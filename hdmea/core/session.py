"""Session = one experiment (one retina): a list of BRW files that are concatenated
in order, an output folder, the per-file table (stimulus / group / start / end),
and the cached concatenated recording (Zarr)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import (LogFn, ProgFn, _nolog, _noprog, extract_stimulus, group_from_filename,
                     DEFAULT_STIM_REGEX, save_json, load_json, resolve_align_folder_from_brw)

SESSION_FILE = "hdmea_session.json"
ZARR_NAME = "rec_concat_zarr.zarr"


@dataclass
class Session:
    brw_files: list = field(default_factory=list)
    out_folder: str = ""
    table: pd.DataFrame | None = None          # one row per BRW file
    settings: dict = field(default_factory=dict)  # GUI parameters per tab

    # runtime (not saved)
    rec: object = None          # concatenated recording (Zarr cache)
    sorting: object = None      # spike sorting result
    sorting_folder: str = ""
    triggers: dict | None = None  # trigger JSON contents
    triggers_path: str = ""

    # ------------------------------------------------------------------
    @property
    def zarr_path(self) -> str:
        return os.path.join(self.out_folder, ZARR_NAME)

    @property
    def session_path(self) -> str:
        return os.path.join(self.out_folder, SESSION_FILE)

    @property
    def fs(self) -> float:
        if self.rec is not None:
            return float(self.rec.get_sampling_frequency())
        if self.table is not None and "fs" in self.table:
            return float(self.table["fs"].iloc[0])
        raise RuntimeError("No recording loaded")

    @property
    def master_csv(self) -> str:
        return os.path.join(self.out_folder, "unit_metrics_master.csv")

    def groups(self) -> list:
        if self.table is None:
            return []
        seen = []
        for g in self.table["group"]:
            if g not in seen:
                seen.append(g)
        return seen

    def stimuli(self) -> list:
        if self.table is None:
            return []
        seen = []
        for s in self.table["stimulus"]:
            if s not in seen:
                seen.append(s)
        return seen

    # ------------------------------------------------------------------
    def save(self) -> None:
        os.makedirs(self.out_folder, exist_ok=True)
        d = dict(
            brw_files=list(self.brw_files),
            out_folder=self.out_folder,
            table=self.table.to_dict(orient="list") if self.table is not None else None,
            settings=self.settings,
            sorting_folder=self.sorting_folder,
            triggers_path=self.triggers_path,
        )
        save_json(d, self.session_path)

    @classmethod
    def load(cls, path: str) -> "Session":
        if os.path.isdir(path):
            path = os.path.join(path, SESSION_FILE)
        d = load_json(path)
        s = cls(brw_files=d.get("brw_files", []), out_folder=d.get("out_folder") or os.path.dirname(path),
                settings=d.get("settings", {}))
        # if the folder was moved, trust the location of the json
        if not os.path.isdir(s.out_folder):
            s.out_folder = os.path.dirname(path)
        if d.get("table"):
            s.table = pd.DataFrame(d["table"])
            if "align_folder" not in s.table:
                s.table["align_folder"] = [resolve_align_folder_from_brw(p) for p in s.table["path"]]
        s.sorting_folder = d.get("sorting_folder", "")
        s.triggers_path = d.get("triggers_path", "")
        return s


# ----------------------------------------------------------------------
def default_out_folder(brw_files) -> str:
    """Same rule as the notebook: <dir of first BRW>/<common filename prefix>."""
    base_dir = os.path.dirname(brw_files[0])
    names = [os.path.splitext(os.path.basename(f))[0] for f in brw_files]
    prefix = os.path.commonprefix(names).rstrip("_-") or "session"
    return os.path.join(base_dir, prefix)


def read_brw(path: str, mea_pitch: float = 42.0, fill_gaps: str = "zeros"):
    import spikeinterface.extractors as se
    return se.read_biocam(file_path=str(path), mea_pitch=mea_pitch, fill_gaps_strategy=fill_gaps)


def build_record_table(brw_files, stim_regex: str = DEFAULT_STIM_REGEX, drug_tokens=("PTX",),
                       mea_pitch: float = 42.0, log: LogFn = _nolog) -> pd.DataFrame:
    rows = []
    for fp in brw_files:
        rec = read_brw(fp, mea_pitch)
        dur = float(rec.get_total_duration())
        name = os.path.basename(fp)
        rows.append(dict(
            File=name, path=str(fp),
            stimulus=extract_stimulus(name, stim_regex),
            group=group_from_filename(name, drug_tokens),
            duration_s=round(dur, 4),
            fs=float(rec.get_sampling_frequency()),
            channels=int(rec.get_num_channels()),
            dtype=str(rec.get_dtype()),
            align_folder=resolve_align_folder_from_brw(fp),
        ))
        log(f"  {name}: {dur:.1f} s, {rec.get_num_channels()} ch, {rec.get_sampling_frequency():.1f} Hz")
    df = pd.DataFrame(rows)
    return recompute_starts(df)


def recompute_starts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    starts = np.concatenate([[0.0], np.cumsum(df["duration_s"].astype(float).to_numpy())[:-1]])
    df["start"] = starts
    df["end"] = starts + df["duration_s"].astype(float).to_numpy()
    return df


def load_or_build_zarr(session: Session, rebuild: bool = False, mea_pitch: float = 42.0,
                       n_jobs: int = -1, chunk_duration: str = "1s", mp_context=None,
                       log: LogFn = _nolog, progress: ProgFn = _noprog):
    """Concatenate BRWs (in table order), convert unsigned->signed, cache to Zarr.
    Mirrors SO_Final cell 2. Returns the cached recording."""
    import spikeinterface as si
    import spikeinterface.preprocessing as spre

    zp = session.zarr_path
    if os.path.exists(zp) and not rebuild:
        log(f"Loading Zarr cache: {zp}")
        rec = _load_extractor(zp)
        try:
            rec.get_channel_locations()
        except Exception:
            log("Zarr has no channel locations – attaching from first BRW")
            rec.set_channel_locations(read_brw(session.brw_files[0], mea_pitch).get_channel_locations())
        _log_rec(rec, log)
        return rec

    files = list(session.table["path"]) if session.table is not None else list(session.brw_files)
    log(f"Building concatenated recording from {len(files)} BRW file(s)…")
    recs = [read_brw(fp, mea_pitch) for fp in files]
    fss = {float(r.get_sampling_frequency()) for r in recs}
    if len(fss) != 1:
        raise RuntimeError(f"Recordings have different sampling rates: {sorted(fss)}")
    recording = si.concatenate_recordings(recs)
    try:
        recording.set_channel_locations(recs[0].get_channel_locations())
    except Exception as e:
        log(f"⚠ Could not attach channel locations: {e!r}")
    if np.dtype(recording.get_dtype()).kind == "u":
        recording = spre.unsigned_to_signed(recording)
        log("Converted unsigned → signed")
    os.makedirs(session.out_folder, exist_ok=True)
    log(f"Writing Zarr cache (this takes a while): {zp}")
    job_kw = dict(n_jobs=n_jobs, chunk_duration=chunk_duration, progress_bar=True)
    if mp_context:
        job_kw["mp_context"] = mp_context
    rec = recording.save(folder=zp, format="zarr", overwrite=True, **job_kw)
    progress(1.0)
    _log_rec(rec, log)
    return rec


def _load_extractor(path):
    import spikeinterface as si
    if hasattr(si, "load"):
        return si.load(path)
    return si.load_extractor(path)


def _log_rec(rec, log):
    log(f"Recording: {rec.get_num_channels()} ch | {rec.get_sampling_frequency():.2f} Hz | "
        f"{rec.get_total_duration():.1f} s | dtype {rec.get_dtype()}")
