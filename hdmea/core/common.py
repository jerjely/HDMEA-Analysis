"""Shared helpers used across the HD-MEA pipeline (ported from the notebooks)."""
from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# progress / logging plumbing (GUI passes real callbacks, CLI uses print)
# --------------------------------------------------------------------------
LogFn = Callable[[str], None]
ProgFn = Callable[[float], None]


def _nolog(msg: str) -> None:
    print(msg)


def _noprog(frac: float) -> None:
    pass


class Cancelled(Exception):
    """Raised inside long loops when the user presses Cancel."""


def check_cancel(cancel_flag) -> None:
    if cancel_flag is not None and cancel_flag():
        raise Cancelled("Cancelled by user")


# --------------------------------------------------------------------------
# BioCam channel helpers (64 x 64 array, 1-based row/col)
# --------------------------------------------------------------------------
def rc_to_id(row: int, col: int, ncols: int = 64) -> str:
    """1-based (row, col) -> string channel_id used by read_biocam."""
    return str((int(row) - 1) * ncols + int(col))


def id_to_rc(ch_id, ncols: int = 64):
    n = int(ch_id)
    return (n - 1) // ncols + 1, (n - 1) % ncols + 1


def safe_channel_id(rec, ch_id):
    """Return a channel id that exists in rec.channel_ids (fixes str/int mismatch)."""
    ids = list(rec.channel_ids)
    if ch_id in ids:
        return ch_id
    s = str(ch_id)
    for cid in ids:
        if str(cid) == s:
            return cid
    try:
        ci = int(ch_id)
        for cid in ids:
            try:
                if int(str(cid)) == ci:
                    return cid
            except Exception:
                pass
    except Exception:
        pass
    raise ValueError(f"Channel id {ch_id} not found in recording")


# --------------------------------------------------------------------------
# File-name parsing
# --------------------------------------------------------------------------
DEFAULT_STIM_REGEX = r"_(SO\d+|BC\d+|P\d+|L\d+|S\d+)\b"


def extract_stimulus(filename: str, regex: str = DEFAULT_STIM_REGEX) -> str:
    m = re.search(regex, str(filename))
    if m:
        return re.sub(r"\d+", "", m.group(1))
    return "?"


def group_from_filename(fname: str, drug_tokens: Iterable[str] = ("PTX",), default: str = "CTRL") -> str:
    up = str(fname).upper()
    for tok in drug_tokens:
        tok = tok.strip()
        if tok and tok.upper() in up:
            return tok.upper()
    return default


def resolve_align_folder_from_brw(brw_path: str) -> str:
    """PsychoPy/lux side-car folder = BRW path without extension."""
    base = os.path.splitext(os.path.basename(brw_path))[0]
    return os.path.join(os.path.dirname(brw_path), base)


def align_folder_of(row) -> str:
    """PsychoPy/lux folder for a table row: explicit 'align_folder' column if set,
    otherwise the BRW path without extension (notebook convention)."""
    af = getattr(row, "align_folder", None) if not isinstance(row, dict) else row.get("align_folder")
    if isinstance(af, str) and af.strip() and af.strip().lower() != "nan":
        return af.strip()
    path = row["path"] if isinstance(row, dict) else row.path
    return resolve_align_folder_from_brw(path)


def has_lux_files(folder: str) -> bool:
    return bool(folder) and os.path.isfile(os.path.join(folder, "lux_stream.csv")) and \
        os.path.isfile(os.path.join(folder, "events_GH.csv"))


_RUN_TS = re.compile(r"(\d{8})_(\d{6})")


def match_run_folders(table: pd.DataFrame, parent: str, max_gap_min: float = 15.0, log: LogFn = _nolog):
    """Assign PsychoPy/lux run folders (names containing YYYYMMDD_HHMMSS, e.g. so_veml_20251217_143012)
    under `parent` to BRW files by start time. BRW start = file modification time − duration
    (BrainWave writes until the recording stops). Returns {row index: folder}."""
    import datetime as dt
    runs = []
    for root, dirs, files in os.walk(parent):
        if "lux_stream.csv" in files or "events_GH.csv" in files:
            m = _RUN_TS.search(os.path.basename(root))
            if m:
                runs.append((dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), root))
    log(f"Found {len(runs)} run folder(s) with a timestamp under {parent}")
    out, used = {}, set()
    for i, r in table.iterrows():
        try:
            end = dt.datetime.fromtimestamp(os.path.getmtime(r["path"]))
        except OSError:
            continue
        start = end - dt.timedelta(seconds=float(r["duration_s"]))
        best = None
        for t, f in runs:
            if f in used:
                continue
            gap = abs((t - start).total_seconds()) / 60
            if gap <= max_gap_min and (best is None or gap < best[0]):
                best = (gap, f, t)
        if best:
            out[i] = best[1]; used.add(best[1])
            log(f"  {r['File']}: BRW start {start:%H:%M:%S} → {os.path.basename(best[1])} (Δ {best[0]:.1f} min)")
        else:
            log(f"  {r['File']}: BRW start {start:%H:%M:%S} → no run folder within {max_gap_min:.0f} min")
    return out


def pick_time_col(df: pd.DataFrame) -> str:
    for c in ("t_s_psychopy", "tThisFlipGlobal"):
        if c in df.columns:
            return c
    for c in df.columns:
        n = c.lower()
        if "time" in n or n.startswith("t"):
            return c
    num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not num:
        raise KeyError(f"No numeric time-like columns in {list(df.columns)}")
    return num[0]


def find_first_existing(folder: str, candidates: Iterable[str]) -> str:
    for fn in candidates:
        p = os.path.join(folder, fn)
        if os.path.isfile(p):
            return p
    lower = {f.lower(): f for f in os.listdir(folder)} if os.path.isdir(folder) else {}
    for fn in candidates:
        if fn.lower() in lower:
            return os.path.join(folder, lower[fn.lower()])
    raise FileNotFoundError(f"None of {list(candidates)} found in {folder}")


# --------------------------------------------------------------------------
# Event pairing
# --------------------------------------------------------------------------
def best_offset_pair(a_times, b_times):
    """If b is longer than a, slide b to the offset that best matches a
    (min median |a-b|). Returns (b_aligned, offset)."""
    a = np.asarray(a_times, float)
    b = np.asarray(b_times, float)
    if len(a) == 0 or len(b) == 0:
        return b[:0], 0
    if len(b) <= len(a):
        return b, 0
    best_off, best_score = 0, None
    for off in range(len(b) - len(a) + 1):
        score = np.median(np.abs(a - b[off:off + len(a)]))
        if best_score is None or score < best_score:
            best_score, best_off = score, off
    return b[best_off:best_off + len(a)], int(best_off)


def trim_pair(a, b):
    n = min(len(a), len(b))
    return np.asarray(a[:n], float), np.asarray(b[:n], float)


# --------------------------------------------------------------------------
# Master unit table (semicolon CSV + XLSX, merged on unit_id)
# --------------------------------------------------------------------------
def dedupe_columns_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df


def merge_into_master(new_df: pd.DataFrame, master_csv: str, also_xlsx: bool = True,
                      log: LogFn = _nolog) -> pd.DataFrame:
    """Outer-merge new columns into the master table; new values overwrite old ones."""
    new_df = new_df.copy()
    new_df.columns = [c.lower() if c != "unit_id" else c for c in new_df.columns]
    new_df["unit_id"] = new_df["unit_id"].astype(int)
    if os.path.exists(master_csv):
        old = pd.read_csv(master_csv, sep=";", low_memory=False)
        old.columns = [c.strip().replace("﻿", "") for c in old.columns]
        old.columns = [c.lower() if c != "unit_id" else c for c in old.columns]
        old = dedupe_columns_keep_last(old)
        old = old[[c for c in old.columns if not c.endswith("_new")]]
        old["unit_id"] = pd.to_numeric(old["unit_id"], errors="coerce").astype(int)
        merged = old.merge(new_df, on="unit_id", how="outer", suffixes=("", "_new"))
        for col in new_df.columns:
            if col == "unit_id":
                continue
            nc = col + "_new"
            if nc in merged.columns:
                merged[col] = merged[nc]
                merged.drop(columns=[nc], inplace=True)
        out = dedupe_columns_keep_last(merged)
    else:
        out = new_df
    out = out.sort_values("unit_id").reset_index(drop=True)
    out.to_csv(master_csv, sep=";", index=False)
    if also_xlsx:
        try:
            out.to_excel(os.path.splitext(master_csv)[0] + ".xlsx", index=False)
        except Exception as e:  # openpyxl missing or file open in Excel
            log(f"⚠ Could not write XLSX ({e})")
    log(f"Master table updated: {master_csv}  ({len(out)} units, {out.shape[1]} columns)")
    return out


def sorting_signature(sorting) -> str:
    ids = np.array(sorted(list(sorting.unit_ids)), dtype=int)
    src = np.concatenate([ids[:200], ids[-200:]]) if ids.size else ids
    return hashlib.md5(src.tobytes()).hexdigest()


def spike_times_s(sorting, unit_id, fs: float) -> np.ndarray:
    st = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=np.float64) / fs
    st.sort()
    return st


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def new_figure(figsize=(6, 4)):
    """Thread-safe matplotlib Figure (no pyplot state)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize, dpi=100)
    FigureCanvasAgg(fig)
    return fig
