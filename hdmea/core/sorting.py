"""Spike sorting (HerdingSpikes, full array), electrode activity map and unit
localisation from HS2_detected.hdf5 (SO_Final cells 5, 6, 12)."""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

from .common import LogFn, ProgFn, _nolog, _noprog, check_cancel, new_figure

HS_DEFAULTS = dict(
    freq_min=350.0, freq_max=6000.0, common_ref="global",
    threshold=8.5, neighbor_radius=15.0, inner_radius=5.0, peak_jitter=0.15,
    left_cutout_time=0.6, right_cutout_time=1.2,
)


def preprocess(rec, freq_min=350.0, freq_max=6000.0, common_ref="global"):
    import spikeinterface.preprocessing as spre
    r = rec
    if np.dtype(r.get_dtype()).kind == "u":
        r = spre.unsigned_to_signed(r)
    r = spre.bandpass_filter(r, freq_min=freq_min, freq_max=freq_max)
    if common_ref and common_ref != "none":
        r = spre.common_reference(r, reference=common_ref)
    return r


def hs_root(out_folder):
    return os.path.join(out_folder, "hs_full")


def list_runs(out_folder):
    root = hs_root(out_folder)
    if not os.path.isdir(root):
        return []
    runs = [os.path.join(root, d) for d in os.listdir(root)
            if d.startswith("hs_full_") and os.path.isdir(os.path.join(root, d))]
    return sorted(runs, key=os.path.getmtime, reverse=True)


def read_sorting(folder):
    import spikeinterface.sorters as ss
    try:
        return ss.read_sorter_folder(folder, register_recording=False)
    except TypeError:
        return ss.read_sorter_folder(sorter_name="herdingspikes", folder_path=folder, register_recording=False)


def run_herdingspikes(rec, out_folder, params: dict, extra_params: dict | None = None,
                      time_range=None, log: LogFn = _nolog):
    import spikeinterface.sorters as ss
    import spikeinterface.preprocessing as spre
    p = {**HS_DEFAULTS, **(params or {})}
    r = preprocess(rec, p["freq_min"], p["freq_max"], p["common_ref"])
    if time_range:
        r = r.time_slice(start_time=float(time_range[0]), end_time=float(time_range[1]))
        log(f"Sorting only {time_range[0]}–{time_range[1]} s (test run)")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(hs_root(out_folder), f"hs_full_{stamp}")
    os.makedirs(hs_root(out_folder), exist_ok=True)
    sorter_kw = {k: p[k] for k in ("threshold", "neighbor_radius", "inner_radius", "peak_jitter",
                                   "left_cutout_time", "right_cutout_time")}
    sorter_kw.update(extra_params or {})
    log(f"Running HerdingSpikes → {folder}\n  params: {sorter_kw}")
    ss.run_sorter(sorter_name="herdingspikes", recording=r, folder=folder,
                  remove_existing_folder=False, with_output=False, verbose=True, **sorter_kw)
    sorting = read_sorting(folder)
    log(f"Sorting finished: {len(sorting.unit_ids)} units")
    return sorting, folder


# ----------------------------------------------------------------------
def hs2_detected_path(run_folder):
    for p in (os.path.join(run_folder, "sorter_output", "HS2_detected.hdf5"),
              os.path.join(run_folder, "HS2_detected.hdf5")):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"HS2_detected.hdf5 not found in {run_folder}")


def _read_hs2(path):
    import h5py
    with h5py.File(path, "r") as f:
        ch = np.asarray(f["ch"][:])
        t = np.asarray(f["t"][:])
    ch = np.where(np.isfinite(ch), ch, -1).astype(np.int64)
    t = np.where(np.isfinite(t), t, -1).astype(np.int64)
    return ch, t


def activity_map(rec, run_folder, channel_base="auto", log: LogFn = _nolog):
    """Spikes/s per electrode from HerdingSpikes detections."""
    ch, _ = _read_hs2(hs2_detected_path(run_folder))
    xy = np.asarray(rec.get_channel_locations(), float)
    n = xy.shape[0]
    cands = {"0-based": ch, "1-based": ch - 1}
    if channel_base == "auto":
        def spread(idx):
            pts = xy[np.clip(idx[:200000], 0, n - 1)]
            return float(np.nanstd(pts[:, 0]) + np.nanstd(pts[:, 1]))
        base = max(cands, key=lambda k: spread(cands[k]))
        log(f"Channel numbering (auto): {base}")
    else:
        base = channel_base
    idx = np.clip(cands[base], 0, n - 1)
    dur = rec.get_num_frames() / float(rec.get_sampling_frequency())
    rate = np.bincount(idx, minlength=n) / max(dur, 1e-12)
    return xy, rate, base


def activity_figure(xy, rate, title="Electrode activity"):
    fig = new_figure((6, 5.5))
    ax = fig.add_subplot(111)
    vmax = np.percentile(rate[rate > 0], 99) if np.any(rate > 0) else 1.0
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=rate, s=9, marker="s", cmap="hot", vmin=0, vmax=vmax)
    fig.colorbar(sc, ax=ax, label="spikes/s", fraction=0.046)
    ax.set_aspect("equal")
    ax.set_facecolor("k")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
def localize_units(rec, sorting, run_folder, out_folder, min_spikes=500, tol=5,
                   max_spikes_offset=30_000, max_spikes_per_unit=400, offset_band=250,
                   channel_base="1-based", log: LogFn = _nolog, progress: ProgFn = _noprog,
                   cancel=None) -> pd.DataFrame:
    """Dominant HS-detected channel per unit -> MEA (x, y). Vectorised version of
    SO_Final cell 12 (global time offset search + per-unit channel vote)."""
    ch, t = _read_hs2(hs2_detected_path(run_folder))
    order = np.argsort(t, kind="stable")
    t, ch = t[order], ch[order]
    ut, first = np.unique(t, return_index=True)
    uch = ch[first]
    log(f"HS detections: {len(t):,} | unique times: {len(ut):,}")

    def nearest(samples):
        k = np.searchsorted(ut, samples)
        k0 = np.clip(k - 1, 0, ut.size - 1)
        k1 = np.clip(k, 0, ut.size - 1)
        d0 = np.abs(ut[k0] - samples); d1 = np.abs(ut[k1] - samples)
        use1 = d1 < d0
        kk = np.where(use1, k1, k0)
        return kk, np.where(use1, d1, d0)

    # --- global offset between sorting spike times and detections ---
    uids = np.array(list(sorting.unit_ids))
    cap = max(50, max_spikes_offset // max(1, len(uids)))
    samp = np.sort(np.concatenate([np.asarray(sorting.get_unit_spike_train(u), np.int64)[:cap] for u in uids]
                                  or [np.array([], np.int64)]))[:max_spikes_offset]
    kk0, _ = nearest(samp)
    off0 = int(np.median(ut[kk0] - samp)) if samp.size else 0
    cands = np.arange(off0 - offset_band, off0 + offset_band + 1)
    rates = np.empty(cands.size)
    for i, o in enumerate(cands):
        _, dd = nearest(samp + o)
        # hit rate within ±tol; ties (plateau) broken by the smallest mean distance
        rates[i] = (np.mean(dd <= tol) - 1e-4 * np.mean(np.minimum(dd, tol + 1))) if samp.size else 0
    est = int(cands[np.argmax(rates)])
    log(f"Estimated detection offset = {est} samples (hit-rate {rates.max() * 100:.1f}%)")

    # --- per unit dominant channel ---
    xy = np.asarray(rec.get_channel_locations(), float)
    shift = 1 if channel_base == "1-based" else 0
    rows = []
    for i, u in enumerate(uids):
        check_cancel(cancel)
        st_all = np.asarray(sorting.get_unit_spike_train(u), np.int64)
        nsp = st_all.size
        if nsp < min_spikes:
            continue
        st = st_all[:max_spikes_per_unit] + est
        kk, dd = nearest(st)
        hits = uch[kk[dd <= tol]]
        hits = hits[hits >= 0]
        if hits.size == 0:
            continue
        dom = int(np.argmax(np.bincount(hits)))
        idx = dom - shift
        if idx < 0 or idx >= xy.shape[0]:
            continue
        rows.append(dict(mea_unit_id=int(u), mea_x_um=float(xy[idx, 0]), mea_y_um=float(xy[idx, 1]),
                         dominant_channel=dom, confidence=float(hits.size / st.size), n_spikes=int(nsp)))
        if i % 50 == 0:
            progress(i / len(uids))
    df = pd.DataFrame(rows).sort_values("mea_unit_id").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["mea_unit_id", "mea_x_um", "mea_y_um", "dominant_channel", "confidence", "n_spikes"])
    out_csv = unit_xy_csv(out_folder)
    df.to_csv(out_csv, index=False)
    log(f"Localised {len(df)} / {len(uids)} units → {out_csv}")
    progress(1.0)
    return df


def unit_xy_csv(out_folder):
    return os.path.join(out_folder, "unit_xy_map_from_HS2_detected.csv")


def load_unit_xy(out_folder) -> dict:
    p = unit_xy_csv(out_folder)
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    xc = "mea_x_um_true" if "mea_x_um_true" in df else "mea_x_um"
    yc = "mea_y_um_true" if "mea_y_um_true" in df else "mea_y_um"
    return {int(u): (float(x), float(y)) for u, x, y in zip(df["mea_unit_id"], df[xc], df[yc])}


def locations_figure(rec, df):
    xy = np.asarray(rec.get_channel_locations(), float)
    fig = new_figure((6, 5.5))
    ax = fig.add_subplot(111)
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="0.85", marker="s")
    if len(df):
        sc = ax.scatter(df["mea_x_um"], df["mea_y_um"], s=16, c=df["confidence"], cmap="viridis",
                        edgecolor="k", linewidths=0.3)
        fig.colorbar(sc, ax=ax, label="detection confidence", fraction=0.046)
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"Unit locations (n={len(df)})")
    fig.tight_layout()
    return fig
