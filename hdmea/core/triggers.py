"""Stimulus trigger extraction on the MEA analog/trigger electrode, with optional
lux-sensor correction from the PsychoPy side-car files (SO_Final cell 4).

Per BRW file:
  1. Detect ON/OFF edges on the trigger electrode (hysteresis + min pulse width).
  2. If <brw-without-ext>/lux_stream.csv and events_GH.csv exist: pair the lux
     ON edges with the G (ON TTL) events and the lux OFF edges with H events, and
     add the per-file median delay (TTL -> real light) to the MEA edges.
  3. Otherwise add a fixed shift (old notebooks used +0.12 s).
Results are split by stimulus and group (CTRL / PTX …) and saved as JSON in the
same structure the notebooks used (per_stim_group[stim][group]["ON"/"OFF"]).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .common import (LogFn, ProgFn, _nolog, _noprog, check_cancel, rc_to_id, safe_channel_id,
                     align_folder_of, pick_time_col, best_offset_pair, trim_pair,
                     save_json, new_figure)

DEFAULTS = dict(
    row=1, col=2,
    auto_thresholds=False,
    low_thr=-3.02e4, high_thr=-2.99e4, mid_thr=-3.00e4,
    min_pw_s=0.05,
    chunk_s=10.0,
    use_lux=True,
    lux_thr=50.0,
    per_file_median=True,
    ignore_off_for_so=True,
    fixed_shift_s=0.0,      # used when no lux files (old notebooks: 0.12)
    start_margin_s=0.0,
    end_margin_s=0.25,
)


# ----------------------------------------------------------------------
def auto_thresholds(rec, ch_id, t0=0.0, dur=20.0):
    """Percentile-based thresholds (Biocam_sorting cell 5)."""
    fs = float(rec.get_sampling_frequency())
    cid = safe_channel_id(rec, ch_id)
    y = rec.get_traces(channel_ids=[cid], start_frame=int(t0 * fs),
                       end_frame=int(min(rec.get_num_frames(), (t0 + dur) * fs))).ravel().astype(float)
    p10, p90 = np.percentile(y, [10, 90])
    return dict(low_thr=p10 + 0.40 * (p90 - p10), high_thr=p10 + 0.60 * (p90 - p10), mid_thr=(p10 + p90) / 2)


def detect_edges(rec, ch_id, t0, t1, low_thr, high_thr, mid_thr, min_pw_s=0.05, chunk_s=10.0,
                 cancel=None):
    """Hysteresis edge detector, identical in result to the notebook's per-sample
    state machine but vectorised (only loops over edges, not samples)."""
    fs = float(rec.get_sampling_frequency())
    start_f = int(t0 * fs)
    end_f = min(int(t1 * fs), int(rec.get_num_frames()))
    min_pw = int(min_pw_s * fs)
    chunk = max(1, int(chunk_s * fs))
    cid = safe_channel_id(rec, ch_id)

    on, off = [], []
    state = None
    last_edge = -10 ** 12
    cur = start_f
    while cur < end_f:
        check_cancel(cancel)
        nxt = min(end_f, cur + chunk)
        y = rec.get_traces(channel_ids=[cid], start_frame=cur, end_frame=nxt).ravel()
        if y.size == 0:
            break
        if state is None:
            state = 1 if y[0] >= mid_thr else 0
        hi = np.flatnonzero(y >= high_thr) + cur
        lo = np.flatnonzero(y <= low_thr) + cur
        pos = cur
        while True:
            arr = hi if state == 0 else lo
            k = np.searchsorted(arr, max(pos, last_edge + min_pw))
            if k >= arr.size:
                break
            g = int(arr[k])
            (on if state == 0 else off).append(g / fs)
            state = 1 - state
            last_edge = g
            pos = g + 1
        cur = nxt
    return np.asarray(on, float), np.asarray(off, float)


def read_trace(rec, ch_id, t0, t1, max_points=200_000):
    fs = float(rec.get_sampling_frequency())
    cid = safe_channel_id(rec, ch_id)
    s, e = int(t0 * fs), min(int(t1 * fs), int(rec.get_num_frames()))
    y = rec.get_traces(channel_ids=[cid], start_frame=s, end_frame=e).ravel().astype(float)
    t = np.arange(y.size) / fs + s / fs
    step = max(1, y.size // max_points)
    return t[::step], y[::step]


# ----------------------------------------------------------------------
def lux_edges(lux_csv, thr):
    df = pd.read_csv(lux_csv)
    tc = pick_time_col(df)
    if "lux" not in df.columns:
        raise KeyError(f"'lux' column missing in {lux_csv}")
    t = df[tc].to_numpy(float)
    x = df["lux"].to_numpy(float)
    on = x >= float(thr)
    rise = np.where(on[1:] & ~on[:-1])[0] + 1
    fall = np.where(~on[1:] & on[:-1])[0] + 1
    return t[rise], t[fall]


def gh_events(events_csv):
    df = pd.read_csv(events_csv)
    if "event" not in df.columns:
        raise KeyError(f"'event' column missing in {events_csv}")
    tc = pick_time_col(df)
    ev = df["event"].astype(str).str.strip().str.upper()
    G = df.loc[ev == "G", tc].sort_values().to_numpy(float)
    H = df.loc[ev == "H", tc].sort_values().to_numpy(float)
    return G, H


# ----------------------------------------------------------------------
def extract_triggers(rec, table: pd.DataFrame, params: dict, out_folder: str,
                     log: LogFn = _nolog, progress: ProgFn = _noprog, cancel=None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    ch_id = rc_to_id(p["row"], p["col"])
    if p["auto_thresholds"]:
        th = auto_thresholds(rec, ch_id)
        log(f"Auto thresholds: low={th['low_thr']:.1f} high={th['high_thr']:.1f} mid={th['mid_thr']:.1f}")
    else:
        th = dict(low_thr=p["low_thr"], high_thr=p["high_thr"], mid_thr=p["mid_thr"])

    dbg = []
    n = len(table)
    for i, r in enumerate(table.itertuples(index=False)):
        check_cancel(cancel)
        stim, grp = str(r.stimulus), str(r.group)
        t0, t1 = float(r.start), float(r.end)
        folder = align_folder_of(r)
        lux_p = os.path.join(folder, "lux_stream.csv")
        ev_p = os.path.join(folder, "events_GH.csv")
        have_lux = p["use_lux"] and os.path.exists(lux_p) and os.path.exists(ev_p)

        mea_on, mea_off = detect_edges(rec, ch_id, t0, t1, th["low_thr"], th["high_thr"], th["mid_thr"],
                                       p["min_pw_s"], p["chunk_s"], cancel)
        keep = lambda a: a[(a >= t0 + p["start_margin_s"]) & (a <= t1 - p["end_margin_s"])]
        mea_on, mea_off = keep(mea_on), keep(mea_off)
        skip_off = p["ignore_off_for_so"] and stim == "SO"

        dON = np.array([]); dOFF = np.array([])
        offG = offH = 0
        if have_lux:
            l_on, l_off = lux_edges(lux_p, p["lux_thr"])
            G, H = gh_events(ev_p)
            Ga, offG = best_offset_pair(l_on, G)
            l_on_p, G_p = trim_pair(l_on, Ga)
            dON = l_on_p - G_p
            dON = dON[(l_on_p + t0) <= (t1 - p["end_margin_s"])]
            if not skip_off:
                Ha, offH = best_offset_pair(l_off, H)
                l_off_p, H_p = trim_pair(l_off, Ha)
                dOFF = l_off_p - H_p
                dOFF = dOFF[(l_off_p + t0) <= (t1 - p["end_margin_s"])]
            if p["per_file_median"]:
                ON_corr = mea_on + (float(np.median(dON)) if dON.size else 0.0)
                OFF_corr = mea_off + (float(np.median(dOFF)) if dOFF.size else 0.0)
            else:
                k = min(len(mea_on), len(dON)); ON_corr = mea_on[:k] + dON[:k]
                k = min(len(mea_off), len(dOFF)); OFF_corr = mea_off[:k] + dOFF[:k]
        else:
            ON_corr = mea_on + p["fixed_shift_s"]
            OFF_corr = mea_off + p["fixed_shift_s"]
        if skip_off:
            OFF_corr = np.array([], float)

        dbg.append(dict(
            file=r.File, brw_path=r.path, group=grp, stim=stim, t0=t0, t1=t1,
            lux_used=bool(have_lux), n_mea_on=int(len(mea_on)), n_mea_off=int(len(mea_off)),
            n_ON_corr=int(len(ON_corr)), n_OFF_corr=int(len(OFF_corr)),
            dON_median_ms=float(np.median(dON) * 1000) if dON.size else np.nan,
            dOFF_median_ms=float(np.median(dOFF) * 1000) if dOFF.size else np.nan,
            offG=int(offG), offH=int(offH),
            ON_corr_mea=np.asarray(ON_corr, float).tolist(),
            OFF_corr_mea=np.asarray(OFF_corr, float).tolist(),
        ))
        msg = (f"[{i + 1}/{n}] {r.File} ({stim}/{grp}): ON={len(ON_corr)} OFF={len(OFF_corr)}"
               + (f" | lux dON={dbg[-1]['dON_median_ms']:.1f} ms" if have_lux and dON.size else
                  " | no lux files → fixed shift" if not have_lux else ""))
        log(msg)
        progress((i + 1) / n)

    dbg_df = pd.DataFrame(dbg)
    cat = lambda ser: np.sort(np.concatenate([np.asarray(x, float) for x in ser])) if len(ser) else np.array([])
    per_stim, per_stim_group = {}, {}
    for stim in dbg_df["stim"].unique():
        sub = dbg_df[dbg_df.stim == stim]
        per_stim[stim] = dict(ON=cat(sub.ON_corr_mea).tolist(), OFF=cat(sub.OFF_corr_mea).tolist())
        per_stim_group[stim] = {}
        for grp in sub["group"].unique():
            s2 = sub[sub.group == grp]
            per_stim_group[stim][grp] = dict(ON=cat(s2.ON_corr_mea).tolist(), OFF=cat(s2.OFF_corr_mea).tolist())

    out = dict(row=int(p["row"]), col=int(p["col"]), ch_id=str(ch_id), thresholds=th, params=p,
               dbg_df=dbg_df.to_dict(orient="list"), per_stim=per_stim, per_stim_group=per_stim_group,
               ON_corr_all=cat(dbg_df.ON_corr_mea).tolist(), OFF_corr_all=cat(dbg_df.OFF_corr_mea).tolist())
    path = triggers_json_path(out_folder, p["row"], p["col"])
    save_json(out, path)
    log(f"Saved triggers: {path}")
    out["_path"] = path
    return out


def triggers_json_path(out_folder, row, col):
    return os.path.join(out_folder, f"ON_OFF_luxcorr_MEA_r{int(row)}c{int(col)}.json")


def get_times(trig: dict, stim: str, grp: str, edge: str) -> np.ndarray:
    try:
        a = np.asarray(trig["per_stim_group"][stim][grp][edge], float)
    except Exception:
        return np.array([], float)
    a = a[np.isfinite(a)]
    a.sort()
    return a


def summary_table(trig: dict) -> pd.DataFrame:
    rows = []
    for stim, g in trig.get("per_stim_group", {}).items():
        for grp, d in g.items():
            rows.append(dict(stimulus=stim, group=grp, n_ON=len(d.get("ON", [])), n_OFF=len(d.get("OFF", []))))
    return pd.DataFrame(rows)


def qc_figure(rec, trig: dict, t0: float, t1: float):
    ch = trig.get("ch_id", rc_to_id(trig.get("row", 1), trig.get("col", 2)))
    t, y = read_trace(rec, ch, t0, t1)
    if y.size == 0:
        end = rec.get_total_duration()
        raise ValueError(f"No samples in {t0:.0f}–{t1:.0f} s: the loaded recording is only {end:.1f} s long. "
                         f"If the session table is longer, the Zarr cache is stale – rebuild it (Session tab).")
    fig = new_figure((10, 3))
    ax = fig.add_subplot(111)
    ax.plot(t, y, lw=0.6, color="k", label="trigger electrode")
    on = np.asarray(trig.get("ON_corr_all", []), float)
    off = np.asarray(trig.get("OFF_corr_all", []), float)
    med = np.median(y) if y.size else 0
    rng = (np.ptp(y) if y.size else 1) or 1
    sel = on[(on >= t0) & (on <= t1)]
    ax.plot(sel, np.full_like(sel, med), "|", ms=14, color="tab:red", label="ON (corrected)")
    sel = off[(off >= t0) & (off <= t1)]
    ax.plot(sel, np.full_like(sel, med - 0.1 * rng), "|", ms=14, color="tab:blue", label="OFF (corrected)")
    th = trig.get("thresholds", {})
    for k, ls in (("low_thr", "--"), ("high_thr", "--"), ("mid_thr", ":")):
        if k in th:
            ax.axhline(th[k], ls=ls, alpha=0.4, color="gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ADC")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def lux_figure(rec, table_row, trig_params: dict, ch_id=None):
    """Diagnostic for one file: lux trace with G/H events (PsychoPy time) and the
    trigger electrode for the same file (file-local MEA time)."""
    p = {**DEFAULTS, **(trig_params or {})}
    folder = align_folder_of(table_row)
    t0, t1 = float(table_row["start"]), float(table_row["end"])
    fig = new_figure((11, 6))
    a1 = fig.add_subplot(211); a2 = fig.add_subplot(212)
    lux_p, ev_p = os.path.join(folder, "lux_stream.csv"), os.path.join(folder, "events_GH.csv")
    info = []
    if os.path.exists(lux_p):
        df = pd.read_csv(lux_p)
        tc = pick_time_col(df)
        a1.plot(df[tc], df["lux"], lw=0.7, color="k", label="lux")
        a1.axhline(p["lux_thr"], ls="--", color="tab:orange", lw=0.8, label=f"lux thr {p['lux_thr']:g}")
        l_on, l_off = lux_edges(lux_p, p["lux_thr"])
        info.append(f"lux ON edges {len(l_on)}, OFF {len(l_off)}")
    else:
        a1.text(0.5, 0.5, f"lux_stream.csv not found in\n{folder}", ha="center", va="center", transform=a1.transAxes)
    if os.path.exists(ev_p):
        G, H = gh_events(ev_p)
        for g in G:
            a1.axvline(g, color="tab:green", ls="--", lw=0.7, alpha=0.6)
        for h in H:
            a1.axvline(h, color="tab:purple", ls=":", lw=0.7, alpha=0.6)
        a1.plot([], [], color="tab:green", ls="--", label=f"G ({len(G)})")
        a1.plot([], [], color="tab:purple", ls=":", label=f"H ({len(H)})")
        info.append(f"G {len(G)}, H {len(H)}")
    a1.set_xlabel("PsychoPy time (s)"); a1.set_ylabel("lux"); a1.legend(fontsize=8, loc="upper right")
    a1.set_title(f"{table_row['File']} — {os.path.basename(folder)}  |  " + ", ".join(info), fontsize=9)
    ch = ch_id or rc_to_id(p["row"], p["col"])
    t, y = read_trace(rec, ch, t0, t1)
    if y.size:
        a2.plot(t - t0, y, lw=0.6, color="k")
        for k, ls in (("low_thr", "--"), ("high_thr", "--"), ("mid_thr", ":")):
            a2.axhline(p[k], ls=ls, color="gray", alpha=0.5)
        a2.set_title(f"Trigger electrode r{p['row']}c{p['col']}: min {y.min():.0f}, max {y.max():.0f}", fontsize=9)
    else:
        a2.text(0.5, 0.5, f"No MEA samples for {t0:.0f}–{t1:.0f} s (recording is {rec.get_total_duration():.0f} s)",
                ha="center", va="center", transform=a2.transAxes)
    a2.set_xlabel("Time from file start (s)"); a2.set_ylabel("ADC")
    fig.tight_layout()
    return fig
