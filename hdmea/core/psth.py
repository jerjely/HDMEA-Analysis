"""Light-response PSTH metrics per unit, per group (CTRL / PTX …) and edge (ON/OFF),
with optional opto reference. Port of SO_Final cell 7 (+ cell 8 filter)."""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences

from .common import (LogFn, ProgFn, _nolog, _noprog, check_cancel, new_figure, sorting_signature,
                     merge_into_master, spike_times_s)
from .triggers import get_times

DEFAULTS = dict(
    stim_key="L", pre_time=0.5, post_time=3.0, bin_size=0.025, smooth_sigma=1.0,
    min_spikes=150, peak_window_s=1.0, min_peak_rate_Hz=3.0, min_peak_delta_Hz=2.0,
    min_peak_delta_global_Hz=0.0, min_peak_z=2.5, min_peak_prom_Hz=1.0, min_peak_distance_s=0.05,
    save_plots=False, plot_min_trials=5, plot_min_spikes=5, ref_group="CTRL", opto_stim="",
)


class PsthEngine:
    def __init__(self, params: dict):
        self.p = {**DEFAULTS, **(params or {})}
        p = self.p
        self.bins = np.arange(-p["pre_time"], p["post_time"] + p["bin_size"], p["bin_size"])
        self.bc = (self.bins[:-1] + self.bins[1:]) / 2
        self.min_dist_bins = max(1, int(round(p["min_peak_distance_s"] / p["bin_size"])))

    def collect(self, spikes, events):
        p = self.p
        if events is None or len(events) == 0 or spikes.size == 0:
            return None, None, 0
        rel = []
        tot = 0
        for t0 in events:
            lo = np.searchsorted(spikes, t0 - p["pre_time"], "left")
            hi = np.searchsorted(spikes, t0 + p["post_time"], "right")
            r = spikes[lo:hi] - t0
            rel.append(r)
            tot += r.size
        if tot == 0:
            return None, None, 0
        counts, _ = np.histogram(np.concatenate(rel), bins=self.bins)
        rate = counts / (p["bin_size"] * len(events))
        return rel, gaussian_filter1d(rate.astype(float), sigma=p["smooth_sigma"]), tot

    def peak_and_tau(self, rate, unit_mean_rate):
        p = self.p
        bc = self.bc
        if rate is None:
            return (np.nan,) * 6
        pre = rate[bc < 0] if np.any(bc < 0) else np.array([0.0])
        baseline = float(np.mean(pre))
        std_eff = max(float(np.std(pre)), 0.2)
        sm = (bc >= 0) & (bc <= p["peak_window_s"])
        if not np.any(sm):
            return baseline, np.nan, np.nan, np.nan, np.nan, np.nan
        y, x = rate[sm], bc[sm]
        peaks, _ = find_peaks(y, prominence=p["min_peak_prom_Hz"], distance=self.min_dist_bins)
        if peaks.size == 0:
            return baseline, np.nan, np.nan, np.nan, np.nan, np.nan
        prom, _, _ = peak_prominences(y, peaks)
        best, best_z = None, -np.inf
        for k, pk in enumerate(peaks):
            pr = float(y[pk])
            z = (pr - baseline) / std_eff
            if (pr < p["min_peak_rate_Hz"] or pr - baseline < p["min_peak_delta_Hz"]
                    or pr - unit_mean_rate < p["min_peak_delta_global_Hz"] or z < p["min_peak_z"]
                    or prom[k] < p["min_peak_prom_Hz"]):
                continue
            if z > best_z:
                best_z, best = z, pk
        if best is None:
            return baseline, np.nan, np.nan, np.nan, np.nan, np.nan
        peak_rate, t_peak = float(y[best]), float(x[best])
        target = peak_rate / math.e
        gi = np.where(sm)[0][best]
        t_tau = tau = np.nan
        for k in range(gi + 1, len(bc)):
            if rate[k] <= target:
                t1, r1, t2, r2 = bc[k - 1], rate[k - 1], bc[k], rate[k]
                frac = (target - r1) / (r2 - r1) if r2 != r1 else 0.0
                t_tau = float(t1 + frac * (t2 - t1))
                tau = t_tau - t_peak
                break
        return baseline, peak_rate, t_peak, target, tau, t_tau


def _interval_medians(on, off):
    n = min(len(on), len(off))
    if n == 0:
        return np.nan, np.nan
    on2off = off[:n] - on[:n]
    off2on = on[1:n + 1] - off[:len(on[1:n + 1])] if len(on) > 1 else np.array([])
    return (float(np.median(on2off)) if on2off.size else np.nan,
            float(np.median(off2on)) if off2on.size else np.nan)


def _isi_windows(on, off):
    w = [(float(off[i]), float(on[i + 1])) for i in range(len(off)) if i + 1 < len(on) and on[i + 1] > off[i]]
    return w, float(sum(e - s for s, e in w))


def _mean_rate_windows(spikes, wins, total, fallback):
    if wins and total > 0 and spikes.size:
        n = sum(int(np.searchsorted(spikes, e) - np.searchsorted(spikes, s)) for s, e in wins)
        return n / total
    return fallback


def build_conditions(trig, stim, groups):
    cond = {}
    for g in groups:
        on, off = get_times(trig, stim, g, "ON"), get_times(trig, stim, g, "OFF")
        on_i, off_i = _interval_medians(on, off)
        w, tot = _isi_windows(on, off)
        cond[(g, "ON")] = dict(times=on, interval=on_i, isi=w, total=tot)
        cond[(g, "OFF")] = dict(times=off, interval=off_i, isi=w, total=tot)
    return cond


def opto_times(trig, opto_stim):
    if not opto_stim:
        return np.array([], float)
    on = np.asarray(trig.get("per_stim", {}).get(opto_stim, {}).get("ON", []), float)
    return np.sort(on[np.isfinite(on)])


def run_light_responses(sorting, fs, duration_s, trig, out_folder, params, groups=None,
                        log: LogFn = _nolog, progress: ProgFn = _noprog, cancel=None) -> pd.DataFrame:
    eng = PsthEngine(params)
    p = eng.p
    stim = p["stim_key"]
    if stim not in trig.get("per_stim_group", {}):
        raise KeyError(f"Stimulus '{stim}' not in triggers. Available: {list(trig.get('per_stim_group', {}))}")
    groups = groups or list(trig["per_stim_group"][stim].keys())
    cond = build_conditions(trig, stim, groups)
    for (g, e), c in cond.items():
        log(f"  {stim} {g} {e}: {len(c['times'])} events")
    opto = opto_times(trig, p["opto_stim"])
    if opto.size:
        log(f"  Opto ({p['opto_stim']}): {opto.size} events")

    sig = sorting_signature(sorting)
    ids = np.array(sorted(list(sorting.unit_ids)), dtype=int)
    counts = np.array([len(sorting.get_unit_spike_train(u)) for u in ids])
    valid = ids[counts >= int(p["min_spikes"])]
    log(f"Valid units (≥{p['min_spikes']} spikes): {len(valid)} / {len(ids)}")

    plot_dir = os.path.join(out_folder, "PSTH_line_raster")
    ref = p["ref_group"] if p["ref_group"] in groups else groups[0]
    rows = []
    for i, u in enumerate(valid, 1):
        check_cancel(cancel)
        sp = spike_times_s(sorting, int(u), fs)
        fallback = sp.size / max(duration_s, 1e-12)
        row = dict(unit_id=int(u), sorting_signature=sig, stim_key=stim, min_spikes_filter=int(p["min_spikes"]))
        for (g, e), c in cond.items():
            umr = _mean_rate_windows(sp, c["isi"], c["total"], fallback)
            rel, rate, nwin = eng.collect(sp, c["times"])
            base, pr, pt, tgt, tau, ttau = eng.peak_and_tau(rate, umr)
            pre = f"{stim}_{g}_{e}".lower()
            row.update({f"{pre}_has_events": int(len(c["times"]) > 0), f"{pre}_n_events": int(len(c["times"])),
                        f"{pre}_n_spikes_in_window": int(nwin),
                        f"{pre}_interval_s": c["interval"], f"{pre}_peak_time_s": pt,
                        f"{pre}_peak_rate_hz": pr, f"{pre}_baseline_rate_hz": base,
                        f"{pre}_target_level_hz": tgt, f"{pre}_tau_s": tau})
            if p["save_plots"] and rel is not None and len(c["times"]) >= p["plot_min_trials"] \
                    and nwin >= p["plot_min_spikes"]:
                d = os.path.join(plot_dir, g, e); os.makedirs(d, exist_ok=True)
                fig = single_psth_figure(eng, rel, rate, f"Unit {u} {stim} {g} {e}", c["interval"], tgt, pt, ttau)
                fig.savefig(os.path.join(d, f"unit_{int(u):04d}_PSTH_{stim}_{g}_{e}.png"), dpi=150)
        if opto.size:
            rel, rate, nwin = eng.collect(sp, opto)
            base, pr, pt, tgt, tau, ttau = eng.peak_and_tau(rate, fallback)
            row.update(dict(opto_n_events=int(opto.size), opto_n_spikes_in_window=int(nwin),
                            opto_peak_time_s=pt, opto_peak_rate_hz=pr, opto_baseline_rate_hz=base, opto_tau_s=tau))
        on_ok = np.isfinite(row.get(f"{stim}_{ref}_on_peak_rate_hz".lower(), np.nan))
        off_ok = np.isfinite(row.get(f"{stim}_{ref}_off_peak_rate_hz".lower(), np.nan))
        row["psth_response_type"] = "ON-OFF" if on_ok and off_ok else "ON" if on_ok else "OFF" if off_ok else ""
        if opto.size:
            row["pv_status"] = "PV+" if np.isfinite(row.get("opto_peak_rate_hz", np.nan)) else ""
        rows.append(row)
        if i % 25 == 0 or i == len(valid):
            progress(i / max(1, len(valid)))
    df = pd.DataFrame(rows)
    base = pd.DataFrame({"unit_id": ids, "n_spikes_total": counts})
    df = base.merge(df, on="unit_id", how="left")
    master = merge_into_master(df, os.path.join(out_folder, "unit_metrics_master.csv"), log=log)
    # PSTH-valid subset (cell 8)
    oc, fc = f"{stim}_{ref}_on_peak_rate_hz".lower(), f"{stim}_{ref}_off_peak_rate_hz".lower()
    if oc in master and fc in master:
        sub = master[master[oc].notna() | master[fc].notna()]
        sub.to_csv(os.path.join(out_folder, "unit_metrics_master_PSTH_valid.csv"), sep=";", index=False)
        log(f"PSTH-valid units ({ref} ON or OFF peak): {len(sub)}")
    return df


# ----------------------------------------------------------------------
def single_psth_figure(eng, rel, rate, title, interval, target, peak_t, tau_t):
    p = eng.p
    fig = new_figure((6, 4))
    ax_r = fig.add_subplot(211)
    ax_p = fig.add_subplot(212, sharex=ax_r)
    for tr, r in enumerate(rel, 1):
        if r.size:
            ax_r.vlines(r, tr - 0.4, tr + 0.4, color="k", lw=0.6)
    ax_r.axvline(0, color="r", ls="--", lw=1)
    ax_r.set_ylabel("Trials"); ax_r.set_title(title)
    ax_p.plot(eng.bc, rate, color="k", lw=1.5)
    ax_p.axvline(0, color="r", ls="--", lw=1)
    for v, kw in ((interval, dict(color="gray", ls=":")), (peak_t, dict(color="gray", ls=":")),
                  (tau_t, dict(color="tab:blue", ls="--"))):
        if np.isfinite(v):
            ax_p.axvline(v, lw=1, **kw)
    if np.isfinite(target):
        ax_p.axhline(target, color="gray", ls="--", lw=1)
    ax_p.set_xlim(-p["pre_time"], p["post_time"])
    ax_p.set_xlabel("Time (s)"); ax_p.set_ylabel("Rate (Hz)")
    fig.tight_layout()
    return fig


def unit_figure(sorting, fs, trig, unit_id, params, groups=None, duration_s=None):
    """Grid: one column per (group, edge), raster on top, PSTH below."""
    eng = PsthEngine(params)
    p = eng.p
    stim = p["stim_key"]
    groups = groups or list(trig.get("per_stim_group", {}).get(stim, {}).keys())
    cond = build_conditions(trig, stim, groups)
    sp = spike_times_s(sorting, int(unit_id), fs)
    fallback = sp.size / max(duration_s or (sp[-1] if sp.size else 1), 1e-12)
    keys = list(cond.keys())
    fig = new_figure((3.2 * max(1, len(keys)), 4.6))
    gs = fig.add_gridspec(2, max(1, len(keys)), height_ratios=[2, 1])
    for j, k in enumerate(keys):
        c = cond[k]
        ax_r = fig.add_subplot(gs[0, j]); ax_p = fig.add_subplot(gs[1, j], sharex=ax_r)
        rel, rate, _ = eng.collect(sp, c["times"])
        ax_r.set_title(f"{k[0]} {k[1]}  (n={len(c['times'])})", fontsize=9)
        if rel is not None:
            for tr, r in enumerate(rel, 1):
                if r.size:
                    ax_r.vlines(r, tr - 0.4, tr + 0.4, color="k", lw=0.5)
            umr = _mean_rate_windows(sp, c["isi"], c["total"], fallback)
            base, pr, pt, tgt, tau, ttau = eng.peak_and_tau(rate, umr)
            ax_p.plot(eng.bc, rate, color="k", lw=1.3)
            if np.isfinite(pt):
                ax_p.plot([pt], [pr], "o", color="tab:orange", ms=5)
                ax_p.axhline(tgt, color="gray", ls="--", lw=0.8)
            if np.isfinite(ttau):
                ax_p.axvline(ttau, color="tab:blue", ls="--", lw=0.8)
            if np.isfinite(c["interval"]):
                ax_p.axvline(c["interval"], color="gray", ls=":", lw=0.8)
        for a in (ax_r, ax_p):
            a.axvline(0, color="r", ls="--", lw=0.8)
        ax_p.set_xlim(-p["pre_time"], p["post_time"])
        ax_p.set_xlabel("Time (s)", fontsize=8)
        if j == 0:
            ax_r.set_ylabel("Trial"); ax_p.set_ylabel("Hz")
    fig.suptitle(f"Unit {unit_id} — {stim}  ({sp.size} spikes)", fontsize=10)
    fig.tight_layout()
    return fig
