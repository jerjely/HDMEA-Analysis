"""Oscillatory 'ringing' of light responses (SO_Final cell 20): baseline-subtracted
fine PSTH, candidate peaks, best consecutive peak triplet (lowest period CV),
f_triplet (descriptive) and osc_strict (CV <= max). CTRL vs drug, paired summary."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .common import LogFn, ProgFn, _nolog, _noprog, check_cancel, new_figure, merge_into_master, spike_times_s
from .triggers import get_times

DEFAULTS = dict(
    stim_key="L", edge="ON", suffix="first40_dualf", n_trials_limit=40,
    pre_time=0.2, post_time=0.6, bin_size=0.0025, smooth_sigma_bins=2.0,
    ring_t0=-0.01, ring_t1=0.30, min_peak_rate_hz=5.0, min_peak_dist_s=0.01, prom_frac=0.12,
    min_peaks=3, max_cv_period=0.45, require_damping=False, allow_damping_bumps=1,
    ref_group="CTRL", test_group="PTX",
)


def _col(stim, grp, edge, metric, suffix):
    return f"{stim}_{grp}_{edge}_{metric}_{suffix}".lower()


class RingEngine:
    def __init__(self, params):
        self.p = {**DEFAULTS, **(params or {})}

    def events(self, trig, grp):
        p = self.p
        ev = get_times(trig, p["stim_key"], grp, p["edge"])
        return ev[:min(len(ev), int(p["n_trials_limit"]))]

    def psth(self, spikes, ev):
        p = self.p
        bins = np.arange(-p["pre_time"], p["post_time"] + p["bin_size"], p["bin_size"])
        bc = (bins[:-1] + bins[1:]) / 2
        rel = []
        for t0 in ev:
            lo = np.searchsorted(spikes, t0 - p["pre_time"]); hi = np.searchsorted(spikes, t0 + p["post_time"], "right")
            rel.append(spikes[lo:hi] - t0)
        flat = np.concatenate(rel) if rel and any(r.size for r in rel) else np.array([])
        counts, _ = np.histogram(flat, bins=bins)
        rate = counts / (p["bin_size"] * max(1, len(ev)))
        sm = gaussian_filter1d(rate.astype(float), p["smooth_sigma_bins"]) if p["smooth_sigma_bins"] > 0 else rate.astype(float)
        base = sm[bc < 0].mean() if np.any(bc < 0) else 0.0
        return bc, sm - base, rel

    def detect(self, bc, rate):
        p = self.p
        empty = (np.array([]), np.array([]), np.array([]), np.array([]), np.nan, [], np.nan, False)
        w = (bc >= p["ring_t0"]) & (bc <= p["ring_t1"])
        x, t = rate[w], bc[w]
        if x.size < 5:
            return empty
        amax = float(np.max(x))
        if not np.isfinite(amax) or amax < p["min_peak_rate_hz"]:
            return empty
        peaks, _ = find_peaks(x, prominence=max(1e-6, p["prom_frac"] * amax),
                              distance=max(1, int(round(p["min_peak_dist_s"] / p["bin_size"]))))
        peaks = np.array([q for q in peaks if x[q] >= p["min_peak_rate_hz"]], dtype=int)
        if peaks.size == 0:
            return empty
        ct, ca = t[peaks], x[peaks]
        if ct.size < p["min_peaks"]:
            return ct, ca, np.array([]), np.array([]), np.nan, [], np.nan, False

        def cv(tp):
            per = np.diff(tp)
            if per.size < 2 or np.mean(per) <= 0:
                return np.inf, per
            return float(np.std(per) / np.mean(per)), per

        best, best_cv, best_per = None, np.inf, None
        for i in range(len(ct) - 2):
            c, per = cv(ct[i:i + 3])
            if not np.isfinite(c):
                continue
            if p["require_damping"] and int(np.sum(np.diff(ca[i:i + 3]) > 0)) > p["allow_damping_bumps"]:
                continue
            if c < best_cv:
                best, best_cv, best_per = (ct[i:i + 3], ca[i:i + 3]), c, per
        if best is None:
            ut, ua = ct[:3], ca[:3]
            best_cv, best_per = cv(ut)
        else:
            ut, ua = best
        T = float(np.median(best_per)) if best_per is not None and best_per.size else np.nan
        f = 1.0 / T if np.isfinite(T) and T > 0 else np.nan
        osc = bool(np.isfinite(best_cv) and best_cv <= p["max_cv_period"])
        return ct, ca, ut, ua, f, (1000 * best_per).tolist() if best_per is not None else [], float(best_cv), osc


def run_ringing(sorting, fs, trig, out_folder, params, log: LogFn = _nolog, progress: ProgFn = _noprog,
                cancel=None):
    eng = RingEngine(params)
    p = eng.p
    groups = [p["ref_group"], p["test_group"]]
    ev = {g: eng.events(trig, g) for g in groups}
    for g in groups:
        log(f"  {p['stim_key']} {g} {p['edge']}: using {len(ev[g])} events")
    ids = np.array(sorted(list(sorting.unit_ids)), dtype=int)
    e = p["edge"].lower()
    rows = []
    for k, u in enumerate(ids, 1):
        check_cancel(cancel)
        sp = spike_times_s(sorting, int(u), fs)
        row = {"unit_id": int(u)}
        for g in groups:
            bc, rate, _ = eng.psth(sp, ev[g])
            ct, ca, ut, ua, f, dms, cvp, osc = eng.detect(bc, rate)
            row.update({_col(p["stim_key"], g, e, "ring_triplet_f_hz", p["suffix"]): f,
                        _col(p["stim_key"], g, e, "ring_triplet_cv", p["suffix"]): cvp,
                        _col(p["stim_key"], g, e, "ring_triplet_n_cand", p["suffix"]): int(len(ct)),
                        _col(p["stim_key"], g, e, "ring_osc_strict", p["suffix"]): int(osc)})
        rows.append(row)
        if k % 100 == 0 or k == len(ids):
            progress(k / len(ids))
    df = pd.DataFrame(rows)
    merge_into_master(df, os.path.join(out_folder, "unit_metrics_master.csv"), log=log)
    paired = paired_table(df, p)
    log(f"Reference-group strict oscillators: n={len(paired)}; lost in {p['test_group']}: "
        f"{int((paired['f_test_zero'] == 0).sum()) if len(paired) else 0}")
    return df, paired


def paired_table(df, p):
    st, e, s = p["stim_key"], p["edge"].lower(), p["suffix"]
    fr, ft = _col(st, p["ref_group"], e, "ring_triplet_f_hz", s), _col(st, p["test_group"], e, "ring_triplet_f_hz", s)
    orf, ot = _col(st, p["ref_group"], e, "ring_osc_strict", s), _col(st, p["test_group"], e, "ring_osc_strict", s)
    if fr not in df:
        return pd.DataFrame(columns=["unit_id", "f_ref", "f_test_zero", "df"])
    sub = df[(df[orf] == 1) & np.isfinite(df[fr])].copy()
    sub["f_ref"] = sub[fr].astype(float)
    sub["f_test_zero"] = np.where((sub[ot] == 1) & np.isfinite(sub[ft]), sub[ft].astype(float), 0.0)
    sub["df"] = sub["f_test_zero"] - sub["f_ref"]
    return sub[["unit_id", "f_ref", "f_test_zero", "df"]]


def unit_figure(sorting, fs, trig, unit_id, params):
    eng = RingEngine(params)
    p = eng.p
    sp = spike_times_s(sorting, int(unit_id), fs)
    groups = [p["ref_group"], p["test_group"]]
    fig = new_figure((13, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 2, 1, 2])
    for j, g in enumerate(groups):
        ev = eng.events(trig, g)
        bc, rate, rel = eng.psth(sp, ev)
        ct, ca, ut, ua, f, dms, cvp, osc = eng.detect(bc, rate)
        ar = fig.add_subplot(gs[0, 2 * j]); ap = fig.add_subplot(gs[0, 2 * j + 1])
        for i, r in enumerate(rel):
            if r.size:
                ar.vlines(r, i + 0.6, i + 1.4, lw=0.45, color="k")
        ar.axvline(0, color="r", ls="--", lw=1); ar.set_title(f"{g} raster", fontsize=9)
        ar.set_xlabel(f"Time from {p['edge']} (s)")
        ap.plot(bc, rate, lw=1.8)
        ap.axvline(0, color="r", ls="--", lw=1)
        ap.axvspan(p["ring_t0"], p["ring_t1"], alpha=0.08)
        if ct.size:
            ap.plot(ct, ca, "o", ms=5, alpha=0.35, color="gray")
        if ut.size:
            ap.plot(ut, ua, "o", ms=7, color="tab:orange")
        ttl = f"Unit {unit_id} {g} | osc={int(osc)}" + (f" | f≈{f:.1f} Hz" if np.isfinite(f) else "")
        ap.set_title(ttl, fontsize=9); ap.set_ylabel("Δ rate (Hz)")
        ap.text(0.99, 0.02, f"Δt(ms)={np.round(dms, 1)} CV≈{cvp:.2f}" if ut.size else "triplet: NA",
                transform=ap.transAxes, ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    return fig


def summary_figure(paired, p):
    fig = new_figure((11, 3.6))
    a1, a2, a3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
    if len(paired):
        fr, ft, d = paired["f_ref"].values, paired["f_test_zero"].values, paired["df"].values
        a1.hist(d[np.isfinite(d)], bins=40); a1.axvline(0, ls="--", lw=1, color="k")
        a1.set_xlabel(f"Δf = {p['test_group']} − {p['ref_group']} (Hz)"); a1.set_ylabel("Units")
        a2.boxplot([fr, ft], positions=[1, 2], widths=0.5, showfliers=False)
        for y1, y2 in zip(fr, ft):
            a2.plot([1, 2], [y1, y2], alpha=0.35, lw=0.8, color="gray")
        a2.set_xticks([1, 2]); a2.set_xticklabels([p["ref_group"], p["test_group"]]); a2.set_ylabel("f (Hz)")
        mx = float(np.nanmax(np.r_[fr, ft, [1.0]]))
        bins = np.arange(0, mx + 1, 1.0)
        a3.hist(fr, bins=bins, alpha=0.45, label=p["ref_group"])
        a3.hist(ft, bins=bins, alpha=0.45, label=f"{p['test_group']} (0 if lost)")
        a3.legend(fontsize=8); a3.set_xlabel("Oscillation frequency (Hz)")
    a1.set_title(f"n={len(paired)} {p['ref_group']} oscillators", fontsize=9)
    fig.tight_layout()
    return fig
