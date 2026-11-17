"""Pairwise cross-correlation (CCF) of spontaneous activity between nearby units,
classified as No-peak / Unimodal-narrow / Unimodal-wide / Bimodal
(Biocam_sorting cells 50–56). Electrical (gap-junction) coupling typically gives
a narrow peak within a few ms of 0, bimodal peaks flank 0.

The CCF is the notebook's mean-centred correlation of 1-ms binned counts, but it
is computed exactly from spike times only within the lag window (no dense FFT),
so whole-array runs are fast."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from .common import LogFn, ProgFn, _nolog, _noprog, check_cancel, new_figure

DEFAULTS = dict(
    bin_size_s=0.001, lag_ms=50.0, min_spikes=100, max_dist_um=500.0, smooth_sigma_ms=0.8,
    snr_min=5.0, min_width_ms=4.0, narrow_wide_split_ms=15.0, min_abs_peak=0.8, edge_factor=1.5,
    mean_ratio_min=3.0, central_peak_limit_ms=7.0, bimo_min_rel_height=0.70, bimo_min_lag_diff_ms=4.0,
    bimo_max_lag_diff_ms=12.0, bimo_valley_drop=0.25, bimo_valley_noise_sd=0.5,
    far_dom_lag_ms=15.0, far_dom_ratio=1.5,
)


class BinnedTrains:
    def __init__(self, sorting, fs, t_start, t_end, bin_s):
        self.t0, self.t1, self.bin = float(t_start), float(t_end), float(bin_s)
        self.n = int(np.ceil((self.t1 - self.t0) / self.bin))
        self.idx = {}
        for u in sorting.unit_ids:
            st = np.asarray(sorting.get_unit_spike_train(u), float) / fs
            st = st[(st >= self.t0) & (st < self.t1)]
            b = np.floor((st - self.t0) / self.bin).astype(np.int64)
            self.idx[int(u)] = np.sort(np.clip(b, 0, self.n - 1))

    def count(self, u):
        return int(self.idx[int(u)].size)

    def ccf(self, ua, ub, K):
        """c[k] = Σ_t (a[t+k]-ā)(b[t]-b̄), k = -K..K (same as np.correlate(a0, b0, 'full'))."""
        ia, ib = self.idx[int(ua)], self.idx[int(ub)]
        n = self.n
        lags = np.arange(-K, K + 1)
        if ia.size == 0 or ib.size == 0:
            return lags, np.zeros(lags.size)
        lo = np.searchsorted(ia, ib - K, "left")
        hi = np.searchsorted(ia, ib + K, "right")
        cnt = hi - lo
        tot = int(cnt.sum())
        raw = np.zeros(2 * K + 1)
        if tot:
            rep_b = np.repeat(ib, cnt)
            starts = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
            pos = np.arange(tot) + starts
            raw = np.bincount(ia[pos] - rep_b + K, minlength=2 * K + 1).astype(float)
        abar, bbar = ia.size / n, ib.size / n
        # overlap of valid t for lag k: t in [max(0,-k), min(n, n-k))
        tlo = np.maximum(0, -lags); thi = np.minimum(n, n - lags)
        sum_b = np.searchsorted(ib, thi, "left") - np.searchsorted(ib, tlo, "left")
        sum_a = np.searchsorted(ia, thi + lags, "left") - np.searchsorted(ia, tlo + lags, "left")
        cc = raw - abar * sum_b - bbar * sum_a + (thi - tlo) * abar * bbar
        return lags, cc


def classify(lags_ms, ccf, p):
    lags_ms = np.asarray(lags_ms, float); ccf = np.asarray(ccf, float)
    dt = lags_ms[1] - lags_ms[0]
    sm = gaussian_filter1d(ccf, sigma=max(1, int(round(p["smooth_sigma_ms"] / dt))))
    cm = np.abs(lags_ms) <= 50.0
    lc, cc = lags_ms[cm], sm[cm]
    bm = np.abs(lags_ms) > 40.0
    bsd = np.std(sm[bm]) if np.any(bm) else np.std(sm)
    nope = ("No-peak", 0.0, np.nan, [])
    if bsd <= 0:
        return nope
    peaks, _ = find_peaks(cc, prominence=2.0 * bsd)
    if len(peaks) == 0:
        return nope
    ph = cc[peaks]; bi = int(np.argmax(ph)); bh = float(ph[bi]); bl = float(lc[peaks[bi]])
    snr = bh / bsd
    if snr < p["snr_min"] or bh < p["min_abs_peak"]:
        return "No-peak", snr, np.nan, []
    mc = float(np.mean(cc))
    if mc <= 0 or bh / mc < p["mean_ratio_min"]:
        return "No-peak", snr, np.nan, []
    ne = min(5, len(cc) // 4)
    if bh <= p["edge_factor"] * float(np.max(np.r_[cc[:ne], cc[-ne:]])):
        return "No-peak", snr, np.nan, []
    if abs(bl) > p["central_peak_limit_ms"]:
        return "No-peak", snr, np.nan, []
    w = float(peak_widths(cc, [peaks[bi]], rel_height=0.5)[0][0] * dt)
    if w < p["min_width_ms"]:
        return "No-peak", snr, w, [bl]
    if len(peaks) >= 2:
        o = np.argsort(ph)[::-1][:2]
        p1, p2 = peaks[o]; h1, h2 = float(ph[o[0]]), float(ph[o[1]])
        l1, l2 = float(lc[p1]), float(lc[p2]); ld = abs(l2 - l1)
        if (h2 >= p["bimo_min_rel_height"] * h1 and p["bimo_min_lag_diff_ms"] <= ld <= p["bimo_max_lag_diff_ms"]
                and abs(l1) <= p["central_peak_limit_ms"] and abs(l2) <= p["central_peak_limit_ms"]):
            i0, i1 = sorted([p1, p2]); v = float(np.min(cc[i0:i1 + 1]))
            if v < (1 - p["bimo_valley_drop"]) * min(h1, h2) and v < min(h1, h2) - p["bimo_valley_noise_sd"] * bsd:
                return "Bimodal", snr, ld, [l1, l2]
    far = np.abs(lc[peaks]) >= p["far_dom_lag_ms"]
    if np.any(far) and bh < p["far_dom_ratio"] * float(np.max(ph[far])):
        return "No-peak", snr, np.nan, []
    return ("Unimodal-narrow" if w < p["narrow_wide_split_ms"] else "Unimodal-wide"), snr, w, [bl]


def run_coupling(sorting, fs, window, unit_xy: dict, out_folder, params, hubs=None,
                 log: LogFn = _nolog, progress: ProgFn = _noprog, cancel=None):
    """window = (t_start, t_end) in s. Each unordered pair is evaluated once (hub < partner)."""
    p = {**DEFAULTS, **(params or {})}
    bt = BinnedTrains(sorting, fs, window[0], window[1], p["bin_size_s"])
    K = int(round(p["lag_ms"] / 1000 / p["bin_size_s"]))
    lags_ms = np.arange(-K, K + 1) * p["bin_size_s"] * 1000
    units = [u for u in sorted(bt.idx) if bt.count(u) >= p["min_spikes"] and u in unit_xy]
    log(f"Window {window[0]:.1f}–{window[1]:.1f} s | {len(units)} units with ≥{p['min_spikes']} spikes and a location")
    xy = np.array([unit_xy[u] for u in units]) if units else np.zeros((0, 2))
    all_hubs = not hubs
    hubs = units if all_hubs else [u for u in hubs if u in units]
    hub_set = set(hubs)
    rows = []
    uidx = {u: i for i, u in enumerate(units)}
    for k, h in enumerate(hubs, 1):
        check_cancel(cancel)
        d = np.hypot(*(xy - xy[uidx[h]]).T)
        for j in np.where((d <= p["max_dist_um"]) & (np.arange(len(units)) != uidx[h]))[0]:
            q = units[j]
            if q in hub_set and q < h:   # each unordered pair only once
                continue
            _, cc = bt.ccf(h, q, K)
            cat, snr, w, pl = classify(lags_ms, cc, p)
            rows.append(dict(hub=h, partner=q, distance_um=float(d[j]), category=cat, coupling_strength=float(snr),
                             peak_width_ms=float(w) if np.isfinite(w) else np.nan,
                             peak_lags_ms=";".join(f"{x:.1f}" for x in pl)))
        if k % 10 == 0 or k == len(hubs):
            progress(k / max(1, len(hubs)))
    df = pd.DataFrame(rows, columns=["hub", "partner", "distance_um", "category", "coupling_strength",
                                     "peak_width_ms", "peak_lags_ms"])
    od = os.path.join(out_folder, "CCF_coupling"); os.makedirs(od, exist_ok=True)
    df.to_csv(os.path.join(od, "ccf_pairs.csv"), index=False)
    log(f"Pairs evaluated: {len(df)} → {od}\n" + df["category"].value_counts().to_string())
    return df, bt


def pair_figure(bt: BinnedTrains, a, b, p, row=None):
    p = {**DEFAULTS, **(p or {})}
    K = int(round(p["lag_ms"] / 1000 / p["bin_size_s"]))
    lags, cc = bt.ccf(a, b, K)
    lags_ms = lags * p["bin_size_s"] * 1000
    sm = gaussian_filter1d(cc, sigma=max(1, int(round(p["smooth_sigma_ms"] / (lags_ms[1] - lags_ms[0])))))
    cat, snr, w, pl = classify(lags_ms, cc, p)
    fig = new_figure((6, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(lags_ms, cc, color="0.75", lw=0.8, label="raw")
    ax.plot(lags_ms, sm, color="k", lw=1.5, label=f"smoothed σ={p['smooth_sigma_ms']} ms")
    ax.axvline(0, ls=":", lw=1)
    for l in pl:
        ax.axvline(l, ls="--", color="tab:red", lw=1)
    ax.set_xlabel("Lag (ms)  (+ = first unit later)"); ax.set_ylabel("CCF (a.u.)")
    ax.set_title(f"{a} × {b}: {cat}, SNR={snr:.1f}" + (f", width={w:.1f} ms" if np.isfinite(w) else ""), fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def network_figure(df, unit_xy, categories=("Unimodal-narrow", "Unimodal-wide", "Bimodal"), min_snr=0.0,
                   highlight=None):
    fig = new_figure((6.4, 6))
    ax = fig.add_subplot(111)
    xy = np.array(list(unit_xy.values())) if unit_xy else np.zeros((0, 2))
    if xy.size:
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c="0.8")
    col = {"Unimodal-narrow": "tab:red", "Unimodal-wide": "tab:orange", "Bimodal": "tab:blue"}
    sub = df[df.category.isin(categories) & (df.coupling_strength >= min_snr)]
    if highlight is not None:
        sub = sub[(sub.hub == highlight) | (sub.partner == highlight)]
    for c in categories:
        s = sub[sub.category == c]
        for _, r in s.iterrows():
            (x1, y1), (x2, y2) = unit_xy[int(r.hub)], unit_xy[int(r.partner)]
            ax.plot([x1, x2], [y1, y2], color=col.get(c, "k"), lw=0.8, alpha=0.8)
        ax.plot([], [], color=col.get(c, "k"), label=f"{c} ({len(s)})")
    if highlight is not None and highlight in unit_xy:
        ax.plot(*unit_xy[highlight], "k*", ms=12)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title("CCF coupling network" + (f" — hub {highlight}" if highlight is not None else ""), fontsize=9)
    fig.tight_layout()
    return fig
