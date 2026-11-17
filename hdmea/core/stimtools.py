"""Stimulus utilities: SO movie generator (Untitled2 cell 0), lux-sensor
TTL→light delay calibration (Untitled4) and camera-based TTL/screen delay
(Untitled2 cell 1), with the edge-case fixes noted in review."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .common import LogFn, ProgFn, _nolog, _noprog, new_figure

SO_DEFAULTS = dict(n_pix=200, retina_width_mm=4.0, f_min=0.21, f_max=2.1, n_scales=10,
                   frames_per_scale=100, percentile=10.0, seed=0)


# ----------------------------------------------------------------------
def generate_so_movie(params, out_path=None, log: LogFn = _nolog, progress: ProgFn = _noprog):
    p = {**SO_DEFAULTS, **(params or {})}
    n = int(p["n_pix"])
    rng = np.random.default_rng(int(p["seed"]))
    freqs = np.fft.fftfreq(n, d=p["retina_width_mm"] / n)
    FX, FY = np.meshgrid(freqs, freqs, indexing="xy")
    cutoffs = np.linspace(p["f_min"], p["f_max"], int(p["n_scales"]))
    movie = np.zeros((len(cutoffs) * int(p["frames_per_scale"]), n, n), np.float32)
    k = 0
    for i, fc in enumerate(cutoffs):
        mask = (np.abs(FX) <= fc) & (np.abs(FY) <= fc)
        for _ in range(int(p["frames_per_scale"])):
            img = (rng.random((n, n)) > 0.5).astype(float)
            filt = np.fft.ifft2(np.fft.fft2(img) * mask).real
            thr = np.percentile(filt, p["percentile"])
            movie[k] = np.where(filt < thr, -1.0, 1.0)
            k += 1
        progress((i + 1) / len(cutoffs))
        log(f"  cutoff {fc:.3f} cyc/mm done")
    if out_path:
        np.save(out_path, movie)
        log(f"Saved {movie.shape} → {out_path}")
    return movie, cutoffs


def so_preview_figure(movie, cutoffs, fps):
    fig = new_figure((10, 2.4))
    idx = np.linspace(0, len(cutoffs) - 1, min(5, len(cutoffs))).astype(int)
    for j, i in enumerate(idx):
        ax = fig.add_subplot(1, len(idx), j + 1)
        ax.imshow(movie[i * fps], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"{cutoffs[i]:.2f} cyc/mm", fontsize=8); ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
LUX_DEFAULTS = dict(smooth_w=5, k_sigma=6.0, search_win_s=0.25, min_dlux_thr=1.0)


def _otsu(x, nbins=256):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    hist, edges = np.histogram(x, bins=nbins)
    c = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(hist); w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * c) / np.maximum(w1, 1e-12)
    m2 = (np.cumsum((hist * c)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    return float(c[int(np.argmax(w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2))])


def lux_delay(run_dir, params, log: LogFn = _nolog):
    """G (TTL ON) → first lux ON edge delay. Fixes vs. the notebook: duplicate
    PsychoPy timestamps are averaged instead of producing dt=0, and the derivative
    threshold has a floor (min_dlux_thr) so it can't collapse to 0 when the lux
    reading is quantised (MAD = 0)."""
    p = {**LUX_DEFAULTS, **(params or {})}
    ev = pd.read_csv(os.path.join(run_dir, "events_GH.csv"))
    lux = pd.read_csv(os.path.join(run_dir, "lux_stream.csv"))
    te = "tThisFlipGlobal" if "tThisFlipGlobal" in ev else "t_s_psychopy"
    tl = "tThisFlipGlobal" if "tThisFlipGlobal" in lux else "t_s_psychopy"
    n_dup = int(lux[tl].duplicated().sum())
    lux = lux.groupby(tl, as_index=False)["lux"].mean().sort_values(tl)
    t = lux[tl].to_numpy(float); y = lux["lux"].to_numpy(float)
    G = np.sort(ev.loc[ev["event"].astype(str).str.upper() == "G", te].to_numpy(float))
    H = np.sort(ev.loc[ev["event"].astype(str).str.upper() == "H", te].to_numpy(float))
    ys = pd.Series(y).rolling(int(p["smooth_w"]), center=True, min_periods=1).mean().to_numpy()
    dl = np.diff(ys) / np.diff(t); td = t[1:]
    lvl = _otsu(ys)
    lo = np.median(ys[ys < lvl]) if np.any(ys < lvl) else np.nan
    hi = np.median(ys[ys >= lvl]) if np.any(ys >= lvl) else np.nan
    band = 0.15 * (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else 0.0
    steady = (ys[1:] < lvl - band) | (ys[1:] > lvl + band)
    ds = dl[steady]
    med = float(np.nanmedian(ds)) if ds.size else 0.0
    mad = float(np.nanmedian(np.abs(ds - med))) * 1.4826 if ds.size else 0.0
    thr_raw = med + p["k_sigma"] * mad
    thr = max(thr_raw, float(p["min_dlux_thr"]))
    m_on = (dl > thr) & (ys[1:] > lvl)
    cross = np.sort(td[np.where(m_on[1:] & ~m_on[:-1])[0] + 1])
    rows = []
    for g in G:
        k = np.searchsorted(cross, g)
        if k < cross.size and cross[k] <= g + p["search_win_s"]:
            rows.append(dict(G_time_s=g, ON_edge_time_s=cross[k], delay_ms=(cross[k] - g) * 1000))
    df = pd.DataFrame(rows)
    log(f"Lux samples {len(t)} (merged {n_dup} duplicate timestamps) | level thr {lvl:.2f} lux | "
        f"dLux thr {thr:.2f} lux/s" + (f" (raw {thr_raw:.2f} → floor applied)" if thr_raw < thr else ""))
    if len(df):
        log(f"G→ON delay: mean {df.delay_ms.mean():.1f} ms | median {df.delay_ms.median():.1f} ms | "
            f"SD {df.delay_ms.std(ddof=0):.1f} ms | n={len(df)}/{len(G)}")
    else:
        log("No ON edges found within the search window")
    return dict(t=t, y=y, ys=ys, td=td, dl=dl, G=G, H=H, lvl=lvl, thr=thr, edges=df, smooth_w=p["smooth_w"])


def lux_figure(res):
    fig = new_figure((11, 5))
    a1 = fig.add_subplot(211); a2 = fig.add_subplot(212, sharex=a1)
    a1.plot(res["t"], res["y"], lw=0.8, label="lux"); a1.plot(res["t"], res["ys"], lw=0.8, label="smoothed")
    a1.axhline(res["lvl"], ls="--", color="gray", lw=0.8)
    for g in res["G"]:
        a1.axvline(g, ls="--", alpha=0.3, color="tab:green")
    for h in res["H"]:
        a1.axvline(h, ls=":", alpha=0.3, color="tab:purple")
    if len(res["edges"]):
        for e in res["edges"].ON_edge_time_s:
            a1.axvline(e, color="tab:red", lw=1)
    a1.set_ylabel("Lux"); a1.legend(fontsize=8, loc="upper right")
    a1.set_title("Lux + G (green dashed) / H (purple dotted) / detected ON edges (red)", fontsize=9)
    a2.plot(res["td"], res["dl"], lw=0.8); a2.axhline(res["thr"], ls="--", color="tab:red", lw=0.8)
    a2.set_ylabel("dLux/dt"); a2.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
def camera_ttl_delay(csv_path, fps=29.99, max_lag_frames=15, log: LogFn = _nolog):
    """Screen (Mean1) vs LED/TTL (Mean2) brightness from a camera ROI measurement.
    Cross-correlation lag, plus per-edge delays where each LED edge is paired to the
    nearest screen edge within ±max_lag_frames (the notebook paired by order, which
    mis-matched edges when the counts differed)."""
    df = pd.read_csv(csv_path)
    cmap = {c.lower(): c for c in df.columns}
    m1 = df[cmap.get("mean1", "Mean1")].to_numpy(float); m2 = df[cmap.get("mean2", "Mean2")].to_numpy(float)
    z1 = (m1 - m1.mean()) / (m1.std() + 1e-9); z2 = (m2 - m2.mean()) / (m2.std() + 1e-9)
    corr = np.correlate(z2, z1, mode="full")
    lags = np.arange(-len(z1) + 1, len(z2))
    win = np.abs(lags) <= max(max_lag_frames, 1) * 4
    best = int(lags[win][np.argmax(corr[win])])

    def edges(x):
        d = np.diff(x)
        thr = 0.5 * (np.max(d) + np.median(d))
        e = np.where(d > thr)[0] + 1
        keep = np.r_[True, np.diff(e) > 1] if e.size else e.astype(bool)
        return e[keep]

    e1, e2 = edges(m1), edges(m2)
    pairs = []
    for b in e2:
        if e1.size:
            j = int(np.argmin(np.abs(e1 - b)))
            if abs(int(e1[j]) - int(b)) <= max_lag_frames:
                pairs.append((int(e1[j]), int(b), int(b) - int(e1[j])))
    dt = 1000.0 / fps
    out = dict(xcorr_lag_frames=best, xcorr_lag_ms=best * dt, n_screen_edges=int(e1.size), n_led_edges=int(e2.size),
               pairs=pairs, edge_delay_ms=(float(np.mean([p[2] for p in pairs])) * dt if pairs else np.nan),
               m1=m1, m2=m2, e1=e1, e2=e2, fps=fps)
    log(f"Cross-corr lag: {best} frames = {best * dt:.1f} ms (LED {'leads' if best < 0 else 'lags'} screen)\n"
        f"Edges: screen {e1.size}, LED {e2.size}, paired {len(pairs)} → mean {out['edge_delay_ms']:.1f} ms "
        f"(resolution ±{dt:.0f} ms)")
    return out


def camera_figure(res):
    fig = new_figure((10, 3.2))
    ax = fig.add_subplot(111)
    fr = np.arange(len(res["m1"]))
    ax.plot(fr, (res["m1"] - res["m1"].min()) / (np.ptp(res["m1"]) or 1), label="screen (Mean1)")
    ax.plot(fr, (res["m2"] - res["m2"].min()) / (np.ptp(res["m2"]) or 1), label="LED/TTL (Mean2)")
    for a, b, _ in res["pairs"]:
        ax.annotate("", xy=(b, 1.05), xytext=(a, 1.05), arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
    ax.set_xlabel("Camera frame"); ax.set_ylabel("normalised"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
