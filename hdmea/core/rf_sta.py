"""Sparse-object (SO) receptive-field mapping by spike-triggered average.
Port of SO_Final cell 10 ("noise-template + residual + local ROI"):
  STA at several lags -> best lag = most negative -> jitter-null z-map ->
  residual = z_best − z_noise (median of FAR lags) -> ROI around residual minimum ->
  connected component containing the minimum = RF; weighted centroid."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .common import (LogFn, ProgFn, _nolog, _noprog, check_cancel, new_figure, find_first_existing,
                     best_offset_pair, trim_pair, save_json, spike_times_s)

DEFAULTS = dict(
    lags="0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25",
    n_null=40, jitter_s=0.5, seed=0, roi_radius=45, rel_frac=0.35, min_rf_pix=60,
    resid_smooth_sigma=2.0, display_smooth=0.5, min_spikes=500, min_jitter_spikes=200,
    far_lag_gap_s=0.15, min_far_lags=3, max_null_spikes=6000, save_pngs=True, save_failed_pngs=False,
    max_unit_id=0, include_ids="",
)
SO_FRAMES_CANDS = ["SO_frames.csv", "so_frames.csv"]
PSY_FRAMES_CANDS = ["psychopy_frames.csv", "psychoPy_frames.csv", "psy_frames.csv"]


def parse_lags(s):
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, float)
    return np.array([float(x) for x in str(s).replace(";", ",").split(",") if x.strip()], float)


class SoTiming:
    """Stimulus movie + frame timestamps converted to the MEA timebase."""

    def __init__(self, so_npy, so_folder, so_on_times_mea, gap_fix=True, log: LogFn = _nolog):
        self.stim = np.load(so_npy, mmap_mode="r")
        if self.stim.dtype != np.float32:
            self.stim = self.stim.astype(np.float32)
        self.mu32 = self.stim.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.T, self.H, self.W = self.stim.shape
        df = pd.read_csv(find_first_existing(so_folder, SO_FRAMES_CANDS))
        tcol = "tThisFlipGlobal" if "tThisFlipGlobal" in df else "t_s_psychopy" if "t_s_psychopy" in df else None
        fcol = "frame_index" if "frame_index" in df else "frame" if "frame" in df else None
        if tcol is None or fcol is None:
            raise KeyError(f"SO_frames.csv needs tThisFlipGlobal/t_s_psychopy and frame_index/frame; got {list(df.columns)}")
        t_psy = df[tcol].to_numpy(np.float64)
        frames = df[fcol].to_numpy(np.int64)
        psy = pd.read_csv(find_first_existing(so_folder, PSY_FRAMES_CANDS))
        if "SO_stim.started" not in psy:
            raise KeyError("psychopy_frames.csv missing 'SO_stim.started'")
        psy_on = psy["SO_stim.started"].dropna().to_numpy(np.float64)
        mea_on = np.sort(np.asarray(so_on_times_mea, float))
        if mea_on.size == 0 or psy_on.size == 0:
            raise ValueError("No SO ON triggers or no SO_stim.started entries")
        a = psy_on[:len(mea_on)] if len(mea_on) < len(psy_on) else psy_on
        mea_al, self.off_idx = best_offset_pair(a, mea_on)
        pp, mp = trim_pair(a, mea_al)
        self.offset = float(np.median(mp - pp))
        self.n_pairs = len(pp)
        log(f"PsychoPy→MEA offset: {self.offset:.4f} s (pairs={self.n_pairs}, idx offset={self.off_idx})")
        tf = (t_psy + self.offset).astype(np.float32)
        o = np.argsort(tf)
        self.t_frames, self.frames = tf[o], frames[o]
        self.tmin, self.tmax = float(self.t_frames[0]), float(self.t_frames[-1])
        dts = np.diff(self.t_frames)
        self.frame_dt = float(np.median(dts)) if dts.size else 0.0
        # gap fix: a spike more than 1.5 frame intervals after the last flip is not
        # "seeing" that frame (inter-trial gray screen) – the notebook counted it.
        self.max_age = 1.5 * self.frame_dt if gap_fix and self.frame_dt > 0 else np.inf
        n_gap = int(np.sum(dts > 1.5 * self.frame_dt)) if self.frame_dt > 0 else 0
        log(f"SO frames: {len(self.frames)} flips, frame interval {self.frame_dt * 1000:.1f} ms, "
            f"{n_gap} gap(s) between trials" + (" (excluded)" if gap_fix else ""))

    def sta(self, spikes, lag):
        tq = spikes - np.float32(lag)
        ii = np.searchsorted(self.t_frames, tq) - 1
        ok = (ii >= 0) & (ii < len(self.frames))
        if np.isfinite(self.max_age):
            ok &= (tq - self.t_frames[np.clip(ii, 0, len(self.frames) - 1)]) <= self.max_age
        ii = ii[ok]
        if ii.size == 0:
            return np.full((self.H, self.W), np.nan, np.float32), 0
        fr = (self.frames[ii] % self.T).astype(np.int64)
        cnt = np.bincount(fr, minlength=self.T).astype(np.float32)
        nz = np.flatnonzero(cnt)
        w = cnt[nz] / float(fr.size)
        acc = np.tensordot(w, np.asarray(self.stim[nz], np.float32) - self.mu32[None], axes=(0, 0))
        return acc.astype(np.float32), int(fr.size)


def _component(mask, z, minpix):
    from skimage.measure import label
    lab0 = label(mask.astype(bool))
    sizes = np.bincount(lab0.ravel())
    keep = sizes >= int(minpix); keep[0] = False      # drop objects smaller than minpix
    m = keep[lab0]
    lab = label(m)
    if lab.max() == 0:
        return m, None, (np.nan, np.nan), "no_components"
    ymin, xmin = np.unravel_index(np.nanargmin(z), z.shape)
    k = lab[ymin, xmin]
    if k == 0:
        return m, None, (np.nan, np.nan), "min_not_in_component"
    comp = lab == k
    w = (-z).clip(min=0) * comp
    s = float(w.sum())
    ys, xs = np.indices(z.shape)
    if s <= 0:
        yy, xx = np.where(comp)
        cy, cx = float(yy.mean()), float(xx.mean())
    else:
        cy, cx = float((ys * w).sum() / s), float((xs * w).sum() / s)
    return m, comp, (cy, cx), "ok"


def analyse_unit(timing: SoTiming, spikes, p, rng):
    """Returns dict with maps + result row (or row only if rejected early)."""
    from skimage.draw import disk
    lags = parse_lags(p["lags"])
    stas = np.stack([timing.sta(spikes, float(l))[0] for l in lags])
    mins = np.array([np.nanmin(s) if np.isfinite(s).any() else np.nan for s in stas])
    if not np.isfinite(mins).any():
        return dict(row=dict(ok=False, reason="no_frames_in_range"))
    bi = int(np.nanargmin(mins))
    best, best_lag = stas[bi], float(lags[bi])
    sp0 = spikes
    if p["max_null_spikes"] and spikes.size > int(p["max_null_spikes"]):
        sp0 = spikes[rng.choice(spikes.size, int(p["max_null_spikes"]), replace=False)]
    mean = np.zeros((timing.H, timing.W), np.float32); M2 = np.zeros_like(mean); n = 0
    for _ in range(int(p["n_null"])):
        sj = sp0 + rng.uniform(-p["jitter_s"], p["jitter_s"], sp0.size).astype(np.float32)
        sj = sj[(sj > timing.tmin) & (sj < timing.tmax)]
        if sj.size < p["min_jitter_spikes"]:
            continue
        img, _ = timing.sta(np.sort(sj), best_lag)
        if not np.isfinite(img).any():
            continue
        n += 1
        d = img - mean; mean += d / n; M2 += d * (img - mean)
    row = dict(best_lag_s=best_lag, raw_min_val=float(mins[bi]), n_null=int(n))
    if n < max(10, int(p["n_null"]) // 3):
        row.update(ok=False, reason="too_few_null_samples")
        return dict(row=row)
    sd = np.sqrt(M2 / max(1, n - 1) + 1e-9).astype(np.float32)
    z_best = (best - mean) / sd
    far = np.where(np.abs(lags - best_lag) >= p["far_lag_gap_s"])[0]
    if far.size < p["min_far_lags"]:
        far = np.array([i for i in range(len(lags)) if i != bi], int)
    z_noise = (np.nanmedian(stas[far], axis=0) - mean) / sd
    resid = gaussian_filter((z_best - z_noise).astype(np.float32), p["resid_smooth_sigma"])
    rmin = float(np.nanmin(resid))
    ymin, xmin = np.unravel_index(np.nanargmin(resid), resid.shape)
    roi = np.zeros(resid.shape, bool)
    rr, cc = disk((ymin, xmin), int(p["roi_radius"]), shape=roi.shape); roi[rr, cc] = True
    thr = float(p["rel_frac"] * rmin)
    mask = (resid <= thr) & roi; mask[ymin, xmin] = True
    m_clean, comp, (cy, cx), reason = _component(mask, resid, p["min_rf_pix"])
    ok = comp is not None and comp.sum() > 0
    row.update(resid_min=rmin, thr=thr, min_x=int(xmin), min_y=int(ymin), mask_area=int(m_clean.sum()),
               comp_area=int(comp.sum()) if comp is not None else 0, centroid_cx_px=float(cx),
               centroid_cy_px=float(cy), n_far_lags=int(far.size), ok=bool(ok), reason=reason)
    return dict(row=row, stas=stas, lags=lags, best=best, z_best=z_best, z_noise=z_noise, resid=resid,
                roi=roi, mask=m_clean, comp=comp, cx=cx, cy=cy, best_lag=best_lag)


def panel_figure(unit_id, res, display_smooth=0.5):
    best = res["best"]
    fin = best[np.isfinite(best)]
    v = max(float(np.nanpercentile(np.abs(fin), 99)) if fin.size else 1.0, 1e-12)
    disp = gaussian_filter(best, display_smooth) if display_smooth else best
    ymin, xmin = np.unravel_index(np.nanargmin(res["resid"]), res["resid"].shape)
    fig = new_figure((15, 6))
    axs = [fig.add_subplot(2, 5, i + 1) for i in range(10)]
    axs[0].imshow(disp, cmap="bwr", vmin=-v, vmax=v); axs[0].set_title(f"Unit {unit_id} | best {res['best_lag']:.3f}s")
    axs[1].imshow(res["z_best"], cmap="viridis"); axs[1].plot([xmin], [ymin], "ro", ms=5); axs[1].set_title("z_best")
    axs[2].imshow(res["z_noise"], cmap="viridis"); axs[2].set_title("z_noise (far lags)")
    axs[3].imshow(res["resid"], cmap="viridis"); axs[3].plot([xmin], [ymin], "ro", ms=5); axs[3].set_title("residual z")
    axs[4].imshow(res["roi"], cmap="gray"); axs[4].set_title("ROI")
    axs[5].imshow(res["mask"], cmap="gray"); axs[5].set_title("mask (clean)")
    rf = np.zeros_like(best)
    if res["comp"] is not None:
        rf[res["comp"]] = best[res["comp"]]
    axs[6].imshow(rf, cmap="bwr", vmin=-v, vmax=v)
    if np.isfinite(res["cx"]):
        axs[6].plot([res["cx"]], [res["cy"]], "wo", ms=8); axs[6].plot([res["cx"]], [res["cy"]], "ko", ms=3)
    axs[6].set_title("RF only" + ("" if res["row"]["ok"] else " (rejected)"))
    for i in range(3):
        if i < len(res["lags"]):
            axs[7 + i].imshow(res["stas"][i], cmap="bwr", vmin=-v, vmax=v)
            axs[7 + i].set_title(f"STA {res['lags'][i]:.3f}s")
    for a in axs:
        a.axis("off")
    for a in axs:
        a.title.set_fontsize(8)
    fig.tight_layout()
    return fig


def rf_out_dir(out_folder, group):
    return os.path.join(out_folder, f"SO_RF_{group}")


def rf_csv(out_folder, group):
    return os.path.join(rf_out_dir(out_folder, group), f"SO_RF_centroids_residTemplate_{group}.csv")


def run_rf_batch(sorting, fs, timing: SoTiming, group, out_folder, params, log: LogFn = _nolog,
                 progress: ProgFn = _noprog, cancel=None):
    p = {**DEFAULTS, **(params or {})}
    rng = np.random.default_rng(int(p["seed"]))
    od = rf_out_dir(out_folder, group)
    acc_dir, fail_dir, mask_dir = (os.path.join(od, d) for d in ("accepted_unit_pngs", "failed_unit_pngs", "rf_masks_npz"))
    for d in (acc_dir, fail_dir, mask_dir):
        os.makedirs(d, exist_ok=True)
    ids = np.array(sorted(list(sorting.unit_ids)), dtype=int)
    if p["max_unit_id"]:
        ids = ids[ids <= int(p["max_unit_id"])]
    if str(p["include_ids"]).strip():
        inc = [int(x) for x in str(p["include_ids"]).replace(";", ",").split(",") if x.strip()]
        ids = ids[np.isin(ids, inc)]
    log(f"[{group}] RF batch on {len(ids)} units → {od}")
    rows, accepted = [], 0
    for i, u in enumerate(ids, 1):
        check_cancel(cancel)
        sp = spike_times_s(sorting, int(u), fs).astype(np.float32)
        base = dict(mea_unit_id=int(u), group=group, n_spikes_total=int(sp.size),
                    psy_to_mea_offset_s=timing.offset, off_index=int(timing.off_idx))
        if sp.size < p["min_spikes"]:
            rows.append({**base, "ok": False, "reason": "too_few_spikes"})
            continue
        res = analyse_unit(timing, sp, p, rng)
        row = {**base, **res["row"]}
        rows.append(row)
        if "best" in res and (row["ok"] or p["save_failed_pngs"]) and p["save_pngs"]:
            fig = panel_figure(int(u), res, p["display_smooth"])
            fig.savefig(os.path.join(acc_dir if row["ok"] else fail_dir,
                                     f"mea_{int(u):04d}_SO_RF_{group}.png"), dpi=110)
        if row.get("ok"):
            accepted += 1
            np.savez_compressed(os.path.join(mask_dir, f"mea_{int(u):04d}_rfmask_{group}.npz"),
                                mea_unit_id=int(u), comp=res["comp"].astype(np.uint8),
                                cx_px=float(res["cx"]), cy_px=float(res["cy"]), best_lag_s=res["best_lag"],
                                resid_min=row["resid_min"], thr=row["thr"])
        if i % 10 == 0 or i == len(ids):
            progress(i / len(ids))
            if i % 50 == 0:
                log(f"  {i}/{len(ids)} processed | accepted {accepted}")
    df = pd.DataFrame(rows)
    df.to_csv(rf_csv(out_folder, group), index=False)
    save_json(dict(params=p, offset=timing.offset), os.path.join(od, "rf_params.json"))
    log(f"[{group}] accepted {accepted} / {len(ids)} → {rf_csv(out_folder, group)}")
    return df
