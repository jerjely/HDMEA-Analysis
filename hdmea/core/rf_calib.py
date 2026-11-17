"""RF stability between conditions and stimulus-pixel → MEA-µm calibration from
stable anchor units (SO_Final cells 13, 14, 18)."""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd

from .common import LogFn, _nolog, save_json, load_json, new_figure

STAB_DEFAULTS = dict(max_shift_px=12.0, min_comp_area=80, require_ok_both=True)
CAL_DEFAULTS = dict(model="similarity", use_ransac=True, ransac_iters=1200, inlier_thresh_um=220.0,
                    min_inliers=30, anchor_source="MEAN", try_orientations=True)


def calib_dir(out_folder):
    return os.path.join(out_folder, "RF_calibration")


def _load_rf(csv, g):
    df = pd.read_csv(csv).copy()
    df["mea_unit_id"] = df["mea_unit_id"].astype(int)
    df["ok"] = df["ok"].astype(bool) if "ok" in df else True
    if "comp_area" not in df:
        df["comp_area"] = np.nan
    return df.rename(columns={"centroid_cx_px": f"cx_{g}", "centroid_cy_px": f"cy_{g}", "ok": f"ok_{g}",
                              "comp_area": f"area_{g}"})[["mea_unit_id", f"cx_{g}", f"cy_{g}", f"ok_{g}", f"area_{g}"]]


def rf_stability(csv_a, csv_b, ga, gb, out_folder, params, log: LogFn = _nolog):
    p = {**STAB_DEFAULTS, **(params or {})}
    m = _load_rf(csv_a, ga).merge(_load_rf(csv_b, gb), on="mea_unit_id", how="inner")
    m["dx_px"] = m[f"cx_{ga}"] - m[f"cx_{gb}"]
    m["dy_px"] = m[f"cy_{ga}"] - m[f"cy_{gb}"]
    m["shift_px"] = np.hypot(m["dx_px"], m["dy_px"])
    mask = np.isfinite(m["shift_px"])
    if p["require_ok_both"]:
        mask &= m[f"ok_{ga}"] & m[f"ok_{gb}"]
    if p["min_comp_area"]:
        mask &= (m[f"area_{ga}"] >= p["min_comp_area"]) & (m[f"area_{gb}"] >= p["min_comp_area"])
    use = m[mask].copy()
    use["stable"] = use["shift_px"] <= p["max_shift_px"]
    d = calib_dir(out_folder); os.makedirs(d, exist_ok=True)
    use.to_csv(os.path.join(d, "rf_all_common_filtered.csv"), index=False)
    use[use.stable].to_csv(os.path.join(d, "rf_stable_units.csv"), index=False)
    use[~use.stable].to_csv(os.path.join(d, "rf_unstable_units.csv"), index=False)
    log(f"Common units {len(m)} → after filters {len(use)} → stable (≤{p['max_shift_px']} px): {int(use.stable.sum())}")
    return use


# ----------------------------------------------------------------------
def orient(P, swapxy=False, flipx=False, flipy=False):
    Q = np.asarray(P, float).copy()
    if swapxy:
        Q = Q[:, [1, 0]]
    if flipx:
        Q[:, 0] = -Q[:, 0]
    if flipy:
        Q[:, 1] = -Q[:, 1]
    return Q


def fit_similarity(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    s = np.trace((X @ R).T @ Y) / max((X ** 2).sum(), 1e-12)
    return float(s), R, mq - s * (mp @ R)


def fit_affine(P, Q):
    B, *_ = np.linalg.lstsq(np.c_[P, np.ones(len(P))], Q, rcond=None)
    return B[:2].T, B[2]


def _predict(model, P, fit):
    if model == "similarity":
        s, R, t = fit
        return s * (P @ R) + t
    A, t = fit
    return P @ A.T + t


def _fit(model, P, Q):
    return fit_similarity(P, Q) if model == "similarity" else fit_affine(P, Q)


def _ransac(P, Q, model, iters, thr, min_in, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    if len(P) < 3:
        return np.arange(len(P))
    for _ in range(int(iters)):
        idx = rng.choice(len(P), 3, replace=False)
        try:
            err = np.linalg.norm(_predict(model, P, _fit(model, P[idx], Q[idx])) - Q, axis=1)
        except Exception:
            continue
        inl = np.where(err <= thr)[0]
        if inl.size < min_in:
            continue
        sc = (inl.size, -float(np.median(err[inl])))
        if best is None or sc > best[0]:
            best = (sc, inl)
    return np.arange(len(P)) if best is None else best[1]


def qc(err):
    e = np.asarray(err, float); e = e[np.isfinite(e)]
    if not e.size:
        return {}
    return dict(n=int(e.size), median=float(np.median(e)), p90=float(np.percentile(e, 90)), max=float(e.max()),
                pct_le_100=float((e <= 100).mean() * 100), pct_le_200=float((e <= 200).mean() * 100))


def fit_rf_to_mea(stable_df, ga, gb, unit_xy: dict, out_folder, params, log: LogFn = _nolog):
    p = {**CAL_DEFAULTS, **(params or {})}
    a = stable_df.copy()
    if p["anchor_source"].upper() == "MEAN":
        a["rf_x_px"] = 0.5 * (a[f"cx_{ga}"] + a[f"cx_{gb}"]); a["rf_y_px"] = 0.5 * (a[f"cy_{ga}"] + a[f"cy_{gb}"])
    else:
        a["rf_x_px"] = a[f"cx_{ga}"]; a["rf_y_px"] = a[f"cy_{ga}"]
    a = a[a["mea_unit_id"].isin(list(unit_xy.keys()))].copy()
    a["mea_x_um"] = [unit_xy[int(u)][0] for u in a["mea_unit_id"]]
    a["mea_y_um"] = [unit_xy[int(u)][1] for u in a["mea_unit_id"]]
    a = a.dropna(subset=["rf_x_px", "rf_y_px", "mea_x_um", "mea_y_um"])
    log(f"Anchor pairs (stable ∩ localised): {len(a)}")
    if len(a) < 3:
        raise ValueError("Need at least 3 anchor units with both RF centroid and MEA location")
    P0 = a[["rf_x_px", "rf_y_px"]].to_numpy(float); Q0 = a[["mea_x_um", "mea_y_um"]].to_numpy(float)
    model = p["model"]
    opts = [(s, fx, fy) for s in (False, True) for fx in (False, True) for fy in (False, True)] \
        if p["try_orientations"] else [(False, False, False)]
    best = None
    min_in = min(int(p["min_inliers"]), max(3, len(a) // 2))
    for sw, fx, fy in opts:
        P = orient(P0, sw, fx, fy)
        inl = _ransac(P, Q0, model, p["ransac_iters"], p["inlier_thresh_um"], min_in) if p["use_ransac"] \
            else np.arange(len(P))
        try:
            fit = _fit(model, P[inl], Q0[inl])
        except Exception:
            continue
        err = np.linalg.norm(_predict(model, P, fit) - Q0, axis=1)
        sc = (inl.size, -qc(err[inl]).get("median", 1e9))
        if best is None or sc > best["score"]:
            best = dict(score=sc, o=(sw, fx, fy), inl=inl)
    sw, fx, fy = best["o"]
    P = orient(P0, sw, fx, fy)
    fit = _fit(model, P[best["inl"]], Q0[best["inl"]])
    pred = _predict(model, P, fit)
    err = np.linalg.norm(pred - Q0, axis=1)
    tr = dict(model=model, orientation=dict(swapxy=sw, flipx=fx, flipy=fy), n_anchor_pairs=len(P0),
              n_inliers=int(best["inl"].size), inlier_thresh_um=float(p["inlier_thresh_um"]),
              anchor_rf_source=p["anchor_source"], qc_all=qc(err), qc_inliers=qc(err[best["inl"]]))
    if model == "similarity":
        s, R, t = fit
        tr.update(scale_um_per_px=s, rotation_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])),
                  R=R.tolist(), translation_um=list(map(float, t)))
    else:
        A, t = fit
        tr.update(A=A.tolist(), translation_um=list(map(float, t)))
    d = calib_dir(out_folder); os.makedirs(d, exist_ok=True)
    save_json(tr, os.path.join(d, f"RF_to_MEA_transform_{model}.json"))
    a["pred_x_um"], a["pred_y_um"], a["err_um"] = pred[:, 0], pred[:, 1], err
    a["is_inlier"] = False; a.iloc[best["inl"], a.columns.get_loc("is_inlier")] = True
    a.to_csv(os.path.join(d, f"RF_to_MEA_pairs_{model}.csv"), index=False)
    log(f"Best orientation swap={sw} flipx={fx} flipy={fy} | inliers {best['inl'].size}/{len(P0)} | "
        f"median err {tr['qc_inliers'].get('median', np.nan):.1f} µm"
        + (f" | scale {tr['scale_um_per_px']:.2f} µm/px, rot {tr['rotation_deg']:.2f}°" if model == "similarity" else ""))
    return tr, a


def px_to_um(P, tr):
    o = tr["orientation"]
    Q = orient(np.asarray(P, float), o["swapxy"], o["flipx"], o["flipy"])
    if tr["model"] == "similarity":
        return tr["scale_um_per_px"] * (Q @ np.asarray(tr["R"], float)) + np.asarray(tr["translation_um"], float)
    return Q @ np.asarray(tr["A"], float).T + np.asarray(tr["translation_um"], float)


def project_centroids(rf_csv_path, group, tr, out_folder):
    df = pd.read_csv(rf_csv_path)
    df = df[df["ok"].astype(bool)].copy() if "ok" in df else df
    xy = px_to_um(df[["centroid_cx_px", "centroid_cy_px"]].to_numpy(float), tr)
    df["rf_x_um"], df["rf_y_um"] = xy[:, 0], xy[:, 1]
    df.to_csv(os.path.join(calib_dir(out_folder), f"{group}_RF_centroids_in_MEA.csv"), index=False)
    return df


def load_transform(out_folder, model="similarity"):
    p = os.path.join(calib_dir(out_folder), f"RF_to_MEA_transform_{model}.json")
    return load_json(p) if os.path.exists(p) else None


# ----------------------------------------------------------------------
def anchors_figure(pairs, tr):
    fig = new_figure((11, 4.6))
    a1, a2 = fig.add_subplot(121), fig.add_subplot(122)
    a1.scatter(pairs["mea_x_um"], pairs["mea_y_um"], s=12, alpha=0.5, label="unit location (HS2)")
    a1.scatter(pairs["pred_x_um"], pairs["pred_y_um"], s=12, alpha=0.5, label="RF centroid → MEA")
    inl = pairs[pairs.is_inlier]
    for _, r in inl.iterrows():
        a1.plot([r.mea_x_um, r.pred_x_um], [r.mea_y_um, r.pred_y_um], lw=0.5, color="gray")
    a1.set_aspect("equal"); a1.legend(fontsize=8); a1.set_xlabel("x (µm)"); a1.set_ylabel("y (µm)")
    a1.set_title(f"Anchors ({tr['model']}) inliers {tr['n_inliers']}/{tr['n_anchor_pairs']}", fontsize=9)
    a2.hist(pairs["err_um"], bins=40); a2.set_xlabel("anchor error (µm)"); a2.set_ylabel("count")
    fig.tight_layout()
    return fig


def contour_figure(unit_id, groups, out_folder, tr, unit_xy=None, max_dist_um=None, lim=(0, 2700)):
    from skimage.measure import find_contours
    from .rf_sta import rf_out_dir
    fig = new_figure((6, 6))
    ax = fig.add_subplot(111)
    cents, msgs = {}, []
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for k, g in enumerate(groups):
        npz = os.path.join(rf_out_dir(out_folder, g), "rf_masks_npz", f"mea_{int(unit_id):04d}_rfmask_{g}.npz")
        if not os.path.exists(npz):
            msgs.append(f"{g}: no RF"); continue
        z = np.load(npz)
        comp = z["comp"].astype(float)
        for c in find_contours(comp, 0.5):
            xy = px_to_um(np.c_[c[:, 1], c[:, 0]], tr)
            ax.plot(xy[:, 0], xy[:, 1], color=colors[k % 4], lw=1.6)
        cxy = px_to_um(np.array([[float(z["cx_px"]), float(z["cy_px"])]]), tr)[0]
        cents[g] = cxy
        ax.plot(*cxy, "o", color=colors[k % 4], label=f"{g} RF")
    if len(cents) == 2 and max_dist_um:
        (g1, c1), (g2, c2) = cents.items()
        dist = float(np.hypot(*(c1 - c2)))
        msgs.append(f"{g1}–{g2} centroid distance {dist:.0f} µm" + (" ⚠ > gate" if dist > max_dist_um else ""))
    if unit_xy and int(unit_id) in unit_xy:
        ax.plot(*unit_xy[int(unit_id)], "k*", ms=12, label="unit (electrode)")
    if lim:
        ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Unit {unit_id} RF on MEA" + ("\n" + "; ".join(msgs) if msgs else ""), fontsize=9)
    fig.tight_layout()
    return fig


def all_centroids_figure(proj: dict, unit_xy=None):
    fig = new_figure((6, 6))
    ax = fig.add_subplot(111)
    if unit_xy:
        xy = np.array(list(unit_xy.values()))
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c="0.75", label="unit locations")
    for g, df in proj.items():
        ax.scatter(df["rf_x_um"], df["rf_y_um"], s=8, alpha=0.5, label=f"{g} RF centroids")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.tight_layout()
    return fig
