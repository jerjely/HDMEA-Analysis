"""Built-in self-test: runs every pipeline stage on a small synthetic dataset and
opens the GUI offscreen. Used by the installer builds (a failing self-test stops
the release) and available to users as `HDMEA-Analysis --selftest`.

Exit code 0 = all checks passed, 1 = at least one failed."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback


class _Log:
    def __init__(self, path=None):
        self.f = open(path, "w", encoding="utf-8", buffering=1) if path else None

    def __call__(self, msg):
        line = str(msg)
        try:
            print(line, flush=True)
        except Exception:
            pass
        if self.f:
            self.f.write(line + "\n")


def run_selftest(log_path=None, keep=False) -> int:
    log = _Log(log_path)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from . import __version__
    log(f"HDMEA-Analysis {__version__} self-test | Python {sys.version.split()[0]} | {sys.platform} | "
        f"frozen={getattr(sys, 'frozen', False)}")
    root = tempfile.mkdtemp(prefix="hdmea_selftest_")
    results = []
    ctx = {}

    def step(name, fn):
        t0 = time.time()
        try:
            fn()
            results.append((name, True, ""))
            log(f"[PASS] {name} ({time.time() - t0:.1f} s)")
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
            log(f"[FAIL] {name}: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    try:
        step("imports", lambda: _imports(ctx))
        step("synthetic data + Zarr cache (2 spawn workers)", lambda: _data(ctx, root, log))
        step("triggers with lux correction", lambda: _triggers(ctx, log))
        step("unit localisation + activity map", lambda: _localise(ctx, log))
        step("light responses (PSTH)", lambda: _psth(ctx, log))
        step("ringing", lambda: _ringing(ctx, log))
        step("SO RF mapping + RF→MEA fit", lambda: _rf(ctx, log))
        step("CCF coupling (exact vs FFT)", lambda: _coupling(ctx, log))
        step("stimulus tools", lambda: _tools(ctx, log))
        step("GUI (offscreen)", lambda: _gui(ctx, log))
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    n_fail = sum(1 for r in results if not r[1])
    log(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    for name, ok, msg in results:
        if not ok:
            log(f"  FAILED: {name} — {msg}")
    log("SELFTEST " + ("OK" if n_fail == 0 else "FAILED"))
    return 0 if n_fail == 0 else 1


# ----------------------------------------------------------------------
def _imports(ctx):
    import numpy, scipy, pandas, h5py, zarr, skimage, matplotlib, openpyxl  # noqa: F401
    import spikeinterface, spikeinterface.extractors, spikeinterface.preprocessing  # noqa: F401
    import spikeinterface.sorters  # noqa: F401
    import PySide6  # noqa: F401
    ctx["si"] = spikeinterface


def _data(ctx, root, log):
    import numpy as np
    import pandas as pd
    import h5py
    import spikeinterface as si
    from .core import session as sess_mod
    from .core.stimtools import generate_so_movie

    rng = np.random.default_rng(1)
    fs = 2000.0
    pre = ctx.get("prefix", "ST_R1")
    names = [f"{pre}_C_SO1.brw", f"{pre}_C_L1.brw", f"{pre}_PTX_SO1.brw", f"{pre}_PTX_L1.brw"]
    durs = [80.0, 50.0, 80.0, 50.0]
    nch = 64
    locs = np.array([[(i % 8) * 300.0, (i // 8) * 300.0] for i in range(nch)])
    starts = np.r_[0, np.cumsum(durs)[:-1]]
    movie, _ = generate_so_movie(dict(n_pix=48, n_scales=3, frames_per_scale=30), os.path.join(root, "so.npy"),
                                 log=lambda m: None)
    on_true, rec_folders = {}, {}
    for k, (nm, st, du) in enumerate(zip(names, starts, durs)):
        so = "SO" in nm
        ons = (5 + np.arange(3) * 25.0) if so else (2 + np.arange(int((du - 6) / 4)) * 4.0)
        pw = 20.0 if so else 2.0
        on_true[nm] = ons + st
        n = int(du * fs)
        tr = rng.normal(0, 5, (n, nch)).astype(np.int16)
        trig = np.full(n, -30500, np.int16)
        for o in ons:
            trig[int(o * fs):int((o + pw) * fs)] = -29800
        tr[:, 1] = trig
        rec = si.NumpyRecording([tr], fs, channel_ids=[str(i + 1) for i in range(nch)])
        rec.set_channel_locations(locs)
        rf = os.path.join(root, "rec_" + nm[:-4])
        rec.save(folder=rf, format="binary", overwrite=True, n_jobs=1, verbose=False)
        rec_folders[nm] = rf
        side = os.path.join(root, nm[:-4]); os.makedirs(side)
        G = ons + 3.0; H = G + pw
        tl = np.arange(0, du, 0.01); lux = np.zeros_like(tl)
        for g in G:
            lux[(tl >= g + 0.2) & (tl < g + pw + 0.2)] = 100
        pd.DataFrame(dict(t_s_psychopy=tl, lux=lux)).to_csv(os.path.join(side, "lux_stream.csv"), index=False)
        pd.DataFrame(dict(event=["G"] * len(G) + ["H"] * len(H), t_s_psychopy=np.r_[G, H])).to_csv(
            os.path.join(side, "events_GH.csv"), index=False)
        if so:
            tf, fi = [], []
            for g in G:
                t = g + np.arange(0, pw, 0.2); tf += list(t); fi += list(np.arange(len(t)) % movie.shape[0])
            pd.DataFrame(dict(tThisFlipGlobal=tf, frame_index=fi)).to_csv(os.path.join(side, "SO_frames.csv"), index=False)
            pd.DataFrame({"SO_stim.started": G}).to_csv(os.path.join(side, "psychopy_frames.csv"), index=False)
        open(os.path.join(root, nm), "wb").close()

    # session through the real code path (BRW reader swapped for the binary stand-ins)
    orig = sess_mod.read_brw
    sess_mod.read_brw = lambda path, mea_pitch=42.0, fill_gaps="zeros": sess_mod._load_extractor(
        rec_folders[os.path.basename(path)])
    try:
        s = sess_mod.Session(brw_files=[os.path.join(root, n) for n in names], out_folder=os.path.join(root, "out"))
        s.table = sess_mod.build_record_table(s.brw_files, log=lambda m: None)
        # 'spawn' = how Windows/macOS start workers; checks freeze_support() in the frozen app
        s.rec = sess_mod.load_or_build_zarr(s, rebuild=True, n_jobs=2, mp_context="spawn", log=log)
    finally:
        sess_mod.read_brw = orig
    assert abs(s.rec.get_total_duration() - sum(durs)) < 0.01, "Zarr duration mismatch"
    assert list(s.table["stimulus"]) == ["SO", "L", "SO", "L"] and list(s.table["group"]) == ["CTRL", "CTRL", "PTX", "PTX"]

    # synthetic units
    total = int(sum(durs) * fs)
    dur = sum(durs)
    L_on = np.concatenate([on_true[n] + 0.2 for n in names if n.endswith("L1.brw")])
    spk, rfc = {}, {}

    def pois(rate, a, b):
        return np.sort(rng.uniform(a, b, rng.poisson(rate * (b - a))))
    for u in range(1, 21):
        parts = [pois(2.0, 0, dur)]
        if u <= 10:
            for o in L_on:
                ctrl = o < starts[2]
                for k in range(4 if ctrl else 1):
                    parts.append(o + 0.03 + k * 0.05 + rng.normal(0, 0.003, rng.poisson(3)))
        cy, cx = rng.integers(12, 36, 2); rfc[u] = (cx, cy)
        for nm in names:
            if "SO" not in nm:
                continue
            for g in on_true[nm] + 0.2:
                for j, t in enumerate(g + np.arange(0, 20, 0.2)):
                    if movie[j % movie.shape[0]][cy - 3:cy + 3, cx - 3:cx + 3].mean() < 0.4:   # ≥30 % of the patch dark
                        parts.append(t + 0.1 + rng.normal(0, 0.005, rng.poisson(6)))
        spk[u] = np.sort(np.concatenate(parts))
    base = pois(8, 0, dur)
    spk[21] = np.sort(np.r_[base, pois(4, 0, dur)])
    spk[22] = np.sort(np.r_[base + 0.001 + rng.normal(0, 0.002, base.size), pois(4, 0, dur)])
    units = {u: np.clip(np.round(v * fs), 0, total - 1).astype(np.int64) for u, v in spk.items()}
    sorting = si.NumpySorting.from_unit_dict(units, fs)
    run = os.path.join(s.out_folder, "hs_full", "hs_full_selftest", "sorter_output"); os.makedirs(run)
    uloc = {u: int(rng.integers(0, nch)) for u in units}
    uloc[22] = uloc[21] + 1 if uloc[21] < nch - 1 else uloc[21] - 1
    with h5py.File(os.path.join(run, "HS2_detected.hdf5"), "w") as f:
        f["t"] = np.concatenate([units[u] for u in units]) + 3
        f["ch"] = np.concatenate([np.full(units[u].size, uloc[u] + 1) for u in units])
    s.sorting, s.sorting_folder = sorting, os.path.dirname(run)
    ctx.update(s=s, fs=fs, root=root, names=names, uloc=uloc, locs=locs, rfc=rfc, starts=starts, dur=dur)


def _triggers(ctx, log):
    import numpy as np
    from .core import triggers
    s = ctx["s"]
    trig = triggers.extract_triggers(s.rec, s.table, dict(row=1, col=2, chunk_s=7.3), s.out_folder, log=lambda m: None)
    s.triggers = trig
    on = np.asarray(trig["per_stim_group"]["L"]["CTRL"]["ON"])
    assert len(on) == 11, f"expected 11 L CTRL ON events, got {len(on)}"
    assert np.allclose(on[:3], ctx["starts"][1] + 2 + np.arange(3) * 4 + 0.2, atol=0.01), "lux-corrected times off"
    assert len(trig["per_stim_group"]["SO"]["PTX"]["ON"]) == 3
    triggers.qc_figure(s.rec, trig, 80, 100)
    triggers.lux_figure(s.rec, s.table.iloc[0].to_dict(), {})


def _localise(ctx, log):
    import numpy as np
    from .core import sorting as hs
    s = ctx["s"]
    df = hs.localize_units(s.rec, s.sorting, s.sorting_folder, s.out_folder, min_spikes=50, log=lambda m: None)
    bad = [int(r.mea_unit_id) for r in df.itertuples()
           if not np.allclose(ctx["locs"][ctx["uloc"][int(r.mea_unit_id)]], [r.mea_x_um, r.mea_y_um])]
    assert len(df) == 22 and not bad, f"localisation wrong for units {bad}"
    xy, rate, base = hs.activity_map(s.rec, s.sorting_folder, log=lambda m: None)
    hs.activity_figure(xy, rate); hs.locations_figure(s.rec, df)


def _psth(ctx, log):
    from .core import psth
    s = ctx["s"]
    df = psth.run_light_responses(s.sorting, ctx["fs"], ctx["dur"], s.triggers, s.out_folder,
                                  dict(stim_key="L", min_spikes=50), log=lambda m: None)
    typ = df.set_index("unit_id").loc[1:10, "psth_response_type"].astype(str)
    assert typ.str.contains("ON").all(), f"responders not detected: {typ.to_dict()}"
    assert os.path.exists(os.path.join(s.out_folder, "unit_metrics_master.csv"))
    psth.unit_figure(s.sorting, ctx["fs"], s.triggers, 3, dict(stim_key="L"), duration_s=ctx["dur"])


def _ringing(ctx, log):
    import numpy as np
    from .core import ringing
    s = ctx["s"]
    _, paired = ringing.run_ringing(s.sorting, ctx["fs"], s.triggers, s.out_folder, dict(stim_key="L"),
                                    log=lambda m: None)
    assert len(paired) >= 6 and np.nanmedian(paired.f_ref) > 15, f"ringing: n={len(paired)}"
    ringing.unit_figure(s.sorting, ctx["fs"], s.triggers, 3, dict(stim_key="L"))
    ringing.summary_figure(paired, ringing.DEFAULTS)


def _rf(ctx, log):
    import numpy as np
    from .core import rf_sta, rf_calib, triggers, sorting as hs
    s, root = ctx["s"], ctx["root"]
    p = dict(roi_radius=10, min_rf_pix=6, n_null=20, min_spikes=100, min_jitter_spikes=50, include_ids=",".join(map(str, range(1, 13))),
             save_pngs=True)
    acc = {}
    pre = ctx.get("prefix", "ST_R1")
    for g, nm in (("CTRL", f"{pre}_C_SO1"), ("PTX", f"{pre}_PTX_SO1")):
        tim = rf_sta.SoTiming(os.path.join(root, "so.npy"), os.path.join(root, nm),
                              triggers.get_times(s.triggers, "SO", g, "ON"), log=lambda m: None)
        df = rf_sta.run_rf_batch(s.sorting, ctx["fs"], tim, g, s.out_folder, p, log=lambda m: None)
        acc[g] = df[df.ok == True]  # noqa: E712
    c = acc["CTRL"]
    err = [np.hypot(r.centroid_cx_px - ctx["rfc"][r.mea_unit_id][0] + 0.5,
                    r.centroid_cy_px - ctx["rfc"][r.mea_unit_id][1] + 0.5) for r in c.itertuples()]
    assert len(c) >= 4 and np.median(err) < 3, f"RF: accepted {len(c)}, median error {np.median(err) if err else 'NA'}"
    stab = rf_calib.rf_stability(rf_sta.rf_csv(s.out_folder, "CTRL"), rf_sta.rf_csv(s.out_folder, "PTX"), "CTRL",
                                 "PTX", s.out_folder, dict(min_comp_area=5), log=lambda m: None)
    uxy = hs.load_unit_xy(s.out_folder)
    tr, pairs = rf_calib.fit_rf_to_mea(stab[stab.stable], "CTRL", "PTX", uxy, s.out_folder,
                                       dict(min_inliers=3, inlier_thresh_um=3000), log=lambda m: None)
    rf_calib.anchors_figure(pairs, tr)
    rf_calib.contour_figure(int(c.mea_unit_id.iloc[0]), ["CTRL", "PTX"], s.out_folder, tr, uxy, 400)


def _coupling(ctx, log):
    import numpy as np
    from scipy.signal import fftconvolve
    from .core import coupling, sorting as hs
    s = ctx["s"]
    w = (ctx["starts"][1], ctx["starts"][1] + 50)
    bt = coupling.BinnedTrains(s.sorting, ctx["fs"], w[0], w[1], 0.001)
    a = np.bincount(bt.idx[21], minlength=bt.n).astype(float); b = np.bincount(bt.idx[22], minlength=bt.n).astype(float)
    full = fftconvolve(a - a.mean(), (b - b.mean())[::-1], mode="full")
    K = 50; mid = bt.n - 1
    _, cc = bt.ccf(21, 22, K)
    assert np.allclose(full[mid - K:mid + K + 1], cc, atol=1e-6), "CCF differs from FFT reference"
    df, bt = coupling.run_coupling(s.sorting, ctx["fs"], w, hs.load_unit_xy(s.out_folder), s.out_folder,
                                   dict(max_dist_um=5000), log=lambda m: None)
    hit = df[(df.hub == 21) & (df.partner == 22)]
    assert len(hit) and hit.category.iloc[0] == "Unimodal-narrow", "coupled pair not classified as narrow"
    coupling.pair_figure(bt, 21, 22, {}); coupling.network_figure(df, hs.load_unit_xy(s.out_folder))


def _tools(ctx, log):
    import numpy as np
    import pandas as pd
    from .core import stimtools
    root = ctx["root"]
    res = stimtools.lux_delay(os.path.join(root, ctx.get("prefix", "ST_R1") + "_C_L1"), {}, log=lambda m: None)
    assert len(res["edges"]) and abs(res["edges"].delay_ms.median() - 200) < 15, "lux delay not ~200 ms"
    stimtools.lux_figure(res)
    m1 = np.zeros(300); m2 = np.zeros(300)
    for s0 in (40, 120, 200):
        m1[s0:s0 + 30] = 100; m2[s0 - 5:s0 + 25] = 80
    p = os.path.join(root, "cam.csv")
    pd.DataFrame({" ": np.arange(300), "Mean1": m1 + np.random.rand(300), "Mean2": m2 + np.random.rand(300)}).to_csv(p, index=False)
    r = stimtools.camera_ttl_delay(p, log=lambda m: None)
    assert r["xcorr_lag_frames"] == -5 and len(r["pairs"]) == 3, "camera delay wrong"
    stimtools.camera_figure(r)


def _gui(ctx, log):
    import matplotlib
    matplotlib.use("QtAgg", force=True)
    from PySide6.QtWidgets import QApplication, QMessageBox
    from .gui.main_window import MainWindow
    from . import __version__
    msgs = []
    for n in ("warning", "information", "critical"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: (msgs.append(a[2] if len(a) > 2 else a), QMessageBox.Ok)[1]))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
    app = QApplication.instance() or QApplication(["selftest"])
    w = MainWindow()
    w.show()
    assert __version__ in w.windowTitle(), "version missing from window title"
    w.session = ctx["s"]
    w.notify_session_changed()
    for i in range(w.tabs.count()):
        w.tabs.setCurrentIndex(i); app.processEvents()
    tabs = {t.key: t for t in w.tab_list}
    tabs["psth"].form.set(dict(stim_key="L")); tabs["psth"].unit.setValue(3)
    tabs["psth"].plot_unit()
    t0 = time.time()
    while w.worker is not None and w.worker.isRunning():
        app.processEvents(); time.sleep(0.02)
        assert time.time() - t0 < 120, "GUI worker timeout"
    for _ in range(20):
        app.processEvents()
    assert tabs["psth"].plot.figure is not None, f"PSTH plot not shown (dialogs: {msgs})"
    assert w.grab().width() > 100
    w.session = type(ctx["s"])()  # detach so closeEvent doesn't save into the temp folder
    w.close()
