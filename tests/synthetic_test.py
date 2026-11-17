"""End-to-end test of the core modules on synthetic data (no BRW / HerdingSpikes needed)."""
import os, sys, shutil, json
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spikeinterface as si
from hdmea.core import triggers, psth, ringing, rf_sta, rf_calib, coupling, sorting as hs, stimtools
from hdmea.core.common import rc_to_id

ROOT = "/tmp/hdmea_test"
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
OUT = os.path.join(ROOT, "R1"); os.makedirs(OUT)
fs = 2000.0
rng = np.random.default_rng(1)

# ---- recording: 4 files (C_SO1, C_L1, PTX_SO1, PTX_L1), 8x8 grid, trigger on channel "2"
names = ["X_R1_C_SO1.brw", "X_R1_C_L1.brw", "X_R1_PTX_SO1.brw", "X_R1_PTX_L1.brw"]
durs = [120.0, 60.0, 120.0, 60.0]
nch = 64
ids = [str(i + 1) for i in range(nch)]
locs = np.array([[((i) % 8) * 300.0, ((i) // 8) * 300.0] for i in range(nch)])
total = int(sum(durs) * fs)
tr = np.full(total, -30500, np.int16)
starts = np.r_[0, np.cumsum(durs)[:-1]]
on_true = {}
for k, (nm, st, du) in enumerate(zip(names, starts, durs)):
    if "SO" in nm:
        ons = st + 5 + np.arange(3) * 30.0; pw = 25.0
    else:
        ons = st + 2 + np.arange(int((du - 6) / 4)) * 4.0; pw = 2.0
    on_true[nm] = ons
    for o in ons:
        tr[int(o * fs):int((o + pw) * fs)] = -29800
    folder = os.path.join(ROOT, nm[:-4]); os.makedirs(folder)
    # lux & events in psychopy time = MEA local time + 3 s, light lags TTL by 0.2 s
    G = ons - st + 3.0; H = G + pw
    tl = np.arange(0, du, 0.01)
    lux = np.zeros_like(tl)
    for g in G:
        lux[(tl >= g + 0.2) & (tl < g + pw + 0.2)] = 100
    pd.DataFrame(dict(t_s_psychopy=tl, lux=lux)).to_csv(os.path.join(folder, "lux_stream.csv"), index=False)
    ev = pd.DataFrame(dict(event=["G"] * len(G) + ["H"] * len(H), t_s_psychopy=np.r_[G, H]))
    ev.to_csv(os.path.join(folder, "events_GH.csv"), index=False)
traces = (rng.normal(0, 5, (total, nch))).astype(np.int16)
traces[:, 1] = tr
rec = si.NumpyRecording([traces], fs, channel_ids=ids)
rec.set_channel_locations(locs)
table = pd.DataFrame(dict(File=names, path=[os.path.join(ROOT, n) for n in names],
                          stimulus=["SO", "L", "SO", "L"], group=["CTRL", "CTRL", "PTX", "PTX"],
                          duration_s=durs, start=starts, end=starts + np.array(durs)))

trig = triggers.extract_triggers(rec, table, dict(row=1, col=2, chunk_s=7.3), OUT)
for nm in names:
    print(nm, "true ON (+0.2 lux):", (on_true[nm][:3] + 0.2).round(3))
print("L CTRL ON first 3:", np.round(trig["per_stim_group"]["L"]["CTRL"]["ON"][:3], 3))
assert np.allclose(trig["per_stim_group"]["L"]["CTRL"]["ON"][:3], on_true[names[1]][:3] + 0.2, atol=0.01)
assert len(trig["per_stim_group"]["SO"]["CTRL"]["OFF"]) == 0
print(triggers.summary_table(trig))
triggers.qc_figure(rec, trig, 120, 140).savefig(os.path.join(ROOT, "qc.png"))

# ---- SO movie + frame logs
movie, cut = stimtools.generate_so_movie(dict(n_pix=60, n_scales=3, frames_per_scale=40), os.path.join(ROOT, "so.npy"))
for nm, st in zip(names, starts):
    if "SO" not in nm:
        continue
    folder = os.path.join(ROOT, nm[:-4])
    G = on_true[nm] - st + 3.0
    tf, fi = [], []
    for g in G:
        t = g + np.arange(0, 25, 0.2)
        tf += list(t); fi += list(np.arange(len(t)) % movie.shape[0])
    pd.DataFrame(dict(tThisFlipGlobal=tf, frame_index=fi)).to_csv(os.path.join(folder, "SO_frames.csv"), index=False)
    pd.DataFrame({"SO_stim.started": G}).to_csv(os.path.join(folder, "psychopy_frames.csv"), index=False)

# ---- synthetic units
dur = sum(durs)
spk = {}
rf_center = {}
def poisson(rate, t0, t1):
    n = rng.poisson(rate * (t1 - t0)); return np.sort(rng.uniform(t0, t1, n))
L_on = np.concatenate([on_true[n] + 0.2 for n in names if "L" in n.split("_")[-1]])
for u in range(1, 31):
    s = [poisson(2.0, 0, dur)]
    if u <= 15:   # ON responders, ringing at 20 Hz in CTRL only
        for o in L_on:
            ctrl = o < starts[2]
            for k in range(4 if ctrl else 1):
                s.append(o + 0.03 + k * 0.05 + rng.normal(0, 0.003, rng.poisson(3)))
    # SO responders: dark in a patch -> spike 0.1 s later
    cy, cx = rng.integers(15, 45, 2); rf_center[u] = (cx, cy)
    for nm, st in zip(names, starts):
        if "SO" not in nm: continue
        for g in on_true[nm] + 0.2:
            for j, t in enumerate(g + np.arange(0, 25, 0.2)):
                fr = movie[j % movie.shape[0]]
                if fr[cy - 3:cy + 3, cx - 3:cx + 3].mean() < 0:
                    s.append(t + 0.1 + rng.normal(0, 0.005, rng.poisson(4)))
    spk[u] = np.sort(np.concatenate(s))
# coupled pair 31/32 (1 ms lag, narrow)
base = poisson(8, 0, dur)
spk[31] = np.sort(np.r_[base, poisson(4, 0, dur)])
spk[32] = np.sort(np.r_[base + 0.001 + rng.normal(0, 0.002, base.size), poisson(4, 0, dur)])
units = {u: (np.clip(np.round(v * fs), 0, total - 1)).astype(np.int64) for u, v in spk.items()}
sorting = si.NumpySorting.from_unit_dict(units, fs)

# ---- fake HS2_detected
run = os.path.join(OUT, "hs_full", "hs_full_20260101_000000", "sorter_output"); os.makedirs(run)
uloc = {u: rng.integers(0, nch) for u in units}
uloc[32] = uloc[31] + 1 if uloc[31] < nch - 1 else uloc[31] - 1
t_all = np.concatenate([units[u] for u in units]); ch_all = np.concatenate([np.full(units[u].size, uloc[u] + 1) for u in units])
with h5py.File(os.path.join(run, "HS2_detected.hdf5"), "w") as f:
    f["t"] = t_all + 3; f["ch"] = ch_all
loc = hs.localize_units(rec, sorting, os.path.dirname(run), OUT, min_spikes=50)
ok = all(np.allclose(locs[uloc[int(r.mea_unit_id)]], [r.mea_x_um, r.mea_y_um]) for r in loc.itertuples())
print("localisation correct:", ok, len(loc)); assert ok
xy, rate, base_ = hs.activity_map(rec, os.path.dirname(run)); print("activity base:", base_)

# ---- PSTH
df = psth.run_light_responses(sorting, fs, dur, trig, OUT, dict(stim_key="L", min_spikes=50, save_plots=True))
print(df[["unit_id", "l_ctrl_on_peak_time_s", "l_ptx_on_peak_rate_hz", "psth_response_type"]].head(17).to_string())
assert (df.set_index("unit_id").loc[1:15, "psth_response_type"].str.contains("ON")).all()
psth.unit_figure(sorting, fs, trig, 3, dict(stim_key="L"), duration_s=dur).savefig(os.path.join(ROOT, "psth_u3.png"))

# ---- ringing
rdf, paired = ringing.run_ringing(sorting, fs, trig, OUT, dict(stim_key="L"))
print(paired.head())
assert len(paired) >= 10 and np.nanmedian(paired.f_ref) > 15
ringing.unit_figure(sorting, fs, trig, 3, dict(stim_key="L")).savefig(os.path.join(ROOT, "ring_u3.png"))
ringing.summary_figure(paired, ringing.DEFAULTS).savefig(os.path.join(ROOT, "ring_sum.png"))

# ---- RF mapping per group
rf = {}
for g, nm in (("CTRL", names[0]), ("PTX", names[2])):
    so_on = triggers.get_times(trig, "SO", g, "ON")
    tim = rf_sta.SoTiming(os.path.join(ROOT, "so.npy"), os.path.join(ROOT, nm[:-4]), so_on)
    rf[g] = rf_sta.run_rf_batch(sorting, fs, tim, g, OUT, dict(roi_radius=12, min_rf_pix=8, n_null=15, min_spikes=100,
                                                               include_ids="1,2,3,4,5,6,7,8,9,10,11,12"))
c = rf["CTRL"]; c = c[c.ok == True]
err = [np.hypot(r.centroid_cx_px - rf_center[r.mea_unit_id][0] + 0.5, r.centroid_cy_px - rf_center[r.mea_unit_id][1] + 0.5) for r in c.itertuples()]
print("RF accepted", len(c), "median centroid error px", np.median(err))
assert len(c) >= 5 and np.median(err) < 3

# ---- calibration
stab = rf_calib.rf_stability(rf_sta.rf_csv(OUT, "CTRL"), rf_sta.rf_csv(OUT, "PTX"), "CTRL", "PTX", OUT,
                             dict(min_comp_area=5))
uxy = hs.load_unit_xy(OUT)
tr, pairs = rf_calib.fit_rf_to_mea(stab[stab.stable], "CTRL", "PTX", uxy, OUT, dict(min_inliers=3, inlier_thresh_um=3000))
rf_calib.anchors_figure(pairs, tr).savefig(os.path.join(ROOT, "anch.png"))
rf_calib.contour_figure(1, ["CTRL", "PTX"], OUT, tr, uxy, 400).savefig(os.path.join(ROOT, "contour.png"))

# ---- coupling (check exact CCF against the notebook's FFT method)
from scipy.signal import fftconvolve
bt = coupling.BinnedTrains(sorting, fs, starts[1], starts[1] + 60, 0.001)
a = np.bincount(bt.idx[31], minlength=bt.n).astype(float); b = np.bincount(bt.idx[32], minlength=bt.n).astype(float)
full = fftconvolve(a - a.mean(), (b - b.mean())[::-1], mode="full")
K = 50; mid = bt.n - 1
lags, cc = bt.ccf(31, 32, K)
print("CCF max abs diff vs FFT:", np.max(np.abs(full[mid - K:mid + K + 1] - cc)))
assert np.allclose(full[mid - K:mid + K + 1], cc, atol=1e-6)
cdf, bt = coupling.run_coupling(sorting, fs, (starts[1], starts[1] + 60), uxy, OUT, dict(max_dist_um=5000))
print(cdf[cdf.category != "No-peak"])
coupling.pair_figure(bt, 31, 32, {}).savefig(os.path.join(ROOT, "ccf.png"))
coupling.network_figure(cdf, uxy).savefig(os.path.join(ROOT, "net.png"))
print("ALL CORE TESTS PASSED")

# ---- export for the GUI test: per-file recordings (stand-ins for BRW) + sorting
for nm, st, du in zip(names, starts, durs):
    seg = rec.frame_slice(int(st * fs), int((st + du) * fs))
    seg.save(folder=os.path.join(ROOT, "fake_" + nm[:-4]), overwrite=True)
sorting.save(folder=os.path.join(ROOT, "sorting_saved"), overwrite=True)
print("exported for GUI test")
