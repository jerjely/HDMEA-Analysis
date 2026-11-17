"""Drive the GUI end-to-end on the synthetic data exported by synthetic_test.py."""
import os, sys, shutil, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("QtAgg")
from PySide6.QtWidgets import QApplication, QMessageBox
import spikeinterface as si

ROOT = "/tmp/hdmea_test"
OUT2 = os.path.join(ROOT, "GUI_R1"); shutil.rmtree(OUT2, ignore_errors=True)
SHOTS = os.path.join(ROOT, "shots"); os.makedirs(SHOTS, exist_ok=True)

from hdmea.core import session as sess_mod
sess_mod.read_brw = lambda path, mea_pitch=42.0, fill_gaps="zeros": si.load(
    os.path.join(os.path.dirname(path), "fake_" + os.path.splitext(os.path.basename(path))[0]))

errors = []
for name in ("warning", "information", "critical"):
    setattr(QMessageBox, name, staticmethod(lambda *a, n=name: (errors.append(a[2] if len(a) > 2 else a), print(f"[{n}]", a[1:3]))[0]))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

app = QApplication([])
from hdmea.gui.main_window import MainWindow
w = MainWindow(); w.resize(1500, 950); w.show()
logs = []
_orig = w.log
w.log = lambda m: (logs.append(str(m)), _orig(m))


def wait(win=None):
    win = win or w
    t0 = time.time()
    while True:
        while win.worker is not None and win.worker.isRunning():
            app.processEvents(); time.sleep(0.02)
            assert time.time() - t0 < 600, "timeout"
        for _ in range(30):
            app.processEvents(); time.sleep(0.005)
        if not (win.worker is not None and win.worker.isRunning()):
            break


def shot(i, name):
    w.tabs.setCurrentIndex(i); app.processEvents(); time.sleep(0.1); app.processEvents()
    w.grab().save(os.path.join(SHOTS, f"{name}.png"))


T = {t.key: t for t in w.tab_list}
names = ["X_R1_C_SO1.brw", "X_R1_C_L1.brw", "X_R1_PTX_SO1.brw", "X_R1_PTX_L1.brw"]
w.session.brw_files = [os.path.join(ROOT, n) for n in names]
T["session"].out_edit.setText(OUT2)
T["session"].fill_table()
T["session"].read_headers(); wait()
assert w.session.table is not None and len(w.session.table) == 4, "headers"
T["session"].build_zarr(); wait()
assert w.session.rec is not None, "zarr"
print("rec:", w.session.rec.get_total_duration(), w.session.rec.get_dtype())
shot(0, "1_session")

T["triggers"].form.set(dict(chunk_s=7.3)); T["triggers"].run_extract(); wait()
assert w.session.triggers and len(w.session.triggers["per_stim_group"]["L"]["CTRL"]["ON"]) == 13
shot(1, "2_triggers")
T["triggers"].lux_file.setCurrentIndex(0); T["triggers"].plot_lux(); wait()
shot(1, "2_triggers_lux")
# auto-match: rename side-car folders to so_veml_<timestamp> under a parent and match back
import datetime as dt
par = os.path.join(ROOT, "runs"); shutil.rmtree(par, ignore_errors=True); os.makedirs(par)
for r in w.session.table.itertuples():
    end = dt.datetime.fromtimestamp(os.path.getmtime(r.path)) if os.path.exists(r.path) else None
for i, r in w.session.table.iterrows():
    open(r["path"], "a").close()   # make sure the stand-in BRW exists so it has an mtime
    st = dt.datetime.fromtimestamp(os.path.getmtime(r["path"])) - dt.timedelta(seconds=r["duration_s"]) + dt.timedelta(minutes=i * 0.5)
    shutil.copytree(os.path.join(ROOT, r["File"][:-4]), os.path.join(par, f"so_veml_{st:%Y%m%d_%H%M%S}"))
from hdmea.core.common import match_run_folders
m = match_run_folders(w.session.table, par, log=print)
assert len(m) == 4, m

# sorting: use saved sorting + fake HS detections folder
src = os.path.join(ROOT, "R1", "hs_full"); shutil.copytree(src, os.path.join(OUT2, "hs_full"))
run = os.path.join(OUT2, "hs_full", "hs_full_20260101_000000")
w.session.sorting = si.load(os.path.join(ROOT, "sorting_saved")); w.session.sorting_folder = run
w.notify_session_changed()
T["sorting"].loc_form.set(dict(min_spikes=50)); T["sorting"].localise(); wait()
assert os.path.exists(os.path.join(OUT2, "unit_xy_map_from_HS2_detected.csv"))
shot(2, "3_sorting_loc")
T["sorting"].activity(); wait()
shot(2, "3_sorting_activity")

T["psth"].form.set(dict(min_spikes=50)); T["psth"].run_batch(); wait()
assert T["psth"].table.model_.rowCount() > 10
T["psth"].unit.setValue(3); T["psth"].plot_unit(); wait()
shot(3, "4_psth")

T["ringing"].run_batch(); wait()
assert T["ringing"].paired is not None and len(T["ringing"].paired) > 5
T["ringing"].unit.setValue(3); T["ringing"].plot_unit(); wait()
shot(4, "5_ringing")

rf = T["rf"]
rf.npy.setText(os.path.join(ROOT, "so.npy"))
rf.form.set(dict(roi_radius=12, min_rf_pix=8, n_null=15, min_spikes=100, include_ids="1,2,3,4,5,6,7,8,9,10"))
print("RF folders:", {g: e.text() for g, e in rf.group_edits.items()})
rf.run_batch(); wait()
assert os.path.exists(os.path.join(OUT2, "SO_RF_CTRL", "SO_RF_centroids_residTemplate_CTRL.csv"))
rf.unit.setValue(1); rf.show_unit(); wait()
shot(5, "6_rf")

cal = T["calib"]
cal.stab.set(dict(min_comp_area=5)); cal.run_stab()
cal.cal.set(dict(min_inliers=3, inlier_thresh_um=3000)); cal.run_fit(); app.processEvents()
cal.unit.setValue(1); cal.show_unit()
shot(6, "7_calib")

cp = T["coupling"]
cp.win.setCurrentIndex(2)  # C_L1 file
cp.form.set(dict(max_dist_um=5000)); cp.run_batch(); wait()
assert cp.df is not None and ((cp.df.hub == 31) & (cp.df.partner == 32) & (cp.df.category == "Unimodal-narrow")).any()
cp.table.selectRow(0); cp._row_clicked()
shot(7, "8_coupling")

tl = T["tools"]
tl.so.form.set(dict(n_pix=64, n_scales=5, frames_per_scale=10, out=os.path.join(ROOT, "so_gui.npy")))
tl.gen_so(); wait()
tl.lux.form.set(dict(run_dir=os.path.join(ROOT, "X_R1_C_L1"))); tl.sub.setCurrentIndex(1); tl.run_lux(); wait()
# camera csv: LED leads screen by 5 frames
fr = np.arange(400); m1 = np.zeros(400); m2 = np.zeros(400)
for s0 in (50, 150, 250, 350):
    m1[s0:s0 + 40] = 100; m2[s0 - 5:s0 + 35] = 80
pd.DataFrame({" ": fr, "Mean1": m1 + np.random.rand(400), "Mean2": m2 + np.random.rand(400)}).to_csv(
    os.path.join(ROOT, "cam.csv"), index=False)
tl.cam.form.set(dict(csv=os.path.join(ROOT, "cam.csv"))); tl.sub.setCurrentIndex(2); tl.run_cam(); wait()
tl.sub.setCurrentIndex(1)
shot(8, "9_tools_lux")

# save + reopen
w.save_session()
w2 = MainWindow(); w2.open_session(os.path.join(OUT2, "hdmea_session.json"))
wait(w2)
print("reopened:", w2.status_lbl.text())
assert w2.session.rec is not None and w2.session.triggers
print("dialog messages:", errors)
fails = [l for l in logs if "failed" in l or "⚠ error" in l]
print("failures in log:", fails)
assert not fails
print("GUI TEST PASSED")
