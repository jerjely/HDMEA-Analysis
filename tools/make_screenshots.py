"""Regenerate the README screenshots (docs/screenshots/*.jpg) from the synthetic
self-test dataset. Run from the repository root:

    python tools/make_screenshots.py
"""
import os
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "screenshots")


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import matplotlib
    matplotlib.use("QtAgg")
    from PIL import Image
    from PySide6.QtWidgets import QApplication, QMessageBox
    from hdmea import selftest as st
    from hdmea.gui.main_window import MainWindow

    os.makedirs(OUT, exist_ok=True)
    for n in ("warning", "information", "critical"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.Ok))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

    ctx = {"prefix": "2026_01_20_Thy1GCamP_R3"}
    root = tempfile.mkdtemp(prefix="hdmea_shots_")
    quiet = lambda m: None
    st._imports(ctx); st._data(ctx, root, quiet); st._triggers(ctx, quiet); st._localise(ctx, quiet)
    st._psth(ctx, quiet); st._ringing(ctx, quiet); st._rf(ctx, quiet); st._coupling(ctx, quiet)

    app = QApplication.instance() or QApplication(["shots"])
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    w.session = ctx["s"]
    w.session.save()
    w.notify_session_changed()
    T = {t.key: t for t in w.tab_list}

    def wait():
        t0 = time.time()
        while True:
            while w.worker is not None and w.worker.isRunning():
                app.processEvents(); time.sleep(0.02)
                if time.time() - t0 > 300:
                    raise TimeoutError
            for _ in range(40):
                app.processEvents(); time.sleep(0.005)
            if not (w.worker is not None and w.worker.isRunning()):
                return

    def shot(tab_index, name):
        w.tabs.setCurrentIndex(tab_index)
        for _ in range(40):
            app.processEvents(); time.sleep(0.005)
        img = w.grab().toImage()
        p = os.path.join(OUT, name + ".png")
        img.save(p)
        im = Image.open(p).convert("RGB")
        if im.width != 1400:
            im = im.resize((1400, round(im.height * 1400 / im.width)), Image.LANCZOS)
        im.save(os.path.join(OUT, name + ".jpg"), quality=88, optimize=True)
        os.remove(p)
        print("saved", name)

    w.log_view.clear()
    T["session"].fill_table()
    shot(0, "01_session")

    tr = T["triggers"]
    tr.lux_file.setCurrentIndex(1); tr.plot_lux(); wait()
    shot(1, "02_triggers_lux")

    ps = T["psth"]
    ps.form.set(dict(stim_key="L", min_spikes=50)); ps.on_session_changed()
    ps.unit.setValue(3); ps.plot_unit(); wait()
    shot(3, "03_light_responses")

    rg = T["ringing"]
    rg.form.set(dict(stim_key="L")); rg.summary(); rg.unit.setValue(3); rg.plot_unit(); wait()
    shot(4, "04_ringing")

    rf = T["rf"]
    rf.npy.setText(os.path.join(root, "so.npy"))
    rf.form.set(dict(roi_radius=10, min_rf_pix=6, n_null=20, min_spikes=100))
    rf.show_table()
    best = None
    import pandas as pd
    from hdmea.core import rf_sta
    df = pd.read_csv(rf_sta.rf_csv(w.session.out_folder, "CTRL"))
    best = int(df[df.ok == True].sort_values("resid_min").mea_unit_id.iloc[0])  # noqa: E712
    rf.view_group.setCurrentText("CTRL"); rf.unit.setValue(best); rf.show_unit(); wait()
    shot(5, "05_rf_mapping")

    cp = T["coupling"]
    cp.win.setCurrentIndex(2)
    cp.form.set(dict(max_dist_um=5000)); cp.run_batch(); wait()
    cp.cat.setCurrentText("coupled only")
    cp.table.selectRow(0); cp._row_clicked()
    shot(7, "06_coupling")

    w.session = type(w.session)()
    w.close()


if __name__ == "__main__":
    main()
