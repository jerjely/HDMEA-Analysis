"""Stimulus tools: SO movie generator, lux delay calibration, camera TTL delay."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QGroupBox, QWidget, QTabWidget, QMessageBox, QSplitter

from ..core import stimtools
from .main_window import TabBase
from .widgets import ParamForm, MplPanel, DataTable, button

SO_SPEC = [
    ("n_pix", "Pixels (square)", "int", 200, dict(min=8, max=2000)),
    ("retina_width_mm", "Projected width on retina (mm)", "float", 4.0, dict(decimals=3)),
    ("f_min", "Lowest cutoff (cyc/mm)", "float", 0.21, dict(decimals=3)),
    ("f_max", "Highest cutoff (cyc/mm)", "float", 2.1, dict(decimals=3)),
    ("n_scales", "Number of scales", "int", 10, dict(min=1)),
    ("frames_per_scale", "Frames per scale", "int", 100, dict(min=1)),
    ("percentile", "Dark fraction (percentile)", "float", 10.0, dict(decimals=1, tip="10 → 10 % of pixels dark (−1)")),
    ("seed", "Random seed", "int", 0, dict(tip="Seeded so the movie can be regenerated exactly")),
    ("out", "Save as", "savefile", "", dict(filter="NumPy (*.npy)")),
]
LUX_SPEC = [
    ("run_dir", "Run folder (events_GH.csv + lux_stream.csv)", "dir", ""),
    ("smooth_w", "Smoothing window (samples)", "int", 5, dict(min=1)),
    ("k_sigma", "Derivative threshold (× MAD)", "float", 6.0, dict(decimals=2)),
    ("min_dlux_thr", "Derivative threshold floor (lux/s)", "float", 1.0, dict(decimals=2)),
    ("search_win_s", "Search window after G (s)", "float", 0.25, dict(decimals=3)),
]
CAM_SPEC = [
    ("csv", "Camera ROI CSV (Mean1 = screen, Mean2 = LED)", "file", "", dict(filter="CSV (*.csv)")),
    ("fps", "Camera frame rate (Hz)", "float", 29.99, dict(decimals=3)),
    ("max_lag_frames", "Max |lag| for pairing (frames)", "int", 15, dict(min=1)),
]


class _Tool(QWidget):
    def __init__(self, tab, spec, run_text, run_fn):
        super().__init__()
        self.tab = tab
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.form = ParamForm(spec)
        left.addWidget(self.form)
        left.addWidget(button(run_text, run_fn, primary=True))
        left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(460)
        right = QSplitter(Qt.Vertical)
        self.plot = MplPanel(); self.table = DataTable()
        right.addWidget(self.plot); right.addWidget(self.table); right.setSizes([520, 160])
        lay.addWidget(lw); lay.addWidget(right, 1)


class ToolsTab(TabBase):
    title = "9 · Stimulus tools"
    key = "tools"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QVBoxLayout(self)
        self.sub = QTabWidget()
        self.so = _Tool(self, SO_SPEC, "Generate SO movie", self.gen_so)
        self.lux = _Tool(self, LUX_SPEC, "Measure TTL → light delay", self.run_lux)
        self.cam = _Tool(self, CAM_SPEC, "Measure camera delay", self.run_cam)
        self.sub.addTab(self.so, "SO movie generator")
        self.sub.addTab(self.lux, "Lux sensor delay")
        self.sub.addTab(self.cam, "Camera TTL delay")
        lay.addWidget(self.sub)

    def gen_so(self):
        p = self.so.form.get()
        out = p.pop("out")
        if out and os.path.exists(out) and QMessageBox.question(
                self, "Overwrite?", f"{out} exists. Overwrite? (RF maps computed with the old file "
                                    f"will no longer match)") != QMessageBox.Yes:
            return

        def job(log, progress, cancel):
            movie, cut = stimtools.generate_so_movie(p, out or None, log, progress)
            return stimtools.so_preview_figure(movie, cut, int(p["frames_per_scale"]))
        self.run(job, on_done=self.so.plot.show_figure, label="SO movie")

    def run_lux(self):
        p = self.lux.form.get()
        d = p.pop("run_dir")
        if not os.path.isdir(d):
            QMessageBox.information(self, "Folder", "Choose a so_veml_* run folder."); return

        def job(log, progress, cancel):
            res = stimtools.lux_delay(d, p, log)
            return res, stimtools.lux_figure(res)

        def done(r):
            res, fig = r
            self.lux.plot.show_figure(fig); self.lux.table.set_df(res["edges"])
        self.run(job, on_done=done, label="Lux delay")

    def run_cam(self):
        p = self.cam.form.get()
        if not os.path.isfile(p["csv"]):
            QMessageBox.information(self, "CSV", "Choose the camera ROI CSV."); return

        def job(log, progress, cancel):
            res = stimtools.camera_ttl_delay(p["csv"], p["fps"], p["max_lag_frames"], log)
            return res, stimtools.camera_figure(res)

        def done(r):
            import pandas as pd
            res, fig = r
            self.cam.plot.show_figure(fig)
            self.cam.table.set_df(pd.DataFrame(res["pairs"], columns=["screen_frame", "led_frame", "delay_frames"]))
        self.run(job, on_done=done, label="Camera delay")

    def get_settings(self):
        return dict(so=self.so.form.get(), lux=self.lux.form.get(), cam=self.cam.form.get())

    def set_settings(self, d):
        d = d or {}
        self.so.form.set(d.get("so", {})); self.lux.form.set(d.get("lux", {})); self.cam.form.set(d.get("cam", {}))
