"""Analysis tabs: light responses, ringing, SO RF mapping, RF→MEA calibration, CCF coupling."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QGroupBox, QSplitter, QWidget, QLabel, QSpinBox,
                               QComboBox, QCheckBox, QDoubleSpinBox, QFormLayout, QMessageBox, QLineEdit)

from ..core import psth, ringing, rf_sta, rf_calib, coupling, sorting as hs
from ..core.common import align_folder_of
from .main_window import TabBase
from .widgets import ParamForm, PathEdit, MplPanel, DataTable, button, hbox


def trig_stims(s):
    return list((s.triggers or {}).get("per_stim_group", {}).keys())


def trig_groups(s, stim=None):
    pg = (s.triggers or {}).get("per_stim_group", {})
    if stim and stim in pg:
        return list(pg[stim].keys())
    out = []
    for g in pg.values():
        for k in g:
            if k not in out:
                out.append(k)
    return out or s.groups()


def unit_spin():
    sp = QSpinBox(); sp.setRange(0, 10 ** 7); sp.setFixedWidth(90)
    return sp


def read_master(s):
    p = s.master_csv
    if s.out_folder and os.path.exists(p):
        try:
            df = pd.read_csv(p, sep=";", low_memory=False)
            df.columns = [c.lower() if c != "unit_id" else c for c in df.columns]
            return df
        except Exception:
            return None
    return None


# ======================================================================
PSTH_SPEC = [
    ("stim_key", "Stimulus", "choice", "L", dict(choices=["L"], editable=True)),
    ("ref_group", "Reference group (response type)", "choice", "CTRL", dict(choices=["CTRL"], editable=True)),
    ("opto_stim", "Opto stimulus (optional)", "choice", "", dict(choices=[""], editable=True,
                                                                  tip="A stimulus code whose ON times are opto pulses → PV_status")),
    ("pre_time", "Pre (s)", "float", 0.5, dict(decimals=3)),
    ("post_time", "Post (s)", "float", 3.0, dict(decimals=3)),
    ("bin_size", "Bin (s)", "float", 0.025, dict(decimals=4, step=0.005)),
    ("smooth_sigma", "Smoothing σ (bins)", "float", 1.0, dict(decimals=2)),
    ("min_spikes", "Min spikes per unit", "int", 150, dict(min=0)),
    ("peak_window_s", "Peak search window (s)", "float", 1.0, dict(decimals=3)),
    ("min_peak_rate_Hz", "Min peak rate (Hz)", "float", 3.0, dict(decimals=2)),
    ("min_peak_delta_Hz", "Min peak − baseline (Hz)", "float", 2.0, dict(decimals=2)),
    ("min_peak_delta_global_Hz", "Min peak − ISI rate (Hz)", "float", 0.0, dict(decimals=2)),
    ("min_peak_z", "Min peak z", "float", 2.5, dict(decimals=2)),
    ("min_peak_prom_Hz", "Min prominence (Hz)", "float", 1.0, dict(decimals=2)),
    ("min_peak_distance_s", "Min peak distance (s)", "float", 0.05, dict(decimals=3)),
    ("save_plots", "Save PNG per unit/condition", "bool", False),
]


class PsthTab(TabBase):
    title = "4 · Light responses"
    key = "psth"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.form = ParamForm(PSTH_SPEC, scroll=True)
        g = QGroupBox("PSTH / raster metrics"); QVBoxLayout(g).addWidget(self.form)
        left.addWidget(g, 1)
        left.addWidget(button("Run for all units → master table", self.run_batch, primary=True))
        self.only_resp = QCheckBox("Show responsive units only"); self.only_resp.setChecked(True)
        self.only_resp.toggled.connect(self.show_table)
        self.unit = unit_spin()
        left.addWidget(self.only_resp)
        left.addWidget(hbox(QLabel("Unit"), self.unit, button("Plot unit", self.plot_unit), "stretch"))
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(430)
        right = QSplitter(Qt.Vertical)
        self.table = DataTable(); self.table.clicked.connect(self._row_clicked)
        self.plot = MplPanel()
        right.addWidget(self.table); right.addWidget(self.plot); right.setSizes([260, 480])
        lay.addWidget(lw); lay.addWidget(right, 1)
        self.master = None

    def run_batch(self):
        if not self.mw.require(rec=True, sorting=True, triggers=True):
            return
        s, p = self.session, self.form.get()

        def job(log, progress, cancel):
            return psth.run_light_responses(s.sorting, s.fs, s.rec.get_total_duration(), s.triggers, s.out_folder,
                                            p, log=log, progress=progress, cancel=cancel)
        self.run(job, on_done=lambda df: self.on_session_changed(), label="Light responses")

    def show_table(self):
        df = self.master
        if df is None:
            self.table.set_df(pd.DataFrame()); return
        st = self.form.get()["stim_key"].lower()
        cols = ["unit_id", "n_spikes_total", "psth_response_type", "pv_status"] + \
               [c for c in df.columns if c.startswith(st + "_") and (c.endswith("_peak_time_s") or
                                                                    c.endswith("_peak_rate_hz") or c.endswith("_tau_s"))]
        cols = [c for c in cols if c in df.columns]
        d = df[cols]
        if self.only_resp.isChecked() and "psth_response_type" in d:
            d = d[d["psth_response_type"].fillna("").astype(str).str.len() > 0]
        self.table.set_df(d)

    def _row_clicked(self, *_):
        r = self.table.selected_row()
        if r is not None:
            self.unit.setValue(int(r["unit_id"])); self.plot_unit()

    def plot_unit(self):
        if not self.mw.require(rec=True, sorting=True, triggers=True):
            return
        s, p, u = self.session, self.form.get(), self.unit.value()
        if u not in set(int(x) for x in s.sorting.unit_ids):
            self.log(f"Unit {u} not in sorting"); return
        self.run(lambda log, progress, cancel: psth.unit_figure(s.sorting, s.fs, s.triggers, u, p,
                                                                duration_s=s.rec.get_total_duration()),
                 on_done=self.plot.show_figure, label=f"PSTH unit {u}")

    def on_session_changed(self):
        s = self.session
        st = trig_stims(s)
        if st:
            self.form.set_choices("stim_key", st)
            self.form.set_choices("opto_stim", [""] + st)
            self.form.set_choices("ref_group", trig_groups(s, self.form.get()["stim_key"]))
        self.master = read_master(s)
        self.show_table()

    def get_settings(self):
        return dict(form=self.form.get())

    def set_settings(self, d):
        self.form.set((d or {}).get("form", {}))


# ======================================================================
RING_SPEC = [
    ("stim_key", "Stimulus", "choice", "L", dict(choices=["L"], editable=True)),
    ("edge", "Edge", "choice", "ON", dict(choices=["ON", "OFF"])),
    ("ref_group", "Reference group", "choice", "CTRL", dict(choices=["CTRL"], editable=True)),
    ("test_group", "Test group", "choice", "PTX", dict(choices=["PTX"], editable=True)),
    ("suffix", "Column suffix", "str", "first40_dualf"),
    ("n_trials_limit", "Use first N trials", "int", 40, dict(min=1)),
    ("pre_time", "Pre (s)", "float", 0.2, dict(decimals=3)),
    ("post_time", "Post (s)", "float", 0.6, dict(decimals=3)),
    ("bin_size", "Bin (s)", "float", 0.0025, dict(decimals=4, step=0.0005)),
    ("smooth_sigma_bins", "Smoothing σ (bins)", "float", 2.0, dict(decimals=2)),
    ("ring_t0", "Peak window start (s)", "float", -0.01, dict(decimals=3)),
    ("ring_t1", "Peak window end (s)", "float", 0.30, dict(decimals=3)),
    ("min_peak_rate_hz", "Min peak Δrate (Hz)", "float", 5.0, dict(decimals=2)),
    ("min_peak_dist_s", "Min peak distance (s)", "float", 0.01, dict(decimals=3)),
    ("prom_frac", "Prominence (× max)", "float", 0.12, dict(decimals=3)),
    ("max_cv_period", "Max period CV (strict)", "float", 0.45, dict(decimals=3)),
    ("require_damping", "Require damping", "bool", False),
]


class RingingTab(TabBase):
    title = "5 · Ringing"
    key = "ringing"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.form = ParamForm(RING_SPEC, scroll=True)
        g = QGroupBox("Oscillation (triplet) detection"); QVBoxLayout(g).addWidget(self.form)
        left.addWidget(g, 1)
        left.addWidget(button("Run for all units → master table", self.run_batch, primary=True))
        self.unit = unit_spin()
        left.addWidget(hbox(QLabel("Unit"), self.unit, button("Plot unit", self.plot_unit), "stretch"))
        left.addWidget(button("Summary plots", self.summary))
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(430)
        right = QSplitter(Qt.Vertical)
        self.table = DataTable(); self.table.clicked.connect(self._row_clicked)
        self.plot = MplPanel()
        right.addWidget(self.table); right.addWidget(self.plot); right.setSizes([240, 480])
        lay.addWidget(lw); lay.addWidget(right, 1)
        self.paired = None

    def run_batch(self):
        if not self.mw.require(rec=True, sorting=True, triggers=True):
            return
        s, p = self.session, self.form.get()

        def job(log, progress, cancel):
            return ringing.run_ringing(s.sorting, s.fs, s.triggers, s.out_folder, p, log, progress, cancel)

        def done(res):
            _, self.paired = res
            self.table.set_df(self.paired)
            self.plot.show_figure(ringing.summary_figure(self.paired, {**ringing.DEFAULTS, **p}))
        self.run(job, on_done=done, label="Ringing")

    def summary(self):
        p = {**ringing.DEFAULTS, **self.form.get()}
        df = read_master(self.session)
        if df is None:
            return
        self.paired = ringing.paired_table(df, p)
        self.table.set_df(self.paired)
        self.plot.show_figure(ringing.summary_figure(self.paired, p))

    def _row_clicked(self, *_):
        r = self.table.selected_row()
        if r is not None:
            self.unit.setValue(int(r["unit_id"])); self.plot_unit()

    def plot_unit(self):
        if not self.mw.require(rec=True, sorting=True, triggers=True):
            return
        s, p, u = self.session, self.form.get(), self.unit.value()
        self.run(lambda log, progress, cancel: ringing.unit_figure(s.sorting, s.fs, s.triggers, u, p),
                 on_done=self.plot.show_figure, label=f"Ringing unit {u}")

    def on_session_changed(self):
        s = self.session
        st = trig_stims(s)
        if st:
            self.form.set_choices("stim_key", st)
            g = trig_groups(s, self.form.get()["stim_key"])
            self.form.set_choices("ref_group", g); self.form.set_choices("test_group", g)
            if len(g) > 1 and self.form.get()["test_group"] == self.form.get()["ref_group"]:
                self.form.set({"test_group": g[1]})

    def get_settings(self):
        return dict(form=self.form.get())

    def set_settings(self, d):
        self.form.set((d or {}).get("form", {}))


# ======================================================================
RF_SPEC = [
    ("lags", "Lags (s)", "str", rf_sta.DEFAULTS["lags"]),
    ("min_spikes", "Min spikes", "int", 500, dict(min=0)),
    ("n_null", "Null jitter repeats", "int", 40, dict(min=5)),
    ("jitter_s", "Jitter ± (s)", "float", 0.5, dict(decimals=3)),
    ("max_null_spikes", "Max spikes for null", "int", 6000, dict(min=0)),
    ("far_lag_gap_s", "Noise template: |lag − best| ≥ (s)", "float", 0.15, dict(decimals=3)),
    ("resid_smooth_sigma", "Residual smoothing σ (px)", "float", 2.0, dict(decimals=2)),
    ("roi_radius", "ROI radius (px)", "int", 45, dict(min=1)),
    ("rel_frac", "Threshold (× residual min)", "float", 0.35, dict(decimals=3)),
    ("min_rf_pix", "Min RF area (px)", "int", 60, dict(min=1)),
    ("gap_fix", "Ignore spikes in inter-trial gaps", "bool", True,
     dict(tip="Spikes > 1.5 frame intervals after the last flip are not assigned to that frame "
              "(the notebook assigned them to the trial's last frame)")),
    ("save_pngs", "Save panel PNGs (accepted)", "bool", True),
    ("include_ids", "Only these unit IDs (comma)", "str", ""),
    ("seed", "Random seed", "int", 0),
]


class RfTab(TabBase):
    title = "6 · RF mapping (SO)"
    key = "rf"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        gi = QGroupBox("Stimulus + timing")
        fi = QFormLayout(gi)
        self.npy = PathEdit("file", "", "NumPy (*.npy)")
        self.so_key = QComboBox(); self.so_key.setEditable(True); self.so_key.addItem("SO")
        fi.addRow("SO movie (.npy)", self.npy)
        fi.addRow("SO stimulus code", self.so_key)
        self.folder_box = QWidget(); self.folder_form = QFormLayout(self.folder_box)
        self.folder_form.setContentsMargins(0, 0, 0, 0)
        fi.addRow(QLabel("SO folder per group (SO_frames.csv + psychopy_frames.csv):"))
        fi.addRow(self.folder_box)
        left.addWidget(gi)
        self.form = ParamForm(RF_SPEC, scroll=True)
        g = QGroupBox("STA / RF detection"); QVBoxLayout(g).addWidget(self.form)
        left.addWidget(g, 1)
        left.addWidget(button("Run RF batch (checked groups)", self.run_batch, primary=True))
        self.view_group = QComboBox(); self.view_group.currentTextChanged.connect(self.show_table)
        self.unit = unit_spin()
        left.addWidget(hbox(QLabel("View"), self.view_group, QLabel("unit"), self.unit,
                            button("Show RF", self.show_unit)))
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(460)
        right = QSplitter(Qt.Vertical)
        self.table = DataTable(); self.table.clicked.connect(self._row_clicked)
        self.plot = MplPanel()
        right.addWidget(self.table); right.addWidget(self.plot); right.setSizes([230, 520])
        lay.addWidget(lw); lay.addWidget(right, 1)
        self.group_edits = {}
        self.group_checks = {}
        self._timing = {}

    def _build_group_rows(self):
        while self.folder_form.rowCount():
            self.folder_form.removeRow(0)
        self.group_edits, self.group_checks = {}, {}
        s = self.session
        key = self.so_key.currentText()
        for g in trig_groups(s, key) or s.groups():
            ed = PathEdit("dir", self._auto_folder(g))
            chk = QCheckBox(g); chk.setChecked(True)
            self.group_edits[g], self.group_checks[g] = ed, chk
            self.folder_form.addRow(chk, ed)
        self.view_group.blockSignals(True)
        cur = self.view_group.currentText()
        self.view_group.clear(); self.view_group.addItems(list(self.group_edits))
        if cur:
            self.view_group.setCurrentText(cur)
        self.view_group.blockSignals(False)

    def _auto_folder(self, g):
        s = self.session
        saved = s.settings.get(self.key, {}).get("folders", {}).get(g)
        if saved:
            return saved
        if s.table is None:
            return ""
        rows = s.table[(s.table.stimulus == self.so_key.currentText()) & (s.table.group == g)]
        return align_folder_of(rows.iloc[0].to_dict()) if len(rows) else ""

    def _params(self):
        p = self.form.get()
        gap = p.pop("gap_fix")
        return p, gap

    def _timing_args(self, g):
        """Collect everything from the widgets in the GUI thread."""
        from ..core.triggers import get_times
        _, gap = self._params()
        so_on = get_times(self.session.triggers, self.so_key.currentText(), g, "ON")
        return (g, self.npy.text(), self.group_edits[g].text(), bool(gap), tuple(np.round(so_on, 6)))

    def _timing_for(self, args, log):
        if args not in self._timing:
            g, npy, folder, gap, so_on = args
            if not so_on:
                raise RuntimeError(f"No '{self.so_key.currentText()}' ON triggers for group {g}")
            self._timing[args] = rf_sta.SoTiming(npy, folder, np.asarray(so_on), gap_fix=gap, log=log)
        return self._timing[args]

    def run_batch(self):
        if not self.mw.require(rec=True, sorting=True, triggers=True):
            return
        if not os.path.isfile(self.npy.text()):
            QMessageBox.information(self, "SO movie", "Choose the SO stimulus .npy file."); return
        groups = [g for g, c in self.group_checks.items() if c.isChecked()]
        targs = {g: self._timing_args(g) for g in groups}
        s = self.session
        p, _ = self._params()

        def job(log, progress, cancel):
            out = {}
            for i, g in enumerate(groups):
                tim = self._timing_for(targs[g], log)
                out[g] = rf_sta.run_rf_batch(s.sorting, s.fs, tim, g, s.out_folder, p, log,
                                             lambda f, i=i: progress((i + f) / len(groups)), cancel)
            return out

        def done(res):
            self.mw.save_session(quiet=True)
            if res:
                self.view_group.setCurrentText(list(res)[0])
            self.show_table()
        self.run(job, on_done=done, label="SO RF batch")

    def show_table(self, *_):
        g = self.view_group.currentText()
        p = rf_sta.rf_csv(self.session.out_folder, g) if g and self.session.out_folder else ""
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            cols = [c for c in ["mea_unit_id", "ok", "reason", "n_spikes_total", "best_lag_s", "resid_min", "comp_area",
                                "centroid_cx_px", "centroid_cy_px"] if c in df]
            self.table.set_df(df[cols].sort_values(["ok", "resid_min"], ascending=[False, True])
                              if "resid_min" in df else df[cols])
        else:
            self.table.set_df(pd.DataFrame())

    def _row_clicked(self, *_):
        r = self.table.selected_row()
        if r is not None:
            self.unit.setValue(int(r["mea_unit_id"])); self.show_unit()

    def show_unit(self):
        if not self.mw.require(rec=True, sorting=True, triggers=True):
            return
        g, u, s = self.view_group.currentText(), self.unit.value(), self.session
        if g not in self.group_edits or not os.path.isfile(self.npy.text()):
            QMessageBox.information(self, "RF", "Choose the SO movie and a group first."); return
        p, _ = self._params()
        targs = self._timing_args(g)

        def job(log, progress, cancel):
            tim = self._timing_for(targs, log)
            sp = rf_sta.spike_times_s(s.sorting, u, s.fs).astype(np.float32)
            res = rf_sta.analyse_unit(tim, sp, {**rf_sta.DEFAULTS, **p}, np.random.default_rng(int(p["seed"])))
            if "best" not in res:
                raise RuntimeError(f"Unit {u}: {res['row'].get('reason')}")
            return rf_sta.panel_figure(u, res, rf_sta.DEFAULTS["display_smooth"])
        self.run(job, on_done=self.plot.show_figure, label=f"RF unit {u} ({g})")

    def on_session_changed(self):
        st = trig_stims(self.session)
        if st:
            cur = self.so_key.currentText()
            self.so_key.clear(); self.so_key.addItems(st)
            self.so_key.setCurrentText(cur if cur in st else ("SO" if "SO" in st else st[0]))
        self._build_group_rows()
        self.show_table()

    def get_settings(self):
        return dict(form=self.form.get(), npy=self.npy.text(), so_key=self.so_key.currentText(),
                    folders={g: e.text() for g, e in self.group_edits.items()})

    def set_settings(self, d):
        d = d or {}
        self.form.set(d.get("form", {})); self.npy.setText(d.get("npy", ""))
        if d.get("so_key"):
            self.so_key.setCurrentText(d["so_key"])
        self._timing = {}


# ======================================================================
STAB_SPEC = [
    ("max_shift_px", "Max centroid shift (px)", "float", 12.0, dict(decimals=1)),
    ("min_comp_area", "Min RF area (px)", "int", 80, dict(min=0)),
    ("require_ok_both", "Require accepted in both", "bool", True),
]
CAL_SPEC = [
    ("model", "Model", "choice", "similarity", dict(choices=["similarity", "affine"])),
    ("anchor_source", "Anchor RF centroid", "choice", "MEAN", dict(choices=["MEAN", "A only"])),
    ("try_orientations", "Search flips / axis swap", "bool", True),
    ("use_ransac", "RANSAC", "bool", True),
    ("ransac_iters", "RANSAC iterations", "int", 1200, dict(min=10)),
    ("inlier_thresh_um", "Inlier threshold (µm)", "float", 220.0, dict(decimals=1)),
    ("min_inliers", "Min inliers", "int", 30, dict(min=3)),
]


class CalibTab(TabBase):
    title = "7 · RF → MEA"
    key = "calib"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.ga = QComboBox(); self.gb = QComboBox()
        left.addWidget(hbox(QLabel("Group A"), self.ga, QLabel("Group B"), self.gb))
        g1 = QGroupBox("1. RF stability A vs B"); v1 = QVBoxLayout(g1)
        self.stab = ParamForm(STAB_SPEC); v1.addWidget(self.stab)
        v1.addWidget(button("Find stable anchor units", self.run_stab, primary=True))
        left.addWidget(g1)
        g2 = QGroupBox("2. Fit stimulus px → MEA µm"); v2 = QVBoxLayout(g2)
        self.cal = ParamForm(CAL_SPEC); v2.addWidget(self.cal)
        v2.addWidget(button("Fit transform on anchors + project all RFs", self.run_fit, primary=True))
        left.addWidget(g2)
        g3 = QGroupBox("3. View"); v3 = QVBoxLayout(g3)
        self.unit = unit_spin()
        self.gate = QDoubleSpinBox(); self.gate.setRange(0, 1e5); self.gate.setValue(400)
        v3.addWidget(hbox(QLabel("Unit"), self.unit, QLabel("A–B gate µm"), self.gate))
        v3.addWidget(hbox(button("RF outline(s) on MEA", self.show_unit), button("All centroids", self.show_all)))
        left.addWidget(g3)
        left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(430)
        right = QSplitter(Qt.Vertical)
        self.table = DataTable(); self.table.clicked.connect(self._row_clicked)
        self.plot = MplPanel()
        right.addWidget(self.table); right.addWidget(self.plot); right.setSizes([220, 520])
        lay.addWidget(lw); lay.addWidget(right, 1)
        self.stable = None

    def _groups(self):
        return self.ga.currentText(), self.gb.currentText()

    def run_stab(self):
        s = self.session
        ga, gb = self._groups()
        ca, cb = rf_sta.rf_csv(s.out_folder, ga), rf_sta.rf_csv(s.out_folder, gb)
        if not (os.path.exists(ca) and os.path.exists(cb)):
            QMessageBox.information(self, "RF results missing", f"Run the RF batch for {ga} and {gb} first."); return
        self.stable = rf_calib.rf_stability(ca, cb, ga, gb, s.out_folder, self.stab.get(), log=self.log)
        self.table.set_df(self.stable)

    def run_fit(self):
        s = self.session
        ga, gb = self._groups()
        if self.stable is None:
            p = os.path.join(rf_calib.calib_dir(s.out_folder), "rf_all_common_filtered.csv")
            if os.path.exists(p):
                self.stable = pd.read_csv(p)
            else:
                self.run_stab()
        if self.stable is None:
            return
        uxy = hs.load_unit_xy(s.out_folder)
        if not uxy:
            QMessageBox.information(self, "No unit locations", "Run 'Localise units' in the Sorting tab first."); return
        p = self.cal.get()
        p["anchor_source"] = "MEAN" if p["anchor_source"] == "MEAN" else "A"
        try:
            tr, pairs = rf_calib.fit_rf_to_mea(self.stable[self.stable.stable.astype(bool)], ga, gb, uxy,
                                               s.out_folder, p, log=self.log)
        except Exception as e:
            QMessageBox.warning(self, "Fit failed", str(e)); return
        for g in (ga, gb):
            rf_calib.project_centroids(rf_sta.rf_csv(s.out_folder, g), g, tr, s.out_folder)
        self.table.set_df(pairs)
        self.plot.show_figure(rf_calib.anchors_figure(pairs, tr))

    def _row_clicked(self, *_):
        r = self.table.selected_row()
        if r is not None and "mea_unit_id" in r:
            self.unit.setValue(int(r["mea_unit_id"])); self.show_unit()

    def show_unit(self):
        s = self.session
        tr = rf_calib.load_transform(s.out_folder, self.cal.get()["model"])
        if tr is None:
            QMessageBox.information(self, "No transform", "Fit the transform first (step 2)."); return
        fig = rf_calib.contour_figure(self.unit.value(), list(dict.fromkeys(self._groups())), s.out_folder, tr,
                                      hs.load_unit_xy(s.out_folder), self.gate.value())
        self.plot.show_figure(fig)

    def show_all(self):
        s = self.session
        proj = {}
        for g in dict.fromkeys(self._groups()):
            p = os.path.join(rf_calib.calib_dir(s.out_folder), f"{g}_RF_centroids_in_MEA.csv")
            if os.path.exists(p):
                proj[g] = pd.read_csv(p)
        if not proj:
            QMessageBox.information(self, "Nothing projected", "Fit the transform first (step 2)."); return
        self.plot.show_figure(rf_calib.all_centroids_figure(proj, hs.load_unit_xy(s.out_folder)))

    def on_session_changed(self):
        g = trig_groups(self.session)
        for cb, i in ((self.ga, 0), (self.gb, 1)):
            cur = cb.currentText()
            cb.clear(); cb.addItems(g)
            if cur in g:
                cb.setCurrentText(cur)
            elif len(g) > i:
                cb.setCurrentIndex(i)
        self.stable = None

    def get_settings(self):
        return dict(stab=self.stab.get(), cal=self.cal.get())

    def set_settings(self, d):
        d = d or {}
        self.stab.set(d.get("stab", {})); self.cal.set(d.get("cal", {}))


# ======================================================================
CCF_SPEC = [
    ("bin_size_s", "Bin (s)", "float", 0.001, dict(decimals=4, step=0.0005)),
    ("lag_ms", "Lag window ± (ms)", "float", 50.0, dict(decimals=1)),
    ("min_spikes", "Min spikes in window", "int", 100, dict(min=1)),
    ("max_dist_um", "Max pair distance (µm)", "float", 500.0, dict(decimals=1)),
    ("smooth_sigma_ms", "CCF smoothing σ (ms)", "float", 0.8, dict(decimals=2)),
    ("snr_min", "Min SNR", "float", 5.0, dict(decimals=2)),
    ("central_peak_limit_ms", "Peak must be within ± (ms)", "float", 7.0, dict(decimals=2)),
    ("min_width_ms", "Min peak width (ms)", "float", 4.0, dict(decimals=2)),
    ("narrow_wide_split_ms", "Narrow/wide split (ms)", "float", 15.0, dict(decimals=2)),
    ("min_abs_peak", "Min absolute peak", "float", 0.8, dict(decimals=3)),
    ("mean_ratio_min", "Peak / mean ratio", "float", 3.0, dict(decimals=2)),
    ("edge_factor", "Peak / edge factor", "float", 1.5, dict(decimals=2)),
]


class CouplingTab(TabBase):
    title = "8 · Coupling (CCF)"
    key = "coupling"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        gw = QGroupBox("Time window (spontaneous activity)"); fw = QFormLayout(gw)
        self.win = QComboBox(); self.win.currentIndexChanged.connect(self._win_changed)
        self.t0 = QDoubleSpinBox(); self.t0.setRange(0, 1e7); self.t1 = QDoubleSpinBox(); self.t1.setRange(0, 1e7)
        fw.addRow("Segment", self.win); fw.addRow("Start (s)", self.t0); fw.addRow("End (s)", self.t1)
        left.addWidget(gw)
        self.form = ParamForm(CCF_SPEC, scroll=True)
        g = QGroupBox("CCF classification"); QVBoxLayout(g).addWidget(self.form)
        left.addWidget(g, 1)
        self.hubs = QLineEdit(); self.hubs.setPlaceholderText("empty = all units as hubs")
        left.addWidget(hbox(QLabel("Hubs"), self.hubs))
        left.addWidget(button("Compute CCFs + classify", self.run_batch, primary=True))
        self.cat = QComboBox(); self.cat.addItems(["coupled only", "all", "Unimodal-narrow", "Unimodal-wide", "Bimodal",
                                                   "No-peak"])
        self.cat.currentTextChanged.connect(self.show_table)
        self.min_snr = QDoubleSpinBox(); self.min_snr.setRange(0, 1e4); self.min_snr.setValue(5)
        left.addWidget(hbox(QLabel("Show"), self.cat, QLabel("SNR ≥"), self.min_snr))
        left.addWidget(hbox(button("Network map", self.network), button("Network of selected hub", self.network_hub)))
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(430)
        right = QSplitter(Qt.Vertical)
        self.table = DataTable(); self.table.clicked.connect(self._row_clicked)
        self.plot = MplPanel()
        right.addWidget(self.table); right.addWidget(self.plot); right.setSizes([260, 480])
        lay.addWidget(lw); lay.addWidget(right, 1)
        self.df = None; self.bt = None; self._bt_key = None

    def _win_changed(self, i):
        d = self.win.itemData(i)
        if d:
            self.t0.setValue(d[0]); self.t1.setValue(d[1])

    def _window(self):
        return (self.t0.value(), self.t1.value())

    def run_batch(self):
        if not self.mw.require(sorting=True):
            return
        s, p, w = self.session, self.form.get(), self._window()
        if w[1] <= w[0]:
            QMessageBox.information(self, "Window", "End must be after start."); return
        uxy = hs.load_unit_xy(s.out_folder)
        if not uxy:
            QMessageBox.information(self, "No unit locations", "Run 'Localise units' in the Sorting tab first."); return
        hubs = [int(x) for x in self.hubs.text().replace(";", ",").split(",") if x.strip()] or None

        def job(log, progress, cancel):
            return coupling.run_coupling(s.sorting, s.fs, w, uxy, s.out_folder, p, hubs, log, progress, cancel)

        def done(res):
            self.df, self.bt = res
            self._bt_key = (w, p["bin_size_s"])
            self.show_table()
            self.network()
        self.run(job, on_done=done, label="CCF coupling")

    def show_table(self, *_):
        if self.df is None:
            p = os.path.join(self.session.out_folder or "", "CCF_coupling", "ccf_pairs.csv")
            if self.session.out_folder and os.path.exists(p):
                self.df = pd.read_csv(p)
            else:
                self.table.set_df(pd.DataFrame()); return
        c = self.cat.currentText()
        d = self.df
        if c == "coupled only":
            d = d[(d.category != "No-peak") & (d.coupling_strength >= self.min_snr.value())]
        elif c != "all":
            d = d[d.category == c]
        self.table.set_df(d.sort_values("coupling_strength", ascending=False))

    def _ensure_bt(self):
        s, p, w = self.session, self.form.get(), self._window()
        if self.bt is None or self._bt_key != (w, p["bin_size_s"]):
            self.bt = coupling.BinnedTrains(s.sorting, s.fs, w[0], w[1], p["bin_size_s"])
            self._bt_key = (w, p["bin_size_s"])
        return self.bt

    def _row_clicked(self, *_):
        r = self.table.selected_row()
        if r is None or self.session.sorting is None:
            return
        bt = self._ensure_bt()
        self.plot.show_figure(coupling.pair_figure(bt, int(r["hub"]), int(r["partner"]), self.form.get()))

    def network(self, highlight=None):
        if self.df is None:
            self.show_table()
        if self.df is None:
            return
        uxy = hs.load_unit_xy(self.session.out_folder)
        self.plot.show_figure(coupling.network_figure(self.df, uxy, min_snr=self.min_snr.value(),
                                                      highlight=highlight))

    def network_hub(self):
        r = self.table.selected_row()
        if r is not None:
            self.network(int(r["hub"]))

    def on_session_changed(self):
        s = self.session
        self.win.blockSignals(True)
        cur = self.win.currentText()
        self.win.clear(); self.win.addItem("custom", None)
        if s.table is not None:
            for r in s.table.itertuples():
                self.win.addItem(f"{r.File} ({r.stimulus}/{r.group}) {r.start:.0f}–{r.end:.0f} s",
                                 (float(r.start), float(r.end)))
        i = self.win.findText(cur)
        self.win.setCurrentIndex(i if i >= 0 else 0)
        self.win.blockSignals(False)
        self.df = None; self.bt = None
        self.show_table()

    def get_settings(self):
        return dict(form=self.form.get(), t0=self.t0.value(), t1=self.t1.value(), hubs=self.hubs.text())

    def set_settings(self, d):
        d = d or {}
        self.form.set(d.get("form", {}))
        self.t0.setValue(d.get("t0", 0)); self.t1.setValue(d.get("t1", 0)); self.hubs.setText(d.get("hubs", ""))
