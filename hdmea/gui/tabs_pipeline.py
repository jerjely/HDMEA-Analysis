"""Session, Triggers and Sorting tabs."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QListWidget, QTableWidget, QTableWidgetItem,
                               QFileDialog, QGroupBox, QSplitter, QWidget, QLabel, QCheckBox, QMessageBox,
                               QComboBox, QDoubleSpinBox, QHeaderView, QAbstractItemView, QLineEdit)

from ..core import session as sess_mod, triggers, sorting as hs
from ..core.common import DEFAULT_STIM_REGEX, load_json, align_folder_of, has_lux_files, match_run_folders
from .main_window import TabBase
from .widgets import ParamForm, PathEdit, MplPanel, DataTable, button, hbox


# ======================================================================
def load_state(session, log=print, progress=None, cancel=None):
    """Lazy-load whatever already exists for a session (zarr, triggers, sorting)."""
    out = {}
    if os.path.exists(session.zarr_path):
        out["rec"] = sess_mod.load_or_build_zarr(session, rebuild=False, log=log)
    tp = session.triggers_path
    if not tp and session.out_folder and os.path.isdir(session.out_folder):
        cands = [f for f in os.listdir(session.out_folder) if f.startswith("ON_OFF_luxcorr_MEA_") and f.endswith(".json")]
        tp = os.path.join(session.out_folder, cands[0]) if cands else ""
    if tp and os.path.exists(tp):
        out["triggers"] = load_json(tp); out["triggers_path"] = tp
        log(f"Loaded triggers: {tp}")
    sf = session.sorting_folder
    if not sf:
        runs = hs.list_runs(session.out_folder) if session.out_folder else []
        sf = runs[0] if runs else ""
    if sf and os.path.isdir(sf):
        try:
            out["sorting"] = hs.read_sorting(sf); out["sorting_folder"] = sf
            log(f"Loaded sorting: {sf} ({len(out['sorting'].unit_ids)} units)")
        except Exception as e:
            log(f"⚠ Could not load sorting {sf}: {e}")
    return out


def apply_state(mw, st):
    s = mw.session
    if "rec" in st:
        s.rec = st["rec"]
        if s.table is not None and abs(s.rec.get_total_duration() - float(s.table["duration_s"].sum())) > 1.0:
            mw.log(f"⚠ Zarr cache is {s.rec.get_total_duration():.1f} s but the table adds up to "
                   f"{float(s.table['duration_s'].sum()):.1f} s – rebuild it in the Session tab "
                   f"(tick 'Rebuild Zarr even if it exists').")
    if "triggers" in st:
        s.triggers, s.triggers_path = st["triggers"], st["triggers_path"]
    if "sorting" in st:
        s.sorting, s.sorting_folder = st["sorting"], st["sorting_folder"]
    mw.notify_session_changed()


# ======================================================================
class SessionTab(TabBase):
    title = "1 · Session"
    key = "session"
    COLS = ["File", "stimulus", "group", "duration_s", "start", "end", "lux", "PsychoPy/lux folder"]

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()

        g1 = QGroupBox("BRW files (concatenated in this order)")
        v = QVBoxLayout(g1)
        v.addWidget(hbox(button("Add BRW…", self.add_files), button("Remove", self.remove_sel),
                         button("▲", lambda: self.move(-1)), button("▼", lambda: self.move(1)),
                         button("Clear", self.clear_files)))
        self.out_edit = PathEdit("dir", "")
        v.addWidget(QLabel("Output folder (results + Zarr cache):"))
        v.addWidget(self.out_edit)
        self.form = ParamForm([
            ("stim_regex", "Stimulus regex", "str", DEFAULT_STIM_REGEX,
             dict(tip="First group = stimulus code; digits are stripped (SO1 → SO)")),
            ("drug_tokens", "Drug tokens", "str", "PTX",
             dict(tip="Comma-separated. A file whose name contains a token gets that group, otherwise CTRL")),
            ("mea_pitch", "MEA pitch (µm)", "float", 42.0, dict(decimals=1)),
            ("n_jobs", "Zarr write jobs", "int", -1, dict(min=-1, max=128, tip="-1 = all CPU cores")),
        ])
        v.addWidget(self.form)
        v.addWidget(button("Read file headers → table", self.read_headers, primary=True))
        left.addWidget(g1)

        g2 = QGroupBox("Concatenated recording")
        v2 = QVBoxLayout(g2)
        self.rebuild_chk = QCheckBox("Rebuild Zarr even if it exists")
        v2.addWidget(self.rebuild_chk)
        v2.addWidget(button("Build / load concatenated recording", self.build_zarr, primary=True,
                            tip="BRW → concatenate → unsigned→signed → Zarr cache in the output folder"))
        v2.addWidget(button("Save session", lambda: self.mw.save_session()))
        left.addWidget(g2)
        left.addStretch(1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Per-file table — edit <b>stimulus</b> and <b>group</b> by double-clicking. "
                               "Groups drive CTRL/PTX splits everywhere."))
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._item_changed)
        right.addWidget(self.table)
        right.addWidget(hbox(button("Set PsychoPy/lux folder for selected file…", self.set_align_folder),
                             button("Auto-match run folders by time…", self.auto_match,
                                    tip="Pick the parent folder of the so_veml_YYYYMMDD_HHMMSS runs; each BRW gets "
                                        "the run whose timestamp is closest to its start"),
                             "stretch"))
        self.info = QLabel("")
        right.addWidget(self.info)

        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(470)
        rw = QWidget(); rw.setLayout(right)
        lay.addWidget(lw); lay.addWidget(rw, 1)
        self._filling = False

    # -------------------------------------------------------------- files
    def _files(self):
        s = self.session
        if s.table is not None:
            return list(s.table["path"])
        return list(s.brw_files)

    def add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add BRW files", self.mw.qs.value("brw_dir", ""), "BioCam (*.brw)")
        if not ps:
            return
        self.mw.qs.setValue("brw_dir", os.path.dirname(ps[0]))
        files = self._files() + [p for p in ps if p not in self._files()]
        self.session.brw_files = files
        self.session.table = None
        if not self.out_edit.text():
            self.out_edit.setText(sess_mod.default_out_folder(files))
        self.fill_table()
        self.log(f"{len(files)} BRW file(s) in list – press 'Read file headers'.")

    def remove_sel(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        s = self.session
        if s.table is not None:
            s.table = sess_mod.recompute_starts(s.table.drop(index=s.table.index[rows]).reset_index(drop=True))
            s.brw_files = list(s.table["path"])
        else:
            for r in rows:
                s.brw_files.pop(r)
        self._order_changed()

    def move(self, d):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if len(rows) != 1:
            return
        r = rows[0]; n = r + d
        s = self.session
        if n < 0 or n >= len(self._files()):
            return
        if s.table is not None:
            idx = list(range(len(s.table))); idx[r], idx[n] = idx[n], idx[r]
            s.table = sess_mod.recompute_starts(s.table.iloc[idx].reset_index(drop=True))
            s.brw_files = list(s.table["path"])
        else:
            s.brw_files[r], s.brw_files[n] = s.brw_files[n], s.brw_files[r]
        self._order_changed()
        self.table.selectRow(n)

    def clear_files(self):
        self.session.brw_files = []; self.session.table = None; self.fill_table()

    def _order_changed(self):
        self.fill_table()
        if self.session.rec is not None:
            self.log("⚠ File list/order changed – rebuild the Zarr cache so times match the table.")

    # -------------------------------------------------------------- table
    def fill_table(self):
        self._filling = True
        s = self.session
        if s.table is not None:
            df = s.table
        else:
            df = pd.DataFrame(dict(File=[os.path.basename(f) for f in s.brw_files]))
        self.table.setRowCount(len(df))
        for i in range(len(df)):
            folder = align_folder_of(df.iloc[i].to_dict()) if "path" in df else ""
            for j, c in enumerate(self.COLS):
                if c == "lux":
                    v = "✓" if has_lux_files(folder) else "–"
                elif c == "PsychoPy/lux folder":
                    v = folder
                else:
                    v = df[c].iloc[i] if c in df else ""
                txt = f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
                it = QTableWidgetItem(txt)
                if c not in ("stimulus", "group"):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                else:
                    it.setBackground(QColor("#fff6d5"))
                self.table.setItem(i, j, it)
        self._filling = False
        if s.table is not None:
            st = ", ".join(f"{a}/{b}: {v}" for (a, b), v in s.table.groupby(["stimulus", "group"], sort=False).size().items())
            self.info.setText(f"Total {s.table['duration_s'].sum() / 60:.1f} min | {st}")
        else:
            self.info.setText("")

    def _item_changed(self, it):
        if self._filling or self.session.table is None:
            return
        c = self.COLS[it.column()]
        if c in ("stimulus", "group"):
            self.session.table.loc[it.row(), c] = it.text().strip()
            self.mw.notify_session_changed()

    def set_align_folder(self):
        s = self.session
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if s.table is None or not rows:
            QMessageBox.information(self, "Select a file", "Read the headers, then select a row."); return
        start = align_folder_of(s.table.iloc[rows[0]].to_dict())
        f = QFileDialog.getExistingDirectory(self, "Folder with lux_stream.csv / events_GH.csv / SO_frames.csv",
                                             start if os.path.isdir(start) else os.path.dirname(start))
        if not f:
            return
        if "align_folder" not in s.table:
            s.table["align_folder"] = ""
        for r in rows:
            s.table.loc[s.table.index[r], "align_folder"] = f
        self.fill_table(); self.mw.notify_session_changed()
        self.log(f"PsychoPy/lux folder set for {len(rows)} file(s): {f}" +
                 ("" if has_lux_files(f) else "  (⚠ no lux_stream.csv + events_GH.csv there)"))

    def auto_match(self):
        s = self.session
        if s.table is None:
            QMessageBox.information(self, "Read headers", "Press 'Read file headers' first."); return
        parent = QFileDialog.getExistingDirectory(self, "Parent folder of the PsychoPy run folders",
                                                  self.mw.qs.value("runs_dir", ""))
        if not parent:
            return
        self.mw.qs.setValue("runs_dir", parent)
        m = match_run_folders(s.table, parent, log=self.log)
        if "align_folder" not in s.table:
            s.table["align_folder"] = ""
        for i, f in m.items():
            s.table.loc[i, "align_folder"] = f
        self.fill_table(); self.mw.notify_session_changed()
        self.log(f"Matched {len(m)} / {len(s.table)} file(s). Check the 'PsychoPy/lux folder' column, "
                 f"then re-run Extract triggers.")

    def read_headers(self):
        files = self._files()
        if not files:
            QMessageBox.information(self, "No files", "Add BRW files first."); return
        p = self.form.get()
        toks = [t.strip() for t in p["drug_tokens"].split(",") if t.strip()]

        def job(log, progress, cancel):
            return sess_mod.build_record_table(files, p["stim_regex"], toks, p["mea_pitch"], log=log)

        def done(df):
            self.session.table = df
            self.session.brw_files = list(df["path"])
            self._sync_out()
            self.fill_table()
            self.mw.notify_session_changed()
        self.run(job, on_done=done, label="Read BRW headers")

    def _sync_out(self):
        self.session.out_folder = self.out_edit.text() or (sess_mod.default_out_folder(self._files()) if self._files() else "")
        self.out_edit.setText(self.session.out_folder)

    def build_zarr(self):
        if self.session.table is None:
            QMessageBox.information(self, "Read headers", "Press 'Read file headers' first."); return
        self._sync_out()
        os.makedirs(self.session.out_folder, exist_ok=True)
        self.mw.save_session(quiet=True)
        p = self.form.get()
        rebuild = self.rebuild_chk.isChecked()
        s = self.session

        def job(log, progress, cancel):
            return sess_mod.load_or_build_zarr(s, rebuild=rebuild, mea_pitch=p["mea_pitch"], n_jobs=p["n_jobs"],
                                               log=log, progress=progress)

        def done(rec):
            s.rec = rec
            dur_tab = float(s.table["duration_s"].sum())
            if abs(rec.get_total_duration() - dur_tab) > 1.0:
                msg = (f"The Zarr cache is {rec.get_total_duration():.1f} s long but the BRW files in the table add "
                       f"up to {dur_tab:.1f} s.\n\nThe cache was probably built earlier from other files (e.g. by "
                       f"the notebook, which uses the same folder and name). Times after the end of the cache "
                       f"have no data, so triggers/spikes there are missing.\n\nRebuild the cache now?")
                self.log("⚠ " + msg.splitlines()[0])
                if QMessageBox.question(self, "Zarr cache doesn't match", msg) == QMessageBox.Yes:
                    self.rebuild_chk.setChecked(True)
                    s.rec = None
                    self.build_zarr()
                    return
            self.mw.notify_session_changed()
        self.run(job, on_done=done, label="Concatenate → Zarr")

    def autoload(self):
        s = self.session
        self.run(lambda log, progress, cancel: load_state(s, log), on_done=lambda st: apply_state(self.mw, st),
                 label="Load session data")

    # -------------------------------------------------------------- settings
    def get_settings(self):
        return dict(form=self.form.get())

    def set_settings(self, d):
        self.form.set((d or {}).get("form", {}))

    def on_session_changed(self):
        self.out_edit.setText(self.session.out_folder)
        self.fill_table()


# ======================================================================
TRIG_SPEC = [
    ("row", "Trigger electrode row", "int", 1, dict(min=1, max=64)),
    ("col", "Trigger electrode col", "int", 2, dict(min=1, max=64)),
    ("auto_thresholds", "Auto thresholds (10/90 pct)", "bool", False),
    ("low_thr", "Low threshold (OFF)", "float", -30200.0, dict(decimals=1, step=10)),
    ("high_thr", "High threshold (ON)", "float", -29900.0, dict(decimals=1, step=10)),
    ("mid_thr", "Mid (initial state)", "float", -30000.0, dict(decimals=1, step=10)),
    ("min_pw_s", "Min pulse width (s)", "float", 0.05, dict(decimals=3)),
    ("use_lux", "Lux correction (lux_stream / events_GH)", "bool", True,
     dict(tip="Uses <brw name>/lux_stream.csv + events_GH.csv next to each BRW")),
    ("lux_thr", "Lux threshold", "float", 50.0, dict(decimals=1)),
    ("per_file_median", "Per-file median delay", "bool", True,
     dict(tip="Avoids the ~100 ms lux sampling jitter; otherwise per-pulse delays")),
    ("ignore_off_for_so", "SO: ignore OFF edges", "bool", True),
    ("fixed_shift_s", "Fixed shift if no lux files (s)", "float", 0.0, dict(decimals=3,
                                                                              tip="Old notebooks used +0.12 s")),
    ("start_margin_s", "Drop events < X s after file start", "float", 0.0, dict(decimals=2)),
    ("end_margin_s", "Drop events < X s before file end", "float", 0.25, dict(decimals=2)),
    ("chunk_s", "Read chunk (s)", "float", 10.0, dict(decimals=1)),
]


class TriggersTab(TabBase):
    title = "2 · Triggers"
    key = "triggers"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.form = ParamForm(TRIG_SPEC)
        g = QGroupBox("Trigger detection"); QVBoxLayout(g).addWidget(self.form)
        left.addWidget(g)
        left.addWidget(button("Extract triggers", self.run_extract, primary=True))
        left.addWidget(button("Load trigger JSON…", self.load_json))
        gq = QGroupBox("QC plot")
        self.t0 = QDoubleSpinBox(); self.t0.setRange(0, 1e6); self.t0.setValue(0)
        self.t1 = QDoubleSpinBox(); self.t1.setRange(0, 1e6); self.t1.setValue(60)
        vq = QVBoxLayout(gq)
        vq.addWidget(hbox(QLabel("from"), self.t0, QLabel("to"), self.t1, QLabel("s")))
        vq.addWidget(button("Plot trigger trace + events", self.plot_qc))
        self.lux_file = QComboBox()
        vq.addWidget(QLabel("Lux signal + G/H events for one file:"))
        vq.addWidget(self.lux_file)
        vq.addWidget(button("Plot lux + trigger electrode", self.plot_lux))
        left.addWidget(gq)
        left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(430)
        right = QSplitter(Qt.Vertical)
        self.sum_tab = DataTable(); self.dbg_tab = DataTable()
        self.plot = MplPanel()
        right.addWidget(self.sum_tab); right.addWidget(self.dbg_tab); right.addWidget(self.plot)
        right.setSizes([140, 200, 420])
        lay.addWidget(lw); lay.addWidget(right, 1)

    def run_extract(self):
        if not self.mw.require(rec=True):
            return
        s, p = self.session, self.form.get()

        def job(log, progress, cancel):
            return triggers.extract_triggers(s.rec, s.table, p, s.out_folder, log, progress, cancel)

        def done(trig):
            s.triggers, s.triggers_path = trig, trig.pop("_path", "")
            self.mw.save_session(quiet=True)
            self.mw.notify_session_changed()
            self.plot_qc(auto=True)
        self.run(job, on_done=done, label="Extract triggers")

    def load_json(self):
        p, _ = QFileDialog.getOpenFileName(self, "Trigger JSON", self.session.out_folder, "JSON (*.json)")
        if p:
            self.session.triggers, self.session.triggers_path = load_json(p), p
            self.mw.notify_session_changed()

    def plot_qc(self, auto=False):
        s = self.session
        if s.rec is None or not s.triggers:
            if not auto:
                self.mw.require(rec=True, triggers=True)
            return
        t0, t1 = self.t0.value(), self.t1.value()
        if auto and s.triggers.get("ON_corr_all"):
            t0 = max(0.0, s.triggers["ON_corr_all"][0] - 5); t1 = t0 + 40
            self.t0.setValue(t0); self.t1.setValue(t1)
        if t1 <= t0:
            return
        self.run(lambda log, progress, cancel: triggers.qc_figure(s.rec, s.triggers, t0, t1),
                 on_done=self.plot.show_figure, label="QC plot")

    def plot_lux(self):
        s = self.session
        i = self.lux_file.currentIndex()
        if s.table is None or i < 0 or not self.mw.require(rec=True):
            return
        row, p = s.table.iloc[i].to_dict(), self.form.get()
        self.run(lambda log, progress, cancel: triggers.lux_figure(s.rec, row, p), on_done=self.plot.show_figure,
                 label=f"Lux plot {row['File']}")

    def on_session_changed(self):
        s = self.session
        cur = self.lux_file.currentIndex()
        self.lux_file.clear()
        if s.table is not None:
            for r in s.table.itertuples():
                self.lux_file.addItem(f"{r.File}  [{'lux ✓' if has_lux_files(align_folder_of(r)) else 'no lux'}]")
            if 0 <= cur < self.lux_file.count():
                self.lux_file.setCurrentIndex(cur)
        trig = self.session.triggers
        if trig:
            self.sum_tab.set_df(triggers.summary_table(trig))
            dbg = pd.DataFrame(trig.get("dbg_df", {}))
            keep = [c for c in dbg.columns if c not in ("ON_corr_mea", "OFF_corr_mea", "brw_path")]
            self.dbg_tab.set_df(dbg[keep] if len(dbg) else dbg)
            if "params" in trig:
                self.form.set(trig["params"])
        else:
            self.sum_tab.set_df(pd.DataFrame()); self.dbg_tab.set_df(pd.DataFrame())

    def get_settings(self):
        return dict(form=self.form.get())

    def set_settings(self, d):
        self.form.set((d or {}).get("form", {}))


# ======================================================================
HS_SPEC = [
    ("freq_min", "Bandpass low (Hz)", "float", 350.0, dict(decimals=1)),
    ("freq_max", "Bandpass high (Hz)", "float", 6000.0, dict(decimals=1)),
    ("common_ref", "Common reference", "choice", "global", dict(choices=["global", "local", "none"])),
    ("threshold", "HS threshold", "float", 8.5, dict(decimals=2)),
    ("neighbor_radius", "neighbor_radius (µm)", "float", 15.0, dict(decimals=1)),
    ("inner_radius", "inner_radius (µm)", "float", 5.0, dict(decimals=1)),
    ("peak_jitter", "peak_jitter (ms)", "float", 0.15, dict(decimals=3)),
    ("left_cutout_time", "left_cutout_time (ms)", "float", 0.6, dict(decimals=2)),
    ("right_cutout_time", "right_cutout_time (ms)", "float", 1.2, dict(decimals=2)),
    ("extra", "Extra HS params (JSON)", "str", "{}", dict(tip='e.g. {"clustering_bandwidth": 4.0}')),
    ("test_end_s", "Test: sort only first N s (0 = all)", "float", 0.0, dict(decimals=0, step=60)),
]
LOC_SPEC = [
    ("min_spikes", "Min spikes", "int", 500, dict(min=0)),
    ("tol", "Match tolerance (samples)", "int", 5, dict(min=0, max=100)),
    ("max_spikes_per_unit", "Spikes per unit for vote", "int", 400, dict(min=10)),
    ("channel_base", "HS channel numbering", "choice", "1-based",
     dict(choices=["1-based", "0-based"], tip="Notebook assumed 1-based ('ch − 1'). Check the activity map log.")),
]


class SortingTab(TabBase):
    title = "3 · Sorting"
    key = "sorting"

    def __init__(self, mw):
        super().__init__(mw)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        g = QGroupBox("HerdingSpikes (full array)")
        v = QVBoxLayout(g)
        self.form = ParamForm(HS_SPEC)
        v.addWidget(self.form)
        v.addWidget(button("Run HerdingSpikes", self.run_hs, primary=True))
        left.addWidget(g)
        g2 = QGroupBox("Existing sortings")
        v2 = QVBoxLayout(g2)
        self.runs = QComboBox()
        v2.addWidget(self.runs)
        v2.addWidget(hbox(button("Refresh", self.refresh_runs), button("Load selected", self.load_selected, primary=True),
                          button("Other folder…", self.load_other)))
        left.addWidget(g2)
        g3 = QGroupBox("Unit locations / activity")
        v3 = QVBoxLayout(g3)
        self.loc_form = ParamForm(LOC_SPEC)
        v3.addWidget(self.loc_form)
        v3.addWidget(hbox(button("Electrode activity map", self.activity), button("Localise units", self.localise,
                                                                                    primary=True)))
        left.addWidget(g3)
        left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(430)
        right = QSplitter(Qt.Vertical)
        self.plot = MplPanel(); self.loc_tab = DataTable()
        right.addWidget(self.plot); right.addWidget(self.loc_tab); right.setSizes([520, 200])
        lay.addWidget(lw); lay.addWidget(right, 1)

    def refresh_runs(self):
        self.runs.clear()
        if self.session.out_folder:
            for r in hs.list_runs(self.session.out_folder):
                self.runs.addItem(os.path.basename(r), r)
        if self.session.sorting_folder:
            i = self.runs.findData(self.session.sorting_folder)
            if i >= 0:
                self.runs.setCurrentIndex(i)

    def run_hs(self):
        if not self.mw.require(rec=True):
            return
        p = self.form.get()
        try:
            extra = json.loads(p.pop("extra") or "{}")
        except Exception as e:
            QMessageBox.warning(self, "Extra params", f"Invalid JSON: {e}"); return
        tend = p.pop("test_end_s")
        s = self.session
        try:
            import herdingspikes  # noqa
        except ImportError:
            QMessageBox.warning(self, "HerdingSpikes missing",
                                "The 'herdingspikes' package isn't installed in this environment.\n"
                                "Install it with:  pip install herdingspikes"); return

        def job(log, progress, cancel):
            return hs.run_herdingspikes(s.rec, s.out_folder, p, extra, (0, tend) if tend > 0 else None, log)

        def done(res):
            s.sorting, s.sorting_folder = res
            self.mw.save_session(quiet=True); self.refresh_runs(); self.mw.notify_session_changed()
        self.run(job, on_done=done, label="HerdingSpikes")

    def _load(self, folder):
        s = self.session

        def done(srt):
            s.sorting, s.sorting_folder = srt, folder
            self.log(f"Sorting loaded: {folder} ({len(srt.unit_ids)} units)")
            self.mw.save_session(quiet=True); self.mw.notify_session_changed()
        self.run(lambda log, progress, cancel: hs.read_sorting(folder), on_done=done, label="Load sorting")

    def load_selected(self):
        f = self.runs.currentData()
        if f:
            self._load(f)

    def load_other(self):
        f = QFileDialog.getExistingDirectory(self, "Sorter output folder (run_sorter folder)", self.session.out_folder)
        if f:
            self._load(f)

    def activity(self):
        if not self.mw.require(rec=True):
            return
        s = self.session
        if not s.sorting_folder:
            QMessageBox.information(self, "No HS run", "Load a HerdingSpikes run first."); return

        def job(log, progress, cancel):
            xy, rate, base = hs.activity_map(s.rec, s.sorting_folder, "auto", log)
            return hs.activity_figure(xy, rate, f"Electrode activity (HS detections, {base})")
        self.run(job, on_done=self.plot.show_figure, label="Activity map")

    def localise(self):
        if not self.mw.require(rec=True, sorting=True):
            return
        s, p = self.session, self.loc_form.get()

        def job(log, progress, cancel):
            df = hs.localize_units(s.rec, s.sorting, s.sorting_folder, s.out_folder, min_spikes=p["min_spikes"],
                                   tol=p["tol"], max_spikes_per_unit=p["max_spikes_per_unit"],
                                   channel_base=p["channel_base"], log=log, progress=progress, cancel=cancel)
            return df, hs.locations_figure(s.rec, df)

        def done(res):
            df, fig = res
            self.loc_tab.set_df(df); self.plot.show_figure(fig)
            self.mw.notify_session_changed()
        self.run(job, on_done=done, label="Localise units")

    def on_session_changed(self):
        self.refresh_runs()
        p = hs.unit_xy_csv(self.session.out_folder) if self.session.out_folder else ""
        if p and os.path.exists(p) and self.loc_tab.model_.rowCount() == 0:
            self.loc_tab.set_df(pd.read_csv(p))

    def get_settings(self):
        return dict(form=self.form.get(), loc=self.loc_form.get())

    def set_settings(self, d):
        d = d or {}
        self.form.set(d.get("form", {})); self.loc_form.set(d.get("loc", {}))
