"""Main window: session handling, background task runner, log dock, tabs."""
from __future__ import annotations

import os
import time

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QDockWidget, QPlainTextEdit, QProgressBar,
                               QPushButton, QLabel, QMessageBox, QFileDialog, QApplication)

from .. import __version__, APP_TITLE, resource_path
from ..core.session import Session
from .widgets import Worker


class TabBase(QWidget):
    title = "Tab"
    key = "tab"

    def __init__(self, mw):
        super().__init__()
        self.mw = mw

    @property
    def session(self) -> Session:
        return self.mw.session

    def run(self, fn, *args, on_done=None, label="", **kwargs):
        return self.mw.run_task(fn, *args, on_done=on_done, label=label or self.title, **kwargs)

    def log(self, msg):
        self.mw.log(msg)

    # settings persistence -------------------------------------------------
    def get_settings(self) -> dict:
        return {}

    def set_settings(self, d: dict):
        pass

    def on_session_changed(self):
        """Called after a session is opened/created or state changed (rec/sorting/triggers)."""
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} {__version__}")
        self.setWindowIcon(QIcon(resource_path("icon_256.png")))
        self.resize(1450, 920)
        self.session = Session()
        self.worker = None
        self._t0 = 0.0
        self.qs = QSettings("SzarkaLab", "HDMEA-Analysis")

        # log dock
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        self.log_view.setStyleSheet("font-family: Consolas, Menlo, monospace; font-size: 11px;")
        dock = QDockWidget("Log", self); dock.setWidget(self.log_view)
        dock.setObjectName("logdock")
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        # status bar
        self.status_lbl = QLabel("No session")
        self.prog = QProgressBar(); self.prog.setRange(0, 1000); self.prog.setFixedWidth(260)
        self.prog.setTextVisible(False); self.prog.hide()
        self.task_lbl = QLabel("")
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.clicked.connect(self.cancel_task)
        self.cancel_btn.hide()
        sb = self.statusBar()
        sb.addWidget(self.status_lbl, 1)
        sb.addPermanentWidget(self.task_lbl); sb.addPermanentWidget(self.prog); sb.addPermanentWidget(self.cancel_btn)

        # tabs
        from .tabs_pipeline import SessionTab, TriggersTab, SortingTab
        from .tabs_analysis import PsthTab, RingingTab, RfTab, CalibTab, CouplingTab
        from .tabs_tools import ToolsTab
        self.tabs = QTabWidget()
        self.tab_list = [SessionTab(self), TriggersTab(self), SortingTab(self), PsthTab(self), RingingTab(self),
                         RfTab(self), CalibTab(self), CouplingTab(self), ToolsTab(self)]
        for t in self.tab_list:
            self.tabs.addTab(t, t.title)
        self.setCentralWidget(self.tabs)

        # menu
        m = self.menuBar().addMenu("&Session")
        for text, slot, sc in (("New session…", self.new_session, QKeySequence.New),
                               ("Open session…", self.open_session_dialog, QKeySequence.Open),
                               ("Save session", self.save_session, QKeySequence.Save)):
            a = QAction(text, self); a.triggered.connect(slot); a.setShortcut(sc); m.addAction(a)
        self.recent_menu = m.addMenu("Open recent")
        m.addSeparator()
        a = QAction("Open output folder", self); a.triggered.connect(self.open_out_folder); m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self); a.triggered.connect(self.close); m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("About", self); a.triggered.connect(self.about); h.addAction(a)
        self._fill_recent()
        geo = self.qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.log(f"{APP_TITLE} {__version__} ready. Start in the Session tab (add BRW files) "
                 f"or Session ▸ Open session….")

    # ------------------------------------------------------------------
    def log(self, msg):
        stamp = time.strftime("%H:%M:%S")
        for line in str(msg).splitlines() or [""]:
            self.log_view.appendPlainText(f"[{stamp}] {line}")
        sb = self.log_view.verticalScrollBar(); sb.setValue(sb.maximum())

    def refresh_status(self):
        s = self.session
        parts = [os.path.basename(s.out_folder) if s.out_folder else "unsaved session"]
        if s.table is not None:
            parts.append(f"{len(s.table)} files")
        parts.append("rec ✓" if s.rec is not None else "rec –")
        parts.append(f"triggers ✓" if s.triggers else "triggers –")
        parts.append(f"sorting ✓ ({len(s.sorting.unit_ids)} units)" if s.sorting is not None else "sorting –")
        self.status_lbl.setText("  |  ".join(parts))

    def notify_session_changed(self):
        for t in self.tab_list:
            try:
                t.on_session_changed()
            except Exception as e:
                self.log(f"⚠ {t.title}: {e}")
        self.refresh_status()

    # ------------------------------------------------------------------
    def run_task(self, fn, *args, on_done=None, label="", **kwargs):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Another task is still running. Wait or press Cancel.")
            return None
        w = Worker(fn, *args, **kwargs)
        w.log.connect(self.log)
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(lambda res: self._task_done(res, on_done, label))
        w.failed.connect(lambda msg: self._task_failed(msg, label))
        self.worker = w
        self.prog.setRange(0, 0); self.prog.show(); self.cancel_btn.show()   # busy until first progress
        self.task_lbl.setText(f"{label}…")
        self._t0 = time.time()
        self.log(f"▶ {label}")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        w.start()
        return w

    def _on_progress(self, v):
        if self.prog.maximum() == 0:
            self.prog.setRange(0, 1000)
        self.prog.setValue(v)

    def _end_task(self):
        QApplication.restoreOverrideCursor()
        self.prog.hide(); self.cancel_btn.hide(); self.task_lbl.setText("")

    def _release_worker(self):
        # keep a reference until the thread has really stopped (avoids "QThread destroyed while running")
        if self.worker is not None:
            old = self.worker
            self._finished_workers = [w for w in getattr(self, "_finished_workers", []) if w.isRunning()] + [old]
            self.worker = None

    def _task_done(self, res, on_done, label):
        self._release_worker()
        self._end_task()
        self.log(f"✔ {label} finished in {time.time() - self._t0:.1f} s")
        if on_done is not None:
            try:
                on_done(res)
            except Exception as e:
                import traceback
                self.log(f"⚠ error while showing results: {e}\n{traceback.format_exc()}")
        self.refresh_status()

    def _task_failed(self, msg, label):
        self._release_worker()
        self._end_task()
        self.log(f"✖ {label} failed: {msg}")
        first = msg.splitlines()[0] if msg else "Unknown error"
        if first != "Cancelled.":
            QMessageBox.warning(self, f"{label} failed", first + "\n\n(full traceback in the Log panel)")

    def cancel_task(self):
        if self.worker is not None:
            self.worker.cancel()
            self.log("Cancel requested – stops at the next checkpoint (SpikeInterface writes/sorting can't be interrupted).")

    # ------------------------------------------------------------------
    def require(self, rec=False, sorting=False, triggers=False, out=True):
        s = self.session
        missing = []
        if out and not s.out_folder:
            missing.append("an output folder (Session tab)")
        if rec and s.rec is None:
            missing.append("the concatenated recording (Session tab ▸ Build / load)")
        if triggers and not s.triggers:
            missing.append("stimulus triggers (Triggers tab)")
        if sorting and s.sorting is None:
            missing.append("a spike sorting (Sorting tab)")
        if missing:
            QMessageBox.information(self, "Missing input", "This step needs:\n• " + "\n• ".join(missing))
            return False
        return True

    def collect_settings(self):
        for t in self.tab_list:
            try:
                self.session.settings[t.key] = t.get_settings()
            except Exception:
                pass

    def save_session(self, quiet=False):
        s = self.session
        if not s.out_folder:
            if not quiet:
                QMessageBox.information(self, "No output folder", "Set an output folder in the Session tab first.")
            return False
        self.collect_settings()
        s.save()
        self._add_recent(s.session_path)
        if not quiet:
            self.log(f"Session saved: {s.session_path}")
        self.refresh_status()
        return True

    def new_session(self):
        if self.session.out_folder and QMessageBox.question(self, "New session",
                                                            "Save the current session first?") == QMessageBox.Yes:
            self.save_session()
        self.session = Session()
        for t in self.tab_list:
            t.set_settings({})
        self.notify_session_changed()
        self.tabs.setCurrentIndex(0)
        self.log("New empty session")

    def open_session_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open session", self.qs.value("last_dir", ""),
                                           "HD-MEA session (hdmea_session.json)")
        if p:
            self.open_session(p)

    def open_session(self, path):
        try:
            s = Session.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Open failed", str(e)); return
        self.session = s
        self.qs.setValue("last_dir", os.path.dirname(path))
        for t in self.tab_list:
            t.set_settings(s.settings.get(t.key, {}))
        self._add_recent(s.session_path)
        self.log(f"Opened session: {s.session_path}")
        self.notify_session_changed()
        self.tab_list[0].autoload()

    def open_out_folder(self):
        if self.session.out_folder and os.path.isdir(self.session.out_folder):
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.session.out_folder))

    def _add_recent(self, p):
        rec = [x for x in (self.qs.value("recent", []) or []) if x != p]
        rec.insert(0, p)
        self.qs.setValue("recent", rec[:10])
        self._fill_recent()

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in (self.qs.value("recent", []) or []):
            a = QAction(p, self); a.triggered.connect(lambda _=False, x=p: self.open_session(x))
            self.recent_menu.addAction(a)

    def about(self):
        QMessageBox.about(self, "About", f"<b>{APP_TITLE}</b><br>Version {__version__}<br><br>"
                          "BioCam HD-MEA retina pipeline built from the lab notebooks "
                          "(Biocam_sorting, Untitled3, SO_Final, Untitled2, Untitled4).<br><br>"
                          "Session → Triggers → Sorting → Light responses / Ringing / RF mapping / "
                          "RF→MEA / Coupling, plus stimulus tools.<br><br>"
                          "© Gergely Szarka — MIT licence<br>"
                          "<a href='https://github.com/jerjely/HDMEA-Analysis'>github.com/jerjely/HDMEA-Analysis</a>")

    def closeEvent(self, ev):
        if self.worker is not None and self.worker.isRunning():
            if QMessageBox.question(self, "Quit", "A task is running. Quit anyway?") != QMessageBox.Yes:
                ev.ignore(); return
            self.worker.cancel()
        if self.session.out_folder:
            try:
                self.save_session(quiet=True)
            except Exception:
                pass
        self.qs.setValue("geometry", self.saveGeometry())
        super().closeEvent(ev)
