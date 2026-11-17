"""Reusable Qt widgets: matplotlib panel, parameter form, background worker, pandas table."""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QLineEdit, QComboBox, QPushButton, QFileDialog, QLabel, QTableView,
                               QSizePolicy, QHeaderView, QAbstractItemView, QScrollArea)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from ..core.common import Cancelled


# ----------------------------------------------------------------------
class MplPanel(QWidget):
    """Shows a matplotlib Figure produced by the core (Agg) inside Qt."""

    def __init__(self, parent=None, placeholder="Results will appear here"):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        self._toolbar = None
        self._ph = QLabel(placeholder)
        self._ph.setAlignment(Qt.AlignCenter)
        self._ph.setStyleSheet("color: gray;")
        self._lay.addWidget(self._ph)
        self.figure = None

    def show_figure(self, fig):
        if self._canvas is not None:
            for wdg in (self._toolbar, self._canvas):   # detach now so the old plot never shows through
                self._lay.removeWidget(wdg); wdg.setParent(None); wdg.deleteLater()
        self._ph.hide()
        try:   # re-flow the layout whenever the panel is resized
            fig.set_layout_engine("constrained")
        except Exception:
            pass
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._lay.addWidget(self._toolbar)
        self._lay.addWidget(self._canvas)
        self._canvas.draw_idle()
        self.figure = fig


# ----------------------------------------------------------------------
class ParamForm(QWidget):
    """spec: list of (key, label, kind, default, extra) where kind in
    int|float|bool|str|choice|file|savefile|dir ; extra = dict(min,max,step,decimals,choices,filter,tip)."""

    def __init__(self, spec, parent=None, scroll=False):
        super().__init__(parent)
        self.widgets = {}
        self.kinds = {}
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        for item in spec:
            key, label, kind, default = item[:4]
            ex = item[4] if len(item) > 4 else {}
            w = self._make(kind, default, ex)
            if ex.get("tip"):
                w.setToolTip(ex["tip"])
            self.widgets[key] = w
            self.kinds[key] = kind
            lab = QLabel(label)
            if ex.get("tip"):
                lab.setToolTip(ex["tip"])
            form.addRow(lab, w)
        if scroll:
            inner = QWidget(); inner.setLayout(form)
            sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(inner)
            lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(sa)
        else:
            self.setLayout(form)

    def _make(self, kind, default, ex):
        if kind == "int":
            w = QSpinBox(); w.setRange(ex.get("min", -10 ** 9), ex.get("max", 10 ** 9)); w.setValue(int(default))
            w.setSingleStep(ex.get("step", 1))
        elif kind == "float":
            w = QDoubleSpinBox(); w.setDecimals(ex.get("decimals", 4))
            w.setRange(ex.get("min", -1e12), ex.get("max", 1e12)); w.setSingleStep(ex.get("step", 0.01))
            w.setValue(float(default))
        elif kind == "bool":
            w = QCheckBox(); w.setChecked(bool(default))
        elif kind == "choice":
            w = QComboBox(); w.setEditable(ex.get("editable", False))
            w.addItems([str(c) for c in ex.get("choices", [])])
            i = w.findText(str(default))
            if i >= 0:
                w.setCurrentIndex(i)
            elif ex.get("editable", False):
                w.setEditText(str(default))
        elif kind in ("file", "savefile", "dir"):
            w = PathEdit(kind, str(default), ex.get("filter", "All files (*)"))
        else:
            w = QLineEdit(str(default))
        return w

    def get(self):
        out = {}
        for k, w in self.widgets.items():
            kind = self.kinds[k]
            if kind in ("int", "float"):
                out[k] = w.value()
            elif kind == "bool":
                out[k] = w.isChecked()
            elif kind == "choice":
                out[k] = w.currentText()
            elif kind in ("file", "savefile", "dir"):
                out[k] = w.text()
            else:
                out[k] = w.text()
        return out

    def set(self, values: dict):
        for k, v in (values or {}).items():
            if k not in self.widgets:
                continue
            w, kind = self.widgets[k], self.kinds[k]
            try:
                if kind == "int":
                    w.setValue(int(v))
                elif kind == "float":
                    w.setValue(float(v))
                elif kind == "bool":
                    w.setChecked(bool(v))
                elif kind == "choice":
                    i = w.findText(str(v))
                    if i >= 0:
                        w.setCurrentIndex(i)
                    elif w.isEditable():
                        w.setEditText(str(v))
                else:
                    w.setText(str(v))
            except Exception:
                pass

    def set_choices(self, key, choices, keep=True):
        w = self.widgets[key]
        cur = w.currentText()
        w.blockSignals(True)
        w.clear(); w.addItems([str(c) for c in choices])
        if keep and cur:
            i = w.findText(cur)
            if i >= 0:
                w.setCurrentIndex(i)
            elif w.isEditable():
                w.setEditText(cur)
        w.blockSignals(False)


class PathEdit(QWidget):
    def __init__(self, kind, text="", filt="All files (*)"):
        super().__init__()
        self.kind, self.filt = kind, filt
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(text)
        b = QPushButton("…"); b.setFixedWidth(28); b.clicked.connect(self._browse)
        lay.addWidget(self.edit); lay.addWidget(b)

    def _browse(self):
        if self.kind == "dir":
            p = QFileDialog.getExistingDirectory(self, "Choose folder", self.edit.text())
        elif self.kind == "savefile":
            p, _ = QFileDialog.getSaveFileName(self, "Save as", self.edit.text(), self.filt)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file", self.edit.text(), self.filt)
        if p:
            self.edit.setText(p)

    def text(self):
        return self.edit.text().strip()

    def setText(self, t):
        self.edit.setText(t)


# ----------------------------------------------------------------------
class Worker(QThread):
    log = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.fn(*self.args, log=self.log.emit,
                          progress=lambda f: self.progress.emit(int(max(0, min(1, f)) * 1000)),
                          cancel=lambda: self._cancel, **self.kwargs)
            self.finished_ok.emit(res)
        except Cancelled:
            self.failed.emit("Cancelled.")
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ----------------------------------------------------------------------
class PandasModel(QAbstractTableModel):
    def __init__(self, df=None):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()

    def set_df(self, df):
        self.beginResetModel(); self.df = df.reset_index(drop=True) if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return self.df.shape[1]

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        v = self.df.iat[idx.row(), idx.column()]
        if role == Qt.DisplayRole:
            if isinstance(v, (float, np.floating)):
                return "" if not np.isfinite(v) else f"{v:.4g}"
            return str(v)
        if role == Qt.UserRole:
            return v
        return None

    def headerData(self, s, o, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self.df.columns[s]) if o == Qt.Horizontal else str(s + 1)

    def sort(self, col, order=Qt.AscendingOrder):
        if self.df.empty:
            return
        self.beginResetModel()
        self.df = self.df.sort_values(self.df.columns[col], ascending=order == Qt.AscendingOrder,
                                      na_position="last").reset_index(drop=True)
        self.endResetModel()


class DataTable(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_ = PandasModel()
        self.setModel(self.model_)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setDefaultSectionSize(20)

    def set_df(self, df):
        self.model_.set_df(df)
        self.resizeColumnsToContents()

    def selected_row(self):
        idx = self.selectionModel().selectedRows()
        if not idx:
            return None
        return self.model_.df.iloc[idx[0].row()]


def hbox(*widgets, stretch_last=False):
    w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if x == "stretch":
            lay.addStretch(1)
        else:
            lay.addWidget(x)
    return w


def button(text, slot, tip=None, primary=False):
    b = QPushButton(text)
    b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    if primary:
        b.setStyleSheet("font-weight: bold;")
    return b
