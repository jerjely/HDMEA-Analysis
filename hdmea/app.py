"""Entry point: python -m hdmea   (or the HDMEA-Analysis executable).

    HDMEA-Analysis [session.json]          start the GUI (optionally open a session)
    HDMEA-Analysis --selftest [--log F]    run the built-in synthetic self-test and exit 0/1
    HDMEA-Analysis --version               print the version
"""
import os
import sys


def _ensure_streams():
    """Windowed (no-console) builds have sys.stdout/stderr = None; SpikeInterface/tqdm
    progress bars would crash on that. Send them to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    d = os.path.join(os.path.expanduser("~"), ".hdmea_analysis")
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, "app.log"), "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def _set_windows_app_id():
    if sys.platform == "win32":
        try:  # own taskbar group + icon instead of python.exe's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("jerjely.HDMEA-Analysis")
        except Exception:
            pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _ensure_streams()
    from . import __version__
    if "--version" in argv:
        print(__version__)
        return 0
    if "--selftest" in argv:
        from .selftest import run_selftest
        log = argv[argv.index("--log") + 1] if "--log" in argv and argv.index("--log") + 1 < len(argv) else None
        return run_selftest(log_path=log)

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    import matplotlib
    matplotlib.use("QtAgg")
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from . import APP_TITLE, resource_path
    from .gui.main_window import MainWindow

    _set_windows_app_id()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(resource_path("icon_256.png")))
    app.setStyle("Fusion")
    w = MainWindow()
    files = [a for a in argv if not a.startswith("-") and os.path.exists(a)]
    if files:
        w.open_session(files[0])
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
