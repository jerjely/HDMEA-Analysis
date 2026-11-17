"""HDMEA-Analysis — BioCam HD-MEA retina analysis."""
from ._version import __version__

APP_NAME = "HDMEA-Analysis"
APP_TITLE = "HD-MEA Analysis"


def resource_path(name: str) -> str:
    """Path to a bundled resource (works from source and inside PyInstaller builds)."""
    import os
    import sys
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = os.path.join(base, "hdmea", "resources", name)
        if os.path.exists(p):
            return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", name)
