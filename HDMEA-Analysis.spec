# PyInstaller build file — used identically on Windows, macOS and Linux.
#   pyinstaller --noconfirm HDMEA-Analysis.spec
# Output: dist/HDMEA-Analysis/ (Windows/Linux) or dist/HDMEA-Analysis.app (macOS)
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

APP = "HDMEA-Analysis"
ROOT = os.path.abspath(SPECPATH)
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"',
                    open(os.path.join(ROOT, "hdmea", "_version.py")).read()).group(1)
NUM_VERSION = re.match(r"\d+(\.\d+){0,2}", VERSION).group(0) if re.match(r"\d", VERSION) else "0.0.0"

# Packages that are imported dynamically (SpikeInterface extractors/preprocessing, neo rawio, zarr codecs)
hidden = []
for pkg in ("spikeinterface.core", "spikeinterface.extractors", "spikeinterface.preprocessing",
            "spikeinterface.sorters", "neo.rawio", "neo.io", "numcodecs", "zarr", "probeinterface"):
    hidden += collect_submodules(pkg, filter=lambda n: ".tests" not in n and "test_" not in n)
hidden += ["hdmea.selftest", "openpyxl", "h5py", "skimage.measure", "skimage.draw",
           "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg"]

datas = [(os.path.join(ROOT, "hdmea", "resources"), os.path.join("hdmea", "resources"))]
for pkg in ("spikeinterface", "probeinterface", "neo"):
    datas += collect_data_files(pkg, excludes=["**/tests/**"])
for pkg in ("spikeinterface", "neo", "probeinterface", "zarr", "numcodecs", "numpy", "scipy", "pandas",
            "h5py", "scikit-image", "matplotlib", "PySide6", "tqdm", "threadpoolctl", "openpyxl"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Optional heavy / driver-dependent packages are never bundled (see README).
excludes = ["herdingspikes", "torch", "tensorflow", "jax", "IPython", "jupyter", "notebook", "ipykernel",
            "tkinter", "PyQt5", "PyQt6", "PySide2", "spikeinterface_gui", "phy", "kilosort", "mountainsort5",
            "psychopy", "nidaqmx", "pytest", "sphinx", "numba", "cupy", "dask", "distributed", "sklearn"]

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP,
    debug=False,
    strip=False,
    upx=False,
    console=False,            # windowed app; the self-test writes its result to a log file
    icon=os.path.join(ROOT, "hdmea", "resources", "icon.ico" if sys.platform == "win32" else "icon.icns"),
    target_arch=None,         # native arch of the runner (arm64 on macos-15, x86_64 on macos-15-intel)
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP}.app",
        icon=os.path.join(ROOT, "hdmea", "resources", "icon.icns"),
        bundle_identifier="io.github.jerjely.hdmea-analysis",
        version=NUM_VERSION,
        info_plist={
            "CFBundleName": APP,
            "CFBundleDisplayName": "HD-MEA Analysis",
            "CFBundleShortVersionString": NUM_VERSION,
            "CFBundleVersion": NUM_VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "© Gergely Szarka — MIT licence",
        },
    )
