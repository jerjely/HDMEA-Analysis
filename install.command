#!/usr/bin/env bash
# HDMEA-Analysis — install from source on macOS (double-click in Finder).
# Creates a virtual environment in .venv next to this file and installs the requirements.
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ not found. Install it from https://www.python.org/downloads/macos/ and run this again."
  read -r -p "Press Enter to close"; exit 1
fi
echo "Using $($PY --version)"
"$PY" -m venv .venv || { read -r -p "venv failed. Press Enter"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || { read -r -p "Install failed. Press Enter"; exit 1; }
echo
echo "Optional: HerdingSpikes for 'Run HerdingSpikes' (needs Xcode command line tools: xcode-select --install)"
python -m pip install herdingspikes || echo "HerdingSpikes not installed - the app works without it."
echo
echo "Installed. Start the app with start.command"
read -r -p "Press Enter to close"
