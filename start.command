#!/usr/bin/env bash
# HDMEA-Analysis — start from source on macOS (double-click in Finder).
cd "$(dirname "$0")" || exit 1
if [ ! -d .venv ]; then echo "Run install.command first."; read -r -p "Press Enter"; exit 1; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m hdmea "$@"
