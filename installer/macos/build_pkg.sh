#!/usr/bin/env bash
# Build a .pkg that installs HDMEA-Analysis.app into /Applications.
#   installer/macos/build_pkg.sh <version> <output.pkg>
set -euo pipefail
VERSION="$1"; OUT="$2"
APP="dist/HDMEA-Analysis.app"
ID="io.github.jerjely.hdmea-analysis"
[ -d "$APP" ] || { echo "missing $APP"; exit 1; }
WORK="$(mktemp -d)"
mkdir -p "$WORK/root"
ditto "$APP" "$WORK/root/HDMEA-Analysis.app"
# Not relocatable: always install into /Applications, even if another copy exists elsewhere.
pkgbuild --analyze --root "$WORK/root" "$WORK/component.plist"
plutil -replace 0.BundleIsRelocatable -bool NO "$WORK/component.plist"
mkdir -p "$(dirname "$OUT")"
pkgbuild --root "$WORK/root" \
         --component-plist "$WORK/component.plist" \
         --install-location /Applications \
         --identifier "$ID" \
         --version "$VERSION" \
         "$OUT"
rm -rf "$WORK"
echo "Built $OUT"
