#!/usr/bin/env bash
# Build mac/build/Meetre.app (release, ad-hoc signed).
#   ./build.sh            release build
#   ./build.sh --debug    debug build
set -euo pipefail
cd "$(dirname "$0")"
MAC="$(pwd)"
ROOT="$(cd .. && pwd)"
CONFIG=release
[[ "${1:-}" == "--debug" ]] && CONFIG=debug

swift build -c "$CONFIG"
BIN="$(swift build -c "$CONFIG" --show-bin-path)/Meetre"

APP="$MAC/build/Meetre.app"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Meetre"

# App icon (cached; delete build/AppIcon.icns to re-render).
ICNS="$MAC/build/AppIcon.icns"
if [[ ! -f "$ICNS" ]]; then
  TMP="$(mktemp -d)"
  swift "$MAC/Scripts/make_icon.swift" "$TMP/icon.png"
  SET="$TMP/AppIcon.iconset"; mkdir -p "$SET"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$TMP/icon.png" --out "$SET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) "$TMP/icon.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$SET" -o "$ICNS"
  rm -rf "$TMP"
fi
cp "$ICNS" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>meetre</string>
  <key>CFBundleDisplayName</key><string>meetre</string>
  <key>CFBundleIdentifier</key><string>net.cubedpixels.meetre.app</string>
  <key>CFBundleExecutable</key><string>Meetre</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>26.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>meetre records your microphone to transcribe meetings on this Mac.</string>
  <key>NSAppleEventsUsageDescription</key><string>meetre saves meeting summaries to Apple Notes.</string>
  <key>NSScreenCaptureUsageDescription</key><string>meetre captures system audio (other participants) via ScreenCaptureKit.</string>
  <key>MeetreRepoRoot</key><string>${ROOT}</string>
</dict>
</plist>
PLIST

# A stable local identity (not ad-hoc): macOS privacy permissions are tied to
# the signature, and an ad-hoc one changes with every build (see the script).
IDENTITY="$("$MAC/Scripts/signing-identity.sh" 2>/dev/null || echo -)"
codesign --force --sign "$IDENTITY" --timestamp=none "$APP"
# This is a build product, not the installed app: keep it out of Launchpad /
# Spotlight's app list so only ~/Applications/Meetre.app shows up there.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
  -u "$APP" >/dev/null 2>&1 || true
touch "$(dirname "$APP")/.metadata_never_index"
echo "✓ Built $APP (v$VERSION, $CONFIG)"
