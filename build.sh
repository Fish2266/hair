#!/bin/zsh
# Builds Hair Game.app next to this script.
set -e
cd "$(dirname "$0")"
APP="Hair Game.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
swiftc -O -enforce-exclusivity=unchecked -swift-version 5 main.swift -o "$APP/Contents/MacOS/HairGame"
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>HairGame</string>
  <key>CFBundleIdentifier</key><string>com.connor.hairgame</string>
  <key>CFBundleName</key><string>Hair Game</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>Hair Game uses your camera to track your face and put hair on it.</string>
</dict></plist>
EOF
# Stable designated requirement so macOS remembers camera permission across rebuilds
codesign --force --sign - --identifier com.connor.hairgame -r='designated => identifier "com.connor.hairgame"' "$APP"
echo "Built $APP"
