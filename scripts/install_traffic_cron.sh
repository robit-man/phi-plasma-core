#!/usr/bin/env bash
# One-shot installer: registers a launchd job to run the traffic snapshot
# daily at 00:30 UTC (which on the US east coast is 8:30 PM the previous
# day, on US west coast is 5:30 PM, in Europe is 1:30 AM).
#
# Why launchd and not crontab? On macOS, launchd is the canonical way to
# schedule periodic jobs. It handles laptop sleep correctly — if the
# machine was asleep at trigger time, launchd runs the job on wake.
# Plain crontab silently misses the slot.
#
# To remove later: launchctl unload ~/Library/LaunchAgents/com.prime007.github-traffic-snapshot.plist

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$HERE/scripts/snapshot_traffic.sh"
PLIST="$HOME/Library/LaunchAgents/com.prime007.github-traffic-snapshot.plist"
LOG_OUT="$HOME/.github_traffic_logs/cron.log"
LOG_ERR="$HOME/.github_traffic_logs/cron.err"

mkdir -p "$(dirname "$LOG_OUT")"

# Ensure the script is executable
chmod +x "$SCRIPT"

# Write the launchd plist
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.prime007.github-traffic-snapshot</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$SCRIPT</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>0</integer>
    <key>Minute</key>
    <integer>30</integer>
  </dict>

  <key>RunAtLoad</key>
  <false/>

  <key>StandardOutPath</key>
  <string>$LOG_OUT</string>

  <key>StandardErrorPath</key>
  <string>$LOG_ERR</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>HOME</key>
    <string>$HOME</string>
  </dict>
</dict>
</plist>
EOF

# Load it
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "Installed: $PLIST"
echo "Logs:      $LOG_OUT  /  $LOG_ERR"
echo ""
echo "Scheduled at 00:30 UTC daily."
echo "To run manually right now:  bash $SCRIPT"
echo "To uninstall:               launchctl unload $PLIST && rm $PLIST"
echo ""
echo "Running once now to verify it works..."
bash "$SCRIPT"
