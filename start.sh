#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
REQUESTED_PORT="${PORT:-}"
PORT="${PORT:-8765}"
DASHBOARD_DEVICE="${DASHBOARD_DEVICE:-cpu}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

set_urls() {
  OPEN_HOST="$HOST"
  if [[ "$OPEN_HOST" == "0.0.0.0" || "$OPEN_HOST" == "::" ]]; then
    OPEN_HOST="127.0.0.1"
  fi
  URL="http://${OPEN_HOST}:${PORT}"
  HEALTH_URL="${URL}/api/health"
}
set_urls

LOG_FILE="$ROOT/logs/dashboard.log"
PID_FILE="$ROOT/logs/dashboard.pid"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs"

is_up() {
  python3 - "$HEALTH_URL" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=0.5) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
}

has_command_console() {
  python3 - "$URL/api/command/presets" <<'PY' >/dev/null 2>&1
import json
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=0.5) as response:
    if response.status != 200:
        raise SystemExit(1)
    payload = json.loads(response.read().decode())
    raise SystemExit(0 if payload.get("presets") else 1)
PY
}

open_browser() {
  if [[ "$OPEN_BROWSER" == "0" ]]; then
    return 0
  fi

  if command -v xdg-open >/dev/null 2>&1; then
    nohup xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v gio >/dev/null 2>&1; then
    nohup gio open "$URL" >/dev/null 2>&1 || true
  elif command -v kde-open >/dev/null 2>&1; then
    nohup kde-open "$URL" >/dev/null 2>&1 || true
  elif command -v gnome-open >/dev/null 2>&1; then
    nohup gnome-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    nohup open "$URL" >/dev/null 2>&1 || true
  else
    echo "[start] no browser opener found; open $URL manually"
  fi
}

wait_for_dashboard() {
  for _ in $(seq 1 120); do
    if is_up; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

if is_up; then
  if has_command_console; then
    echo "[start] dashboard already running at $URL"
    open_browser
    exit 0
  fi
  if [[ -n "$REQUESTED_PORT" ]]; then
    echo "[start] a dashboard is already running at $URL, but it does not expose the Command Console API" >&2
    echo "[start] stop the existing process or rerun with a different PORT, for example: PORT=8876 ./start.sh" >&2
    exit 1
  fi

  echo "[start] port $PORT is occupied by an older dashboard; finding a free localhost port"
  FOUND_PORT=0
  for candidate in $(seq $((PORT + 1)) $((PORT + 20))); do
    PORT="$candidate"
    set_urls
    if ! is_up; then
      FOUND_PORT=1
      break
    fi
  done
  if [[ "$FOUND_PORT" == "0" ]]; then
    echo "[start] could not find a free dashboard port from 8766 through 8785" >&2
    exit 1
  fi
  echo "[start] using $URL instead"
fi

echo "[start] launching Phi-Plasma dashboard at $URL"
echo "[start] logs: $LOG_FILE"
python3 -m phi_plasma.dashboard --host "$HOST" --port "$PORT" --device "$DASHBOARD_DEVICE" >"$LOG_FILE" 2>&1 &
DASH_PID=$!
echo "$DASH_PID" >"$PID_FILE"

cleanup() {
  if kill -0 "$DASH_PID" >/dev/null 2>&1; then
    echo
    echo "[start] stopping dashboard pid $DASH_PID"
    kill "$DASH_PID" >/dev/null 2>&1 || true
    wait "$DASH_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM EXIT

if ! wait_for_dashboard; then
  echo "[start] dashboard failed to become healthy" >&2
  echo "[start] last dashboard log lines:" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "[start] dashboard ready"
echo "[start] use the Command Console panel for training, synthetic data generation, packing, convergence checks, tests, and sampling"
open_browser

wait "$DASH_PID"
