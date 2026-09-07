#!/usr/bin/env bash
# Restart the AdForge server. Spawned DETACHED by the UI, because the process
# it kills is the one serving the request that asked for the restart.
#
#   adforge-restart.sh <pid-to-kill> <port>
set -uo pipefail

OLD_PID="${1:?pid required}"
PORT="${2:-8770}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$APP_DIR/venv/bin/python"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/adforge"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/adforge.log"

port_pid() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u | head -1; }

{ echo; echo "=== restart requested $(date -Is), killing $OLD_PID ==="; } >> "$LOG"

# Let the HTTP response for /settings/restart finish flushing to the browser.
sleep 1

kill "$OLD_PID" 2>/dev/null

# Wait for the socket to actually free. Starting while the old process still
# holds the port gives "address already in use" and leaves nothing running -
# a restart button that stops the app is worse than no button.
for _ in $(seq 1 40); do
    [ -z "$(port_pid)" ] && break
    sleep 0.25
done

if [ -n "$(port_pid)" ]; then
    echo "restart: $OLD_PID would not release :$PORT, sending KILL" >> "$LOG"
    kill -9 "$OLD_PID" 2>/dev/null
    sleep 1
fi

cd "$APP_DIR" || exit 1
setsid nohup "$PY" run.py >> "$LOG" 2>&1 < /dev/null &
echo "restart: started, new pid $!" >> "$LOG"
