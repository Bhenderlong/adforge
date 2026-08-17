#!/usr/bin/env bash
# One-click launcher for the Ubuntu applications menu.
#
# Idempotent on purpose: clicking the menu entry while AdForge is already up
# must open the browser at the running instance, NOT start a second one. Two
# processes on one SQLite file both running the publish tick is how the same
# post goes out twice.
set -uo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$APP_DIR/venv/bin/python"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/adforge"
LOG="$LOG_DIR/adforge.log"
mkdir -p "$LOG_DIR"

# Read the real values rather than hardcoding 8770 - both are editable on the
# Settings page, and a launcher that opens the wrong port looks like a crash.
read -r HOST PORT < <("$PY" -c "
from adforge.config import settings
print(settings.host, settings.port)" 2>/dev/null) || { HOST=127.0.0.1; PORT=8770; }
[ -n "${HOST:-}" ] || HOST=127.0.0.1
[ -n "${PORT:-}" ] || PORT=8770
URL="http://${HOST}:${PORT}/"

fail() {
    # A menu entry that silently does nothing is indistinguishable from a
    # broken .desktop file, so say what went wrong and where to look.
    notify-send -u critical "AdForge failed to start" "$1" 2>/dev/null
    command -v zenity >/dev/null && zenity --error --no-wrap \
        --title="AdForge failed to start" \
        --text="$1\n\nLast lines of $LOG:\n$(tail -n 12 "$LOG" 2>/dev/null | sed 's/&/\&amp;/g; s/</\&lt;/g')" 2>/dev/null
    exit 1
}

# stderr discarded: -S makes curl announce every refused connection, and this
# is polled once per 0.5s during startup. Those are expected, not errors.
alive() { curl -fsS -o /dev/null --max-time 2 "$URL" 2>/dev/null; }

if alive; then
    xdg-open "$URL" >/dev/null 2>&1 &
    exit 0
fi

[ -x "$PY" ] || fail "No virtualenv at $PY. Run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"

# setsid + nohup so the server outlives this script and the launching shell.
cd "$APP_DIR" || fail "Cannot enter $APP_DIR"
{ echo; echo "=== launch $(date -Is) ==="; } >> "$LOG"
setsid nohup "$PY" run.py >> "$LOG" 2>&1 < /dev/null &

# Generation models load lazily, but the DB migration and APScheduler start
# during boot, so allow a slow first run rather than declaring failure early.
for _ in $(seq 1 60); do
    alive && { xdg-open "$URL" >/dev/null 2>&1 & exit 0; }
    sleep 0.5
done

fail "Timed out after 30s waiting for $URL"
