#!/usr/bin/env bash
# Start everything AdForge needs, then open the UI.
#
# Three services, not one:
#   ollama    :11434  writes the copy          (systemd --user unit)
#   ComfyUI   :8189   renders images and video (~/adstudio, its own venv)
#   adforge   :8770   the UI and scheduler
#
# Every step is idempotent and checked by its LISTENING PORT, never by pgrep.
# pgrep -f matches any process whose command line merely contains the pattern -
# including the shell running this script - which during development killed the
# very server it was meant to manage. A port is owned by exactly one process.
set -uo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$APP_DIR/venv/bin/python"
COMFY_DIR="$HOME/adstudio"
COMFY_PY="$COMFY_DIR/venv/bin/python"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/adforge"
LOG="$LOG_DIR/services.log"
mkdir -p "$LOG_DIR"

read -r HOST PORT < <("$PY" -c "
from adforge.config import settings
print(settings.host, settings.port)" 2>/dev/null) || { HOST=127.0.0.1; PORT=8770; }
[ -n "${HOST:-}" ] || HOST=127.0.0.1
[ -n "${PORT:-}" ] || PORT=8770
URL="http://${HOST}:${PORT}/"

say()  { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
up()   { curl -fsS -o /dev/null --max-time 2 "$1" 2>/dev/null; }
note() { notify-send -a AdForge "$1" "${2:-}" 2>/dev/null || true; }

warn_only=()

wait_for() {  # wait_for <url> <seconds> <label>
    local url="$1" secs="$2" label="$3" i
    for ((i = 0; i < secs * 2; i++)); do
        up "$url" && { say "$label is up"; return 0; }
        sleep 0.5
    done
    say "$label did NOT come up within ${secs}s"
    return 1
}

{ echo; echo "=== start-all $(date -Is) ==="; } >> "$LOG"
note "Starting services…" "ollama, ComfyUI, AdForge"

# ---- 1. ollama -------------------------------------------------------------
if up "http://127.0.0.1:11434/api/tags"; then
    say "ollama already running"
else
    say "starting ollama"
    systemctl --user start ollama 2>>"$LOG" || say "systemctl --user start ollama failed"
    wait_for "http://127.0.0.1:11434/api/tags" 30 "ollama" \
        || warn_only+=("ollama (:11434) - no text generation without it")
fi

# ---- 2. ComfyUI ------------------------------------------------------------
# Slowest of the three: it imports torch and enumerates models. 3 minutes is
# generous rather than optimistic, because declaring failure early and then
# having it appear 20s later is worse than waiting.
if up "http://127.0.0.1:8189/system_stats"; then
    say "ComfyUI already running"
elif [ -x "$COMFY_PY" ]; then
    say "starting ComfyUI from $COMFY_DIR"
    ( cd "$COMFY_DIR" && setsid nohup "$COMFY_PY" main.py \
        --listen 127.0.0.1 --port 8189 >> "$LOG_DIR/comfyui.log" 2>&1 < /dev/null & )
    wait_for "http://127.0.0.1:8189/system_stats" 180 "ComfyUI" \
        || warn_only+=("ComfyUI (:8189) - no images or video without it")
else
    say "no ComfyUI venv at $COMFY_PY"
    warn_only+=("ComfyUI not found at $COMFY_DIR - no images or video")
fi

# ---- 3. AdForge ------------------------------------------------------------
# This one is fatal: without it there is nothing to open.
if up "$URL"; then
    say "adforge already running"
else
    [ -x "$PY" ] || {
        note "AdForge failed to start" "No virtualenv at $PY"
        command -v zenity >/dev/null && zenity --error --no-wrap \
            --title="AdForge" --text="No virtualenv at $PY" 2>/dev/null
        exit 1
    }
    say "starting adforge"
    ( cd "$APP_DIR" && setsid nohup "$PY" run.py >> "$LOG_DIR/adforge.log" 2>&1 < /dev/null & )
    wait_for "$URL" 60 "adforge" || {
        note "AdForge failed to start" "Timed out. See $LOG_DIR/adforge.log"
        command -v zenity >/dev/null && zenity --error --no-wrap --title="AdForge" \
            --text="Timed out waiting for $URL\n\n$(tail -n 12 "$LOG_DIR/adforge.log" 2>/dev/null | sed 's/&/\&amp;/g; s/</\&lt;/g')" 2>/dev/null
        exit 1
    }
fi

# A partial start is reported rather than passed off as success: the UI opens
# fine with no ollama and no ComfyUI, and every generation then fails one at a
# time with nothing on screen connecting them to a service that never started.
if [ ${#warn_only[@]} -gt 0 ]; then
    msg=$(printf '%s\n' "${warn_only[@]}")
    say "started WITH PROBLEMS: $msg"
    note "AdForge started, but not everything" "$msg"
else
    note "All services running" "ollama · ComfyUI · AdForge"
fi

xdg-open "$URL" >/dev/null 2>&1 &
exit 0
