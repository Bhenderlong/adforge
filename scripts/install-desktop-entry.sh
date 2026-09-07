#!/usr/bin/env bash
# Install (or refresh) the AdForge entry in the desktop applications menu.
# Per-user, no root: everything lands under ~/.local/share.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
mkdir -p "$APPS" "$ICONS"

chmod +x "$HERE/adforge-launch.sh" "$HERE/adforge-start-all.sh"
cp "$HERE/adforge.svg" "$ICONS/adforge.svg"

# Exec is written with the ABSOLUTE path of this checkout: .desktop files do
# not expand ~ or $HOME, and a relative Exec silently does nothing.
cat > "$APPS/adforge.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=AdForge
GenericName=Marketing Automation
Comment=Local marketing automation for Inferix and Vallorix
Exec=$HERE/adforge-start-all.sh
Icon=adforge
Terminal=false
Categories=Network;
Keywords=marketing;social;inferix;vallorix;adforge;campaign;
# StartupNotify MUST stay false. Nothing here ever maps a window - the visible
# result is a tab in your existing browser, owned by the browser. With it on,
# the desktop waits for a startup-notification handshake that never arrives:
# gtk-launch blocked for a full 3 minutes on an app that was serving HTTP 200
# one second in, and XFCE shows a busy cursor for its whole timeout.
StartupNotify=false
Actions=uionly;

# The default click starts ollama, ComfyUI and AdForge, because a UI with no
# model behind it fails one generation at a time with nothing on screen
# connecting those failures to a service that never started. This action is
# for when the GPU is deliberately busy with something else and you only want
# the queue and the settings.
[Desktop Action uionly]
Name=Open UI only (no ollama or ComfyUI)
Exec=$HERE/adforge-launch.sh
EOF
chmod +x "$APPS/adforge.desktop"

update-desktop-database "$APPS" 2>/dev/null || true
gtk-update-icon-cache -f -t "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" 2>/dev/null || true

# A copy on the Desktop too. XFCE files Categories=Network under "Internet",
# which is not where anyone looks for their own marketing tool.
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ]; then
    cp "$APPS/adforge.desktop" "$DESKTOP_DIR/adforge.desktop"
    chmod +x "$DESKTOP_DIR/adforge.desktop"
    gio set "$DESKTOP_DIR/adforge.desktop" metadata::trusted true 2>/dev/null || true
    echo "Desktop icon:  $DESKTOP_DIR/adforge.desktop"
fi

command -v desktop-file-validate >/dev/null && desktop-file-validate "$APPS/adforge.desktop"
echo "Installed: $APPS/adforge.desktop -> $HERE/adforge-start-all.sh (action: adforge-launch.sh)"
