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
Keywords=marketing;social;inferix;vallorix;adforge;
StartupNotify=true
SingleMainWindow=true
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

command -v desktop-file-validate >/dev/null && desktop-file-validate "$APPS/adforge.desktop"
echo "Installed: $APPS/adforge.desktop -> $HERE/adforge-start-all.sh (action: adforge-launch.sh)"
