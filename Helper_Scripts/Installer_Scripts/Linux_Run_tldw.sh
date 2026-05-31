#!/usr/bin/env bash
# Compatibility launcher for installs created by Linux_Install_Update.sh.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install_dir="${TLDW_INSTALL_DIR:-$script_dir/tldw}"
launcher="$install_dir/quick-launch.sh"

if [ ! -f "$launcher" ]; then
    echo "tldw_server launcher not found at: $launcher" >&2
    echo "Run Linux_Install_Update.sh first, or set TLDW_INSTALL_DIR to a checkout that contains quick-launch.sh." >&2
    exit 1
fi

export TLDW_VENV_DIR="${TLDW_VENV_DIR:-venv}"
export TLDW_SKIP_INSTALL="${TLDW_SKIP_INSTALL:-1}"

cd "$install_dir"
exec bash "$launcher" "$@"
