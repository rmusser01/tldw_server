#!/usr/bin/env bash
# macOS Finder-friendly wrapper for quick-launch.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec "$SCRIPT_DIR/quick-launch.sh" "$@"
