#!/usr/bin/env bash
# Quick-launch local single-user tldw_server without Docker or Make.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD="${TLDW_PYTHON:-python3}"
VENV_DIR="${TLDW_VENV_DIR:-.venv}"
VENV_PYTHON="$VENV_DIR/bin/python"
ENV_FILE="${TLDW_ENV_FILE:-tldw_Server_API/Config_Files/.env}"
HOST="${TLDW_HOST:-127.0.0.1}"
PORT="${TLDW_PORT:-8000}"

echo "=== tldw_server quick launch ==="
echo ""

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "[quick-launch] $PYTHON_CMD not found. Install Python 3.10+ or set TLDW_PYTHON." >&2
    exit 1
fi

"$PYTHON_CMD" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "[quick-launch] Python 3.10+ is required." >&2
    exit 1
}

if [ ! -x "$VENV_PYTHON" ]; then
    echo "[quick-launch] Creating virtualenv at $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

if [ "${TLDW_SKIP_INSTALL:-0}" != "1" ]; then
    echo "[quick-launch] Installing/updating local Python dependencies..."
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
    "$VENV_PYTHON" -m pip install -e .
else
    echo "[quick-launch] Skipping dependency install because TLDW_SKIP_INSTALL=1"
fi

echo "[quick-launch] Configuring local single-user profile..."
"$VENV_PYTHON" -m tldw_Server_API.cli.wizard.cli init \
    --profile local-single \
    --env-file "$ENV_FILE" \
    --default \
    --yes

echo ""
echo "[quick-launch] Starting API at http://$HOST:$PORT"
echo "[quick-launch] Docs:   http://$HOST:$PORT/docs"
echo "[quick-launch] Health: http://$HOST:$PORT/health"
echo ""

TLDW_ENV_FILE="$ENV_FILE" exec "$VENV_PYTHON" -m uvicorn \
    tldw_Server_API.app.main:app \
    --host "$HOST" \
    --port "$PORT"
