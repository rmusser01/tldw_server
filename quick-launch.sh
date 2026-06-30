#!/usr/bin/env bash
# Quick-launch local single-user tldw_server without Docker or Make.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mode="${1:-all}"
PYTHON_CMD="${TLDW_PYTHON:-python3}"
VENV_DIR="${TLDW_VENV_DIR:-.venv}"
INSTALL_MARKER="$VENV_DIR/.initialized"
ENV_FILE="${TLDW_ENV_FILE:-tldw_Server_API/Config_Files/.env}"
HOST="${TLDW_HOST:-127.0.0.1}"
PORT="${TLDW_API_PORT:-${TLDW_PORT:-8000}}"
WEBUI_PORT="${TLDW_WEBUI_PORT:-8080}"
WEBUI_DIR="apps/tldw-frontend"
VENV_PYTHON=""
api_pid=""

usage() {
    cat <<USAGE
Usage: $(basename "$0") [api|webui|all]

Modes:
  api     Start the FastAPI backend only on http://$HOST:$PORT
  webui   Start the Next.js WebUI only on http://127.0.0.1:$WEBUI_PORT
  all     Start the backend, then run the WebUI (default)

Environment:
  TLDW_PYTHON         Python executable for venv creation (default: python3)
  TLDW_VENV_DIR       Virtualenv directory (default: .venv)
  TLDW_ENV_FILE       Env file path (default: tldw_Server_API/Config_Files/.env)
  TLDW_HOST           Backend host (default: 127.0.0.1)
  TLDW_API_PORT       Backend port (default: TLDW_PORT or 8000)
  TLDW_PORT           Legacy backend port override
  TLDW_WEBUI_PORT     WebUI port (default: 8080)
  NEXT_PUBLIC_API_URL Override WebUI API URL
USAGE
}

resolve_venv_python() {
    if [ -x "$VENV_DIR/bin/python" ]; then
        printf '%s\n' "$VENV_DIR/bin/python"
        return 0
    fi

    if [ -x "$VENV_DIR/Scripts/python" ]; then
        printf '%s\n' "$VENV_DIR/Scripts/python"
        return 0
    fi

    if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
        printf '%s\n' "$VENV_DIR/Scripts/python.exe"
        return 0
    fi

    return 1
}

ensure_python() {
    if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
        echo "[quick-launch] $PYTHON_CMD not found. Install Python 3.10+ or set TLDW_PYTHON." >&2
        exit 1
    fi

    "$PYTHON_CMD" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
        echo "[quick-launch] Python 3.10+ is required." >&2
        exit 1
    }
}

ensure_api_environment() {
    ensure_python

    local venv_created=0
    if ! VENV_PYTHON="$(resolve_venv_python)"; then
        echo "[quick-launch] Creating virtualenv at $VENV_DIR"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        venv_created=1
        if ! VENV_PYTHON="$(resolve_venv_python)"; then
            echo "[quick-launch] Could not find Python in $VENV_DIR after creating the virtualenv." >&2
            exit 1
        fi
    fi

    if [ "${TLDW_SKIP_INSTALL:-0}" != "1" ]; then
        if [ "${TLDW_FORCE_INSTALL:-0}" = "1" ] || [ "$venv_created" = "1" ] || [ ! -f "$INSTALL_MARKER" ]; then
            echo "[quick-launch] Installing/updating local Python dependencies..."
            "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
            "$VENV_PYTHON" -m pip install -e .
            touch "$INSTALL_MARKER"
        else
            echo "[quick-launch] Dependency setup already completed; set TLDW_FORCE_INSTALL=1 to reinstall/update."
        fi
    else
        echo "[quick-launch] Skipping dependency install because TLDW_SKIP_INSTALL=1"
    fi

    echo "[quick-launch] Configuring local single-user profile..."
    "$VENV_PYTHON" -m tldw_Server_API.cli.wizard.cli init \
        --profile local-single \
        --env-file "$ENV_FILE" \
        --default \
        --yes
}

ensure_bun() {
    if ! command -v bun >/dev/null 2>&1; then
        echo "[quick-launch] Bun is required to launch the WebUI but was not found in PATH." >&2
        echo "[quick-launch] Install Bun from https://bun.sh/docs/installation, then rerun this launcher." >&2
        exit 1
    fi
}

ensure_webui_dir() {
    if [ ! -d "$WEBUI_DIR" ]; then
        echo "[quick-launch] WebUI directory not found: $SCRIPT_DIR/$WEBUI_DIR" >&2
        echo "[quick-launch] Update your checkout before launching the WebUI." >&2
        exit 1
    fi
}

run_api() {
    echo "=== tldw_server quick launch: API ==="
    echo ""
    ensure_api_environment

    echo ""
    echo "[quick-launch] Starting API at http://$HOST:$PORT"
    echo "[quick-launch] Docs:   http://$HOST:$PORT/docs"
    echo "[quick-launch] Health: http://$HOST:$PORT/health"
    echo ""

    TLDW_ENV_FILE="$ENV_FILE" exec "$VENV_PYTHON" -m uvicorn \
        tldw_Server_API.app.main:app \
        --host "$HOST" \
        --port "$PORT"
}

start_api_background() {
    echo "=== tldw_server quick launch: API + WebUI ==="
    echo ""
    ensure_api_environment

    echo ""
    echo "[quick-launch] Starting API at http://$HOST:$PORT"
    echo "[quick-launch] Docs:   http://$HOST:$PORT/docs"
    echo "[quick-launch] Health: http://$HOST:$PORT/health"
    TLDW_ENV_FILE="$ENV_FILE" "$VENV_PYTHON" -m uvicorn \
        tldw_Server_API.app.main:app \
        --host "$HOST" \
        --port "$PORT" &
    api_pid="$!"
    sleep "${TLDW_API_START_DELAY:-2}"
}

run_webui() {
    ensure_bun
    ensure_webui_dir
    api_url_host="$HOST"
    if [ "$api_url_host" = "0.0.0.0" ]; then
        api_url_host="127.0.0.1"
    fi

    if [ -z "${NEXT_PUBLIC_API_URL:-}" ]; then
        export NEXT_PUBLIC_API_URL="http://$api_url_host:$PORT"
        if [ "$HOST" = "0.0.0.0" ]; then
            echo "[quick-launch] API is bound to 0.0.0.0; using 127.0.0.1 for local browser requests."
            echo "[quick-launch] Set NEXT_PUBLIC_API_URL to your LAN URL for non-local browser clients."
        fi
    fi

    echo ""
    echo "[quick-launch] Starting WebUI at http://127.0.0.1:$WEBUI_PORT"
    echo "[quick-launch] Using API URL: $NEXT_PUBLIC_API_URL"
    echo ""

    cd "$WEBUI_DIR"
    bun run dev -- -p "$WEBUI_PORT"
}

cleanup() {
    if [ -n "$api_pid" ] && kill -0 "$api_pid" >/dev/null 2>&1; then
        echo "[quick-launch] Stopping API..."
        kill "$api_pid" >/dev/null 2>&1 || true
        wait "$api_pid" 2>/dev/null || true
    fi
}

run_all() {
    trap cleanup EXIT INT TERM
    start_api_background
    run_webui
}

case "$mode" in
    api)
        run_api
        ;;
    webui)
        echo "=== tldw_server quick launch: WebUI ==="
        run_webui
        ;;
    all)
        run_all
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "[quick-launch] Unknown launch mode: $mode" >&2
        usage >&2
        exit 2
        ;;
esac
