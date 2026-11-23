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
# Exact loopback names only (no case folding, no other 127.x addresses).
is_loopback_host() {
    case "$1" in
        127.0.0.1 | localhost | ::1) return 0 ;;
    esac
    return 1
}

# Print the host of an http(s) URL, without IPv6 brackets.
url_host() {
    local rest="${1#*://}"
    rest="${rest%%/*}"
    rest="${rest##*@}"
    case "$rest" in
        \[*)
            rest="${rest#\[}"
            printf '%s\n' "${rest%%]*}"
            ;;
        *) printf '%s\n' "${rest%%:*}" ;;
    esac
}

# Browsers and the WebUI reach a 0.0.0.0-bound API through loopback.
api_url_host="$HOST"
if [ "$api_url_host" = "0.0.0.0" ]; then
    api_url_host="127.0.0.1"
fi

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
  TLDW_API_START_DELAY
                      Maximum seconds 'all' waits for the API's /health before
                      starting the WebUI (default: 120, minimum: 5)
  NEXT_PUBLIC_API_URL Browser calls this API URL directly; no automatic sign-in. The
                      production build also reads it from $WEBUI_DIR/.env*
                      (TLDW_WEBUI_DEV=1 ignores those files and uses the API above)
  TLDW_WEBUI_DEV      Set to 1 for the hot-reloading Next.js dev server instead of the
                      production build (WebUI development; several GB of RAM; no
                      automatic sign-in)
  TLDW_WEBUI_EXPOSE_RUNTIME_AUTH
                      Default 1: the WebUI signs browsers on this computer in with
                      SINGLE_USER_API_KEY, so programs on this computer can sign in
                      while it runs. Set 0 (shell, env file, or $WEBUI_DIR/.env*)
                      on a shared computer and paste the API key instead.

In 'all' mode on a loopback API host the API's cookies drop the Secure flag so sign-in
works over plain http; SESSION_COOKIE_SECURE=1 (shell or env file) keeps it, e.g. behind
a TLS proxy. These settings are read from the shell and $ENV_FILE only.

The WebUI is built once and rebuilt when apps/ changes in git or the API address
changes. To force a rebuild, for example after changing other NEXT_PUBLIC_* values or
updating a zip download (which has no git revision),
delete apps/tldw-frontend/.next and launch again.
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
    if is_loopback_host "$HOST"; then
        # Plain-http loopback, as in Dockerfiles/docker-compose.webui.yml: Secure cookies are
        # not reliably kept over http://127.0.0.1 (Safari), so drop Secure and keep CSRF on.
        # A value from the shell or the env file wins.
        export_setting SESSION_COOKIE_SECURE 0
        export_setting CSRF_ENABLED 1
    fi
    TLDW_ENV_FILE="$ENV_FILE" "$VENV_PYTHON" -m uvicorn \
        tldw_Server_API.app.main:app \
        --host "$HOST" \
        --port "$PORT" &
    api_pid="$!"
}

# Wait for /health rather than a fixed sleep: the production WebUI starts within seconds.
# Called just before the WebUI server starts, so a first-launch build overlaps API boot.
# The probe bypasses http_proxy/HTTP_PROXY: the API is on this computer.
wait_for_api() {
    local url="http://$api_url_host:$PORT/health"
    local max_wait="${TLDW_API_START_DELAY:-120}"
    local deadline
    case "$max_wait" in
        '' | *[!0-9]*) max_wait=120 ;;
    esac
    if [ "$max_wait" -lt 5 ]; then
        max_wait=5
    fi
    deadline=$((SECONDS + max_wait))
    echo "[quick-launch] Waiting for the API at $url (up to ${max_wait}s) ..."
    until "$VENV_PYTHON" -c "import sys, urllib.request; urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=2)" "$url" >/dev/null 2>&1; do
        if ! kill -0 "$api_pid" 2>/dev/null; then
            echo "[quick-launch] The API exited during startup; see the messages above." >&2
            exit 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "[quick-launch] The API did not answer at $url within ${max_wait}s; see the messages above, or raise TLDW_API_START_DELAY." >&2
            exit 1
        fi
        sleep 1
    done
}

# Print KEY from dotenv FILE the way the backend reads it (python-dotenv); empty if unreadable.
read_env_value() {
    local python_cmd
    [ -f "$1" ] || return 0
    python_cmd="$(resolve_venv_python)" || return 0
    "$python_cmd" -c "import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2]) or '')" "$1" "$2" || true
}

# Export NAME with the first non-empty value of: the shell, the env file, $2.
export_setting() {
    local value="${!1:-}"
    if [ -z "$value" ]; then
        value="$(read_env_value "$ENV_FILE" "$1")"
    fi
    value="${value:-${2:-}}"
    if [ -n "$value" ]; then
        export "$1=$value"
    fi
}

# `next build` takes minutes and several GB of RAM on a laptop, so build only when needed.
# The key names what a build bakes in: networking profile, API address, and the git tree of
# apps/ (so backend-only updates reuse the build). It is written into .next only after a
# successful build, and removed before building, so a failed build always rebuilds next time.
# ponytail: without git (zip download) the tree is "no-git"; delete .next after updating.
ensure_webui_build() {
    local build_script="$1"
    local key_file="$WEBUI_DIR/.next/quick-launch-build-key"
    local build_key
    build_key="$1 $2 $(git rev-parse --verify --quiet HEAD:apps 2>/dev/null || echo no-git)"

    if [ -f "$WEBUI_DIR/.next/BUILD_ID" ] && [ "$(cat "$key_file" 2>/dev/null)" = "$build_key" ]; then
        echo "[quick-launch] Reusing the existing WebUI build."
        return 0
    fi

    echo "[quick-launch] Building the WebUI; the first launch takes a few minutes..."
    rm -f "$key_file"
    if ! (cd "$WEBUI_DIR" && bun run "$build_script"); then
        echo "[quick-launch] The WebUI build failed. Try 'bun install' in apps/, close other programs to free memory, or set TLDW_WEBUI_DEV=1 to use the dev server." >&2
        exit 1
    fi
    printf '%s\n' "$build_key" >"$key_file"
}

# Print "<file><TAB><value>" for KEY exactly as `next build`/`next start` load it from the
# WebUI's .env.production.local, .env.local, .env.production and .env files: Next's own
# loader (@next/env, reached through next/package.json), so no venv is needed and parsing,
# $VAR expansion and shell precedence match Next. Prints only KEY, and nothing when the
# shell sets KEY or no file does. --env-file stops bun from loading those files itself.
TAB=$'\t'
FRONTEND_ENV_READER="const { createRequire } = require('node:module'); const key = process.argv[1]; const nextEnv = createRequire(require.resolve('next/package.json', { paths: [process.cwd()] }))('@next/env'); const { parsedEnv, loadedEnvFiles } = nextEnv.loadEnvConfig(process.cwd(), false, { info() {}, error() {} }, true); const file = loadedEnvFiles.find((f) => f.env && Object.hasOwn(f.env, key)); if (parsedEnv && Object.hasOwn(parsedEnv, key) && file) process.stdout.write(file.path + '\t' + parsedEnv[key]);"
read_frontend_env() {
    local no_env_file
    no_env_file="$(mktemp)"
    (cd "$WEBUI_DIR" && NODE_ENV=production bun --env-file="$no_env_file" -e "$FRONTEND_ENV_READER" "$1") || true
    rm -f "$no_env_file"
}

# NEXT_PUBLIC_API_URL in the WebUI's own env files (README tells users to copy
# .env.local.example) is a user override too: quickstart builds refuse it.
use_frontend_env_api_url() {
    local found
    found="$(read_frontend_env NEXT_PUBLIC_API_URL)"
    if [ -n "${found#*"$TAB"}" ]; then
        export NEXT_PUBLIC_API_URL="${found#*"$TAB"}"
        echo "[quick-launch] Using NEXT_PUBLIC_API_URL from $WEBUI_DIR/${found%%"$TAB"*} (the browser calls the API directly)."
    fi
}

# Like the Docker WebUI, the WebUI server (never the browser bundle) turns SINGLE_USER_API_KEY
# into a cookie session for browsers on this computer, so nobody pastes a key; programs on this
# computer can sign in too, hence the loopback-only bind and the TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=0
# opt-out (shell, else env file, else the WebUI's .env*, else 1; the default would otherwise
# mask a .env* value, since Next never overrides a set variable). Set after the build so the
# key never reaches it. Unexported settings still reach Next from the WebUI's .env* files.
configure_runtime_sign_in() {
    local reason frontend_flag
    export_setting AUTH_MODE
    export_setting SINGLE_USER_API_KEY
    export_setting SINGLE_USER_SESSION_COOKIE_NAME
    frontend_flag="$(read_frontend_env TLDW_WEBUI_EXPOSE_RUNTIME_AUTH)"
    frontend_flag="${frontend_flag#*"$TAB"}"
    export_setting TLDW_WEBUI_EXPOSE_RUNTIME_AUTH "${frontend_flag:-1}"

    if [ "$TLDW_WEBUI_EXPOSE_RUNTIME_AUTH" != "1" ]; then
        echo "[quick-launch] Automatic sign-in is off (TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=$TLDW_WEBUI_EXPOSE_RUNTIME_AUTH); paste your API key in the WebUI."
    elif [ -z "${SINGLE_USER_API_KEY:-}" ]; then
        if ! resolve_venv_python >/dev/null; then
            reason="no Python virtualenv at $VENV_DIR to read $ENV_FILE"
        elif [ ! -f "$ENV_FILE" ]; then
            reason="no env file at $ENV_FILE"
        else
            reason="SINGLE_USER_API_KEY is not set in $ENV_FILE"
        fi
        echo "[quick-launch] Automatic sign-in is unavailable: $reason. Paste your API key in the WebUI."
    elif [ "$mode" = "webui" ]; then
        echo "[quick-launch] If sign-in does not stick (for example in Safari), start the API with SESSION_COOKIE_SECURE=0, as 'all' mode does on loopback."
    fi
}

run_webui() {
    ensure_bun
    ensure_webui_dir
    if [ "$HOST" = "0.0.0.0" ] && [ -z "${NEXT_PUBLIC_API_URL:-}" ]; then
        echo "[quick-launch] API is bound to 0.0.0.0; using 127.0.0.1 for local browser requests."
        echo "[quick-launch] Set NEXT_PUBLIC_API_URL to your LAN URL for non-local browser clients."
    fi

    if [ "${TLDW_WEBUI_DEV:-0}" = "1" ]; then
        # Developer opt-in: hot reload, but `next dev` holds several GB of RAM.
        export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://$api_url_host:$PORT}"
        if [ -n "$api_pid" ]; then
            wait_for_api
        fi
        echo ""
        echo "[quick-launch] Starting WebUI dev server (TLDW_WEBUI_DEV=1) on all interfaces, port $WEBUI_PORT; open http://127.0.0.1:$WEBUI_PORT"
        echo "[quick-launch] Using API URL: $NEXT_PUBLIC_API_URL"
        echo ""
        cd "$WEBUI_DIR"
        bun run dev -- -p "$WEBUI_PORT"
        return 0
    fi

    if [ -z "${NEXT_PUBLIC_API_URL:-}" ]; then
        use_frontend_env_api_url
    fi
    api_url="${NEXT_PUBLIC_API_URL:-http://$api_url_host:$PORT}"

    if [ -n "${NEXT_PUBLIC_API_URL:-}" ]; then
        # Advanced networking: the browser calls NEXT_PUBLIC_API_URL directly. build:dev is
        # still an optimized `next build`; only its networking profile differs from build:prod.
        export NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE="advanced"
        ensure_webui_build "build:dev" "$api_url"
        # LAN browsers cannot use a loopback API URL, so keep such a WebUI on loopback too.
        if is_loopback_host "$(url_host "$api_url")"; then
            webui_host="127.0.0.1"
        else
            webui_host="0.0.0.0"
        fi
        echo "[quick-launch] Using API URL: $api_url"
    else
        # Quickstart networking, as in the Docker WebUI: the browser talks only to the WebUI,
        # which proxies /api and /health to the backend origin baked in at build time.
        export NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE="quickstart"
        export TLDW_INTERNAL_API_ORIGIN="$api_url"
        ensure_webui_build "build:prod" "$api_url"
        configure_runtime_sign_in
        webui_host="127.0.0.1"
        echo "[quick-launch] WebUI proxies API requests to: $api_url"
    fi

    if [ -n "$api_pid" ]; then
        wait_for_api
    fi
    echo ""
    echo "[quick-launch] Starting WebUI on $webui_host:$WEBUI_PORT; open http://127.0.0.1:$WEBUI_PORT"
    echo "[quick-launch] Next.js warns that \"next start\" does not work with \"output: standalone\"; that warning is expected here."
    echo ""
    cd "$WEBUI_DIR"
    bun run start -- -p "$WEBUI_PORT" -H "$webui_host"
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
