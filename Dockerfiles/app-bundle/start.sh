#!/bin/sh
set -eu

# Compose shell values outrank --env-file; only verified persisted values may resolve.
unset TLDW_PROJECT_ID TLDW_PUBLIC_PORT SINGLE_USER_API_KEY TLDW_GATEWAY_HOP_SECRET SINGLE_USER_SESSION_COOKIE_NAME CSRF_COOKIE_NAME TLDW_BACKEND_IMAGE TLDW_WEBUI_IMAGE TLDW_GATEWAY_IMAGE

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
control_image='__CONTROL_IMAGE_DIGEST__'
trusted_key_id='__TRUSTED_KEY_ID__'
case "$control_image:$trusted_key_id" in
  *'__'*)
    echo 'This source template must be filled with a signed release control image and key.' >&2
    exit 1 ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker is required to start this bundle.' >&2
  exit 1
fi
# The browser URL belongs to this host. A remote daemon's loopback publication
# cannot establish readiness here, even when its bind-mount paths happen to match.
if [ -n "${DOCKER_CONTEXT:-}" ]; then
  docker_endpoint=$(docker context inspect --format '{{.Endpoints.docker.Host}}' "$DOCKER_CONTEXT") || exit 1
elif [ -n "${DOCKER_HOST:-}" ]; then
  docker_endpoint=$DOCKER_HOST
else
  docker_context=$(docker context show) || exit 1
  docker_endpoint=$(docker context inspect --format '{{.Endpoints.docker.Host}}' "$docker_context") || exit 1
fi
case "$docker_endpoint" in
  unix:///*|npipe:////./pipe/*) ;;
  *) echo 'This bundle requires local Docker through a Unix socket or Windows named pipe; remote/TCP daemons cannot serve its local browser URL.' >&2; exit 1 ;;
esac
architecture=$(docker info --format '{{.Architecture}}') || {
  echo 'Docker daemon is unavailable.' >&2
  exit 1
}
docker compose version >/dev/null || {
  echo 'Docker Compose v2 is required.' >&2
  exit 1
}
case "$architecture" in
  x86_64|amd64) platform=linux/amd64 ;;
  aarch64|arm64) platform=linux/arm64 ;;
  *) echo "Unsupported Docker architecture: $architecture" >&2; exit 1 ;;
esac

if [ -n "${TLDW_APP_STATE_DIR:-}" ]; then
  state_root=$TLDW_APP_STATE_DIR
elif [ "$(uname -s)" = Darwin ]; then
  state_root="$HOME/Library/Application Support/tldw/app"
else
  state_root="${XDG_DATA_HOME:-$HOME/.local/share}/tldw/app"
fi
mkdir -p "$state_root"
chmod 700 "$state_root"

verify_only=0
if [ "${1:-}" = --verify-only ]; then verify_only=1; fi
set --
if [ "$verify_only" != 1 ] && [ -n "${TLDW_APP_PUBLIC_PORT:-}" ]; then
  case "$TLDW_APP_PUBLIC_PORT" in
    *[!0-9]*|'') echo 'TLDW_APP_PUBLIC_PORT must be a decimal port.' >&2; exit 1 ;;
  esac
  set -- --public-port "$TLDW_APP_PUBLIC_PORT"
fi

control() {
  command=$1
  shift
  docker run --rm --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --user "$(id -u):$(id -g)" \
    -v "$bundle_dir:/bundle:ro" -v "$state_root:/state" \
    "$control_image" "$command" \
    --state /state/instance --manifest /bundle/manifest.json \
    --signature /bundle/manifest.sig --bundle-root /bundle \
    --platform "$platform" --expected-signer "$trusted_key_id" "$@"
}

control verify "$@" || {
  echo 'Bundle verification failed; no application containers were started.' >&2
  exit 1
}
if [ "${verify_only:-0}" = 1 ]; then exit 0; fi
# Reserve host bindings with Docker before first origin/credentials are committed.
# A failed create owns nothing; only the returned immutable container ID is removed.
preflight_id=''
cleanup_preflight() {
  if [ -n "$preflight_id" ]; then
    if ! docker rm -f "$preflight_id" >/dev/null 2>&1; then
      echo "Port preflight cleanup failed. Recovery container: $preflight_id; state: $state_root." >&2
      return 1
    fi
    preflight_id=''
  fi
}
trap 'cleanup_preflight || exit 1' 0
trap 'exit 1' HUP INT TERM
check_port() {
  requested=$1
  preflight_id=$(docker create --network bridge --read-only --cap-drop ALL \
    --security-opt no-new-privileges -p "127.0.0.1:${requested}:8080" \
    --entrypoint python "$control_image" -c 'import time; time.sleep(120)' 2>/dev/null) || {
    echo 'Unable to create Docker port preflight; no origin was saved.' >&2
    return 2
  }
  case "$preflight_id" in ''|*[!0-9a-f]*) echo 'Invalid preflight resource identity.' >&2; preflight_id=''; return 2 ;; esac
  if [ "${#preflight_id}" -ne 64 ]; then preflight_id=''; return 2; fi
  port_started=0
  if docker start "$preflight_id" >/dev/null 2>&1; then port_started=1; fi
  selected_port=''
  if [ "$port_started" = 1 ] && [ -z "$requested" ]; then
    selected_port=$(docker inspect --format '{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}' "$preflight_id") || {
      cleanup_preflight || return 2
      return 2
    }
    case "$selected_port" in ''|*[!0-9]*) cleanup_preflight || return 2; return 2 ;; esac
  fi
  cleanup_preflight || return 2
  [ "$port_started" = 1 ]
}
if [ ! -e "$state_root/instance" ]; then
  port_result=0
  check_port "${TLDW_APP_PUBLIC_PORT:-8080}" || port_result=$?
  if [ "$port_result" -ne 0 ]; then
    if [ "$port_result" = 2 ]; then exit 1; fi
    if [ -n "${TLDW_APP_PUBLIC_PORT:-}" ]; then
      echo "Public port $TLDW_APP_PUBLIC_PORT is unavailable. Choose another TLDW_APP_PUBLIC_PORT and retry; no origin was saved." >&2
    elif check_port ''; then
      echo "Default port 8080 is unavailable. Available choice: set TLDW_APP_PUBLIC_PORT=$selected_port and run start.sh again; no origin was saved." >&2
    else
      echo 'No available Docker port choice could be confirmed; no origin was saved.' >&2
    fi
    exit 1
  fi
fi
control init "$@" || {
  echo 'Instance initialization failed; no application containers were started.' >&2
  exit 1
}

env_file="$state_root/instance/config.env"
if [ ! -f "$env_file" ]; then
  echo 'Instance configuration is missing after initialization.' >&2
  exit 1
fi
project_id=$(sed -n 's/^TLDW_PROJECT_ID=//p' "$env_file")
public_port=$(sed -n 's/^TLDW_PUBLIC_PORT=//p' "$env_file")
case "$project_id:$public_port" in
  *[!a-zA-Z0-9_:.-]*) echo 'Instance configuration contains unsafe identity values.' >&2; exit 1 ;;
esac
if [ -z "$project_id" ] || [ -z "$public_port" ]; then
  echo 'Instance configuration lacks a project ID or port.' >&2
  exit 1
fi

compose() {
  docker compose --project-name "$project_id" --env-file "$env_file" \
    -f "$bundle_dir/compose.yaml" "$@"
}

compose pull
failed_start() {
  if compose down >/dev/null 2>&1; then
    echo "Application failed readiness; partial services were stopped. Data retained at $state_root." >&2
  else
    echo "Application failed readiness and cleanup failed; services may still be running. Recovery state: $state_root. Retry stop.sh." >&2
  fi
  exit 1
}
if ! compose up -d --no-build --wait --wait-timeout 600; then
  failed_start
fi
# Inspection stays in the pipe: its Env contains credentials and must never be logged.
container_ids=$(compose ps -q app webui gateway) || failed_start
if [ -z "$container_ids" ]; then failed_start; fi
if ! docker inspect $container_ids "${project_id}_private" | \
  docker run --rm -i --network "${project_id}_private" --read-only --cap-drop ALL \
    --security-opt no-new-privileges --user "$(id -u):$(id -g)" \
    -v "$bundle_dir:/bundle:ro" -v "$state_root:/state:ro" \
    "$control_image" ready --state /state/instance --manifest /bundle/manifest.json \
    --signature /bundle/manifest.sig --bundle-root /bundle --platform "$platform" \
    --expected-signer "$trusted_key_id"; then
  failed_start
fi
browser_url="http://127.0.0.1:$public_port/"
echo "Open $browser_url"
if [ "${TLDW_APP_NO_BROWSER:-0}" != 1 ]; then
  if [ "$(uname -s)" = Darwin ] && command -v open >/dev/null 2>&1; then
    open "$browser_url" >/dev/null 2>&1 || true
  elif [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$browser_url" >/dev/null 2>&1 || true
  fi
fi
