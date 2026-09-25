#!/bin/sh
set -eu

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

set --
if [ -n "${TLDW_APP_PUBLIC_PORT:-}" ]; then
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

if command -v lsof >/dev/null 2>&1 && \
   lsof -nP -iTCP:"$public_port" -sTCP:LISTEN >/dev/null 2>&1; then
  own_gateway=$(compose ps -q gateway)
  if [ -z "$own_gateway" ]; then
    echo "Public port $public_port is already occupied." >&2
    exit 1
  fi
fi

compose pull
if ! compose up -d --no-build --wait --wait-timeout 600; then
  compose down >/dev/null 2>&1 || true
  echo 'Application failed readiness; partial services were stopped and data was retained.' >&2
  exit 1
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
