#!/bin/sh
set -eu

# Compose shell values outrank --env-file; only verified persisted values may resolve.
unset TLDW_PROJECT_ID TLDW_PUBLIC_PORT SINGLE_USER_API_KEY TLDW_GATEWAY_HOP_SECRET SINGLE_USER_SESSION_COOKIE_NAME CSRF_COOKIE_NAME TLDW_BACKEND_IMAGE TLDW_WEBUI_IMAGE TLDW_GATEWAY_IMAGE

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ -n "${TLDW_APP_STATE_DIR:-}" ]; then
  state_root=$TLDW_APP_STATE_DIR
elif [ "$(uname -s)" = Darwin ]; then
  state_root="$HOME/Library/Application Support/tldw/app"
else
  state_root="${XDG_DATA_HOME:-$HOME/.local/share}/tldw/app"
fi
"$bundle_dir/start.sh" --verify-only >/dev/null
env_file="$state_root/instance/config.env"
if [ ! -f "$env_file" ]; then
  echo 'No initialized tldw instance was found.' >&2
  exit 1
fi
project_id=$(sed -n 's/^TLDW_PROJECT_ID=//p' "$env_file")
public_port=$(sed -n 's/^TLDW_PUBLIC_PORT=//p' "$env_file")
case "$project_id" in
  ''|*[!a-zA-Z0-9_.-]*) echo 'Instance project ID is invalid.' >&2; exit 1 ;;
esac
case "$public_port" in
  ''|*[!0-9]*) echo 'Instance public port is invalid.' >&2; exit 1 ;;
esac
if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker is required to inspect this bundle.' >&2
  exit 1
fi
docker compose --project-name "$project_id" --env-file "$env_file" \
  -f "$bundle_dir/compose.yaml" ps
echo "Browser URL: http://127.0.0.1:$public_port/"
