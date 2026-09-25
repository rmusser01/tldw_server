#!/bin/sh
set -eu

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ -n "${TLDW_APP_STATE_DIR:-}" ]; then
  state_root=$TLDW_APP_STATE_DIR
elif [ "$(uname -s)" = Darwin ]; then
  state_root="$HOME/Library/Application Support/tldw/app"
else
  state_root="${XDG_DATA_HOME:-$HOME/.local/share}/tldw/app"
fi
env_file="$state_root/instance/config.env"
if [ ! -f "$env_file" ]; then
  echo 'No initialized tldw instance was found.' >&2
  exit 1
fi
project_id=$(sed -n 's/^TLDW_PROJECT_ID=//p' "$env_file")
case "$project_id" in
  ''|*[!a-zA-Z0-9_.-]*) echo 'Instance project ID is invalid.' >&2; exit 1 ;;
esac
if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker is required to stop this bundle.' >&2
  exit 1
fi
docker compose --project-name "$project_id" --env-file "$env_file" \
  -f "$bundle_dir/compose.yaml" down
echo 'Application stopped. Persistent data and credentials were retained.'
