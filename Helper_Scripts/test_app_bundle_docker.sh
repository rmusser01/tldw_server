#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; printf "Bundle smoke failed at line %s (exit %s).\n" "$LINENO" "$status" >&2' ERR

if [[ $# -ne 1 ]]; then
  echo 'Usage: test_app_bundle_docker.sh <extracted-bundle-directory>' >&2
  exit 2
fi
bundle_dir=$(cd "$1" && pwd -P)
test_root=$(mktemp -d)
export TLDW_APP_STATE_DIR="$test_root/instance-data"
active_state_dir=$TLDW_APP_STATE_DIR
export TLDW_APP_PUBLIC_PORT="${TLDW_APP_PUBLIC_PORT:-18080}"
public_url="http://127.0.0.1:$TLDW_APP_PUBLIC_PORT"

cleanup() {
  TLDW_APP_STATE_DIR="$active_state_dir" "$bundle_dir/stop.sh" >/dev/null 2>&1 || true
  rm -rf "$test_root"
}
trap cleanup EXIT

cd "$test_root"
"$bundle_dir/start.sh"
env_file="$TLDW_APP_STATE_DIR/instance/config.env"
first_config_hash=$(sha256sum "$env_file" | awk '{print $1}')
project_id=$(sed -n 's/^TLDW_PROJECT_ID=//p' "$env_file")

ready=0
for _ in $(seq 1 120); do
  if curl --fail --silent --max-time 3 "$public_url/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ $ready != 1 ]]; then
  echo 'Gateway/backend did not become ready within 10 minutes.' >&2
  exit 1
fi

curl --fail --silent --show-error "$public_url/_tldw/status" | grep -q '"ready":true'
curl --fail --silent --show-error "$public_url/favicon.ico" >/dev/null
curl --fail --silent --show-error "$public_url/docs" >/dev/null
page=$(curl --fail --silent --show-error --location "$public_url/")
static_path=$(printf '%s' "$page" | grep -oE '/_next/static/[^" ]+' | head -n 1)
if [[ -z "$static_path" ]]; then
  echo 'Running WebUI did not reference a copied static asset.' >&2
  exit 1
fi
curl --fail --silent --show-error "$public_url$static_path" >/dev/null

session_headers="$test_root/session-headers"
curl --fail --silent --show-error --dump-header "$session_headers" --output /dev/null \
  --request POST --header "Origin: $public_url" \
  "$public_url/api/_tldw-webui/session"
grep -qi '^set-cookie:' "$session_headers"
if [[ $(curl --silent --output /dev/null --write-out '%{http_code}' \
  --header 'Host: hostile.invalid' "$public_url/_tldw/status") != 403 ]]; then
  echo 'Gateway accepted a hostile Host header.' >&2
  exit 1
fi

if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo 'Backend port is directly reachable from the host.' >&2
  exit 1
fi
if curl --fail --silent --max-time 2 http://127.0.0.1:3000/ >/dev/null 2>&1; then
  echo 'Next port is directly reachable from the host.' >&2
  exit 1
fi

docker compose --project-name "$project_id" --env-file "$env_file" \
  -f "$bundle_dir/compose.yaml" exec -T app sh -c \
  'printf durable > /app/Databases/wp1-candidate-sentinel'
(cd /tmp && "$bundle_dir/stop.sh")
"$bundle_dir/start.sh"
second_config_hash=$(sha256sum "$env_file" | awk '{print $1}')
[[ "$first_config_hash" == "$second_config_hash" ]]
docker compose --project-name "$project_id" --env-file "$env_file" \
  -f "$bundle_dir/compose.yaml" exec -T app sh -c \
  'test "$(cat /app/Databases/wp1-candidate-sentinel)" = durable'

bad_bundle="$test_root/tampered-bundle"
cp -R "$bundle_dir" "$bad_bundle"
printf 'tamper' >> "$bad_bundle/manifest.json"
export TLDW_APP_STATE_DIR="$test_root/tampered-state"
if "$bad_bundle/start.sh" >"$test_root/tampered-stdout" 2>"$test_root/tampered-stderr"; then
  echo 'Tampered bundle unexpectedly started.' >&2
  exit 1
fi
[[ ! -e "$TLDW_APP_STATE_DIR/instance" ]]

echo 'Extracted Docker bundle passed startup, static/public assets, auth, isolation, persistence, and tamper checks.'
