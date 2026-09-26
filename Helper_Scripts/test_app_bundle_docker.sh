#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
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
session_cookie_name=$(sed -n 's/^SINGLE_USER_SESSION_COOKIE_NAME=//p' "$env_file")
csrf_cookie_name=$(sed -n 's/^CSRF_COOKIE_NAME=//p' "$env_file")
[[ -n "$session_cookie_name" && -n "$csrf_cookie_name" ]]

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
curl --fail --silent --show-error "$public_url/docs-static/Documentation.md" >/dev/null
curl --fail --silent --show-error "$public_url/static/favicon.ico" >/dev/null

# Exercise the Next documentation API from the extracted bundle, outside a checkout.
documentation_manifest="$test_root/documentation-manifest.json"
documentation_content="$test_root/documentation-content.json"
if [[ $(curl --silent --show-error --max-time 20 --output "$documentation_manifest" \
  --write-out '%{http_code}' "$public_url/api/documentation/manifest") != 200 ]]; then
  echo 'WebUI documentation manifest did not return 200.' >&2
  exit 1
fi
if ! docker compose --project-name "$project_id" --env-file "$env_file" \
  -f "$bundle_dir/compose.yaml" exec -T webui node -e '
    try {
      const manifest = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
      const entries = manifest.docsBySource?.server;
      if (!Array.isArray(entries) || entries.length === 0 || !entries.some(entry =>
        entry.source === "server" && entry.relativePath === "API-related/AuthNZ-API-Guide.md"
      )) process.exit(1);
    } catch { process.exit(1); }
  ' < "$documentation_manifest"; then
  echo 'WebUI documentation manifest omitted the published server guide.' >&2
  exit 1
fi
if [[ $(curl --silent --show-error --max-time 20 --get --output "$documentation_content" \
  --write-out '%{http_code}' --data-urlencode 'source=server' \
  --data-urlencode 'relativePath=API-related/AuthNZ-API-Guide.md' \
  "$public_url/api/documentation/content") != 200 ]]; then
  echo 'WebUI published documentation content did not return 200.' >&2
  exit 1
fi
if ! docker compose --project-name "$project_id" --env-file "$env_file" \
  -f "$bundle_dir/compose.yaml" exec -T webui node -e '
    try {
      const body = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
      if (typeof body.content !== "string" || !body.content.startsWith("# AuthNZ API Guide\n"))
        process.exit(1);
    } catch { process.exit(1); }
  ' < "$documentation_content"; then
  echo 'WebUI published documentation content lacked the expected heading.' >&2
  exit 1
fi
if [[ $(curl --silent --show-error --max-time 20 --get --output /dev/null \
  --write-out '%{http_code}' --data-urlencode 'source=server' \
  --data-urlencode 'relativePath=../Design/private.md' \
  "$public_url/api/documentation/content") != 400 ]]; then
  echo 'WebUI documentation API did not refuse path traversal.' >&2
  exit 1
fi
setup_page=$(curl --fail --silent --show-error --location "$public_url/setup")
case "$setup_page" in
  *'/_next/static/'*) ;;
  *) echo 'Setup did not serve the managed WebUI page.' >&2; exit 1 ;;
esac
page=$(curl --fail --silent --show-error --location "$public_url/")
static_path=$(printf '%s' "$page" | grep -oE '/_next/static/[^" ]+' | head -n 1)
if [[ -z "$static_path" ]]; then
  echo 'Running WebUI did not reference a copied static asset.' >&2
  exit 1
fi
curl --fail --silent --show-error "$public_url$static_path" >/dev/null

session_headers="$test_root/session-headers"
session_cookies="$test_root/session-cookies"
curl --fail --silent --show-error --dump-header "$session_headers" --output /dev/null \
  --cookie-jar "$session_cookies" \
  --request POST --header "Origin: $public_url" \
  "$public_url/api/_tldw-webui/session"
grep -qi "^set-cookie: $session_cookie_name=" "$session_headers"
grep -qi "^set-cookie: $csrf_cookie_name=" "$session_headers"
curl --fail --silent --show-error --cookie "$session_cookies" \
  "$public_url/api/v1/users/me/profile" >/dev/null
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

echo 'Extracted Docker bundle passed startup, static/public assets, published documentation, auth, isolation, persistence, and tamper checks.'
