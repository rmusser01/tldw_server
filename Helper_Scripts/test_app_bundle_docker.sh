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
export TLDW_APP_NO_BROWSER=1
env_file="$TLDW_APP_STATE_DIR/instance/config.env"
project_id=""
source_commit=""
platform=""
lifecycle_passed=0
port_fixture_id=""
evidence_path="$(dirname "$bundle_dir")/lifecycle-evidence.json"
export TLDW_APP_PUBLIC_PORT="${TLDW_APP_PUBLIC_PORT:-18080}"
public_url="http://127.0.0.1:$TLDW_APP_PUBLIC_PORT"

cleanup() {
  local original_exit=$?
  local cleanup_failed=0
  trap - EXIT
  if [[ -n "$port_fixture_id" ]]; then
    docker rm -f "$port_fixture_id" >/dev/null 2>&1 || cleanup_failed=1
  fi
  if [[ -f "$env_file" ]]; then
    project_id=$(sed -n 's/^TLDW_PROJECT_ID=//p' "$env_file" 2>/dev/null) || cleanup_failed=1
    if [[ "$project_id" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
      docker compose --project-name "$project_id" --env-file "$env_file" \
        -f "$bundle_dir/compose.yaml" down --volumes >/dev/null 2>&1 || cleanup_failed=1
    else
      cleanup_failed=1
    fi
  fi
  if [[ $cleanup_failed == 0 ]]; then
    rm -rf "$test_root" >/dev/null 2>&1 || cleanup_failed=1
  fi
  if [[ $cleanup_failed != 0 ]]; then
    printf '%s\n' "$test_root" > "$(dirname "$evidence_path")/.lifecycle-cleanup-recovery" || true
    echo 'Lifecycle cleanup failed; disposable state retained for recovery.' >&2
  fi
  if ! python - "$evidence_path" "$original_exit" "$cleanup_failed" "$lifecycle_passed" "$source_commit" "$platform" <<'PY_CLEANUP'
import json
import sys
from pathlib import Path
path, original, failed, lifecycle, commit, platform = sys.argv[1:]
removed = failed == "0"
passed = original == "0" and removed and lifecycle == "1"
try:
    Path(path).write_text(json.dumps({
        "schema_version": 1, "source_commit": commit, "platform": platform,
        "passed": passed, "owned_resources_removed": removed,
        "checks": {name: {"passed": passed} for name in (
            "signed_start", "ready", "public_assets", "published_documentation",
            "cookie_auth", "private_isolation", "restart_persistence", "tamper_refused",
            "installer_authenticated_readiness", "probe_session_revoked", "occupied_default_offer",
            "occupied_explicit_retry", "established_origin_refused")},
    }, indent=2) + "\n")
except Exception:
    sys.exit("Lifecycle evidence update failed (private details suppressed).")
PY_CLEANUP
  then
    cleanup_failed=1
  fi
  if [[ $original_exit == 0 && $cleanup_failed != 0 ]]; then original_exit=1; fi
  if [[ $original_exit == 0 ]]; then
    echo 'Extracted Docker lifecycle checks and owned cleanup passed.'
  fi
  exit "$original_exit"
}
trap cleanup EXIT

cd "$test_root"
control_image=$(python - "$bundle_dir/manifest.json" <<'PY_CONTROL'
import json, sys
from pathlib import Path
print(next(a["location"] for a in json.loads(Path(sys.argv[1]).read_bytes())["artifacts"] if a["kind"] == "oci" and a["role"] == "control"))
PY_CONTROL
)
# Only this captured ID is owned; keep the blocker running through both failures.
port_fixture_id=$(docker create --read-only --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8080:8080 --entrypoint python "$control_image" -c 'import time; time.sleep(180)')
[[ "$port_fixture_id" =~ ^[a-f0-9]{64}$ ]]
docker start "$port_fixture_id" >/dev/null
if TLDW_APP_PUBLIC_PORT= "$bundle_dir/start.sh" >"$test_root/default-port.log" 2>&1; then
  echo 'Occupied default port unexpectedly initialized.' >&2; exit 1
fi
[[ ! -e "$TLDW_APP_STATE_DIR/instance" ]]
grep -q 'Available choice: set TLDW_APP_PUBLIC_PORT=' "$test_root/default-port.log"
[[ $(docker inspect --format '{{.State.Running}}' "$port_fixture_id") == true ]]
if TLDW_APP_PUBLIC_PORT=8080 "$bundle_dir/start.sh" >"$test_root/explicit-port.log" 2>&1; then
  echo 'Occupied explicit port unexpectedly initialized.' >&2; exit 1
fi
[[ ! -e "$TLDW_APP_STATE_DIR/instance" ]]
[[ $(docker inspect --format '{{.State.Running}}' "$port_fixture_id") == true ]]
docker rm -f "$port_fixture_id" >/dev/null
port_fixture_id=""
# The explicit alternate retry must complete authenticated readiness and revocation.
"$bundle_dir/start.sh" >"$test_root/first-start.log" 2>&1
grep -q '^ready complete for ' "$test_root/first-start.log"
source_commit=$(python - "$bundle_dir/manifest.json" <<'PY_COMMIT'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["source_commit"])
PY_COMMIT
)
platform=${TLDW_CANDIDATE_PLATFORM:-$(python - "$bundle_dir/manifest.json" <<'PY_PLATFORM'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["platforms"][0])
PY_PLATFORM
)}
first_config_hash=$(sha256sum "$env_file" | awk '{print $1}')
if TLDW_APP_PUBLIC_PORT=8080 "$bundle_dir/start.sh" >"$test_root/established-port.log" 2>&1; then
  echo 'Established origin unexpectedly changed.' >&2; exit 1
fi
[[ "$first_config_hash" == "$(sha256sum "$env_file" | awk '{print $1}')" ]]

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

lifecycle_passed=1
