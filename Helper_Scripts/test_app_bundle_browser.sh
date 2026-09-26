#!/usr/bin/env bash
# Qualification-only private routing fixture; never modifies the signed bundle.
set -Eeuo pipefail
umask 077
if [[ $# -ne 1 ]]; then
  echo 'Usage: test_app_bundle_browser.sh <extracted-bundle-directory>' >&2
  exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
bundle_dir=$(cd "$1" && pwd -P)
evidence_path="$(dirname "$bundle_dir")/browser-evidence.json"
source "$repo_root/.venv/bin/activate"
test_root=$(mktemp -d)
chmod 700 "$test_root"
export TLDW_APP_NO_BROWSER=1
cleanup() {
  local original_exit=$?
  local cleanup_failed=0
  trap - EXIT
  for index in 1 2; do
    local env_file="$test_root/instance-$index/instance/config.env"
    if [[ -f "$env_file" ]]; then
      local project_id
      project_id=$(sed -n 's/^TLDW_PROJECT_ID=//p' "$env_file" 2>/dev/null) || cleanup_failed=1
      if [[ "$project_id" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
        if ! docker compose --project-name "$project_id" --env-file "$env_file" \
          -f "$bundle_dir/compose.yaml" down --volumes >/dev/null 2>&1; then
          cleanup_failed=1
        fi
      else
        cleanup_failed=1
      fi
    fi
  done
  if [[ $cleanup_failed == 0 ]]; then
    rm -rf "$test_root" >/dev/null 2>&1 || cleanup_failed=1
  fi
  if [[ $cleanup_failed != 0 ]]; then
    # Private recovery pointer is deliberately outside the uploaded whitelist.
    printf '%s\n' "$test_root" > "$(dirname "$evidence_path")/.browser-cleanup-recovery" || true
    echo 'Cleanup failed; disposable instance state retained for recovery.' >&2
  fi
  if ! python - "$evidence_path" "$cleanup_failed" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    evidence = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "passed": False, "G2": False, "G4": False,
        "G12": False, "checks": {},
    }
    removed = sys.argv[2] == "0"
    evidence["checks"]["owned_resources_removed"] = {"passed": removed}
    if not removed:
        evidence["passed"] = False
        evidence.setdefault("failure_code", "owned_resources_removed")
    path.write_text(json.dumps(evidence, indent=2) + "\n")
except Exception:
    sys.exit("Cleanup evidence update failed (private details suppressed).")
PY
  then
    cleanup_failed=1
  fi
  if [[ $original_exit == 0 && $cleanup_failed != 0 ]]; then
    original_exit=1
  fi
  exit "$original_exit"
}
trap cleanup EXIT
# Keep failure output public and bounded, including failures before browser launch.
python - "$evidence_path" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1, "passed": False, "planned_setup_complete": False,
    "G2": False, "G4": False, "G12": False,
    "checks": {"paired_signed_start_and_runtime_identity": {"passed": False}},
}) + "\n")
PY
cd "$test_root"
for index in 1 2; do
  TLDW_APP_STATE_DIR="$test_root/instance-$index" \
    TLDW_APP_PUBLIC_PORT="$((18080 + index))" "$bundle_dir/start.sh"
  cp "$test_root/instance-$index/instance/config.env" "$test_root/config-$index.before"
done
second_env="$test_root/instance-2/instance/config.env"
second_project=$(sed -n 's/^TLDW_PROJECT_ID=//p' "$second_env")
compose_second() {
  docker compose --project-name "$second_project" --env-file "$second_env" \
    -f "$bundle_dir/compose.yaml" "$@"
}
# Record only volume identities; state/config and trust must survive recreation.
compose_second exec -T app sh -c 'printf browser-qualification > /app/Databases/wp1-browser-sentinel'
app_id=$(compose_second ps -q app)
docker inspect --format '{{json .Mounts}}' "$app_id" > "$test_root/mounts.before.json"
cat > "$test_root/private-routing.yaml" <<'YAML'
services:
  app:
    environment:
      PORT: "18101"
    expose: ["18101"]
    networks:
      private:
        aliases: [backend-qualification]
    healthcheck:
      test: [CMD, python, -c, "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:18101/internal/ready', timeout=3).status == 200 else 1)"]
  webui:
    environment:
      PORT: "18102"
      TLDW_INTERNAL_API_ORIGIN: http://backend-qualification:18101
    expose: ["18102"]
    networks:
      private:
        aliases: [next-qualification]
    healthcheck:
      test: [CMD, node, -e, "fetch('http://127.0.0.1:18102/favicon.ico').then(r=>process.exit(r.status===200?0:1)).catch(()=>process.exit(1))"]
  gateway:
    environment:
      TLDW_INTERNAL_API_ORIGIN: http://backend-qualification:18101
      TLDW_INTERNAL_WEBUI_ORIGIN: http://next-qualification:18102
YAML
compose_second -f "$test_root/private-routing.yaml" up -d --no-build --wait --wait-timeout 600
compose_second -f "$test_root/private-routing.yaml" exec -T app sh -c \
  'test "$(cat /app/Databases/wp1-browser-sentinel)" = browser-qualification'
for index in 1 2; do
  cmp -s "$test_root/config-$index.before" "$test_root/instance-$index/instance/config.env"
done
python - "$test_root" "$bundle_dir" <<'PY'
import json
# Fixed Docker executable and argv only; no shell or user-controlled command.
import subprocess  # nosec B404
import sys
from pathlib import Path

root, bundle = map(Path, sys.argv[1:])

def config(index):
    return dict(line.split("=", 1) for line in
                (root / f"instance-{index}/instance/config.env").read_text().splitlines()
                if "=" in line)

def containers(index):
    cfg = config(index)
    args = ["docker", "compose", "--project-name", cfg["TLDW_PROJECT_ID"],
            "--env-file", str(root / f"instance-{index}/instance/config.env"),
            "-f", str(bundle / "compose.yaml")]
    ids = subprocess.check_output(args + ["ps", "-q", "app", "webui", "gateway"], text=True).split()
    raw = subprocess.check_output(["docker", "inspect", *ids], text=True)  # nosec B603 B607
    items = json.loads(raw)
    return {item["Config"]["Labels"]["com.docker.compose.service"]: item for item in items}

def env(container):
    return dict(item.split("=", 1) for item in container["Config"]["Env"] if "=" in item)

def require(condition):
    if not condition:
        raise ValueError("runtime_identity_failed")

# No inspection output or condition operands are printed: these contain secrets.
try:
    first, second = containers(1), containers(2)
    require(set(first) == set(second) == {"app", "webui", "gateway"})
    require(config(1)["TLDW_PROJECT_ID"] != config(2)["TLDW_PROJECT_ID"])
    require(first["webui"]["Image"] == second["webui"]["Image"])
    for index, items in [(1, first), (2, second)]:
        cfg = config(index)
        for role, key in [("app", "TLDW_BACKEND_IMAGE"), ("webui", "TLDW_WEBUI_IMAGE"), ("gateway", "TLDW_GATEWAY_IMAGE")]:
            require("@sha256:" in cfg[key] and items[role]["Config"]["Image"] == cfg[key])
        for role in ["app", "webui"]:
            require(not any(items[role]["NetworkSettings"]["Ports"].values()))
            require(env(items[role])["SINGLE_USER_API_KEY"] == cfg["SINGLE_USER_API_KEY"])
            require(env(items[role])["SINGLE_USER_SESSION_COOKIE_NAME"] == cfg["SINGLE_USER_SESSION_COOKIE_NAME"])
            require(env(items[role])["CSRF_COOKIE_NAME"] == cfg["CSRF_COOKIE_NAME"])
        require(env(items["webui"])["TLDW_GATEWAY_HOP_SECRET"] == env(items["gateway"])["TLDW_GATEWAY_HOP_SECRET"] == cfg["TLDW_GATEWAY_HOP_SECRET"])
        require(items["gateway"]["NetworkSettings"]["Ports"]["8080/tcp"] == [{"HostIp": "127.0.0.1", "HostPort": cfg["TLDW_PUBLIC_PORT"]}])
    require(env(second["app"])["PORT"] == "18101")
    require(env(second["webui"])["PORT"] == "18102")
    require(env(second["webui"])["TLDW_INTERNAL_API_ORIGIN"] == "http://backend-qualification:18101")
    require(env(second["gateway"])["TLDW_INTERNAL_API_ORIGIN"] == "http://backend-qualification:18101")
    require(env(second["gateway"])["TLDW_INTERNAL_WEBUI_ORIGIN"] == "http://next-qualification:18102")
    for role, alias in [("app", "backend-qualification"), ("webui", "next-qualification")]:
        require(any(alias in network["Aliases"] for network in second[role]["NetworkSettings"]["Networks"].values()))
    before = json.loads((root / "mounts.before.json").read_text())
    require({(item["Name"], item["Destination"]) for item in before if item["Type"] == "volume"} == {(item["Name"], item["Destination"]) for item in second["app"]["Mounts"] if item["Type"] == "volume"})
    public = {"instances": [{"publicUrl": "http://127.0.0.1:" + config(i)["TLDW_PUBLIC_PORT"],
                             "sessionCookieName": config(i)["SINGLE_USER_SESSION_COOKIE_NAME"],
                             "csrfCookieName": config(i)["CSRF_COOKIE_NAME"]} for i in (1, 2)]}
    (root / "public-input.json").write_text(json.dumps(public))
except Exception:
    sys.exit("Paired runtime identity check failed (private details suppressed).")
PY
browser_exit=0
node "$repo_root/apps/tldw-frontend/scripts/qualify-app-bundle-browser.mjs" \
  "$test_root/public-input.json" "$evidence_path" || browser_exit=$?
python - "$evidence_path" <<'PY_EVIDENCE'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
evidence = json.loads(path.read_text())
evidence["checks"]["paired_signed_start_and_runtime_identity"] = {"passed": True}
path.write_text(json.dumps(evidence, indent=2) + "\n")
PY_EVIDENCE
exit "$browser_exit"
