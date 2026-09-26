#!/usr/bin/env bash
# CI-only: build, run, and sign a single-platform candidate in a job-local registry.
set -Eeuo pipefail
umask 077

platform="${TLDW_CANDIDATE_PLATFORM:?Set TLDW_CANDIDATE_PLATFORM}"
evidence_url="${TLDW_EVIDENCE_URL:?Set TLDW_EVIDENCE_URL}"
output_dir="${TLDW_CANDIDATE_OUTPUT:?Set TLDW_CANDIDATE_OUTPUT}"
registry_port="${TLDW_CANDIDATE_REGISTRY_PORT-5000}"
if [[ ! $registry_port =~ ^[1-9][0-9]{0,4}$ ]] || (( registry_port > 65535 )); then
  echo 'Invalid candidate registry port.' >&2
  exit 2
fi
registry="localhost:$registry_port"
case "$platform" in
  linux/amd64|linux/arm64) ;;
  *) echo "Unsupported candidate platform: $platform" >&2; exit 2 ;;
esac
if [[ -e "$output_dir" ]]; then
  echo 'Candidate output must be a new private directory.' >&2
  exit 2
fi
if [[ -n $(git status --porcelain) ]]; then
  echo 'Candidate builds require a clean source checkout.' >&2
  exit 1
fi
source_commit=$(git rev-parse HEAD)

source .venv/bin/activate
mkdir -p "$output_dir" "$output_dir/trust" "$output_dir/bundle"
chmod 700 "$output_dir" "$output_dir/trust"
python - "$output_dir" <<'PY'
import sys
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

root = Path(sys.argv[1])
key = Ed25519PrivateKey.generate()
private = root / "signing.key"
private.write_bytes(key.private_bytes(serialization.Encoding.Raw,
                                      serialization.PrivateFormat.Raw,
                                      serialization.NoEncryption()))
private.chmod(0o600)
(root / "trust" / "ci-test.pub").write_bytes(
    key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
)
PY

registry_id=""
backend_test_id=""
cleanup_registry() {
  if [[ -z "$registry_id" ]]; then return 0; fi
  if [[ ! $registry_id =~ ^[0-9a-f]{64}$ ]]; then
    echo 'Owned registry cleanup refused an invalid identity.' >&2
    return 1
  fi
  if ! docker rm -f -v "$registry_id" >/dev/null 2>&1; then
    printf '%s\n' "$registry_id" > "$output_dir/.registry-cleanup-recovery" || true
    echo 'Registry cleanup failed; owned resource retained for recovery.' >&2
    return 1
  fi
  registry_id=""
}
candidate_exit() {
  local original_exit=$?
  trap - EXIT
  if [[ -n "$backend_test_id" ]]; then
    if [[ ! $backend_test_id =~ ^[0-9a-f]{64}$ ]] || ! docker rm -f -v "$backend_test_id" >/dev/null 2>&1; then
      printf '%s\n' "$backend_test_id" > "$output_dir/.backend-test-cleanup-recovery" || true
      echo 'Backend test cleanup failed; owned resource retained for recovery.' >&2
      if [[ $original_exit == 0 ]]; then original_exit=1; fi
    fi
  fi
  if ! cleanup_registry; then
    if [[ $original_exit == 0 ]]; then original_exit=1; fi
  fi
  if [[ $original_exit != 0 ]]; then
    if [[ -f "$output_dir/evidence.json" ]]; then
      if ! python - "$output_dir/evidence.json" <<'PY_FAILED_EVIDENCE'
import json
import sys
from pathlib import Path
try:
    path = Path(sys.argv[1])
    evidence = json.loads(path.read_text())
    for item in evidence["platforms"].values():
        item.update({gate: False for gate in ("G2", "G4", "G10", "G12")})
    path.write_text(json.dumps(evidence, sort_keys=True))
except Exception:
    sys.exit("Failed-run evidence invalidation failed (private details suppressed).")
PY_FAILED_EVIDENCE
      then
        # A corrupt evidence file is not eligible for the public allowlist.
        rm -f "$output_dir/evidence.json" >/dev/null 2>&1 || echo 'Public evidence removal failed.' >&2
      fi
    fi
    # Only generated public signature/manifest files are invalidated. Keys,
    # inventory, identities and private recovery state remain recoverable.
    rm -f "$output_dir/bundle/manifest.json" "$output_dir/bundle/manifest.sig" >/dev/null 2>&1 || echo 'Public candidate signature removal failed.' >&2
  fi
  exit "$original_exit"
}
trap candidate_exit EXIT
# Capture ownership before start: port-binding failure must still be removable.
if ! registry_id=$(docker create --name tldw-candidate-registry \
  -p "127.0.0.1:$registry_port:5000" registry:2 2>"$output_dir/.registry-create.log"); then
  registry_id=""
  echo 'Candidate registry creation failed (private details suppressed).' >&2
  exit 1
fi
if [[ ! $registry_id =~ ^[0-9a-f]{64}$ ]]; then
  echo 'Candidate registry identity invalid.' >&2
  exit 1
fi
if ! docker start "$registry_id" >"$output_dir/.registry-start.log" 2>&1; then
  echo 'Candidate registry start failed (private details suppressed).' >&2
  exit 1
fi

for role in control backend webui gateway; do
  echo "Candidate image build started: $role."
  case "$role" in
    control)
      dockerfile=Dockerfiles/Dockerfile.control
      target=runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        --build-context "trust=$output_dir/trust" -f "$dockerfile" \
        -t "$registry/tldw/$role:candidate" . ;;
    backend)
      dockerfile=Dockerfiles/Dockerfile.prod
      target=runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        -f "$dockerfile" -t "$registry/tldw/$role:candidate" . ;;
    webui)
      dockerfile=Dockerfiles/Dockerfile.webui
      target=managed-runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        -f "$dockerfile" -t "$registry/tldw/$role:candidate" . ;;
    gateway)
      dockerfile=Dockerfiles/Dockerfile.gateway
      target=runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        -f "$dockerfile" -t "$registry/tldw/$role:candidate" . ;;
  esac
  revision=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$registry/tldw/$role:candidate")
  if [[ $revision != "$source_commit" ]]; then
    echo "$role image source revision does not match the clean checkout." >&2
    exit 1
  fi
  docker push "$registry/tldw/$role:candidate" >"$output_dir/$role-push.log"
  digest=$(sed -n 's/.*digest: \(sha256:[0-9a-f]*\).*/\1/p' "$output_dir/$role-push.log" | tail -n 1)
  if [[ ! $digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Could not capture the pushed $role digest." >&2
    exit 1
  fi
  size=$(docker image inspect --format '{{.Size}}' "$registry/tldw/$role:candidate")
  printf '%s\t%s\t%s\n' "$role" "$registry/tldw/$role@$digest" "$size" \
    >>"$output_dir/images.tsv"
  echo "Candidate image build and local digest capture passed: $role."
done

backend_tag="$registry/tldw/backend:candidate"
webui_tag="$registry/tldw/webui:candidate"
gateway_tag="$registry/tldw/gateway:candidate"
control_tag="$registry/tldw/control:candidate"
python_version=$(docker run --rm --platform "$platform" --entrypoint python "$backend_tag" \
  --version | sed 's/^Python //')
docker run --rm --platform "$platform" --entrypoint python "$backend_tag" \
  -c 'import tldw_profile_core'
node_version=$(docker run --rm --platform "$platform" --entrypoint node "$webui_tag" \
  --version | sed 's/^v//')
gateway_node_version=$(docker run --rm --platform "$platform" --entrypoint node "$gateway_tag" \
  --version | sed 's/^v//')
[[ $python_version =~ ^3\.12\.[0-9]+$ ]]
[[ $node_version =~ ^24\.[0-9]+\.[0-9]+$ ]]
[[ $gateway_node_version == "$node_version" ]]
docker run --rm --platform "$platform" --entrypoint sh "$backend_tag" -c \
  'test ! -e /app/tldw_Server_API/Config_Files/.env && test ! -e /app/Databases/users.db'
docker run --rm --platform "$platform" --entrypoint sh "$webui_tag" -c \
  'test ! -e /app/apps/tldw-frontend/.next/cache && test -s /app/apps/tldw-frontend/public/favicon.ico &&
   test -r /app/Docs/Published/API-related/AuthNZ-API-Guide.md &&
   grep -q "^# AuthNZ API Guide$" /app/Docs/Published/API-related/AuthNZ-API-Guide.md'
docker run --rm --platform "$platform" --entrypoint python \
  --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" "$control_tag" -c \
  'from pathlib import Path; assert not Path("/opt/tldw/signing.key").exists(); assert len(Path("/opt/tldw/trusted-keys/ci-test.pub").read_bytes()) == 32'

# Exercise focused security tests on the actual built backend dependencies.
# Narrow read-only mounts supply the Compose contract and the excluded setup
# test only; production source and MCP tests remain from the built image.
echo 'Focused built-backend qualification started.'
if ! backend_test_id=$(docker create --platform "$platform" --entrypoint sh \
  --mount "type=bind,source=$(pwd)/Dockerfiles/app-bundle,target=/app/Dockerfiles/app-bundle,readonly" \
  --mount "type=bind,source=$(pwd)/tldw_Server_API/tests/Setup/test_managed_gateway_setup.py,target=/app/tldw_Server_API/tests/Setup/test_managed_gateway_setup.py,readonly" \
  --env PYTHONPATH=/app --env TEST_MODE=false \
  --env MCP_AUDIT_LOG_FILE=/app/Databases/mcp-audit.log \
  --env SINGLE_USER_API_KEY=ci-managed-dummy-key-with-at-least-32-characters \
  "$backend_tag" -c '
    set -eu
    python -m venv --system-site-packages /tmp/qualification-venv
    . /tmp/qualification-venv/bin/activate
    python -m pip install pytest==9.0.3 pytest-asyncio==1.3.0 hypothesis==6.138.2
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    python -m pytest -c /dev/null --confcutdir=tldw_Server_API/app/core/MCP_unified/tests \
      -p pytest_asyncio.plugin --asyncio-mode=auto -p no:cacheprovider -q \
      tldw_Server_API/app/core/MCP_unified/tests/test_managed_gateway_ingress.py
    python -m pytest -c /dev/null --confcutdir=tldw_Server_API/tests/Setup \
      -p pytest_asyncio.plugin --asyncio-mode=auto -p no:cacheprovider -q \
      tldw_Server_API/tests/Setup/test_managed_gateway_setup.py
  ' 2>"$output_dir/.backend-test-create.log"); then
  backend_test_id=""
  echo 'Backend qualification container creation failed (private details suppressed).' >&2
  exit 1
fi
[[ $backend_test_id =~ ^[0-9a-f]{64}$ ]]
if ! docker start -a "$backend_test_id" >"$output_dir/.backend-qualification.log" 2>&1; then
  echo 'Focused built-backend qualification failed (private details suppressed).' >&2
  exit 1
fi
if [[ $(docker inspect --format '{{.State.ExitCode}}' "$backend_test_id") != 0 ]]; then
  echo 'Focused built-backend qualification failed (private details suppressed).' >&2
  exit 1
fi
if ! docker rm -v "$backend_test_id" >/dev/null 2>&1; then
  echo 'Backend test cleanup failed (private details suppressed).' >&2
  exit 1
fi
backend_test_id=""
echo 'Focused built-backend MCP and setup qualification passed with owned cleanup.'

export TLDW_CANDIDATE_PYTHON_VERSION="$python_version"
export TLDW_CANDIDATE_NODE_VERSION="$node_version"
python - "$output_dir" "$platform" "$evidence_url" <<'PY'
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root, platform, link = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
arch = platform.split("/")[1]
images = []
for line in (root / "images.tsv").read_text().splitlines():
    role, ref, size = line.split("\t")
    digest = ref.rsplit("@sha256:", 1)[1]
    images.append({
        "id": f"{role}-{arch}", "kind": "oci", "role": role,
        "platform": platform, "source_commit": commit,
        "location": ref, "image_digest": f"sha256:{digest}", "sha256": digest,
        "size_bytes": int(size), "installed_size_bytes": int(size),
    })
control = next(image["location"] for image in images if image["role"] == "control")
inventory = {
    "version": "0.2.0", "source_commit": commit,
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "channel": "ci-candidate", "signer_id": "ci-test", "platforms": [platform],
    "control_image": control, "artifact_base_url": link.rstrip("/") + "/bundle",
    "compatibility": {
        "min_launcher": "0.1.0", "python_version": os.environ["TLDW_CANDIDATE_PYTHON_VERSION"],
        "node_version": os.environ["TLDW_CANDIDATE_NODE_VERSION"],
        "backend_generation": 1, "browser_generation": 1,
        "allowed_upgrade_sources": ["0.1.0"], "components": ["core"],
    },
    "artifacts": images,
    "dependencies": {"lock_digests": {
        "frontend": hashlib.sha256(Path("apps/bun.lock").read_bytes()).hexdigest(),
        "gateway": hashlib.sha256(Path("Dockerfiles/gateway/bun.lock").read_bytes()).hexdigest(),
        "backend": hashlib.sha256(Path("pyproject.toml").read_bytes()).hexdigest(),
    }},
    "data": {"inventory_schema": 1, "migration_generation": 1,
             "rollback_eligible": True,
             "component_catalog_digest": hashlib.sha256(
                 Path("tldw_Server_API/Config_Files/config.txt").read_bytes()).hexdigest()},
}
evidence = {"source_commit": commit, "platforms": {platform: {
    "G2": False, "G4": False, "G10": False, "G12": False,
    "python_version": os.environ["TLDW_CANDIDATE_PYTHON_VERSION"],
    "node_version": os.environ["TLDW_CANDIDATE_NODE_VERSION"], "link": link,
}}}
(root / "inventory.json").write_text(json.dumps(inventory, sort_keys=True))
(root / "evidence.json").write_text(json.dumps(evidence, sort_keys=True))
PY

python -m Helper_Scripts.build_app_bundle \
  --artifacts "$output_dir/inventory.json" --evidence "$output_dir/evidence.json" \
  --signing-key "$output_dir/signing.key" --output "$output_dir/bundle"
echo 'Extracted Docker lifecycle qualification started.'
Helper_Scripts/test_app_bundle_docker.sh "$output_dir/bundle"
echo 'Paired browser transport qualification started.'
Helper_Scripts/test_app_bundle_browser.sh "$output_dir/bundle"

# No qualified evidence or signature exists until EVERY owned resource is gone.
cleanup_registry
echo 'Owned registry and fixture cleanup passed; closing bounded evidence.'
node --input-type=module - "$output_dir" "$source_commit" "$platform" <<'JS_EVIDENCE'
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { completeEvidence } from './apps/tldw-frontend/scripts/qualify-app-bundle-browser.mjs'
const [root, commit, platform] = process.argv.slice(2)
const path = join(root, 'evidence.json')
try {
  const browser = JSON.parse(readFileSync(join(root, 'browser-evidence.json'), 'utf8'))
  const lifecycle = JSON.parse(readFileSync(join(root, 'lifecycle-evidence.json'), 'utf8'))
  for (const item of [browser, lifecycle]) {
    if (item.schema_version !== 1 || item.source_commit !== commit || item.platform !== platform || item.passed !== true) throw new Error('fixture_evidence')
  }
  if (browser.planned_setup_complete !== false || browser.setup_scope !== 'managed_connection_and_initial_wizard_only' ||
      browser.checks.owned_resources_removed?.passed !== true || browser.checks.paired_signed_start_and_runtime_identity?.passed !== true ||
      lifecycle.owned_resources_removed !== true) throw new Error('fixture_evidence')
  completeEvidence(browser)
  const lifecycleChecks = ['signed_start', 'ready', 'public_assets', 'published_documentation', 'cookie_auth', 'private_isolation', 'restart_persistence', 'tamper_refused', 'installer_authenticated_readiness', 'probe_session_revoked', 'occupied_default_offer', 'occupied_explicit_retry', 'established_origin_refused']
  if (lifecycleChecks.some(name => lifecycle.checks[name]?.passed !== true) || Object.values(lifecycle.checks).some(check => check.passed !== true)) throw new Error('fixture_evidence')
  const evidence = JSON.parse(readFileSync(path, 'utf8'))
  if (evidence.source_commit !== commit || !evidence.platforms?.[platform]) throw new Error('fixture_evidence')
  Object.assign(evidence.platforms[platform], { G2: true, G4: true, G12: false,
    setup_scope: browser.setup_scope, planned_setup_complete: false })
  writeFileSync(path, JSON.stringify(evidence, null, 2) + '\n')
} catch {
  console.error('Candidate fixture evidence closure failed (private details suppressed).')
  process.exitCode = 1
}
JS_EVIDENCE
python - "$output_dir" "$platform" <<'PY_TRUST'
import json
import sys
from pathlib import Path
from tldw_Server_API.app.core.Release.manifest import verify_artifact, verify_manifest
root, platform = Path(sys.argv[1]), sys.argv[2]
try:
    bundle = root / "bundle"
    manifest = verify_manifest((bundle / "manifest.json").read_bytes(),
        (bundle / "manifest.sig").read_bytes(),
        {"ci-test": (root / "trust/ci-test.pub").read_bytes()}, platform=platform)
    for artifact in manifest.artifacts:
        if artifact.kind == "file":
            verify_artifact(bundle / artifact.path, artifact)
    path = root / "evidence.json"
    evidence = json.loads(path.read_text())
    evidence["platforms"][platform]["G10"] = True
    path.write_text(json.dumps(evidence, sort_keys=True))
except Exception:
    sys.exit("Candidate local artifact trust failed (private details suppressed).")
PY_TRUST
python -m Helper_Scripts.build_app_bundle \
  --artifacts "$output_dir/inventory.json" --evidence "$output_dir/evidence.json" \
  --signing-key "$output_dir/signing.key" --output "$output_dir/bundle"
if python -m Helper_Scripts.verify_app_bundle \
  --manifest "$output_dir/bundle/manifest.json" \
  --signature "$output_dir/bundle/manifest.sig" \
  --evidence "$output_dir/evidence.json" \
  --trusted-key-id ci-test --trusted-key-file "$output_dir/trust/ci-test.pub" \
  --platform "$platform"; then
  echo 'Incomplete candidate unexpectedly passed the promotion gate.' >&2
  exit 1
fi
# Verify the final exact signature and copied helper bytes after registry removal.
python - "$output_dir" "$platform" <<'PY_FINAL_TRUST'
import sys
from pathlib import Path
from tldw_Server_API.app.core.Release.manifest import verify_artifact, verify_manifest
root, platform = Path(sys.argv[1]), sys.argv[2]
try:
    bundle = root / "bundle"
    manifest = verify_manifest((bundle / "manifest.json").read_bytes(),
        (bundle / "manifest.sig").read_bytes(),
        {"ci-test": (root / "trust/ci-test.pub").read_bytes()}, platform=platform)
    for artifact in manifest.artifacts:
        if artifact.kind == "file":
            verify_artifact(bundle / artifact.path, artifact)
except Exception:
    sys.exit("Final candidate artifact trust failed (private details suppressed).")
PY_FINAL_TRUST
rm "$output_dir/signing.key"
echo "Built provisional local $platform candidate in $output_dir/bundle; G2/G4 passed bounded qualification; G12 remains open."
