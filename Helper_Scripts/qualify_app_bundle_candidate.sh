#!/usr/bin/env bash
# CI-only: build, run, and sign a single-platform candidate in a job-local registry.
set -euo pipefail

platform="${TLDW_CANDIDATE_PLATFORM:?Set TLDW_CANDIDATE_PLATFORM}"
evidence_url="${TLDW_EVIDENCE_URL:?Set TLDW_EVIDENCE_URL}"
output_dir="${TLDW_CANDIDATE_OUTPUT:?Set TLDW_CANDIDATE_OUTPUT}"
case "$platform" in
  linux/amd64|linux/arm64) ;;
  *) echo "Unsupported candidate platform: $platform" >&2; exit 2 ;;
esac
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

docker run -d --name tldw-candidate-registry \
  -p 127.0.0.1:5000:5000 registry:2 >/dev/null
cleanup() { docker rm -f tldw-candidate-registry >/dev/null 2>&1 || true; }
trap cleanup EXIT

for role in control backend webui gateway; do
  case "$role" in
    control)
      dockerfile=Dockerfiles/Dockerfile.control
      target=runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        --build-context "trust=$output_dir/trust" -f "$dockerfile" \
        -t "localhost:5000/tldw/$role:candidate" . ;;
    backend)
      dockerfile=Dockerfiles/Dockerfile.prod
      target=runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        -f "$dockerfile" -t "localhost:5000/tldw/$role:candidate" . ;;
    webui)
      dockerfile=Dockerfiles/Dockerfile.webui
      target=managed-runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        -f "$dockerfile" -t "localhost:5000/tldw/$role:candidate" . ;;
    gateway)
      dockerfile=Dockerfiles/Dockerfile.gateway
      target=runtime
      docker buildx build --platform "$platform" --load --target "$target" \
        --build-arg "TLDW_SOURCE_COMMIT=$source_commit" \
        -f "$dockerfile" -t "localhost:5000/tldw/$role:candidate" . ;;
  esac
  revision=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "localhost:5000/tldw/$role:candidate")
  if [[ $revision != "$source_commit" ]]; then
    echo "$role image source revision does not match the clean checkout." >&2
    exit 1
  fi
  docker push "localhost:5000/tldw/$role:candidate" >"$output_dir/$role-push.log"
  digest=$(sed -n 's/.*digest: \(sha256:[0-9a-f]*\).*/\1/p' "$output_dir/$role-push.log" | tail -n 1)
  if [[ ! $digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Could not capture the pushed $role digest." >&2
    exit 1
  fi
  size=$(docker image inspect --format '{{.Size}}' "localhost:5000/tldw/$role:candidate")
  printf '%s\t%s\t%s\n' "$role" "localhost:5000/tldw/$role@$digest" "$size" \
    >>"$output_dir/images.tsv"
done

backend_tag=localhost:5000/tldw/backend:candidate
webui_tag=localhost:5000/tldw/webui:candidate
gateway_tag=localhost:5000/tldw/gateway:candidate
control_tag=localhost:5000/tldw/control:candidate
python_version=$(docker run --rm --platform "$platform" --entrypoint python "$backend_tag" \
  --version | sed 's/^Python //')
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
  'test ! -e /app/apps/tldw-frontend/.next/cache && test -s /app/apps/tldw-frontend/public/favicon.ico'
docker run --rm --platform "$platform" --entrypoint sh "$control_tag" -c \
  'test ! -e /opt/tldw/signing.key && test -s /opt/tldw/trusted-keys/ci-test.pub'

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
Helper_Scripts/test_app_bundle_docker.sh "$output_dir/bundle"

python - "$output_dir/evidence.json" "$platform" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
evidence = json.loads(path.read_text())
for gate in ("G2", "G4", "G10", "G12"):
    evidence["platforms"][sys.argv[2]][gate] = True
path.write_text(json.dumps(evidence, sort_keys=True))
PY
python -m Helper_Scripts.build_app_bundle \
  --artifacts "$output_dir/inventory.json" --evidence "$output_dir/evidence.json" \
  --signing-key "$output_dir/signing.key" --output "$output_dir/bundle"
python -m Helper_Scripts.verify_app_bundle \
  --manifest "$output_dir/bundle/manifest.json" \
  --signature "$output_dir/bundle/manifest.sig" \
  --evidence "$output_dir/evidence.json" \
  --trusted-key-id ci-test --trusted-key-file "$output_dir/trust/ci-test.pub" \
  --platform "$platform"
rm "$output_dir/signing.key"
echo "Qualified local $platform candidate in $output_dir/bundle (job-local registry only)."
