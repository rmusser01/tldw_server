"""Candidate signing and qualification must not promote partial image sets."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from Helper_Scripts.build_app_bundle import build_candidate
from Helper_Scripts.verify_app_bundle import candidate_is_promotable

PLATFORMS = ("linux/amd64", "linux/arm64")


@pytest.fixture
def signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


@pytest.fixture
def inventory() -> dict[str, object]:
    source_commit = "a" * 40
    artifacts = []
    for platform, letter in (("linux/amd64", "b"), ("linux/arm64", "c")):
        for role, role_letter in (("backend", "1"), ("webui", "2"), ("gateway", "3")):
            digest = letter + role_letter * 63
            artifacts.append(
                {
                    "id": f"{role}-{platform.split('/')[1]}",
                    "kind": "oci",
                    "role": role,
                    "platform": platform,
                    "source_commit": source_commit,
                    "location": f"localhost:5000/tldw/{role}@sha256:{digest}",
                    "image_digest": f"sha256:{digest}",
                    "sha256": digest,
                    "size_bytes": 1024,
                    "installed_size_bytes": 4096,
                }
            )
        artifacts.append(
            {
                "id": f"control-{platform.split('/')[1]}",
                "kind": "oci",
                "role": "control",
                "platform": platform,
                "source_commit": source_commit,
                "location": "localhost:5000/tldw/control@sha256:" + "d" * 64,
                "image_digest": "sha256:" + "d" * 64,
                "sha256": "d" * 64,
                "size_bytes": 512,
                "installed_size_bytes": 2048,
            }
        )
    return {
        "version": "0.2.0",
        "source_commit": source_commit,
        "created_at": "2026-09-25T00:00:00Z",
        "channel": "ci-candidate",
        "signer_id": "ci-test",
        "platforms": list(PLATFORMS),
        "control_image": "localhost:5000/tldw/control@sha256:" + "d" * 64,
        "artifact_base_url": "https://example.invalid/ci-bundle",
        "compatibility": {
            "min_launcher": "0.1.0",
            "python_version": "3.12.7",
            "node_version": "24.6.0",
            "backend_generation": 1,
            "browser_generation": 1,
            "allowed_upgrade_sources": ["0.1.0"],
            "components": ["core"],
        },
        "artifacts": artifacts,
        "dependencies": {"lock_digests": {"gateway": "e" * 64}},
        "data": {
            "inventory_schema": 1,
            "migration_generation": 1,
            "rollback_eligible": True,
            "component_catalog_digest": "f" * 64,
        },
    }


@pytest.fixture
def evidence(inventory: dict[str, object]) -> dict[str, object]:
    return {
        "source_commit": inventory["source_commit"],
        "platforms": {
            platform: {
                "G2": True,
                "G4": True,
                "G10": True,
                "G12": True,
                "python_version": "3.12.7",
                "node_version": "24.6.0",
                "link": "https://example.invalid/ci-evidence",
            }
            for platform in PLATFORMS
        },
    }


def _build(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> tuple[Path, dict[str, bytes]]:
    output = tmp_path / "bundle"
    build_candidate(inventory, evidence, signing_key, output)
    public = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return output, {"ci-test": public}


def test_complete_local_candidate_is_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)
    assert b"__CONTROL_IMAGE_DIGEST__" not in (output / "start.sh").read_bytes()
    assert not (output / "signing.key").exists()


def test_missing_webui_image_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["artifacts"] = [
        artifact
        for artifact in inventory["artifacts"]
        if not (artifact["platform"] == "linux/amd64" and artifact["role"] == "webui")
    ]
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_mixed_source_commits_are_rejected_before_signing(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["artifacts"][0]["source_commit"] = "0" * 40

    with pytest.raises(ValueError, match="source commit"):
        _build(tmp_path, inventory, evidence, signing_key)


def test_runtime_patch_mismatch_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    evidence["platforms"]["linux/arm64"]["node_version"] = "24.0.0"
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_unsupported_runtime_family_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["compatibility"]["node_version"] = "20.19.0"
    for platform in PLATFORMS:
        evidence["platforms"][platform]["node_version"] = "20.19.0"
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_missing_arm64_evidence_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    del evidence["platforms"]["linux/arm64"]
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_tampered_helper_digest_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    output, keys = _build(tmp_path, inventory, evidence, signing_key)
    (output / "start.sh").write_text((output / "start.sh").read_text() + "\n# tampered\n")

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_corrupted_image_digest_is_rejected_before_signing(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["artifacts"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="digest"):
        _build(tmp_path, inventory, evidence, signing_key)


@pytest.mark.parametrize("port", ["", "0", "65536", "05000", "-1", "5000x", "1;echo leak", "localhost:5000"])
def test_registry_port_rejected_before_resources(tmp_path: Path, port: str) -> None:
    """Invalid loopback publication ports never create keys or Docker resources."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["bash", str(root / "Helper_Scripts/qualify_app_bundle_candidate.sh")],
        cwd=root,
        env={
            **os.environ,
            "TLDW_CANDIDATE_PLATFORM": "linux/arm64",
            "TLDW_EVIDENCE_URL": "https://example.invalid/public",
            "TLDW_CANDIDATE_OUTPUT": str(tmp_path / "output"),
            "TLDW_CANDIDATE_REGISTRY_PORT": port,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "Invalid candidate registry port."
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failed,original,expected", [(False, 0, 0), (True, 0, 1), (True, 7, 7)])
def test_lifecycle_smoke_cleanup_removes_only_owned_volumes_or_retains_state(
    tmp_path: Path,
    failed: bool,
    original: int,
    expected: int,
) -> None:
    """Lifecycle cleanup must fail closed, keep private recovery data and errors."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[3]
    shell = (root / "Helper_Scripts/test_app_bundle_docker.sh").read_text()
    cleanup = shell[shell.index("cleanup() {") : shell.index("trap cleanup EXIT")]
    state = tmp_path / "owned-state"
    state.mkdir()
    env_file = state / "config.env"
    env_file.write_text("TLDW_PROJECT_ID=owned-project\nSECRET=private-fixture\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    calls = tmp_path / "calls"
    docker.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{calls}"\necho secret-error >&2\nexit {int(failed)}\n')
    docker.chmod(0o700)
    harness = tmp_path / "cleanup.sh"
    evidence = tmp_path / "lifecycle-evidence.json"
    harness.write_text(
        f"""set -Eeuo pipefail
umask 077
test_root=$1
bundle_dir=$2
env_file=$3
project_id=owned-project
evidence_path=$4
lifecycle_passed=1
source_commit={'a' * 40}
platform=linux/arm64
{cleanup}
trap cleanup EXIT
exit "$5"
"""
    )
    result = subprocess.run(
        ["bash", str(harness), str(state), str(tmp_path), str(env_file), str(evidence), str(original)],
        env={**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected
    assert state.exists() is failed
    assert "secret-error" not in result.stderr
    assert (
        calls.read_text().strip()
        == f"compose --project-name owned-project --env-file {env_file} -f {tmp_path}/compose.yaml down --volumes"
    )
    import json

    public = json.loads(evidence.read_text())
    assert public["passed"] is (not failed and original == 0)
    assert public["owned_resources_removed"] is (not failed)


@pytest.mark.parametrize("failed,original,expected", [(False, 0, 0), (True, 0, 1), (True, 7, 7)])
def test_registry_cleanup_owns_id_and_anonymous_storage(
    tmp_path: Path,
    failed: bool,
    original: int,
    expected: int,
) -> None:
    """Registry cleanup removes its captured ID, never a colliding fixed name."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[3]
    shell = (root / "Helper_Scripts/qualify_app_bundle_candidate.sh").read_text()
    cleanup = shell[shell.index("cleanup_registry() {") : shell.index("trap candidate_exit EXIT")]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{calls}"\necho secret-error >&2\nexit {int(failed)}\n')
    docker.chmod(0o700)
    harness = tmp_path / "cleanup.sh"
    harness.write_text(
        f"""set -Eeuo pipefail
umask 077
output_dir=$1
registry_id={'a' * 64}
backend_test_id=""
{cleanup}
trap candidate_exit EXIT
exit "$2"
"""
    )
    result = subprocess.run(
        ["bash", str(harness), str(tmp_path), str(original)],
        env={**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected
    assert calls.read_text().strip() == "rm -f -v " + "a" * 64
    assert "secret-error" not in result.stderr
    assert (tmp_path / ".registry-cleanup-recovery").exists() is failed


@pytest.mark.parametrize("change", ["", "missing", "false", "source", "platform", "schema", "lifecycle", "cleanup"])
def test_candidate_gate_requires_complete_matching_cleaned_fixture_evidence(
    tmp_path: Path,
    change: str,
) -> None:
    """No signed qualified gates may be derived from partial or mismatched evidence."""
    import json
    import subprocess

    root = Path(__file__).resolve().parents[3]
    shell = (root / "Helper_Scripts/qualify_app_bundle_candidate.sh").read_text()
    marker = 'node --input-type=module - "$output_dir" "$source_commit" "$platform" <<\'JS_EVIDENCE\'\n'
    assert marker in shell, "candidate fixture evidence closure is missing"
    embedded = shell.split(marker, 1)[1].split("\nJS_EVIDENCE", 1)[0]
    required = json.loads(
        subprocess.check_output(
            [
                "node",
                "--input-type=module",
                "-e",
                "import { REQUIRED_CHECKS } from './apps/tldw-frontend/scripts/qualify-app-bundle-browser.mjs'; console.log(JSON.stringify(REQUIRED_CHECKS))",
            ],
            cwd=root,
            text=True,
        )
    )
    browser = {
        "schema_version": 1,
        "passed": True,
        "planned_setup_complete": False,
        "setup_scope": "managed_connection_and_initial_wizard_only",
        "source_commit": "a" * 40,
        "platform": "linux/arm64",
        "checks": {name: {"passed": True} for name in required},
    }
    browser["checks"].update(
        {"owned_resources_removed": {"passed": True}, "paired_signed_start_and_runtime_identity": {"passed": True}}
    )
    lifecycle = {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "platform": "linux/arm64",
        "passed": True,
        "owned_resources_removed": True,
        "checks": {
            name: {"passed": True}
            for name in (
                "signed_start",
                "ready",
                "public_assets",
                "published_documentation",
                "cookie_auth",
                "private_isolation",
                "restart_persistence",
                "tamper_refused",
            )
        },
    }
    if change == "missing":
        del browser["checks"]["notification_sse_cancel_2"]
    if change == "false":
        browser["checks"]["cookie_mcp_websocket_1"]["passed"] = False
    if change == "source":
        browser["source_commit"] = "b" * 40
    if change == "platform":
        browser["platform"] = "linux/amd64"
    if change == "schema":
        browser["schema_version"] = 2
    if change == "lifecycle":
        del lifecycle["checks"]["restart_persistence"]
    if change == "cleanup":
        lifecycle["owned_resources_removed"] = False
    (tmp_path / "browser-evidence.json").write_text(json.dumps(browser))
    (tmp_path / "lifecycle-evidence.json").write_text(json.dumps(lifecycle))
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "platforms": {"linux/arm64": {"G2": False, "G4": False, "G10": False, "G12": False}},
            }
        )
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-", str(tmp_path), "a" * 40, "linux/arm64"],
        input=embedded,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if change == "" else 1)
    evidence = json.loads((tmp_path / "evidence.json").read_text())["platforms"]["linux/arm64"]
    assert evidence["G2"] is (change == "")
    assert evidence["G4"] is (change == "")
    assert evidence["G12"] is False


@pytest.mark.parametrize("mode,expected", [("normal", 0), ("collision", 1), ("start", 1), ("cleanup", 1)])
def test_local_registry_port_refs_and_partial_ownership(
    tmp_path: Path,
    mode: str,
    expected: int,
) -> None:
    """The selected local port is used for every ref; create/start failures are owned safely."""
    import json
    import os
    import subprocess

    root = Path(__file__).resolve().parents[3]
    shell = (root / "Helper_Scripts/qualify_app_bundle_candidate.sh").read_text()
    prefix = shell.split("backend_tag=", 1)[0]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.jsonl"
    git = bin_dir / "git"
    git.write_text('#!/bin/bash\nif [[ $1 == rev-parse ]]; then printf "%s\\n" ' + "a" * 40 + "; fi\n")
    git.chmod(0o700)
    docker = bin_dir / "docker"
    docker.write_text(
        f"""#!{os.sys.executable}
import json, sys
from pathlib import Path
args=sys.argv[1:]
with Path({str(calls)!r}).open('a') as log: log.write(json.dumps(args)+'\\n')
mode={mode!r}
if args[0]=='create':
    if mode=='collision': sys.exit(1)
    print({'b' * 64!r})
elif args[0]=='start' and mode=='start': sys.exit(1)
elif args[0]=='rm' and mode=='cleanup': sys.exit(1)
elif args[:2]==['image','inspect']:
    print('1024' if args[3]=='{{{{.Size}}}}' else {'a' * 40!r})
elif args[0]=='push': print('digest: sha256:' + 'c'*64)
"""
    )
    docker.chmod(0o700)
    harness = tmp_path / "candidate-prefix.sh"
    harness.write_text(prefix)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(harness)],
        cwd=root,
        env={
            **os.environ,
            "PATH": str(bin_dir) + ":" + os.environ["PATH"],
            "TLDW_CANDIDATE_PLATFORM": "linux/arm64",
            "TLDW_EVIDENCE_URL": "https://example.invalid/public",
            "TLDW_CANDIDATE_OUTPUT": str(output),
            "TLDW_CANDIDATE_REGISTRY_PORT": "15000",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected, result.stderr
    actual = [json.loads(line) for line in calls.read_text().splitlines()]
    assert actual[0] == ["create", "--name", "tldw-candidate-registry", "-p", "127.0.0.1:15000:5000", "registry:2"]
    removed = [args for args in actual if args[0] == "rm"]
    assert removed == ([] if mode == "collision" else [["rm", "-f", "-v", "b" * 64]])
    if mode in ("normal", "cleanup"):
        refs = [arg for args in actual for arg in args if arg.startswith("localhost:")]
        assert refs and all(ref.startswith("localhost:15000/tldw/") for ref in refs)
        assert all(
            line.split("\t")[1].startswith("localhost:15000/tldw/")
            for line in (output / "images.tsv").read_text().splitlines()
        )
    assert (output / ".registry-cleanup-recovery").exists() is (mode == "cleanup")


@pytest.mark.parametrize("cleanup_failed,original", [(False, 7), (True, 0)])
def test_candidate_failure_invalidates_qualified_evidence_and_signature(
    tmp_path: Path,
    cleanup_failed: bool,
    original: int,
) -> None:
    """A failed run must never leave a signed success artifact in the allowlist."""
    import json
    import os
    import subprocess

    root = Path(__file__).resolve().parents[3]
    shell = (root / "Helper_Scripts/qualify_app_bundle_candidate.sh").read_text()
    cleanup = shell[shell.index("cleanup_registry() {") : shell.index("trap candidate_exit EXIT")]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f"#!/bin/bash\nexit {int(cleanup_failed)}\n")
    docker.chmod(0o700)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"platforms": {"linux/arm64": dict.fromkeys(("G2", "G4", "G10", "G12"), True)}}))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("public-provisional")
    (bundle / "manifest.sig").write_text("public-signature")
    (tmp_path / "signing.key").write_text("private-recovery-fixture")
    harness = tmp_path / "exit.sh"
    harness.write_text(
        f"""set -Eeuo pipefail
umask 077
output_dir=$1
registry_id={'a' * 64}
backend_test_id=""
{cleanup}
trap candidate_exit EXIT
exit "$2"
"""
    )
    result = subprocess.run(
        ["bash", str(harness), str(tmp_path), str(original)],
        env={**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (original or 1)
    assert not any(json.loads(evidence.read_text())["platforms"]["linux/arm64"].values())
    assert not (bundle / "manifest.sig").exists()
    assert not (bundle / "manifest.json").exists()
    assert (tmp_path / "signing.key").read_text() == "private-recovery-fixture"


def test_candidate_refuses_existing_output_without_changing_recovery_state(tmp_path: Path) -> None:
    """Each run owns a fresh output directory and cannot delete another run's artifacts."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[3]
    sentinel = tmp_path / "signing.key"
    sentinel.write_text("previous-private-recovery-fixture")
    result = subprocess.run(
        ["bash", str(root / "Helper_Scripts/qualify_app_bundle_candidate.sh")],
        cwd=root,
        env={
            **os.environ,
            "TLDW_CANDIDATE_PLATFORM": "linux/arm64",
            "TLDW_CANDIDATE_OUTPUT": str(tmp_path),
            "TLDW_EVIDENCE_URL": "https://example.invalid/public",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "Candidate output must be a new private directory."
    assert sentinel.read_text() == "previous-private-recovery-fixture"


def test_built_backend_qualification_supplies_excluded_setup_test_readonly(tmp_path: Path) -> None:
    """A clean runtime layout needs one test mount without replacing built source."""
    import json
    import os
    import shutil
    import subprocess

    root = Path(__file__).resolve().parents[3]
    excluded = "tldw_Server_API/tests/"
    ignore_rules = (root / ".dockerignore").read_text().splitlines()
    assert excluded in ignore_rules
    assert not any(rule.startswith("!" + excluded) for rule in ignore_rules)
    dockerfile = (root / "Dockerfiles/Dockerfile.prod").read_text()
    assert "COPY --chown=appuser:appuser tldw_Server_API /app/tldw_Server_API" in dockerfile
    setup_test = "tldw_Server_API/tests/Setup/test_managed_gateway_setup.py"
    mcp_test = "tldw_Server_API/app/core/MCP_unified/tests/test_managed_gateway_ingress.py"
    mcp_support = "tldw_Server_API/app/core/MCP_unified/tests/support.py"
    production = "tldw_Server_API/app/api/v1/endpoints/setup.py"
    image = tmp_path / "clean-runtime"
    # Reproduce exactly the relevant COPY/exclusion distinction, without a build.
    for relative in (setup_test, mcp_test, mcp_support, production):
        if relative.startswith(excluded):
            continue
        assert not any(
            relative.startswith(rule.rstrip("/") + "/")
            for rule in ignore_rules
            if rule.endswith("/") and not rule.startswith("#")
        )
        target = image / "app" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    assert not (image / "app" / setup_test).exists()
    assert (image / "app" / mcp_test).is_file()
    assert (image / "app" / mcp_support).is_file()

    shell = (root / "Helper_Scripts/qualify_app_bundle_candidate.sh").read_text()
    block = shell[
        shell.index("# Exercise focused security tests") : shell.index("export TLDW_CANDIDATE_PYTHON_VERSION")
    ]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "mounts.json"
    docker = bin_dir / "docker"
    docker.write_text(
        f"""#!{os.sys.executable}
import json, sys
from pathlib import Path
args=sys.argv[1:]
image=Path({str(image)!r})
root=Path({str(root)!r})
setup={setup_test!r}
mcp={mcp_test!r}
production={production!r}
if args[0]=='create':
    environment=dict(args[i+1].split('=',1) for i,arg in enumerate(args) if arg=='--env')
    if environment.get('MCP_AUDIT_LOG_FILE') != '/app/Databases/mcp-audit.log': sys.exit(1)
    if environment.get('MCP_AUDIT_ENABLED','true') != 'true' or '--user' in args: sys.exit(1)
    mounts=[]
    for i, arg in enumerate(args):
        if arg=='--mount':
            values=args[i+1].split(',')
            fields=dict(item.split('=',1) for item in values if '=' in item)
            if 'readonly' not in values: sys.exit(1)
            mounts.append(fields)
    expected={{'/app/Dockerfiles/app-bundle': str(root/'Dockerfiles/app-bundle'), '/app/'+setup: str(root/setup)}}
    if {{item['target']:item['source'] for item in mounts}} != expected: sys.exit(1)
    def readable(target):
        for mount in mounts:
            destination=mount['target']
            if target==destination or target.startswith(destination+'/'):
                return (Path(mount['source'])/target.removeprefix(destination).lstrip('/')).is_file()
        return (image/target.lstrip('/')).is_file()
    if not all(readable('/app/'+item) for item in (setup,mcp,production)): sys.exit(1)
    if not readable('/app/Dockerfiles/app-bundle/compose.yaml'): sys.exit(1)
    if not (image/'app'/production).is_file(): sys.exit(1)
    command=args[args.index('-c')+1]
    if setup not in command or mcp not in command: sys.exit(1)
    Path({str(capture)!r}).write_text(json.dumps(mounts))
    print({'d' * 64!r})
elif args[0]=='inspect': print('0')
elif args[0] not in ('start','rm'): sys.exit(1)
"""
    )
    docker.chmod(0o700)
    output = tmp_path / "private-output"
    output.mkdir()
    harness = tmp_path / "qualification.sh"
    harness.write_text(
        f"set -Eeuo pipefail\noutput_dir=$1\nplatform=linux/arm64\nbackend_tag=localhost:15000/tldw/backend:candidate\n{block}"
    )
    result = subprocess.run(
        ["bash", str(harness), str(output)],
        cwd=root,
        env={**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    mounts = json.loads(capture.read_text())
    assert len(mounts) == 2
    assert (image / "app" / production).read_bytes() == (root / production).read_bytes()
