"""Contract tests for the candidate-only frontend Dockerfile renderer."""

import hashlib
import importlib.util
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/frontend/render.py"
RUNTIME_FROM = (
    "FROM node:24.20.0-bookworm-slim@sha256:"
    "ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS runtime\n"
)
BUILDER_FROM = (
    "FROM node:24.20.0-bookworm-slim@sha256:"
    "ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS builder\n"
)
CANONICAL = {
    "webui": ROOT / "Dockerfiles/Dockerfile.webui",
    "admin-ui": ROOT / "Dockerfiles/Dockerfile.admin-ui",
}


def renderer():
    """Load the repository-owned renderer without making it a package."""
    assert SCRIPT.is_file(), "frontend candidate renderer is not implemented"
    spec = importlib.util.spec_from_file_location("frontend_candidate_render", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("application", ["webui", "admin-ui"])
def test_render_preserves_canonical_builder_prefix_and_runtime_remainder(application):
    source = CANONICAL[application].read_text()
    prefix, remainder = source.split(RUNTIME_FROM)

    output = renderer().render_candidate(source)

    assert output.startswith(prefix)
    assert output.endswith(remainder)
    assert output == renderer().render_candidate(source)


def test_render_injects_exact_native_runtime_and_authenticated_acquisition_evidence():
    output = renderer().render_candidate(CANONICAL["webui"].read_text())

    assert (
        "FROM ubuntu:24.04@sha256:"
        "33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS runtime\n" in output
    )
    assert "ENV NODE_VERSION=24.20.0" in output
    assert "ARG ZLIB1G_VERSION=1:1.3.dfsg-3.1ubuntu2.2" in output
    assert "ARG LIBC6_VERSION=2.39-0ubuntu8.8" in output
    assert '"zlib1g=${ZLIB1G_VERSION}"' in output
    assert "apt-get update" in output
    assert "apt-get install --download-only" in output
    assert "apt-inrelease" in output
    assert "apt-inrelease.sha256" in output
    assert "zlib1g-download.sha256" in output
    assert "installed-versions.tsv" in output
    assert "dpkg-query" in output
    assert "libstdc++6" in output


def test_render_preserves_docker_run_continuations_and_dpkg_format_escapes():
    output = renderer().render_candidate(CANONICAL["webui"].read_text())

    assert "RUN set -eux; \\\n    evidence=/usr/local/share/tldw-candidate-evidence; \\" in output
    assert r"dpkg-query -W -f='${binary:Package}\t${Version}\n'" in output


def test_render_copies_verified_node_artifacts_and_restores_entrypoint():
    output = renderer().render_candidate(CANONICAL["webui"].read_text())

    assert "COPY --from=builder /usr/local/bin/node /usr/local/bin/node" in output
    assert "COPY --from=builder /usr/local/bin/docker-entrypoint.sh " "/usr/local/bin/docker-entrypoint.sh" in output
    assert "COPY --from=builder /usr/local/LICENSE /usr/local/LICENSE" in output
    assert 'ENTRYPOINT ["docker-entrypoint.sh"]' in output
    assert "/etc/ssl/certs" not in output
    assert "/usr/share/ca-certificates" not in output


@pytest.mark.parametrize(
    "forbidden",
    [
        "--allow-unauthenticated",
        "Acquire::AllowInsecureRepositories",
        "Acquire::AllowDowngradeToInsecureRepositories",
        "trusted=yes",
        "upstream1.3.2",
        "snapshot.ubuntu.com",
    ],
)
def test_render_does_not_weaken_apt_or_claim_snapshot_reproducibility(forbidden):
    output = renderer().render_candidate(CANONICAL["webui"].read_text())
    assert forbidden not in output


@pytest.mark.parametrize("application", ["webui", "admin-ui"])
def test_render_preserves_application_runtime_contract(application):
    source = CANONICAL[application].read_text()
    _, remainder = source.split(RUNTIME_FROM)
    output = renderer().render_candidate(source)

    assert remainder in output
    expected_user = "USER webui" if application == "webui" else "USER adminui"
    assert expected_user in output
    assert "HEALTHCHECK --interval=30s --timeout=5s --retries=5" in output
    assert 'CMD ["node", "server.js"]' in output


@pytest.mark.parametrize("mutation", ["missing", "duplicated", "version", "digest"])
def test_render_rejects_noncanonical_runtime_stage(mutation):
    source = CANONICAL["webui"].read_text()
    if mutation == "missing":
        source = source.replace(RUNTIME_FROM, "")
    elif mutation == "duplicated":
        source = source.replace(RUNTIME_FROM, RUNTIME_FROM + RUNTIME_FROM)
    elif mutation == "version":
        changed = RUNTIME_FROM.replace("node:24.20.0-bookworm-slim", "node:24.20.1-bookworm-slim")
        source = source.replace(RUNTIME_FROM, changed)
    else:
        source = source.replace(RUNTIME_FROM, RUNTIME_FROM.replace("ba849c60", "ca849c60"))

    with pytest.raises(ValueError, match="expected exactly one canonical runtime stage"):
        renderer().render_candidate(source)


@pytest.mark.parametrize("mutation", ["missing", "renamed", "duplicated"])
def test_render_rejects_noncanonical_builder_stage(mutation):
    source = CANONICAL["webui"].read_text()
    if mutation == "missing":
        source = source.replace(BUILDER_FROM, "")
    elif mutation == "renamed":
        source = source.replace(" AS builder\n", " AS compile\n", 1)
    else:
        source = source.replace(BUILDER_FROM, BUILDER_FROM + BUILDER_FROM)

    with pytest.raises(ValueError, match="expected canonical Node builder stage"):
        renderer().render_candidate(source)


@pytest.mark.parametrize("application", ["webui", "admin-ui"])
def test_cli_writes_only_requested_candidate_and_preserves_canonical_sources(tmp_path, application):
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in CANONICAL.values()}
    output = tmp_path / f"Dockerfile.{application}.candidate"

    result = subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), "--application", application, "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert set(tmp_path.iterdir()) == {output}
    assert RUNTIME_FROM not in output.read_text()
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in CANONICAL.values()}
    assert after == before


def test_cli_rejects_changed_source_without_writing_output(tmp_path):
    copied_script = tmp_path / "repo/Dockerfiles/candidates/frontend/render.py"
    copied_script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, copied_script)
    canonical = tmp_path / "repo/Dockerfiles/Dockerfile.webui"
    canonical.write_text(CANONICAL["webui"].read_text().replace(RUNTIME_FROM, ""))
    output = tmp_path / "candidate"

    result = subprocess.run(  # nosec B603
        [sys.executable, str(copied_script), "--application", "webui", "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "expected exactly one canonical runtime stage" in result.stderr
    assert not output.exists()
