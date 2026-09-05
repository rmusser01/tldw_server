"""Contracts for immutable production and reference image identities."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_DOCKERFILES = (
    ROOT / "Dockerfiles/Dockerfile.prod",
    ROOT / "Dockerfiles/Dockerfile.worker",
    ROOT / "Dockerfiles/Dockerfile.audio_gpu_worker",
    ROOT / "Dockerfiles/Dockerfile.webui",
    ROOT / "Dockerfiles/Dockerfile.admin-ui",
)
REFERENCE_IMAGES = ROOT / ".github/supply-chain/reference-images.json"
IMAGE_REF = re.compile(r"^[^@\s]+:[^@\s]+@sha256:[0-9a-f]{64}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_REFERENCE_IMAGES = {
    "caddy": "caddy:",
    "postgres": "postgres:",
    "redis": "redis:",
    "prometheus": "prom/prometheus:",
    "alertmanager": "prom/alertmanager:",
    "grafana": "grafana/grafana:",
}
EXPECTED_REFERENCE_IDENTITIES = {
    "postgres": {
        "reference": (
            "postgres:18.6-alpine3.24@sha256:"
            "d3e1620b530c944afa6e887d22eb899824da68e19c52024bf98f5220c88a65b2"
        ),
        "platform_manifest_digest": (
            "sha256:63bdc97d67b5133bf0e5ebd500bec6d046fa851dc81340d838f0347e616107e8"
        ),
    },
    "prometheus": {
        "reference": (
            "prom/prometheus:v3.14.0@sha256:"
            "5ce7540c3c00ef4ab0c9d2c995c6a5b9c421f44b4a115d97a2c7af3b1c21cbb0"
        ),
        "platform_manifest_digest": (
            "sha256:e906cef998316bbe319f98711e1b4d8613ad37e14b08ff831d7036e77b7464f9"
        ),
    },
}
NODE_24_LTS = (
    "node:24.20.0-bookworm-slim@sha256:"
    "ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e"
)
PYTHON_312_RUNTIME = (
    "python:3.12.14-slim-trixie@sha256:"
    "78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
)
PYTHON_311_RUNTIME = (
    "python:3.11.16-slim-trixie@sha256:"
    "9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534"
)


def _from_reference(line: str) -> str:
    """Return the image token from one Dockerfile FROM instruction."""
    fields = line.split()
    index = 1
    if fields[index].startswith("--platform="):
        index += 1
    return fields[index]


@pytest.mark.parametrize("path", PRODUCTION_DOCKERFILES)
def test_every_production_from_is_digest_pinned(path: Path) -> None:
    """Every production stage retains a readable, immutable base reference."""
    from_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("FROM ")]

    assert from_lines
    for line in from_lines:
        reference = _from_reference(line)
        assert IMAGE_REF.fullmatch(reference), (path, line)
        assert reference.rpartition("@sha256:")[0].rsplit(":", 1)[-1].lower() != "latest"


def test_reference_image_inventory_is_exact_and_digest_bound() -> None:
    """The six operator-controlled runtime images have complete OCI identity."""
    inventory = json.loads(REFERENCE_IMAGES.read_text(encoding="utf-8"))

    assert set(inventory) == {"schema_version", "images"}
    assert inventory["schema_version"] == 1
    images = inventory["images"]
    assert isinstance(images, list)
    assert {record["name"] for record in images} == set(EXPECTED_REFERENCE_IMAGES)
    assert len(images) == len(EXPECTED_REFERENCE_IMAGES)

    for record in images:
        assert set(record) == {
            "name",
            "reference",
            "platform",
            "index_digest",
            "platform_manifest_digest",
        }
        assert record["platform"] == "linux/amd64"
        assert DIGEST.fullmatch(record["index_digest"])
        assert DIGEST.fullmatch(record["platform_manifest_digest"])
        assert record["index_digest"] != record["platform_manifest_digest"]
        assert record["reference"].startswith(EXPECTED_REFERENCE_IMAGES[record["name"]])
        assert record["reference"].endswith("@" + record["index_digest"])
        assert IMAGE_REF.fullmatch(record["reference"])

    records_by_name = {record["name"]: record for record in images}
    for name, expected in EXPECTED_REFERENCE_IDENTITIES.items():
        assert records_by_name[name]["reference"] == expected["reference"]
        assert (
            records_by_name[name]["platform_manifest_digest"]
            == expected["platform_manifest_digest"]
        )


@pytest.mark.parametrize(
    "path",
    (
        ROOT / "Dockerfiles/Dockerfile.webui",
        ROOT / "Dockerfiles/Dockerfile.admin-ui",
    ),
)
def test_next_images_use_supported_node_24_lts(path: Path) -> None:
    """Shipped Next images must not regress to the EOL Node 20 release line."""
    from_references = [
        _from_reference(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("FROM node:")
    ]

    assert from_references == [NODE_24_LTS, NODE_24_LTS]


@pytest.mark.parametrize(
    "path",
    (
        ROOT / "Dockerfiles/Dockerfile.webui",
        ROOT / "Dockerfiles/Dockerfile.admin-ui",
    ),
)
def test_next_runtime_images_do_not_ship_npm(path: Path) -> None:
    """The production runtime contains Node, not an unused package manager."""
    runtime = path.read_text(encoding="utf-8").split(" AS runtime", maxsplit=1)[1]

    assert "rm -rf /usr/local/lib/node_modules/npm" in runtime
    assert "/usr/local/bin/npm" in runtime
    assert "/usr/local/bin/npx" in runtime


@pytest.mark.parametrize(
    ("path", "expected"),
    (
        (ROOT / "Dockerfiles/Dockerfile.prod", PYTHON_312_RUNTIME),
        (ROOT / "Dockerfiles/Dockerfile.worker", PYTHON_311_RUNTIME),
        (ROOT / "Dockerfiles/Dockerfile.audio_gpu_worker", PYTHON_311_RUNTIME),
    ),
)
def test_python_images_use_patch_qualified_trixie_bases(
    path: Path,
    expected: str,
) -> None:
    """Python production images make the exact interpreter and distro legible."""
    from_references = [
        _from_reference(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("FROM python:")
    ]

    assert from_references == [expected, expected]


@pytest.mark.parametrize(
    "path",
    (
        ROOT / "Dockerfiles/Dockerfile.prod",
        ROOT / "Dockerfiles/Dockerfile.worker",
        ROOT / "Dockerfiles/Dockerfile.audio_gpu_worker",
    ),
)
def test_python_runtime_images_do_not_ship_package_managers(path: Path) -> None:
    """The locked production environments do not retain base pip/setuptools."""
    runtime = path.read_text(encoding="utf-8").split(" AS runtime", maxsplit=1)[1]

    assert "/usr/local/lib/python*/site-packages/pip" in runtime
    assert "/usr/local/lib/python*/site-packages/setuptools" in runtime
    assert "/usr/local/bin/pip*" in runtime


def test_embedding_worker_runtime_has_only_required_os_package() -> None:
    """The text embedding worker does not inherit unrelated media tooling."""
    dockerfile = (ROOT / "Dockerfiles/Dockerfile.worker").read_text(encoding="utf-8")
    runtime = dockerfile.split(" AS runtime", maxsplit=1)[1]
    install_block = runtime.split("apt-get install", maxsplit=1)[1].split(
        "rm -rf /var/lib/apt/lists/*",
        maxsplit=1,
    )[0]

    assert "ca-certificates" in install_block
    for unexpected in ("curl", "ffmpeg", "git", "libportaudio2"):
        assert unexpected not in install_block


def test_admin_next_build_executes_with_node_after_frozen_bun_install() -> None:
    """Next builds run under Node while Bun remains the lockfile installer."""
    dockerfile = (ROOT / "Dockerfiles/Dockerfile.admin-ui").read_text(encoding="utf-8")
    builder_line = next(
        line for line in dockerfile.splitlines() if line.startswith("FROM ") and line.endswith(" AS builder")
    )

    assert builder_line.startswith("FROM node:")
    assert "RUN bun install --frozen-lockfile" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "RUN bun run build" not in dockerfile


def test_webui_production_build_caps_next_worker_count() -> None:
    """The reference WebUI build stays within ordinary container memory limits."""
    dockerfile = (ROOT / "Dockerfiles/Dockerfile.webui").read_text(encoding="utf-8")
    builder_line = next(
        line for line in dockerfile.splitlines() if line.startswith("FROM ") and line.endswith(" AS builder")
    )

    assert builder_line.startswith("FROM node:")
    assert "TLDW_NEXT_BUILD_CPUS=1" in dockerfile
    assert "NODE_OPTIONS=--max-old-space-size=6144" in dockerfile
    assert "RUN bun install --frozen-lockfile --cwd /app/apps" in dockerfile
    assert "npm run compile:prod" in dockerfile
    assert "npm run build:prod" not in dockerfile
    assert "bun run build:prod" not in dockerfile
