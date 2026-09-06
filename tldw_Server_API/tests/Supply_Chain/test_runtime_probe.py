"""Reject unbound runtime evidence, unexpected listeners and dependency drift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _blob(root: Path, value: dict) -> str:
    raw = json.dumps(value).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = root / "blobs" / "sha256" / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return f"sha256:{digest}"


def _candidate(root: Path, architecture: str = "amd64") -> tuple[str, str]:
    config = _blob(root, {"os": "linux", "architecture": architecture})
    manifest = _blob(root, {"config": {"digest": config}, "layers": []})
    subject = _blob(
        root,
        {
            "manifests": [
                {
                    "digest": manifest,
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ]
        },
    )
    return subject, config


def test_runtime_identity_is_derived_from_hashed_oci_subject(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import candidate_config

    subject, config = _candidate(tmp_path)
    assert candidate_config(tmp_path, subject) == config


def test_runtime_identity_rejects_corrupted_manifest(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import candidate_config

    subject, _ = _candidate(tmp_path)
    (tmp_path / "blobs/sha256" / subject.split(":")[1]).write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        candidate_config(tmp_path, subject)


def test_runtime_identity_rejects_wrong_config_platform(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import candidate_config

    subject, _ = _candidate(tmp_path, architecture="arm64")
    with pytest.raises(ValueError, match="platform"):
        candidate_config(tmp_path, subject)


@pytest.mark.parametrize("subject", ["latest", "sha256:../../secret", "sha256:" + "A" * 64])
def test_runtime_identity_rejects_untrusted_digest(tmp_path: Path, subject: str) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import candidate_config

    with pytest.raises(ValueError, match="digest"):
        candidate_config(tmp_path, subject)


def test_runtime_identity_rejects_ambiguous_platform(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import candidate_config

    subject, _ = _candidate(tmp_path)
    value = json.loads((tmp_path / "blobs/sha256" / subject.split(":")[1]).read_text())
    value["manifests"] *= 2
    with pytest.raises(ValueError, match="one linux/amd64"):
        candidate_config(tmp_path, _blob(tmp_path, value))


@pytest.mark.parametrize("state,rejected", [("0A", True), ("01", False)])
def test_listener_probe_distinguishes_listening_from_connected_sockets(
    tmp_path: Path,
    state: str,
    rejected: bool,
) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import require_no_listeners

    (tmp_path / "tcp").write_text(f"header\n 0: 00000000:1F40 00000000:0000 {state} 0\n")
    (tmp_path / "tcp6").write_text("header\n")
    if rejected:
        with pytest.raises(ValueError, match="listening"):
            require_no_listeners(tmp_path)
    else:
        require_no_listeners(tmp_path)


def test_listener_probe_fails_closed_without_proc_evidence(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import require_no_listeners

    with pytest.raises(OSError):
        require_no_listeners(tmp_path)


def test_listener_probe_rejects_malformed_socket_table(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import require_no_listeners

    (tmp_path / "tcp").write_text("header\n malformed\n")
    with pytest.raises(ValueError, match="socket"):
        require_no_listeners(tmp_path)


def test_dependency_probe_rejects_installed_version_drift(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import require_locked_versions

    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "chromadb"\nversion = "0.0.0"\n')
    with pytest.raises(ValueError, match="locked version"):
        require_locked_versions(lock)
