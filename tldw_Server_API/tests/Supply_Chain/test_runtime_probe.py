"""Reject unbound runtime evidence, unexpected listeners and dependency drift."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


class _FakeCollection:
    def __init__(self, collection_id: str) -> None:
        self.id = collection_id
        self.ids: list[str] = []
        self.documents: list[str] = []

    def add(self, *, ids: list[str], documents: list[str], embeddings: list[list[float]]) -> None:
        self.ids = ids
        self.documents = documents

    def get(self, *, include: list[str] | None = None) -> dict:
        return {"ids": list(self.ids), "documents": list(self.documents)}

    def query(self, *, query_embeddings: list[list[float]], n_results: int) -> dict:
        return {"ids": [list(self.ids)]}


class _FakeClient:
    def __init__(self, mode: str) -> None:
        self._server = type("FakeServer", (), {})()
        type(self._server).__module__ = "chromadb.api.rust"
        self.mode = mode
        self.collection: _FakeCollection | None = None
        self.calls: list[str] = []

    def __getattr__(self, operation: str):
        if operation not in {"_get", "_count", "_query", "_add", "_update", "_upsert", "_delete"}:
            raise AttributeError(operation)

        def call(**kwargs):
            from chromadb.errors import NotFoundError

            self.calls.append(operation)
            if self.mode == "accept":
                return None
            if self.mode == "unexpected-error":
                raise RuntimeError("unexpected storage failure")
            if self.mode == "corrupt":
                assert self.collection is not None
                self.collection.documents = ["modified"]
            raise NotFoundError("foreign collection")

        return call


def _install_fake_chroma(monkeypatch: pytest.MonkeyPatch, mode: str) -> list[_FakeClient]:
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library

    clients = [_FakeClient(mode), _FakeClient(mode)]
    collections = [_FakeCollection("collection-0"), _FakeCollection("collection-1")]
    for client, collection in zip(clients, collections):
        client.collection = collection
    created = 0

    class FakeManager:
        def __init__(self, *, user_id: str, user_embedding_config: dict) -> None:
            nonlocal created
            index = created
            created += 1
            self.client = clients[index]
            self.collection = collections[index]
            self.user_chroma_path = f"/fake/user-{index}"

        def get_or_create_collection(self, name: str) -> _FakeCollection:
            return self.collection

        def close(self) -> None:
            return None

    monkeypatch.setattr(ChromaDB_Library, "ChromaDBManager", FakeManager)
    return clients


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


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("accept", "foreign collection UUID"),
        ("unexpected-error", "foreign collection UUID"),
        ("corrupt", "original Chroma collection changed"),
    ],
)
def test_storage_probe_fails_closed_on_invalid_foreign_uuid_behavior(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    message: str,
) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import probe_chroma

    _install_fake_chroma(monkeypatch, mode)
    with pytest.raises(ValueError, match=message):
        probe_chroma()


def test_storage_probe_checks_every_low_level_operation_in_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import probe_chroma

    clients = _install_fake_chroma(monkeypatch, "isolated")
    assert probe_chroma() == "chromadb.api.rust"
    assert [client.calls for client in clients] == [
        ["_get", "_count", "_query", "_add", "_update", "_upsert", "_delete"],
        ["_get", "_count", "_query", "_add", "_update", "_upsert", "_delete"],
    ]


def test_os_facts_record_absent_facilities(monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts.Supply_Chain import runtime_probe

    monkeypatch.setattr(Path, "is_file", lambda path: False)
    assert runtime_probe.probe_os_facts() == {
        "perl": {"path": "/usr/bin/perl", "status": "absent"},
        "systemd_homed": {
            "paths": {
                "/usr/lib/systemd/systemd-homed": False,
                "/lib/systemd/systemd-homed": False,
            }
        },
    }


def test_os_facts_record_observed_perl_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts.Supply_Chain import runtime_probe

    monkeypatch.setattr(Path, "is_file", lambda path: True)

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        assert command == ["/usr/bin/perl", "-V:ivsize", "-V:ptrsize"]
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 5}
        return subprocess.CompletedProcess(command, 0, "ivsize='8';\nptrsize='8';\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert runtime_probe.probe_os_facts() == {
        "perl": {"path": "/usr/bin/perl", "status": "observed", "ivsize": 8, "ptrsize": 8},
        "systemd_homed": {
            "paths": {
                "/usr/lib/systemd/systemd-homed": True,
                "/lib/systemd/systemd-homed": True,
            }
        },
    }


@pytest.mark.parametrize(
    ("output", "exception"),
    [
        ("ivsize='8';\n", None),
        (None, subprocess.CalledProcessError(1, ["/usr/bin/perl"])),
    ],
)
def test_os_facts_fail_closed_when_perl_is_unobserved(
    monkeypatch: pytest.MonkeyPatch,
    output: str | None,
    exception: subprocess.CalledProcessError | None,
) -> None:
    from Helper_Scripts.Supply_Chain import runtime_probe

    monkeypatch.setattr(Path, "is_file", lambda path: True)

    def run(*args, **kwargs):
        if exception is not None:
            raise exception
        return subprocess.CompletedProcess(args[0], 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="Perl"):
        runtime_probe.probe_os_facts()


def test_probe_evidence_records_os_facts_and_foreign_uuid_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Helper_Scripts.Supply_Chain import runtime_probe

    lock = tmp_path / "uv.lock"
    lock.write_text("")
    absent = {
        "perl": {"path": "/usr/bin/perl", "status": "absent"},
        "systemd_homed": {
            "paths": {
                "/usr/lib/systemd/systemd-homed": False,
                "/lib/systemd/systemd-homed": False,
            }
        },
    }
    monkeypatch.setattr(runtime_probe, "require_locked_versions", lambda path: {})
    monkeypatch.setattr(runtime_probe, "require_no_listeners", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_probe, "probe_chroma", lambda **kwargs: "chromadb.api.rust")
    monkeypatch.setattr(runtime_probe, "probe_crypto", lambda: "cryptography.backend")
    monkeypatch.setattr(runtime_probe, "probe_os_facts", lambda: absent, raising=False)
    monkeypatch.setattr("sys.argv", ["runtime_probe.py", "probe", "--lock", str(lock)])

    runtime_probe.main()

    evidence = json.loads(capsys.readouterr().out)
    assert evidence["os_facts"] == absent
    assert evidence["checks"]["foreign_collection_uuid_isolation"] is True
