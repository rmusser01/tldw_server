from pathlib import Path

import pytest

from tldw_Server_API.app.core.exceptions import InvalidStoragePathError

pytestmark = pytest.mark.unit


def test_write_json_artifact_records_manifest(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test query",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)

    payload = {"focus_areas": ["history", "market structure"]}
    artifact = store.write_json(
        owner_user_id=1,
        session_id=session.id,
        artifact_name="plan.json",
        payload=payload,
        phase="drafting_plan",
        job_id="123",
    )

    assert artifact.byte_size > 0
    artifact_path = Path(artifact.storage_path)
    assert artifact_path.exists()
    assert artifact_path.name.startswith("artifact_")
    assert artifact_path.suffix == ".json"

    manifest = db.list_artifacts(session.id)
    assert manifest[0].artifact_name == "plan.json"
    assert not (tmp_path / "outputs" / "research" / session.id / "plan.json").exists()
    assert store.read_json(session_id=session.id, artifact_name="plan.json") == payload

    records = [{"url": "https://example.test", "rank": 1}]
    jsonl_artifact = store.write_jsonl(
        owner_user_id=1,
        session_id=session.id,
        artifact_name="sources.jsonl",
        records=records,
        phase="drafting_plan",
        job_id="123",
    )
    text_artifact = store.write_text(
        owner_user_id=1,
        session_id=session.id,
        artifact_name="summary.txt",
        content="summary",
        phase="drafting_plan",
        job_id="123",
    )

    manifest_names = {artifact.artifact_name for artifact in db.list_artifacts(session.id)}
    assert {"plan.json", "sources.jsonl", "summary.txt"}.issubset(manifest_names)
    assert Path(jsonl_artifact.storage_path).name.startswith("artifact_")
    assert Path(text_artifact.storage_path).name.startswith("artifact_")
    assert store.read_jsonl(session_id=session.id, artifact_name="sources.jsonl") == records
    assert store.read_text(session_id=session.id, artifact_name="summary.txt") == "summary"


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "nested/artifact.json", "..\\outside.json"])
def test_artifact_store_rejects_non_filename_artifact_names(tmp_path, name):
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=None)
    with pytest.raises(InvalidStoragePathError):
        store._artifact_path("session", name)


@pytest.mark.parametrize("session_id", ["../outside", "/absolute/session", "nested/session", "..\\outside"])
def test_artifact_store_hashes_session_ids_into_confined_directories(tmp_path, session_id):
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=None)
    path = store._artifact_path(session_id, "plan.json")

    assert path.parent.parent == (tmp_path / "outputs" / "research").resolve()


def test_artifact_store_rejects_session_directory_symlink_escape(tmp_path):
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=None)
    outside = tmp_path / "outside"
    outside.mkdir()
    session_dir = store._artifact_path("session", "plan.json").parent
    session_dir.rmdir()
    session_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact session path escapes"):
        store._artifact_path("session", "plan.json")


def test_artifact_store_rejects_artifact_symlink_escape(tmp_path):
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=None)
    path = store._artifact_path("session", "plan.json")
    path.symlink_to(tmp_path / "outside.json")

    with pytest.raises(ValueError, match="artifact path escapes"):
        store._artifact_path("session", "plan.json")


@pytest.mark.parametrize("target_kind", ["case-sibling", "same-session", "regular"])
def test_artifact_windows_canonical_leaf_respects_exact_session_root(monkeypatch, target_kind):
    import ntpath
    from pathlib import PureWindowsPath
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Research import artifact_store
    from tldw_Server_API.app.core.Utils import path_utils

    session_name = artifact_store.ResearchArtifactStore._safe_session_component("session")
    session_root = PureWindowsPath(r"C:\Outputs\research") / session_name
    resolutions = []

    class CanonicalWindowsPath(PureWindowsPath):
        def mkdir(self, *, parents=False, exist_ok=False):
            pass

        def resolve(self, *, strict=False):
            resolutions.append(str(self))
            if self.name.startswith("artifact_"):
                if target_kind == "case-sibling":
                    return CanonicalWindowsPath(session_root.parent / session_name.upper() / "private.json")
                if target_kind == "same-session":
                    return CanonicalWindowsPath(session_root / "MixedCase.json")
            return self

    windows_os = SimpleNamespace(path=ntpath, sep="\\", pardir="..")
    monkeypatch.setattr(ntpath, "islink", lambda _path: False)
    monkeypatch.setattr(path_utils, "os", windows_os)
    monkeypatch.setattr(artifact_store, "os", windows_os, raising=False)
    monkeypatch.setattr(artifact_store, "Path", CanonicalWindowsPath)
    store = artifact_store.ResearchArtifactStore(base_dir=r"C:\Outputs", db=None)
    if target_kind == "case-sibling":
        with pytest.raises(ValueError, match="artifact path escapes session directory"):
            store._artifact_path("session", "plan.json")
    else:
        result = store._artifact_path("session", "plan.json")
        assert str(result.parent) == str(session_root)
        if target_kind == "same-session":
            assert result.name == "MixedCase.json"
    assert any("artifact_" in value for value in resolutions)


def test_artifact_store_preserves_same_session_leaf_link(tmp_path):
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=None)
    path = store._artifact_path("session", "plan.json")
    target = path.parent / "MixedCase.json"
    target.write_text("{}", encoding="utf-8")
    path.symlink_to(target)
    assert store._artifact_path("session", "plan.json") == target


def test_artifact_store_rejects_leaf_link_to_session_root_before_versioning(tmp_path):
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=None)
    path = store._artifact_path("session", "plan.json")
    path.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="artifact path escapes session directory"):
        store._versioned_artifact_path("session", "plan.json", 1)


def test_artifact_store_rejects_research_root_symlink_escape(tmp_path):
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output_dir / "research").symlink_to(outside, target_is_directory=True)
    store = ResearchArtifactStore(base_dir=output_dir, db=None)

    with pytest.raises(ValueError, match="artifact research path escapes"):
        store._artifact_path("session", "plan.json")

    assert list(outside.iterdir()) == []


@pytest.fixture
def stored_artifact(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1", query="read boundary", source_policy="balanced",
        autonomy_mode="autonomous", limits_json={},
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    artifact = store.write_text(
        owner_user_id="1", session_id=session.id, artifact_name="payload.json",
        content='{"safe": true}', phase="drafting_plan", job_id=None,
    )
    return store, session.id, artifact


@pytest.mark.parametrize("reader", ["read_text", "read_json", "read_jsonl"])
@pytest.mark.parametrize("attack", ["leaf_symlink", "outside_record", "other_session_record"])
def test_artifact_readers_reject_paths_outside_recorded_session(tmp_path, stored_artifact, reader, attack):
    store, session_id, artifact = stored_artifact
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": "outside"}', encoding="utf-8")
    if attack == "leaf_symlink":
        recorded_path = Path(artifact.storage_path)
        recorded_path.unlink()
        recorded_path.symlink_to(outside)
    else:
        if attack == "other_session_record":
            outside = store._artifact_path("different-session", "payload.json")
            outside.write_text('{"secret": "other session"}', encoding="utf-8")
        store.db.record_artifact(
            session_id=session_id, artifact_name=artifact.artifact_name,
            artifact_version=2, storage_path=str(outside), content_type="application/json",
            byte_size=outside.stat().st_size, checksum="fixture", phase="drafting_plan",
        )

    with pytest.raises(ValueError, match="artifact storage path"):
        getattr(store, reader)(session_id=session_id, artifact_name=artifact.artifact_name)


@pytest.mark.parametrize("reader", ["read_text", "read_json", "read_jsonl"])
def test_artifact_readers_preserve_missing_recorded_file_behavior(stored_artifact, reader):
    store, session_id, artifact = stored_artifact
    Path(artifact.storage_path).unlink()

    assert getattr(store, reader)(session_id=session_id, artifact_name=artifact.artifact_name) is None


@pytest.mark.parametrize("writer", ["write_json", "write_jsonl", "write_text"])
def test_artifact_writers_reject_cross_session_directory_alias(stored_artifact, writer):
    store, victim_id, victim_artifact = stored_artifact
    attacker = store.db.create_session(
        owner_user_id="1", query="attacker", source_policy="balanced",
        autonomy_mode="autonomous", limits_json={},
    )
    victim_path = Path(victim_artifact.storage_path)
    previous_content = victim_path.read_bytes()
    previous_files = set(victim_path.parent.iterdir())
    attacker_dir = store._artifact_path(attacker.id, victim_artifact.artifact_name).parent
    attacker_dir.rmdir()
    attacker_dir.symlink_to(victim_path.parent, target_is_directory=True)
    content = {
        "write_json": {"payload": {"attacker": True}},
        "write_jsonl": {"records": [{"attacker": True}]},
        "write_text": {"content": "attacker overwrite"},
    }[writer]

    with pytest.raises(ValueError, match="artifact session path escapes"):
        getattr(store, writer)(
            owner_user_id="1", session_id=attacker.id,
            artifact_name=victim_artifact.artifact_name, phase="test", job_id=None,
            **content,
        )

    assert victim_path.read_bytes() == previous_content
    assert set(victim_path.parent.iterdir()) == previous_files
    assert store.db.list_artifacts(attacker.id) == []


@pytest.mark.parametrize(
    ("writer", "reader", "old", "new"),
    [
        ("write_json", "read_json", {"payload": {"old": True}}, {"payload": {"new": True}}),
        ("write_jsonl", "read_jsonl", {"records": [{"old": True}]}, {"records": [{"new": True}]}),
        ("write_text", "read_text", {"content": "old"}, {"content": "new"}),
    ],
)
def test_control_change_during_file_write_preserves_registered_content_and_no_unpublished_files(
    tmp_path,
    monkeypatch,
    writer,
    reader,
    old,
    new,
):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchPhaseChangedError, ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous", limits_json={}
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    args = {
        "owner_user_id": "1",
        "session_id": session.id,
        "artifact_name": "controlled",
        "phase": session.phase,
        "job_id": None,
    }
    original = getattr(store, writer)(**args, **old)
    before = getattr(store, reader)(session_id=session.id, artifact_name="controlled")
    assert before == next(iter(old.values()))  # nosec B101 -- Establish an actual readable baseline before testing a rejected update.
    alias = store._artifact_path(session.id, "controlled")
    alias.write_bytes(b"retained legacy alias")
    files = set(alias.parent.iterdir())
    # This unit isolates the Research control fence; real Jobs lease fencing has a separate SDK regression.
    db.bind_phase_writer(db.get_session(session.id), lambda: None)
    atomic_write = store._write_bytes_atomically
    changed = []

    def cancel_after_file(path, data):
        atomic_write(path, data)
        if not changed:
            changed.append(True)
            ResearchSessionsDB(db.db_path).update_control_state(session.id, control_state="cancel_requested")

    monkeypatch.setattr(store, "_write_bytes_atomically", cancel_after_file)
    with pytest.raises(ResearchPhaseChangedError):
        getattr(store, writer)(**args, **new)
    assert alias.read_bytes() == b"retained legacy alias"  # nosec B101 -- A rejected attempt cannot replace preexisting functional bytes.
    assert getattr(store, reader)(session_id=session.id, artifact_name="controlled") == before  # nosec B101 -- Canonical reads retain the acknowledged registered version.
    assert db.list_artifacts(session.id) == [original] and set(alias.parent.iterdir()) == files  # nosec B101 -- Remove this attempt's unpublished version without touching prior files.


@pytest.mark.parametrize("readback_unavailable", [False, True])
def test_artifact_unknown_commit_ack_preserves_registered_file(tmp_path, monkeypatch, readback_unavailable):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous", limits_json={}
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    record, read = db.record_artifact_with_event, db.list_artifacts
    committed = []

    def lose_ack(**kwargs):
        result = record(**kwargs)
        committed.append(result[0])
        raise RuntimeError("controlled manifest acknowledgement")

    def manifest(session_id):
        if committed and readback_unavailable:
            raise RuntimeError("controlled manifest readback unavailable")
        return read(session_id)

    monkeypatch.setattr(db, "record_artifact_with_event", lose_ack)
    monkeypatch.setattr(db, "list_artifacts", manifest)
    with pytest.raises(RuntimeError, match="manifest acknowledgement") as error:
        store.write_json(
            owner_user_id="1",
            session_id=session.id,
            artifact_name="plan.json",
            payload={"query": "Rowan"},
            phase="drafting_plan",
            job_id=None,
        )
    assert len(read(session.id)) == 1 and Path(committed[0].storage_path).is_file()  # nosec B101 -- Unknown acknowledgements never delete a registered canonical version.
    assert bool(getattr(error.value, "__notes__", [])) is readback_unavailable  # nosec B101 -- An unavailable readback remains explicit rather than being treated as rollback.


def test_artifact_atomic_replace_failure_removes_its_owned_temporary_file(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1", query="Rowan", source_policy="local_only", autonomy_mode="autonomous", limits_json={}
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)

    def interrupt_replace(*args, **kwargs):
        raise RuntimeError("controlled atomic replace failure")

    monkeypatch.setattr(Path, "replace", interrupt_replace)
    with pytest.raises(RuntimeError, match="atomic replace"):
        store.write_json(
            owner_user_id="1",
            session_id=session.id,
            artifact_name="plan.json",
            payload={"query": "Rowan"},
            phase="drafting_plan",
            job_id=None,
        )
    assert (
        db.list_artifacts(session.id) == []
        and [path for path in (tmp_path / "outputs").rglob("*") if path.is_file()] == []
    )  # nosec B101 -- Failed writes leave neither partial metadata nor their owned temporary file.
