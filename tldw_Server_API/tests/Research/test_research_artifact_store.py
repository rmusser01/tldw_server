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
    victim_path = store._artifact_path(victim_id, victim_artifact.artifact_name)
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
