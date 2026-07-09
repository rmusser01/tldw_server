from pathlib import Path

import pytest


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
