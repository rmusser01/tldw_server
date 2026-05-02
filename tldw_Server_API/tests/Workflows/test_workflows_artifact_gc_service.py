from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.services import workflows_artifact_gc_service as gc_mod


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_artifact_gc_appends_deletion_evidence(monkeypatch, tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-artifact-gc"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "artifact-gc", "version": 1, "steps": []},
    )

    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("artifact payload", encoding="utf-8")
    artifact_id = "artifact-gc-1"
    db.add_artifact(
        artifact_id=artifact_id,
        tenant_id="default",
        run_id=run_id,
        step_run_id=None,
        type="text",
        uri=f"file://{artifact_path}",
        metadata={"purpose": "retention-test"},
    )

    monkeypatch.setattr(
        gc_mod,
        "create_workflows_database",
        lambda backend=None: db,
    )
    monkeypatch.setattr(
        gc_mod,
        "get_content_backend_instance",
        lambda: None,
    )
    monkeypatch.setattr(
        gc_mod,
        "_now_utc",
        lambda: gc_mod.datetime(2030, 1, 1, tzinfo=gc_mod.timezone.utc),
    )
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_RETENTION_DAYS", "30")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC", "1")

    stop_event = asyncio.Event()
    task = asyncio.create_task(gc_mod.run_workflows_artifact_gc_worker(stop_event))

    async def _wait_for_gc() -> None:
        while artifact_path.exists() or db.get_artifact(artifact_id) is not None:
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(_wait_for_gc(), timeout=3)
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=3)

    assert not artifact_path.exists()
    assert db.get_artifact(artifact_id) is None

    events = db.get_events(run_id, types=["artifact_gc"])
    assert events, "Expected artifact GC to append workflow evidence"
    payload = events[-1]["payload_json"]
    assert payload["artifact_id"] == artifact_id
    assert payload["status"] == "deleted"
    assert payload["file_deleted"] is True
    assert payload["row_deleted"] is True
    assert payload["source"] == "artifact_gc"
