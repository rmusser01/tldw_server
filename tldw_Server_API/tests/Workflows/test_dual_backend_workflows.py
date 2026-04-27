"""Dual-backend regression tests for the Workflows persistence layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import pytest


pytestmark = pytest.mark.integration


def _now_iso() -> str:


    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def test_workflow_definition_and_run_roundtrip(workflows_dual_backend_db):


    backend_label, db = workflows_dual_backend_db

    definition_body: Dict[str, object] = {
        "name": "Sample Workflow",
        "version": 1,
        "description": "demo",
        "tags": ["demo", backend_label],
        "steps": [
            {
                "id": "step-1",
                "name": "Echo",
                "type": "prompt",
                "config": {"prompt": "Hello"},
            }
        ],
    }

    workflow_id = db.create_definition(
        tenant_id="tenant-1",
        name="Sample Workflow",
        version=1,
        owner_id="owner-1",
        visibility="private",
        description=definition_body["description"],
        tags=definition_body["tags"],
        definition=definition_body,
    )

    fetched = db.get_definition(workflow_id)
    assert fetched is not None
    assert fetched.tenant_id == "tenant-1"
    assert fetched.name == "Sample Workflow"

    definitions = db.list_definitions(tenant_id="tenant-1", owner_id="owner-1")
    assert any(d.id == workflow_id for d in definitions)

    run_id = f"run-{backend_label}"
    snapshot = {
        "name": fetched.name,
        "version": fetched.version,
        "steps": definition_body["steps"],
    }

    db.create_run(
        run_id=run_id,
        tenant_id="tenant-1",
        user_id="runner",
        inputs={"input": "value"},
        workflow_id=workflow_id,
        definition_version=fetched.version,
        definition_snapshot=snapshot,
        idempotency_key=f"idem-{backend_label}",
        session_id="session-1",
    )

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "queued"

    db.update_run_status(
        run_id,
        status="running",
        status_reason="started",
        started_at=_now_iso(),
        tokens_input=42,
    )
    db.set_cancel_requested(run_id, cancel=True)
    assert db.is_cancel_requested(run_id) is True

    db.update_run_status(
        run_id,
        status="completed",
        outputs={"foo": "bar"},
        ended_at=_now_iso(),
        duration_ms=1234,
        tokens_output=100,
        cost_usd=0.12,
    )

    # Events round-trip
    seq = db.append_event(
        tenant_id="tenant-1",
        run_id=run_id,
        event_type="run_completed",
        payload={"status": "completed"},
    )
    assert seq == 1
    events = db.get_events(run_id)
    assert events and events[0]["event_seq"] == seq

    # Step runs and artifacts
    step_run_id = f"{run_id}:step-1"
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="tenant-1",
        run_id=run_id,
        step_id="step-1",
        name="Echo",
        step_type="prompt",
        status="running",
        inputs={"config": {"prompt": "Hello"}},
    )
    db.update_step_attempt(step_run_id=step_run_id, attempt=1)
    db.update_step_lock_and_heartbeat(
        step_run_id=step_run_id,
        locked_by="tester",
        lock_ttl_seconds=30,
    )
    db.update_step_subprocess(
        step_run_id=step_run_id,
        pid=123,
        workdir="/tmp/workdir",  # nosec B108
        stdout_path="stdout.log",
    )
    db.complete_step_run(
        step_run_id=step_run_id,
        status="succeeded",
        outputs={"response": "Hello"},
    )
    step_runs = db.list_step_runs(run_id=run_id)
    assert len(step_runs) == 1
    assert step_runs[0]["step_run_id"] == step_run_id
    assert step_runs[0]["status"] == "succeeded"
    attempt_id = db.create_step_attempt(
        tenant_id="tenant-1",
        run_id=run_id,
        step_run_id=step_run_id,
        step_id="step-1",
        attempt_number=1,
        status="running",
        metadata={"backend": backend_label},
    )
    db.complete_step_attempt(
        attempt_id=attempt_id,
        status="succeeded",
        metadata={"backend": backend_label, "result": "ok"},
    )
    attempts = db.list_step_attempts(run_id=run_id, step_id="step-1")
    assert len(attempts) == 1
    assert attempts[0]["attempt_id"] == attempt_id
    assert attempts[0]["status"] == "succeeded"
    assert attempts[0]["metadata_json"]["backend"] == backend_label

    db.add_artifact(
        artifact_id=f"artifact-{backend_label}",
        tenant_id="tenant-1",
        run_id=run_id,
        step_run_id=step_run_id,
        type="text",
        uri="s3://bucket/object",
        size_bytes=12,
        mime_type="text/plain",
        checksum_sha256="deadbeef",
        metadata={"source": backend_label},
    )

    artifacts = db.list_artifacts_for_run(run_id)
    assert artifacts and artifacts[0]["type"] == "text"
    artifact = db.get_artifact(f"artifact-{backend_label}")
    assert artifact is not None and artifact["artifact_id"] == f"artifact-{backend_label}"

    orphans = db.find_orphan_step_runs(_now_iso())
    assert isinstance(orphans, list)

    subprocesses = db.find_running_subprocesses_for_run(run_id)
    assert isinstance(subprocesses, list)

    # Cleanup flag should succeed without raising
    assert db.soft_delete_definition(workflow_id) is True


def test_step_attempt_contract_exists_on_both_backends(workflows_dual_backend_db):
    _backend_label, db = workflows_dual_backend_db

    assert callable(getattr(db, "create_step_attempt", None))
    assert callable(getattr(db, "complete_step_attempt", None))
    assert callable(getattr(db, "list_step_attempts", None))
    assert callable(getattr(db, "list_step_runs", None))
