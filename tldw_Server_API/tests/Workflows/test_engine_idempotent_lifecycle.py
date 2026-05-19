import pytest

from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine


pytestmark = pytest.mark.unit


def _create_run(db: WorkflowsDatabase, run_id: str) -> None:
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "cancel-idempotent", "version": 1, "steps": []},
    )


def test_duplicate_cancel_request_is_already_applied(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-idem-cancel"
    _create_run(db, run_id)

    engine = WorkflowEngine(db)
    first = engine.cancel(run_id)
    second = engine.cancel(run_id)

    assert first == "applied"
    assert second == "already_applied"


def test_cancel_records_acknowledged_event_once(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-cancel-ack"
    _create_run(db, run_id)

    engine = WorkflowEngine(db)
    first = engine.cancel(run_id)
    second = engine.cancel(run_id)

    assert first == "applied"
    assert second == "already_applied"
    ack_events = db.get_events(run_id, types=["cancel_acknowledged"])
    assert len(ack_events) == 1
    payload = ack_events[0].get("payload_json") or ack_events[0].get("payload") or {}
    assert payload.get("reason") == "cancelled_by_user"


def test_duplicate_pause_resume_requests_are_already_applied(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-idem-pause-resume"
    _create_run(db, run_id)
    engine = WorkflowEngine(db)

    first_pause = engine.pause(run_id)
    second_pause = engine.pause(run_id)
    first_resume = engine.resume(run_id)
    second_resume = engine.resume(run_id)

    assert first_pause == "applied"
    assert second_pause == "already_applied"
    assert first_resume == "applied"
    assert second_resume == "already_applied"


def test_pause_rejects_terminal_run_without_mutating_status(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-pause-terminal"
    _create_run(db, run_id)
    db.update_run_status(run_id, status="succeeded")

    engine = WorkflowEngine(db)
    result = engine.pause(run_id)

    run = db.get_run(run_id)
    assert result == "invalid_state"
    assert run is not None
    assert run.status == "succeeded"
    assert db.get_events(run_id, types=["run_paused"]) == []


def test_resume_requires_paused_status_without_mutating_run(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-resume-invalid"
    _create_run(db, run_id)
    db.update_run_status(run_id, status="waiting_human", status_reason="awaiting_review")

    engine = WorkflowEngine(db)
    result = engine.resume(run_id)

    run = db.get_run(run_id)
    assert result == "invalid_state"
    assert run is not None
    assert run.status == "waiting_human"
    assert db.get_events(run_id, types=["run_resumed"]) == []


def test_cancel_rejects_terminal_run_without_mutating_status(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-cancel-terminal"
    _create_run(db, run_id)
    db.update_run_status(run_id, status="succeeded")

    engine = WorkflowEngine(db)
    result = engine.cancel(run_id)

    run = db.get_run(run_id)
    assert result == "invalid_state"
    assert run is not None
    assert run.status == "succeeded"
    assert db.get_events(run_id, types=["run_cancelled"]) == []


@pytest.mark.asyncio
async def test_control_run_cancel_duplicate_returns_already_applied(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-idem-control-cancel"
    _create_run(db, run_id)
    # Route checks owner-or-admin and tenant.
    db._conn.execute("UPDATE workflow_runs SET user_id = ? WHERE run_id = ?", ("42", run_id))
    db._conn.commit()

    request = type("Req", (), {"headers": {}})()
    user = User(id=42, username="tester", email="t@e.com", is_active=True, is_admin=True, tenant_id="default")

    first = await workflows_ep.control_run(run_id=run_id, action="cancel", request=request, current_user=user, db=db)
    second = await workflows_ep.control_run(run_id=run_id, action="cancel", request=request, current_user=user, db=db)

    assert first.get("ok") is True
    assert second.get("ok") is True
    assert first.get("result") == "applied"
    assert second.get("result") == "already_applied"
