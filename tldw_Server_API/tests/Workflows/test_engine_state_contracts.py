import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows import engine as engine_mod
from tldw_Server_API.app.core.Workflows.engine import _is_allowed_transition


def test_state_contract_rejects_invalid_transition() -> None:
    assert _is_allowed_transition("running", "queued") is False


def test_state_contract_allows_pause_transitions() -> None:
    assert _is_allowed_transition("queued", "paused") is True
    assert _is_allowed_transition("running", "paused") is True
    assert _is_allowed_transition("paused", "running") is True


def test_append_event_warns_on_persistence_failure(tmp_path, monkeypatch) -> None:
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "append-warning", "version": 1, "steps": []},
    )

    def _raise_append_event(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("event-store-down")

    warnings: list[tuple[object, ...]] = []

    monkeypatch.setattr(db, "append_event", _raise_append_event)
    monkeypatch.setattr(
        engine_mod.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    engine = engine_mod.WorkflowEngine(db)
    engine._append_event(run_id, "run_started", {"mode": "async"})

    assert warnings
    assert "append_event failed" in str(warnings[0][0])


def test_control_transition_reports_missing_run(tmp_path) -> None:
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    engine = engine_mod.WorkflowEngine(db)

    assert engine._control_transition(
        "missing-run",
        target_status="cancelled",
        op_key="missing-run:cancel",
    ) == ("not_found", "unknown")


def test_engine_now_iso_is_timezone_aware() -> None:
    parsed = datetime.fromisoformat(engine_mod.WorkflowEngine._now_iso())

    assert parsed.tzinfo is not None


def test_engine_duration_ms_since_accepts_timezone_aware_started_at() -> None:
    started_at = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()

    duration_ms = engine_mod.WorkflowEngine._duration_ms_since(started_at)

    assert duration_ms is not None
    assert duration_ms >= 0


@pytest.mark.asyncio
async def test_invalid_transition_sets_invariant_violation(tmp_path, monkeypatch) -> None:
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "invalid-transition", "version": 1, "steps": []},
    )

    monkeypatch.setattr(engine_mod, "_is_allowed_transition", lambda *_: False)

    engine = engine_mod.WorkflowEngine(db)
    await engine.start_run(run_id)

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "failed"
    assert run.status_reason == "invariant_violation"

    events = db.get_events(run_id)
    rejected = [event for event in events if event["event_type"] == "transition_rejected"]
    assert rejected


@pytest.mark.asyncio
async def test_success_transition_is_guarded(tmp_path, monkeypatch) -> None:
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "guarded-success",
            "version": 1,
            "steps": [
                {"id": "s1", "type": "log", "config": {"message": "hello"}},
            ],
        },
    )

    monkeypatch.setattr(
        engine_mod,
        "_is_allowed_transition",
        lambda current, target: (current, target) == ("queued", "running"),
    )

    engine = engine_mod.WorkflowEngine(db)
    await engine.start_run(run_id)

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "failed"
    assert run.status_reason == "invariant_violation"

    rejected = [event for event in db.get_events(run_id) if event["event_type"] == "transition_rejected"]
    assert any(event["payload_json"].get("to") == "succeeded" for event in rejected)


@pytest.mark.asyncio
async def test_wait_transition_is_guarded(tmp_path, monkeypatch) -> None:
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "guarded-wait",
            "version": 1,
            "steps": [
                {
                    "id": "review",
                    "type": "wait_for_human",
                    "config": {"assigned_to_user_id": "7"},
                },
            ],
        },
    )

    monkeypatch.setattr(
        engine_mod,
        "_is_allowed_transition",
        lambda current, target: (current, target) == ("queued", "running"),
    )

    engine = engine_mod.WorkflowEngine(db)
    await engine.start_run(run_id)

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "failed"
    assert run.status_reason == "invariant_violation"

    rejected = [event for event in db.get_events(run_id) if event["event_type"] == "transition_rejected"]
    assert any(event["payload_json"].get("to") == "waiting_human" for event in rejected)
