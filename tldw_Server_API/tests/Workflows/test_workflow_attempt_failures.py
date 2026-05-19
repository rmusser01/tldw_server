import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows import engine as engine_mod


pytestmark = pytest.mark.unit


def _create_run(
    db: WorkflowsDatabase,
    *,
    run_id: str,
    step_type: str,
    retry: int,
    config: dict,
) -> None:
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": f"{step_type}-attempt-failures",
            "version": 1,
            "steps": [{"id": "s1", "type": step_type, "retry": retry, "config": config}],
        },
    )


@pytest.mark.asyncio
async def test_retrying_step_records_multiple_attempt_rows(tmp_path, monkeypatch):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-attempt-history"
    _create_run(
        db,
        run_id=run_id,
        step_type="prompt",
        retry=1,
        config={"template": "ok"},
    )

    state = {"calls": 0}

    async def _flaky_adapter(_config, _context):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("transient_network_error")
        return {"text": "ok"}

    async def _fast_sleep(_duration: float):
        return None

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _flaky_adapter)
    monkeypatch.setattr(engine_mod.asyncio, "sleep", _fast_sleep)

    engine = engine_mod.WorkflowEngine(db)
    await engine.start_run(run_id)

    attempts = db.list_step_attempts(run_id=run_id, step_id="s1")
    assert len(attempts) == 2
    assert [attempt["attempt_number"] for attempt in attempts] == [1, 2]
    assert attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["ended_at"] is not None
    assert attempts[0]["reason_code_core"] == "transient_network_error"
    assert bool(attempts[0]["retryable"]) is True
    assert attempts[0]["metadata_json"]["retry_recommendation"] == "safe"
    assert attempts[0]["metadata_json"]["blame_scope"] == "step"
    assert attempts[0]["metadata_json"]["failure_envelope"]["reason_code_core"] == "transient_network_error"
    assert attempts[1]["status"] == "succeeded"
    assert attempts[1]["ended_at"] is not None
    assert attempts[1]["reason_code_core"] is None


@pytest.mark.asyncio
async def test_unsafe_step_types_do_not_report_safe_replay(tmp_path, monkeypatch):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-unsafe-replay-guidance"
    _create_run(
        db,
        run_id=run_id,
        step_type="webhook",
        retry=0,
        config={"url": "https://example.invalid/hook"},
    )

    async def _transient_adapter(_config, _context):
        raise RuntimeError("transient_network_error")

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _transient_adapter)

    engine = engine_mod.WorkflowEngine(db)
    await engine.start_run(run_id)

    attempts = db.list_step_attempts(run_id=run_id, step_id="s1")
    assert len(attempts) == 1
    assert attempts[0]["reason_code_core"] == "transient_network_error"
    assert bool(attempts[0]["retryable"]) is True
    assert attempts[0]["metadata_json"]["retry_recommendation"] == "conditional"
    assert attempts[0]["metadata_json"]["failure_envelope"]["reason_code_core"] == "transient_network_error"
    assert attempts[0]["metadata_json"]["step_capability"]["replay_safe"] is False
