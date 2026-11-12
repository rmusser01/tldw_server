"""A run that a detached engine task continues does not inherit its creator's ChaCha operation (TASK-13897).

The creator is an HTTP request (ChaChaOperationMiddleware) or a voice command, each inside
``chacha_operation(independent=True)``. That operation has closed by the time the detached
task runs a step, so a step that joins it, as any ChaCha owner without ``independent=True``
does, raised ClosedChaChaOperationError.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import operation_scope
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows import engine as engine_module
from tldw_Server_API.app.core.Workflows.engine import RunMode, WorkflowEngine

pytestmark = pytest.mark.integration

_REAL_GET_ADAPTER = engine_module.get_adapter
_JOINS = {"id": "s", "type": "summarize", "config": {}}
_FIRST = {"id": "first", "type": "log", "config": {"message": "first"}}


async def _joins_the_current_operation(_config, _context):
    """A ChaCha step that does not open an independent operation."""
    with chacha_operation():
        return {"joined": True}


def _create_run(db: WorkflowsDatabase, run_id: str, steps: list[dict]) -> None:
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": run_id, "version": 1, "steps": steps},
    )


async def _continued_by_a_request(db: WorkflowsDatabase, engine: WorkflowEngine) -> str:
    """POST /runs/{id}/retry, approve or reject: the route starts continue_run and returns."""
    _create_run(db, "run", [_FIRST, _JOINS])
    db.update_run_status("run", status="waiting_human")
    with chacha_operation(independent=True):
        asyncio.create_task(engine.continue_run("run", after_step_id="first"))  # as the route does
    return "run"


async def _continued_after_a_human_step_timed_out(db: WorkflowsDatabase, engine: WorkflowEngine) -> str:
    """A SYNC run reaches a human step, whose timeout task later continues it."""
    human = {
        "id": "h",
        "type": "wait_for_human",
        "config": {"assigned_to_user_id": "user", "timeout_seconds": 0.1},
        "on_timeout": "s",
    }
    _create_run(db, "run", [human, _JOINS])
    with chacha_operation(independent=True):
        await engine.start_run("run", RunMode.SYNC)
    return "run"


async def _continued_by_the_orphan_reaper(db: WorkflowsDatabase, engine: WorkflowEngine) -> str:
    """A SYNC run's start reaps another user's stale run and continues it."""
    _create_run(db, "orphan", [_JOINS])
    db.create_step_run(
        step_run_id="orphan:s:1", tenant_id="default", run_id="orphan", step_id="s", name="s", step_type="summarize"
    )
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    db._conn.execute(
        "UPDATE workflow_step_runs SET heartbeat_at = ?, status = 'running' WHERE step_run_id = ?",
        (stale, "orphan:s:1"),
    )
    db._conn.commit()
    _create_run(db, "run", [_FIRST])
    with chacha_operation(independent=True):
        await engine.start_run("run", RunMode.SYNC)
    return "orphan"


@pytest.mark.parametrize(
    "continue_run",
    [_continued_by_a_request, _continued_after_a_human_step_timed_out, _continued_by_the_orphan_reaper],
    ids=["request_task", "human_step_timeout", "orphan_reaper"],
)
async def test_a_run_continued_by_a_detached_task_runs_outside_its_creators_operation(tmp_path, continue_run) -> None:
    """On dev the step raised ClosedChaChaOperationError ('The ChaCha operation has finished') and the run failed."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    engine = WorkflowEngine(db)
    with patch.object(
        engine_module,
        "get_adapter",
        side_effect=lambda t: _joins_the_current_operation if t == "summarize" else _REAL_GET_ADAPTER(t),
    ):
        run_id = await continue_run(db, engine)
        for _ in range(200):  # the detached task runs after its creator has returned
            if db.get_run(run_id).status in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)

    finished = db.get_run(run_id)
    assert (finished.status, finished.error) == ("succeeded", None)
    assert finished.outputs_json == '{"joined": true}'


async def test_a_human_steps_timeout_task_starts_outside_its_creators_operation(tmp_path) -> None:
    """The task sleeps for the step's timeout, which can be days; on dev it held its creator's operation all along."""
    engine = WorkflowEngine(WorkflowsDatabase(str(tmp_path / "wf.db")))
    before = asyncio.all_tasks()
    with chacha_operation(independent=True):  # the request or voice command whose SYNC run reached the human step
        engine._schedule_human_timeout("run", "h", 3600, None)
    [timeout_task] = [task for task in asyncio.all_tasks() - before if task.get_coro().__name__ == "_timeout_task"]
    try:
        inherited = timeout_task.get_context().get(operation_scope._current_operation)
    finally:
        timeout_task.cancel()

    assert inherited is None
