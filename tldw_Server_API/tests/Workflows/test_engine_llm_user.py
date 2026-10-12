"""A workflow step sees its run's owner as the LLM usage user (spec §1)."""

import asyncio
import threading
import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    current_llm_user_id,
    llm_call_context,
)
from tldw_Server_API.app.core.Workflows import engine as engine_mod

pytestmark = pytest.mark.unit


def _create_run(db: WorkflowsDatabase, user_id: str) -> str:
    """A one-step run owned by ``user_id``."""
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id, tenant_id="default", user_id=user_id, inputs={}, workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "probe", "version": 1, "steps": [{"id": "s1", "type": "provider_step", "config": {}}]},
    )
    return run_id


async def test_a_step_awaited_directly_sees_the_run_owner_and_the_caller_keeps_its_own(tmp_path, monkeypatch) -> None:
    """Inside the run the owner is the user; afterwards the caller's user is back."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "7")
    seen: list[tuple[int | None, str]] = []

    async def _adapter(_config, _context):
        """Record what the step sees."""
        seen.append((current_llm_user_id(), current_llm_call_context().operation))
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    with llm_call_context(user_id=3):
        await engine_mod.WorkflowEngine(db).start_run(run_id)
        assert current_llm_user_id() == 3
    assert seen == [(7, "workflow")]


async def test_a_spawned_run_on_its_own_thread_sees_the_run_owner(tmp_path, monkeypatch) -> None:
    """A run started through the scheduler's thread still sees its owner."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "9")
    seen: list[tuple[int | None, str]] = []
    done = threading.Event()

    async def _adapter(_config, _context):
        """Record what the step sees, then signal."""
        seen.append((current_llm_user_id(), current_llm_call_context().operation))
        done.set()
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    engine = engine_mod.WorkflowEngine(db)
    engine_mod.WorkflowScheduler._spawn(engine.start_run(run_id))  # noqa: SLF001 - the thread hand-off under test
    assert await asyncio.to_thread(done.wait, 10)
    assert seen == [(9, "workflow")]


async def test_a_run_with_a_non_numeric_owner_runs_with_no_user(tmp_path, monkeypatch) -> None:
    """An owner that is not a number sets no user, and the run still executes."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "service-account")
    seen: list[int | None] = []

    async def _adapter(_config, _context):
        """Record what the step sees."""
        seen.append(current_llm_user_id())
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    await engine_mod.WorkflowEngine(db).start_run(run_id)
    assert seen == [None]


async def test_a_resumed_run_sees_the_run_owner(tmp_path, monkeypatch) -> None:
    """continue_run is its own entry: the step it resumes into sees the owner too."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id, tenant_id="default", user_id="7", inputs={}, workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "probe", "version": 1,
            "steps": [
                {"id": "s1", "type": "provider_step", "config": {}},
                {"id": "s2", "type": "provider_step", "config": {}},
            ],
        },
    )
    seen: list[tuple[int | None, str]] = []

    async def _adapter(_config, _context):
        """Record what the resumed step sees."""
        seen.append((current_llm_user_id(), current_llm_call_context().operation))
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    with llm_call_context(user_id=3):
        await engine_mod.WorkflowEngine(db).continue_run(run_id, after_step_id="s1", last_outputs={})
        assert current_llm_user_id() == 3
    assert seen == [(7, "workflow")]
