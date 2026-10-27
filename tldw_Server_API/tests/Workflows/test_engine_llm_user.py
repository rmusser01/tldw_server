"""A workflow step sees its run's owner as the LLM usage user (spec §1)."""

import asyncio
import sqlite3
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


async def test_a_database_error_on_the_accounting_read_does_not_fail_the_run(tmp_path, monkeypatch) -> None:
    """A locked database on the owner read costs the run its user, not its execution."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "7")
    seen: list[tuple[int | None, str]] = []
    real_get_run = db.get_run
    calls: list[str] = []

    def _flaky_get_run(*args, **kwargs):
        """Fail on the first call, as a locked database would; then behave."""
        calls.append(run_id)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_get_run(*args, **kwargs)

    async def _adapter(_config, _context):
        """Record what the step sees."""
        seen.append((current_llm_user_id(), current_llm_call_context().operation))
        return {"ok": True}

    monkeypatch.setattr(db, "get_run", _flaky_get_run)
    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    await engine_mod.WorkflowEngine(db).start_run(run_id)
    assert seen == [(None, "workflow")]


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


async def test_a_refused_llm_call_fails_the_step_and_the_run_records_it(tmp_path, monkeypatch) -> None:
    """A run on its own thread and loop is checked at the call (spec §4b): no provider call, a failed run."""
    from types import SimpleNamespace
    from typing import Any

    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

    refuse_llm_calls(monkeypatch)
    provider_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _name: SimpleNamespace(chat=provider_calls.append)),
    )
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "7")
    done = threading.Event()

    async def _adapter(_config, _context):
        """Make an LLM call through the shared call layer, as the workflow llm step does."""
        try:
            return await chat_service.perform_chat_api_call_async(
                api_provider="openai", messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini",
                api_key="test-key", app_config={},
            )
        finally:
            done.set()

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    engine = engine_mod.WorkflowEngine(db)
    engine_mod.WorkflowScheduler._spawn(engine.start_run(run_id))  # noqa: SLF001 - the thread hand-off under test
    assert await asyncio.to_thread(done.wait, 10)
    for _ in range(100):
        run = db.get_run(run_id)
        if run is not None and run.status == "failed":
            break
        await asyncio.sleep(0.05)
    assert run is not None and run.status == "failed"
    assert (run.error or "").startswith("402") and "limit_exceeded" in run.error
    assert provider_calls == []
