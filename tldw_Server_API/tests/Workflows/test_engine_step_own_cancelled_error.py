"""A step that raises its own CancelledError fails the run; a real cancellation still propagates (TASK-13891).

Each case runs through both step loops: ``start_run``'s and ``continue_run``'s, which is the
loop every detached task (a route, a human step's timeout, the orphan reaper) runs.
"""

import asyncio
import contextlib
import sqlite3
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows import engine as engine_module
from tldw_Server_API.app.core.Workflows.adapters.control import flow
from tldw_Server_API.app.core.Workflows.engine import RunMode, WorkflowEngine

pytestmark = pytest.mark.integration

_REAL_GET_ADAPTER = engine_module.get_adapter
_ENTRIES = ["start_run", "continue_run"]

_MAP = {"id": "s", "type": "map", "config": {"items": [0, 1], "step": {"type": "summarize", "config": {}}}}
_PARALLEL_FAIL_FAST = {
    "id": "s",
    "type": "parallel",
    "config": {"fail_fast": True, "steps": [{"type": "summarize", "config": {"n": n}} for n in range(2)]},
}
_PLAIN = {"id": "s", "type": "summarize", "config": {"n": 0}}


class PoolTimeout(psycopg.OperationalError):
    """psycopg_pool.PoolTimeout's shape: the PostgreSQL pool raises it unwrapped from getconn() when it is exhausted.

    A stand-in, because psycopg_pool comes only with the db_postgres extra, which the macOS and Windows jobs lack.
    """


_UNREADABLE_FLAG = [sqlite3.OperationalError("database is locked"), PoolTimeout("couldn't get a connection after 30.00 sec")]


def _run(db: WorkflowsDatabase, engine: WorkflowEngine, entry: str, step: dict):
    """Create a run whose step under test is ``step``, and return the coroutine that runs it."""
    first = [] if entry == "start_run" else [{"id": "first", "type": "summarize", "config": {"n": 1}}]
    db.create_run(
        run_id="run",
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "run", "version": 1, "steps": [*first, step]},
    )
    if entry == "start_run":
        return engine.start_run("run", RunMode.SYNC)
    db.update_run_status("run", status="waiting_human")  # e.g. approved: the route continues the run
    return engine.continue_run("run", after_step_id="first")


@contextlib.contextmanager
def _summarize_is(sub_step):
    """Route every ``summarize`` step and sub-step to ``sub_step``; other step types stay real."""
    with patch.object(flow, "get_adapter", return_value=sub_step), patch.object(
        engine_module, "get_adapter", side_effect=lambda t: sub_step if t == "summarize" else _REAL_GET_ADAPTER(t)
    ):
        yield


def _events(db: WorkflowsDatabase) -> list[str]:
    return [event["event_type"] for event in db.get_events("run")]


async def _raises_its_own_cancelled_error(config, context):
    """Item or step 0 raises CancelledError itself; nothing is cancelling it."""
    if context.get("item", config.get("n")) == 0:
        raise asyncio.CancelledError()
    return {"ok": True}


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("step", [_MAP, _PARALLEL_FAIL_FAST, _PLAIN], ids=["map", "parallel_fail_fast", "plain"])
async def test_a_step_that_raises_its_own_cancelled_error_fails_the_run(tmp_path, step, entry) -> None:
    """On dev the CancelledError left the engine, and the run stayed 'running' for the orphan reaper."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, step)

    with _summarize_is(_raises_its_own_cancelled_error):
        await asyncio.wait_for(run, timeout=10)

    finished = db.get_run("run")
    assert (finished.status, finished.error) == ("failed", "step_raised_cancelled_error")
    assert _events(db)[-2:] == ["step_failed", "run_failed"]


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_step_timeout_is_still_a_step_timeout(tmp_path, entry) -> None:
    """The step's own timeout cancels it, and wait_for turns that into TimeoutError, as on dev."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, {"id": "s", "type": "summarize", "timeout_seconds": 1, "config": {}})

    async def never_returns(_config, _context):
        await asyncio.Event().wait()

    with _summarize_is(never_returns):
        await asyncio.wait_for(run, timeout=10)

    finished = db.get_run("run")
    assert finished.status == "failed"
    assert "step_timeout" in _events(db)


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_cancelled_run_task_still_propagates_and_leaves_the_run_for_the_reaper(tmp_path, entry) -> None:
    """Cancelling the run's task (e.g. at shutdown) still raises CancelledError and leaves the run 'running', as on dev."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, {"id": "s", "type": "summarize", "config": {}})
    started = asyncio.Event()

    async def runs_until_cancelled(_config, _context):
        started.set()
        await asyncio.Event().wait()

    with _summarize_is(runs_until_cancelled):
        task = asyncio.create_task(run)
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert db.get_run("run").status == "running"
    assert "run_failed" not in _events(db)


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_run_cancelled_through_the_api_stays_cancelled_when_its_step_then_raises(tmp_path, entry) -> None:
    """engine.cancel() (the run-cancel route) cancels no task, so cancelling() is 0; dev kept 'cancelled', and so must this."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    engine = WorkflowEngine(db)
    run = _run(db, engine, entry, {"id": "s", "type": "summarize", "config": {}})
    started, release = asyncio.Event(), asyncio.Event()

    async def raises_after_the_cancel(_config, _context):
        started.set()
        await release.wait()
        raise asyncio.CancelledError()

    with _summarize_is(raises_after_the_cancel):
        task = asyncio.create_task(run)
        await asyncio.wait_for(started.wait(), timeout=10)
        cancelled = engine.cancel("run")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    finished = db.get_run("run")
    assert (cancelled, finished.status, finished.error) == ("applied", "cancelled", None)
    assert "run_failed" not in _events(db)


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("error", _UNREADABLE_FLAG, ids=["sqlite_locked", "pool_timeout"])
async def test_an_unreadable_cancel_flag_still_propagates_the_cancelled_error(tmp_path, error, entry) -> None:
    """If the clause cannot read the cancel flag, the step's CancelledError propagates and the run stays 'running', as on dev."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), entry, {"id": "s", "type": "summarize", "config": {}})
    read_flag, stepped = db.is_cancel_requested, asyncio.Event()

    def locked_once_the_step_runs(run_id):
        if stepped.is_set():
            raise error
        return read_flag(run_id)

    async def raises_cancelled_error(_config, _context):
        stepped.set()
        raise asyncio.CancelledError()

    with _summarize_is(raises_cancelled_error), patch.object(db, "is_cancel_requested", side_effect=locked_once_the_step_runs):
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=10)

    assert db.get_run("run").status == "running"
    assert "run_failed" not in _events(db)


async def _cancel_the_run_task(db: WorkflowsDatabase, entry: str, read_once_the_step_runs) -> BaseException | None:
    """Cancel the run's task mid-step, as shutdown does, and return what left it.

    Once the step runs, the cancel flag is read through ``read_once_the_step_runs``.
    """
    run = _run(db, WorkflowEngine(db), entry, {"id": "s", "type": "summarize", "config": {}})
    read_flag, started = db.is_cancel_requested, asyncio.Event()

    def read(run_id):
        return read_once_the_step_runs(run_id) if started.is_set() else read_flag(run_id)

    async def runs_until_cancelled(_config, _context):
        started.set()
        await asyncio.Event().wait()

    with _summarize_is(runs_until_cancelled), patch.object(db, "is_cancel_requested", side_effect=read):
        task = asyncio.create_task(run)
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        [left] = await asyncio.gather(task, return_exceptions=True)
    return left


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_cancelled_run_task_propagates_when_the_cancel_flag_cannot_be_read(tmp_path, entry) -> None:
    """A real cancel during a pool outage: before the fix the flag read's PoolTimeout escaped instead (clean review I1)."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    def pool_exhausted(_run_id):
        raise PoolTimeout("couldn't get a connection after 30.00 sec")

    left = await _cancel_the_run_task(db, entry, pool_exhausted)

    assert type(left) is asyncio.CancelledError


@pytest.mark.parametrize("entry", _ENTRIES)
async def test_a_cancelled_run_task_reads_no_cancel_flag(tmp_path, entry) -> None:
    """The flag only tells a step's own CancelledError from an API cancel; a cancelled task needs no blocking DB read."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    reads: list[str] = []

    left = await _cancel_the_run_task(db, entry, lambda run_id: reads.append(run_id) or False)

    assert type(left) is asyncio.CancelledError
    assert reads == []


async def test_a_steps_own_cancelled_error_is_recorded_on_its_span(tmp_path) -> None:
    """start_run records each step failure on its tracing span; the step's own CancelledError too (clean review M3)."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run = _run(db, WorkflowEngine(db), "start_run", _PLAIN)
    recorded = MagicMock()

    with _summarize_is(_raises_its_own_cancelled_error), patch.object(engine_module, "record_span_exception", recorded):
        await asyncio.wait_for(run, timeout=10)

    assert [str(call.args[0]) for call in recorded.call_args_list] == ["step_raised_cancelled_error"]
