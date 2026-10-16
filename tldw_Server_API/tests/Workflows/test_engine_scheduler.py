import os

os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("SINGLE_USER_API_KEY", "test-key")

from loguru import logger

logger.remove()

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine, WorkflowScheduler, RunMode
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowRun, WorkflowsDatabase
from tldw_Server_API.app.api.v1.schemas.workflows import RunRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.api.v1.endpoints.workflows import (
    _wait_for_run_completion,
    _wait_for_run_visibility,
    run_saved,
)


TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def _workflow_run(run_id: str, status: str) -> WorkflowRun:
    return WorkflowRun(
        run_id=run_id,
        tenant_id="tenant",
        workflow_id=1,
        status=status,
        status_reason=None,
        user_id="user",
        inputs_json="{}",
        outputs_json=None,
        error=None,
        duration_ms=None,
        created_at="2026-01-01T00:00:00Z",
        started_at=None,
        ended_at=None,
        definition_version=1,
        definition_snapshot_json="{}",
        idempotency_key=None,
        session_id=None,
    )


class _TransientMissingRunDB:
    def __init__(self) -> None:
        self.calls = 0

    def get_run(self, run_id: str):
        self.calls += 1
        if self.calls == 1:
            return None
        return _workflow_run(run_id, "succeeded")


@pytest.fixture(autouse=True)
def reset_engine_state(monkeypatch):
    WorkflowScheduler._inst = None
    WorkflowEngine._RUN_SECRETS.clear()
    yield
    WorkflowScheduler._inst = None
    WorkflowEngine._RUN_SECRETS.clear()


@pytest.fixture
def workflows_db(tmp_path: Path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(db_path=str(db_path))
    try:
        yield db
    finally:
        try:
            db._conn.close()
        except Exception:
            _ = None


def _wait_for_status(db: WorkflowsDatabase, run_id: str, timeout: float = 10.0) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        with sqlite3.connect(db.db_path) as read_conn:
            row = read_conn.execute(
                "SELECT status FROM workflow_runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        last_status = row[0] if row else None
        if last_status in TERMINAL_STATES.union({"waiting_human", "waiting_approval"}):
            return str(last_status)
        time.sleep(0.05)
    raise AssertionError(
        "Run did not reach a terminal or waiting state within the timeout "
        f"(run_id={run_id!r}, last_status={last_status!r})"
    )


def _wait_for_scheduler_idle(scheduler: WorkflowScheduler, timeout: float = 10.0) -> dict[str, int]:
    deadline = time.time() + timeout
    last_stats = scheduler.stats()
    while time.time() < deadline:
        last_stats = scheduler.stats()
        if (
            last_stats["queue_depth"] == 0
            and last_stats["active_tenants"] == 0
            and last_stats["active_workflows"] == 0
        ):
            return last_stats
        time.sleep(0.02)
    return last_stats


def test_scheduler_releases_slot_on_step_failure(workflows_db: WorkflowsDatabase):
    scheduler = WorkflowScheduler.instance()
    definition = {
        "name": "force-failure",
        "steps": [
            {
                "id": "step1",
                "type": "prompt",
                "config": {"template": "oops", "force_error": True},
            }
        ],
    }
    run_id = "fail-run"
    workflows_db.create_run(
        run_id=run_id,
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    engine = WorkflowEngine(workflows_db)
    engine.submit(run_id, RunMode.ASYNC)

    status = _wait_for_status(workflows_db, run_id)
    assert status == "failed"

    stats = _wait_for_scheduler_idle(scheduler)
    assert stats["active_tenants"] == 0
    assert stats["active_workflows"] == 0
    assert stats["queue_depth"] == 0


def test_drain_pending_reports_active_run(
    workflows_db: WorkflowsDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = WorkflowScheduler.instance()
    spawned = []

    def fake_spawn(coro):
        spawned.append(coro)
        coro.close()

    monkeypatch.setattr(WorkflowScheduler, "_spawn", staticmethod(fake_spawn))

    definition = {
        "name": "active-run",
        "steps": [{"id": "step1", "type": "prompt", "config": {"template": "hi"}}],
    }
    run_id = "active-run"
    workflows_db.create_run(
        run_id=run_id,
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )

    WorkflowEngine(workflows_db).submit(run_id, RunMode.ASYNC)

    assert spawned
    assert scheduler.stats()["active_tenants"] == 1
    assert scheduler.drain_pending(run_id) is True


def test_waiting_run_keeps_secrets_and_releases_slot(workflows_db: WorkflowsDatabase):
    scheduler = WorkflowScheduler.instance()
    definition = {
        "name": "await-approval",
        "steps": [
            {
                "id": "wait",
                "type": "wait_for_human",
                "config": {"assigned_to_user_id": "user"},
                "on_success": "next",
            },
            {"id": "next", "type": "log", "config": {"message": "done"}},
        ],
    }
    run_id = "wait-run"
    workflows_db.create_run(
        run_id=run_id,
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    WorkflowEngine.set_run_secrets(run_id, {"fixture": "value"})
    engine = WorkflowEngine(workflows_db)
    engine.submit(run_id, RunMode.ASYNC)

    status = _wait_for_status(workflows_db, run_id)
    assert status == "waiting_human"
    assert WorkflowEngine._RUN_SECRETS.get(run_id) is not None

    stats = _wait_for_scheduler_idle(scheduler, timeout=10.0)
    assert stats["active_tenants"] == 0
    assert stats["active_workflows"] == 0


def test_continue_run_clears_secrets(workflows_db: WorkflowsDatabase):
    scheduler = WorkflowScheduler.instance()
    definition = {
        "name": "resume-flow",
        "steps": [
            {
                "id": "wait",
                "type": "wait_for_human",
                "config": {"assigned_to_user_id": "user"},
                "on_success": "next",
            },
            {"id": "next", "type": "log", "config": {"message": "finished"}},
        ],
    }
    run_id = "resume-run"
    workflows_db.create_run(
        run_id=run_id,
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    WorkflowEngine.set_run_secrets(run_id, {"fixture": "value"})
    engine = WorkflowEngine(workflows_db)
    engine.submit(run_id, RunMode.ASYNC)
    status = _wait_for_status(workflows_db, run_id)
    assert status == "waiting_human"

    asyncio.run(engine.continue_run(run_id, after_step_id="wait", last_outputs={"approved": True}))

    status = _wait_for_status(workflows_db, run_id)
    assert status == "succeeded"
    assert WorkflowEngine._RUN_SECRETS.get(run_id) is None
    stats = _wait_for_scheduler_idle(scheduler)
    assert stats["active_tenants"] == 0
    assert stats["active_workflows"] == 0


def test_continue_run_recovers_succeeded_step_outputs_for_exact_resume(
    workflows_db: WorkflowsDatabase,
    monkeypatch,
):
    definition = {
        "name": "audio-resume",
        "steps": [
            {"id": "compose_script", "type": "log", "config": {}},
            {"id": "generate_audio", "type": "log", "config": {}},
        ],
    }
    run_id = "audio-resume-run"
    workflows_db.create_run(
        run_id=run_id,
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    workflows_db.create_step_run(
        step_run_id="compose-success",
        tenant_id="tenant",
        run_id=run_id,
        step_id="compose_script",
        name="compose_script",
        step_type="log",
    )
    workflows_db.complete_step_run(
        step_run_id="compose-success",
        outputs={"sections": [{"speaker": "HOST", "text": "Ready script"}]},
    )
    workflows_db.update_run_status(run_id, "failed", error="tts failed")
    captured: dict[str, object] = {}

    async def fake_adapter(self, step_type, config, context, last_outputs, run_id, step_run_id=None):
        captured.update(context)
        return {"artifact_id": "audio-final"}

    monkeypatch.setattr(WorkflowEngine, "_run_step_adapter", fake_adapter)
    engine = WorkflowEngine(workflows_db)

    asyncio.run(
        engine.continue_run(
            run_id,
            after_step_id="",
            next_step_id="generate_audio",
            retry_resume=True,
        )
    )

    assert captured["compose_script"] == {
        "sections": [{"speaker": "HOST", "text": "Ready script"}]
    }
    assert workflows_db.get_run(run_id).status == "succeeded"


def test_adapter_returned_research_checkpoint_wait_sets_run_waiting_human_without_human_timeout(
    workflows_db: WorkflowsDatabase,
    monkeypatch,
):
    scheduler = WorkflowScheduler.instance()
    scheduled_timeouts: list[tuple[str, str, int, str | None]] = []

    async def _fake_adapter(_config, _context):
        return {
            "__status__": "waiting_human",
            "reason": "research_checkpoint",
            "run_id": "research-session-9",
            "research_phase": "awaiting_source_review",
            "research_control_state": "running",
            "research_checkpoint_id": "checkpoint-3",
            "research_checkpoint_type": "sources_review",
            "research_console_url": "/research?run=research-session-9",
            "active_poll_seconds": 0.25,
        }

    def _fake_get_adapter(name: str):
        if name == "fake_research_wait":
            return _fake_adapter
        from tldw_Server_API.app.core.Workflows.adapters import get_adapter as _real_get_adapter

        return _real_get_adapter(name)

    def _fake_schedule(self, run_id: str, step_id: str, timeout_seconds: int, on_timeout: str | None):
        scheduled_timeouts.append((run_id, step_id, timeout_seconds, on_timeout))

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.engine.get_adapter",
        _fake_get_adapter,
    )
    monkeypatch.setattr(
        WorkflowEngine,
        "_schedule_human_timeout",
        _fake_schedule,
    )

    definition = {
        "name": "adapter-wait-research-checkpoint",
        "steps": [
            {
                "id": "wait",
                "type": "fake_research_wait",
                "config": {"timeout_seconds": 30},
            },
        ],
    }
    run_id = "adapter-wait-run"
    workflows_db.create_run(
        run_id=run_id,
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    engine = WorkflowEngine(workflows_db)
    engine.submit(run_id, RunMode.ASYNC)

    status = _wait_for_status(workflows_db, run_id)
    assert status == "waiting_human"

    step_run = workflows_db.get_latest_step_run(run_id=run_id, step_id="wait")
    assert step_run is not None
    assert step_run["status"] == "waiting_human"
    assert scheduled_timeouts == []

    stats = _wait_for_scheduler_idle(scheduler)
    assert stats["active_tenants"] == 0
    assert stats["active_workflows"] == 0


def test_run_saved_sync_waits_for_completion(workflows_db: WorkflowsDatabase, monkeypatch):
    definition_doc = {
        "name": "sync-run",
        "steps": [{"id": "log", "type": "log", "config": {"message": "hi"}}],
    }
    workflow_id = workflows_db.create_definition(
        tenant_id="tenant",
        name="sync-run",
        version=1,
        owner_id="user",
        visibility="private",
        description=None,
        tags=None,
        definition=definition_doc,
    )
    user = User(id="user", username="tester", tenant_id="tenant")
    request_body = RunRequest(inputs={})
    response = asyncio.run(
        run_saved(
            workflow_id=workflow_id,
            mode="sync",
            request=None,
            body=request_body,
            current_user=user,
            db=workflows_db,
            audit_service=None,
        )
    )
    assert response.status == "succeeded"
    stats = _wait_for_scheduler_idle(WorkflowScheduler.instance())
    assert stats["active_tenants"] == 0
    assert stats["active_workflows"] == 0


def test_wait_for_run_completion_tolerates_transient_missing_run():
    db = _TransientMissingRunDB()

    run = asyncio.run(
        _wait_for_run_completion(
            db,
            "transient-run",
            timeout_seconds=1.0,
            poll_interval=0.001,
        )
    )

    assert run.status == "succeeded"
    assert db.calls == 2


def test_wait_for_run_visibility_tolerates_transient_missing_run():
    db = _TransientMissingRunDB()

    run = asyncio.run(
        _wait_for_run_visibility(
            db,
            "transient-run",
            grace_seconds=1.0,
            poll_interval=0.001,
        )
    )

    assert run.status == "succeeded"
    assert db.calls == 2


def test_scheduler_releases_admission_when_owner_capture_is_missing(
    workflows_db: WorkflowsDatabase, monkeypatch: pytest.MonkeyPatch
):
    """Release the admitted slot even when the engine cannot re-read its owner."""
    scheduler = WorkflowScheduler.instance()
    engine = WorkflowEngine(workflows_db)
    tenant = "admitted-tenant"
    definition = {
        "name": "missing-owner-capture",
        "steps": [{"id": "log", "type": "log", "config": {"message": "done"}}],
    }
    workflow_id = workflows_db.create_definition(
        tenant_id=tenant,
        name="missing-owner-capture",
        version=1,
        owner_id="user",
        visibility="private",
        description=None,
        tags=None,
        definition=definition,
    )
    run_id = "missing-owner-capture"
    workflows_db.create_run(
        run_id=run_id,
        tenant_id=tenant,
        user_id="user",
        inputs={},
        workflow_id=workflow_id,
        definition_version=1,
        definition_snapshot=definition,
    )
    spawned: list = []
    injected_reads = []
    accounting_reads = []
    notifications = []
    notified_scheduler = []
    native_get_run = workflows_db.get_run
    native_accounting_read = engine._llm_user_for_run
    native_start_impl = engine._start_run_impl
    native_notify = scheduler.notify_finished

    def read_accounting_owner(rid):
        owner = native_accounting_read(rid)
        accounting_reads.append((rid, owner))
        return owner

    async def start_with_missing_owner(*args, **kwargs):
        # Arm only after native accounting, before the implementation captures ownership.
        notified_scheduler.append(kwargs.get("scheduler_for_notify"))
        assert accounting_reads == [(run_id, "user")]
        with monkeypatch.context() as owner_patch:

            def missing_owner_once(rid):
                owner_patch.setattr(workflows_db, "get_run", native_get_run)
                injected_reads.append(rid)
                return None

            owner_patch.setattr(workflows_db, "get_run", missing_owner_once)
            await native_start_impl(*args, **kwargs)

    def notify(tenant_id, admitted_workflow_id, run_id=None):
        notifications.append(
            {
                "same_scheduler": scheduler is notified_scheduler[0],
                "tenant": tenant_id,
                "workflow_id": admitted_workflow_id,
                "run_id": run_id,
            }
        )
        native_notify(tenant_id, admitted_workflow_id, run_id=run_id)

    monkeypatch.setattr(WorkflowScheduler, "_spawn", staticmethod(spawned.append))
    monkeypatch.setattr(engine, "_llm_user_for_run", read_accounting_owner)
    monkeypatch.setattr(engine, "_start_run_impl", start_with_missing_owner)
    monkeypatch.setattr(scheduler, "notify_finished", notify)
    try:
        engine.submit(run_id, RunMode.SYNC)
        assert len(spawned) == 1
        assert scheduler.stats() == {
            "queue_depth": 0,
            "active_tenants": 1,
            "active_workflows": 1,
        }
        assert tenant != engine.config.tenant_id
        asyncio.run(spawned[0])
    finally:
        for coroutine in spawned:
            coroutine.close()

    assert injected_reads == [run_id]
    persisted_run = native_get_run(run_id)
    assert persisted_run is not None
    assert persisted_run.status == "succeeded"
    assert len(notifications) == 1
    assert notifications[0]["same_scheduler"] is True
    assert scheduler.stats() == {
        "queue_depth": 0,
        "active_tenants": 0,
        "active_workflows": 0,
    }, str(
        {
            "admitted": {
                "run_id": run_id,
                "tenant": tenant,
                "workflow_id": workflow_id,
            },
            "notifications": notifications,
            "stats": scheduler.stats(),
            "active_runs": scheduler._active_runs,
        }
    )


@pytest.fixture
def scheduler_admissions(workflows_db: WorkflowsDatabase, monkeypatch):
    """Hold native run coroutines while exercising admission and release."""
    scheduler = WorkflowScheduler.instance()
    engine = WorkflowEngine(workflows_db)
    spawned: list = []
    definition = {
        "name": "admission",
        "steps": [{"id": "log", "type": "log", "config": {"message": "done"}}],
    }
    workflow_id = workflows_db.create_definition(
        tenant_id="tenant",
        name="admission",
        version=1,
        owner_id="user",
        visibility="private",
        description=None,
        tags=None,
        definition=definition,
    )

    def create_run(run_id, tenant="tenant", with_workflow=True):
        workflows_db.create_run(
            run_id=run_id,
            tenant_id=tenant,
            user_id="user",
            inputs={},
            workflow_id=workflow_id if with_workflow else None,
            definition_version=1,
            definition_snapshot=definition,
        )
        return workflow_id if with_workflow else None

    monkeypatch.setattr(WorkflowScheduler, "_spawn", staticmethod(spawned.append))
    try:
        yield scheduler, engine, spawned, create_run
    finally:
        for coroutine in spawned:
            coroutine.close()


@pytest.mark.parametrize("tenant", ["tenant", ""])
@pytest.mark.parametrize("with_workflow", [True, False])
@pytest.mark.parametrize("matching_keys", [True, False])
def test_scheduler_releases_recorded_admission(scheduler_admissions, tenant, with_workflow, matching_keys):
    scheduler, engine, _, create_run = scheduler_admissions
    workflow_id = create_run("owned", tenant, with_workflow)
    engine.submit("owned")

    scheduler.notify_finished(
        tenant if matching_keys else "wrong",
        workflow_id if matching_keys else -1,
        run_id="owned",
    )

    assert scheduler.stats() == {
        "queue_depth": 0,
        "active_tenants": 0,
        "active_workflows": 0,
    }
    assert scheduler.drain_pending("owned") is False


@pytest.mark.parametrize("notification", ["unknown", "duplicate", "idless"])
def test_scheduler_unowned_completion_keeps_other_admissions(scheduler_admissions, notification):
    scheduler, engine, spawned, create_run = scheduler_admissions
    scheduler.tenant_limit = 1
    workflow_id = create_run("first")
    create_run("second")
    create_run("third")
    for run_id in ["first", "second", "third"]:
        engine.submit(run_id)
    if notification == "duplicate":
        scheduler.notify_finished("tenant", workflow_id, run_id="first")
    before = scheduler.stats()
    started = len(spawned)

    scheduler.notify_finished(
        "tenant",
        workflow_id,
        run_id={"unknown": "absent", "duplicate": "first", "idless": None}[notification],
    )

    assert scheduler.stats() == before
    assert len(spawned) == started


@pytest.mark.parametrize("duplicate", ["active", "queued"])
def test_scheduler_duplicate_submission_owns_one_slot(scheduler_admissions, duplicate):
    scheduler, engine, spawned, create_run = scheduler_admissions
    scheduler.tenant_limit = 1
    workflow_id = create_run("first")
    create_run("second")
    engine.submit("first")
    if duplicate == "queued":
        engine.submit("second")
    engine.submit("first" if duplicate == "active" else "second")
    assert len(spawned) == 1
    assert scheduler.queue_depth() == (1 if duplicate == "queued" else 0)

    scheduler.notify_finished("tenant", workflow_id, run_id="first")
    if duplicate == "queued":
        assert len(spawned) == 2
        assert scheduler.queue_depth() == 0
        scheduler.notify_finished("tenant", workflow_id, run_id="second")
    assert scheduler.stats() == {
        "queue_depth": 0,
        "active_tenants": 0,
        "active_workflows": 0,
    }


def test_scheduler_releases_runs_in_fifo_order(scheduler_admissions, workflows_db: WorkflowsDatabase):
    scheduler, engine, spawned, create_run = scheduler_admissions
    scheduler.tenant_limit = 1
    for run_id in ["first", "second", "third"]:
        create_run(run_id)
        engine.submit(run_id)
    asyncio.run(spawned[0])
    assert len(spawned) == 2
    asyncio.run(spawned[1])
    second_run = workflows_db.get_run("second")
    assert second_run is not None
    assert second_run.status == "succeeded"
    third_run = workflows_db.get_run("third")
    assert third_run is not None
    assert third_run.status == "queued"
    assert len(spawned) == 3
    asyncio.run(spawned[2])
    assert scheduler.stats() == {
        "queue_depth": 0,
        "active_tenants": 0,
        "active_workflows": 0,
    }


def test_scheduler_drain_preserves_active_and_removes_queued_run(scheduler_admissions, workflows_db: WorkflowsDatabase):
    scheduler, engine, spawned, create_run = scheduler_admissions
    scheduler.tenant_limit = 1
    workflow_id = create_run("first")
    create_run("second")
    create_run("third")
    for run_id in ["first", "second", "third"]:
        engine.submit(run_id)
    assert scheduler.drain_pending("first") is True
    assert scheduler.drain_pending("second") is True
    assert scheduler.queue_depth() == 1
    scheduler.notify_finished("tenant", workflow_id, run_id="first")
    assert len(spawned) == 2
    asyncio.run(spawned[1])
    third_run = workflows_db.get_run("third")
    assert third_run is not None
    assert third_run.status == "succeeded"
    second_run = workflows_db.get_run("second")
    assert second_run is not None
    assert second_run.status == "queued"
    assert scheduler.stats() == {
        "queue_depth": 0,
        "active_tenants": 0,
        "active_workflows": 0,
    }


@pytest.mark.parametrize("entry", ["start", "continue"])
def test_unadmitted_engine_completion_cannot_release_other_run(
    scheduler_admissions, workflows_db: WorkflowsDatabase, entry
):
    scheduler, engine, spawned, create_run = scheduler_admissions
    scheduler.tenant_limit = 1
    create_run("admitted")
    create_run("direct")
    create_run("queued")
    engine.submit("admitted")
    engine.submit("queued")
    before = scheduler.stats()

    if entry == "continue":
        workflows_db.update_run_status("direct", status="waiting_human")
        asyncio.run(engine.continue_run("direct", after_step_id="log"))
    else:
        asyncio.run(engine.start_run("direct"))

    direct_run = workflows_db.get_run("direct")
    assert direct_run is not None
    assert direct_run.status == "succeeded"
    assert scheduler.stats() == before
    assert len(spawned) == 1


def test_scheduler_skips_blocked_queue_head_without_losing_fifo_scan(
    scheduler_admissions, workflows_db: WorkflowsDatabase
):
    scheduler, engine, spawned, create_run = scheduler_admissions
    scheduler.tenant_limit = 1
    workflow_id = create_run("workflow-owner", tenant="tenant-a")
    create_run("tenant-owner", tenant="tenant-b", with_workflow=False)
    create_run("blocked", tenant="tenant-b")
    create_run("eligible", tenant="tenant-c")
    for run_id in ["workflow-owner", "tenant-owner", "blocked", "eligible"]:
        engine.submit(run_id)
    assert len(spawned) == 2
    assert scheduler.queue_depth() == 2

    scheduler.notify_finished("tenant-a", workflow_id, run_id="workflow-owner")
    assert len(spawned) == 3
    asyncio.run(spawned[2])
    eligible_run = workflows_db.get_run("eligible")
    assert eligible_run is not None
    assert eligible_run.status == "succeeded"
    blocked_run = workflows_db.get_run("blocked")
    assert blocked_run is not None
    assert blocked_run.status == "queued"
    scheduler.notify_finished("tenant-b", None, run_id="tenant-owner")
    assert len(spawned) == 4
    asyncio.run(spawned[3])
    blocked_run = workflows_db.get_run("blocked")
    assert blocked_run is not None
    assert blocked_run.status == "succeeded"
    assert scheduler.stats() == {
        "queue_depth": 0,
        "active_tenants": 0,
        "active_workflows": 0,
    }
