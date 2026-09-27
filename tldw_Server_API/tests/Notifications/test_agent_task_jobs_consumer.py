"""Agent-task Jobs consumer tests (TASK-13021).

Mirrors the reminder consumer's test shapes: real per-user databases
under a tmp USER_DB_BASE_DIR (preview-gated definition creation, real run
rows, real user notifications), an injected stub executor, and direct
``handle_agent_task_job`` invocation. No reimplementation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    DefinitionRow,
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.Scheduled_Tasks import agent_task_jobs
from tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs import (
    handle_agent_task_job,
    register_executor,
)
from tldw_Server_API.app.core.Scheduled_Tasks.execution_certification import (
    ExecutionCertification,
)

pytestmark = pytest.mark.unit

SLOT = "2026-08-21T09:00:00+00:00"


def _certified_execution() -> ExecutionCertification:
    observed_at = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
    return ExecutionCertification(
        outcome="certified",
        deployment_class_id="sha256:" + ("1" * 64),
        evidence_id="sha256:" + ("2" * 64),
        evidence_source="server_verified",
        observed_at=observed_at,
        expires_at=observed_at + timedelta(hours=24),
        reason_codes=(),
    )


@pytest.fixture()
def consumer_env(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_agent_task_consumer"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("JOBS_DB_PATH", str(base_dir / "jobs.db"))
    _orig_executors = dict(agent_task_jobs._EXECUTORS)
    agent_task_jobs._EXECUTORS.clear()
    try:
        yield
    finally:
        agent_task_jobs._EXECUTORS.clear()
        agent_task_jobs._EXECUTORS.update(_orig_executors)
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def _create_definition(
    user_id: int,
    *,
    input_config: dict[str, Any] | None = None,
    lifecycle: str = "configured",
    family: str = "recurring_question",
    notification_policy: dict[str, Any] | None = None,
) -> DefinitionRow:
    db = ScheduledTasksDatabase.for_user(user_id=user_id)
    db.ensure_schema()
    preview = db.create_preview(
        owner_id=user_id,
        mode="create",
        family=family,
        definition_id=None,
        definition_version=None,
        status="valid",
        payload_hash=f"hash-{datetime.now(timezone.utc).timestamp()}",
        normalized_config={},
        validation_errors=[],
        warnings=[],
        risk_class=None,
        visibility_policy="owner",
        schedule_preview={"kind": "daily", "at": "09:00"},
        redaction_policy={"fields": ["input.message"], "mode": "metadata_only"},
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        created_by="test",
    )
    return db.create_definition(
        owner_id=user_id,
        family=family,
        name="Daily Digest",
        description=None,
        lifecycle=lifecycle,
        health="ready",
        schedule={"kind": "daily", "at": "09:00"},
        input=input_config or {"question": "What changed today?"},
        visibility_policy="owner",
        notification_policy=notification_policy or {},
        approval_policy={},
        preview_id=preview.id,
        created_by="test",
        updated_by="test",
    )


def _job(definition: DefinitionRow, user_id: int, *, slot: str = SLOT) -> dict[str, Any]:
    return {
        "id": 77,
        "owner_user_id": user_id,
        "job_type": "agent_task_run",
        "payload": {
            "definition_id": definition.id,
            "user_id": user_id,
            "family": definition.family,
            "scheduled_for": slot,
        },
    }


def _latest_notification(user_id: int) -> Any | None:
    cdb = CollectionsDatabase.for_user(user_id=user_id)
    rows = cdb.list_user_notifications(limit=5)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Run rows, dedupe, lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_run_records_and_notifies(consumer_env) -> None:
    user_id = 1010
    definition = _create_definition(user_id)
    register_executor("recurring_question", lambda d, p: asyncio.sleep(0, result="42"))

    result = await handle_agent_task_job(_job(definition, user_id))

    assert result["status"] == "succeeded"
    assert result.get("deduped") is None
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    slot_key = datetime.fromisoformat(SLOT).astimezone(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    run = sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id, run_slot_key=slot_key
    )
    assert run is not None and run["status"] == "succeeded"
    assert run["result_summary"] == "42"

    notification = _latest_notification(user_id)
    assert notification is not None
    assert notification.kind == "automation_run_succeeded"
    assert "Daily Digest" in str(notification.title)
    # Traceability: source_job_id references the JOBS pipeline id, not the
    # run row id (review #6).
    assert str(notification.source_job_id) == "77"


@pytest.mark.asyncio
async def test_redelivered_job_for_terminal_slot_is_recorded_noop(consumer_env) -> None:
    user_id = 1011
    definition = _create_definition(user_id)
    register_executor("recurring_question", lambda d, p: asyncio.sleep(0, result="42"))
    job = _job(definition, user_id)

    first = await handle_agent_task_job(job)
    second = await handle_agent_task_job(job)

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert second.get("deduped") is True


@pytest.mark.asyncio
async def test_redelivered_terminal_run_dedupes_when_definition_is_unavailable(
    consumer_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return the recorded terminal run when its definition later disappears."""
    user_id = 1022
    definition = _create_definition(user_id)
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    executor = AsyncMock(return_value="42")
    register_executor("recurring_question", executor)
    job = _job(definition, user_id)

    first = await handle_agent_task_job(job, scheduled_db=sdb)
    monkeypatch.setattr(sdb, "get_definition", Mock(return_value=None))
    second = await handle_agent_task_job(job, scheduled_db=sdb)

    assert first["status"] == "succeeded"
    assert second == {
        "status": "succeeded",
        "definition_id": definition.id,
        "run_id": first["run_id"],
        "deduped": True,
    }
    assert executor.await_count == 1


@pytest.mark.asyncio
async def test_injected_database_cannot_dedupe_another_owners_run(
    consumer_env: None,
) -> None:
    """Conceal recorded runs when an injected repository belongs to another owner."""
    definition_owner_id = 1023
    job_owner_id = 1024
    definition = _create_definition(definition_owner_id)
    sdb = ScheduledTasksDatabase.for_user(user_id=definition_owner_id)
    executor = AsyncMock(return_value="42")
    register_executor("recurring_question", executor)

    owner_result = await handle_agent_task_job(
        _job(definition, definition_owner_id), scheduled_db=sdb
    )
    result = await handle_agent_task_job(
        _job(definition, job_owner_id), scheduled_db=sdb
    )

    assert owner_result["status"] == "succeeded"
    assert result == {
        "status": "skipped",
        "definition_id": definition.id,
        "run_id": None,
        "reason": "definition_missing",
    }
    assert executor.await_count == 1


@pytest.mark.asyncio
async def test_paused_definition_skips_with_reason(consumer_env) -> None:
    user_id = 1012
    definition = _create_definition(user_id, lifecycle="paused")
    register_executor("recurring_question", lambda d, p: asyncio.sleep(0, result="never"))

    result = await handle_agent_task_job(_job(definition, user_id))

    assert result["status"] == "skipped"
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id, run_slot_key=SLOT
    )
    assert run is not None
    assert run["status"] == "skipped"
    assert run["error"] == "definition_paused"


@pytest.mark.asyncio
async def test_missing_definition_skips_without_side_effects(
    consumer_env: None,
) -> None:
    """Skip a missing definition without creating dependent resources."""
    user_id = 1013
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    sdb.ensure_schema()
    missing_definition_id = "definition-never-created"
    executor = AsyncMock(return_value="must not execute")
    register_executor("recurring_question", executor)
    log_messages: list[str] = []
    sink_id = agent_task_jobs.logger.add(log_messages.append, format="{message}")

    job = {
        "id": 78,
        "owner_user_id": user_id,
        "job_type": "agent_task_run",
        # A definition id that was never created: unique to this test.
        "payload": {
            "definition_id": missing_definition_id,
            "user_id": user_id,
            "family": "recurring_question",
            "scheduled_for": SLOT,
            "prompt": "private prompt must not be logged",
        },
    }
    try:
        result = await handle_agent_task_job(job)
    finally:
        agent_task_jobs.logger.remove(sink_id)

    assert result == {
        "status": "skipped",
        "definition_id": missing_definition_id,
        "run_id": None,
        "reason": "definition_missing",
    }
    assert sdb.get_scheduled_task_run_by_slot(
        definition_id=missing_definition_id, run_slot_key=SLOT
    ) is None
    audits, total = sdb.list_audit_events(
        owner_id=user_id, definition_id=missing_definition_id
    )
    assert audits == []
    assert total == 0
    assert _latest_notification(user_id) is None
    executor.assert_not_awaited()
    assert [message.strip() for message in log_messages] == [
        "Automation Job skipped because its definition is unavailable "
        f"(definition_id={missing_definition_id} user_id={user_id} job_id=78)"
    ]
    assert "private prompt" not in log_messages[0]


@pytest.mark.asyncio
async def test_cross_owner_definition_is_treated_as_missing(
    consumer_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conceal another owner's definition as an unavailable resource."""
    definition_owner_id = 1020
    job_owner_id = 1021
    definition = _create_definition(definition_owner_id)
    sdb = ScheduledTasksDatabase.for_user(user_id=job_owner_id)
    sdb.ensure_schema()
    executor = AsyncMock(return_value="must not execute")
    register_executor("recurring_question", executor)
    fake_logger = Mock()
    monkeypatch.setattr(agent_task_jobs, "logger", fake_logger)

    result = await handle_agent_task_job(
        {
            "id": 79,
            "owner_user_id": job_owner_id,
            "job_type": "agent_task_run",
            "payload": {
                "definition_id": definition.id,
                "user_id": job_owner_id,
                "family": definition.family,
                "scheduled_for": SLOT,
                "prompt": "private prompt must not be logged",
            },
        }
    )

    assert result == {
        "status": "skipped",
        "definition_id": definition.id,
        "run_id": None,
        "reason": "definition_missing",
    }
    assert sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id, run_slot_key=SLOT
    ) is None
    audits, total = sdb.list_audit_events(
        owner_id=job_owner_id, definition_id=definition.id
    )
    assert audits == []
    assert total == 0
    assert _latest_notification(job_owner_id) is None
    executor.assert_not_awaited()
    fake_logger.warning.assert_called_once_with(
        "Automation Job skipped because its definition is unavailable "
        "(definition_id={definition_id} user_id={user_id} job_id={job_id})",
        definition_id=definition.id,
        user_id=job_owner_id,
        job_id=79,
    )


# ---------------------------------------------------------------------------
# Phase-1 boundary (enforced by the consumer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queued_agent_job_is_blocked_before_registered_executor(
    consumer_env,
) -> None:
    user_id = 1025
    definition = _create_definition(
        user_id,
        family="agent_task",
        input_config={
            "agent_ref": "agent:triage",
            "message_ref": "redacted:example",
        },
    )
    executor = AsyncMock(return_value="must not execute")
    register_executor("agent_task", executor)

    result = await handle_agent_task_job(
        _job(definition, user_id),
        execution_certification_resolver=_certified_execution,
        execution_stack_ready_resolver=lambda: False,
    )

    assert result["status"] == "skipped"  # nosec B101
    assert result["definition_id"] == definition.id  # nosec B101
    assert result["run_id"] is not None  # nosec B101
    assert result["reason"] == "agent_execution_stack_unimplemented"  # nosec B101
    executor.assert_not_awaited()
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id,
        run_slot_key=SLOT,
    )
    assert run is not None  # nosec B101
    assert run["status"] == "skipped"  # nosec B101
    assert run["error"] == "agent_execution_stack_unimplemented"  # nosec B101
    audits, _total = sdb.list_audit_events(
        owner_id=user_id,
        definition_id=definition.id,
    )
    assert any(event.event_type == "run_skipped" for event in audits)  # nosec B101


@pytest.mark.asyncio
async def test_tool_requesting_config_skips_with_actionable_reason(consumer_env) -> None:
    user_id = 1014
    definition = _create_definition(
        user_id, input_config={"question": "q", "tools": ["fs_read", "http_fetch"]}
    )
    register_executor("recurring_question", lambda d, p: asyncio.sleep(0, result="never"))

    result = await handle_agent_task_job(_job(definition, user_id))

    assert result["status"] == "skipped"
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id, run_slot_key=SLOT
    )
    assert run["error"] == "tools_not_executable_in_phase1"
    assert "approval-escalation" in (run["result_summary"] or "")


@pytest.mark.asyncio
async def test_no_executor_fails_honestly(consumer_env) -> None:
    user_id = 1015
    definition = _create_definition(user_id)

    result = await handle_agent_task_job(_job(definition, user_id))

    # Phase 1: an unwired family skips with an actionable reason (only
    # recurring_question has a production executor; agent_task messages
    # are redacted at rest) -- it is not a failure.
    assert result["status"] == "skipped"
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id, run_slot_key=SLOT
    )
    assert run["error"] == "family_not_wired_for_execution:recurring_question"


# ---------------------------------------------------------------------------
# Timeout status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_deadline_records_timed_out(consumer_env) -> None:
    user_id = 1016
    definition = _create_definition(user_id)

    async def _slow(d: DefinitionRow, p: dict[str, Any]) -> str:
        await asyncio.sleep(30)
        return "never"

    register_executor("recurring_question", _slow)

    result = await handle_agent_task_job(
        _job(definition, user_id), execution_timeout_seconds=0.05
    )

    assert result["status"] == "timed_out"
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id, run_slot_key=SLOT
    )
    assert run["status"] == "timed_out"
    notification = _latest_notification(user_id)
    assert notification is not None
    assert notification.kind == "automation_run_timed_out"


# ---------------------------------------------------------------------------
# Executor failure, notification policy, health, audit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_exception_records_failed(consumer_env) -> None:
    user_id = 1017
    definition = _create_definition(user_id)

    async def _boom(d: DefinitionRow, p: dict[str, Any]) -> str:
        raise RuntimeError("model exploded")

    register_executor("recurring_question", _boom)

    result = await handle_agent_task_job(_job(definition, user_id))

    assert result["status"] == "failed"
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.get_scheduled_task_run_by_slot(
        definition_id=definition.id, run_slot_key=SLOT
    )
    assert "RuntimeError" in (run["error"] or "")


@pytest.mark.asyncio
async def test_notification_policy_enabled_false_silences(consumer_env) -> None:
    user_id = 1018
    definition = _create_definition(
        user_id, notification_policy={"enabled": False}
    )
    register_executor("recurring_question", lambda d, p: asyncio.sleep(0, result="42"))

    result = await handle_agent_task_job(_job(definition, user_id))

    assert result["status"] == "succeeded"
    assert _latest_notification(user_id) is None


@pytest.mark.asyncio
async def test_failed_run_degrades_health_and_audits(consumer_env) -> None:
    user_id = 1019
    definition = _create_definition(user_id)

    async def _boom(d: DefinitionRow, p: dict[str, Any]) -> str:
        raise RuntimeError("nope")

    register_executor("recurring_question", _boom)

    await handle_agent_task_job(_job(definition, user_id))

    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    updated = sdb.get_definition(owner_id=user_id, definition_id=definition.id)
    assert updated.health == "degraded"
    audits, _total = sdb.list_audit_events(
        owner_id=user_id, definition_id=definition.id
    )
    assert any(a.event_type == "run_failed" for a in audits)


@pytest.mark.asyncio
async def test_terminal_persistence_failure_fails_job_before_notification(consumer_env, monkeypatch):
    user_id = 1030
    definition = _create_definition(user_id)
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    register_executor("recurring_question", lambda d, p: asyncio.sleep(0, result="42"))
    original = sdb.update_scheduled_task_run_status

    def unavailable(**kwargs):
        raise RuntimeError("terminal write unavailable")

    monkeypatch.setattr(sdb, "update_scheduled_task_run_status", unavailable)
    with pytest.raises(agent_task_jobs.ScheduledTaskPersistenceError, match="terminal run persistence failed"):
        await handle_agent_task_job(_job(definition, user_id), scheduled_db=sdb)
    assert _latest_notification(user_id) is None
    monkeypatch.setattr(sdb, "update_scheduled_task_run_status", original)
    result = await handle_agent_task_job(_job(definition, user_id), scheduled_db=sdb)
    assert result["status"] == "succeeded"


@pytest.mark.asyncio
async def test_concurrent_deliveries_execute_scheduled_slot_once(consumer_env):
    user_id = 1031
    definition = _create_definition(user_id)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def executor(d, p):
        calls.append(d.id)
        started.set()
        await release.wait()
        return "42"

    register_executor("recurring_question", executor)
    first = asyncio.create_task(handle_agent_task_job(_job(definition, user_id)))
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        with pytest.raises(agent_task_jobs.ScheduledTaskClaimBusy):
            await asyncio.wait_for(handle_agent_task_job(_job(definition, user_id)), timeout=0.2)
        assert calls == [definition.id]
    finally:
        release.set()
        await first


def test_scheduled_run_claim_fences_stale_completion_and_release(consumer_env):
    user_id = 1032
    definition = _create_definition(user_id)
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.create_scheduled_task_run(
        definition_id=definition.id,
        owner_id=user_id,
        scheduled_for=SLOT,
        job_id="77",
        run_slot_utc=SLOT,
        run_slot_key=SLOT,
        status="running",
    )
    kwargs = dict(owner_id=user_id, run_id=run["id"], job_id="77", lease_id="lease-1")
    assert sdb.claim_scheduled_task_run(**kwargs, claim_id="first")
    assert not sdb.claim_scheduled_task_run(**kwargs, claim_id="duplicate")
    sdb.release_scheduled_task_run_claim(owner_id=user_id, run_id=run["id"], claim_id="wrong")
    assert not sdb.claim_scheduled_task_run(**kwargs, claim_id="second")
    sdb.release_scheduled_task_run_claim(owner_id=user_id, run_id=run["id"], claim_id="first")
    assert sdb.claim_scheduled_task_run(**kwargs, claim_id="second")
    sdb.release_scheduled_task_run_claim(owner_id=user_id, run_id=run["id"], claim_id="first")
    with pytest.raises(ValueError, match="stale execution claim"):
        sdb.update_scheduled_task_run_status(run_id=run["id"], status="succeeded", execution_claim_id="first")
    assert not sdb.claim_scheduled_task_run(**kwargs, claim_id="third")
    sdb.update_scheduled_task_run_status(run_id=run["id"], status="succeeded", execution_claim_id="second")
    assert not sdb.claim_scheduled_task_run(**kwargs, claim_id="third")


@pytest.mark.asyncio
async def test_reclaimed_jobs_lease_requires_explicit_stopped_claim_reconciliation(consumer_env):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    user_id = 1033
    definition = _create_definition(user_id)
    manager = JobManager()
    manager.create_job(
        domain="scheduled_tasks",
        job_type="agent_task_run",
        queue="default",
        payload=_job(definition, user_id)["payload"],
        owner_user_id=str(user_id),
    )
    original = manager.acquire_next_job(domain="scheduled_tasks", queue="default", worker_id="first", lease_seconds=60)
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.create_scheduled_task_run(
        definition_id=definition.id,
        owner_id=user_id,
        scheduled_for=SLOT,
        job_id=str(original["id"]),
        run_slot_utc=SLOT,
        run_slot_key=SLOT,
        status="running",
    )
    assert sdb.claim_scheduled_task_run(
        owner_id=user_id,
        run_id=run["id"],
        claim_id="abandoned",
        job_id=str(original["id"]),
        lease_id=original["lease_id"],
    )
    executor = AsyncMock(return_value="recovered")
    register_executor("recurring_question", executor)
    with pytest.raises(agent_task_jobs.ScheduledTaskClaimBusy):
        await handle_agent_task_job(original, scheduled_db=sdb)
    executor.assert_not_awaited()
    assert manager.release_job(int(original["id"]), worker_id="first", lease_id=original["lease_id"])
    replacement = manager.acquire_next_job(
        domain="scheduled_tasks", queue="default", worker_id="second", lease_seconds=60
    )
    with pytest.raises(ValueError, match="stale Jobs lease"):
        await handle_agent_task_job(original, scheduled_db=sdb, jobs_manager=manager)
    with pytest.raises(agent_task_jobs.ScheduledTaskClaimBusy):
        await handle_agent_task_job(replacement, scheduled_db=sdb, jobs_manager=manager)
    # The prior attempt was never started in this test; explicit reconciliation is safe.
    sdb.release_scheduled_task_run_claim(owner_id=user_id, run_id=run["id"], claim_id="abandoned")
    result = await handle_agent_task_job(replacement, scheduled_db=sdb, jobs_manager=manager)
    assert result["status"] == "succeeded"
    assert executor.await_count == 1


@pytest.mark.asyncio
async def test_worker_retries_terminal_persistence_failure_with_same_manager(monkeypatch):
    from tldw_Server_API.app.services import agent_task_jobs_worker as worker
    from tldw_Server_API.app.core.Scheduled_Tasks import automation_executors

    stop = asyncio.Event()
    captured = {}

    class Manager:
        def acquire_next_job(self, **kwargs):
            return {"id": 77, "lease_id": "lease", "job_type": "agent_task_run"}

        def fail_job(self, job_id, **kwargs):
            captured.update(kwargs)
            stop.set()

    manager = Manager()

    async def handler(job, **kwargs):
        captured["manager"] = kwargs.get("jobs_manager")
        error_class = getattr(agent_task_jobs, "ScheduledTaskPersistenceError", RuntimeError)
        raise error_class("terminal persistence failed")

    monkeypatch.setattr(worker, "JobManager", lambda: manager, raising=False)
    monkeypatch.setattr(worker, "jobs_manager_from_env", lambda: manager, raising=False)
    monkeypatch.setattr(worker, "handle_agent_task_job", handler)
    monkeypatch.setattr(automation_executors, "register_automation_executors", lambda: None)
    await worker.run_agent_task_jobs_worker(stop)
    assert captured["retryable"] is True
    assert captured["manager"] is manager


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", [None, {"owner_user_id": "9999", "status": "completed"}])
async def test_unverifiable_prior_jobs_claim_is_not_stolen(consumer_env, previous):
    user_id = 1034
    definition = _create_definition(user_id)
    sdb = ScheduledTasksDatabase.for_user(user_id=user_id)
    run = sdb.create_scheduled_task_run(
        definition_id=definition.id,
        owner_id=user_id,
        scheduled_for=SLOT,
        job_id="78",
        run_slot_utc=SLOT,
        run_slot_key=SLOT,
        status="running",
    )
    assert sdb.claim_scheduled_task_run(
        owner_id=user_id, run_id=run["id"], claim_id="prior", job_id="78", lease_id="unverifiable"
    )
    job = {**_job(definition, user_id), "lease_id": "current", "status": "processing"}
    manager = Mock()
    manager.get_job.side_effect = lambda job_id: job if job_id == 77 else previous
    executor = AsyncMock(return_value="must not run")
    register_executor("recurring_question", executor)
    with pytest.raises(agent_task_jobs.ScheduledTaskClaimBusy):
        await handle_agent_task_job(job, scheduled_db=sdb, jobs_manager=manager)
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_replaced_lease_cannot_overlap_cancellation_resistant_executor(consumer_env):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    user_id = 1035
    definition = _create_definition(user_id)
    manager = JobManager()
    manager.create_job(
        domain="scheduled_tasks",
        job_type="agent_task_run",
        queue="default",
        payload=_job(definition, user_id)["payload"],
        owner_user_id=str(user_id),
    )
    original = manager.acquire_next_job(domain="scheduled_tasks", queue="default", worker_id="old", lease_seconds=60)
    started, cancelling, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def executor(d, p):
        calls.append(d.id)
        if len(calls) > 1:
            return "duplicate execution"
        started.set()
        try:
            await stopped.wait()
        except asyncio.CancelledError:
            cancelling.set()
            await stopped.wait()
        return "old result"

    register_executor("recurring_question", executor)
    first = asyncio.create_task(handle_agent_task_job(original, jobs_manager=manager))
    await asyncio.wait_for(started.wait(), timeout=2)
    first.cancel()
    await asyncio.wait_for(cancelling.wait(), timeout=2)
    assert manager.release_job(int(original["id"]), worker_id="old", lease_id=original["lease_id"])
    replacement = manager.acquire_next_job(domain="scheduled_tasks", queue="default", worker_id="new", lease_seconds=60)
    try:
        with pytest.raises(RuntimeError, match="claim.*active|reconciliation"):
            await handle_agent_task_job(replacement, jobs_manager=manager)
        assert calls == [definition.id]
    finally:
        stopped.set()
        with pytest.raises(asyncio.CancelledError):
            await first
    result = await handle_agent_task_job(replacement, jobs_manager=manager)
    assert result["status"] == "succeeded"
    assert calls == [definition.id, definition.id]


@pytest.mark.asyncio
async def test_worker_does_not_ack_or_fail_a_busy_execution_claim(monkeypatch):
    from tldw_Server_API.app.services import agent_task_jobs_worker as worker
    from tldw_Server_API.app.core.Scheduled_Tasks import automation_executors

    stop = asyncio.Event()
    transitions = []

    class Manager:
        def acquire_next_job(self, **kwargs):
            return {"id": 77, "lease_id": "lease", "job_type": "agent_task_run"}

        def fail_job(self, *args, **kwargs):
            transitions.append("failed")

        def complete_job(self, *args, **kwargs):
            transitions.append("completed")

    async def handler(job, **kwargs):
        stop.set()
        error_class = getattr(agent_task_jobs, "ScheduledTaskClaimBusy", RuntimeError)
        raise error_class("claim active; reconciliation required")

    monkeypatch.setattr(worker, "jobs_manager_from_env", Manager)
    monkeypatch.setattr(worker, "handle_agent_task_job", handler)
    monkeypatch.setattr(automation_executors, "register_automation_executors", lambda: None)
    await worker.run_agent_task_jobs_worker(stop)
    assert transitions == []


@pytest.mark.asyncio
async def test_repeated_handler_cancellation_retains_live_executor_claim(consumer_env):
    user_id = 1036
    definition = _create_definition(user_id)
    started, cancelling, stop_executor = asyncio.Event(), asyncio.Event(), asyncio.Event()
    executor_tasks = []

    async def executor(d, p):
        executor_tasks.append(asyncio.current_task())
        if len(executor_tasks) > 1:
            return "duplicate execution"
        started.set()
        try:
            await stop_executor.wait()
        except asyncio.CancelledError:
            cancelling.set()
            await stop_executor.wait()
        return "old result"

    register_executor("recurring_question", executor)
    first = asyncio.create_task(handle_agent_task_job(_job(definition, user_id)))
    await asyncio.wait_for(started.wait(), timeout=2)
    first.cancel()
    await asyncio.wait_for(cancelling.wait(), timeout=2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    try:
        assert not executor_tasks[0].done()
        with pytest.raises(agent_task_jobs.ScheduledTaskClaimBusy):
            await handle_agent_task_job(_job(definition, user_id))
        assert len(executor_tasks) == 1
    finally:
        stop_executor.set()
        await asyncio.gather(*executor_tasks, return_exceptions=True)
