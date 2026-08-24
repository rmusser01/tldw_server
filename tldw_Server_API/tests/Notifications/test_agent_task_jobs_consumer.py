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

pytestmark = pytest.mark.unit

SLOT = "2026-08-21T09:00:00+00:00"


@pytest.fixture()
def consumer_env(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_agent_task_consumer"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("JOBS_DB_PATH", str(base_dir / "jobs.db"))
    _orig_executors = dict(getattr(__import__(
        "tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs", fromlist=["_EXECUTORS"]
    ), "_EXECUTORS"))
    __import__(
        "tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs", fromlist=["_EXECUTORS"]
    )._EXECUTORS.clear()
    try:
        yield
    finally:
        module = __import__(
            "tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs",
            fromlist=["_EXECUTORS"],
        )
        module._EXECUTORS.clear()
        module._EXECUTORS.update(_orig_executors)
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
