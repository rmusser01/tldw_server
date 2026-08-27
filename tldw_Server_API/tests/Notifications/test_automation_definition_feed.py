"""Automation definition scheduler feed tests (TASK-13020).

Mirrors test_reminders_scheduler.py's shapes: real per-user
ScheduledTasksDatabase under a tmp USER_DB_BASE_DIR (preview-gated
definition creation through the REAL DB methods), direct fire-path
invocation, and captured ``create_job`` kwargs. No reimplementation.

The arming model under test: each occurrence is its own DateTrigger job
with the slot bound into the job args at arm time, so a late callback
still enqueues the occurrence it was armed for (review finding #7).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    DefinitionRow,
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.Scheduled_Tasks.execution_certification import (
    ExecutionCertification,
)
from tldw_Server_API.app.services.reminders_scheduler import _normalize_slot_to_utc_iso
from tldw_Server_API.app.services.scheduled_task_automation_scheduler import (
    ARMED_HEALTH,
    AUTOMATION_DOMAIN,
    AUTOMATION_JOB_TYPE,
    _AutomationScheduler,
    _next_occurrence,
    build_trigger,
)

pytestmark = pytest.mark.unit


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
def automation_scheduler_env(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_automation_scheduler"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("JOBS_DB_PATH", str(base_dir / "jobs.db"))
    try:
        yield
    finally:
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
    schedule: dict[str, Any],
    lifecycle: str = "configured",
    family: str = "recurring_question",
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
        schedule_preview=schedule,
        redaction_policy={"fields": [], "mode": "none"},
        expires_at=(
            datetime.now(timezone.utc) + timedelta(hours=24)
        ).isoformat(),
        created_by="test",
    )
    return db.create_definition(
        owner_id=user_id,
        family=family,
        name="Test Definition",
        description=None,
        lifecycle=lifecycle,
        health="execution_unavailable",
        schedule=schedule,
        input={"question": "What changed today?"},
        visibility_policy="owner",
        notification_policy={},
        approval_policy={},
        preview_id=preview.id,
        created_by="test",
        updated_by="test",
    )


def _capture_jobs(scheduler: _AutomationScheduler) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []

    def _capture(**kwargs: Any) -> dict[str, int]:
        created.append(kwargs)
        return {"id": len(created)}

    scheduler._jobs.create_job = _capture  # type: ignore[method-assign]
    return created


def _bare_scheduler(user_id: int) -> _AutomationScheduler:
    """A scheduler with a warmed per-user DB cache and NO APS instance.

    The fire path only needs the DB cache; arming needs APS and is tested
    separately against a real AsyncIOScheduler.
    """
    scheduler = _AutomationScheduler()
    scheduler._get_db(user_id)
    return scheduler


# ---------------------------------------------------------------------------
# Trigger building — all five kinds, plus honest-refusal cases
# ---------------------------------------------------------------------------


def test_build_trigger_one_time() -> None:
    trigger, reason = build_trigger(
        {"kind": "one_time", "run_at": "2026-08-21T09:00:00+00:00"}
    )
    assert trigger is not None and reason is None


def test_build_trigger_interval() -> None:
    trigger, reason = build_trigger({"kind": "interval", "seconds": 900})
    assert trigger is not None and reason is None


def test_build_trigger_daily_and_weekly() -> None:
    daily, _ = build_trigger({"kind": "daily", "at": "09:30", "timezone": "UTC"})
    weekly, _ = build_trigger({"kind": "weekly", "weekday": 0, "at": "09:30"})
    assert daily is not None and weekly is not None


def test_build_trigger_cron() -> None:
    trigger, reason = build_trigger({"kind": "cron", "cron": "*/5 * * * *"})
    assert trigger is not None and reason is None


@pytest.mark.parametrize(
    "schedule,expect_fragment",
    [
        ({"kind": "one_time"}, "run_at"),
        ({"kind": "interval", "seconds": 0}, "seconds"),
        ({"kind": "interval"}, "seconds"),
        ({"kind": "daily", "at": "nope"}, "at"),
        ({"kind": "daily", "at": "25:00"}, "range"),
        ({"kind": "cron", "cron": ""}, "expression"),
        ({"kind": "hourly"}, "unsupported"),
    ],
)
def test_build_trigger_refuses_unusable_schedules(
    schedule: dict[str, Any], expect_fragment: str
) -> None:
    trigger, reason = build_trigger(schedule)
    assert trigger is None
    assert expect_fragment in reason


# ---------------------------------------------------------------------------
# Next-occurrence derivation
# ---------------------------------------------------------------------------


def test_next_occurrence_one_time_past_returns_none() -> None:
    trigger, _ = build_trigger(
        {"kind": "one_time", "run_at": "2020-01-01T00:00:00+00:00"}
    )
    assert _next_occurrence(trigger, after=datetime.now(timezone.utc)) is None


def test_next_occurrence_periodic_returns_next_boundary() -> None:
    trigger, _ = build_trigger({"kind": "daily", "at": "09:00", "timezone": "UTC"})
    after = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    nxt = _next_occurrence(trigger, after=after)
    assert nxt is not None
    assert nxt == datetime(2026, 8, 22, 9, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Enumeration (review finding #5: list_definitions returns (rows, total))
# ---------------------------------------------------------------------------


def test_configured_definitions_returns_plain_row_list(automation_scheduler_env) -> None:
    user_id = 989
    _create_definition(user_id, schedule={"kind": "cron", "cron": "0 9 * * *"})
    _create_definition(
        user_id, schedule={"kind": "cron", "cron": "0 10 * * *"}, lifecycle="paused"
    )

    scheduler = _AutomationScheduler()
    rows = scheduler._configured_definitions(user_id)

    # The paused definition must be excluded; iterating the result must
    # yield DefinitionRow objects, never a (rows, total) tuple's tail int.
    assert len(rows) == 1
    assert all(isinstance(row, DefinitionRow) for row in rows)
    assert rows[0].lifecycle == "configured"


# ---------------------------------------------------------------------------
# The fire path (enqueue shape, lifecycle gate, early-skip, idempotency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_due_slot_enqueues_job_once(automation_scheduler_env) -> None:
    user_id = 990
    definition = _create_definition(user_id, schedule={"kind": "cron", "cron": "* * * * *"})
    slot = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    scheduler = _bare_scheduler(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(
        definition.id, user_id, _normalize_slot_to_utc_iso(slot)
    )

    assert len(created) == 1
    assert created[0]["domain"] == AUTOMATION_DOMAIN
    assert created[0]["job_type"] == AUTOMATION_JOB_TYPE
    assert created[0]["owner_user_id"] == user_id
    assert created[0]["payload"]["definition_id"] == definition.id
    assert created[0]["payload"]["family"] == "recurring_question"
    assert created[0]["payload"]["scheduled_for"] == _normalize_slot_to_utc_iso(slot)
    assert created[0]["idempotency_key"] == (
        f"definition:{definition.id}:{_normalize_slot_to_utc_iso(slot)}"
    )


@pytest.mark.asyncio
async def test_late_fire_still_enqueues_the_armed_slot(automation_scheduler_env) -> None:
    """Misfire correctness (review finding #7).

    The callback runs 90s after the boundary it was armed for; the slot,
    scheduled_for, and idempotency key must all name the ARMED occurrence,
    not anything derived from the late wall clock.
    """
    user_id = 996
    definition = _create_definition(user_id, schedule={"kind": "cron", "cron": "* * * * *"})
    slot = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        seconds=90
    )

    scheduler = _bare_scheduler(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(
        definition.id, user_id, _normalize_slot_to_utc_iso(slot)
    )

    assert len(created) == 1
    assert created[0]["payload"]["scheduled_for"] == _normalize_slot_to_utc_iso(slot)
    assert created[0]["idempotency_key"] == (
        f"definition:{definition.id}:{_normalize_slot_to_utc_iso(slot)}"
    )


@pytest.mark.asyncio
async def test_same_slot_fires_one_idempotency_key(automation_scheduler_env) -> None:
    """Two scheduler passes over the same slot produce the identical key.

    The Jobs layer returns the same row for a duplicate key, so identical
    keys ARE the dedupe across restarts and concurrent rescans.
    """
    user_id = 991
    definition = _create_definition(user_id, schedule={"kind": "cron", "cron": "* * * * *"})
    slot_iso = _normalize_slot_to_utc_iso(
        datetime.now(timezone.utc).replace(second=0, microsecond=0)
    )
    scheduler = _bare_scheduler(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id, slot_iso)
    await scheduler._run_definition_schedule(definition.id, user_id, slot_iso)

    assert len(created) == 2
    assert created[0]["idempotency_key"] == created[1]["idempotency_key"]


@pytest.mark.asyncio
async def test_non_configured_lifecycle_never_fires(automation_scheduler_env) -> None:
    user_id = 992
    definition = _create_definition(
        user_id, schedule={"kind": "cron", "cron": "* * * * *"}, lifecycle="paused"
    )
    scheduler = _bare_scheduler(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id, "2026-08-21T09:00:00+00:00")

    assert created == []


@pytest.mark.asyncio
async def test_future_slot_skips_early_fire(automation_scheduler_env) -> None:
    user_id = 993
    definition = _create_definition(user_id, schedule={"kind": "one_time", "run_at": ""})
    run_at = datetime.now(timezone.utc) + timedelta(hours=2)
    db = ScheduledTasksDatabase.for_user(user_id=user_id)
    db.update_definition(
        owner_id=user_id,
        definition_id=definition.id,
        patch={"schedule": {"kind": "one_time", "run_at": run_at.isoformat()}},
    )
    scheduler = _bare_scheduler(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(
        definition.id, user_id, _normalize_slot_to_utc_iso(run_at)
    )

    assert created == []


@pytest.mark.asyncio
async def test_unusable_schedule_never_fires(automation_scheduler_env) -> None:
    user_id = 994
    # The DB stores whatever the schedule dict carried; the feed must skip
    # a definition whose per-kind fields are junk even though 'kind' is
    # among the supported five.
    definition = _create_definition(user_id, schedule={"kind": "interval"})
    scheduler = _bare_scheduler(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id, "2026-08-21T09:00:00+00:00")

    assert created == []


# ---------------------------------------------------------------------------
# Arming against a real APScheduler instance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arms_next_occurrence_with_slot_bound_in_args(automation_scheduler_env) -> None:
    user_id = 997
    definition = _create_definition(user_id, schedule={"kind": "daily", "at": "23:59"})

    scheduler = _AutomationScheduler()
    scheduler._aps = AsyncIOScheduler(timezone="UTC")
    scheduler._aps.start()
    try:
        assert scheduler._arm(definition, user_id) is True

        jobs = scheduler._aps.get_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.id == f"automation:{definition.id}"
        # One DateTrigger occurrence with the slot bound into args:
        (arg_definition_id, arg_user_id, arg_slot_iso) = job.args
        assert arg_definition_id == definition.id
        assert arg_user_id == user_id
        assert datetime.fromisoformat(arg_slot_iso).hour == 23
        assert datetime.fromisoformat(arg_slot_iso).minute == 59
    finally:
        scheduler._aps.shutdown(wait=False)
        scheduler._aps = None


@pytest.mark.asyncio
async def test_agent_definition_never_arms_during_load_rescan_or_reconcile(
    automation_scheduler_env,
    monkeypatch,
) -> None:
    user_id = 998
    definition = _create_definition(
        user_id,
        family="agent_task",
        schedule={"kind": "daily", "at": "23:59"},
    )
    scheduler = _AutomationScheduler()
    scheduler._execution_certification_resolver = _certified_execution
    scheduler._execution_stack_ready_resolver = lambda: False
    scheduler._aps = AsyncIOScheduler(timezone="UTC")
    scheduler._aps.start()
    monkeypatch.setattr(scheduler, "_enumerate_user_ids", lambda: {user_id})
    try:
        await scheduler._load_all()
        assert scheduler._aps.get_jobs() == []  # nosec B101

        await scheduler._rescan_once()
        assert scheduler._aps.get_jobs() == []  # nosec B101

        scheduler._started = True
        await scheduler.reconcile_definition(
            definition_id=definition.id,
            user_id=user_id,
        )
        assert scheduler._aps.get_jobs() == []  # nosec B101
    finally:
        scheduler._aps.shutdown(wait=False)
        scheduler._aps = None


@pytest.mark.asyncio
async def test_reconcile_blocked_agent_removes_job_and_marks_unavailable(
    automation_scheduler_env,
) -> None:
    """A newly blocked Agent must not retain an armed job or ready health."""

    user_id = 996
    definition = _create_definition(
        user_id,
        family="agent_task",
        schedule={"kind": "daily", "at": "23:59"},
    )
    scheduler = _AutomationScheduler(
        execution_certification_resolver=_certified_execution,
        execution_stack_ready_resolver=lambda: True,
    )
    scheduler._aps = AsyncIOScheduler(timezone="UTC")
    scheduler._aps.start()
    scheduler._started = True
    try:
        assert scheduler._arm(definition, user_id) is True
        ready = scheduler._get_db(user_id).get_definition(
            owner_id=user_id,
            definition_id=definition.id,
        )
        assert ready is not None
        assert ready.health == ARMED_HEALTH
        assert scheduler._aps.get_job(f"automation:{definition.id}") is not None

        scheduler._execution_stack_ready_resolver = lambda: False
        await scheduler.reconcile_definition(
            definition_id=definition.id,
            user_id=user_id,
        )

        blocked = scheduler._get_db(user_id).get_definition(
            owner_id=user_id,
            definition_id=definition.id,
        )
        assert blocked is not None
        assert blocked.health == "execution_unavailable"
        assert scheduler._aps.get_job(f"automation:{definition.id}") is None
        audits, _total = scheduler._get_db(user_id).list_audit_events(
            owner_id=user_id,
            definition_id=definition.id,
        )
        assert any(audit.event_type == "scheduler_blocked" for audit in audits)
    finally:
        scheduler._aps.shutdown(wait=False)
        scheduler._aps = None


@pytest.mark.asyncio
async def test_agent_definition_race_refuses_again_at_fire(
    automation_scheduler_env,
) -> None:
    user_id = 999
    definition = _create_definition(
        user_id,
        family="agent_task",
        schedule={"kind": "cron", "cron": "* * * * *"},
    )
    scheduler = _bare_scheduler(user_id)
    scheduler._execution_certification_resolver = _certified_execution
    scheduler._execution_stack_ready_resolver = lambda: False
    created = _capture_jobs(scheduler)
    slot = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    await scheduler._fire(
        definition.id,
        user_id,
        _normalize_slot_to_utc_iso(slot),
    )

    assert created == []  # nosec B101


# ---------------------------------------------------------------------------
# Health honesty (AC#4)
# ---------------------------------------------------------------------------


def test_mark_ready_flips_health_with_audit_and_no_version_churn(automation_scheduler_env) -> None:
    user_id = 995
    definition = _create_definition(user_id, schedule={"kind": "cron", "cron": "0 9 * * *"})
    db = ScheduledTasksDatabase.for_user(user_id=user_id)
    assert definition.health == "execution_unavailable"

    scheduler = _AutomationScheduler()
    scheduler._get_db(user_id)
    scheduler._mark_ready(definition, user_id)

    updated = db.get_definition(owner_id=user_id, definition_id=definition.id)
    assert updated.health == ARMED_HEALTH
    assert updated.version == definition.version + 1

    audits, _total = db.list_audit_events(owner_id=user_id, definition_id=definition.id)
    assert any(a.event_type == "scheduler_armed" for a in audits)

    # Second pass with fresh row state is a no-op: no version churn.
    scheduler._mark_ready(updated, user_id)
    again = db.get_definition(owner_id=user_id, definition_id=definition.id)
    assert again.version == updated.version
