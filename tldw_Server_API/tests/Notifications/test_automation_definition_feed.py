"""Automation definition scheduler feed tests (TASK-13020).

Mirrors test_reminders_scheduler.py's shapes: real per-user
ScheduledTasksDatabase under a tmp USER_DB_BASE_DIR (preview-gated
definition creation through the REAL DB methods), direct fire-path
invocation, and captured ``create_job`` kwargs. No reimplementation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.services.reminders_scheduler import _normalize_slot_to_utc_iso
from tldw_Server_API.app.services.scheduled_task_automation_scheduler import (
    ARMED_HEALTH,
    AUTOMATION_DOMAIN,
    AUTOMATION_JOB_TYPE,
    _AutomationScheduler,
    build_trigger,
    compute_run_slot,
)

pytestmark = pytest.mark.unit


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
    schedule: dict,
    lifecycle: str = "configured",
    family: str = "recurring_question",
) -> "object":
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


def _capture_jobs(scheduler: _AutomationScheduler) -> list[dict]:
    created: list[dict] = []

    def _capture(**kwargs):
        created.append(kwargs)
        return {"id": len(created)}

    scheduler._jobs.create_job = _capture  # type: ignore[method-assign]
    return created


# ---------------------------------------------------------------------------
# Trigger building — all five kinds, plus honest-refusal cases
# ---------------------------------------------------------------------------


def test_build_trigger_one_time():
    trigger, reason = build_trigger({"kind": "one_time", "run_at": "2026-08-21T09:00:00+00:00"})
    assert trigger is not None and reason is None


def test_build_trigger_interval():
    trigger, reason = build_trigger({"kind": "interval", "seconds": 900})
    assert trigger is not None and reason is None


def test_build_trigger_daily_and_weekly():
    daily, _ = build_trigger({"kind": "daily", "at": "09:30", "timezone": "UTC"})
    weekly, _ = build_trigger({"kind": "weekly", "weekday": 0, "at": "09:30"})
    assert daily is not None and weekly is not None


def test_build_trigger_cron():
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
def test_build_trigger_refuses_unusable_schedules(schedule, expect_fragment):
    trigger, reason = build_trigger(schedule)
    assert trigger is None
    assert expect_fragment in reason


# ---------------------------------------------------------------------------
# Slot computation
# ---------------------------------------------------------------------------


def test_compute_run_slot_one_time_uses_run_at():
    run_at = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
    slot = compute_run_slot({"kind": "one_time", "run_at": run_at.isoformat()}, None)
    assert slot == run_at


def test_compute_run_slot_periodic_returns_current_slot():
    trigger, _ = build_trigger({"kind": "daily", "at": "09:00", "timezone": "UTC"})
    now = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
    slot = compute_run_slot({"kind": "daily", "at": "09:00"}, trigger, now=now)
    assert slot is not None
    assert slot.hour == 9 and slot.minute == 0 and slot <= now


# ---------------------------------------------------------------------------
# The fire path (enqueue shape, lifecycle gate, early-skip, idempotency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_due_slot_enqueues_job_once(automation_scheduler_env):
    user_id = 990
    definition = _create_definition(user_id, schedule={"kind": "cron", "cron": "* * * * *"})

    scheduler = _AutomationScheduler()
    scheduler._get_db(user_id)  # warm the cache against the real DB
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id=user_id)

    assert len(created) == 1
    assert created[0]["domain"] == AUTOMATION_DOMAIN
    assert created[0]["job_type"] == AUTOMATION_JOB_TYPE
    assert created[0]["owner_user_id"] == user_id
    assert created[0]["payload"]["definition_id"] == definition.id
    assert created[0]["payload"]["family"] == "recurring_question"
    assert created[0]["idempotency_key"].startswith(f"definition:{definition.id}:")


@pytest.mark.asyncio
async def test_same_slot_fires_one_idempotency_key(automation_scheduler_env):
    """Two scheduler passes over the same slot produce the identical key.

    The Jobs layer returns the same row for a duplicate key, so identical
    keys ARE the dedupe across restarts and concurrent rescans.
    """
    user_id = 991
    definition = _create_definition(user_id, schedule={"kind": "cron", "cron": "* * * * *"})
    scheduler = _AutomationScheduler()
    scheduler._get_db(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id=user_id)
    await scheduler._run_definition_schedule(definition.id, user_id=user_id)

    assert len(created) == 2
    assert created[0]["idempotency_key"] == created[1]["idempotency_key"]


@pytest.mark.asyncio
async def test_non_configured_lifecycle_never_fires(automation_scheduler_env):
    user_id = 992
    definition = _create_definition(
        user_id, schedule={"kind": "cron", "cron": "*/5 * * * *"}, lifecycle="paused"
    )
    scheduler = _AutomationScheduler()
    scheduler._get_db(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id=user_id)

    assert created == []


@pytest.mark.asyncio
async def test_future_one_time_slot_skips_early_fire(automation_scheduler_env):
    user_id = 993
    run_at = datetime.now(timezone.utc) + timedelta(hours=2)
    definition = _create_definition(
        user_id, schedule={"kind": "one_time", "run_at": run_at.isoformat()}
    )
    scheduler = _AutomationScheduler()
    scheduler._get_db(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id=user_id)

    assert created == []


@pytest.mark.asyncio
async def test_unusable_schedule_never_fires(automation_scheduler_env):
    user_id = 994
    # The DB stores whatever the schedule dict carried; the feed must skip
    # a definition whose per-kind fields are junk even though 'kind' is
    # among the supported five.
    definition = _create_definition(user_id, schedule={"kind": "interval"})
    scheduler = _AutomationScheduler()
    scheduler._get_db(user_id)
    created = _capture_jobs(scheduler)

    await scheduler._run_definition_schedule(definition.id, user_id=user_id)

    assert created == []


# ---------------------------------------------------------------------------
# Health honesty (AC#4)
# ---------------------------------------------------------------------------


def test_mark_ready_flips_health_with_audit_and_no_version_churn(automation_scheduler_env):
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
