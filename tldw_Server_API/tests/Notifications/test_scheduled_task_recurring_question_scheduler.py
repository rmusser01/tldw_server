from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskDefinitionCreateRequest,
    ScheduledTaskPreviewCreateRequest,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_jobs import (
    RECURRING_QUESTION_JOB_TYPE,
    RECURRING_QUESTION_QUEUE,
    SCHEDULED_TASKS_DOMAIN,
    build_scheduled_run_idempotency_key,
)
from tldw_Server_API.app.services.scheduled_task_automation_service import ScheduledTaskAutomationError
from tldw_Server_API.app.services.scheduled_task_recurring_question_scheduler import (
    _RecurringQuestionScheduler,
    _normalize_day_of_week,
    _normalize_slot_to_utc_iso,
    _parse_time_fields,
)
from tldw_Server_API.app.services.scheduled_task_recurring_question_service import (
    ScheduledTaskRecurringQuestionService,
)

pytestmark = pytest.mark.unit

OWNER_ID = 9901
ACTOR = "recurring-question-scheduler-test"


@pytest.fixture()
def recurring_question_scheduler_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    base_dir = tmp_path / "users"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("SCHEDULED_TASKS_RECURRING_QUESTION_SCHEDULER_TZ", "UTC")
    try:
        yield base_dir
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def _service_for_user(user_id: int) -> tuple[ScheduledTaskRecurringQuestionService, ScheduledTasksDatabase]:
    repo = ScheduledTasksDatabase.for_user(user_id)
    repo.ensure_schema()
    return ScheduledTaskRecurringQuestionService(repository=repo, job_manager=JobManager()), repo


def _payload(
    *,
    name: str,
    schedule: dict[str, Any] | None = None,
    family: str = "recurring_question",
) -> ScheduledTaskPreviewCreateRequest:
    if family == "agent_task":
        input_payload = {"agent_ref": "agent:daily-summary", "message": "Summarize new information."}
        visibility_policy = {"mode": "metadata_only"}
    else:
        input_payload = {"question": "What has changed in my research corpus?"}
        visibility_policy = {"mode": "findings_only"}
    return ScheduledTaskPreviewCreateRequest(
        family=family,
        name=name,
        description="Scheduler test definition",
        input=input_payload,
        config={
            "scope": {"mode": "all_searchable_library"},
            "generation_mode": "optional",
        },
        schedule=schedule or {"kind": "daily", "time": "09:00", "timezone": "UTC"},
        visibility_policy=visibility_policy,
        notification_policy={"channels": ["in_app"]},
        approval_policy={"required": False},
    )


def _create_definition(
    service: ScheduledTaskRecurringQuestionService,
    *,
    name: str,
    lifecycle: str = "configured",
    schedule: dict[str, Any] | None = None,
    family: str = "recurring_question",
):
    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(name=name, schedule=schedule, family=family),
    )
    return service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(
            preview_id=preview.id,
            initial_lifecycle=lifecycle,
        ),
    )


class _FakeAPSJob:
    def __init__(self, job_id: str) -> None:
        self.id = job_id


class _FakeAPS:
    def __init__(self, existing: list[str] | None = None) -> None:
        self.added: dict[str, dict[str, Any]] = {}
        self.removed: list[str] = []
        self._existing = list(existing or [])

    def add_job(self, func, *, trigger, id: str, args: list[Any], **kwargs: Any) -> None:
        self.added[id] = {"func": func, "trigger": trigger, "args": args, "kwargs": kwargs}

    def remove_job(self, job_id: str) -> None:
        self.removed.append(job_id)
        self.added.pop(job_id, None)

    def get_jobs(self) -> list[_FakeAPSJob]:
        ids = set(self._existing) | set(self.added)
        return [_FakeAPSJob(job_id) for job_id in ids]


@pytest.mark.asyncio
async def test_rescan_registers_only_open_configured_recurring_questions(
    recurring_question_scheduler_env,
) -> None:
    service, repo = _service_for_user(OWNER_ID)
    configured = _create_definition(service, name="Configured recurring question")
    paused = _create_definition(service, name="Paused recurring question", lifecycle="paused")
    archived = _create_definition(service, name="Archived recurring question")
    solved = _create_definition(service, name="Solved recurring question")
    agent_task = _create_definition(service, name="Agent task", family="agent_task")
    service.archive_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=archived.id)
    service.mark_solved(owner_id=OWNER_ID, actor=ACTOR, definition_id=solved.id)
    disabled = _create_definition(service, name="Disabled recurring question")
    repo.update_definition(
        owner_id=OWNER_ID,
        definition_id=disabled.id,
        patch={"lifecycle": "disabled", "updated_by": ACTOR},
        expected_version=disabled.version,
    )

    scheduler = _RecurringQuestionScheduler()
    fake_aps = _FakeAPS(
        existing=[
            scheduler._job_id(OWNER_ID, paused.id),
            scheduler._job_id(OWNER_ID, archived.id),
            scheduler._job_id(OWNER_ID, agent_task.id),
        ]
    )
    scheduler._aps = fake_aps
    scheduler._started = True

    await scheduler.rescan()

    assert set(fake_aps.added) == {scheduler._job_id(OWNER_ID, configured.id)}  # nosec B101
    assert scheduler._job_id(OWNER_ID, paused.id) in fake_aps.removed  # nosec B101
    assert scheduler._job_id(OWNER_ID, archived.id) in fake_aps.removed  # nosec B101
    assert scheduler._job_id(OWNER_ID, agent_task.id) in fake_aps.removed  # nosec B101


@pytest.mark.asyncio
async def test_rescan_preserves_existing_registered_jobs(
    recurring_question_scheduler_env,
) -> None:
    service, _repo = _service_for_user(OWNER_ID)
    configured = _create_definition(service, name="Already registered")
    scheduler = _RecurringQuestionScheduler()
    job_id = scheduler._job_id(OWNER_ID, configured.id)
    fake_aps = _FakeAPS(existing=[job_id])
    scheduler._aps = fake_aps
    scheduler._started = True

    await scheduler.rescan()

    assert fake_aps.removed == []  # nosec B101
    assert fake_aps.added == {}  # nosec B101


@pytest.mark.asyncio
async def test_rescan_isolates_per_user_repository_errors(
    recurring_question_scheduler_env,
) -> None:
    service, repo = _service_for_user(OWNER_ID)
    configured = _create_definition(service, name="Healthy user")

    class BrokenUserScheduler(_RecurringQuestionScheduler):
        def _enumerate_user_ids(self) -> set[int]:
            return {123, OWNER_ID}

        def _get_repo(self, owner_id: int) -> ScheduledTasksDatabase:
            if owner_id == 123:
                raise RuntimeError("locked user database")
            return repo

    scheduler = BrokenUserScheduler()
    fake_aps = _FakeAPS()
    scheduler._aps = fake_aps
    scheduler._started = True

    await scheduler.rescan()

    assert set(fake_aps.added) == {scheduler._job_id(OWNER_ID, configured.id)}  # nosec B101


@pytest.mark.asyncio
async def test_rescan_skips_invalid_schedule_without_blocking_valid_definitions(
    recurring_question_scheduler_env,
) -> None:
    service, _repo = _service_for_user(OWNER_ID)
    invalid = _create_definition(
        service,
        name="Invalid cron recurring question",
        schedule={"kind": "cron", "cron": "not-valid-cron", "timezone": "UTC"},
    )
    valid = _create_definition(service, name="Valid recurring question")
    scheduler = _RecurringQuestionScheduler()
    fake_aps = _FakeAPS()
    scheduler._aps = fake_aps
    scheduler._started = True

    await scheduler.rescan()

    assert set(fake_aps.added) == {scheduler._job_id(OWNER_ID, valid.id)}  # nosec B101
    assert scheduler._job_id(OWNER_ID, invalid.id) in fake_aps.removed  # nosec B101


@pytest.mark.asyncio
async def test_reconcile_all_stale_runs_isolates_per_user_errors(
    recurring_question_scheduler_env,
) -> None:
    service, repo = _service_for_user(OWNER_ID)
    definition = _create_definition(service, name="Healthy reconcile user")
    run = repo.create_run(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        definition_version=definition.version,
        trigger_reason="scheduled",
        status="queued",
        outcome="none",
        scope_snapshot={"mode": "all_searchable_library", "resolved_sources": ["media_db"]},
        finding_policy_snapshot={"preset": "balanced_findings"},
        rag_request_snapshot={"query": "What changed?"},
        run_summary={"message": "Queued before crash."},
        schedule_slot="2026-07-01T16:00:00+00:00",
    )
    with sqlite3.connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE scheduled_task_runs SET updated_at = ? WHERE owner_id = ? AND id = ?",
            ["2026-07-01T16:00:00+00:00", OWNER_ID, run.id],
        )

    class BrokenUserScheduler(_RecurringQuestionScheduler):
        def _enumerate_user_ids(self) -> set[int]:
            return {123, OWNER_ID}

        def _get_service(self, owner_id: int) -> ScheduledTaskRecurringQuestionService:
            if owner_id == 123:
                raise RuntimeError("locked user database")
            return service

    scheduler = BrokenUserScheduler()

    repaired = await scheduler.reconcile_all_stale_runs()

    assert repaired == {OWNER_ID: [run.id]}  # nosec B101


@pytest.mark.asyncio
async def test_enqueue_due_slot_uses_scheduled_idempotency_and_creates_one_run(
    recurring_question_scheduler_env,
) -> None:
    service, repo = _service_for_user(OWNER_ID)
    definition = _create_definition(service, name="Scheduled execution")
    scheduler = _RecurringQuestionScheduler()
    scheduler._service_cache[OWNER_ID] = service
    due = datetime(2026, 7, 1, 16, 30, tzinfo=timezone.utc)
    slot = _normalize_slot_to_utc_iso(due)

    first = await scheduler._enqueue_due_slot(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        definition_version=definition.version,
        schedule_slot=due,
    )
    replay = await scheduler._enqueue_due_slot(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        definition_version=definition.version,
        schedule_slot=due,
    )

    assert replay.id == first.id  # nosec B101
    runs, total = repo.list_runs(owner_id=OWNER_ID, definition_id=definition.id)
    assert total == 1  # nosec B101
    assert runs[0].trigger_reason == "scheduled"  # nosec B101
    assert runs[0].schedule_slot == slot  # nosec B101

    jobs = JobManager().list_jobs(
        domain=SCHEDULED_TASKS_DOMAIN,
        queue=RECURRING_QUESTION_QUEUE,
        job_type=RECURRING_QUESTION_JOB_TYPE,
        owner_user_id=str(OWNER_ID),
    )
    assert len(jobs) == 1  # nosec B101
    assert jobs[0]["idempotency_key"] == build_scheduled_run_idempotency_key(  # nosec B101
        definition_id=definition.id,
        definition_version=definition.version,
        schedule_slot=slot,
    )
    assert jobs[0]["payload"]["run_id"] == first.id  # nosec B101


@pytest.mark.asyncio
async def test_enqueue_due_slot_rejects_overlapping_scheduled_runs(
    recurring_question_scheduler_env,
) -> None:
    service, repo = _service_for_user(OWNER_ID)
    definition = _create_definition(service, name="Overlap policy")
    scheduler = _RecurringQuestionScheduler()
    scheduler._service_cache[OWNER_ID] = service
    await scheduler._enqueue_due_slot(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        definition_version=definition.version,
        schedule_slot=datetime(2026, 7, 1, 16, 30, tzinfo=timezone.utc),
    )

    with pytest.raises(ScheduledTaskAutomationError, match="run_in_progress"):
        await scheduler._enqueue_due_slot(
            owner_id=OWNER_ID,
            definition_id=definition.id,
            definition_version=definition.version,
            schedule_slot=datetime(2026, 7, 1, 17, 30, tzinfo=timezone.utc),
        )

    _runs, total = repo.list_runs(owner_id=OWNER_ID, definition_id=definition.id)
    assert total == 1  # nosec B101


def test_reconcile_stale_runs_marks_failed_with_repair_reason(
    recurring_question_scheduler_env,
) -> None:
    service, repo = _service_for_user(OWNER_ID)
    definition = _create_definition(service, name="Stale run repair")
    run = repo.create_run(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        definition_version=definition.version,
        trigger_reason="scheduled",
        status="queued",
        outcome="none",
        scope_snapshot={"mode": "all_searchable_library", "resolved_sources": ["media_db"]},
        finding_policy_snapshot={"preset": "balanced_findings"},
        rag_request_snapshot={"query": "What changed?"},
        run_summary={"message": "Queued before crash."},
        schedule_slot="2026-07-01T16:00:00+00:00",
    )
    stale_updated_at = "2026-07-01T16:00:00+00:00"
    with sqlite3.connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE scheduled_task_runs SET updated_at = ? WHERE owner_id = ? AND id = ?",
            [stale_updated_at, OWNER_ID, run.id],
        )

    repaired = service.reconcile_stale_runs(
        owner_id=OWNER_ID,
        actor="scheduled-task-reconciler",
        now=datetime(2026, 7, 1, 16, 30, tzinfo=timezone.utc),
        stale_after=timedelta(minutes=10),
    )

    updated = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    assert repaired == [run.id]  # nosec B101
    assert updated is not None  # nosec B101
    assert updated.status == "failed"  # nosec B101
    assert updated.outcome == "degraded"  # nosec B101
    assert updated.failure_reason["code"] == "scheduler_repair_stale_run"  # nosec B101
    audit_events = repo.list_audit_events(owner_id=OWNER_ID, definition_id=definition.id, limit=20, offset=0)[0]
    assert "run.repaired" in {event.event_type for event in audit_events}  # nosec B101


def test_weekly_day_strings_are_trimmed_for_cron_trigger() -> None:
    assert _normalize_day_of_week("mon, wed, fri") == "mon,wed,fri"  # nosec B101


def test_invalid_daily_time_raises_instead_of_defaulting_to_midnight() -> None:
    with pytest.raises(ValueError, match="invalid schedule time"):
        _parse_time_fields({"time": "not-a-time"})


def test_reconcile_orphaned_completed_job_marks_run_needs_attention(
    recurring_question_scheduler_env,
) -> None:
    service, repo = _service_for_user(OWNER_ID)
    definition = _create_definition(service, name="Orphaned job repair")
    job = JobManager().create_job(
        domain=SCHEDULED_TASKS_DOMAIN,
        queue=RECURRING_QUESTION_QUEUE,
        job_type=RECURRING_QUESTION_JOB_TYPE,
        payload={"definition_id": definition.id, "owner_user_id": str(OWNER_ID)},
        owner_user_id=str(OWNER_ID),
    )
    jobs_db_path = Path(os.environ["JOBS_DB_PATH"])
    with sqlite3.connect(jobs_db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'completed', result = ? WHERE id = ?",
            ['{"message":"Worker finished without updating run."}', int(job["id"])],
        )
    run = repo.create_run(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        definition_version=definition.version,
        trigger_reason="scheduled",
        status="running",
        outcome="none",
        scope_snapshot={"mode": "all_searchable_library", "resolved_sources": ["media_db"]},
        finding_policy_snapshot={"preset": "balanced_findings"},
        rag_request_snapshot={"query": "What changed?"},
        run_summary={"message": "Running before worker exit."},
        job_id=str(job["id"]),
        schedule_slot="2026-07-01T16:00:00+00:00",
    )
    with sqlite3.connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE scheduled_task_runs SET updated_at = ? WHERE owner_id = ? AND id = ?",
            ["2026-07-01T16:00:00+00:00", OWNER_ID, run.id],
        )

    repaired = service.reconcile_stale_runs(
        owner_id=OWNER_ID,
        actor="scheduled-task-reconciler",
        now=datetime(2026, 7, 1, 16, 30, tzinfo=timezone.utc),
        stale_after=timedelta(minutes=10),
    )

    updated = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    assert repaired == [run.id]  # nosec B101
    assert updated is not None  # nosec B101
    assert updated.status == "failed"  # nosec B101
    assert updated.failure_reason["code"] == "job_completed_without_run_finalization"  # nosec B101
    assert updated.run_summary["needs_attention"] is True  # nosec B101
