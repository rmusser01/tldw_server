"""Tests for ACP Schedule CRUD endpoints and scheduler routing.

Covers:
1. Create schedule stores acp_config_json in DB
2. List schedules returns only ACP schedules (not workflow schedules)
3. Update schedule modifies cron and config
4. Delete schedule removes it
5. _load_all() routes ACP schedules to _add_acp_job not _add_job
6. _load_all() still routes workflow schedules to _add_job
7. Invalid cron expression returns error
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import (
    WorkflowSchedule,
    WorkflowsSchedulerDB,
)
from tldw_Server_API.app.services import workflows_scheduler as workflows_scheduler_mod
from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def scheduler():
    """Provide a started scheduler service; stop on teardown."""
    svc = get_workflows_scheduler()
    asyncio.run(svc.start())
    yield svc
    try:
        asyncio.run(svc.stop())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Create schedule stores acp_config_json in DB
# ---------------------------------------------------------------------------

def test_create_acp_schedule_stores_config(scheduler):
    svc = scheduler
    acp_config = {
        "prompt": "Summarize new emails",
        "cwd": "/workspace",
        "agent_type": "coding",
        "model": "claude-sonnet-4-20250514",
        "sandbox_enabled": True,
    }
    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="Daily agent run",
        cron="0 9 * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        acp_config_json=json.dumps(acp_config),
    )

    s = svc.get(sid)
    assert s is not None
    assert s.acp_config_json is not None
    parsed = json.loads(s.acp_config_json)
    assert parsed["prompt"] == "Summarize new emails"
    assert parsed["cwd"] == "/workspace"
    assert parsed["agent_type"] == "coding"
    assert parsed["model"] == "claude-sonnet-4-20250514"
    assert parsed["sandbox_enabled"] is True
    assert s.name == "Daily agent run"
    assert s.cron == "0 9 * * *"


# ---------------------------------------------------------------------------
# 2. List schedules returns only ACP schedules
# ---------------------------------------------------------------------------

def test_list_returns_only_acp_schedules(scheduler):
    svc = scheduler

    # Create a plain workflow schedule (no acp_config_json)
    wf_id = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=42,
        name="workflow-only",
        cron="*/10 * * * *",
        timezone="UTC",
        inputs={"step": 1},
        run_mode="async",
        validation_mode="block",
        enabled=True,
    )

    # Create an ACP schedule
    acp_id = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="ACP schedule",
        cron="0 12 * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        acp_config_json=json.dumps({"prompt": "Run daily check"}),
    )

    # List all schedules for the user
    all_schedules = svc.list(tenant_id="default", user_id="1", limit=100)

    # Filter to ACP-only (as the endpoint would)
    acp_only = [s for s in all_schedules if s.acp_config_json]
    workflow_only = [s for s in all_schedules if not s.acp_config_json]

    acp_ids = [s.id for s in acp_only]
    wf_ids = [s.id for s in workflow_only]

    assert acp_id in acp_ids
    assert wf_id not in acp_ids
    assert wf_id in wf_ids


# ---------------------------------------------------------------------------
# 3. Update schedule modifies cron and config
# ---------------------------------------------------------------------------

def test_update_acp_schedule(scheduler):
    svc = scheduler

    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="Update me",
        cron="0 8 * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        acp_config_json=json.dumps({"prompt": "Original prompt", "cwd": "/old"}),
    )

    # Update cron and acp_config_json
    new_config = {"prompt": "Updated prompt", "cwd": "/new"}
    ok = svc.update(sid, {
        "cron": "30 10 * * *",
        "acp_config_json": json.dumps(new_config),
    })
    assert ok is True

    s = svc.get(sid)
    assert s is not None
    assert s.cron == "30 10 * * *"
    parsed = json.loads(s.acp_config_json)
    assert parsed["prompt"] == "Updated prompt"
    assert parsed["cwd"] == "/new"


# ---------------------------------------------------------------------------
# 4. Delete schedule removes it
# ---------------------------------------------------------------------------

def test_delete_acp_schedule(scheduler):
    svc = scheduler

    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="Delete me",
        cron="0 6 * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        acp_config_json=json.dumps({"prompt": "Bye"}),
    )

    assert svc.get(sid) is not None
    ok = svc.delete(sid)
    assert ok is True
    assert svc.get(sid) is None


# ---------------------------------------------------------------------------
# 5. _load_all() routes ACP schedules to _add_acp_job
# ---------------------------------------------------------------------------

def test_list_registered_schedules_falls_back_to_list_schedules(monkeypatch):
    """Older/test scheduler DB handles may not expose list_all_schedules."""
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    acp_schedule = WorkflowSchedule(
        id="legacy-acp-1",
        tenant_id="default",
        user_id="7",
        workflow_id=None,
        name="legacy acp schedule",
        cron="0 9 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=0,
        coalesce=True,
        jitter_sec=0,
        acp_config_json='{"prompt": "legacy"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    class FakeDB:
        def __init__(self):
            self.calls: list[dict[str, Any]] = []

        def list_schedules(self, **kwargs):
            self.calls.append(kwargs)
            return [acp_schedule]

    fake_db = FakeDB()
    monkeypatch.setattr(svc, "_get_db", lambda _uid: fake_db)

    schedules = svc._list_registered_schedules(7)

    assert schedules == [acp_schedule]
    assert fake_db.calls == [
        {"tenant_id": "default", "user_id": None, "limit": 1000, "offset": 0}
    ]


@pytest.mark.asyncio
async def test_load_all_routes_acp_to_add_acp_job(monkeypatch, tmp_path):
    """When _load_all encounters a schedule with acp_config_json, it must
    call _add_acp_job instead of _add_job."""
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    await svc.start()

    acp_schedule = WorkflowSchedule(
        id="acp-1",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="acp schedule",
        cron="0 9 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=0,
        coalesce=True,
        jitter_sec=0,
        acp_config_json='{"prompt": "hello"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    class FakeDB:
        def list_schedules(self, **kwargs):
            return [acp_schedule]

    monkeypatch.setattr(svc, "_get_db", lambda uid: FakeDB())
    monkeypatch.setattr(
        workflows_scheduler_mod.DatabasePaths,
        "get_user_db_base_dir",
        lambda: tmp_path,
    )
    (tmp_path / "1").mkdir()

    add_job_calls = []
    add_acp_job_calls = []
    monkeypatch.setattr(svc, "_add_job", lambda s, uid: add_job_calls.append((s.id, uid)))
    monkeypatch.setattr(svc, "_add_acp_job", lambda s, uid: add_acp_job_calls.append((s.id, uid)))

    await svc._load_all()

    assert len(add_acp_job_calls) >= 1, "Expected _add_acp_job to be called for ACP schedule"
    assert any(call[0] == "acp-1" for call in add_acp_job_calls)
    assert not any(call[0] == "acp-1" for call in add_job_calls), \
        "_add_job should NOT be called for ACP schedules"

    await svc.stop()


# ---------------------------------------------------------------------------
# 6. _load_all() still routes workflow schedules to _add_job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_all_routes_workflow_to_add_job(monkeypatch, tmp_path):
    """When _load_all encounters a schedule without acp_config_json, it must
    call _add_job (not _add_acp_job)."""
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    await svc.start()

    workflow_schedule = WorkflowSchedule(
        id="wf-1",
        tenant_id="default",
        user_id="1",
        workflow_id=42,
        name="workflow schedule",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs_json='{"step": 1}',
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=300,
        coalesce=True,
        jitter_sec=0,
        acp_config_json=None,
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    class FakeDB:
        def list_schedules(self, **kwargs):
            return [workflow_schedule]

    monkeypatch.setattr(svc, "_get_db", lambda uid: FakeDB())
    monkeypatch.setattr(
        workflows_scheduler_mod.DatabasePaths,
        "get_user_db_base_dir",
        lambda: tmp_path,
    )
    (tmp_path / "1").mkdir()

    add_job_calls = []
    add_acp_job_calls = []
    monkeypatch.setattr(svc, "_add_job", lambda s, uid: add_job_calls.append((s.id, uid)))
    monkeypatch.setattr(svc, "_add_acp_job", lambda s, uid: add_acp_job_calls.append((s.id, uid)))

    await svc._load_all()

    assert len(add_job_calls) >= 1, "Expected _add_job to be called for workflow schedule"
    assert any(call[0] == "wf-1" for call in add_job_calls)
    assert not any(call[0] == "wf-1" for call in add_acp_job_calls), \
        "_add_acp_job should NOT be called for workflow schedules"

    await svc.stop()


# ---------------------------------------------------------------------------
# 7. Invalid cron expression returns error
# ---------------------------------------------------------------------------

def test_invalid_cron_raises_on_add_acp_job(scheduler):
    """_add_acp_job should handle invalid cron gracefully (log warning, no crash)."""
    svc = scheduler

    bad_schedule = WorkflowSchedule(
        id="bad-cron-1",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="bad cron",
        cron="not a valid cron",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=300,
        coalesce=True,
        jitter_sec=0,
        acp_config_json='{"prompt": "test"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    # Should not raise -- invalid cron is handled gracefully
    svc._add_acp_job(bad_schedule, 1)

    # The bad job should NOT be registered in APScheduler
    jobs = svc._aps.get_jobs() if svc._aps else []
    assert not any(j.id == "bad-cron-1" for j in jobs)


def test_invalid_cron_via_endpoint_validation():
    """Validate cron expressions using the endpoint validation helper."""
    from tldw_Server_API.app.api.v1.endpoints.acp_schedules import _validate_cron
    from fastapi import HTTPException

    # Valid cron should not raise
    _validate_cron("0 9 * * *")
    _validate_cron("*/5 * * * *")

    # Invalid cron should raise HTTPException 422
    with pytest.raises(HTTPException) as exc_info:
        _validate_cron("not a cron")
    assert exc_info.value.status_code == 422
    assert "Invalid cron" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_add_acp_job_applies_explicit_concurrency_metadata(monkeypatch):
    """ACP schedule APS jobs should expose queue vs skip concurrency behavior."""
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    captured_jobs: list[dict[str, Any]] = []
    history_updates: list[dict[str, Any]] = []

    class FakeAPS:
        def remove_job(self, _job_id: str) -> None:
            return None

        def add_job(self, func, **kwargs):
            captured_jobs.append({"func": func, **kwargs})

    class FakeDB:
        def set_history(self, _schedule_id: str, **kwargs):
            history_updates.append(kwargs)

    monkeypatch.setattr(svc, "_get_db", lambda _uid: FakeDB())
    svc._aps = FakeAPS()  # type: ignore[assignment]

    queue_schedule = WorkflowSchedule(
        id="acp-queue",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="queue mode",
        cron="0 9 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="queue",
        misfire_grace_sec=123,
        coalesce=False,
        jitter_sec=0,
        acp_config_json='{"prompt": "queue"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    skip_schedule = WorkflowSchedule(
        id="acp-skip",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="skip mode",
        cron="0 10 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=0,
        coalesce=True,
        jitter_sec=0,
        acp_config_json='{"prompt": "skip"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    svc._add_acp_job(queue_schedule, 1)
    svc._add_acp_job(skip_schedule, 1)

    assert captured_jobs[0]["id"] == "acp-queue"
    assert captured_jobs[0]["max_instances"] == 3
    assert captured_jobs[0]["coalesce"] is False
    assert captured_jobs[0]["misfire_grace_time"] == 123
    assert captured_jobs[1]["id"] == "acp-skip"
    assert captured_jobs[1]["max_instances"] == 1
    assert captured_jobs[1]["coalesce"] is True
    assert captured_jobs[1]["misfire_grace_time"] == 0
    assert history_updates

def test_create_with_message_list_prompt(scheduler):
    """ACP prompts can be a list of message dicts, not just a string."""
    svc = scheduler
    prompt = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Summarize today's logs."},
    ]
    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="Multi-turn",
        cron="0 0 * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        acp_config_json=json.dumps({"prompt": prompt}),
    )

    s = svc.get(sid)
    assert s is not None
    parsed = json.loads(s.acp_config_json)
    assert isinstance(parsed["prompt"], list)
    assert len(parsed["prompt"]) == 2


@pytest.mark.asyncio
async def test_run_acp_schedule_submits_acp_run():
    """_run_acp_schedule should submit an acp_run task (not workflow_run)."""
    svc = get_workflows_scheduler()
    await svc.start()

    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="acp-fire-test",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        acp_config_json=json.dumps({
            "prompt": "Run check",
            "cwd": "/workspace",
            "agent_type": "coding",
        }),
    )

    captured: dict[str, Any] = {}

    class _StubScheduler:
        async def submit(self, *args: Any, **kwargs: Any) -> str:
            captured["handler"] = kwargs.get("handler")
            captured["payload"] = kwargs.get("payload")
            captured["queue_name"] = kwargs.get("queue_name")
            return "task-acp-1"

    svc._core_scheduler = _StubScheduler()  # type: ignore[attr-defined]

    await svc._run_acp_schedule(sid, 1)  # type: ignore[attr-defined]

    assert captured.get("handler") == "acp_run", f"Expected acp_run, got {captured.get('handler')}"
    assert captured.get("queue_name") == "acp"
    payload = captured.get("payload", {})
    assert payload.get("prompt") == "Run check"
    assert payload.get("cwd") == "/workspace"
    assert payload.get("agent_type") == "coding"

    await svc.stop()


@pytest.mark.asyncio
async def test_run_acp_schedule_records_skipped_disabled(monkeypatch):
    """Disabled schedules should leave an operator-visible skipped state."""
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    disabled_schedule = WorkflowSchedule(
        id="acp-disabled",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="disabled",
        cron="0 9 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=False,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=300,
        coalesce=True,
        jitter_sec=0,
        acp_config_json='{"prompt": "disabled"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    history_updates: list[dict[str, Any]] = []

    class FakeDB:
        def get_schedule(self, _schedule_id: str):
            return disabled_schedule

        def set_history(self, _schedule_id: str, **kwargs):
            history_updates.append(kwargs)

    monkeypatch.setattr(svc, "_get_db", lambda _uid: FakeDB())

    await svc._run_acp_schedule("acp-disabled", 1)

    assert history_updates == [{"last_status": "skipped_disabled"}]


@pytest.mark.asyncio
async def test_run_acp_schedule_records_error_on_submit_failure(monkeypatch):
    """ACP schedule submit failures should be visible through schedule history."""
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    enabled_schedule = WorkflowSchedule(
        id="acp-submit-fails",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="submit fails",
        cron="0 9 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=300,
        coalesce=True,
        jitter_sec=0,
        acp_config_json='{"prompt": "fail"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    history_updates: list[dict[str, Any]] = []

    class FakeDB:
        def get_schedule(self, _schedule_id: str):
            return enabled_schedule

        def set_history(self, _schedule_id: str, **kwargs):
            history_updates.append(kwargs)

    class FailingScheduler:
        async def submit(self, **_kwargs):
            raise RuntimeError("scheduler submit failed")

    monkeypatch.setattr(svc, "_get_db", lambda _uid: FakeDB())
    svc._core_scheduler = FailingScheduler()  # type: ignore[assignment]

    await svc._run_acp_schedule("acp-submit-fails", 1)

    assert history_updates[0]["last_status"] == "pending"
    assert history_updates[-1] == {"last_status": "error"}


@pytest.mark.asyncio
async def test_rescan_routes_acp_schedules(monkeypatch, tmp_path):
    """_rescan_once should also route ACP schedules to _add_acp_job."""
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    await svc.start()

    acp_schedule = WorkflowSchedule(
        id="acp-rescan-1",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="acp rescan",
        cron="0 9 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=300,
        coalesce=True,
        jitter_sec=0,
        acp_config_json='{"prompt": "rescan test"}',
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    class FakeDB:
        def list_schedules(self, **kwargs):
            return [acp_schedule]

    monkeypatch.setattr(svc, "_get_db", lambda uid: FakeDB())
    monkeypatch.setattr(
        workflows_scheduler_mod.DatabasePaths,
        "get_user_db_base_dir",
        lambda: tmp_path,
    )
    (tmp_path / "1").mkdir()

    add_job_calls = []
    add_acp_job_calls = []
    monkeypatch.setattr(svc, "_add_job", lambda s, uid: add_job_calls.append((s.id, uid)))
    monkeypatch.setattr(svc, "_add_acp_job", lambda s, uid: add_acp_job_calls.append((s.id, uid)))

    await svc._rescan_once()

    assert any(call[0] == "acp-rescan-1" for call in add_acp_job_calls), \
        "Expected _add_acp_job to be called during rescan for ACP schedule"
    assert not any(call[0] == "acp-rescan-1" for call in add_job_calls), \
        "_add_job should NOT be called during rescan for ACP schedule"

    await svc.stop()


def test_acp_config_json_in_update_schedule_db(scheduler):
    """update_schedule handles acp_config_json dict serialization."""
    svc = scheduler

    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="config-update",
        cron="0 8 * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        acp_config_json=json.dumps({"prompt": "original"}),
    )

    # Update with a dict value (should be serialized)
    db = svc._get_db(1)
    ok = db.update_schedule(sid, {"acp_config_json": {"prompt": "updated via dict"}})
    assert ok is True

    s = db.get_schedule(sid)
    assert s is not None
    parsed = json.loads(s.acp_config_json)
    assert parsed["prompt"] == "updated via dict"


def test_schedule_response_exposes_operator_state_and_concurrency():
    """Schedule responses should expose state needed for operator triage."""
    from tldw_Server_API.app.api.v1.endpoints.acp_schedules import _schedule_to_response

    schedule = WorkflowSchedule(
        id="acp-visible-state",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="visible state",
        cron="0 9 * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="queue",
        misfire_grace_sec=120,
        coalesce=False,
        jitter_sec=0,
        acp_config_json='{"prompt": "state"}',
        last_run_at="2026-05-10T09:00:00+00:00",
        next_run_at="2026-05-11T09:00:00+00:00",
        last_status="error",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    response = _schedule_to_response(schedule)

    assert response["last_status"] == "error"
    assert response["next_run_at"] == "2026-05-11T09:00:00+00:00"
    assert response["concurrency_mode"] == "queue"
    assert response["misfire_grace_sec"] == 120
    assert response["coalesce"] is False


@pytest.mark.asyncio
async def test_create_acp_schedule_passes_concurrency_controls(monkeypatch):
    """The ACP schedule API should persist explicit queue-vs-skip behavior."""
    from tldw_Server_API.app.api.v1.endpoints import acp_schedules as schedules_api

    class User:
        id = 1

    class FakeScheduler:
        def __init__(self):
            self.create_kwargs: dict[str, Any] = {}

        def create(self, **kwargs):
            self.create_kwargs = kwargs
            return "acp-concurrency"

        def get(self, _schedule_id: str):
            return WorkflowSchedule(
                id="acp-concurrency",
                tenant_id="default",
                user_id="1",
                workflow_id=None,
                name=self.create_kwargs["name"],
                cron=self.create_kwargs["cron"],
                timezone=self.create_kwargs["timezone"],
                inputs_json="{}",
                run_mode="async",
                validation_mode="block",
                enabled=True,
                require_online=False,
                concurrency_mode=self.create_kwargs["concurrency_mode"],
                misfire_grace_sec=self.create_kwargs["misfire_grace_sec"],
                coalesce=self.create_kwargs["coalesce"],
                jitter_sec=0,
                acp_config_json=self.create_kwargs["acp_config_json"],
                last_run_at=None,
                next_run_at="2026-05-11T09:00:00+00:00",
                last_status=None,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )

    fake_scheduler = FakeScheduler()
    monkeypatch.setattr(schedules_api, "_get_scheduler", lambda: fake_scheduler)

    response = await schedules_api.create_acp_schedule(
        schedules_api.ACPScheduleCreate(
            name="Queue run",
            cron="0 9 * * *",
            timezone="UTC",
            prompt="run",
            concurrency_mode="queue",
            misfire_grace_sec=120,
            coalesce=False,
        ),
        user=User(),
    )

    assert fake_scheduler.create_kwargs["concurrency_mode"] == "queue"
    assert fake_scheduler.create_kwargs["misfire_grace_sec"] == 120
    assert fake_scheduler.create_kwargs["coalesce"] is False
    assert response["concurrency_mode"] == "queue"
    assert response["misfire_grace_sec"] == 120
    assert response["next_run_at"] == "2026-05-11T09:00:00+00:00"


@pytest.mark.asyncio
async def test_create_acp_schedule_preserves_zero_misfire_grace_sec(monkeypatch):
    """Explicit zero misfire grace should survive API create and response mapping."""
    from tldw_Server_API.app.api.v1.endpoints import acp_schedules as schedules_api

    class User:
        id = 1

    class FakeScheduler:
        def __init__(self):
            self.create_kwargs: dict[str, Any] = {}

        def create(self, **kwargs):
            self.create_kwargs = kwargs
            return "acp-zero-misfire"

        def get(self, _schedule_id: str):
            return WorkflowSchedule(
                id="acp-zero-misfire",
                tenant_id="default",
                user_id="1",
                workflow_id=None,
                name=self.create_kwargs["name"],
                cron=self.create_kwargs["cron"],
                timezone=self.create_kwargs["timezone"],
                inputs_json="{}",
                run_mode="async",
                validation_mode="block",
                enabled=True,
                require_online=False,
                concurrency_mode=self.create_kwargs["concurrency_mode"],
                misfire_grace_sec=self.create_kwargs["misfire_grace_sec"],
                coalesce=self.create_kwargs["coalesce"],
                jitter_sec=0,
                acp_config_json=self.create_kwargs["acp_config_json"],
                last_run_at=None,
                next_run_at=None,
                last_status=None,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            )

    fake_scheduler = FakeScheduler()
    monkeypatch.setattr(schedules_api, "_get_scheduler", lambda: fake_scheduler)

    response = await schedules_api.create_acp_schedule(
        schedules_api.ACPScheduleCreate(
            name="Zero misfire",
            cron="0 9 * * *",
            timezone="UTC",
            prompt="run",
            concurrency_mode="skip",
            misfire_grace_sec=0,
            coalesce=True,
        ),
        user=User(),
    )

    assert fake_scheduler.create_kwargs["misfire_grace_sec"] == 0
    assert response["misfire_grace_sec"] == 0


def test_acp_schedule_rejects_negative_misfire_grace_sec():
    """Negative misfire grace values should fail Pydantic/FastAPI validation."""
    from tldw_Server_API.app.api.v1.endpoints import acp_schedules as schedules_api

    with pytest.raises(ValidationError):
        schedules_api.ACPScheduleCreate(
            name="Bad misfire",
            cron="0 9 * * *",
            timezone="UTC",
            prompt="run",
            misfire_grace_sec=-1,
        )

    with pytest.raises(ValidationError):
        schedules_api.ACPScheduleUpdate(misfire_grace_sec=-1)
