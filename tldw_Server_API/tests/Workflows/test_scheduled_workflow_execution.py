"""Real scheduled-workflow routing, admission and snapshot regressions."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.api.v1.endpoints import scheduler_workflows as routes
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Scheduler import Scheduler, SchedulerConfig
from tldw_Server_API.app.core.Scheduler.base.exceptions import SchedulerError
from tldw_Server_API.app.core.Scheduler.handlers import workflows as handler
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine, WorkflowScheduler
from tldw_Server_API.app.services import workflows_scheduler as recurring

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
MARKER = "DUE-ORBIT-742"
DEFINITION = {
    "name": "scheduled-marker",
    "steps": [{"id": "marker", "type": "template_render", "config": {"template": "{{ inputs.marker }}"}}],
}


@pytest.fixture
def owned_db(tmp_path, monkeypatch):
    db = WorkflowsDatabase(db_path=str(tmp_path / "workflows.db"))
    previous_scheduler = WorkflowScheduler._inst
    WorkflowScheduler._inst = None
    monkeypatch.setattr(handler, "_get_wf_db", lambda: db)
    yield db
    WorkflowScheduler._inst = previous_scheduler
    db.close()


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "schedules.db"))
    monkeypatch.delenv("WORKFLOWS_MINT_VIRTUAL_KEYS", raising=False)
    svc = recurring._WFRecurringScheduler()
    monkeypatch.setattr(routes, "get_workflows_scheduler", lambda: svc)
    yield svc
    for db in svc._db_cache.values():
        db.backend.get_pool().close_all()


def saved_definition(db, *, tenant="default", owner="1", definition=None):
    return db.create_definition(
        tenant_id=tenant, name="scheduled-marker", version=7, owner_id=owner,
        visibility="private", description=None, tags=[],
        definition=DEFINITION if definition is None else definition,
    )


def create_schedule(svc, workflow_id, *, enabled=True):
    return svc.create(
        tenant_id="default", user_id="1", workflow_id=workflow_id,
        name="due-marker", cron="* * * * *", timezone="UTC",
        inputs={"marker": MARKER}, run_mode="async", validation_mode="block", enabled=enabled,
    )


async def wait_for(predicate):
    async with asyncio.timeout(5):
        while not (value := predicate()):
            await asyncio.sleep(0.02)
        return value


@pytest.mark.parametrize("tenant, expected", [(None, "default"), ("", "default"), ("   ", "default"), (" tenant-a ", "tenant-a"), ("None", "None")])
async def test_create_uses_canonical_tenant_claim(service, tenant, expected):
    user = User(id=1, username="owner", is_active=True, tenant_id=tenant)
    response = await routes.create_schedule(
        routes.ScheduleCreateRequest(cron="* * * * *", enabled=False), current_user=user,
    )
    record = service.get(response["id"])
    assert record.tenant_id == expected  # nosec B101 # pytest behavior oracle, not production authorization
    assert record.user_id == "1"  # nosec B101 # pytest behavior oracle, not production authorization


@pytest.mark.parametrize("tenant", [None, "", "   "])
async def test_list_finds_default_tenant_schedule_for_absent_claim(service, tenant):
    schedule_id = create_schedule(service, None, enabled=False)
    response = await routes.list_schedules(
        owner=None, limit=50, offset=0,
        current_user=User(id=1, username="owner", is_active=True, tenant_id=tenant),
    )
    assert [record.id for record in response] == [schedule_id]  # nosec B101 # pytest behavior oracle, not production authorization


async def test_due_schedule_admits_real_workflows_worker_and_executes_saved_marker(service, owned_db, tmp_path):
    core = Scheduler(SchedulerConfig(
        database_url=f"sqlite:///{tmp_path}/tasks.db", base_path=tmp_path / "scheduler",
        min_workers=1, max_workers=2, write_buffer_flush_interval=0.01,
    ))
    await core.start()
    service._core_scheduler = core
    workflow_id = saved_definition(owned_db)
    schedule_id = create_schedule(service, workflow_id)
    try:
        assert core.get_status()["worker_pool"]["workers_by_queue"] == {"default": 1}  # nosec B101 # pytest behavior oracle, not production authorization
        await service._run_schedule(schedule_id, user_id=1)
        assert core.get_status()["worker_pool"]["workers_by_queue"].get("workflows", 0) >= 1  # nosec B101 # pytest behavior oracle, not production authorization
        runs = await wait_for(lambda: owned_db.list_runs(tenant_id="default", user_id="1", workflow_id=workflow_id))
        run = await wait_for(lambda: (r if (r := owned_db.get_run(runs[0].run_id)).status == "succeeded" else None))
        assert json.loads(run.outputs_json)["text"] == MARKER  # nosec B101 # pytest behavior oracle, not production authorization
        assert run.definition_version == 7  # nosec B101 # pytest behavior oracle, not production authorization
        assert json.loads(run.definition_snapshot_json) == DEFINITION  # nosec B101 # pytest behavior oracle, not production authorization
        assert owned_db.list_step_runs(run_id=run.run_id)[0]["step_id"] == "marker"  # nosec B101 # pytest behavior oracle, not production authorization
        assert service.get(schedule_id).last_status == "queued"  # nosec B101 # pytest behavior oracle, not production authorization
    finally:
        await core.stop()


@pytest.mark.parametrize("failure", ["capacity", "scheduler_error"])
async def test_worker_admission_failure_refuses_enqueue(service, owned_db, tmp_path, monkeypatch, failure):
    core = Scheduler(SchedulerConfig(
        database_url=f"sqlite:///{tmp_path}/tasks.db", base_path=tmp_path / "scheduler",
        min_workers=1, max_workers=1, write_buffer_flush_interval=0.01,
    ))
    await core.start()
    service._core_scheduler = core
    if failure == "scheduler_error":
        monkeypatch.setattr(core, "scale_workers", AsyncMock(side_effect=SchedulerError("worker unavailable")))
    schedule_id = create_schedule(service, saved_definition(owned_db))
    try:
        await service._run_schedule(schedule_id, user_id=1)
        assert service.get(schedule_id).last_status == "error"  # nosec B101 # pytest behavior oracle, not production authorization
        await core.write_buffer.flush()
        assert await core.backend.get_ready_tasks("workflows") == []  # nosec B101 # pytest behavior oracle, not production authorization
        assert owned_db.list_runs(tenant_id="default", user_id="1") == []  # nosec B101 # pytest behavior oracle, not production authorization
    finally:
        await core.stop()


async def test_existing_workflows_workers_are_not_downscaled(service, owned_db, tmp_path):
    core = Scheduler(SchedulerConfig(
        database_url=f"sqlite:///{tmp_path}/tasks.db", base_path=tmp_path / "scheduler",
        min_workers=0, max_workers=2, write_buffer_flush_interval=0.01,
    ))
    await core.start()
    service._core_scheduler = core
    await core.scale_workers(2, "workflows")
    schedule_id = create_schedule(service, saved_definition(owned_db))
    try:
        await service._run_schedule(schedule_id, user_id=1)
        runs = await wait_for(lambda: owned_db.list_runs(tenant_id="default", user_id="1"))
        await wait_for(lambda: owned_db.get_run(runs[0].run_id).status == "succeeded")
        assert core.get_status()["worker_pool"]["workers_by_queue"]["workflows"] == 2  # nosec B101 # pytest behavior oracle, not production authorization
        assert json.loads(owned_db.get_run(runs[0].run_id).outputs_json)["text"] == MARKER  # nosec B101 # pytest behavior oracle, not production authorization
    finally:
        await core.stop()


async def test_saved_handler_freezes_owned_definition_and_version(owned_db):
    workflow_id = saved_definition(owned_db)
    payload = {"user_id": "1", "tenant_id": "default", "workflow_id": workflow_id,
               "inputs": {"marker": MARKER}, "mode": "sync", "metadata": {"source": "schedule"},
               "definition_snapshot": {"steps": []}}
    result = await handler.workflow_run(payload)
    run = owned_db.get_run(result["run_id"])
    assert json.loads(run.outputs_json)["text"] == MARKER  # nosec B101 # pytest behavior oracle, not production authorization
    assert run.definition_version == 7  # nosec B101 # pytest behavior oracle, not production authorization
    assert json.loads(run.definition_snapshot_json) == {**DEFINITION, "metadata": {"source": "schedule"}}  # nosec B101 # pytest behavior oracle, not production authorization
    assert payload["definition_snapshot"] == {"steps": []}  # nosec B101 # pytest behavior oracle, not production authorization


@pytest.mark.parametrize("boundary", ["foreign_owner", "foreign_tenant", "missing", "malformed"])
async def test_saved_definition_refusal_precedes_quota_and_run_creation(owned_db, monkeypatch, boundary):
    workflow_id = 999 if boundary == "missing" else saved_definition(
        owned_db, owner="2" if boundary == "foreign_owner" else "1",
        tenant="other" if boundary == "foreign_tenant" else "default",
        definition=[] if boundary == "malformed" else None,
    )
    consume = AsyncMock(return_value=quota_checks.UNLIMITED)
    monkeypatch.setattr(quota_checks, "workflows_runs_consume", consume)
    with pytest.raises(ValueError):
        await handler.workflow_run({"user_id": "1", "tenant_id": "default", "workflow_id": workflow_id,
                                    "inputs": {"marker": MARKER}, "mode": "sync"})
    assert consume.await_count == 0  # nosec B101 # pytest behavior oracle, not production authorization
    assert owned_db.list_runs(tenant_id="default", user_id="1") == []  # nosec B101 # pytest behavior oracle, not production authorization


async def test_valid_saved_definition_still_obeys_quota_refusal(owned_db, monkeypatch):
    workflow_id = saved_definition(owned_db)
    monkeypatch.setattr(quota_checks, "workflows_runs_consume", AsyncMock(return_value=SimpleNamespace(allowed=False)))
    with pytest.raises(RuntimeError, match="Daily workflow run quota exceeded"):
        await handler.workflow_run({"user_id": "1", "tenant_id": "default", "workflow_id": workflow_id})
    assert owned_db.list_runs(tenant_id="default", user_id="1") == []  # nosec B101 # pytest behavior oracle, not production authorization


async def test_ad_hoc_definition_executes_without_saved_lookup(owned_db):
    result = await handler.workflow_run({"user_id": "1", "tenant_id": "default",
                                       "definition_snapshot": DEFINITION, "inputs": {"marker": MARKER}, "mode": "sync"})
    run = owned_db.get_run(result["run_id"])
    assert json.loads(run.outputs_json)["text"] == MARKER  # nosec B101 # pytest behavior oracle, not production authorization
    assert run.workflow_id is None  # nosec B101 # pytest behavior oracle, not production authorization
    assert run.definition_version is None  # nosec B101 # pytest behavior oracle, not production authorization


@pytest.mark.parametrize("foreign", ["owner", "tenant"])
async def test_resume_preserves_existing_owner_and_tenant_refusal(owned_db, foreign):
    owned_db.create_run(run_id="resume", tenant_id="other" if foreign == "tenant" else "default",
                        user_id="2" if foreign == "owner" else "1", inputs={}, definition_snapshot=DEFINITION)
    with pytest.raises(ValueError, match="resume run not found"):
        await handler.workflow_run({"user_id": "1", "tenant_id": "default", "resume_run_id": "resume", "resume_step_id": "marker"})
    assert owned_db.get_run("resume").status == "queued"  # nosec B101 # pytest behavior oracle, not production authorization


async def test_cancelled_owned_resume_does_not_execute_again(owned_db):
    owned_db.create_run(run_id="resume", tenant_id="default", user_id="1", inputs={"marker": MARKER}, definition_snapshot=DEFINITION)
    WorkflowEngine(owned_db).cancel("resume")
    await handler.workflow_run({"user_id": "1", "tenant_id": "default", "resume_run_id": "resume", "resume_step_id": "marker"})
    assert owned_db.get_run("resume").status == "cancelled"  # nosec B101 # pytest behavior oracle, not production authorization
    assert owned_db.list_step_runs(run_id="resume") == []  # nosec B101 # pytest behavior oracle, not production authorization


@pytest.mark.parametrize("capacity", [1, 2])
async def test_run_now_uses_same_real_worker_admission(service, owned_db, tmp_path, monkeypatch, capacity):
    from fastapi import HTTPException

    core = Scheduler(SchedulerConfig(
        database_url=f"sqlite:///{tmp_path}/tasks.db", base_path=tmp_path / "scheduler",
        min_workers=1, max_workers=capacity, write_buffer_flush_interval=0.01,
    ))
    await core.start()
    async def existing_core():
        return core
    monkeypatch.setattr(routes, "get_global_scheduler", existing_core)
    workflow_id = saved_definition(owned_db)
    schedule_id = create_schedule(service, workflow_id, enabled=False)
    user = User(id=1, username="owner", is_active=True, tenant_id="default")
    try:
        if capacity == 1:
            with pytest.raises(HTTPException) as refused:
                await routes.run_now(schedule_id, current_user=user)
            assert refused.value.status_code == 503  # nosec B101 # pytest refusal oracle
            await core.write_buffer.flush()
            assert await core.backend.get_ready_tasks("workflows") == []  # nosec B101 # pytest no-enqueue oracle
        else:
            response = await routes.run_now(schedule_id, current_user=user)
            assert core.get_status()["worker_pool"]["workers_by_queue"].get("workflows", 0) >= 1  # nosec B101 # pytest admission oracle
            runs = await wait_for(lambda: owned_db.list_runs(tenant_id="default", user_id="1", workflow_id=workflow_id))
            run = await wait_for(lambda: (r if (r := owned_db.get_run(runs[0].run_id)).status == "succeeded" else None))
            assert json.loads(run.outputs_json)["text"] == MARKER  # nosec B101 # pytest actual execution oracle
            assert await core.get_task(response["task_id"]) is not None  # nosec B101 # pytest actual task oracle
    finally:
        await core.stop()
