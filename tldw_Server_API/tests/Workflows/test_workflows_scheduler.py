from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import (
    WorkflowSchedule,
    WorkflowsSchedulerDB,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.services import workflows_scheduler as workflows_scheduler_mod
from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler


pytestmark = pytest.mark.unit


def _schedule(schedule_id: str, *, inputs_json: str = "{}", user_id: str = "1") -> WorkflowSchedule:
    return WorkflowSchedule(
        id=schedule_id,
        tenant_id="default",
        user_id=user_id,
        workflow_id=None,
        name=schedule_id,
        cron="*/5 * * * *",
        timezone="UTC",
        inputs_json=inputs_json,
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
        jitter_sec=0,
        acp_config_json=None,
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture()
def client_admin(monkeypatch, auth_headers):
    async def override_user():
        # Admin user for owner overrides
        u = User(id=1, username="admin", email=None, is_active=True)
        setattr(u, "is_admin", True)
        setattr(u, "tenant_id", "default")
        return u

    fastapi_app.dependency_overrides[get_request_user] = override_user

    # Ensure scheduler is started for tests that need APScheduler instance
    svc = get_workflows_scheduler()
    asyncio.run(svc.start())

    with TestClient(fastapi_app, headers=auth_headers) as client:
        yield client, svc

    # Teardown
    try:
        asyncio.run(svc.stop())
    except Exception:
        _ = None
    fastapi_app.dependency_overrides.clear()


def test_cron_validation_422(client_admin):
    client, _ = client_admin
    bad = {
        "cron": "not a cron",
        "timezone": "UTC",
        "inputs": {},
    }
    r = client.post("/api/v1/scheduler/workflows/dry-run", json=bad)
    assert r.status_code == 422


def test_dry_run_returns_next_run(client_admin):
    client, _ = client_admin
    body = {
        "cron": "*/15 * * * *",
        "timezone": "UTC",
        "inputs": {"x": 1},
    }
    r = client.post("/api/v1/scheduler/workflows/dry-run", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("valid") is True
    assert isinstance(data.get("next_run_at"), str) and len(data["next_run_at"]) > 0


def test_concurrency_mode_mapping_and_job_defaults(client_admin):
    client, svc = client_admin
    # queue mode should set max_instances > 1 and coalesce False
    body = {
        "workflow_id": None,
        "name": "q1",
        "cron": "*/30 * * * *",
        "timezone": "UTC",
        "inputs": {},
        "run_mode": "async",
        "validation_mode": "block",
        "enabled": True,
        "concurrency_mode": "queue",
        "misfire_grace_sec": 123,
        "coalesce": False,
    }
    rid = client.post("/api/v1/scheduler/workflows", json=body).json()["id"]
    jobs = svc._aps.get_jobs() if getattr(svc, "_aps", None) else []  # type: ignore[attr-defined]
    assert jobs, "Expected a scheduled job to be registered"
    job = next((j for j in jobs if j.id == rid), jobs[0])
    # APScheduler job exposes attributes for these settings in 3.x
    assert getattr(job, "max_instances", 1) > 1
    assert getattr(job, "misfire_grace_time", 0) == 123
    assert getattr(job, "coalesce", True) is False


def test_update_invalid_cron_returns_422(client_admin):
    client, _ = client_admin
    # Create a valid schedule
    body = {
        "name": "u1",
        "cron": "*/10 * * * *",
        "timezone": "UTC",
        "inputs": {},
        "run_mode": "async",
        "validation_mode": "block",
        "enabled": True,
    }
    sid = client.post("/api/v1/scheduler/workflows", json=body).json()["id"]
    # Attempt to update with invalid cron
    r = client.patch(f"/api/v1/scheduler/workflows/{sid}", json={"cron": "not a cron"})
    assert r.status_code == 422


def test_next_run_persisted_after_create(client_admin):
    client, _ = client_admin
    body = {
        "name": "p1",
        "cron": "*/7 * * * *",
        "timezone": "UTC",
        "inputs": {},
        "run_mode": "async",
        "validation_mode": "block",
        "enabled": True,
    }
    sid = client.post("/api/v1/scheduler/workflows", json=body).json()["id"]
    resp = client.get(f"/api/v1/scheduler/workflows/{sid}")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("next_run_at"), str) and len(data["next_run_at"]) > 0


def test_list_all_schedules_preserves_acp_config_json(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "scheduler.db"))
    db = WorkflowsSchedulerDB()
    acp_config = '{"prompt":"summarize"}'

    db.create_schedule(
        id="acp-schedule",
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="ACP schedule",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs={},
        acp_config_json=acp_config,
    )

    schedules = db.list_all_schedules(limit=10)

    assert len(schedules) == 1
    assert schedules[0].acp_config_json == acp_config


def test_build_schedule_payload_defaults_malformed_inputs_to_empty_dict():
    payload = workflows_scheduler_mod.build_schedule_payload(_schedule("bad-json", inputs_json="{not-json"))

    assert payload["inputs"] == {}
    assert payload["workflow_id"] is None


def test_list_registered_schedules_pages_until_short_page(monkeypatch):
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    first_page = [_schedule(f"sched-{idx}") for idx in range(1000)]
    second_page = [_schedule("sched-1000")]
    calls: list[tuple[int, int]] = []

    class _PagedDB:
        def list_all_schedules(self, **kwargs):
            calls.append((kwargs["limit"], kwargs["offset"]))
            if kwargs["offset"] == 0:
                return first_page
            if kwargs["offset"] == 1000:
                return second_page
            return []

    monkeypatch.setattr(svc, "_get_db", lambda uid: _PagedDB())

    schedules = svc._list_registered_schedules(1)

    assert len(schedules) == 1001
    assert calls == [(1000, 0), (1000, 1000)]


def test_get_tolerates_default_and_user_db_lookup_errors(monkeypatch, tmp_path):
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    target = object()

    class _BrokenDefaultDB:
        def get_schedule(self, _schedule_id: str):
            raise BackendDatabaseError("SQLite error: no such table: workflow_schedules")

    class _BrokenUserDB:
        def get_schedule(self, _schedule_id: str):
            raise BackendDatabaseError("SQLite error: no such table: workflow_schedules")

    class _GoodUserDB:
        def get_schedule(self, schedule_id: str):
            return target if schedule_id == "wf-1" else None

    (tmp_path / "101").mkdir()
    (tmp_path / "202").mkdir()
    (tmp_path / "not-a-user").mkdir()

    monkeypatch.setattr(svc, "_db", _BrokenDefaultDB())
    monkeypatch.setattr(
        workflows_scheduler_mod.DatabasePaths,
        "get_user_db_base_dir",
        lambda: tmp_path,
    )

    def _get_db(uid: int):
        if uid == 101:
            return _BrokenUserDB()
        if uid == 202:
            return _GoodUserDB()
        raise AssertionError(f"Unexpected user id: {uid}")

    monkeypatch.setattr(svc, "_get_db", _get_db)

    assert svc.get("wf-1") is target


@pytest.mark.asyncio
async def test_load_all_discovers_non_default_tenant_schedules(monkeypatch, tmp_path):
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    loaded: list[tuple[str, int | None]] = []
    schedule = WorkflowSchedule(
        id="sched-tenant-a",
        tenant_id="tenant-a",
        user_id="1",
        workflow_id=None,
        name="Tenant A schedule",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
        jitter_sec=0,
        acp_config_json=None,
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    class _StubDB:
        def list_schedules(self, **kwargs):
            raise AssertionError(f"tenant-filtered schedule lookup should not be used during scheduler bootstrap: {kwargs}")

        def list_all_schedules(self, **kwargs):
            return [schedule]

    (tmp_path / "1").mkdir()
    monkeypatch.setattr(
        workflows_scheduler_mod.DatabasePaths,
        "get_user_db_base_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(svc, "_get_db", lambda uid: _StubDB())
    monkeypatch.setattr(svc, "_add_job", lambda schedule_obj, user_id=None: loaded.append((schedule_obj.id, user_id)))

    await svc._load_all()  # type: ignore[attr-defined]

    assert loaded == [("sched-tenant-a", 1)]


@pytest.mark.asyncio
async def test_load_all_uses_schedule_owner_when_shared_backend_returns_duplicates(monkeypatch, tmp_path):
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    loaded: list[tuple[str, int | None]] = []
    schedule = WorkflowSchedule(
        id="sched-shared",
        tenant_id="default",
        user_id="77",
        workflow_id=None,
        name="Shared schedule",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
        jitter_sec=0,
        acp_config_json=None,
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    class _SharedDB:
        def list_all_schedules(self, **kwargs):
            return [schedule]

    (tmp_path / "1").mkdir()
    (tmp_path / "77").mkdir()
    monkeypatch.setattr(
        workflows_scheduler_mod.DatabasePaths,
        "get_user_db_base_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(svc, "_get_db", lambda uid: _SharedDB())
    monkeypatch.setattr(svc, "_add_job", lambda schedule_obj, user_id=None: loaded.append((schedule_obj.id, user_id)))

    await svc._load_all()  # type: ignore[attr-defined]

    assert loaded == [("sched-shared", 77)]


@pytest.mark.asyncio
async def test_history_updates_on_fire(monkeypatch):
    # Start service directly without TestClient overhead for this unit test
    svc = get_workflows_scheduler()
    await svc.start()
    # Create a schedule via DB helper to avoid HTTP
    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="hist",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
    )

    class _StubScheduler:
        async def submit(self, *args: Any, **kwargs: Any) -> str:
            return "task-1"

    # Monkeypatch core scheduler to avoid real submission
    svc._core_scheduler = _StubScheduler()  # type: ignore[attr-defined]

    # Fire the job manually
    await svc._run_schedule(sid)  # type: ignore[attr-defined]

    s = svc.get(sid)
    assert s is not None
    # last_run_at populated, last_status moved to queued
    assert isinstance(s.last_run_at, str) and len(s.last_run_at) > 0
    assert s.last_status in ("pending", "queued", "error", "running")

    await svc.stop()


@pytest.mark.asyncio
async def test_run_schedule_routes_watchlist_backed_schedules_to_watchlists_queue(monkeypatch):
    svc = get_workflows_scheduler()
    await svc.start()
    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="watchlist-fire",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs={"watchlist_job_id": 7},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
    )

    captured: dict[str, Any] = {}

    class _StubScheduler:
        async def submit(self, *args: Any, **kwargs: Any) -> str:
            captured["handler"] = kwargs.get("handler")
            captured["queue_name"] = kwargs.get("queue_name")
            captured["payload"] = kwargs.get("payload")
            return "task-watchlist-fire"

    svc._core_scheduler = _StubScheduler()  # type: ignore[attr-defined]

    await svc._run_schedule(sid)  # type: ignore[attr-defined]

    assert captured["handler"] == "watchlist_run"
    assert captured["queue_name"] == "watchlists"
    assert captured["payload"]["inputs"]["watchlist_job_id"] == 7

    await svc.stop()


@pytest.mark.asyncio
async def test_run_schedule_submits_with_resolved_owner_user_id(monkeypatch):
    monkeypatch.delenv("WORKFLOWS_MINT_VIRTUAL_KEYS", raising=False)
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    schedule = _schedule("owner-fallback", user_id="legacy-owner")
    captured: dict[str, Any] = {}

    class _StubDB:
        def get_schedule(self, schedule_id: str) -> WorkflowSchedule | None:
            return schedule if schedule_id == "owner-fallback" else None

        def set_history(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _StubScheduler:
        async def submit(self, *args: Any, **kwargs: Any) -> str:
            captured["payload"] = kwargs.get("payload")
            captured["metadata"] = kwargs.get("metadata")
            return "task-owner-fallback"

    monkeypatch.setattr(svc, "_get_db", lambda uid: _StubDB())
    svc._core_scheduler = _StubScheduler()  # type: ignore[attr-defined]

    await svc._run_schedule("owner-fallback", user_id=42)  # type: ignore[attr-defined]

    assert captured["payload"]["user_id"] == "42"
    assert captured["metadata"] == {"user_id": "42"}


def test_run_now_routes_watchlist_backed_schedules_to_watchlists_queue(client_admin, monkeypatch):
    client, svc = client_admin
    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="watchlist-now",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs={"watchlist_job_id": 42},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
    )

    captured: dict[str, Any] = {}

    class _StubScheduler:
        async def submit(self, handler: str, *args: Any, **kwargs: Any) -> str:
            captured["handler"] = handler
            captured["payload"] = kwargs.get("payload")
            captured["queue_name"] = kwargs.get("queue_name")
            captured["metadata"] = kwargs.get("metadata")
            return "task-watchlist-now"

    async def _get_global_scheduler_stub():
        return _StubScheduler()

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.scheduler_workflows.get_global_scheduler",
        _get_global_scheduler_stub,
    )

    response = client.post(f"/api/v1/scheduler/workflows/{sid}/run-now")

    assert response.status_code == 200, response.text
    assert response.json() == {"task_id": "task-watchlist-now"}
    assert captured["handler"] == "watchlist_run"
    assert captured["queue_name"] == "watchlists"
    assert captured["payload"]["inputs"]["watchlist_job_id"] == 42
    assert captured["metadata"] == {"user_id": "1"}


@pytest.mark.asyncio
async def test_start_workflows_scheduler_enabled_with_y(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_ENABLED", "y")
    calls = {"start": 0, "stop": 0}

    class _StubService:
        async def start(self) -> None:
            calls["start"] += 1

        async def stop(self) -> None:
            calls["stop"] += 1

    stub = _StubService()
    monkeypatch.setattr(workflows_scheduler_mod, "get_workflows_scheduler", lambda: stub)

    task = await workflows_scheduler_mod.start_workflows_scheduler()
    assert task is not None
    assert calls["start"] == 1

    await workflows_scheduler_mod.stop_workflows_scheduler(task)
    assert calls["stop"] == 1


@pytest.mark.asyncio
async def test_run_schedule_mints_virtual_key_when_enabled_with_y(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_MINT_VIRTUAL_KEYS", "y")
    svc = get_workflows_scheduler()
    await svc.start()
    sid = svc.create(
        tenant_id="default",
        user_id="1",
        workflow_id=None,
        name="vk-y",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=True,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
    )

    captured: dict[str, Any] = {}

    class _StubScheduler:
        async def submit(self, *args: Any, **kwargs: Any) -> str:
            captured["payload"] = kwargs.get("payload")
            return "task-vk"

    class _StubJWTService:
        def __init__(self, settings) -> None:  # noqa: ANN001
            self.settings = settings

        def create_virtual_access_token(self, **kwargs: Any) -> str:
            captured["vk_kwargs"] = kwargs
            return "vk-token-y"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.jwt_service.JWTService",
        _StubJWTService,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.settings.get_settings",
        lambda: object(),
    )
    svc._core_scheduler = _StubScheduler()  # type: ignore[attr-defined]

    await svc._run_schedule(sid)  # type: ignore[attr-defined]

    assert captured.get("vk_kwargs") is not None
    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("secrets", {}).get("jwt") == "vk-token-y"

    await svc.stop()


@pytest.mark.asyncio
async def test_run_acp_schedule_logs_malformed_acp_config_json(monkeypatch):
    svc = workflows_scheduler_mod._WFRecurringScheduler()
    schedule = WorkflowSchedule(
        id="bad-acp-json",
        tenant_id="default",
        user_id="5",
        workflow_id=None,
        name="Bad ACP JSON",
        cron="*/5 * * * *",
        timezone="UTC",
        inputs_json="{}",
        run_mode="async",
        validation_mode="block",
        enabled=True,
        require_online=False,
        concurrency_mode="skip",
        misfire_grace_sec=60,
        coalesce=True,
        jitter_sec=0,
        acp_config_json="{not-json",
        last_run_at=None,
        next_run_at=None,
        last_status=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    warnings: list[tuple[object, ...]] = []
    captured: dict[str, Any] = {}

    class _StubDB:
        def get_schedule(self, schedule_id: str) -> WorkflowSchedule | None:
            return schedule if schedule_id == "bad-acp-json" else None

        def set_history(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _StubScheduler:
        async def submit(self, *args: Any, **kwargs: Any) -> str:
            captured["payload"] = kwargs.get("payload")
            return "task-bad-acp-json"

    monkeypatch.setattr(svc, "_get_db", lambda uid: _StubDB())
    monkeypatch.setattr(
        workflows_scheduler_mod.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )
    svc._core_scheduler = _StubScheduler()  # type: ignore[attr-defined]

    await svc._run_acp_schedule("bad-acp-json", 5)  # type: ignore[attr-defined]

    assert captured["payload"]["prompt"] == ""
    assert any("malformed acp_config_json" in str(args[0]) for args in warnings)
