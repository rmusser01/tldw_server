from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock

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


class _CoreSchedulerStub:
    def __init__(self, *, task_id: str = "task-stub") -> None:
        self.task_id = task_id
        self.submissions: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.stop_calls = 0
        self.worker_pool = self
        self.leader_election = AsyncMock()
        self.leader_election.config.leader_ttl_seconds = 60

    def get_status(self) -> dict[str, Any]:
        return {"workers_by_queue": {"workflows": 1, "watchlists": 1}}

    async def submit(self, *args: Any, **kwargs: Any) -> str:
        self.submissions.append((args, kwargs))
        return self.task_id

    async def stop(self) -> None:
        self.stop_calls += 1


def _patch_core_scheduler(monkeypatch: pytest.MonkeyPatch, stub: _CoreSchedulerStub) -> None:
    async def _get_global_scheduler_stub() -> _CoreSchedulerStub:
        return stub

    monkeypatch.setattr(workflows_scheduler_mod, "get_global_scheduler", _get_global_scheduler_stub)


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
    _patch_core_scheduler(monkeypatch, _CoreSchedulerStub())
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



@pytest.mark.parametrize(
    "update",
    [
        {"timezone": "C12/Not_A_Zone"},
        {"timezone": "C12/Not_A_Zone", "name": "must-not-persist"},
        {"timezone": "C12/Not_A_Zone", "cron": "*/20 * * * *"},
    ],
)
def test_update_invalid_timezone_preserves_saved_schedule(client_admin, update):
    client, svc = client_admin
    created = client.post(
        "/api/v1/scheduler/workflows",
        json={
            "name": "timezone-update",
            "cron": "*/10 * * * *",
            "timezone": "UTC",
            "enabled": False,
        },
    )
    sid = created.json()["id"]
    before = svc.get(sid)
    response = client.patch(f"/api/v1/scheduler/workflows/{sid}", json=update)
    assert response.status_code == 422  # nosec B101 -- pytest request-validation oracle.
    assert svc.get(sid) == before  # nosec B101 -- rejected patches must preserve every saved field.


def test_update_valid_timezone_alone_preserves_cron(client_admin):
    client, svc = client_admin
    created = client.post(
        "/api/v1/scheduler/workflows",
        json={
            "name": "valid-timezone-update",
            "cron": "*/10 * * * *",
            "timezone": "UTC",
            "enabled": False,
        },
    )
    sid = created.json()["id"]
    response = client.patch(f"/api/v1/scheduler/workflows/{sid}", json={"timezone": "Europe/London"})
    saved = svc.get(sid)
    assert response.status_code == 200  # nosec B101 -- valid timezone-only edits remain supported.
    assert saved.timezone == "Europe/London" and saved.cron == "*/10 * * * *"  # nosec B101 -- unchanged authored cron retained.


def test_managed_schedule_definition_is_edited_in_watchlists(client_admin):
    client, scheduler = client_admin
    sid = scheduler.create(
        tenant_id="default", user_id="1", workflow_id=None,
        managed_watchlist_job_id=9, name="Managed monitor", cron="0 9 * * *",
        timezone="UTC", inputs={"watchlist_job_id": 9}, run_mode="async",
        validation_mode="block", enabled=True,
    )
    path = f"/api/v1/scheduler/workflows/{sid}"
    response = client.patch(path, json={"cron": "0 10 * * *"})
    assert response.status_code == 409
    assert response.json()["detail"] == "edit_managed_schedule_in_watchlists"
    assert client.patch(path, json={"name": "Renamed monitor"}).json() == {"ok": True}
    row = scheduler.get(sid)
    assert row is not None
    assert (row.cron, row.name, row.managed_watchlist_job_id) == ("0 9 * * *", "Renamed monitor", 9)


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


@pytest.mark.asyncio
async def test_persisted_schedule_dispatch_has_one_owner_and_can_fail_over(monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
    from tldw_Server_API.app.core.Scheduler.backends.sqlite_backend import SQLiteBackend
    from tldw_Server_API.app.core.Scheduler.config import SchedulerConfig
    from tldw_Server_API.app.core.Scheduler.core.leader_election import LeaderElection

    config = SchedulerConfig(database_url=f"sqlite:///{tmp_path / 'queue.db'}")
    backends = [SQLiteBackend(config), SQLiteBackend(config)]
    cores = [_CoreSchedulerStub(), _CoreSchedulerStub()]
    for core, backend in zip(cores, backends, strict=True):
        await backend.connect()
        core.leader_election = LeaderElection(backend, config)
    core_iter = iter(cores)

    async def get_core():
        return next(core_iter)

    monkeypatch.setattr(workflows_scheduler_mod, "get_global_scheduler", get_core)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "scheduler.db"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    first = workflows_scheduler_mod._WFRecurringScheduler()
    second = workflows_scheduler_mod._WFRecurringScheduler()

    watchlists = WatchlistsDatabase.for_user(1)
    job = watchlists.create_job(
        name="Feed monitor", description=None, scope_json="{}", schedule_expr="*/15 * * * *",
        schedule_timezone="UTC", active=True, max_concurrency=None, per_host_delay_ms=None,
        retry_policy_json=None, output_prefs_json=None,
    )
    schedule_id = first.create(
        tenant_id="default", user_id="1", workflow_id=None, name=f"watchlist:{job.id}:Feed monitor",
        cron=job.schedule_expr, timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True,
    )
    assert watchlists.set_job_schedule_id(job.id, schedule_id)

    try:
        await first.start()
        await second.start()
        pending = datetime.now(timezone.utc) + timedelta(minutes=3, seconds=17)
        first._aps.modify_job(schedule_id, next_run_time=pending)
        await first._rescan_once()
        assert first._aps.get_job(schedule_id).next_run_time == pending
        due = datetime.now(timezone.utc) + timedelta(milliseconds=100)
        first._aps.modify_job(schedule_id, next_run_time=due)
        job = watchlists.update_job(job.id, {"schedule_expr": "*/10 * * * *"})
        assert second.update(schedule_id, {"cron": "*/10 * * * *"})
        await asyncio.sleep(0.25)
        assert [len(core.submissions) for core in cores] == [0, 0]
        await first._rescan_once()
        assert "*/10" in str(first._aps.get_job(schedule_id).trigger)
        await second._rescan_once()
        due = datetime.now(timezone.utc) + timedelta(milliseconds=100)
        for service in (first, second):
            service._aps.modify_job(schedule_id, next_run_time=due)
        first._get_db(1).set_history(schedule_id, next_run_at=due.isoformat())
        await asyncio.sleep(0.25)
        assert [len(core.submissions) for core in cores] == [1, 0]
        _, submission = cores[0].submissions[0]
        assert submission["payload"]["watchlist_schedule_id"] == schedule_id
        assert submission["payload"]["watchlist_schedule_cron"] == "*/10 * * * *"
        assert submission["payload"]["watchlist_job_revision"] == job.updated_at
        await first._rescan_once()
        assert datetime.fromisoformat(first.get(schedule_id).next_run_at) > due

        await first.stop()
        assert await second._refresh_cron_owner()
        due = datetime.now(timezone.utc) + timedelta(milliseconds=80)
        second._aps.reschedule_job(schedule_id, trigger="date", run_date=due)
        await asyncio.sleep(0.2)
        assert [len(core.submissions) for core in cores] == [1, 1]
    finally:
        await first.stop()
        await second.stop()
        for backend in backends:
            await backend.disconnect()


def test_failed_schedule_delete_preserves_live_trigger(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "scheduler.db"))
    scheduler = workflows_scheduler_mod._WFRecurringScheduler()
    schedule_id = scheduler.create(
        tenant_id="default", user_id="1", workflow_id=None, name="Monitor",
        cron="*/15 * * * *", timezone="UTC", inputs={}, run_mode="async",
        validation_mode="block", enabled=True,
    )
    store = scheduler._get_db(1)
    live = {schedule_id}
    stored_when_removed = []

    def remove_job(target):
        stored_when_removed.append(store.get_schedule(target))
        live.remove(target)

    scheduler._aps = SimpleNamespace(remove_job=remove_job)

    def refuse_delete(target, *, user_id, managed_watchlist_job_id):
        assert (target, user_id, managed_watchlist_job_id) == (schedule_id, "1", None)
        raise OSError("store unavailable")

    with monkeypatch.context() as fault:
        fault.setattr(store, "delete_schedule", refuse_delete)
        with pytest.raises(OSError, match="store unavailable"):
            scheduler.delete(schedule_id, user_id=1)
    assert live == {schedule_id}
    assert store.get_schedule(schedule_id) is not None

    assert scheduler.delete(schedule_id, user_id=1)
    assert not live
    assert stored_when_removed == [None]


def test_shared_postgres_scheduler_reuses_the_database_and_one_scan(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

    config = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    db = SimpleNamespace(backend=SimpleNamespace(config=config), get_schedule=lambda _: None)
    monkeypatch.setattr(workflows_scheduler_mod, "WorkflowsSchedulerDB", lambda: db)
    scheduler = workflows_scheduler_mod._WFRecurringScheduler()
    assert scheduler._get_db(10) is scheduler._get_db(11) is db
    assert len(scheduler._schedule_scan_user_ids()) == 1
    assert scheduler.get("missing") is None


@pytest.mark.asyncio
async def test_scheduler_start_closes_retired_postgres_pool(pg_temp_db, monkeypatch):
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_DATABASE_URL", pg_temp_db["dsn"])
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", raising=False)
    _patch_core_scheduler(monkeypatch, _CoreSchedulerStub())

    scheduler = workflows_scheduler_mod._WFRecurringScheduler()
    retired_pool = scheduler._db.backend.get_pool()
    try:
        await scheduler.start()

        assert retired_pool.get_stats()["closed"]  # nosec B101
    finally:
        await scheduler.stop()
        retired_pool.close_all()
        scheduler._db.backend.get_pool().close_all()


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
    core_scheduler = _CoreSchedulerStub(task_id="task-1")
    _patch_core_scheduler(monkeypatch, core_scheduler)
    svc = get_workflows_scheduler()
    await svc.start()
    try:
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

        # Fire the job manually
        await svc._run_schedule(sid)  # type: ignore[attr-defined]

        s = svc.get(sid)
        assert s is not None
        # last_run_at populated, last_status moved to queued
        assert isinstance(s.last_run_at, str) and len(s.last_run_at) > 0
        assert s.last_status in ("pending", "queued", "error", "running")
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_run_schedule_routes_watchlist_backed_schedules_to_watchlists_queue(monkeypatch):
    core_scheduler = _CoreSchedulerStub(task_id="task-watchlist-fire")
    _patch_core_scheduler(monkeypatch, core_scheduler)
    svc = get_workflows_scheduler()
    await svc.start()
    try:
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

        await svc._run_schedule(sid)  # type: ignore[attr-defined]

        _args, kwargs = core_scheduler.submissions[-1]
        assert kwargs.get("handler") == "watchlist_run"
        assert kwargs.get("queue_name") == "watchlists"
        assert kwargs.get("payload", {})["inputs"]["watchlist_job_id"] == 7
    finally:
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
        def __init__(self) -> None:
            self.worker_pool = self

        def get_status(self) -> dict[str, Any]:
            return {"workers_by_queue": {"workflows": 1}}

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
        def __init__(self) -> None:
            self.worker_pool = self

        def get_status(self) -> dict[str, Any]:
            return {"workers_by_queue": {"watchlists": 1}}

        async def submit(self, handler: str, *args: Any, **kwargs: Any) -> str:
            captured["handler"] = handler
            captured["payload"] = kwargs.get("payload")
            captured["queue_name"] = kwargs.get("queue_name")
            captured["metadata"] = kwargs.get("metadata")
            return "task-watchlist-now"

    async def _get_global_scheduler_stub():
        return _StubScheduler()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Scheduler.get_global_scheduler",
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
    core_scheduler = _CoreSchedulerStub(task_id="task-vk")
    _patch_core_scheduler(monkeypatch, core_scheduler)
    svc = get_workflows_scheduler()
    await svc.start()
    try:
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

        await svc._run_schedule(sid)  # type: ignore[attr-defined]

        assert captured.get("vk_kwargs") is not None
        _args, kwargs = core_scheduler.submissions[-1]
        payload = kwargs.get("payload")
        assert isinstance(payload, dict)
        assert payload.get("secrets", {}).get("jwt") == "vk-token-y"
    finally:
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
