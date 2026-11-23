from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_user(monkeypatch):
    async def override_user():
        return User(id=9201, username="scheduler-user", email=None, is_active=True)

    base_dir = Path.cwd() / "Databases" / "test_user_dbs_scheduler_controls"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_ENABLED", "false")

    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_job_scheduler_controls_roundtrip(client_user):
    r = client_user.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Scheduler Controls",
            "scope": {"tags": ["scheduler-controls"]},
            "schedule_expr": "*/15 * * * *",
            "timezone": "UTC",
            "active": True,
            "max_concurrency": 3,
            "per_host_delay_ms": 250,
        },
    )
    assert r.status_code == 200, r.text
    created = r.json()
    job_id = int(created["id"])
    assert created["max_concurrency"] == 3
    assert created["per_host_delay_ms"] == 250
    assert created["next_run_at"] is not None

    r = client_user.get(f"/api/v1/watchlists/jobs/{job_id}")
    assert r.status_code == 200, r.text
    fetched = r.json()
    assert fetched["max_concurrency"] == 3
    assert fetched["per_host_delay_ms"] == 250

    r = client_user.get("/api/v1/watchlists/jobs")
    assert r.status_code == 200, r.text
    rows = [row for row in r.json().get("items", []) if int(row["id"]) == job_id]
    assert len(rows) == 1
    assert rows[0]["max_concurrency"] == 3
    assert rows[0]["per_host_delay_ms"] == 250


def test_update_scheduler_controls_and_run_timestamps(client_user):
    r = client_user.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Job Update",
            "scope": {"tags": ["scheduler-update"]},
            "schedule_expr": "*/30 * * * *",
            "timezone": "UTC",
            "active": True,
            "max_concurrency": 1,
            "per_host_delay_ms": 50,
        },
    )
    assert r.status_code == 200, r.text
    job_id = int(r.json()["id"])

    r = client_user.patch(
        f"/api/v1/watchlists/jobs/{job_id}",
        json={
            "schedule_expr": "*/10 * * * *",
            "timezone": "UTC",
            "max_concurrency": 5,
            "per_host_delay_ms": 500,
            "active": False,
        },
    )
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["max_concurrency"] == 5
    assert updated["per_host_delay_ms"] == 500
    assert updated["active"] is False
    assert updated["next_run_at"] is not None

    # run-now should refresh history timestamps even after scheduler control updates
    r = client_user.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text

    r = client_user.get(f"/api/v1/watchlists/jobs/{job_id}")
    assert r.status_code == 200, r.text
    fetched = r.json()
    assert fetched["last_run_at"] is not None
    assert fetched["next_run_at"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule", [None, ""], ids=["null", "blank"])
async def test_clear_schedule_removes_persisted_cron_and_preserves_manual_runs(tmp_path, monkeypatch, schedule):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import JobCreateRequest, JobUpdateRequest
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
    from tldw_Server_API.app.services import workflows_scheduler

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "scheduler.db"))
    user = User(id=9201, username="scheduler-user", email=None, is_active=True)
    db = WatchlistsDatabase(user.id)
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    created = await watchlists.create_job(
        JobCreateRequest(name="Feed monitor", scope={"tags": ["news"]}, schedule_expr="0 8 * * *", timezone="UTC"),
        current_user=user,
        db=db,
    )
    schedule_id = db.get_job(created.id).wf_schedule_id
    assert schedule_id and scheduler.get(schedule_id) and created.next_run_at
    assert not scheduler.delete(schedule_id, user_id=user.id + 1)
    assert scheduler.get(schedule_id) is not None

    retained = await watchlists.update_job(created.id, JobUpdateRequest(name="Renamed"), current_user=user, db=db)
    assert retained.schedule_expr == created.schedule_expr
    assert retained.next_run_at == created.next_run_at

    payload = JobUpdateRequest(schedule_expr=schedule)
    with monkeypatch.context() as fault:
        def refuse_delete(*args, **kwargs):
            raise OSError("scheduler store unavailable")

        fault.setattr(scheduler, "delete", refuse_delete)
        with pytest.raises(HTTPException) as failure:
            await watchlists.update_job(created.id, payload, current_user=user, db=db)
        with pytest.raises(HTTPException) as blocked_edit:
            await watchlists.update_job(
                created.id, JobUpdateRequest(schedule_expr="0 9 * * *"), current_user=user, db=db,
            )
        assert (blocked_edit.value.status_code, blocked_edit.value.detail) == (503, "job_schedule_cleanup_pending")
    assert failure.value.status_code == 503
    assert failure.value.detail == "job_schedule_cleanup_pending"
    pending = db.get_job(created.id)
    assert (pending.schedule_expr, pending.next_run_at, pending.wf_schedule_id) == (None, None, schedule_id)
    assert scheduler.get(schedule_id) is not None

    cleared = await watchlists.update_job(created.id, payload, current_user=user, db=db)
    assert (cleared.schedule_expr, cleared.next_run_at, cleared.active) == (None, None, True)
    assert db.get_job(created.id).wf_schedule_id is None
    assert workflows_scheduler._WFRecurringScheduler().get(schedule_id) is None

    async def run_now(user_id, job_id, **kwargs):
        assert (user_id, job_id) == (user.id, created.id)
        return {"run_id": db.create_run(job_id).id}

    monkeypatch.setattr(watchlists._pipeline, "run_watchlist_job", run_now)
    manual = await watchlists.trigger_run(created.id, current_user=user, db=db)
    assert manual.job_id == created.id

    await watchlists.update_job(
        created.id, JobUpdateRequest(schedule_expr="0 9 * * *"), current_user=user, db=db,
    )
    new_id = db.get_job(created.id).wf_schedule_id
    assert new_id and new_id != schedule_id and scheduler.get(new_id).cron == "0 9 * * *"
    assert scheduler.delete(new_id, user_id=user.id)
    repaired = await watchlists.update_job(
        created.id, JobUpdateRequest(schedule_expr="0 10 * * *"), current_user=user, db=db,
    )
    replacement_id = db.get_job(created.id).wf_schedule_id
    assert replacement_id and replacement_id != new_id
    assert repaired.schedule_expr == scheduler.get(replacement_id).cron == "0 10 * * *"


@pytest.fixture()
def scheduled_collections_monitor(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = WatchlistsDatabase(9201)
    job = db.create_job(
        name="Collections feed", description=None, scope_json="{}", schedule_expr="0 * * * *",
        schedule_timezone="UTC", active=True, max_concurrency=None, per_host_delay_ms=None,
        retry_policy_json=None, output_prefs_json=None,
    )
    return db, job


def _link_legacy_collections_job(db, job, source):
    import json

    return db.update_job(job.id, {
        "scope_json": json.dumps({"sources": [source.id]}),
        "output_prefs_json": json.dumps({"collections_origin": "feed"}),
    })


def _bind_collections_job(db, job, source):
    import json

    db.backend.execute(
        "UPDATE scrape_jobs SET collections_feed_binding_json = ? WHERE id = ? AND user_id = ?",
        (json.dumps([source.id, source.created_at]), job.id, db.user_id),
    )
    return _link_legacy_collections_job(db, job, source)


def test_stale_run_history_does_not_restore_cleared_next_run(scheduled_collections_monitor):
    db, job = scheduled_collections_monitor
    db.update_job(job.id, {"schedule_expr": None})
    completed = "2026-11-01T08:00:00+00:00"
    db.set_job_history(job.id, last_run_at=completed, next_run_at="2026-11-01T09:00:00+00:00")
    row = db.get_job(job.id)
    assert row.last_run_at == completed
    assert row.next_run_at is None


def test_stale_run_history_preserves_replacement_next_run(scheduled_collections_monitor):
    db, initial = scheduled_collections_monitor
    db.set_job_schedule_id(initial.id, "old")
    old = db.get_job(initial.id)
    db.update_job(old.id, {"schedule_expr": None})
    db.update_job(old.id, {"schedule_expr": old.schedule_expr})
    db.set_job_schedule_id(old.id, "replacement")
    replacement_next = "2026-11-01T12:00:00+00:00"
    db.set_job_history(old.id, next_run_at=replacement_next)

    completed = "2026-11-01T08:00:00+00:00"
    db.set_job_history(
        old.id, last_run_at=completed, next_run_at="2026-11-01T09:00:00+00:00", expected_job=old,
    )
    row = db.get_job(old.id)
    assert row.last_run_at == completed
    assert row.next_run_at == replacement_next


@pytest.mark.parametrize("scenario", ["active", "in-flight", "manual"])
def test_feed_promotion_requires_current_schedule(scheduled_collections_monitor, scenario):
    from tldw_Server_API.app.core.Watchlists.pipeline import _maybe_promote_feed_schedule

    db, job = scheduled_collections_monitor
    observed_job = db.update_job(job.id, {"schedule_expr": None}) if scenario == "manual" else job
    if scenario == "in-flight":
        db.update_job(job.id, {"schedule_expr": None})
    prefs = {"collections_schedule": {
        "mode": "hourly_then_daily", "created_at": "2000-01-01T00:00:00+00:00",
        "daily_expr": "0 9 * * *",
    }}
    _maybe_promote_feed_schedule(db=db, job=observed_job, job_output_prefs=prefs)
    row = db.get_job(job.id)
    if scenario == "active":
        assert row.schedule_expr == "0 9 * * *"
        assert row.next_run_at is not None
        assert prefs["collections_schedule"]["promoted"] is True
        return
    assert (row.schedule_expr, row.next_run_at, row.output_prefs_json) == (None, None, None)
    assert "promoted" not in prefs["collections_schedule"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["concurrent-link", "watchlist-store"])
async def test_schedule_clear_does_not_delete_before_winning_job_update(
    scheduled_collections_monitor, monkeypatch, tmp_path, failure,
):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import JobUpdateRequest
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "scheduler.db"))
    user = User(id=9201, username="scheduler-user", email=None, is_active=True)
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)

    def new_cron():
        return scheduler.create(
            tenant_id="default", user_id=str(user.id), workflow_id=None, name="Collections feed",
            cron=job.schedule_expr, timezone="UTC", inputs={"watchlist_job_id": job.id},
            run_mode="async", validation_mode="block", enabled=True,
        )

    schedule_id = new_cron()
    db.set_job_schedule_id(job.id, schedule_id)
    persisted = db.update_job
    winning_id = schedule_id

    def race_or_fail(job_id, patch, **kwargs):
        nonlocal winning_id
        if failure == "watchlist-store":
            raise OSError("watchlist database unavailable")
        winning_id = new_cron()
        db.set_job_schedule_id(job_id, winning_id)
        return persisted(job_id, patch, **kwargs)

    monkeypatch.setattr(db, "update_job", race_or_fail)
    with pytest.raises(HTTPException) as error:
        await watchlists.update_job(job.id, JobUpdateRequest(schedule_expr=None), current_user=user, db=db)
    assert error.value.status_code == (409 if failure == "concurrent-link" else 503)
    row = db.get_job(job.id)
    assert (row.schedule_expr, row.wf_schedule_id) == (job.schedule_expr, winning_id)
    assert scheduler.get(winning_id) is not None
    assert scheduler.get(schedule_id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["current", "cleared", "replaced", "edited-cron"])
async def test_queued_watchlist_cron_checks_current_attachment(scheduled_collections_monitor, monkeypatch, scenario):
    from tldw_Server_API.app.core.Scheduler.handlers import watchlists as watchlist_tasks

    db, job = scheduled_collections_monitor
    db.set_job_schedule_id(job.id, "queued")
    if scenario == "cleared":
        db.update_job(job.id, {"schedule_expr": None})
    elif scenario == "replaced":
        db.set_job_schedule_id(job.id, "replacement")
    elif scenario == "edited-cron":
        db.update_job(job.id, {"schedule_expr": "0 9 * * *"})

    class ReachedRunStart(Exception):
        pass

    def stop_at_run_start(*args, **kwargs):
        raise ReachedRunStart

    monkeypatch.setattr(type(db), "create_scheduled_run", stop_at_run_start)
    task = {
        "user_id": job.user_id, "tenant_id": "default", "inputs": {"watchlist_job_id": job.id},
        "watchlist_schedule_id": "queued", "watchlist_schedule_cron": job.schedule_expr,
    }
    if scenario == "current":
        with pytest.raises(ReachedRunStart):
            await watchlist_tasks.watchlist_run(task)
    else:
        result = await watchlist_tasks.watchlist_run(task)
        assert (result["status"], result["run_id"]) == ("skipped", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("write", ["clear", "pause-resume"])
async def test_managed_run_insert_rejects_a_write_after_job_read(
    scheduled_collections_monitor, monkeypatch, write,
):
    from tldw_Server_API.app.core.Scheduler.handlers import watchlists as watchlist_tasks
    from tldw_Server_API.app.core.Watchlists import pipeline

    db, job = scheduled_collections_monitor
    db.set_job_schedule_id(job.id, "scheduled")
    observed_get_job = type(db).get_job
    changed = False

    def get_job_then_edit(self, job_id, *, connection=None):
        nonlocal changed
        observed = observed_get_job(self, job_id, connection=connection)
        if not changed:
            changed = True
            if write == "clear":
                db.update_job(job.id, {"schedule_expr": None})
            else:
                db.update_job(job.id, {"active": False})
                db.update_job(job.id, {"active": True})
        return observed

    def reject_manual_insert(*args, **kwargs):
        pytest.fail("a scheduled task cannot create a manual run")

    monkeypatch.setattr(type(db), "get_job", get_job_then_edit)
    monkeypatch.setattr(type(db), "create_run", reject_manual_insert)
    monkeypatch.setattr(pipeline.CollectionsDatabase, "for_user", lambda *_: object())
    result = await watchlist_tasks.watchlist_run({
        "user_id": job.user_id, "tenant_id": "default", "inputs": {"watchlist_job_id": job.id},
        "watchlist_schedule_id": "scheduled", "watchlist_schedule_cron": job.schedule_expr,
    })
    assert changed
    assert (result["status"], result["run_id"]) == ("skipped", None)
    assert db.list_runs_for_job(job.id, limit=1, offset=0) == ([], 0)


def test_managed_run_insert_accepts_current_revision(scheduled_collections_monitor):
    db, job = scheduled_collections_monitor
    assert db.set_job_schedule_id(job.id, "scheduled")
    admitted = db.create_scheduled_run(db.get_job(job.id))
    assert admitted is not None
    assert (admitted.job_id, admitted.status) == (job.id, "running")
    assert db.list_runs_for_job(job.id, limit=1, offset=0) == ([admitted], 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_first", [False, True], ids=["edit", "re-enable"])
async def test_explicit_cron_disables_collections_auto_promotion(
    scheduled_collections_monitor, monkeypatch, tmp_path, clear_first,
):
    import json

    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import JobUpdateRequest
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "promotion.db"))
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", workflows_scheduler._WFRecurringScheduler())
    db.update_job(job.id, {"output_prefs_json": json.dumps({
        "collections_schedule": {
            "mode": "hourly_then_daily", "created_at": "2000-01-01T00:00:00+00:00",
            "daily_expr": "0 9 * * *",
        },
    })})
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)

    async def patch_cron(expr):
        return await watchlists.update_job(job.id, JobUpdateRequest(schedule_expr=expr), current_user=user, db=db)

    unchanged = await patch_cron(job.schedule_expr)
    assert unchanged.output_prefs["collections_schedule"]["mode"] == "hourly_then_daily"
    if clear_first:
        await patch_cron(None)
    edited = await patch_cron("45 8 * * *")
    assert edited.output_prefs["collections_schedule"]["mode"] == "manual"
    persisted = db.get_job(job.id)
    prefs = json.loads(persisted.output_prefs_json)
    watchlists._pipeline._maybe_promote_feed_schedule(db=db, job=persisted, job_output_prefs=prefs)
    assert db.get_job(job.id).schedule_expr == "45 8 * * *"


@pytest.mark.asyncio
@pytest.mark.parametrize("cleared", [False, True], ids=["tag-current", "skip-cleared"])
async def test_renamed_managed_cron_remains_fenced(
    scheduled_collections_monitor, monkeypatch, tmp_path, cleared,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "renamed-schedule.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    schedule_id = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None, name=f"watchlist:{job.id}:Monitor",
        cron=job.schedule_expr, timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True,
    )
    assert db.set_job_schedule_id(job.id, schedule_id)
    assert scheduler.update(schedule_id, {"name": "Renamed monitor"})
    schedule = scheduler.get(schedule_id)
    assert schedule is not None
    if cleared:
        # Simulate a Watchlist clear committed while the scheduler store cleanup is pending.
        db.update_job(job.id, {"schedule_expr": None})

    submit = AsyncMock(return_value="test-task")
    scheduler._core_scheduler = SimpleNamespace(submit=submit)
    monkeypatch.setattr(scheduler, "_may_submit_cron", AsyncMock(return_value=True))
    monkeypatch.setattr(workflows_scheduler, "ensure_schedule_queue_worker", AsyncMock())
    await scheduler._run_schedule(
        schedule_id, int(job.user_id), expected_cron=scheduler._cron_definition(schedule),
    )
    if cleared:
        submit.assert_not_awaited()
    else:
        submit.assert_awaited_once()
        payload = submit.await_args.kwargs["payload"]
        assert (payload["watchlist_schedule_id"], payload["watchlist_schedule_cron"]) == (schedule_id, job.schedule_expr)


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", ["timezone", "pause-resume"])
async def test_queued_managed_cron_cannot_start_after_its_definition_changes(
    scheduled_collections_monitor, monkeypatch, tmp_path, edit,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Scheduler.handlers import watchlists as watchlist_tasks
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "queued-schedule.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    schedule_id = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None, name=f"watchlist:{job.id}:Monitor",
        cron=job.schedule_expr, timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True,
    )
    assert db.set_job_schedule_id(job.id, schedule_id)
    schedule = scheduler.get(schedule_id)
    assert schedule is not None
    submit = AsyncMock(return_value="test-task")
    scheduler._core_scheduler = SimpleNamespace(submit=submit)
    monkeypatch.setattr(scheduler, "_may_submit_cron", AsyncMock(return_value=True))
    monkeypatch.setattr(workflows_scheduler, "ensure_schedule_queue_worker", AsyncMock())
    await scheduler._run_schedule(
        schedule_id, int(job.user_id), expected_cron=scheduler._cron_definition(schedule),
    )
    submit.assert_awaited_once()
    payload = submit.await_args.kwargs["payload"]

    if edit == "timezone":
        db.update_job(job.id, {"schedule_timezone": "America/Los_Angeles"})
    else:
        db.update_job(job.id, {"active": False})
        db.update_job(job.id, {"active": True})

    def stop_at_run_start(*args, **kwargs):
        pytest.fail("a retired occurrence reached run creation")

    monkeypatch.setattr(type(db), "create_scheduled_run", stop_at_run_start)
    monkeypatch.setattr(type(db), "create_run", stop_at_run_start)
    result = await watchlist_tasks.watchlist_run(payload)
    assert (result["status"], result["run_id"]) == ("skipped", None)


@pytest.mark.asyncio
async def test_collections_replacement_preserves_link_until_retired_cron_is_deleted(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    import json

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds
    from tldw_Server_API.app.api.v1.schemas.collections_feeds_schemas import CollectionsFeedUpdateRequest
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    source = db.create_source(
        name="Feed monitor", url="https://example.invalid/feed.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
    )
    job = _bind_collections_job(db, job, source)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "collections-schedule.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    monkeypatch.setattr(collections_feeds._Personalization, "record_watchlist_source_updated", lambda **_: None)
    schedule_id = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None, name=f"watchlist:{job.id}:Feed",
        cron=job.schedule_expr, timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True,
    )
    assert db.set_job_schedule_id(job.id, schedule_id)
    # This is the state after clear commits but scheduler cleanup fails.
    db.update_job(job.id, {"schedule_expr": None})

    def delete_unavailable(*_, **__):
        raise OSError("scheduler store unavailable")

    patch = CollectionsFeedUpdateRequest(name="Replacement feed", schedule_expr="30 9 * * *")
    with monkeypatch.context() as failed:
        failed.setattr(scheduler, "delete", delete_unavailable)
        with pytest.raises(HTTPException) as exc:
            await collections_feeds.update_feed_subscription(source.id, patch, current_user=user, db=db)
    assert exc.value.status_code == 503
    assert db.get_job(job.id).wf_schedule_id == schedule_id
    assert db.get_source(source.id).name == "Feed monitor"
    assert scheduler.get(schedule_id) is not None

    response = await collections_feeds.update_feed_subscription(source.id, patch, current_user=user, db=db)
    current = db.get_job(job.id)
    assert response.schedule_expr == current.schedule_expr == "30 9 * * *"
    assert current.wf_schedule_id and current.wf_schedule_id != schedule_id
    assert scheduler.get(schedule_id) is None
    assert scheduler.get(current.wf_schedule_id).cron == current.schedule_expr


@pytest.mark.asyncio
async def test_watchlist_schedule_read_error_is_not_a_missing_link(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import JobUpdateRequest
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "unavailable-schedule.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    schedule_id = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None, name=f"watchlist:{job.id}:Feed",
        cron=job.schedule_expr, timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True,
    )
    assert db.set_job_schedule_id(job.id, schedule_id)
    schedule_type = type(scheduler._get_db(int(job.user_id)))
    get_schedule = schedule_type.get_schedule

    def fail_read(self, key):
        if key == schedule_id:
            raise OSError("scheduler store unavailable")
        return get_schedule(self, key)

    patch = JobUpdateRequest(schedule_expr="30 9 * * *")
    with monkeypatch.context() as failed:
        failed.setattr(schedule_type, "get_schedule", fail_read)
        await watchlists.update_job(job.id, patch, current_user=user, db=db)
    assert db.get_job(job.id).wf_schedule_id == schedule_id
    assert scheduler.get(schedule_id).cron == job.schedule_expr

    await watchlists.update_job(job.id, patch, current_user=user, db=db)
    replacement_id = db.get_job(job.id).wf_schedule_id
    assert replacement_id and replacement_id != schedule_id
    assert scheduler.get(schedule_id) is None
    assert scheduler.get(replacement_id).cron == patch.schedule_expr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "producer", ["create", "restore", "collections", "collections-fallback", "collections-post-persist"],
)
async def test_a_stale_schedule_registration_cannot_replace_the_winning_link(
    scheduled_collections_monitor, monkeypatch, tmp_path, producer,
):
    from tldw_Server_API.app.api.v1.endpoints import collections_feeds, watchlists
    from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import JobCreateRequest
    from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import WorkflowsSchedulerDB
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "attachment-race.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    real_persist = WorkflowsSchedulerDB.create_schedule
    attempts = []
    losing = winning = None
    target_id = job.id

    def persist_then_replace(self, **kwargs):
        nonlocal losing, winning, target_id
        attempts.append(kwargs["id"])
        real_persist(self, **kwargs)
        losing = kwargs["id"]
        target_id = kwargs["inputs"]["watchlist_job_id"]
        latest = db.update_job(target_id, {"schedule_expr": "15 9 * * *"})
        winning = "winning-schedule"
        real_persist(self, **{**kwargs, "id": winning, "cron": latest.schedule_expr})
        assert db.set_job_schedule_id(target_id, winning, expected_job=latest)

    monkeypatch.setattr(WorkflowsSchedulerDB, "create_schedule", persist_then_replace)
    if producer == "collections-fallback":
        def unavailable(**_):
            raise OSError("live scheduler unavailable")

        monkeypatch.setattr(scheduler, "create", unavailable)
    elif producer == "collections-post-persist":
        real_create = scheduler.create

        def fail_after_write(**kwargs):
            real_create(**kwargs)
            raise OSError("live scheduler unavailable after persistence")

        monkeypatch.setattr(scheduler, "create", fail_after_write)
    if producer == "create":
        await watchlists.create_job(
            JobCreateRequest(name="Monitor", scope={"tags": ["news"]}, schedule_expr=job.schedule_expr, timezone="UTC"),
            current_user=user, db=db,
        )
    elif producer == "restore":
        assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]
        await watchlists.restore_job(job.id, current_user=user, db=db)
    else:
        collections_feeds._register_schedule(db, job, current_user=user)

    assert losing is not None and winning is not None
    assert attempts == [losing]
    current = db.get_job(target_id)
    assert (current.wf_schedule_id, current.schedule_expr) == (winning, "15 9 * * *")
    assert scheduler.get(winning).cron == current.schedule_expr
    assert scheduler.get(losing) is None


def test_fallback_attachment_only_retires_the_owners_schedule(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "shared-schedules.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    foreign = scheduler.create(
        tenant_id="default", user_id=str(int(job.user_id) + 1), workflow_id=123,
        name="Other owner's schedule", cron="0 9 * * *", timezone="UTC", inputs={},
        run_mode="async", validation_mode="block", enabled=True,
    )
    assert db.set_job_schedule_id(job.id, foreign)

    def unavailable(**_):
        raise OSError("live registration unavailable")

    monkeypatch.setattr(scheduler, "create", unavailable)
    workflows_scheduler.register_watchlist_schedule(
        db, db.get_job(job.id), tenant_id="default", timezone="UTC",
    )

    current = db.get_job(job.id)
    assert current.wf_schedule_id != foreign
    assert scheduler.get(current.wf_schedule_id).user_id == str(job.user_id)
    untouched = scheduler.get(foreign)
    assert untouched is not None
    assert untouched.user_id == str(int(job.user_id) + 1)


@pytest.mark.asyncio
async def test_delayed_mirror_write_cannot_supersede_the_newest_schedule(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    import asyncio
    from threading import Event

    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import JobUpdateRequest
    from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import WorkflowsSchedulerDB
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "mirror-race.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    workflows_scheduler.register_watchlist_schedule(db, job, tenant_id="default", timezone="UTC")
    assert db.get_job(job.id).wf_schedule_id is not None

    first, latest = "15 8 * * *", "45 9 * * *"
    paused_before_mirror = Event()
    release_mirror = Event()
    original_update = WorkflowsSchedulerDB.update_schedule
    original_create = WorkflowsSchedulerDB.create_schedule

    def wait_for_latest(cron):
        if cron == first:
            paused_before_mirror.set()
            assert release_mirror.wait(timeout=5)

    def delay_update(self, schedule_id, patch):
        wait_for_latest(patch.get("cron"))
        return original_update(self, schedule_id, patch)

    def delay_create(self, **kwargs):
        wait_for_latest(kwargs["cron"])
        return original_create(self, **kwargs)

    monkeypatch.setattr(WorkflowsSchedulerDB, "update_schedule", delay_update)
    monkeypatch.setattr(WorkflowsSchedulerDB, "create_schedule", delay_create)

    async def change(cron):
        return await watchlists.update_job(job.id, JobUpdateRequest(schedule_expr=cron), current_user=user, db=db)

    def change_first():
        return asyncio.run(change(first))

    pending = asyncio.create_task(asyncio.to_thread(change_first))
    try:
        assert await asyncio.to_thread(paused_before_mirror.wait, 5)
        await change(latest)
    finally:
        release_mirror.set()
        await pending

    current = db.get_job(job.id)
    assert current.schedule_expr == latest
    assert scheduler.get(current.wf_schedule_id).cron == latest


@pytest.mark.asyncio
async def test_renamed_detached_managed_schedule_cannot_fall_back_to_a_manual_run(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "detached-schedule.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    workflows_scheduler.register_watchlist_schedule(db, job, tenant_id="default", timezone="UTC")
    linked = db.get_job(job.id)
    assert linked.wf_schedule_id is not None
    assert scheduler.update(linked.wf_schedule_id, {"name": "Renamed monitor"})
    assert db.set_job_schedule_id(job.id, None, expected_job=linked)
    old = scheduler.get(linked.wf_schedule_id)
    assert old is not None

    submit = AsyncMock()
    scheduler._core_scheduler = SimpleNamespace(submit=submit)
    monkeypatch.setattr(scheduler, "_may_submit_cron", AsyncMock(return_value=True))
    monkeypatch.setattr(workflows_scheduler, "ensure_schedule_queue_worker", AsyncMock())
    await scheduler._run_schedule(old.id, int(job.user_id), expected_cron=scheduler._cron_definition(old))
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_collections_job_conflict_leaves_the_feed_unchanged(scheduled_collections_monitor, monkeypatch):
    import json

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds
    from tldw_Server_API.app.api.v1.schemas.collections_feeds_schemas import CollectionsFeedUpdateRequest
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistJobChangedError

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    source = db.create_source(
        name="Original feed", url="https://example.invalid/feed.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
        tags=["original"],
    )
    job = _bind_collections_job(db, job, source)

    def stale(*args, **kwargs):
        raise WatchlistJobChangedError("schedule version changed")

    monkeypatch.setattr(db, "update_job", stale)
    with pytest.raises(HTTPException) as exc:
        await collections_feeds.update_feed_subscription(
            source.id,
            CollectionsFeedUpdateRequest(name="Rejected edit", tags=["changed"], schedule_expr="30 9 * * *"),
            current_user=user, db=db,
        )
    assert exc.value.status_code == 409
    current = db.get_source(source.id)
    assert current.name == source.name
    assert current.tags == source.tags


@pytest.mark.asyncio
async def test_collections_combined_edit_rolls_back_when_tag_write_fails(scheduled_collections_monitor, monkeypatch):
    import json

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds
    from tldw_Server_API.app.api.v1.schemas.collections_feeds_schemas import CollectionsFeedUpdateRequest

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    source = db.create_source(
        name="Original feed", url="https://example.invalid/combined.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
        tags=["original"],
    )
    source = db.get_source(source.id)
    job = _bind_collections_job(db, job, source)
    sync_calls = []
    monkeypatch.setattr(collections_feeds, "_sync_job_schedule", lambda *_args, **_kwargs: sync_calls.append(True))
    monkeypatch.setattr(collections_feeds._Personalization, "record_watchlist_source_updated", lambda **_: None)
    set_tags = db.set_source_tags

    def fail_after_write(*args, **kwargs):
        set_tags(*args, **kwargs)
        raise OSError("tag store failed")

    patch = CollectionsFeedUpdateRequest(name="Updated feed", tags=["changed"], schedule_expr="30 9 * * *")
    with monkeypatch.context() as failed:
        failed.setattr(db, "set_source_tags", fail_after_write)
        with pytest.raises(HTTPException) as exc:
            await collections_feeds.update_feed_subscription(source.id, patch, current_user=user, db=db)
    assert exc.value.status_code == 400
    assert db.get_source(source.id) == source
    assert db.get_job(job.id) == job
    assert sync_calls == []

    result = await collections_feeds.update_feed_subscription(source.id, patch, current_user=user, db=db)
    assert result.name == "Updated feed"
    assert result.tags == ["changed"]
    assert result.schedule_expr == db.get_job(job.id).schedule_expr == "30 9 * * *"
    assert sync_calls == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["delete", "acknowledge"])
async def test_schedule_retry_retires_an_older_generation_after_cleanup_error(
    scheduled_collections_monitor, monkeypatch, tmp_path, failure,
):
    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.api.v1.schemas.watchlists_schemas import JobUpdateRequest
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "cleanup-retry.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    workflows_scheduler.register_watchlist_schedule(db, job, tenant_id="default", timezone="UTC")
    retired_id = db.get_job(job.id).wf_schedule_id
    assert retired_id is not None
    delete = scheduler.delete

    def unavailable(schedule_id, **kwargs):
        if schedule_id == retired_id:
            raise OSError("schedule store unavailable during cleanup")
        return delete(schedule_id, **kwargs)

    def unacknowledged(job_id, schedule_id):
        assert (job_id, schedule_id) == (job.id, retired_id)
        raise OSError("retirement acknowledgement unavailable")

    patch = JobUpdateRequest(schedule_expr="30 9 * * *")
    with monkeypatch.context() as failed:
        if failure == "delete":
            failed.setattr(scheduler, "delete", unavailable)
        else:
            failed.setattr(db, "acknowledge_job_schedule_retirement", unacknowledged)
        await watchlists.update_job(job.id, patch, current_user=user, db=db)
    winner = db.get_job(job.id).wf_schedule_id
    assert winner is not None and winner != retired_id
    assert (scheduler.get(retired_id) is not None) == (failure == "delete")
    assert db.list_retired_job_schedule_ids(job.id) == [retired_id]

    # Retry with a fresh service object; an unattached row is not by itself retired.
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    db = type(db).for_user(job.user_id)
    candidate = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None,
        name="Candidate", cron="45 9 * * *", timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True, managed_watchlist_job_id=job.id,
    )
    await watchlists.update_job(job.id, patch, current_user=user, db=db)
    assert db.get_job(job.id).wf_schedule_id == winner
    assert scheduler.get(winner) is not None
    assert scheduler.get(retired_id) is None
    assert scheduler.get(candidate) is not None
    assert db.list_retired_job_schedule_ids(job.id) == []


def test_failed_retirement_record_rolls_back_the_schedule_attachment(scheduled_collections_monitor, monkeypatch):
    db, job = scheduled_collections_monitor
    assert db.set_job_schedule_id(job.id, "old")
    expected = db.get_job(job.id)
    execute = db.backend.execute

    def fail_retirement(sql, *args, **kwargs):
        if "INSERT INTO watchlist_schedule_retirements" in sql:
            raise OSError("retirement record failed")
        return execute(sql, *args, **kwargs)

    monkeypatch.setattr(db.backend, "execute", fail_retirement)
    with pytest.raises(OSError, match="retirement record failed"):
        db.set_job_schedule_id(job.id, "new", expected_job=expected, retire_previous=True)
    assert db.get_job(job.id).wf_schedule_id == "old"
    assert not db.list_retired_job_schedule_ids(job.id)


def test_retirement_scan_failure_does_not_block_a_schedule_edit(scheduled_collections_monitor, monkeypatch, tmp_path):
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "receipt-read.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    workflows_scheduler.register_watchlist_schedule(db, job, tenant_id="default", timezone="UTC")
    old = db.get_job(job.id).wf_schedule_id
    assert old is not None
    updated = db.update_job(job.id, {"schedule_expr": "15 9 * * *"})

    def unavailable(job_id):
        assert job_id == job.id
        raise OSError("retirement scan unavailable")

    with monkeypatch.context() as failed:
        failed.setattr(db, "list_retired_job_schedule_ids", unavailable)
        workflows_scheduler.register_watchlist_schedule(db, updated, tenant_id="default", timezone="UTC")
    winner = db.get_job(job.id).wf_schedule_id
    assert winner is not None and winner != old
    assert scheduler.get(winner).cron == updated.schedule_expr
    assert db.list_retired_job_schedule_ids(job.id) == [old]


def test_restored_monitor_does_not_readmit_its_previous_schedule_generation(scheduled_collections_monitor):
    db, job = scheduled_collections_monitor
    assert db.set_job_schedule_id(job.id, "prior-generation")
    before_delete = db.get_job(job.id)
    assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]

    restored = db.restore_job(job.id)
    assert (restored.schedule_expr, restored.schedule_timezone, restored.active) == (
        before_delete.schedule_expr, before_delete.schedule_timezone, before_delete.active,
    )
    assert restored.wf_schedule_id is None
    assert restored.updated_at != before_delete.updated_at
    with pytest.raises(KeyError, match="job_restore_not_found"):
        db.restore_job(job.id)


def test_restore_leaves_a_new_job_with_the_same_sqlite_id_unchanged(scheduled_collections_monitor):
    db, job = scheduled_collections_monitor
    assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]
    replacement = db.create_job(
        name="New monitor", description=None, scope_json="{}", schedule_expr="15 9 * * *",
        schedule_timezone="UTC", active=True, max_concurrency=None, per_host_delay_ms=None,
        retry_policy_json=None, output_prefs_json=None,
    )
    # Simulate an occupied ID created by the old allocator.
    db.backend.execute(
        "UPDATE scrape_jobs SET id = ? WHERE id = ? AND user_id = ?",
        (job.id, replacement.id, job.user_id),
    )
    replacement = db.get_job(job.id)

    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistJobChangedError

    with pytest.raises(WatchlistJobChangedError):
        db.restore_job(job.id)
    assert db.get_job(job.id) == replacement
    with pytest.raises(WatchlistJobChangedError):
        db.delete_job_reversible(job.id, undo_window_seconds=60)
    assert db.get_job(job.id) == replacement
    with pytest.raises(WatchlistJobChangedError):
        db.restore_job(job.id)
    assert db.get_job(job.id) == replacement
    assert db.delete_job(job.id)
    assert db.restore_job(job.id).name == job.name


def test_new_job_preserves_an_unexpired_undo_record(scheduled_collections_monitor):
    db, job = scheduled_collections_monitor
    assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]
    newer = db.create_job(
        name="New monitor", description=None, scope_json="{}", schedule_expr="15 9 * * *",
        schedule_timezone="UTC", active=True, max_concurrency=None, per_host_delay_ms=None,
        retry_policy_json=None, output_prefs_json=None,
    )
    assert db.delete_job_reversible(newer.id, undo_window_seconds=60)[0]

    assert db.restore_job(job.id).name == job.name
    assert db.restore_job(newer.id).name == newer.name


@pytest.mark.asyncio
async def test_delete_does_not_remove_a_concurrently_attached_schedule(scheduled_collections_monitor, monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import watchlists

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    delete = db.delete_job_reversible

    def attach_before_delete(job_id, **kwargs):
        assert db.set_job_schedule_id(job_id, "new-generation", expected_job=job)
        return delete(job_id, **kwargs)

    monkeypatch.setattr(db, "delete_job_reversible", attach_before_delete)
    with pytest.raises(HTTPException) as exc:
        await watchlists.delete_job(job.id, current_user=user, db=db)
    assert (exc.value.status_code, exc.value.detail) == (409, "job_schedule_changed")
    assert db.get_job(job.id).wf_schedule_id == "new-generation"
    with pytest.raises(KeyError, match="job_restore_not_found"):
        db.restore_job(job.id)


@pytest.mark.asyncio
async def test_delete_records_a_renamed_legacy_schedule_and_restore_uses_a_new_one(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "delete-recovery.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    old_id = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None, name="Renamed monitor",
        cron=job.schedule_expr, timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True,
    )
    assert db.set_job_schedule_id(job.id, old_id, expected_job=job)

    def unavailable(*args, **kwargs):
        raise OSError("scheduler deletion unavailable")

    with monkeypatch.context() as failed:
        failed.setattr(scheduler, "delete", unavailable)
        await watchlists.delete_job(job.id, current_user=user, db=db)
    with pytest.raises(KeyError, match="job_not_found"):
        db.get_job(job.id)
    old = scheduler.get(old_id)
    assert old is not None and old.managed_watchlist_job_id == job.id
    assert db.list_retired_job_schedule_ids(job.id) == [old_id]

    submit = AsyncMock()
    scheduler._core_scheduler = SimpleNamespace(submit=submit)
    monkeypatch.setattr(scheduler, "_may_submit_cron", AsyncMock(return_value=True))
    monkeypatch.setattr(workflows_scheduler, "ensure_schedule_queue_worker", AsyncMock())
    await scheduler._run_schedule(old.id, int(job.user_id), expected_cron=scheduler._cron_definition(old))
    submit.assert_not_awaited()

    await watchlists.restore_job(job.id, current_user=user, db=db)
    restored = db.get_job(job.id)
    assert restored.wf_schedule_id and restored.wf_schedule_id != old_id
    assert scheduler.get(restored.wf_schedule_id).cron == job.schedule_expr
    assert scheduler.get(old_id) is None
    assert db.list_retired_job_schedule_ids(job.id) == []


@pytest.mark.asyncio
async def test_restore_still_registers_schedule_if_history_write_fails(scheduled_collections_monitor, monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "restore-history.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]

    def unavailable(*args, **kwargs):
        raise DatabaseError("history store unavailable")

    monkeypatch.setattr(db, "set_job_history", unavailable)
    response = await watchlists.restore_job(job.id, current_user=user, db=db)
    assert response.id == job.id
    restored = db.get_job(job.id)
    assert restored.wf_schedule_id is not None
    assert scheduler.get(restored.wf_schedule_id).cron == job.schedule_expr


def test_schedule_retirement_uses_the_owner_store_if_the_scheduler_is_unavailable(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "fallback-retirement.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    workflows_scheduler.register_watchlist_schedule(db, job, tenant_id="default", timezone="UTC")
    old_id = db.get_job(job.id).wf_schedule_id
    assert old_id is not None and scheduler.get(old_id) is not None
    edited = db.update_job(job.id, {"schedule_expr": "30 9 * * *"})

    def unavailable():
        raise OSError("default scheduler store unavailable")

    monkeypatch.setattr(workflows_scheduler, "get_workflows_scheduler", unavailable)
    workflows_scheduler.register_watchlist_schedule(db, edited, tenant_id="default", timezone="UTC")
    winner = db.get_job(job.id).wf_schedule_id
    assert winner is not None and winner != old_id
    assert scheduler.get(winner).cron == edited.schedule_expr
    assert scheduler.get(old_id) is None
    assert db.list_retired_job_schedule_ids(job.id) == []


@pytest.mark.asyncio
async def test_deleting_a_feed_records_all_pending_schedule_retirements(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    import json

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    source = db.create_source(
        name="Feed to delete", url="https://example.invalid/retiring.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
    )
    job = _bind_collections_job(db, job, source)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "feed-retirements.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_deleted", lambda **_: None)
    workflows_scheduler.register_watchlist_schedule(db, job, tenant_id="default", timezone="UTC")
    older = db.get_job(job.id).wf_schedule_id
    assert older is not None

    def unavailable(*args, **kwargs):
        raise OSError("scheduler deletion unavailable")

    with monkeypatch.context() as failed:
        failed.setattr(scheduler, "delete", unavailable)
        edited = db.update_job(job.id, {"schedule_expr": "15 9 * * *"})
        workflows_scheduler.register_watchlist_schedule(db, edited, tenant_id="default", timezone="UTC")
        newer = db.get_job(job.id).wf_schedule_id
        assert newer is not None and newer != older
        assert db.list_retired_job_schedule_ids(job.id) == [older]
        assert await collections_feeds.delete_feed_subscription(source.id, current_user=user, db=db) == {"success": True}

    with pytest.raises(KeyError, match="source_not_found"):
        db.get_source(source.id)
    with pytest.raises(KeyError, match="job_not_found"):
        db.get_job(job.id)
    assert set(db.list_retired_job_schedule_ids(job.id)) == {older, newer}
    assert scheduler.get(older) is not None and scheduler.get(newer) is not None


@pytest.mark.asyncio
async def test_feed_delete_rolls_back_source_if_job_delete_fails(scheduled_collections_monitor, monkeypatch):
    import json

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    created = db.create_source(
        name="Feed to keep", url="https://example.invalid/retained.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
        tags=["keep"],
    )
    source = db.get_source(created.id)
    job = _bind_collections_job(db, job, source)
    execute = db.backend.execute

    def unavailable(query, *args, **kwargs):
        if "DELETE FROM scrape_jobs" in query:
            raise OSError("job store failed during deletion")
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(db.backend, "execute", unavailable)
    activity = []
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_deleted", lambda **_: activity.append(True))
    with pytest.raises(HTTPException) as error:
        await collections_feeds.delete_feed_subscription(source.id, current_user=user, db=db)
    assert (error.value.status_code, error.value.detail) == (503, "feed_delete_failed")
    assert db.get_source(source.id) == source
    assert db.get_job(job.id) == job
    assert not activity


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_exists", [True, False])
@pytest.mark.parametrize("legacy_content_schema", [False, True])
async def test_scheduler_rescan_retires_a_row_after_its_feed_is_deleted(
    scheduled_collections_monitor, monkeypatch, tmp_path, schedule_exists, legacy_content_schema,
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    source = db.create_source(
        name="Deleted feed", url="https://example.invalid/deleted.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
    )
    job = _bind_collections_job(db, job, source)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "rescan-retirement.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    monkeypatch.setattr(workflows_scheduler, "_INSTANCE", scheduler)
    workflows_scheduler.register_watchlist_schedule(db, job, tenant_id="default", timezone="UTC")
    linked = db.get_job(job.id)
    assert linked.wf_schedule_id is not None
    db.delete_feed_and_job(source, linked_job_id=job.id, expected_job=linked, retire_schedule=True)
    assert db.list_retired_job_schedule_ids(job.id) == [linked.wf_schedule_id]
    if legacy_content_schema:
        db.backend.execute("ALTER TABLE scrape_jobs DROP COLUMN collections_feed_binding_json", ())
    if not schedule_exists:
        assert scheduler.delete(linked.wf_schedule_id, user_id=int(job.user_id))

    scheduler._aps = SimpleNamespace(get_jobs=lambda: [])
    holds_lease = AsyncMock(return_value=False)
    monkeypatch.setattr(scheduler, "_may_submit_cron", holds_lease)
    add_job = Mock()
    monkeypatch.setattr(scheduler, "_add_job", add_job)
    unknown_owner = int(job.user_id) + 1
    monkeypatch.setattr(scheduler, "_schedule_scan_user_ids", lambda: [int(job.user_id), unknown_owner])
    await scheduler._rescan_once()
    assert (scheduler.get(linked.wf_schedule_id) is not None) == schedule_exists
    assert db.list_retired_job_schedule_ids(job.id) == [linked.wf_schedule_id]

    holds_lease.return_value = True
    add_job.reset_mock()
    await scheduler._rescan_once()

    assert scheduler.get(linked.wf_schedule_id) is None
    assert db.list_retired_job_schedule_ids(job.id) == []
    add_job.assert_not_called()
    assert not (tmp_path / str(unknown_owner)).exists()


@pytest.mark.asyncio
async def test_scheduler_rescan_does_not_create_a_missing_owner_scheduler_store(
    scheduled_collections_monitor, monkeypatch, tmp_path,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    owner = int(job.user_id)
    monkeypatch.setattr(DatabasePaths, "get_single_user_id", lambda: owner + 100)
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", raising=False)
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_DATABASE_URL", raising=False)
    scheduler = workflows_scheduler._WFRecurringScheduler()
    scheduler._aps = SimpleNamespace(get_jobs=lambda: [])
    monkeypatch.setattr(scheduler, "_may_submit_cron", AsyncMock(return_value=True))
    monkeypatch.setattr(scheduler, "_schedule_scan_user_ids", lambda: [owner])
    with db.transaction() as conn:
        db._record_job_schedule_retirement(job.id, "pending-retirement", connection=conn)

    await scheduler._rescan_once()

    assert not (tmp_path / str(owner) / DatabasePaths.WORKFLOWS_SUBDIR).exists()
    assert db.list_retired_job_schedule_ids(job.id) == ["pending-retirement"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_store", ["content", "scheduler"])
async def test_scheduler_rescan_recovers_mixed_stores_with_owner_scope(
    scheduled_collections_monitor, monkeypatch, tmp_path, shared_store,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope
    from tldw_Server_API.app.services import workflows_scheduler

    db, job = scheduled_collections_monitor
    owner = int(job.user_id)
    monkeypatch.setenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", str(tmp_path / "mixed-rescan.db"))
    scheduler = workflows_scheduler._WFRecurringScheduler()
    retired = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None, name="Retired",
        cron="0 * * * *", timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True, managed_watchlist_job_id=job.id,
    )
    candidate = scheduler.create(
        tenant_id="default", user_id=job.user_id, workflow_id=None, name="Never attached",
        cron="0 * * * *", timezone="UTC", inputs={"watchlist_job_id": job.id},
        run_mode="async", validation_mode="block", enabled=True, managed_watchlist_job_id=job.id,
    )
    with db.transaction() as conn:
        db._record_job_schedule_retirement(job.id, retired, connection=conn)
    # Shared-store branches are exercised with a local SQLite transport, not live PostgreSQL.
    scopes = []
    if shared_store == "content":
        def execute(query, *args, **kwargs):
            scope = get_scope()
            assert scope is not None
            scopes.append((scope.user_id, scope.is_admin))
            return db.backend.execute(query, *args, **kwargs)

        shared_backend = SimpleNamespace(execute=execute, table_exists=db.backend.table_exists)
    else:
        shared_backend = None
        monkeypatch.setattr(scheduler._db.backend.config, "backend_type", BackendType.POSTGRESQL)
        monkeypatch.setattr(scheduler, "_local_schedule_user_ids", lambda: [owner])
    monkeypatch.setattr(type(db), "shared_retirement_backend", staticmethod(lambda: shared_backend))
    monkeypatch.setattr(scheduler, "_schedule_scan_user_ids", lambda: [owner + 100])
    monkeypatch.setattr(scheduler, "_may_submit_cron", AsyncMock(return_value=True))
    monkeypatch.setattr(scheduler, "_add_job", Mock())
    scheduler._aps = SimpleNamespace(get_jobs=lambda: [])

    await scheduler._rescan_once()

    assert scheduler._db.get_schedule(retired) is None
    assert scheduler._db.get_schedule(candidate) is not None
    assert db.list_retired_job_schedule_ids(job.id) == []
    if shared_store == "content":
        assert scopes[0] == (None, True)
        assert len(scopes) > 1 and set(scopes[1:]) == {(owner, False)}


def test_new_job_does_not_inherit_retained_run_history(scheduled_collections_monitor):
    db, job = scheduled_collections_monitor
    old_run = db.create_run(job.id)
    assert db.delete_job(job.id)
    newer = db.create_job(
        name="New monitor", description=None, scope_json="{}", schedule_expr=None,
        schedule_timezone=None, active=False, max_concurrency=None, per_host_delay_ms=None,
        retry_policy_json=None, output_prefs_json=None,
    )
    assert db.list_runs_for_job(newer.id, limit=10, offset=0) == ([], 0)
    assert old_run.job_id != newer.id


def test_hard_delete_does_not_transfer_watchlist_membership_to_a_new_source(scheduled_collections_monitor):
    db, _ = scheduled_collections_monitor
    source = db.create_source(name="Original", url="https://example.invalid/original.xml", source_type="rss")
    watchlist = db.create_watchlist(name="Original only")
    db.add_source_to_watchlist(watchlist.id, source.id)

    assert db.delete_source(source.id)
    db.create_source(name="Replacement", url="https://example.invalid/replacement.xml", source_type="rss")

    assert db.list_watchlist_sources(watchlist.id, limit=10, offset=0) == ([], 0)


@pytest.mark.asyncio
async def test_feed_delete_refuses_an_unrelated_job_in_the_same_account(scheduled_collections_monitor, monkeypatch):
    import json

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds

    db, unrelated = scheduled_collections_monitor
    user = User(id=int(unrelated.user_id), username="scheduler-user", email=None, is_active=True)
    feed = db.create_source(
        name="Stale feed", url="https://example.invalid/stale.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": unrelated.id}),
    )
    unrelated = _link_legacy_collections_job(db, unrelated, feed)
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_deleted", lambda **_: None)

    with pytest.raises(HTTPException) as error:
        await collections_feeds.delete_feed_subscription(feed.id, current_user=user, db=db)
    assert error.value.status_code == 409
    assert db.get_job(unrelated.id) == unrelated
    assert db.get_source(feed.id).name == feed.name


@pytest.mark.asyncio
async def test_feed_delete_refuses_if_the_missing_linked_job_is_restored(scheduled_collections_monitor, monkeypatch):
    import json

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    feed = db.create_source(
        name="Feed", url="https://example.invalid/restore-race.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
    )
    _bind_collections_job(db, job, feed)
    assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]
    delete = db.delete_feed_and_job

    def restore_before_delete(*args, **kwargs):
        assert kwargs["expected_job"] is None
        type(db).for_user(job.user_id).restore_job(job.id)
        return delete(*args, **kwargs)

    monkeypatch.setattr(db, "delete_feed_and_job", restore_before_delete)
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_deleted", lambda **_: None)
    with pytest.raises(HTTPException) as error:
        await collections_feeds.delete_feed_subscription(feed.id, current_user=user, db=db)
    assert error.value.status_code == 409
    assert db.get_source(feed.id).name == feed.name
    assert json.loads(db.get_job(job.id).scope_json) == {"sources": [feed.id]}


@pytest.mark.asyncio
@pytest.mark.parametrize("belongs_to_feed", [True, False])
async def test_feed_delete_only_consumes_its_own_job_undo(scheduled_collections_monitor, monkeypatch, belongs_to_feed):
    import json

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds

    db, job = scheduled_collections_monitor
    user = User(id=int(job.user_id), username="scheduler-user", email=None, is_active=True)
    feed = db.create_source(
        name="Feed", url="https://example.invalid/undo.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
    )
    if belongs_to_feed:
        _bind_collections_job(db, job, feed)
    assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_deleted", lambda **_: None)

    if belongs_to_feed:
        assert await collections_feeds.delete_feed_subscription(feed.id, current_user=user, db=db) == {"success": True}
        with pytest.raises(KeyError, match="job_restore_not_found"):
            db.restore_job(job.id)
    else:
        with pytest.raises(HTTPException) as error:
            await collections_feeds.delete_feed_subscription(feed.id, current_user=user, db=db)
        assert error.value.status_code == 409
        assert db.get_source(feed.id).name == feed.name
        assert db.restore_job(job.id).id == job.id


def test_collections_settings_cannot_supply_a_new_job_link():
    from tldw_Server_API.app.api.v1.endpoints import collections_feeds

    updates = {"collections_feed_job_id": 4, "collections_job_id": 5, "collections": {"job_id": 6, "extra": True}}
    result = collections_feeds._merge_settings({}, updates)
    assert collections_feeds._extract_job_id(result) is None
    assert result["collections"]["extra"] is True
    for original in ({"collections_feed_job_id": 7}, {"collections": {"job_id": 8}}):
        updated = collections_feeds._merge_settings(original, updates)
        assert collections_feeds._extract_job_id(updated) == collections_feeds._extract_job_id(original)


def test_legacy_collections_job_survives_concurrent_additive_binding_upgrade(scheduled_collections_monitor, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, local

    db, old_job = scheduled_collections_monitor
    source = db.create_source(name="Legacy feed", url="https://example.invalid/legacy.xml", source_type="rss")
    job = _link_legacy_collections_job(db, old_job, source)
    db.backend.execute("ALTER TABLE scrape_jobs DROP COLUMN collections_feed_binding_json", ())
    lockstep = Barrier(2, timeout=10)
    probes = local()
    get_table_info = db.backend.get_table_info

    def coordinate(table, connection=None):
        result = get_table_info(table, connection=connection)
        if table == "scrape_jobs" and connection is None:
            probes.count = getattr(probes, "count", 0) + 1
            if probes.count == 2:
                lockstep.wait()
        return result

    monkeypatch.setattr(db.backend, "get_table_info", coordinate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(db.ensure_schema) for _ in range(2)]
        for future in futures:
            future.result(timeout=20)
    restored_schema = db.get_job(job.id)
    assert restored_schema == job


def test_feed_delete_rejects_a_new_undo_seen_only_in_the_final_snapshot(scheduled_collections_monitor, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistJobChangedError

    db, _ = scheduled_collections_monitor
    source = db.create_source(name="Feed", url="https://example.invalid/re-delete.xml", source_type="rss")
    execute = db.backend.execute
    inserted = []

    def observe_new_undo(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if "DELETE FROM deleted_jobs" in query and "RETURNING" in query:
            assert result.first is None
            execute(
                "INSERT INTO deleted_jobs (user_id, job_id, payload_json, deleted_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (db.user_id, 999, "{}", "2026-11-19", "2026-11-20"), connection=kwargs["connection"],
            )
            inserted.append(True)
        return result

    monkeypatch.setattr(db.backend, "execute", observe_new_undo)
    with pytest.raises(WatchlistJobChangedError):
        db.delete_feed_and_job(source, linked_job_id=999, expected_job=None, retire_schedule=False)
    assert inserted == [True]
    assert db.get_source(source.id).name == source.name


@pytest.mark.asyncio
async def test_new_feed_binding_survives_edit_and_undo(scheduled_collections_monitor, monkeypatch):
    import json

    from fastapi import BackgroundTasks, HTTPException

    from tldw_Server_API.app.api.v1.endpoints import collections_feeds
    from tldw_Server_API.app.api.v1.schemas.collections_feeds_schemas import CollectionsFeedCreateRequest

    db, unrelated = scheduled_collections_monitor
    user = User(id=int(unrelated.user_id), username="scheduler-user", email=None, is_active=True)
    monkeypatch.setattr(collections_feeds, "_register_schedule", lambda *_, **__: None)
    monkeypatch.setattr(collections_feeds._Personalization, "record_watchlist_source_created", lambda **_: None)
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_deleted", lambda **_: None)
    bg = BackgroundTasks()
    created = await collections_feeds.create_feed_subscription(
        CollectionsFeedCreateRequest(url="https://example.invalid/private.xml", active=False,
                                     settings={"collections_job_id": unrelated.id}),
        bg, current_user=user, db=db,
    )
    assert created.job_id is not None and created.job_id != unrelated.id
    assert not bg.tasks
    job = db.update_job(created.job_id, {"scope_json": "{}", "output_prefs_json": "{}"})
    assert db.delete_job_reversible(job.id, undo_window_seconds=60)[0]
    restored = db.restore_job(job.id)
    feed = db.get_source(created.id)
    assert restored.belongs_to_collections_feed(feed)

    other = db.create_source(
        name="Wrong feed", url="https://example.invalid/wrong.xml", source_type="rss",
        settings_json=json.dumps({"collections_origin": "feed", "collections_feed_job_id": job.id}),
    )
    readback = await collections_feeds.get_feed_subscription(other.id, current_user=user, db=db)
    assert readback.job_id is None
    with pytest.raises(HTTPException) as error:
        await collections_feeds.delete_feed_subscription(other.id, current_user=user, db=db)
    assert error.value.status_code == 409
    assert await collections_feeds.delete_feed_subscription(feed.id, current_user=user, db=db) == {"success": True}
    assert db.get_job(unrelated.id) == unrelated
