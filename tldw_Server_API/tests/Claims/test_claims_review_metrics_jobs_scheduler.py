from __future__ import annotations

import asyncio
import threading
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.core.Jobs.operations.contracts import AdmissionResult, OperationOutcome
from tldw_Server_API.app.services import claims_review_metrics_scheduler as service

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_scheduler_environment(monkeypatch):
    for name in (
        "CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED",
        "CLAIMS_REVIEW_METRICS_INTERVAL_SEC",
        "CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS",
        "CLAIMS_REVIEW_METRICS_JOBS_ENABLED",
        "CLAIMS_JOBS_ENABLED",
        "CLAIMS_JOBS_WORKER_ENABLED",
        "AUTH_MODE",
        "CLAIMS_JOBS_QUEUE",
        "CLAIMS_JOBS_MAX_RETRIES_REVIEW_METRICS",
    ):
        monkeypatch.delenv(name, raising=False)


def config(**values):
    return service.resolve_scheduler_config(
        {
            "CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED": True,
            "CLAIMS_REVIEW_METRICS_JOBS_ENABLED": True,
            "CLAIMS_JOBS_ENABLED": True,
            "AUTH_MODE": "multi_user",
            **values,
        }
    )


@pytest.mark.parametrize(
    "global_flag,metrics_flag,worker_flag,expected",
    [
        (True, True, False, "jobs"),
        (True, False, True, "local"),
        (False, True, True, "local"),
        (False, False, False, "local"),
    ],
)
def test_route_ignores_worker_flag(global_flag, metrics_flag, worker_flag, expected):
    resolved = config(
        CLAIMS_JOBS_ENABLED=global_flag,
        CLAIMS_REVIEW_METRICS_JOBS_ENABLED=metrics_flag,
        CLAIMS_JOBS_WORKER_ENABLED=worker_flag,
    )
    assert resolved.mode == expected


def test_false_environment_value_overrides_true_setting(monkeypatch):
    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", "false")
    assert config().enabled is False


@pytest.mark.parametrize("raw,expected", [(None, 86400), ("bad", 86400), (0, 86400), (-1, 86400), (1, 60), (100, 100)])
def test_interval_normalization(raw, expected):
    assert config(CLAIMS_REVIEW_METRICS_INTERVAL_SEC=raw).interval_seconds == expected


def test_large_numeric_interval_disables_only_scheduler():
    resolved = config(CLAIMS_REVIEW_METRICS_INTERVAL_SEC=10**100)
    assert not resolved.enabled


def test_configuration_is_a_snapshot():
    values = {"CLAIMS_JOBS_QUEUE": "first", "CLAIMS_JOBS_ENABLED": True, "CLAIMS_REVIEW_METRICS_JOBS_ENABLED": True}
    resolved = service.resolve_scheduler_config(values)
    values["CLAIMS_JOBS_QUEUE"] = "second"
    assert resolved.job_settings["CLAIMS_JOBS_QUEUE"] == "first"


def test_scheduler_preserves_shared_claims_queue_default():
    assert config().job_settings["CLAIMS_JOBS_QUEUE"] == "default"


def test_window_floors_fractional_pre_epoch_timestamp():
    now = datetime(1969, 12, 31, 23, 59, 59, 500000, tzinfo=timezone.utc)
    scheduled, _, _ = service.capture_window(now, 60, 2)
    assert scheduled == "1969-12-31T23:59:00Z"


@given(
    st.datetimes(timezones=st.just(timezone.utc), min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1)),
    st.integers(min_value=60, max_value=86400 * 365),
    st.integers(min_value=1, max_value=366),
)
def test_real_window_derivation(now, interval, lookback):
    scheduled_for, start_date, end_date = service.capture_window(now, interval, lookback)
    slot = datetime.fromisoformat(scheduled_for.replace("Z", "+00:00"))
    assert slot <= now < slot + timedelta(seconds=interval)
    assert end_date == now.date()
    assert start_date == now.date() - timedelta(days=lookback - 1)


async def owners(*args, **kwargs):
    for owner in ["42", "43"]:
        yield owner


@pytest.mark.asyncio
async def test_callback_isolates_owner_failure_and_offloads_admission(monkeypatch):
    monkeypatch.setattr(service, "_iter_owner_ids", owners, raising=False)
    calls = []
    event_loop_thread = threading.get_ident()

    def admit(**kwargs):
        calls.append((kwargs, threading.get_ident()))
        if kwargs["owner_user_id"] == "42":
            raise ValueError("private SQL detail")
        return AdmissionResult.applied(row={"id": 2})

    monkeypatch.setattr(service.claims_jobs, "enqueue_claims_review_metrics", admit, raising=False)
    result = await service.run_review_metrics_callback(
        config(),
        now=datetime(2026, 9, 7, 23, 59, tzinfo=timezone.utc),
        job_manager=object(),
    )
    assert result["failed"] == 1 and result["accepted"] == 1
    assert all(thread != event_loop_thread for _, thread in calls)
    assert {call["end_date"] for call, _ in calls} == {"2026-09-07"}


@pytest.mark.asyncio
async def test_transient_admission_reuses_identity_and_counts_replay(monkeypatch):
    monkeypatch.setattr(service, "_iter_owner_ids", owners, raising=False)
    calls = []

    def admit(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("private connection")
        return AdmissionResult.existing(row={"id": 1})

    monkeypatch.setattr(service.claims_jobs, "enqueue_claims_review_metrics", admit, raising=False)
    result = await service.run_review_metrics_callback(config(), job_manager=object())
    assert calls[0] == calls[1]
    assert result["deduplicated"] == 2


@pytest.mark.parametrize("wrapped", [False, True])
async def test_native_postgres_admission_timeout_retries_same_owner_identity(monkeypatch, wrapped):
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    error = pytest.importorskip("psycopg").errors.ConnectionTimeout("private DSN")
    if wrapped:
        wrapper = DatabaseError("private storage detail")
        wrapper.__cause__ = error
        error = wrapper
    monkeypatch.setattr(service, "_iter_owner_ids", owners)
    calls = []

    def admit(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        return AdmissionResult.applied(row={"id": 1})

    monkeypatch.setattr(service.claims_jobs, "enqueue_claims_review_metrics", admit)
    result = await service.run_review_metrics_callback(config(), job_manager=object())
    assert calls[0] == calls[1]
    assert result["accepted"] == 2 and result["failed"] == 0


@pytest.mark.asyncio
async def test_shutdown_during_retry_prevents_later_admissions(monkeypatch):
    stop = asyncio.Event()
    monkeypatch.setattr(service, "_iter_owner_ids", owners, raising=False)
    calls = []

    def admit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(outcome=OperationOutcome.BACKEND_CONFLICT)

    monkeypatch.setattr(service.claims_jobs, "enqueue_claims_review_metrics", admit, raising=False)
    callback = asyncio.create_task(service.run_review_metrics_callback(config(), stop_event=stop, job_manager=object()))
    while not calls:
        await asyncio.sleep(0)
    stop.set()
    await asyncio.wait_for(callback, 1)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_startup_defers_io_and_registers_coalesced_unlimited_grace(monkeypatch):
    captured = {}
    stopped = asyncio.Event()

    class Scheduler:
        def __init__(self, **kwargs):
            pass

        def add_job(self, callback, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

        def pause(self):
            pass

        def shutdown(self, wait):
            stopped.set()

    monkeypatch.setattr(service, "AsyncIOScheduler", Scheduler, raising=False)
    monkeypatch.setattr(service, "settings", {"CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED": True})
    task = await service.start_claims_review_metrics_scheduler()
    assert task is not None and task.get_name() == "claims_review_metrics_scheduler"
    await asyncio.sleep(0)
    assert captured["max_instances"] == 1 and captured["coalesce"] is True
    assert captured["misfire_grace_time"] is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_postgres_discovery_pages_off_loop_and_stops_before_next_page(monkeypatch):
    monkeypatch.setattr(service.content_db_settings, "backend_type", BackendType.POSTGRESQL)
    stop = asyncio.Event()
    main_thread = threading.get_ident()
    calls = []
    page = [str(owner) for owner in range(100, 200)]

    def fetch_page(start, end, after):
        calls.append((start, end, after, threading.get_ident()))
        return page

    monkeypatch.setattr(service, "_postgres_owner_page", fetch_page)
    seen = []
    async for owner in service._iter_owner_ids(config(), date(2026, 9, 6), date(2026, 9, 7), stop):
        seen.append(owner)
        if len(seen) == 100:
            stop.set()
    assert seen == page
    assert len(calls) == 1 and calls[0][-1] != main_thread


@pytest.mark.asyncio
async def test_postgres_keyset_fanout_skips_invalid_owners(monkeypatch):
    monkeypatch.setattr(service.content_db_settings, "backend_type", BackendType.POSTGRESQL)
    cursors = []
    page = [str(owner) for owner in range(100, 200)]

    def fetch_page(start, end, after):
        cursors.append(after)
        return page if after is None else ["200", "invalid"]

    monkeypatch.setattr(service, "_postgres_owner_page", fetch_page)
    seen = [
        owner async for owner in service._iter_owner_ids(config(), date(2026, 9, 6), date(2026, 9, 7), asyncio.Event())
    ]
    assert seen == page + ["200"]
    assert cursors == [None, "199"]


@pytest.mark.parametrize("failure", [False, True])
def test_postgres_discovery_scope_closes_session_before_restoration(monkeypatch, failure):
    scopes = []

    class Session:
        def __enter__(self):
            scopes.append(get_scope())
            return self

        def __exit__(self, *args):
            scopes.append(get_scope())

        def list_claims_review_user_ids_page(self, **kwargs):
            assert kwargs["limit"] == 100
            if failure:
                raise RuntimeError("private SQL")
            return ["42"]

    monkeypatch.setattr(service, "managed_media_database", lambda **kwargs: Session())
    with scoped_context(user_id=9, org_ids=[4], team_ids=[5], is_admin=False) as original:
        if failure:
            with pytest.raises(RuntimeError):
                service._postgres_owner_page(date(2026, 9, 6), date(2026, 9, 7), None)
        else:
            assert service._postgres_owner_page(date(2026, 9, 6), date(2026, 9, 7), None) == ["42"]
        assert get_scope() is original
    assert len(scopes) == 2
    assert all(
        scope.is_admin and scope.user_id is None and not scope.org_ids and not scope.team_ids for scope in scopes
    )


@pytest.mark.asyncio
async def test_local_route_never_admits_and_missing_owner_does_not_block_later_owner(monkeypatch):
    monkeypatch.setattr(service, "_iter_owner_ids", owners)
    calls = []

    def aggregate(owner, start, end):
        calls.append(owner)
        if owner == "42":
            raise FileNotFoundError("private path")
        return 1

    monkeypatch.setattr(service, "_aggregate_owner", aggregate)
    monkeypatch.setattr(
        service.claims_jobs,
        "enqueue_claims_review_metrics",
        lambda **kwargs: pytest.fail("local route attempted admission"),
    )
    result = await service.run_review_metrics_callback(config(CLAIMS_REVIEW_METRICS_JOBS_ENABLED=False))
    assert calls == ["42", "43"] and result["failed"] == 0


@pytest.mark.asyncio
async def test_typed_terminal_admission_never_runs_local_fallback(monkeypatch):
    monkeypatch.setattr(service, "_iter_owner_ids", owners)
    calls = []

    def reject(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(outcome=OperationOutcome.ADMISSION_REJECTED)

    monkeypatch.setattr(service.claims_jobs, "enqueue_claims_review_metrics", reject)
    monkeypatch.setattr(service, "_aggregate_owner", lambda *args: pytest.fail("Jobs route ran locally"))
    result = await service.run_review_metrics_callback(config())
    assert len(calls) == 2 and result["failed"] == 2


@pytest.mark.asyncio
async def test_delayed_callback_uses_current_repair_window_not_stale_misfire_date(monkeypatch):
    monkeypatch.setattr(service, "_iter_owner_ids", owners)
    calls = []
    monkeypatch.setattr(
        service.claims_jobs,
        "enqueue_claims_review_metrics",
        lambda **kwargs: (calls.append(kwargs), AdmissionResult.applied(row={"id": 1}))[1],
    )
    await service.run_review_metrics_callback(config(), now=datetime(2026, 9, 8, 0, 10, tzinfo=timezone.utc))
    assert {call["scheduled_for"] for call in calls} == {"2026-09-08T00:00:00Z"}
    assert {(call["start_date"], call["end_date"]) for call in calls} == {("2026-09-07", "2026-09-08")}


def test_local_missing_owner_does_not_create_owner_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(service.content_db_settings, "backend_type", BackendType.SQLITE)
    monkeypatch.setattr(service.DatabasePaths, "resolve_user_db_base_dir", lambda **kwargs: tmp_path)
    with pytest.raises(FileNotFoundError):
        service._aggregate_owner("42", date(2026, 9, 6), date(2026, 9, 7))
    assert not (tmp_path / "42").exists()


def test_missing_discovery_base_is_not_created(monkeypatch, tmp_path):
    base = tmp_path / "absent"
    monkeypatch.setattr(service.DatabasePaths, "resolve_user_db_base_dir", lambda **kwargs: base)
    assert service._enumerate_sqlite_user_ids(single_user_mode=False) == []
    assert not base.exists()


@pytest.mark.asyncio
async def test_repeated_lifecycle_cancellation_still_shuts_down_scheduler(monkeypatch):
    started = asyncio.Event()
    draining = asyncio.Event()
    state = {}
    real_wait = asyncio.wait

    class Scheduler:
        def __init__(self, **kwargs):
            pass

        def add_job(self, callback, **kwargs):
            state["callback"] = callback

        def start(self):
            state["work"] = asyncio.create_task(state["callback"]())

        def pause(self):
            pass

        def shutdown(self, wait):
            state["shutdown"] = True

    async def blocked_callback(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    async def record_drain(*args, **kwargs):
        draining.set()
        return await real_wait(*args, **kwargs)

    monkeypatch.setattr(service, "AsyncIOScheduler", Scheduler)
    monkeypatch.setattr(service, "settings", {"CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED": True})
    monkeypatch.setattr(service, "run_review_metrics_callback", blocked_callback)
    monkeypatch.setattr(service.asyncio, "wait", record_drain)
    task = await service.start_claims_review_metrics_scheduler()
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        await asyncio.wait_for(draining.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.get("shutdown") is True
    finally:
        state["work"].cancel()
        await asyncio.gather(state["work"], return_exceptions=True)
