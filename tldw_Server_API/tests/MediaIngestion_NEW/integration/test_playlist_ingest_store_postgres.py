from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

from tldw_Server_API.tests._plugins.postgres import pg_temp_db as _pg_temp_db_fixture  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


class _FixedClock:
    def __init__(self) -> None:
        self.current = NOW

    def now_utc(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


def test_playlist_store_postgres_matches_sqlite_contract(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    manager = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
    store = PlaylistIngestStore(manager)
    preflight = store.create_preflight(
        "pg-owner",
        source_url="https://example.com/pg-playlist",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "pg-owner",
        preflight.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": f"pg-occ-{ordinal}",
                "ordinal": ordinal,
                "occurrence_index_for_source": 1,
                "source_url": f"https://example.com/pg-video/{ordinal}",
                "source_kind": "url",
                "availability": "available",
                "duplicate_status": "not_found",
                "display_metadata": {"title": f"PG video {ordinal}"},
            }
            for ordinal in range(1, 4)
        ],
    )
    first = store.list_preflight_items("pg-owner", preflight.preflight_id, limit=2)
    second = store.list_preflight_items(
        "pg-owner",
        preflight.preflight_id,
        limit=2,
        cursor=first.next_cursor,
    )
    assert [item.occurrence_id for item in (*first, *second)] == [
        "pg-occ-1",
        "pg-occ-2",
        "pg-occ-3",
    ]

    materialized = store.create_materialization(
        "pg-owner",
        preflight_id=preflight.preflight_id,
        occurrence_ids=["pg-occ-3", "pg-occ-1"],
    )
    run = store.create_run("pg-owner", materialization_ids=[materialized.id])
    assert store.compare_and_set_run_item_state(
        "pg-owner",
        run.run_id,
        "pg-occ-1",
        expected_state="staged",
        new_state="running",
    )
    event = store.append_run_event(
        "pg-owner",
        run.run_id,
        event_type="item_running",
        occurrence_id="pg-occ-1",
        state="running",
        expected_version=1,
    )
    assert store.list_run_events("pg-owner", run.run_id)[0].event_id == event.event_id
    assert store.get_run("pg-owner", run.run_id).version == 2


def test_postgres_ready_snapshot_guard_rejects_cancelled_linked_job(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
        PlaylistPreflightLeaseLostError,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    manager = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
    store = PlaylistIngestStore(manager)
    job = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload={"preflight_id": "guarded"},
        owner_user_id="pg-guard-owner",
    )
    preflight = store.create_preflight(
        "pg-guard-owner",
        source_url="https://example.com/guarded",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
        job_id=int(job["id"]),
    )
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="pg-guard-test",
        job_type="playlist_preflight",
    )
    assert claimed is not None
    assert manager.cancel_job(int(job["id"]), reason="race") is True

    with pytest.raises(PlaylistPreflightLeaseLostError) as exc_info:
        store.replace_preflight_snapshot(
            "pg-guard-owner",
            preflight.preflight_id,
            status="ready",
            items=[],
            expected_job_id=int(claimed["id"]),
            expected_lease_id=str(claimed["lease_id"]),
            expected_worker_id=str(claimed["worker_id"]),
        )

    assert exc_info.value.cancelled is True
    assert store.get_preflight("pg-guard-owner", preflight.preflight_id).status == "pending"


def test_postgres_reclaimed_job_fences_stale_preflight_writer(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
        PlaylistPreflightLeaseLostError,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    manager_a = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
    manager_b = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
    store_a = PlaylistIngestStore(manager_a)
    store_b = PlaylistIngestStore(manager_b)
    job = manager_a.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload={"preflight_id": "pg-stale"},
        owner_user_id="pg-stale-owner",
    )
    preflight = store_a.create_preflight(
        "pg-stale-owner",
        source_url="https://example.com/pg-stale",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
        job_id=int(job["id"]),
    )
    stale_claim = manager_a.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="pg-worker-a",
        job_type="playlist_preflight",
    )
    assert stale_claim is not None
    connection = manager_a._connect()
    try:
        with manager_a._pg_cursor(connection) as cursor:
            cursor.execute(
                "UPDATE jobs SET leased_until = NOW() - INTERVAL '1 second' WHERE id = %s",
                (int(job["id"]),),
            )
        connection.commit()
    finally:
        connection.close()
    active_claim = manager_b.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="pg-worker-b",
        job_type="playlist_preflight",
    )
    assert active_claim is not None

    with pytest.raises(PlaylistPreflightLeaseLostError):
        store_a.replace_preflight_snapshot(
            "pg-stale-owner",
            preflight.preflight_id,
            status="ready",
            items=[],
            expected_job_id=int(stale_claim["id"]),
            expected_lease_id=str(stale_claim["lease_id"]),
            expected_worker_id=str(stale_claim["worker_id"]),
        )

    assert store_a.get_preflight("pg-stale-owner", preflight.preflight_id).status == "pending"
    store_b.replace_preflight_snapshot(
        "pg-stale-owner",
        preflight.preflight_id,
        status="ready",
        items=[],
        expected_job_id=int(active_claim["id"]),
        expected_lease_id=str(active_claim["lease_id"]),
        expected_worker_id=str(active_claim["worker_id"]),
    )
    assert store_b.get_preflight("pg-stale-owner", preflight.preflight_id).status == "ready"


def test_postgres_no_cas_concurrent_events_both_persist(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    clock = _FixedClock()
    seed = PlaylistIngestStore(JobManager(backend="postgres", db_url=dsn, clock=clock))
    preflight = seed.create_preflight(
        "pg-events-owner",
        source_url="https://example.com/events",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    seed.replace_preflight_snapshot(
        "pg-events-owner",
        preflight.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": "pg-event-occ",
                "ordinal": 1,
                "source_url": "https://example.com/event/1",
                "source_kind": "url",
            }
        ],
    )
    materialized = seed.create_materialization(
        "pg-events-owner",
        preflight_id=preflight.preflight_id,
        occurrence_ids=["pg-event-occ"],
    )
    run = seed.create_run("pg-events-owner", materialization_ids=[materialized.id])

    def append(index: int):
        store = PlaylistIngestStore(JobManager(backend="postgres", db_url=dsn, clock=clock))
        return store.append_run_event("pg-events-owner", run.run_id, event_type=f"event-{index}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        events = list(pool.map(append, range(2)))

    assert len(seed.list_run_events("pg-events-owner", run.run_id)) == 2
    assert sorted(event.event_id for event in events) == [
        event.event_id for event in seed.list_run_events("pg-events-owner", run.run_id)
    ]
    assert seed.get_run("pg-events-owner", run.run_id).version == 3


def test_postgres_cleanup_race_leaves_no_orphan_children(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    clock = _FixedClock()

    def new_store():
        return PlaylistIngestStore(JobManager(backend="postgres", db_url=dsn, clock=clock))

    seed = new_store()
    pending = seed.create_preflight(
        "pg-cleanup-owner",
        source_url="https://example.com/pending",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    ready = seed.create_preflight(
        "pg-cleanup-owner",
        source_url="https://example.com/ready",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    seed.replace_preflight_snapshot(
        "pg-cleanup-owner",
        ready.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": "pg-cleanup-occ",
                "ordinal": 1,
                "source_url": "https://example.com/cleanup/1",
                "source_kind": "url",
            }
        ],
    )
    materialized = seed.create_materialization(
        "pg-cleanup-owner",
        preflight_id=ready.preflight_id,
        occurrence_ids=["pg-cleanup-occ"],
        expires_at=NOW + timedelta(hours=1),
    )
    run = seed.create_run(
        "pg-cleanup-owner",
        materialization_ids=[materialized.id],
        expires_at=NOW + timedelta(hours=1),
    )

    release = Event()
    snapshot_locked = Event()
    event_locked = Event()
    cleanup_started = Event()
    snapshot_store = new_store()
    event_store = new_store()
    cleanup_store = new_store()

    original_snapshot_query = snapshot_store._query
    original_event_query = event_store._query
    original_cleanup_query = cleanup_store._query

    def pause_snapshot(db, sql, params=()):
        result = original_snapshot_query(db, sql, params)
        if "SELECT status" in sql and "FOR UPDATE" in sql:
            snapshot_locked.set()
            assert release.wait(10)
        return result

    def pause_event(db, sql, params=()):
        result = original_event_query(db, sql, params)
        if "SELECT version" in sql and "FOR UPDATE" in sql:
            event_locked.set()
            assert release.wait(10)
        return result

    def observe_cleanup(db, sql, params=()):
        if "SELECT preflight_id" in sql and "FOR UPDATE" in sql:
            cleanup_started.set()
        return original_cleanup_query(db, sql, params)

    monkeypatch.setattr(snapshot_store, "_query", pause_snapshot)
    monkeypatch.setattr(event_store, "_query", pause_event)
    monkeypatch.setattr(cleanup_store, "_query", observe_cleanup)

    with ThreadPoolExecutor(max_workers=3) as pool:
        snapshot_future = pool.submit(
            snapshot_store.replace_preflight_snapshot,
            "pg-cleanup-owner",
            pending.preflight_id,
            status="ready",
            items=[
                {
                    "occurrence_id": "pg-late-occ",
                    "ordinal": 1,
                    "source_url": "https://example.com/late",
                    "source_kind": "url",
                }
            ],
        )
        event_future = pool.submit(
            event_store.append_run_event,
            "pg-cleanup-owner",
            run.run_id,
            event_type="final-event",
        )
        assert snapshot_locked.wait(10)
        assert event_locked.wait(10)
        clock.advance(timedelta(hours=2))
        cleanup_future = pool.submit(
            cleanup_store.cleanup_expired,
            "pg-cleanup-owner",
            now=clock.now_utc(),
        )
        assert cleanup_started.wait(10)
        release.set()
        snapshot_future.result(timeout=10)
        event_future.result(timeout=10)
        cleanup_future.result(timeout=10)

    manager = JobManager(backend="postgres", db_url=dsn, clock=clock)
    connection = manager._connect()
    try:
        with manager._pg_cursor(connection) as cursor:
            for table in (
                "playlist_preflight_items",
                "playlist_materialization_items",
                "media_ingest_run_items",
                "media_ingest_run_events",
            ):
                cursor.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE owner_user_id = %s", ("pg-cleanup-owner",)
                )  # nosec B608
                assert cursor.fetchone()["count"] == 0
    finally:
        connection.close()
