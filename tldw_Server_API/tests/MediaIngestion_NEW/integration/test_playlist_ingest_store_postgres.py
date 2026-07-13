from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event, Lock

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


def test_postgres_occurrence_job_reservation_and_binding_match_sqlite(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    manager = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
    store = PlaylistIngestStore(manager)
    run = store.create_validated_run(
        "pg-job-owner",
        items=[
            {
                "occurrence_id": "pg-job-occ",
                "input_kind": "direct_url",
                "source_url": "https://www.youtube.com/watch?v=alpha123456",
                "normalized_source_id": "youtube:video:alpha123456",
                "source_kind": "youtube_video",
                "display_metadata": {"title": "Alpha"},
                "state": "staged",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    identity = "playlist-ingest-v1:pg-derived"
    reserved = store.prepare_run_item_job_submission(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        attempt=1,
        batch_id="pg-job-batch",
        idempotency_identity=identity,
        source_kind="url",
        planned_item_id=None,
    )
    repeated = store.prepare_run_item_job_submission(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        attempt=1,
        batch_id="pg-other-batch",
        idempotency_identity=identity,
        source_kind="url",
        planned_item_id=None,
    )
    assert repeated.batch_id == "pg-job-batch"
    with pytest.raises(PlaylistIngestConflictError, match="reservation"):
        store.reset_run_item_job_submission(
            "pg-job-owner",
            run.run_id,
            "pg-job-occ",
            attempt=1,
            batch_id="pg-other-batch",
            idempotency_identity=identity,
            submission_lease_token=reserved.submission_lease_token,
        )
    assert store.get_run_item("pg-job-owner", run.run_id, "pg-job-occ").state == "submit_pending"
    job = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": run.run_id,
            "occurrence_id": "pg-job-occ",
            "attempt": 1,
            "batch_id": "pg-job-batch",
            "idempotency_key": identity,
            "source": reserved.source_url,
            "source_kind": "url",
            "options": {"overwrite_existing": False},
        },
        batch_group="pg-job-batch",
        owner_user_id="pg-job-owner",
        idempotency_key=identity,
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        _transaction_guard=lambda db: store.assert_run_item_submission_lease_for_job_create(
            db,
            "pg-job-owner",
            run.run_id,
            "pg-job-occ",
            attempt=1,
            batch_id="pg-job-batch",
            idempotency_identity=identity,
            submission_lease_token=reserved.submission_lease_token,
        ),
    )
    exact = manager.get_job_by_idempotency(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        idempotency_key=identity,
        owner_user_id="pg-job-owner",
        batch_group="pg-job-batch",
    )
    wrong_batch = manager.get_job_by_idempotency(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        idempotency_key=identity,
        owner_user_id="pg-job-owner",
        batch_group="pg-other-batch",
    )
    assert exact is not None and int(exact["id"]) == int(job["id"])
    assert wrong_batch is None
    bound = store.bind_run_item_job(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="pg-job-batch",
        idempotency_identity=identity,
        submission_lease_token=reserved.submission_lease_token,
    )
    repeated = store.bind_run_item_job(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="pg-job-batch",
        idempotency_identity=identity,
    )

    assert reserved.state == "submit_pending"
    assert bound == repeated
    assert bound.state == "queued"
    assert bound.job_id == int(job["id"])
    assert store.get_run("pg-job-owner", run.run_id).batch_ids == ["pg-job-batch"]

    running = store.reconcile_run_item_job(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        expected_job_id=int(job["id"]),
        expected_attempt=1,
        state="running",
        outcome=None,
        progress_percent=45.0,
        progress_message="download",
        retryable=False,
        media_id=None,
    )
    failed = store.reconcile_run_item_job(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        expected_job_id=int(job["id"]),
        expected_attempt=1,
        state="terminal",
        outcome="processing_failed",
        progress_percent=45.0,
        progress_message="download",
        retryable=True,
        media_id=None,
    )
    retried, changed = store.retry_run_item(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        expected_attempt=1,
        resolved_media_id=None,
    )
    cancelled = store.request_run_item_cancellation(
        "pg-job-owner",
        run.run_id,
        "pg-job-occ",
        expected_attempt=2,
    )
    minimum_event_id, maximum_event_id = store.run_event_bounds("pg-job-owner", run.run_id)

    assert running.state == "running"
    assert failed.outcome == "processing_failed"
    assert changed is True
    assert retried.state == "staged"
    assert retried.attempt == 2
    assert retried.job_id is None
    assert cancelled.state == "terminal"
    assert cancelled.outcome == "cancelled"
    assert minimum_event_id is not None
    assert maximum_event_id is not None and maximum_event_id >= minimum_event_id


def test_postgres_duplicate_action_state_matches_sqlite_contract(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    store = PlaylistIngestStore(JobManager(backend="postgres", db_url=dsn, clock=_FixedClock()))
    run = store.create_validated_run(
        "pg-action-owner",
        items=[
            {
                "occurrence_id": "pg-action-occ",
                "input_kind": "direct_url",
                "source_url": "https://example.com/existing",
                "normalized_source_id": "url:https://example.com/existing",
                "source_kind": "generic_url",
                "display_metadata": {"title": "Existing"},
                "state": "staged",
                "action": "include_existing",
                "metadata_patch": None,
            }
        ],
    )

    prepared = store.prepare_nonprocessing_run_item("pg-action-owner", run.run_id, "pg-action-occ")
    resolved = store.resolve_nonprocessing_run_item(
        "pg-action-owner",
        run.run_id,
        "pg-action-occ",
        outcome="included_existing",
        media_id=17,
    )
    repeated = store.resolve_nonprocessing_run_item(
        "pg-action-owner",
        run.run_id,
        "pg-action-occ",
        outcome="included_existing",
        media_id=17,
    )

    assert prepared.state == "preparing"
    assert resolved == repeated
    assert store.get_run("pg-action-owner", run.run_id).status == "completed"

    failed_run = store.create_validated_run(
        "pg-action-owner",
        items=[
            {
                "occurrence_id": "pg-failed-action-occ",
                "input_kind": "direct_url",
                "source_url": "https://example.com/failed-existing",
                "normalized_source_id": "url:https://example.com/failed-existing",
                "source_kind": "generic_url",
                "display_metadata": {"title": "Failed existing"},
                "state": "staged",
                "action": "include_existing",
                "metadata_patch": None,
            }
        ],
    )
    store.prepare_nonprocessing_run_item(
        "pg-action-owner",
        failed_run.run_id,
        "pg-failed-action-occ",
    )
    failed = store.resolve_nonprocessing_run_item(
        "pg-action-owner",
        failed_run.run_id,
        "pg-failed-action-occ",
        outcome="metadata_update_failed",
        media_id=18,
    )

    assert failed.outcome == "metadata_update_failed"
    assert store.get_run("pg-action-owner", failed_run.run_id).status == "completed"


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
                "normalized_source_id": f"url:https://example.com/pg-video/{ordinal}",
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
    locked_run = store.create_validated_run(
        "pg-owner",
        items=[
            {
                "occurrence_id": "pg-occ-3",
                "input_kind": "materialized_playlist_item",
                "materialization_id": materialized.materialization_id,
                "source_url": "https://example.com/pg-video/3",
                "normalized_source_id": "url:https://example.com/pg-video/3",
                "source_kind": "url",
                "display_metadata": {"title": "PG video 3"},
                "state": "staged",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    assert store.list_run_items("pg-owner", locked_run.run_id)[0].occurrence_id == "pg-occ-3"
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


def test_postgres_validated_mixed_manifest_matches_sqlite_contract(pg_temp_db, monkeypatch):
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
    run = store.create_validated_run(
        "pg-manifest-owner",
        items=[
            {
                "occurrence_id": "pg-url",
                "input_kind": "direct_url",
                "source_url": "https://example.com/video",
                "normalized_source_id": "url:https://example.com/video",
                "source_kind": "generic_url",
                "display_metadata": {"title": "URL"},
                "state": "staged",
                "action": "overwrite",
                "metadata_patch": {"title": "Reviewed"},
            },
            {
                "occurrence_id": "pg-file",
                "input_kind": "file_stub",
                "source_url": None,
                "normalized_source_id": None,
                "source_kind": "file",
                "display_metadata": {"name": "local.mp3", "size_bytes": 12},
                "state": "awaiting_upload",
                "action": "ingest",
                "metadata_patch": None,
            },
        ],
    )

    items = list(store.list_run_items("pg-manifest-owner", run.run_id, limit=10))
    events = list(store.list_run_events("pg-manifest-owner", run.run_id, limit=10))
    assert [(item.occurrence_id, item.state, item.action) for item in items] == [
        ("pg-url", "staged", "overwrite"),
        ("pg-file", "awaiting_upload", "ingest"),
    ]
    assert len(events) == 2
    assert store.get_run("pg-manifest-owner", run.run_id).version == 2
    assert manager.list_jobs(domain="media_ingest", owner_user_id="pg-manifest-owner", limit=10) == []


def test_postgres_live_staging_reference_is_owner_scoped(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    clock = _FixedClock()
    store = PlaylistIngestStore(JobManager(backend="postgres", db_url=dsn, clock=clock))

    def create_staged_run(occurrence_id: str, batch_id: str, identity: str):
        run = store.create_validated_run(
            "pg-staging-owner",
            items=[
                {
                    "occurrence_id": occurrence_id,
                    "input_kind": "file_stub",
                    "source_url": None,
                    "normalized_source_id": None,
                    "source_kind": "file",
                    "display_metadata": {"name": f"{occurrence_id}.mp3"},
                    "state": "awaiting_upload",
                    "action": "ingest",
                    "metadata_patch": None,
                }
            ],
        )
        reserved = store.prepare_run_item_job_submission(
            "pg-staging-owner",
            run.run_id,
            occurrence_id,
            attempt=1,
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_queue="default",
            source_kind="file",
            planned_item_id=None,
        )
        store.record_run_item_staging(
            "pg-staging-owner",
            run.run_id,
            occurrence_id,
            attempt=1,
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_lease_token=reserved.submission_lease_token,
            temp_dir="/tmp/media_ingest_job_pg_shared_alias",
        )
        return run

    expired = create_staged_run("pg-expired", "pg-expired-batch", "playlist-ingest-v1:pg-expired")
    clock.advance(timedelta(days=8))
    create_staged_run("pg-live", "pg-live-batch", "playlist-ingest-v1:pg-live")

    assert store.has_live_run_item_staging_reference(
        "pg-staging-owner",
        "/tmp/media_ingest_job_pg_shared_alias",
        excluding_run_id=expired.run_id,
        excluding_occurrence_id="pg-expired",
    )
    assert not store.has_live_run_item_staging_reference(
        "pg-other-owner",
        "/tmp/media_ingest_job_pg_shared_alias",
        excluding_run_id=expired.run_id,
        excluding_occurrence_id="pg-expired",
    )


def test_postgres_submission_lease_takeover_race_has_one_new_owner(pg_temp_db, monkeypatch):
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
    run = seed.create_validated_run(
        "pg-lease-owner",
        items=[
            {
                "occurrence_id": "pg-lease-occ",
                "input_kind": "direct_url",
                "source_url": "https://www.youtube.com/watch?v=alpha123456",
                "normalized_source_id": "youtube:video:alpha123456",
                "source_kind": "youtube_video",
                "display_metadata": {"title": "Alpha"},
                "state": "staged",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    seed.prepare_run_item_job_submission(
        "pg-lease-owner",
        run.run_id,
        "pg-lease-occ",
        attempt=1,
        batch_id="pg-lease-batch",
        idempotency_identity="playlist-ingest-v1:pg-lease",
        source_kind="url",
        planned_item_id=None,
        submission_lease_token="pg-lease-crashed",
        submission_lease_seconds=5,
    )
    clock.advance(timedelta(seconds=6))

    def take_over(token: str):
        record = new_store().prepare_run_item_job_submission(
            "pg-lease-owner",
            run.run_id,
            "pg-lease-occ",
            attempt=1,
            batch_id="ignored-batch",
            idempotency_identity="playlist-ingest-v1:ignored",
            source_kind="url",
            planned_item_id=None,
            submission_lease_token=token,
            submission_lease_seconds=30,
        )
        return token, record.submission_lease_token, record.submission_lease_generation

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(take_over, ["pg-racer-a", "pg-racer-b"]))

    assert sum(proposed == stored for proposed, stored, _generation in results) == 1
    assert {generation for _proposed, _stored, generation in results} == {2}


def test_postgres_expired_cleanup_cancels_only_held_job_and_releases_quota(
    pg_temp_db,
    monkeypatch,
):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_MEDIA_INGEST_USER_pg-cleanup-held", "1")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.exceptions import JobSubmissionLimitError
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        _RUN_BOUND_JOB_SENTINEL,
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    clock = _FixedClock()
    manager = JobManager(backend="postgres", db_url=dsn, clock=clock)
    store = PlaylistIngestStore(manager)

    def seed(owner: str, occurrence_id: str, identity: str):
        run = store.create_validated_run(
            owner,
            items=[
                {
                    "occurrence_id": occurrence_id,
                    "input_kind": "direct_url",
                    "source_url": f"https://example.com/{occurrence_id}",
                    "normalized_source_id": f"url:https://example.com/{occurrence_id}",
                    "source_kind": "generic_url",
                    "display_metadata": {"title": occurrence_id},
                    "state": "staged",
                    "action": "ingest",
                    "metadata_patch": None,
                }
            ],
        )
        reserved = store.prepare_run_item_job_submission(
            owner,
            run.run_id,
            occurrence_id,
            attempt=1,
            batch_id=f"batch-{occurrence_id}",
            idempotency_identity=identity,
            source_kind="url",
            planned_item_id=None,
        )
        job = manager.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={
                "run_id": run.run_id,
                "occurrence_id": occurrence_id,
                "attempt": 1,
                "batch_id": f"batch-{occurrence_id}",
                "idempotency_key": identity,
                "source": reserved.source_url,
                "source_kind": "url",
                "options": {"overwrite_existing": False},
            },
            owner_user_id=owner,
            batch_group=f"batch-{occurrence_id}",
            idempotency_key=identity,
            available_at=_RUN_BOUND_JOB_SENTINEL,
        )
        return run, reserved, job

    held_run, _held_reserved, held_job = seed(
        "pg-cleanup-held",
        "pg-cleanup-held-occ",
        "playlist-ingest-v1:pg-cleanup-held",
    )
    terminal_run, terminal_reserved, terminal_job = seed(
        "pg-cleanup-terminal",
        "pg-cleanup-terminal-occ",
        "playlist-ingest-v1:pg-cleanup-terminal",
    )
    store.bind_run_item_job(
        "pg-cleanup-terminal",
        terminal_run.run_id,
        "pg-cleanup-terminal-occ",
        attempt=1,
        job_id=int(terminal_job["id"]),
        batch_id="batch-pg-cleanup-terminal-occ",
        idempotency_identity="playlist-ingest-v1:pg-cleanup-terminal",
        submission_lease_token=terminal_reserved.submission_lease_token,
    )
    connection = manager._connect()
    try:
        with manager._pg_cursor(connection) as cursor:
            cursor.execute("UPDATE jobs SET status = 'completed' WHERE id = %s", (int(terminal_job["id"]),))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(JobSubmissionLimitError, match="max queued"):
        manager.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={"source": "https://example.com/quota-before-cleanup"},
            owner_user_id="pg-cleanup-held",
        )

    clock.advance(timedelta(days=8))
    assert store.cleanup_expired_resources("pg-cleanup-held", now=clock.now_utc())["runs"] == 1
    assert store.cleanup_expired_resources("pg-cleanup-terminal", now=clock.now_utc())["runs"] == 1

    assert manager.get_job(int(held_job["id"]))["status"] == "cancelled"
    assert manager.get_job(int(terminal_job["id"]))["status"] == "completed"
    admitted = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={"source": "https://example.com/quota-after-cleanup"},
        owner_user_id="pg-cleanup-held",
    )
    assert admitted["status"] == "queued"
    connection = manager._connect()
    try:
        with manager._pg_cursor(connection) as cursor:
            cursor.execute(
                """
                SELECT scheduled_count FROM job_counters
                WHERE domain = 'media_ingest' AND queue = 'default'
                  AND job_type = 'media_ingest_item'
                """
            )
            counter = cursor.fetchone()
        assert counter is not None
        assert int(counter["scheduled_count"]) == 0
    finally:
        connection.close()
    assert held_run.run_id != terminal_run.run_id


def test_postgres_expired_cleanup_rolls_back_held_cancellation_on_delete_failure(
    pg_temp_db,
    monkeypatch,
):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        _RUN_BOUND_JOB_SENTINEL,
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    clock = _FixedClock()
    manager = JobManager(backend="postgres", db_url=dsn, clock=clock)
    store = PlaylistIngestStore(manager)
    run = store.create_validated_run(
        "pg-cleanup-rollback",
        items=[
            {
                "occurrence_id": "pg-cleanup-rollback-occ",
                "input_kind": "direct_url",
                "source_url": "https://example.com/rollback",
                "normalized_source_id": "url:https://example.com/rollback",
                "source_kind": "generic_url",
                "display_metadata": {"title": "Rollback"},
                "state": "staged",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    identity = "playlist-ingest-v1:pg-cleanup-rollback"
    reserved = store.prepare_run_item_job_submission(
        "pg-cleanup-rollback",
        run.run_id,
        "pg-cleanup-rollback-occ",
        attempt=1,
        batch_id="pg-cleanup-rollback-batch",
        idempotency_identity=identity,
        source_kind="url",
        planned_item_id=None,
    )
    job = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": run.run_id,
            "occurrence_id": "pg-cleanup-rollback-occ",
            "attempt": 1,
            "batch_id": "pg-cleanup-rollback-batch",
            "idempotency_key": identity,
            "source": reserved.source_url,
            "source_kind": "url",
            "options": {"overwrite_existing": False},
        },
        owner_user_id="pg-cleanup-rollback",
        batch_group="pg-cleanup-rollback-batch",
        idempotency_key=identity,
        available_at=_RUN_BOUND_JOB_SENTINEL,
    )
    clock.advance(timedelta(days=8))
    original_query = store._query

    def fail_after_cancel(db, sql, params=()):
        if "DELETE FROM media_ingest_run_events" in sql:
            raise RuntimeError("synthetic postgres cleanup delete failure")
        return original_query(db, sql, params)

    monkeypatch.setattr(store, "_query", fail_after_cancel)

    with pytest.raises(RuntimeError, match="synthetic postgres cleanup delete failure"):
        store.cleanup_expired_resources("pg-cleanup-rollback", now=clock.now_utc())

    assert manager.get_job(int(job["id"]))["status"] == "queued"
    connection = manager._connect()
    try:
        with manager._pg_cursor(connection) as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS count FROM media_ingest_runs WHERE run_id = %s",
                (run.run_id,),
            )
            remaining = cursor.fetchone()
        assert remaining is not None
        assert int(remaining["count"]) == 1
    finally:
        connection.close()


def test_postgres_validated_manifest_rejects_stale_materialization_authority(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    manager = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
    store = PlaylistIngestStore(manager)
    preflight = store.create_preflight(
        "pg-race-owner",
        source_url="https://example.com/playlist",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "pg-race-owner",
        preflight.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": "pg-race-occ",
                "ordinal": 1,
                "occurrence_index_for_source": 1,
                "source_url": "https://example.com/video",
                "normalized_source_id": "url:https://example.com/video",
                "source_kind": "generic_url",
                "availability": "available",
                "duplicate_status": "not_found",
                "selected_by_default": True,
                "display_metadata": {"title": "Authoritative"},
            }
        ],
    )
    materialized = store.create_materialization(
        "pg-race-owner",
        preflight_id=preflight.preflight_id,
        occurrence_ids=["pg-race-occ"],
    )
    authoritative = store.resolve_materialization_occurrences(
        "pg-race-owner",
        [(materialized.materialization_id, "pg-race-occ")],
    )[0]
    with store._connection(write=True) as db:
        store._query(
            db,
            "UPDATE playlist_materializations SET status = 'expired' WHERE materialization_id = ?",
            (materialized.materialization_id,),
        )

    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        store.create_validated_run(
            "pg-race-owner",
            items=[
                {
                    "occurrence_id": authoritative.occurrence_id,
                    "input_kind": "materialized_playlist_item",
                    "materialization_id": authoritative.materialization_id,
                    "source_url": authoritative.source_url,
                    "normalized_source_id": authoritative.normalized_source_id,
                    "source_kind": authoritative.source_kind,
                    "display_metadata": authoritative.display_metadata,
                    "state": "staged",
                    "action": "ingest",
                    "metadata_patch": None,
                }
            ],
        )

    with store._connection(write=False) as db:
        count = store._query(
            db,
            "SELECT COUNT(*) AS count FROM media_ingest_runs WHERE owner_user_id = ?",
            ("pg-race-owner",),
        ).fetchone()
    assert int(dict(count)["count"]) == 0


def test_postgres_preflight_admission_serializes_empty_capacity_predicate(pg_temp_db, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_GLOBAL_CAPACITY", "1")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_OWNER_CAPACITY", "1")
    dsn = str(pg_temp_db["dsn"])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
        PlaylistPreflightBusyError,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

    ensure_jobs_tables_pg(dsn)
    barrier = Event()
    ready_count = 0
    ready_lock = Lock()

    def create() -> str:
        nonlocal ready_count
        manager = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
        with ready_lock:
            ready_count += 1
            if ready_count == 2:
                barrier.set()
        assert barrier.wait(10)
        try:
            return (
                PlaylistIngestService(manager)
                .create_preflight(
                    "pg-capacity-owner",
                    url="https://www.youtube.com/playlist?list=PLpgcapacity",
                    max_items=20,
                    timeout_seconds=10,
                )
                .preflight_id
            )
        except PlaylistPreflightBusyError:
            return "busy"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: create(), range(2)))

    manager = JobManager(backend="postgres", db_url=dsn, clock=_FixedClock())
    assert results.count("busy") == 1
    assert len([result for result in results if result != "busy"]) == 1
    assert len(manager.list_jobs(domain="media_ingest", owner_user_id="pg-capacity-owner", limit=10)) == 1


def test_postgres_preflight_publication_uses_database_clock_and_is_immediately_acquirable(pg_temp_db, monkeypatch):
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
    preflight = store.reserve_preflight(
        "pg-publication-owner",
        source_url="https://www.youtube.com/playlist?list=PLpgpublication",
        source_kind="youtube_playlist",
        playlist_id="PLpgpublication",
        ttl_seconds=600,
        global_capacity=2,
        owner_capacity=1,
    )
    payload = {"preflight_id": preflight.preflight_id, "max_items": 20, "timeout_seconds": 10}
    job = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload=payload,
        owner_user_id="pg-publication-owner",
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )

    bound = store.bind_preflight_job(
        "pg-publication-owner",
        preflight.preflight_id,
        int(job["id"]),
        expected_queue="default",
        expected_payload=payload,
    )
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="pg-publication-worker",
        job_type="playlist_preflight",
    )

    assert bound.job_id == int(job["id"])
    assert claimed is not None
    assert int(claimed["id"]) == int(job["id"])


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
