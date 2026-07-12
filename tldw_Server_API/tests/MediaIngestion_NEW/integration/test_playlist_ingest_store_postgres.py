from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.tests._plugins.postgres import pg_temp_db as _pg_temp_db_fixture  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


class _FixedClock:
    def now_utc(self) -> datetime:
        return NOW


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
