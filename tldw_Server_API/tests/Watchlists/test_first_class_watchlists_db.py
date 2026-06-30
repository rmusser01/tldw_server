from __future__ import annotations

import json
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


pytestmark = pytest.mark.unit


def _make_db(tmp_path, *, user_id: int = 123) -> WatchlistsDatabase:
    db_path = tmp_path / f"watchlists_{user_id}.db"
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    return WatchlistsDatabase(user_id=user_id, backend=backend)


def test_supplied_backend_schema_cache_uses_target_key(tmp_path) -> None:
    db_path = tmp_path / "watchlists_external.db"
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    original_keys = set(WatchlistsDatabase._schema_init_keys)
    WatchlistsDatabase._schema_init_keys.clear()
    try:
        WatchlistsDatabase(user_id=123, backend=backend)

        assert str(db_path) in WatchlistsDatabase._schema_init_keys
        assert f"backend:{id(backend)}" not in WatchlistsDatabase._schema_init_keys
    finally:
        try:
            backend.get_pool().close_all()
        except Exception:
            _ = None
        WatchlistsDatabase._schema_init_keys.clear()
        WatchlistsDatabase._schema_init_keys.update(original_keys)


def _create_source(db: WatchlistsDatabase, *, label: str, watchlist_id: int | None = None):
    return db.create_source(
        name=f"{label} Feed",
        url=f"https://example.com/{label}/rss.xml",
        source_type="rss",
        active=True,
        settings_json=None,
        tags=["news"],
        group_ids=[],
        watchlist_id=watchlist_id,
    )


def _create_job(db: WatchlistsDatabase, *, label: str, source_id: int, watchlist_id: int | None = None):
    return db.create_job(
        name=f"{label} Daily",
        description=f"{label} monitor",
        scope_json=json.dumps({"sources": [int(source_id)]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=0,
        retry_policy_json=json.dumps({}),
        output_prefs_json=json.dumps({}),
        job_filters_json=None,
        watchlist_id=watchlist_id,
    )


def _create_run_item(db: WatchlistsDatabase, *, label: str, job_id: int, source_id: int):
    run = db.create_run(job_id, status="finished")
    item = db.record_scraped_item(
        run_id=int(run.id),
        job_id=job_id,
        source_id=source_id,
        media_id=None,
        media_uuid=None,
        url=f"https://example.com/{label}/story",
        title=f"{label} story",
        summary=f"{label} summary",
        published_at=None,
        tags=["news"],
        status="ingested",
    )
    return run, item


def test_default_watchlist_created_once_and_backfills_jobs_and_sources(tmp_path):
    db = _make_db(tmp_path)
    source = _create_source(db, label="imported")
    job = _create_job(db, label="imported", source_id=int(source.id))

    default = db.ensure_default_watchlist()

    assert default.name == "Imported Watchlist"
    assert default.domain == "general"
    assert default.status == "active"
    assert db.ensure_default_watchlist().id == default.id

    sources, source_total = db.list_watchlist_sources(default.id, limit=50, offset=0)
    assert source_total == 1
    assert [int(row.id) for row in sources] == [int(source.id)]
    assert db.get_job(int(job.id)).watchlist_id == default.id

    db.backend.execute("DELETE FROM watchlist_sources WHERE watchlist_id = ? AND source_id = ?", (default.id, source.id))
    db.backend.execute("UPDATE scrape_jobs SET watchlist_id = NULL WHERE id = ?", (job.id,))

    db.backfill_default_watchlist_scope(int(default.id))
    db.backfill_default_watchlist_scope(int(default.id))

    backfilled_sources, backfilled_source_total = db.list_watchlist_sources(default.id, limit=50, offset=0)
    assert backfilled_source_total == 1
    assert [int(row.id) for row in backfilled_sources] == [int(source.id)]
    assert db.get_job(int(job.id)).watchlist_id == default.id


def test_watchlist_lifecycle_preserves_memberships_and_job_scope(tmp_path):
    db = _make_db(tmp_path)
    watchlist = db.create_watchlist(
        name="Healthcare Ransomware",
        description="Track hospital impact",
        objective="Find ransomware reports affecting hospitals in Germany",
        domain="cti_osint",
        priority="high",
        tags=["ransomware", "healthcare"],
    )
    source = _create_source(db, label="cti", watchlist_id=int(watchlist.id))
    job = _create_job(db, label="cti", source_id=int(source.id), watchlist_id=int(watchlist.id))

    active_watchlists, total = db.list_watchlists(limit=50, offset=0)
    assert total == 1
    assert [int(row.id) for row in active_watchlists] == [int(watchlist.id)]
    assert json.loads(db.get_watchlist(int(watchlist.id)).tags_json or "[]") == [
        "ransomware",
        "healthcare",
    ]

    archived = db.update_watchlist(int(watchlist.id), {"status": "archived"})
    assert archived.status == "archived"
    assert archived.archived_at

    deleted, restore_expires_at = db.delete_watchlist(int(watchlist.id), restore_window_seconds=60)
    assert deleted is True
    assert restore_expires_at
    assert db.list_watchlists(limit=50, offset=0)[1] == 0
    with pytest.raises(KeyError):
        db.get_watchlist(int(watchlist.id))

    restored = db.restore_watchlist(int(watchlist.id))
    assert restored.id == watchlist.id
    assert restored.deleted_at is None
    assert db.list_watchlist_sources(int(restored.id), limit=50, offset=0)[1] == 1
    assert db.get_job(int(job.id)).watchlist_id == restored.id


def test_source_url_uniqueness_is_unchanged_but_membership_can_be_reused(tmp_path):
    db = _make_db(tmp_path)
    first = db.create_watchlist(name="First", domain="general")
    second = db.create_watchlist(name="Second", domain="news")

    source_a = db.create_source(
        name="Shared Feed",
        url="https://example.com/shared/rss.xml",
        source_type="rss",
        watchlist_id=int(first.id),
    )
    source_b = db.create_source(
        name="Shared Feed Duplicate",
        url="https://example.com/shared/rss.xml",
        source_type="rss",
        watchlist_id=int(second.id),
    )

    assert source_b.id == source_a.id
    first_sources, first_total = db.list_watchlist_sources(int(first.id), limit=50, offset=0)
    second_sources, second_total = db.list_watchlist_sources(int(second.id), limit=50, offset=0)
    assert first_total == 1
    assert second_total == 1
    assert int(first_sources[0].id) == int(source_a.id)
    assert int(second_sources[0].id) == int(source_a.id)


def test_watchlist_scoped_list_helpers_filter_sources_jobs_runs_and_items(tmp_path):
    db = _make_db(tmp_path)
    cti_watchlist = db.create_watchlist(name="CTI", domain="cti_osint")
    news_watchlist = db.create_watchlist(name="News", domain="news")

    cti_source = _create_source(db, label="cti", watchlist_id=int(cti_watchlist.id))
    cti_job = _create_job(db, label="cti", source_id=int(cti_source.id), watchlist_id=int(cti_watchlist.id))
    cti_run, cti_item = _create_run_item(
        db,
        label="cti",
        job_id=int(cti_job.id),
        source_id=int(cti_source.id),
    )

    news_source = _create_source(db, label="news", watchlist_id=int(news_watchlist.id))
    news_job = _create_job(db, label="news", source_id=int(news_source.id), watchlist_id=int(news_watchlist.id))
    news_run, news_item = _create_run_item(
        db,
        label="news",
        job_id=int(news_job.id),
        source_id=int(news_source.id),
    )

    cti_sources, _ = db.list_sources(None, None, 50, 0, watchlist_id=int(cti_watchlist.id))
    cti_jobs, _ = db.list_jobs(None, 50, 0, watchlist_id=int(cti_watchlist.id))
    cti_runs, _ = db.list_runs(None, 50, 0, watchlist_id=int(cti_watchlist.id))
    cti_items, _ = db.list_items(watchlist_id=int(cti_watchlist.id), limit=50, offset=0)

    assert [int(row.id) for row in cti_sources] == [int(cti_source.id)]
    assert [int(row.id) for row in cti_jobs] == [int(cti_job.id)]
    assert [int(row.id) for row in cti_runs] == [int(cti_run.id)]
    assert [int(row.id) for row in cti_items] == [int(cti_item.id)]
    assert int(news_run.id) not in [int(row.id) for row in cti_runs]
    assert int(news_item.id) not in [int(row.id) for row in cti_items]


def test_schema_contains_watchlist_tables_membership_and_job_scope(tmp_path):
    db = _make_db(tmp_path)

    watchlist_columns = {row["name"] for row in db.backend.get_table_info("watchlists")}
    membership_columns = {row["name"] for row in db.backend.get_table_info("watchlist_sources")}
    job_columns = {row["name"] for row in db.backend.get_table_info("scrape_jobs")}

    assert {
        "id",
        "user_id",
        "name",
        "objective",
        "domain",
        "status",
        "priority",
        "tags_json",
        "archived_at",
        "deleted_at",
        "restore_expires_at",
    }.issubset(watchlist_columns)
    assert {"watchlist_id", "source_id", "created_at"}.issubset(membership_columns)
    assert "watchlist_id" in job_columns


class _CapturingPostgresBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self) -> None:
        self.ddl = ""
        self.executed: list[str] = []

    def create_tables(self, ddl: str) -> None:
        self.ddl = ddl

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append(query)

    def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        if table_name == "scrape_jobs":
            return [
                {"name": "wf_schedule_id"},
                {"name": "job_filters_json"},
                {"name": "watchlist_id"},
            ]
        if table_name == "sources":
            return [
                {"name": "defer_until"},
                {"name": "consec_not_modified"},
                {"name": "consec_errors"},
            ]
        if table_name == "scrape_run_items":
            return [{"name": "source_id"}]
        if table_name == "scraped_items":
            return [{"name": "content"}, {"name": "queued_for_briefing"}]
        return []


def test_postgres_schema_text_includes_watchlist_container_contract():
    backend = _CapturingPostgresBackend()

    WatchlistsDatabase(user_id=123, backend=backend)  # type: ignore[arg-type]

    assert "CREATE TABLE IF NOT EXISTS watchlists" in backend.ddl
    assert "id BIGSERIAL PRIMARY KEY" in backend.ddl
    assert "CREATE TABLE IF NOT EXISTS watchlist_sources" in backend.ddl
    assert "watchlist_id BIGINT" in backend.ddl
    assert any("CREATE INDEX IF NOT EXISTS idx_jobs_user_watchlist" in query for query in backend.executed)
