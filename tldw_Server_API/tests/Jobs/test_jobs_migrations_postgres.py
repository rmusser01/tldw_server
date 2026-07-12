
import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg

pytestmark = [pytest.mark.pg_jobs]


EXPECTED_PLAYLIST_TABLES = {
    "playlist_preflights",
    "playlist_preflight_items",
    "playlist_materializations",
    "playlist_materialization_items",
    "media_ingest_runs",
    "media_ingest_run_items",
    "media_ingest_run_events",
}


@pytest.fixture(autouse=True)
def _setup(jobs_pg_dsn):
    return


def test_pg_forward_migration_adds_missing_columns_and_partial_indexes(jobs_pg_dsn):


    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Try to drop a new-ish column to simulate an older schema
            try:
                cur.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS progress_message")
            except Exception:
                _ = None

    # Run ensure to forward-migrate
    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn:
        with conn.cursor() as cur:
            # Column should exist now
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='jobs' AND column_name='progress_message'")
            row = cur.fetchone()
            assert row is not None
            # idx_jobs_acquire_order partial index exists and is queued-only
            cur.execute("""
                SELECT indexname, indexdef FROM pg_indexes
                WHERE schemaname = current_schema() AND tablename = 'jobs' AND indexname = 'idx_jobs_acquire_order'
            """)
            row2 = cur.fetchone()
            assert row2 is not None
            assert "status = 'queued'" in (row2[1] or "")


def test_postgres_schema_has_playlist_ingest_tables(jobs_pg_dsn):
    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = current_schema()
            """
        )
        tables = {row[0] for row in cur.fetchall()}

    assert tables >= EXPECTED_PLAYLIST_TABLES
