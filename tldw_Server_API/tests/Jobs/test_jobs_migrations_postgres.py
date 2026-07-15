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

EXPECTED_PLAYLIST_COLUMNS = {
    "playlist_preflights": {"preflight_id", "owner_user_id", "status", "job_id", "expires_at"},
    "playlist_preflight_items": {"preflight_id", "occurrence_id", "ordinal", "normalized_source_id"},
    "playlist_materializations": {"materialization_id", "preflight_id", "owner_user_id", "status", "expires_at"},
    "playlist_materialization_items": {"materialization_id", "occurrence_id", "ordinal", "normalized_source_id"},
    "media_ingest_runs": {
        "run_id",
        "owner_user_id",
        "client_request_id",
        "request_fingerprint",
        "initialization_token",
        "initialization_expires_at",
        "status",
        "expires_at",
    },
    "media_ingest_run_items": {
        "run_id",
        "occurrence_id",
        "ordinal",
        "normalized_source_id",
        "duplicate_policy",
        "state",
        "outcome",
        "job_id",
        "attempt",
        "submission_queue",
        "staging_temp_dir",
        "submission_lease_token",
        "submission_lease_expires_at",
        "submission_lease_generation",
    },
    "media_ingest_run_events": {"event_id", "run_id", "occurrence_id", "job_id", "state", "outcome"},
}

EXPECTED_JSONB_COLUMNS = {
    "playlist_preflights": {"summary_json", "error_json"},
    "playlist_preflight_items": {"display_metadata_json"},
    "playlist_materialization_items": {"display_metadata_json"},
    "media_ingest_runs": {"processing_options_json", "playlist_summaries_json", "batch_ids_json"},
    "media_ingest_run_items": {"display_metadata_json", "metadata_patch_json"},
    "media_ingest_run_events": {"attrs_json"},
}

EXPECTED_PLAYLIST_INDEXES = {
    "idx_playlist_preflights_owner_status",
    "idx_playlist_preflights_job",
    "idx_playlist_preflights_expiry",
    "idx_playlist_preflight_items_owner_source",
    "idx_playlist_materializations_owner_status",
    "idx_playlist_materializations_expiry",
    "idx_playlist_materialization_items_owner_source",
    "idx_playlist_materialization_items_occurrence",
    "idx_media_ingest_runs_owner_status",
    "idx_media_ingest_runs_expiry",
    "idx_media_ingest_runs_owner_client_request",
    "idx_media_ingest_run_items_owner_state",
    "idx_media_ingest_run_items_source",
    "idx_media_ingest_run_items_job",
    "idx_media_ingest_run_events_owner_run_event",
    "idx_media_ingest_run_events_occurrence",
    "idx_media_ingest_run_events_job",
}

REDUNDANT_PLAYLIST_INDEXES = {
    "idx_playlist_preflight_items_preflight_ordinal",
    "idx_media_ingest_run_items_run_ordinal",
    "idx_media_ingest_run_items_attempt",
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
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='jobs' AND column_name='progress_message'"
            )
            row = cur.fetchone()
            assert row is not None
            # idx_jobs_acquire_order partial index exists and is queued-only
            cur.execute(
                """
                SELECT indexname, indexdef FROM pg_indexes
                WHERE schemaname = current_schema() AND tablename = 'jobs' AND indexname = 'idx_jobs_acquire_order'
            """
            )
            row2 = cur.fetchone()
            assert row2 is not None
            assert "status = 'queued'" in (row2[1] or "")


def test_pg_forward_migration_adds_playlist_submission_authority_columns(jobs_pg_dsn):
    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE media_ingest_run_items DROP COLUMN IF EXISTS submission_queue")
        cur.execute("ALTER TABLE media_ingest_run_items DROP COLUMN IF EXISTS staging_temp_dir")
        cur.execute("ALTER TABLE media_ingest_run_items DROP COLUMN IF EXISTS submission_lease_token")
        cur.execute("ALTER TABLE media_ingest_run_items DROP COLUMN IF EXISTS submission_lease_expires_at")
        cur.execute("ALTER TABLE media_ingest_run_items DROP COLUMN IF EXISTS submission_lease_generation")

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'media_ingest_run_items'
              AND column_name = ANY(%s)
            """,
            (
                [
                    "submission_queue",
                    "staging_temp_dir",
                    "submission_lease_token",
                    "submission_lease_expires_at",
                    "submission_lease_generation",
                ],
            ),
        )
        columns = {row[0] for row in cur.fetchall()}
    assert columns == {
        "submission_queue",
        "staging_temp_dir",
        "submission_lease_token",
        "submission_lease_expires_at",
        "submission_lease_generation",
    }


def test_pg_forward_migration_adds_playlist_run_initialization_authority(jobs_pg_dsn):
    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_media_ingest_runs_owner_client_request")
        cur.execute("ALTER TABLE media_ingest_runs DROP COLUMN IF EXISTS client_request_id")
        cur.execute("ALTER TABLE media_ingest_runs DROP COLUMN IF EXISTS request_fingerprint")
        cur.execute("ALTER TABLE media_ingest_runs DROP COLUMN IF EXISTS initialization_token")
        cur.execute("ALTER TABLE media_ingest_runs DROP COLUMN IF EXISTS initialization_expires_at")

    ensure_jobs_tables_pg(jobs_pg_dsn)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'media_ingest_runs'
              AND column_name = ANY(%s)
            """,
            (
                [
                    "client_request_id",
                    "request_fingerprint",
                    "initialization_token",
                    "initialization_expires_at",
                ],
            ),
        )
        columns = {row[0] for row in cur.fetchall()}
        cur.execute(
            """
            SELECT indexdef FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'media_ingest_runs'
              AND indexname = 'idx_media_ingest_runs_owner_client_request'
            """
        )
        index_row = cur.fetchone()
    assert columns == {
        "client_request_id",
        "request_fingerprint",
        "initialization_token",
        "initialization_expires_at",
    }
    assert index_row is not None
    assert "UNIQUE" in index_row[0]


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

        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s)
            """,
            (list(EXPECTED_PLAYLIST_TABLES),),
        )
        catalog_columns = cur.fetchall()
        columns_by_table = {
            table: {column for row_table, column, _ in catalog_columns if row_table == table}
            for table in EXPECTED_PLAYLIST_TABLES
        }
        jsonb_by_table = {
            table: {
                column
                for row_table, column, data_type in catalog_columns
                if row_table == table and data_type == "jsonb"
            }
            for table in EXPECTED_PLAYLIST_TABLES
        }

        cur.execute(
            """
            SELECT relation.relname, pg_get_constraintdef(constraint_row.oid)
            FROM pg_constraint AS constraint_row
            JOIN pg_class AS relation ON relation.oid = constraint_row.conrelid
            WHERE relation.relname = ANY(%s)
            """,
            (list(EXPECTED_PLAYLIST_TABLES),),
        )
        constraints = {}
        for table, definition in cur.fetchall():
            constraints.setdefault(table, []).append(definition)

        cur.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = ANY(%s)
            """,
            (list(EXPECTED_PLAYLIST_TABLES),),
        )
        indexes = {row[0] for row in cur.fetchall()}

    assert tables >= EXPECTED_PLAYLIST_TABLES
    for table, expected_columns in EXPECTED_PLAYLIST_COLUMNS.items():
        assert columns_by_table[table] >= expected_columns
    for table, expected_jsonb in EXPECTED_JSONB_COLUMNS.items():
        assert jsonb_by_table[table] >= expected_jsonb

    run_item_constraints = " ".join(constraints["media_ingest_run_items"])
    event_constraints = " ".join(constraints["media_ingest_run_events"])
    duplicate_policies = {"skip", "include_existing", "update_metadata_only", "overwrite"}
    run_states = {
        "staged",
        "preparing",
        "awaiting_upload",
        "submit_pending",
        "queued",
        "running",
        "cancellation_requested",
        "status_unavailable",
        "terminal",
    }
    run_outcomes = {
        "completed",
        "included_existing",
        "metadata_updated",
        "skipped_existing",
        "submit_failed",
        "processing_failed",
        "metadata_update_failed",
        "cancelled",
    }
    for value in duplicate_policies | run_states | run_outcomes:
        assert value in run_item_constraints
    for value in run_states | run_outcomes:
        assert value in event_constraints
    assert "outcome IS NOT NULL" in run_item_constraints
    assert "outcome IS NULL" in run_item_constraints
    assert "state IS NULL" in event_constraints
    assert "outcome IS NOT NULL" in event_constraints
    assert "outcome IS NULL" in event_constraints
    assert "UNIQUE (run_id, occurrence_id, attempt)" in run_item_constraints

    assert indexes >= EXPECTED_PLAYLIST_INDEXES
    assert indexes.isdisjoint(REDUNDANT_PLAYLIST_INDEXES)
