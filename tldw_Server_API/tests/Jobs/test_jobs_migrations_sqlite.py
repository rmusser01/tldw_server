import importlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.Jobs.migrations import JOBS_SQLITE_DDL, ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.pg_migrations import JOBS_POSTGRES_DDL

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
    "playlist_preflights": {
        "preflight_id",
        "owner_user_id",
        "status",
        "job_id",
        "summary_json",
        "error_json",
        "expires_at",
    },
    "playlist_preflight_items": {
        "preflight_id",
        "owner_user_id",
        "occurrence_id",
        "ordinal",
        "normalized_source_id",
        "display_metadata_json",
    },
    "playlist_materializations": {
        "materialization_id",
        "preflight_id",
        "owner_user_id",
        "status",
        "expires_at",
    },
    "playlist_materialization_items": {
        "materialization_id",
        "owner_user_id",
        "occurrence_id",
        "ordinal",
        "normalized_source_id",
        "display_metadata_json",
    },
    "media_ingest_runs": {
        "run_id",
        "owner_user_id",
        "status",
        "processing_options_json",
        "playlist_summaries_json",
        "batch_ids_json",
        "expires_at",
    },
    "media_ingest_run_items": {
        "run_id",
        "owner_user_id",
        "occurrence_id",
        "ordinal",
        "normalized_source_id",
        "duplicate_policy",
        "metadata_patch_json",
        "state",
        "outcome",
        "job_id",
        "attempt",
        "display_metadata_json",
        "submission_queue",
        "staging_temp_dir",
    },
    "media_ingest_run_events": {
        "event_id",
        "run_id",
        "owner_user_id",
        "occurrence_id",
        "job_id",
        "state",
        "outcome",
        "attrs_json",
    },
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

PLAYLIST_JSON_COLUMNS = {
    "playlist_preflights": {"summary_json", "error_json"},
    "playlist_preflight_items": {"display_metadata_json"},
    "playlist_materialization_items": {"display_metadata_json"},
    "media_ingest_runs": {"processing_options_json", "playlist_summaries_json", "batch_ids_json"},
    "media_ingest_run_items": {"display_metadata_json", "metadata_patch_json"},
    "media_ingest_run_events": {"attrs_json"},
}


def test_sqlite_schema_has_expected_columns_and_indexes(tmp_path):

    db_path = ensure_jobs_tables(tmp_path / "jobs_mig.db")
    conn = sqlite3.connect(db_path)
    try:
        # Columns present
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for expected in [
            "completion_token",
            "failure_timeline",
            "request_id",
            "trace_id",
            "progress_percent",
            "progress_message",
            "error_code",
            "error_class",
            "error_stack",
        ]:
            assert expected in cols
        # Archive table exists
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs_archive'").fetchone()
        assert row and row[0] == "jobs_archive"
        # Partial unique index for idempotency exists
        idx = [r[1] for r in conn.execute("PRAGMA index_list('jobs')").fetchall()]
        assert any("idx_jobs_idempotent" in x for x in idx)
        # job_counters exists
        row2 = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_counters'").fetchone()
        assert row2 and row2[0] == "job_counters"
        # Idempotent ensure re-runs without error
        ensure_jobs_tables(Path(db_path))
    finally:
        conn.close()


def test_sqlite_schema_has_playlist_ingest_tables(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_ingest.db")
    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}

    assert tables >= EXPECTED_PLAYLIST_TABLES


def test_sqlite_forward_migration_adds_playlist_submission_authority_columns(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_upgrade.db")
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(media_ingest_run_items)")}
        for column in ("submission_queue", "staging_temp_dir"):
            if column in columns:
                conn.execute(f"ALTER TABLE media_ingest_run_items DROP COLUMN {column}")

    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(media_ingest_run_items)")}
    assert {"submission_queue", "staging_temp_dir"} <= columns


def test_sqlite_playlist_ingest_catalog_has_required_columns_indexes_and_uniques(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_ingest_catalog.db")

    with sqlite3.connect(db_path) as conn:
        for table, expected_columns in EXPECTED_PLAYLIST_COLUMNS.items():
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
            assert columns >= expected_columns

        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}
        assert indexes >= EXPECTED_PLAYLIST_INDEXES
        assert indexes.isdisjoint(REDUNDANT_PLAYLIST_INDEXES)

        expected_uniques = {
            "playlist_preflight_items": {
                ("occurrence_id",),
                ("preflight_id", "ordinal"),
            },
            "playlist_materialization_items": {
                ("materialization_id", "ordinal"),
                ("materialization_id", "occurrence_id"),
            },
            "media_ingest_run_items": {
                ("run_id", "ordinal"),
                ("run_id", "occurrence_id"),
                ("run_id", "occurrence_id", "attempt"),
            },
        }
        for table, expected in expected_uniques.items():
            unique_columns = {
                tuple(row[2] for row in conn.execute(f"PRAGMA index_info('{index_row[1]}')").fetchall())
                for index_row in conn.execute(f"PRAGMA index_list('{table}')").fetchall()
                if index_row[2]
            }
            assert unique_columns >= expected


@pytest.mark.parametrize(
    ("insert_sql", "values"),
    [
        pytest.param(
            """
            INSERT INTO playlist_preflights (
                preflight_id, owner_user_id, status, source_url, source_kind, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (None, "owner-1", "pending", "https://example.test/playlist", "youtube", "2099-01-01"),
            id="preflight-id",
        ),
        pytest.param(
            """
            INSERT INTO playlist_materializations (
                materialization_id, preflight_id, owner_user_id, status, expires_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (None, "preflight-1", "owner-1", "ready", "2099-01-01"),
            id="materialization-id",
        ),
        pytest.param(
            """
            INSERT INTO media_ingest_runs (
                run_id, owner_user_id, status, expires_at
            ) VALUES (?, ?, ?, ?)
            """,
            (None, "owner-1", "staged", "2099-01-01"),
            id="run-id",
        ),
    ],
)
def test_sqlite_playlist_ingest_resource_ids_reject_null(tmp_path, insert_sql, values):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_ingest_not_null.db")

    with sqlite3.connect(db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert_sql, values)


@pytest.mark.parametrize(
    ("table", "json_column", "required_columns", "required_values"),
    [
        pytest.param(
            "playlist_preflights",
            column,
            ("preflight_id", "owner_user_id", "status", "source_url", "source_kind", "expires_at"),
            ("preflight-1", "owner-1", "pending", "https://example.test/playlist", "youtube", "2099-01-01"),
            id=f"playlist-preflights-{column}",
        )
        for column in ("summary_json", "error_json")
    ]
    + [
        pytest.param(
            "playlist_preflight_items",
            "display_metadata_json",
            (
                "preflight_id",
                "owner_user_id",
                "occurrence_id",
                "ordinal",
                "occurrence_index_for_source",
                "source_kind",
                "availability",
                "duplicate_status",
            ),
            ("preflight-1", "owner-1", "occ-1", 1, 1, "youtube", "available", "new"),
            id="playlist-preflight-items-display-metadata",
        ),
        pytest.param(
            "playlist_materialization_items",
            "display_metadata_json",
            ("materialization_id", "owner_user_id", "occurrence_id", "ordinal", "source_url", "source_kind"),
            ("materialization-1", "owner-1", "occ-1", 1, "https://example.test/video", "youtube"),
            id="playlist-materialization-items-display-metadata",
        ),
    ]
    + [
        pytest.param(
            "media_ingest_runs",
            column,
            ("run_id", "owner_user_id", "status", "expires_at"),
            ("run-1", "owner-1", "staged", "2099-01-01"),
            id=f"media-ingest-runs-{column}",
        )
        for column in ("processing_options_json", "playlist_summaries_json", "batch_ids_json")
    ]
    + [
        pytest.param(
            "media_ingest_run_items",
            column,
            ("run_id", "owner_user_id", "occurrence_id", "ordinal", "input_kind", "state"),
            ("run-1", "owner-1", "occ-1", 1, "direct_url", "running"),
            id=f"media-ingest-run-items-{column}",
        )
        for column in ("display_metadata_json", "metadata_patch_json")
    ]
    + [
        pytest.param(
            "media_ingest_run_events",
            "attrs_json",
            ("run_id", "owner_user_id", "event_type"),
            ("run-1", "owner-1", "progress"),
            id="media-ingest-run-events-attrs",
        )
    ],
)
def test_sqlite_playlist_ingest_json_columns_reject_invalid_json(
    tmp_path,
    table,
    json_column,
    required_columns,
    required_values,
):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_ingest_json.db")
    columns = (*required_columns, json_column)
    placeholders = ", ".join("?" for _ in columns)

    with sqlite3.connect(db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            (*required_values, "not-json"),
        )


@pytest.mark.parametrize(
    ("duplicate_policy", "state", "outcome"),
    [
        pytest.param("reuse", "running", None, id="invalid-policy"),
        pytest.param(None, "file_reattach_required", None, id="client-only-state"),
        pytest.param(None, "terminal", "unknown", id="invalid-outcome"),
        pytest.param(None, "terminal", None, id="terminal-without-outcome"),
        pytest.param(None, "running", "completed", id="nonterminal-with-outcome"),
    ],
)
def test_sqlite_run_items_reject_invalid_domains_and_state_outcome(
    tmp_path,
    duplicate_policy,
    state,
    outcome,
):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_ingest_domains.db")

    with sqlite3.connect(db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO media_ingest_run_items (
                run_id, owner_user_id, occurrence_id, ordinal, input_kind,
                duplicate_policy, state, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("run-1", "owner-1", "occ-1", 1, "direct_url", duplicate_policy, state, outcome),
        )


@pytest.mark.parametrize(
    ("state", "outcome"),
    [
        pytest.param("terminal", None, id="terminal-without-outcome"),
        pytest.param("running", "completed", id="nonterminal-with-outcome"),
        pytest.param(None, "completed", id="outcome-without-state"),
        pytest.param("file_reattach_required", None, id="client-only-state"),
    ],
)
def test_sqlite_run_events_reject_invalid_state_outcome(tmp_path, state, outcome):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_ingest_event_domains.db")

    with sqlite3.connect(db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO media_ingest_run_events (
                run_id, owner_user_id, event_type, state, outcome
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("run-1", "owner-1", "progress", state, outcome),
        )


def test_sqlite_run_events_allow_partial_and_consistent_state_evidence(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "jobs_playlist_ingest_valid_events.db")

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO media_ingest_run_events (
                run_id, owner_user_id, event_type, state, outcome
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("run-1", "owner-1", "summary", None, None),
                ("run-1", "owner-1", "progress", "running", None),
                ("run-1", "owner-1", "completed", "terminal", "completed"),
            ],
        )

        count = conn.execute("SELECT COUNT(*) FROM media_ingest_run_events").fetchone()[0]

    assert count == 3


def test_playlist_ingest_sqlite_postgres_ddl_has_focused_static_parity():
    sqlite_ddl = " ".join(JOBS_SQLITE_DDL.split()).replace("( ", "(").replace(" )", ")")
    postgres_ddl = " ".join(JOBS_POSTGRES_DDL.split()).replace("( ", "(").replace(" )", ")")

    for table in EXPECTED_PLAYLIST_TABLES:
        declaration = f"CREATE TABLE IF NOT EXISTS {table}"
        assert declaration in sqlite_ddl
        assert declaration in postgres_ddl

    for value in {
        "skip",
        "include_existing",
        "update_metadata_only",
        "overwrite",
        "staged",
        "preparing",
        "awaiting_upload",
        "submit_pending",
        "queued",
        "running",
        "cancellation_requested",
        "status_unavailable",
        "terminal",
        "completed",
        "included_existing",
        "metadata_updated",
        "skipped_existing",
        "submit_failed",
        "processing_failed",
        "metadata_update_failed",
        "cancelled",
    }:
        assert f"'{value}'" in sqlite_ddl
        assert f"'{value}'" in postgres_ddl

    for ddl in (sqlite_ddl, postgres_ddl):
        assert "CHECK (duplicate_policy IS NULL OR duplicate_policy IN" in ddl
        assert "state TEXT NOT NULL CHECK (state IN" in ddl
        assert "outcome TEXT CHECK (outcome IS NULL OR outcome IN" in ddl
        assert (
            "CHECK ((state = 'terminal' AND outcome IS NOT NULL) " "OR (state <> 'terminal' AND outcome IS NULL))"
        ) in ddl
        assert (
            "CHECK ((state IS NULL AND outcome IS NULL) "
            "OR (state IS NOT NULL AND state = 'terminal' AND outcome IS NOT NULL) "
            "OR (state IS NOT NULL AND state <> 'terminal' AND outcome IS NULL))"
        ) in ddl

    for table, columns in PLAYLIST_JSON_COLUMNS.items():
        assert table in sqlite_ddl
        assert table in postgres_ddl
        for column in columns:
            assert f"CHECK ({column} IS NULL OR json_valid({column}))" in sqlite_ddl
            assert f"{column} JSONB" in postgres_ddl

    for index_name in REDUNDANT_PLAYLIST_INDEXES:
        assert index_name not in sqlite_ddl
        assert index_name not in postgres_ddl


def test_jobs_migrations_uses_shared_sqlite_policy_helper(tmp_path):
    jobs_migrations = importlib.import_module("tldw_Server_API.app.core.Jobs.migrations")
    calls: list[dict[str, object]] = []

    def fake_configure(conn, **kwargs):
        calls.append(kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sqlite_policy, "configure_sqlite_connection", fake_configure)
        jobs_migrations = importlib.reload(jobs_migrations)
        jobs_migrations.ensure_jobs_tables(tmp_path / "jobs_helper.db")

    importlib.reload(jobs_migrations)

    assert calls == [
        {
            "use_wal": True,
            "synchronous": "NORMAL",
            "busy_timeout_ms": 5000,
            "foreign_keys": False,
            "temp_store": None,
        }
    ]


def test_ensure_jobs_tables_sanitizes_schema_failure_log(tmp_path, monkeypatch):
    jobs_migrations = importlib.import_module("tldw_Server_API.app.core.Jobs.migrations")
    secret = "sk_jobsMigrationSecret1234567890"
    db_path = tmp_path / secret / "jobs.db"
    fake_logger = MagicMock()

    def fail_connect(path):
        raise sqlite3.OperationalError(f"unable to open database file {path} token={secret}")

    monkeypatch.setattr(jobs_migrations.sqlite3, "connect", fail_connect)
    monkeypatch.setattr(jobs_migrations, "logger", fake_logger)

    result = jobs_migrations.ensure_jobs_tables(db_path)

    assert result == db_path
    fake_logger.warning.assert_called_once()
    warning_args = fake_logger.warning.call_args.args
    rendered = " ".join(str(arg) for arg in warning_args)
    assert "Failed to ensure Jobs schema" in rendered
    assert str(db_path) not in rendered
    assert secret not in rendered
