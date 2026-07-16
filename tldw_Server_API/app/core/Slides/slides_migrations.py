"""Authoritative transactional migrations for the per-user Slides database."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any


SLIDES_SCHEMA_VERSION = 2


class SlidesMigrationError(RuntimeError):
    """Raised when a Slides schema cannot be migrated safely."""


_PRESENTATION_V2_COLUMNS: tuple[tuple[str, str], ...] = (
    (
        "content_kind",
        "TEXT NOT NULL DEFAULT 'structured_slides' "
        "CHECK(content_kind IN ('structured_slides', 'standalone_html'))",
    ),
    ("html_document", "TEXT NULL"),
    ("html_sha256", "TEXT NULL"),
    ("html_bytes", "INTEGER NULL"),
    ("html_slide_count", "INTEGER NULL"),
    ("generation_job_uuid", "TEXT NULL"),
    ("generation_provenance_json", "TEXT NULL"),
)


_RECEIPTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slides_generation_receipts (
    id TEXT PRIMARY KEY
        CHECK(length(id) BETWEEN 1 AND 64),
    owner_user_id TEXT NOT NULL
        CHECK(length(owner_user_id) BETWEEN 1 AND 256),
    digest_key_id TEXT NOT NULL
        CHECK(length(digest_key_id) BETWEEN 1 AND 32),
    idempotency_key_hmac_sha256 TEXT UNIQUE NOT NULL
        CHECK(length(idempotency_key_hmac_sha256) = 64),
    jobs_idempotency_key TEXT UNIQUE NOT NULL
        CHECK(length(jobs_idempotency_key) BETWEEN 1 AND 256),
    client_request_hmac_sha256 TEXT NOT NULL
        CHECK(length(client_request_hmac_sha256) = 64),
    execution_hmac_sha256 TEXT NOT NULL
        CHECK(length(execution_hmac_sha256) = 64),
    job_id INTEGER NULL
        CHECK(job_id IS NULL OR job_id >= 0),
    job_uuid TEXT NULL
        CHECK(job_uuid IS NULL OR length(job_uuid) BETWEEN 1 AND 64),
    presentation_id TEXT NULL REFERENCES presentations(id) ON DELETE SET NULL,
    receipt_status TEXT NOT NULL
        CHECK(receipt_status IN (
            'claimed', 'queued', 'running', 'completed', 'failed', 'cancelled'
        )),
    error_code TEXT NULL
        CHECK(error_code IS NULL OR length(error_code) <= 128),
    error_message TEXT NULL
        CHECK(error_message IS NULL OR length(error_message) <= 1024),
    created_at TEXT NOT NULL
        CHECK(length(created_at) BETWEEN 1 AND 64),
    updated_at TEXT NOT NULL
        CHECK(length(updated_at) BETWEEN 1 AND 64),
    expires_at TEXT NULL
        CHECK(expires_at IS NULL OR length(expires_at) BETWEEN 1 AND 64)
)
"""


_INPUTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slides_generation_inputs (
    receipt_id TEXT PRIMARY KEY
        REFERENCES slides_generation_receipts(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL
        CHECK(length(source_kind) BETWEEN 1 AND 64),
    source_text TEXT NOT NULL,
    source_hmac_sha256 TEXT NOT NULL
        CHECK(length(source_hmac_sha256) = 64),
    source_bytes INTEGER NOT NULL
        CHECK(source_bytes >= 0),
    provenance_json TEXT NOT NULL
        CHECK(length(CAST(provenance_json AS BLOB)) BETWEEN 2 AND 4096),
    html_options_json TEXT NOT NULL
        CHECK(length(CAST(html_options_json AS BLOB)) BETWEEN 2 AND 4096),
    provider TEXT NOT NULL
        CHECK(length(provider) BETWEEN 1 AND 128),
    model TEXT NOT NULL
        CHECK(length(model) BETWEEN 1 AND 256),
    adapter_id TEXT NOT NULL
        CHECK(length(adapter_id) BETWEEN 1 AND 128),
    endpoint_identity TEXT NOT NULL
        CHECK(length(endpoint_identity) BETWEEN 1 AND 2048),
    system_prompt TEXT NOT NULL
        CHECK(length(CAST(system_prompt AS BLOB)) BETWEEN 1 AND 131072),
    prompt_sha256 TEXT NOT NULL
        CHECK(length(prompt_sha256) = 64),
    prompt_contract_version TEXT NOT NULL
        CHECK(length(prompt_contract_version) BETWEEN 1 AND 128),
    input_expires_at TEXT NOT NULL
        CHECK(length(input_expires_at) BETWEEN 1 AND 64),
    created_at TEXT NOT NULL
        CHECK(length(created_at) BETWEEN 1 AND 64)
)
"""


_GENERATION_JOB_UUID_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_presentations_generation_job_uuid
ON presentations(generation_job_uuid)
WHERE generation_job_uuid IS NOT NULL
"""


def _execute_migration_statement(
    conn: sqlite3.Connection,
    statement: str,
    parameters: Sequence[Any] = (),
) -> sqlite3.Cursor:
    """Execute one migration statement."""
    return conn.execute(statement, parameters)


def migrate_slides_schema(conn: sqlite3.Connection) -> None:
    """Migrate an initialized Slides database to schema v2 atomically."""
    if conn.in_transaction:
        raise SlidesMigrationError("Slides migration requires an idle connection")

    conn.execute("BEGIN IMMEDIATE")
    try:
        _execute_migration_statement(
            conn,
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY NOT NULL)",
        )
        version_rows = conn.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        versions = [int(row[0]) for row in version_rows]
        if any(version < 0 or version > SLIDES_SCHEMA_VERSION for version in versions):
            raise SlidesMigrationError(
                f"Unsupported Slides schema versions: {versions!r}"
            )

        table_row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'presentations'"
        ).fetchone()
        if table_row is None:
            raise SlidesMigrationError("presentations table is missing")

        columns = {
            str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
            for row in conn.execute("PRAGMA table_info(presentations)").fetchall()
        }
        for column_name, column_ddl in _PRESENTATION_V2_COLUMNS:
            if column_name not in columns:
                _execute_migration_statement(
                    conn,
                    f"ALTER TABLE presentations ADD COLUMN {column_name} {column_ddl}",
                )

        legacy_schema = not versions or max(versions) < SLIDES_SCHEMA_VERSION
        if legacy_schema:
            fts_table = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'presentations_fts'"
            ).fetchone()
            if fts_table is not None:
                _execute_migration_statement(
                    conn,
                    "INSERT INTO presentations_fts(presentations_fts) "
                    "VALUES ('rebuild')",
                )
            _execute_migration_statement(
                conn,
                """
                UPDATE presentations
                SET content_kind = 'structured_slides',
                    html_document = NULL,
                    html_sha256 = NULL,
                    html_bytes = NULL,
                    html_slide_count = NULL,
                    generation_job_uuid = NULL,
                    generation_provenance_json = NULL
                """,
            )

        _execute_migration_statement(conn, _RECEIPTS_TABLE_SQL)
        _execute_migration_statement(conn, _INPUTS_TABLE_SQL)
        _execute_migration_statement(conn, _GENERATION_JOB_UUID_INDEX_SQL)
        _execute_migration_statement(conn, "DELETE FROM schema_version")
        _execute_migration_statement(
            conn,
            "INSERT INTO schema_version (version) VALUES (?)",
            (SLIDES_SCHEMA_VERSION,),
        )

        normalized = conn.execute(
            "SELECT version FROM schema_version"
        ).fetchall()
        if len(normalized) != 1 or int(normalized[0][0]) != SLIDES_SCHEMA_VERSION:
            raise SlidesMigrationError("Failed to normalize Slides schema version")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
