"""PostgreSQL sequence maintenance helper."""

from __future__ import annotations

from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.backends.postgres_sequences import (
    PostgresSequenceBackend,
    advance_postgres_serial,
)

# Owned serial columns from Media's core, claims/collections, data-table and
# email schema DDL. Shared PostgreSQL schemas also contain other modules' tables.
# The real Media-only schema inventory test guards this list against omissions.
_MEDIA_SEQUENCE_COLUMNS = frozenset({
    ("chunkingtemplates", "id"),
    ("claim_clusters", "id"),
    ("claims", "id"),
    ("claims_monitoring_alerts", "id"),
    ("claims_monitoring_config", "id"),
    ("claims_monitoring_events", "id"),
    ("claims_monitoring_health", "id"),
    ("claims_monitoring_settings", "id"),
    ("claims_notifications", "id"),
    ("claims_review_extractor_metrics_daily", "id"),
    ("claims_review_log", "id"),
    ("claims_review_rules", "id"),
    ("collection_tags", "id"),
    ("content_items", "id"),
    ("data_table_columns", "id"),
    ("data_table_rows", "id"),
    ("data_table_sources", "id"),
    ("data_tables", "id"),
    ("documentstructureindex", "id"),
    ("documentversions", "id"),
    ("email_attachments", "id"),
    ("email_backfill_state", "id"),
    ("email_labels", "id"),
    ("email_messages", "id"),
    ("email_participants", "id"),
    ("email_sources", "id"),
    ("email_sync_state", "id"),
    ("keywords", "id"),
    ("media", "id"),
    ("mediachunks", "id"),
    ("mediafiles", "id"),
    ("mediakeywords", "id"),
    ("output_templates", "id"),
    ("reading_highlights", "id"),
    ("sync_log", "change_id"),
    ("transcripts", "id"),
    ("tts_history", "id"),
    ("unvectorizedmediachunks", "id"),
    ("visualdocuments", "id"),
})


class PostgresSequenceMaintenanceDB(Protocol):
    """Protocol for DB objects exposing a PostgreSQL backend."""

    backend: PostgresSequenceBackend


def sync_postgres_sequences(
    db: PostgresSequenceMaintenanceDB,
    conn: Any,
) -> None:
    """Advance Media serials without reusing allocated IDs hidden by tenant RLS.

    The schema owner calls this inside its existing migration transaction.
    Only a temporary lock savepoint is rolled back; caller work is untouched.
    PostgreSQL's nontransactional setval deliberately retains forward advances.
    """

    backend = db.backend
    sequence_rows = backend.execute(
        """
        SELECT
            sequence_ns.nspname AS sequence_schema,
            seq.relname AS sequence_name,
            tab.relname AS table_name,
            col.attname AS column_name
        FROM pg_class seq
        JOIN pg_namespace sequence_ns ON sequence_ns.oid = seq.relnamespace
        JOIN pg_depend dep ON dep.objid = seq.oid AND dep.deptype = 'a'
        JOIN pg_class tab ON tab.oid = dep.refobjid
        JOIN pg_namespace tab_ns ON tab_ns.oid = tab.relnamespace
        JOIN pg_attribute col ON col.attrelid = tab.oid AND col.attnum = dep.refobjsubid
        WHERE seq.relkind = 'S' AND tab_ns.nspname = 'public'
        ORDER BY sequence_ns.nspname, seq.relname;
        """,
        connection=conn,
    )

    for row in sequence_rows.rows:
        table_name = row.get("table_name")
        column_name = row.get("column_name")
        sequence_schema = row.get("sequence_schema", "public")
        sequence_name = row.get("sequence_name")

        if not table_name or not column_name or not sequence_name:
            continue
        if (table_name, column_name) not in _MEDIA_SEQUENCE_COLUMNS:
            continue

        advance_postgres_serial(
            backend,
            conn,
            table_name=table_name,
            column_name=column_name,
            sequence_schema=sequence_schema,
            sequence_name=sequence_name,
        )


__all__ = [
    "PostgresSequenceMaintenanceDB",
    "sync_postgres_sequences",
]
