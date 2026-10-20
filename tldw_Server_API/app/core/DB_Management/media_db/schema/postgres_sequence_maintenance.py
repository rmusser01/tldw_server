"""PostgreSQL sequence maintenance helper."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.postgres_sequence_maintenance import (
    PostgresSequenceMaintenanceDB,
    sync_owned_postgres_sequences,
)

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


def sync_postgres_sequences(
    db: PostgresSequenceMaintenanceDB,
    conn: Any,
) -> None:
    """Advance only Media serials, retaining the shared allocation/rollback contract."""
    sync_owned_postgres_sequences(db, conn, _MEDIA_SEQUENCE_COLUMNS)


__all__ = [
    "PostgresSequenceMaintenanceDB",
    "sync_postgres_sequences",
]
