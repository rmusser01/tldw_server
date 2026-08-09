"""Package-owned PostgreSQL claims/collections ensure helpers."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    logger = logging.getLogger("media_db_postgres_claims_collection_structures")


class _PostgresClaimsCollectionsBackend(Protocol):
    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: object,
    ) -> object: ...

    def escape_identifier(self, name: str) -> str: ...


class PostgresClaimsCollectionsDB(Protocol):
    backend: _PostgresClaimsCollectionsBackend
    _CLAIMS_TABLE_SQL: str

    def _convert_sqlite_sql_to_postgres_statements(self, sql: str) -> list[str]: ...

    def _ensure_postgres_claims_extensions(self, conn: Any) -> None: ...


def ensure_postgres_claims_tables(db: PostgresClaimsCollectionsDB, conn: Any) -> None:
    """Ensure claims base tables exist on PostgreSQL."""

    statements = db._convert_sqlite_sql_to_postgres_statements(db._CLAIMS_TABLE_SQL)
    create_tables = [s for s in statements if s.strip().upper().startswith("CREATE TABLE")]
    other_statements = [s for s in statements if s not in create_tables]

    for stmt in create_tables:
        try:
            db.backend.execute(stmt, connection=conn)
        except BackendDatabaseError as exc:
            logger.warning(f"Could not ensure Claims base table on PostgreSQL: {exc}")

    # Claims extensions add late columns before index creation.
    db._ensure_postgres_claims_extensions(conn)

    for stmt in other_statements:
        try:
            db.backend.execute(stmt, connection=conn)
        except BackendDatabaseError as exc:
            logger.warning(
                f"Could not ensure Claims index/statement on PostgreSQL: {exc}"
            )


def ensure_postgres_collections_tables(
    db: PostgresClaimsCollectionsDB, conn: Any
) -> None:
    """Ensure collections/content item tables exist on PostgreSQL."""

    backend = db.backend

    try:
        backend.execute(
            (
                "CREATE TABLE IF NOT EXISTS output_templates ("
                "id SERIAL PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, type TEXT NOT NULL, "
                "format TEXT NOT NULL, body TEXT NOT NULL, description TEXT, is_default BOOLEAN NOT NULL DEFAULT FALSE, "
                "created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL)"
            ),
            connection=conn,
        )
        backend.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_output_templates_user_name "
            "ON output_templates(user_id, name)",
            connection=conn,
        )
        backend.execute(
            (
                "CREATE TABLE IF NOT EXISTS reading_highlights ("
                "id SERIAL PRIMARY KEY, user_id TEXT NOT NULL, item_id INTEGER NOT NULL, quote TEXT NOT NULL, "
                "start_offset INTEGER, end_offset INTEGER, color TEXT, note TEXT, created_at TIMESTAMPTZ NOT NULL, "
                "anchor_strategy TEXT NOT NULL DEFAULT 'fuzzy_quote', content_hash_ref TEXT, context_before TEXT, context_after TEXT, "
                "state TEXT NOT NULL DEFAULT 'active')"
            ),
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_highlights_user_item ON reading_highlights(user_id, item_id)",
            connection=conn,
        )
        backend.execute(
            (
                "CREATE TABLE IF NOT EXISTS collection_tags ("
                "id BIGSERIAL PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, "
                "UNIQUE (user_id, name))"
            ),
            connection=conn,
        )
        backend.execute(
            (
                "CREATE TABLE IF NOT EXISTS content_items ("
                "id BIGSERIAL PRIMARY KEY, user_id TEXT NOT NULL, origin TEXT NOT NULL, origin_type TEXT, "
                "origin_id BIGINT, url TEXT, canonical_url TEXT, domain TEXT, title TEXT, summary TEXT, notes TEXT, "
                "content_hash TEXT, word_count INTEGER, published_at TEXT, status TEXT, favorite INTEGER NOT NULL DEFAULT 0, "
                "metadata_json TEXT, media_id BIGINT, job_id BIGINT, run_id BIGINT, source_id BIGINT, read_at TEXT, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            ),
            connection=conn,
        )
        backend.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_canonical "
            "ON content_items(user_id, canonical_url) WHERE canonical_url IS NOT NULL",
            connection=conn,
        )
        backend.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_hash "
            "ON content_items(user_id, content_hash) WHERE content_hash IS NOT NULL",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_items_user_updated "
            "ON content_items(user_id, updated_at DESC)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_items_user_domain "
            "ON content_items(user_id, domain)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_items_job "
            "ON content_items(job_id)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_items_run "
            "ON content_items(run_id)",
            connection=conn,
        )
        backend.execute(
            (
                "CREATE TABLE IF NOT EXISTS content_item_tags ("
                "item_id BIGINT NOT NULL, tag_id BIGINT NOT NULL, "
                "UNIQUE (item_id, tag_id))"
            ),
            connection=conn,
        )
    except BackendDatabaseError as exc:
        logger.warning(f"Could not ensure collections tables on PostgreSQL: {exc}")


def ensure_postgres_claims_extensions(
    db: PostgresClaimsCollectionsDB, conn: Any
) -> None:
    """Ensure claims review/cluster columns and related tables exist on PostgreSQL."""

    backend = db.backend
    ident = backend.escape_identifier

    try:
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('review_status')} TEXT DEFAULT 'pending'",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('reviewer_id')} BIGINT",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('review_group')} TEXT",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('reviewed_at')} TIMESTAMPTZ",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('review_notes')} TEXT",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('review_version')} INTEGER DEFAULT 1",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('review_reason_code')} TEXT",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims')} ADD COLUMN IF NOT EXISTS {ident('claim_cluster_id')} BIGINT",
            connection=conn,
        )
        backend.execute(
            f"UPDATE {ident('claims')} SET {ident('review_status')} = 'pending' "  # nosec B608
            f"WHERE {ident('review_status')} IS NULL",
            connection=conn,
        )
        backend.execute(
            f"UPDATE {ident('claims')} SET {ident('review_version')} = 1 "  # nosec B608
            f"WHERE {ident('review_version')} IS NULL",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_status')} "
            f"ON {ident('claims')} ({ident('review_status')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_reviewer_id')} "
            f"ON {ident('claims')} ({ident('reviewer_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_group')} "
            f"ON {ident('claims')} ({ident('review_group')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_cluster_id')} "
            f"ON {ident('claims')} ({ident('claim_cluster_id')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_review_log')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "claim_id BIGINT NOT NULL, "
                "old_status TEXT, "
                "new_status TEXT, "
                "old_text TEXT, "
                "new_text TEXT, "
                "reviewer_id BIGINT, "
                "notes TEXT, "
                "reason_code TEXT, "
                "action_ip TEXT, "
                "action_user_agent TEXT, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_log_claim')} "
            f"ON {ident('claims_review_log')} ({ident('claim_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_log_reviewer')} "
            f"ON {ident('claims_review_log')} ({ident('reviewer_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_log_created')} "
            f"ON {ident('claims_review_log')} ({ident('created_at')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_review_extractor_metrics_daily')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "report_date DATE NOT NULL, "
                "extractor TEXT NOT NULL, "
                "extractor_version TEXT NOT NULL DEFAULT '', "
                "total_reviewed INTEGER NOT NULL DEFAULT 0, "
                "approved_count INTEGER NOT NULL DEFAULT 0, "
                "rejected_count INTEGER NOT NULL DEFAULT 0, "
                "flagged_count INTEGER NOT NULL DEFAULT 0, "
                "reassigned_count INTEGER NOT NULL DEFAULT 0, "
                "edited_count INTEGER NOT NULL DEFAULT 0, "
                "reason_code_counts_json TEXT, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "UNIQUE (user_id, report_date, extractor, extractor_version))"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_metrics_user')} "
            f"ON {ident('claims_review_extractor_metrics_daily')} ({ident('user_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_metrics_date')} "
            f"ON {ident('claims_review_extractor_metrics_daily')} ({ident('report_date')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_metrics_extractor')} "
            f"ON {ident('claims_review_extractor_metrics_daily')} ({ident('extractor')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_review_rules')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "priority INTEGER NOT NULL DEFAULT 0, "
                "predicate_json TEXT NOT NULL, "
                "reviewer_id BIGINT, "
                "review_group TEXT, "
                "active BOOLEAN NOT NULL DEFAULT TRUE, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_rules_user')} "
            f"ON {ident('claims_review_rules')} ({ident('user_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_review_rules_active')} "
            f"ON {ident('claims_review_rules')} ({ident('active')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_monitoring_settings')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "threshold_ratio DOUBLE PRECISION, "
                "baseline_ratio DOUBLE PRECISION, "
                "slack_webhook_url TEXT, "
                "webhook_url TEXT, "
                "email_recipients TEXT, "
                "enabled BOOLEAN NOT NULL DEFAULT TRUE, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_settings_user')} "
            f"ON {ident('claims_monitoring_settings')} ({ident('user_id')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_monitoring_alerts')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "name TEXT NOT NULL, "
                "alert_type TEXT NOT NULL, "
                "threshold_ratio DOUBLE PRECISION, "
                "baseline_ratio DOUBLE PRECISION, "
                "channels_json TEXT NOT NULL, "
                "slack_webhook_url TEXT, "
                "webhook_url TEXT, "
                "email_recipients TEXT, "
                "enabled BOOLEAN NOT NULL DEFAULT TRUE, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_alerts_user')} "
            f"ON {ident('claims_monitoring_alerts')} ({ident('user_id')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_monitoring_config')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "threshold_ratio DOUBLE PRECISION, "
                "baseline_ratio DOUBLE PRECISION, "
                "slack_webhook_url TEXT, "
                "webhook_url TEXT, "
                "email_recipients TEXT, "
                "enabled BOOLEAN NOT NULL DEFAULT TRUE, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_user')} "
            f"ON {ident('claims_monitoring_config')} ({ident('user_id')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_monitoring_events')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "event_type TEXT NOT NULL, "
                "severity TEXT, "
                "payload_json TEXT, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "delivered_at TIMESTAMPTZ)"
            ),
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims_monitoring_events')} "
            f"ADD COLUMN IF NOT EXISTS {ident('delivered_at')} TIMESTAMPTZ",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_events_user')} "
            f"ON {ident('claims_monitoring_events')} ({ident('user_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_events_user_created_id')} "
            f"ON {ident('claims_monitoring_events')} ({ident('user_id')}, {ident('created_at')}, {ident('id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_events_type')} "
            f"ON {ident('claims_monitoring_events')} ({ident('event_type')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_events_delivered')} "
            f"ON {ident('claims_monitoring_events')} ({ident('delivered_at')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_monitoring_health')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "queue_size INTEGER NOT NULL DEFAULT 0, "
                "worker_count INTEGER, "
                "last_worker_heartbeat TIMESTAMPTZ, "
                "last_processed_at TIMESTAMPTZ, "
                "last_failure_at TIMESTAMPTZ, "
                "last_failure_reason TEXT, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {ident('idx_claims_monitoring_health_user')} "
            f"ON {ident('claims_monitoring_health')} ({ident('user_id')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_analytics_exports')} ("
                "export_id TEXT PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "format TEXT NOT NULL, "
                "status TEXT NOT NULL, "
                "payload_json TEXT, "
                "payload_csv TEXT, "
                "filters_json TEXT, "
                "pagination_json TEXT, "
                "error_message TEXT, "
                "job_id BIGINT, "
                "error_code TEXT, "
                "snapshot_at TIMESTAMPTZ, "
                "snapshot_event_id BIGINT, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims_analytics_exports')} "
            f"ADD COLUMN IF NOT EXISTS {ident('job_id')} BIGINT",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims_analytics_exports')} "
            f"ADD COLUMN IF NOT EXISTS {ident('error_code')} TEXT",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims_analytics_exports')} "
            f"ADD COLUMN IF NOT EXISTS {ident('snapshot_at')} TIMESTAMPTZ",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('claims_analytics_exports')} "
            f"ADD COLUMN IF NOT EXISTS {ident('snapshot_event_id')} BIGINT",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_analytics_exports_user')} "
            f"ON {ident('claims_analytics_exports')} ({ident('user_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_analytics_exports_job_id')} "
            f"ON {ident('claims_analytics_exports')} ({ident('job_id')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claims_notifications')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "kind TEXT NOT NULL, "
                "target_user_id TEXT, "
                "target_review_group TEXT, "
                "resource_type TEXT, "
                "resource_id TEXT, "
                "payload_json TEXT NOT NULL, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "delivered_at TIMESTAMPTZ)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_notifications_user')} "
            f"ON {ident('claims_notifications')} ({ident('user_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_notifications_kind')} "
            f"ON {ident('claims_notifications')} ({ident('kind')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_notifications_target_user')} "
            f"ON {ident('claims_notifications')} ({ident('target_user_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_notifications_review_group')} "
            f"ON {ident('claims_notifications')} ({ident('target_review_group')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_notifications_resource')} "
            f"ON {ident('claims_notifications')} ({ident('resource_type')}, {ident('resource_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claims_notifications_delivered')} "
            f"ON {ident('claims_notifications')} ({ident('delivered_at')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claim_clusters')} ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "canonical_claim_text TEXT, "
                "representative_claim_id BIGINT, "
                "summary TEXT, "
                "cluster_version INTEGER NOT NULL DEFAULT 1, "
                "watchlist_count INTEGER NOT NULL DEFAULT 0, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claim_clusters_user')} "
            f"ON {ident('claim_clusters')} ({ident('user_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claim_clusters_updated')} "
            f"ON {ident('claim_clusters')} ({ident('updated_at')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claim_cluster_membership')} ("
                "cluster_id BIGINT NOT NULL, "
                "claim_id BIGINT NOT NULL, "
                "similarity_score DOUBLE PRECISION, "
                "cluster_joined_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "PRIMARY KEY (cluster_id, claim_id))"
            ),
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_claim_cluster_membership_claim')} "
            f"ON {ident('claim_cluster_membership')} ({ident('claim_id')})",
            connection=conn,
        )

        backend.execute(
            (
                f"CREATE TABLE IF NOT EXISTS {ident('claim_cluster_links')} ("
                "parent_cluster_id BIGINT NOT NULL, "
                "child_cluster_id BIGINT NOT NULL, "
                "relation_type TEXT, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "PRIMARY KEY (parent_cluster_id, child_cluster_id))"
            ),
            connection=conn,
        )
    except BackendDatabaseError as exc:
        logger.warning(f"Could not ensure Claims extensions on Postgres: {exc}")


__all__ = [
    "PostgresClaimsCollectionsDB",
    "ensure_postgres_claims_extensions",
    "ensure_postgres_claims_tables",
    "ensure_postgres_collections_tables",
]
