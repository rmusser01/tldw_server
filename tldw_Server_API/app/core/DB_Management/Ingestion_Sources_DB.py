from __future__ import annotations

import json
from typing import Any


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


async def postgres_source_schema_is_ready(db: Any) -> bool:
    """Prove base relations in the creation namespace without taking DDL locks."""
    cursor = await db.execute(
        """
        SELECT BOOL_AND(COALESCE(
            relation.relkind::text = required.kind
            AND (required.parent_name IS NULL OR (
                index_state.indrelid = parent.oid
                AND index_state.indisvalid AND index_state.indisready
            )), FALSE)) AS ready
        FROM (VALUES
            ('ingestion_sources', 'r', NULL::text),
            ('ingestion_source_state', 'r', NULL),
            ('ingestion_source_snapshots', 'r', NULL),
            ('ingestion_source_items', 'r', NULL),
            ('ingestion_item_events', 'r', NULL),
            ('ingestion_source_artifacts', 'r', NULL),
            ('idx_ingestion_sources_user_id', 'i', 'ingestion_sources'),
            ('idx_ingestion_sources_scheduler', 'i', 'ingestion_sources'),
            ('idx_ingestion_source_state_active_job', 'i', 'ingestion_source_state'),
            ('idx_ingestion_source_snapshots_source_status_id', 'i', 'ingestion_source_snapshots'),
            ('idx_ingestion_source_artifacts_source_kind_status', 'i', 'ingestion_source_artifacts'),
            ('idx_ingestion_item_events_source_item', 'i', 'ingestion_item_events')
        ) AS required(name, kind, parent_name)
        LEFT JOIN pg_catalog.pg_namespace namespace
          ON namespace.nspname = current_schema()
        LEFT JOIN pg_catalog.pg_class relation
          ON relation.relnamespace = namespace.oid AND relation.relname = required.name
        LEFT JOIN pg_catalog.pg_class parent
          ON parent.relnamespace = namespace.oid AND parent.relname = required.parent_name
          AND parent.relkind = 'r'
        LEFT JOIN pg_catalog.pg_index index_state ON index_state.indexrelid = relation.oid
        """
    )
    row = await cursor.fetchone()
    return row is not None and row["ready"] is True


async def ensure_postgres_source_item_presence(db: Any) -> None:
    """Upgrade pre-presence PostgreSQL catalogs while retaining existing rows."""
    column = await db.execute(
        "SELECT 1 FROM pg_catalog.pg_attribute "
        "WHERE attrelid = 'ingestion_source_items'::regclass "
        "AND attname = 'present_in_source' AND NOT attisdropped"
    )
    if await column.fetchone() is not None:
        return
    await db.execute(
        "ALTER TABLE ingestion_source_items "
        "ADD COLUMN IF NOT EXISTS present_in_source INTEGER NOT NULL DEFAULT 1"
    )


async def update_ingestion_source_record(
    db,
    *,
    source_id: int,
    source_type: str,
    sink_type: str,
    policy: str,
    enabled: bool,
    schedule_enabled: bool,
    schedule_config: dict[str, Any],
    config: dict[str, Any],
    updated_at: str,
) -> None:
    """Persist the mutable ingestion source definition fields for a source row."""

    await db.execute(
        """
        UPDATE ingestion_sources
        SET source_type = ?,
            sink_type = ?,
            policy = ?,
            enabled = ?,
            schedule_enabled = ?,
            schedule_config_json = ?,
            config_json = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            str(source_type),
            str(sink_type),
            str(policy),
            1 if enabled else 0,
            1 if schedule_enabled else 0,
            _json_dumps(schedule_config),
            _json_dumps(config),
            str(updated_at),
            int(source_id),
        ),
    )
