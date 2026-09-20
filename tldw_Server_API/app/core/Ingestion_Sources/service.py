from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.Ingestion_Sources_DB import (
    update_ingestion_source_record,
)
from tldw_Server_API.app.core.exceptions import (
    IngestionSourceSchemaError,
    IngestionSourceValidationError,
)
from tldw_Server_API.app.core.Ingestion_Sources.models import (
    SINK_TYPES,
    SOURCE_POLICIES,
    SOURCE_TYPES,
)

_SQLITE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def _json_loads(value: Any, default: Any) -> Any:
    raw = value
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _normalize_choice(value: Any, *, field_name: str, allowed: frozenset[str], default: str | None = None) -> str:
    raw = str(value if value is not None else default or "").strip().lower()
    if raw not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise IngestionSourceValidationError(
            f"Unsupported {field_name} '{raw}'. Allowed values: {allowed_values}"
        )
    return raw


def _normalize_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise IngestionSourceValidationError(f"{field_name} must be a boolean")


def validate_source_sink_pair(*, source_type: str, sink_type: str) -> None:
    if source_type == "git_repository" and sink_type != "notes":
        raise IngestionSourceValidationError(
            "Git repository sources currently support the notes sink only"
        )


def normalize_source_payload(data: dict[str, Any]) -> dict[str, Any]:
    source_type = _normalize_choice(
        data.get("source_type"),
        field_name="source_type",
        allowed=SOURCE_TYPES,
    )
    sink_type = _normalize_choice(
        data.get("sink_type"),
        field_name="sink_type",
        allowed=SINK_TYPES,
    )
    policy = _normalize_choice(
        data.get("policy"),
        field_name="policy",
        allowed=SOURCE_POLICIES,
        default="canonical",
    )
    validate_source_sink_pair(source_type=source_type, sink_type=sink_type)
    enabled = _normalize_bool(data.get("enabled"), field_name="enabled", default=True)
    schedule_config = data.get("schedule") or data.get("schedule_config") or {}
    schedule_enabled = _normalize_bool(
        data.get("schedule_enabled"),
        field_name="schedule_enabled",
        default=bool(schedule_config),
    )
    return {
        "source_type": source_type,
        "sink_type": sink_type,
        "policy": policy,
        "enabled": enabled,
        "schedule_enabled": schedule_enabled,
        "schedule_config": schedule_config if isinstance(schedule_config, dict) else {},
    }


async def ensure_ingestion_sources_schema(db) -> None:
    schema_script = """
        CREATE TABLE IF NOT EXISTS ingestion_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            sink_type TEXT NOT NULL,
            policy TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            schedule_enabled INTEGER NOT NULL DEFAULT 0,
            schedule_config_json TEXT NOT NULL DEFAULT '{}',
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ingestion_source_state (
            source_id INTEGER PRIMARY KEY,
            active_job_id TEXT,
            last_successful_snapshot_id INTEGER,
            last_sync_started_at TEXT,
            last_sync_completed_at TEXT,
            last_sync_status TEXT,
            last_error TEXT,
            FOREIGN KEY(source_id) REFERENCES ingestion_sources(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ingestion_source_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            snapshot_kind TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES ingestion_sources(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ingestion_source_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            normalized_relative_path TEXT NOT NULL,
            content_hash TEXT,
            sync_status TEXT NOT NULL DEFAULT 'pending',
            binding_json TEXT NOT NULL DEFAULT '{}',
            present_in_source INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(source_id, normalized_relative_path),
            FOREIGN KEY(source_id) REFERENCES ingestion_sources(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ingestion_item_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            item_path TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES ingestion_sources(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ingestion_source_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            snapshot_id INTEGER,
            artifact_kind TEXT NOT NULL,
            status TEXT NOT NULL,
            storage_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES ingestion_sources(id) ON DELETE CASCADE,
            FOREIGN KEY(snapshot_id) REFERENCES ingestion_source_snapshots(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ingestion_sources_user_id
            ON ingestion_sources(user_id, id);

        CREATE INDEX IF NOT EXISTS idx_ingestion_sources_scheduler
            ON ingestion_sources(enabled, schedule_enabled, id);

        CREATE INDEX IF NOT EXISTS idx_ingestion_source_state_active_job
            ON ingestion_source_state(active_job_id, source_id);

        CREATE INDEX IF NOT EXISTS idx_ingestion_source_snapshots_source_status_id
            ON ingestion_source_snapshots(source_id, status, id DESC);

        CREATE INDEX IF NOT EXISTS idx_ingestion_source_artifacts_source_kind_status
            ON ingestion_source_artifacts(source_id, artifact_kind, status, id DESC);

        CREATE INDEX IF NOT EXISTS idx_ingestion_item_events_source_item
            ON ingestion_item_events(source_id, item_path, id DESC);
        """
    statement_lines: list[str] = []
    for line in schema_script.splitlines():
        statement_lines.append(line)
        statement = "\n".join(statement_lines).strip()
        if statement and sqlite3.complete_statement(statement):
            await db.execute(statement)
            statement_lines.clear()
    if any(line.strip() for line in statement_lines):
        raise IngestionSourceSchemaError(
            "Incomplete ingestion sources schema statement"
        )
    await _ensure_sqlite_column(
        db,
        table_name="ingestion_source_items",
        column_name="present_in_source",
        column_sql="INTEGER NOT NULL DEFAULT 1",
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    keys = getattr(row, "keys", None)
    if callable(keys):
        return {key: row[key] for key in row.keys()}
    return dict(row)


def _validate_sqlite_identifier(identifier: str) -> str:
    normalized = str(identifier or "").strip()
    if not _SQLITE_IDENTIFIER_PATTERN.fullmatch(normalized):
        raise IngestionSourceValidationError(f"Received unsafe SQL identifier: {identifier!r}")
    return normalized


async def _ensure_sqlite_column(
    db,
    *,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    validated_table_name = _validate_sqlite_identifier(table_name)
    validated_column_name = _validate_sqlite_identifier(column_name)
    try:
        pragma_cur = await db.execute(f"PRAGMA table_info({validated_table_name})")
        columns = {row["name"] for row in await pragma_cur.fetchall()}
    except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError):
        return
    if validated_column_name in columns:
        return
    await db.execute(
        f"ALTER TABLE {validated_table_name} ADD COLUMN {validated_column_name} {column_sql}"
    )


async def create_source(db, *, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_source_payload(payload)
    now = _utc_now_text()
    config_json = _json_dumps(payload.get("config") or {})
    schedule_config_json = _json_dumps(normalized.get("schedule_config") or {})

    cursor = await db.execute(
        """
        INSERT INTO ingestion_sources (
            user_id,
            source_type,
            sink_type,
            policy,
            enabled,
            schedule_enabled,
            schedule_config_json,
            config_json,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            normalized["source_type"],
            normalized["sink_type"],
            normalized["policy"],
            1 if normalized["enabled"] else 0,
            1 if normalized["schedule_enabled"] else 0,
            schedule_config_json,
            config_json,
            now,
            now,
        ),
    )
    source_id = int(cursor.lastrowid)
    await db.execute(
        """
        INSERT INTO ingestion_source_state (
            source_id,
            active_job_id,
            last_successful_snapshot_id,
            last_sync_started_at,
            last_sync_completed_at,
            last_sync_status,
            last_error
        ) VALUES (?, NULL, NULL, NULL, NULL, NULL, NULL)
        """,
        (source_id,),
    )

    row_cur = await db.execute(
        """
        SELECT
            id,
            user_id,
            source_type,
            sink_type,
            policy,
            enabled,
            schedule_enabled,
            schedule_config_json,
            config_json,
            created_at,
            updated_at
        FROM ingestion_sources
        WHERE id = ?
        """,
        (source_id,),
    )
    row = await row_cur.fetchone()
    result = _row_to_dict(row)
    if "config_json" in result:
        result["config"] = _json_loads(result.pop("config_json"), {})
    if "schedule_config_json" in result:
        result["schedule_config"] = _json_loads(result.pop("schedule_config_json"), {})
    if "enabled" in result:
        result["enabled"] = bool(result["enabled"])
    if "schedule_enabled" in result:
        result["schedule_enabled"] = bool(result["schedule_enabled"])
    return result


def _deserialize_source_row(row: Any) -> dict[str, Any]:
    result = _row_to_dict(row)
    if not result:
        return result
    if "config_json" in result:
        result["config"] = _json_loads(result.pop("config_json"), {})
    if "schedule_config_json" in result:
        result["schedule_config"] = _json_loads(result.pop("schedule_config_json"), {})
    if "last_successful_sync_summary_json" in result:
        result["last_successful_sync_summary"] = _json_loads(
            result.pop("last_successful_sync_summary_json"),
            {},
        )
    if "enabled" in result:
        result["enabled"] = bool(result["enabled"])
    if "schedule_enabled" in result:
        result["schedule_enabled"] = bool(result["schedule_enabled"])
    return result


def _deserialize_source_item_row(row: Any) -> dict[str, Any]:
    result = _row_to_dict(row)
    if not result:
        return result
    result["binding"] = _json_loads(result.pop("binding_json", None), {})
    if "present_in_source" in result:
        result["present_in_source"] = bool(result["present_in_source"])
    return result


def _deserialize_source_artifact_row(row: Any) -> dict[str, Any]:
    result = _row_to_dict(row)
    if not result:
        return result
    result["metadata"] = _json_loads(result.pop("metadata_json", None), {})
    return result


async def get_source_by_id(db, *, source_id: int, user_id: int | None = None) -> dict[str, Any]:
    query = """
        SELECT
            s.id,
            s.user_id,
            s.source_type,
            s.sink_type,
            s.policy,
            s.enabled,
            s.schedule_enabled,
            s.schedule_config_json,
            s.config_json,
            s.created_at,
            s.updated_at,
            st.active_job_id,
            st.last_successful_snapshot_id,
            st.last_sync_started_at,
            st.last_sync_completed_at,
            st.last_sync_status,
            st.last_error,
            ss.summary_json AS last_successful_sync_summary_json
        FROM ingestion_sources s
        JOIN ingestion_source_state st ON st.source_id = s.id
        LEFT JOIN ingestion_source_snapshots ss ON ss.id = st.last_successful_snapshot_id
        WHERE s.id = ?
    """
    params: list[Any] = [int(source_id)]
    if user_id is not None:
        query += " AND s.user_id = ?"
        params.append(int(user_id))
    cursor = await db.execute(query, tuple(params))
    row = await cursor.fetchone()
    return _deserialize_source_row(row)


def _source_identity_patch_changed(existing: dict[str, Any], patch: dict[str, Any]) -> bool:
    if "source_type" in patch and patch.get("source_type") not in (None, existing.get("source_type")):
        return True
    if "sink_type" in patch and patch.get("sink_type") not in (None, existing.get("sink_type")):
        return True
    if "config" in patch:
        patch_config = patch.get("config")
        if isinstance(patch_config, dict) and patch_config != (existing.get("config") or {}):
            return True
    return False


async def update_source(
    db,
    *,
    source_id: int,
    user_id: int,
    patch: dict[str, Any],
) -> dict[str, Any]:
    existing = await get_source_by_id(db, source_id=source_id, user_id=user_id)
    if not existing:
        return {}
    identity_changed = _source_identity_patch_changed(existing, patch)
    if identity_changed and existing.get("last_successful_snapshot_id") is not None:
        raise IngestionSourceValidationError(
            "Source identity is immutable after the first successful sync"
        )

    update_requested = any(
        key in patch and patch.get(key) is not None
        for key in ("source_type", "sink_type", "config", "policy", "enabled", "schedule_enabled", "schedule")
    )
    if not update_requested:
        return existing

    source_type_value = existing.get("source_type")
    if "source_type" in patch and patch.get("source_type") is not None:
        source_type_value = _normalize_choice(
            patch.get("source_type"),
            field_name="source_type",
            allowed=SOURCE_TYPES,
        )
    sink_type_value = existing.get("sink_type")
    if "sink_type" in patch and patch.get("sink_type") is not None:
        sink_type_value = _normalize_choice(
            patch.get("sink_type"),
            field_name="sink_type",
            allowed=SINK_TYPES,
        )
    validate_source_sink_pair(
        source_type=str(source_type_value),
        sink_type=str(sink_type_value),
    )
    config_value = existing.get("config") or {}
    if "config" in patch and patch.get("config") is not None:
        config_value = patch.get("config") if isinstance(patch.get("config"), dict) else {}
    policy_value = existing.get("policy")
    if "policy" in patch and patch.get("policy") is not None:
        policy_value = _normalize_choice(
            patch.get("policy"),
            field_name="policy",
            allowed=SOURCE_POLICIES,
        )
    enabled_value = existing.get("enabled")
    if "enabled" in patch and patch.get("enabled") is not None:
        enabled_value = _normalize_bool(
            patch.get("enabled"),
            field_name="enabled",
            default=bool(enabled_value),
        )
    schedule_enabled_value = existing.get("schedule_enabled")
    if "schedule_enabled" in patch and patch.get("schedule_enabled") is not None:
        schedule_enabled_value = _normalize_bool(
            patch.get("schedule_enabled"),
            field_name="schedule_enabled",
            default=bool(schedule_enabled_value),
        )
    schedule_config_value = existing.get("schedule_config") or {}
    if "schedule" in patch and patch.get("schedule") is not None:
        schedule_config = patch.get("schedule")
        schedule_config_value = schedule_config if isinstance(schedule_config, dict) else {}

    await update_ingestion_source_record(
        db,
        source_id=int(source_id),
        source_type=str(source_type_value),
        sink_type=str(sink_type_value),
        policy=str(policy_value),
        enabled=bool(enabled_value),
        schedule_enabled=bool(schedule_enabled_value),
        schedule_config=schedule_config_value,
        config=config_value,
        updated_at=_utc_now_text(),
    )
    updated = await get_source_by_id(db, source_id=source_id, user_id=user_id)
    return updated or {}


async def list_sources_for_scheduler(db) -> list[dict[str, Any]]:
    cursor = await db.execute(
        """
        SELECT
            s.id,
            s.user_id,
            s.source_type,
            s.sink_type,
            s.policy,
            s.enabled,
            s.schedule_enabled,
            s.schedule_config_json,
            s.config_json,
            st.active_job_id,
            st.last_successful_snapshot_id,
            st.last_sync_started_at,
            st.last_sync_completed_at,
            st.last_sync_status,
            st.last_error,
            ss.summary_json AS last_successful_sync_summary_json
        FROM ingestion_sources s
        JOIN ingestion_source_state st ON st.source_id = s.id
        LEFT JOIN ingestion_source_snapshots ss ON ss.id = st.last_successful_snapshot_id
        WHERE s.enabled = 1
          AND s.schedule_enabled = 1
        ORDER BY s.id ASC
        """
    )
    rows = await cursor.fetchall()
    return [_deserialize_source_row(row) for row in rows]


async def list_sources_for_retention_cleanup(db) -> list[dict[str, Any]]:
    cursor = await db.execute(
        """
        SELECT
            s.id,
            s.user_id,
            s.source_type,
            s.sink_type,
            s.policy,
            s.enabled,
            s.schedule_enabled,
            s.schedule_config_json,
            s.config_json,
            st.active_job_id,
            st.last_successful_snapshot_id,
            st.last_sync_started_at,
            st.last_sync_completed_at,
            st.last_sync_status,
            st.last_error,
            ss.summary_json AS last_successful_sync_summary_json
        FROM ingestion_sources s
        JOIN ingestion_source_state st ON st.source_id = s.id
        LEFT JOIN ingestion_source_snapshots ss ON ss.id = st.last_successful_snapshot_id
        WHERE s.source_type = 'archive_snapshot'
        ORDER BY s.id ASC
        """
    )
    rows = await cursor.fetchall()
    return [_deserialize_source_row(row) for row in rows]


async def list_sources_by_user(db, *, user_id: int) -> list[dict[str, Any]]:
    cursor = await db.execute(
        """
        SELECT
            s.id,
            s.user_id,
            s.source_type,
            s.sink_type,
            s.policy,
            s.enabled,
            s.schedule_enabled,
            s.schedule_config_json,
            s.config_json,
            s.created_at,
            s.updated_at,
            st.active_job_id,
            st.last_successful_snapshot_id,
            st.last_sync_started_at,
            st.last_sync_completed_at,
            st.last_sync_status,
            st.last_error,
            ss.summary_json AS last_successful_sync_summary_json
        FROM ingestion_sources s
        JOIN ingestion_source_state st ON st.source_id = s.id
        LEFT JOIN ingestion_source_snapshots ss ON ss.id = st.last_successful_snapshot_id
        WHERE s.user_id = ?
        ORDER BY s.id ASC
        """,
        (int(user_id),),
    )
    rows = await cursor.fetchall()
    return [_deserialize_source_row(row) for row in rows]


async def create_source_snapshot(
    db,
    *,
    source_id: int,
    snapshot_kind: str,
    status: str,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now_text()
    cursor = await db.execute(
        """
        INSERT INTO ingestion_source_snapshots (
            source_id,
            snapshot_kind,
            status,
            summary_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(source_id),
            str(snapshot_kind),
            str(status),
            _json_dumps(summary or {}),
            now,
        ),
    )
    snapshot_id = int(cursor.lastrowid)
    row_cur = await db.execute(
        """
        SELECT id, source_id, snapshot_kind, status, summary_json, created_at
        FROM ingestion_source_snapshots
        WHERE id = ?
        """,
        (snapshot_id,),
    )
    row = _row_to_dict(await row_cur.fetchone())
    row["summary"] = _json_loads(row.pop("summary_json", None), {})
    return row


async def get_source_snapshot_by_id(
    db,
    *,
    snapshot_id: int,
) -> dict[str, Any] | None:
    cursor = await db.execute(
        """
        SELECT id, source_id, snapshot_kind, status, summary_json, created_at
        FROM ingestion_source_snapshots
        WHERE id = ?
        """,
        (int(snapshot_id),),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    result = _row_to_dict(row)
    result["summary"] = _json_loads(result.pop("summary_json", None), {})
    return result


async def get_latest_source_snapshot(
    db,
    *,
    source_id: int,
    status: str | None = None,
) -> dict[str, Any] | None:
    query = """
        SELECT id, source_id, snapshot_kind, status, summary_json, created_at
        FROM ingestion_source_snapshots
        WHERE source_id = ?
    """
    params: list[Any] = [int(source_id)]
    if status is not None:
        query += " AND status = ?"
        params.append(str(status))
    query += " ORDER BY id DESC LIMIT 1"
    cursor = await db.execute(query, tuple(params))
    row = await cursor.fetchone()
    if row is None:
        return None
    result = _row_to_dict(row)
    result["summary"] = _json_loads(result.pop("summary_json", None), {})
    return result


async def list_source_snapshots(
    db,
    *,
    source_id: int,
    status: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT id, source_id, snapshot_kind, status, summary_json, created_at
        FROM ingestion_source_snapshots
        WHERE source_id = ?
    """
    params: list[Any] = [int(source_id)]
    if status is not None:
        query += " AND status = ?"
        params.append(str(status))
    query += " ORDER BY id DESC"
    cursor = await db.execute(query, tuple(params))
    rows = await cursor.fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        result = _row_to_dict(row)
        result["summary"] = _json_loads(result.pop("summary_json", None), {})
        results.append(result)
    return results


async def update_source_snapshot(
    db,
    *,
    snapshot_id: int,
    status: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = await get_source_snapshot_by_id(db, snapshot_id=snapshot_id)
    if not existing:
        raise ValueError(f"Snapshot not found: {snapshot_id}")
    merged_summary = dict(existing.get("summary") or {})
    if summary is not None:
        merged_summary.update(summary)
    await db.execute(
        """
        UPDATE ingestion_source_snapshots
        SET status = ?,
            summary_json = ?
        WHERE id = ?
        """,
        (
            str(status or existing.get("status") or "pending"),
            _json_dumps(merged_summary),
            int(snapshot_id),
        ),
    )
    updated = await get_source_snapshot_by_id(db, snapshot_id=snapshot_id)
    return updated or {}


async def delete_source_snapshot(
    db,
    *,
    snapshot_id: int,
) -> None:
    await db.execute(
        "DELETE FROM ingestion_source_snapshots WHERE id = ?",
        (int(snapshot_id),),
    )


async def create_source_artifact(
    db,
    *,
    source_id: int,
    snapshot_id: int | None,
    artifact_kind: str,
    status: str,
    storage_path: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now_text()
    cursor = await db.execute(
        """
        INSERT INTO ingestion_source_artifacts (
            source_id,
            snapshot_id,
            artifact_kind,
            status,
            storage_path,
            metadata_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(source_id),
            None if snapshot_id is None else int(snapshot_id),
            str(artifact_kind),
            str(status),
            None if storage_path is None else str(storage_path),
            _json_dumps(metadata or {}),
            now,
        ),
    )
    artifact_id = int(cursor.lastrowid)
    artifact = await get_source_artifact_by_id(db, artifact_id=artifact_id)
    return artifact or {}


async def get_source_artifact_by_id(
    db,
    *,
    artifact_id: int,
) -> dict[str, Any] | None:
    cursor = await db.execute(
        """
        SELECT
            id,
            source_id,
            snapshot_id,
            artifact_kind,
            status,
            storage_path,
            metadata_json,
            created_at
        FROM ingestion_source_artifacts
        WHERE id = ?
        """,
        (int(artifact_id),),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _deserialize_source_artifact_row(row)


async def list_source_artifacts(
    db,
    *,
    source_id: int,
    snapshot_id: int | None = None,
    artifact_kind: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            source_id,
            snapshot_id,
            artifact_kind,
            status,
            storage_path,
            metadata_json,
            created_at
        FROM ingestion_source_artifacts
        WHERE source_id = ?
    """
    params: list[Any] = [int(source_id)]
    if snapshot_id is not None:
        query += " AND snapshot_id = ?"
        params.append(int(snapshot_id))
    if artifact_kind is not None:
        query += " AND artifact_kind = ?"
        params.append(str(artifact_kind))
    if status is not None:
        query += " AND status = ?"
        params.append(str(status))
    query += " ORDER BY id DESC"
    cursor = await db.execute(query, tuple(params))
    rows = await cursor.fetchall()
    return [_deserialize_source_artifact_row(row) for row in rows]


async def get_latest_source_artifact(
    db,
    *,
    source_id: int,
    snapshot_id: int | None = None,
    artifact_kind: str | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    query = """
        SELECT
            id,
            source_id,
            snapshot_id,
            artifact_kind,
            status,
            storage_path,
            metadata_json,
            created_at
        FROM ingestion_source_artifacts
        WHERE source_id = ?
    """
    params: list[Any] = [int(source_id)]
    if snapshot_id is not None:
        query += " AND snapshot_id = ?"
        params.append(int(snapshot_id))
    if artifact_kind is not None:
        query += " AND artifact_kind = ?"
        params.append(str(artifact_kind))
    if status is not None:
        query += " AND status = ?"
        params.append(str(status))
    query += " ORDER BY id DESC LIMIT 1"
    cursor = await db.execute(query, tuple(params))
    row = await cursor.fetchone()
    if row is None:
        return None
    return _deserialize_source_artifact_row(row)


async def update_source_artifact(
    db,
    *,
    artifact_id: int,
    snapshot_id: int | None = None,
    status: str | None = None,
    storage_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = await get_source_artifact_by_id(db, artifact_id=artifact_id)
    if not existing:
        raise ValueError(f"Artifact not found: {artifact_id}")
    merged_metadata = dict(existing.get("metadata") or {})
    if metadata is not None:
        merged_metadata.update(metadata)
    await db.execute(
        """
        UPDATE ingestion_source_artifacts
        SET snapshot_id = ?,
            status = ?,
            storage_path = ?,
            metadata_json = ?
        WHERE id = ?
        """,
        (
            int(snapshot_id) if snapshot_id is not None else existing.get("snapshot_id"),
            str(status or existing.get("status") or "pending"),
            (
                str(storage_path)
                if storage_path is not None
                else existing.get("storage_path")
            ),
            _json_dumps(merged_metadata),
            int(artifact_id),
        ),
    )
    updated = await get_source_artifact_by_id(db, artifact_id=artifact_id)
    return updated or {}


async def delete_source_artifact(
    db,
    *,
    artifact_id: int,
) -> None:
    await db.execute(
        "DELETE FROM ingestion_source_artifacts WHERE id = ?",
        (int(artifact_id),),
    )


async def list_source_items(
    db,
    *,
    source_id: int,
    include_absent: bool = True,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            source_id,
            normalized_relative_path,
            content_hash,
            sync_status,
            binding_json,
            present_in_source,
            created_at,
            updated_at
        FROM ingestion_source_items
        WHERE source_id = ?
    """
    params: list[Any] = [int(source_id)]
    if not include_absent:
        query += " AND present_in_source = 1"
    query += " ORDER BY normalized_relative_path ASC"
    cursor = await db.execute(query, tuple(params))
    rows = await cursor.fetchall()
    return [_deserialize_source_item_row(row) for row in rows]


async def get_source_item(
    db,
    *,
    source_id: int,
    normalized_relative_path: str,
) -> dict[str, Any] | None:
    cursor = await db.execute(
        """
        SELECT
            id,
            source_id,
            normalized_relative_path,
            content_hash,
            sync_status,
            binding_json,
            present_in_source,
            created_at,
            updated_at
        FROM ingestion_source_items
        WHERE source_id = ?
          AND normalized_relative_path = ?
        """,
        (int(source_id), str(normalized_relative_path)),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _deserialize_source_item_row(row)


async def get_source_item_by_id(
    db,
    *,
    source_id: int,
    item_id: int,
) -> dict[str, Any] | None:
    cursor = await db.execute(
        """
        SELECT
            id,
            source_id,
            normalized_relative_path,
            content_hash,
            sync_status,
            binding_json,
            present_in_source,
            created_at,
            updated_at
        FROM ingestion_source_items
        WHERE source_id = ?
          AND id = ?
        """,
        (int(source_id), int(item_id)),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _deserialize_source_item_row(row)


async def upsert_source_item(
    db,
    *,
    source_id: int,
    normalized_relative_path: str,
    content_hash: str | None,
    sync_status: str,
    binding: dict[str, Any] | None = None,
    present_in_source: bool,
) -> dict[str, Any]:
    now = _utc_now_text()
    await db.execute(
        """
        INSERT INTO ingestion_source_items (
            source_id,
            normalized_relative_path,
            content_hash,
            sync_status,
            binding_json,
            present_in_source,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, normalized_relative_path) DO UPDATE SET
            content_hash = excluded.content_hash,
            sync_status = excluded.sync_status,
            binding_json = excluded.binding_json,
            present_in_source = excluded.present_in_source,
            updated_at = excluded.updated_at
        """,
        (
            int(source_id),
            str(normalized_relative_path),
            content_hash,
            str(sync_status),
            _json_dumps(binding or {}),
            1 if present_in_source else 0,
            now,
            now,
        ),
    )
    item = await get_source_item(
        db,
        source_id=source_id,
        normalized_relative_path=normalized_relative_path,
    )
    return item or {}


async def update_source_item_state(
    db,
    *,
    item_id: int,
    sync_status: str,
    binding: dict[str, Any] | None = None,
    present_in_source: bool | None = None,
    content_hash: str | None = None,
    clear_content_hash: bool = False,
) -> dict[str, Any]:
    existing_cursor = await db.execute(
        """
        SELECT source_id, binding_json, present_in_source, content_hash
        FROM ingestion_source_items
        WHERE id = ?
        """,
        (int(item_id),),
    )
    existing = _row_to_dict(await existing_cursor.fetchone())
    if not existing:
        raise ValueError(f"Source item not found: {item_id}")
    now = _utc_now_text()
    updated_binding = binding if binding is not None else _json_loads(existing.get("binding_json"), {})
    await db.execute(
        """
        UPDATE ingestion_source_items
        SET sync_status = ?,
            binding_json = ?,
            present_in_source = ?,
            content_hash = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            str(sync_status),
            _json_dumps(updated_binding),
            1 if (existing.get("present_in_source") if present_in_source is None else present_in_source) else 0,
            None
            if clear_content_hash
            else (content_hash if content_hash is not None else existing.get("content_hash")),
            now,
            int(item_id),
        ),
    )
    return (
        await get_source_item_by_id(
            db,
            source_id=int(existing["source_id"]),
            item_id=int(item_id),
        )
        or {}
    )


async def record_ingestion_item_event(
    db,
    *,
    source_id: int,
    item_path: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now_text()
    cursor = await db.execute(
        """
        INSERT INTO ingestion_item_events (
            source_id,
            item_path,
            event_type,
            payload_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(source_id),
            str(item_path),
            str(event_type),
            _json_dumps(payload or {}),
            now,
        ),
    )
    row_cur = await db.execute(
        """
        SELECT id, source_id, item_path, event_type, payload_json, created_at
        FROM ingestion_item_events
        WHERE id = ?
        """,
        (int(cursor.lastrowid),),
    )
    row = _row_to_dict(await row_cur.fetchone())
    row["payload"] = _json_loads(row.pop("payload_json", None), {})
    return row


async def start_source_sync_job(db, *, source_id: int, job_id: str) -> dict[str, Any]:
    now = _utc_now_text()
    await db.execute(
        """
        UPDATE ingestion_source_state
        SET active_job_id = ?,
            last_sync_started_at = ?,
            last_sync_status = ?,
            last_error = NULL
        WHERE source_id = ?
          AND (
              active_job_id IS NULL
              OR active_job_id = ''
              OR active_job_id = ?
          )
        """,
        (str(job_id), now, "running", int(source_id), str(job_id)),
    )
    cursor = await db.execute(
        """
        SELECT
            source_id,
            active_job_id,
            last_successful_snapshot_id,
            last_sync_started_at,
            last_sync_completed_at,
            last_sync_status,
            last_error
        FROM ingestion_source_state
        WHERE source_id = ?
        """,
        (int(source_id),),
    )
    return _row_to_dict(await cursor.fetchone())


async def finish_source_sync_job(
    db,
    *,
    source_id: int,
    job_id: str,
    outcome: str,
    error: str | None = None,
    snapshot_id: int | None = None,
) -> dict[str, Any]:
    now = _utc_now_text()
    cursor = await db.execute(
        """
        UPDATE ingestion_source_state
        SET active_job_id = NULL,
            last_successful_snapshot_id = CASE
                WHEN ? = 'success' AND ? IS NOT NULL THEN ?
                ELSE last_successful_snapshot_id
            END,
            last_sync_completed_at = ?,
            last_sync_status = ?,
            last_error = ?
        WHERE source_id = ?
          AND active_job_id = ?
        """,
        (
            str(outcome),
            snapshot_id,
            snapshot_id,
            now,
            str(outcome),
            None if error is None else str(error),
            int(source_id),
            str(job_id),
        ),
    )
    if getattr(cursor, "rowcount", -1) == 0:
        raise RuntimeError(
            f"Could not finish source {int(source_id)}: active sync job does not match {job_id}"
        )
    cursor = await db.execute(
        """
        SELECT
            source_id,
            active_job_id,
            last_successful_snapshot_id,
            last_sync_started_at,
            last_sync_completed_at,
            last_sync_status,
            last_error
        FROM ingestion_source_state
        WHERE source_id = ?
        """,
        (int(source_id),),
    )
    return _row_to_dict(await cursor.fetchone())
