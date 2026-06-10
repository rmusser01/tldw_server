"""
Per-user SQLite repository for Scheduled Tasks automation definitions.

This module stores durable previews, definitions, audit events, and idempotency
records for the Scheduled Tasks API foundation. The API/service layers own
validation and lifecycle rules; this layer provides owner-scoped persistence.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
    configure_sqlite_connection,
)

_SCHEDULED_TASKS_DB_NAME = "ScheduledTasks.db"
_DISABLED_LOCK_KINDS = {"none", "admin", "security", "system"}


@dataclass(frozen=True)
class PreviewRow:
    id: str
    owner_id: int
    mode: str
    family: str
    definition_id: str | None
    definition_version: int | None
    status: str
    payload_hash: str
    normalized_config: dict[str, Any]
    validation_errors: list[Any]
    warnings: list[Any]
    risk_class: str | None
    visibility_policy: str
    schedule_preview: dict[str, Any]
    redaction_policy: dict[str, Any]
    expires_at: str
    created_by: str
    created_at: str
    consumed_at: str | None
    created_definition_id: str | None


@dataclass(frozen=True)
class DefinitionRow:
    id: str
    owner_id: int
    version: int
    family: str
    name: str
    description: str | None
    lifecycle: str
    health: str
    disabled_lock_kind: str
    disabled_reason: str | None
    schedule: dict[str, Any]
    input: dict[str, Any]
    visibility_policy: str
    notification_policy: dict[str, Any]
    approval_policy: dict[str, Any]
    preview_id: str
    created_by: str
    updated_by: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AuditEventRow:
    id: str
    owner_id: int
    definition_id: str
    event_type: str
    actor: str
    summary: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    created_at: str
    request_id: str | None
    idempotency_key: str | None


@dataclass(frozen=True)
class IdempotencyRecordRow:
    owner_id: int
    route: str
    key: str
    payload_hash: str
    response_ref: dict[str, Any]
    created_at: str
    expires_at: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid4().hex


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_loads(raw_value: str) -> Any:
    return json.loads(raw_value)


def _optional_json_dumps(value: dict[str, Any] | None) -> str | None:
    return None if value is None else _json_dumps(value)


def _optional_json_loads(raw_value: str | None) -> dict[str, Any] | None:
    return None if raw_value is None else _json_loads(raw_value)


def _validate_limit_offset(limit: int, offset: int) -> None:
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    if offset < 0:
        raise ValueError("offset must not be negative")


def _validate_disabled_lock_kind(value: str) -> None:
    if value not in _DISABLED_LOCK_KINDS:
        raise ValueError(f"invalid disabled_lock_kind: {value!r}")


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_expired(expires_at: str) -> bool:
    return _parse_iso_datetime(expires_at) <= datetime.now(timezone.utc)


def _prune_expired_idempotency_record(
    conn: sqlite3.Connection,
    *,
    owner_id: int,
    route: str,
    key: str,
) -> None:
    row = conn.execute(
        """
        SELECT expires_at
        FROM scheduled_task_idempotency
        WHERE owner_id = ? AND route = ? AND key = ?
        """,
        [owner_id, route, key],
    ).fetchone()
    if row is not None and _is_expired(row["expires_at"]):
        conn.execute(
            """
            DELETE FROM scheduled_task_idempotency
            WHERE owner_id = ? AND route = ? AND key = ?
            """,
            [owner_id, route, key],
        )


def _preview_from_row(row: sqlite3.Row | None) -> PreviewRow | None:
    if row is None:
        return None
    return PreviewRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        mode=row["mode"],
        family=row["family"],
        definition_id=row["definition_id"],
        definition_version=row["definition_version"],
        status=row["status"],
        payload_hash=row["payload_hash"],
        normalized_config=_json_loads(row["normalized_config_json"]),
        validation_errors=_json_loads(row["validation_errors_json"]),
        warnings=_json_loads(row["warnings_json"]),
        risk_class=row["risk_class"],
        visibility_policy=row["visibility_policy"],
        schedule_preview=_json_loads(row["schedule_preview_json"]),
        redaction_policy=_json_loads(row["redaction_policy_json"]),
        expires_at=row["expires_at"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        consumed_at=row["consumed_at"],
        created_definition_id=row["created_definition_id"],
    )


def _definition_from_row(row: sqlite3.Row | None) -> DefinitionRow | None:
    if row is None:
        return None
    return DefinitionRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        version=int(row["version"]),
        family=row["family"],
        name=row["name"],
        description=row["description"],
        lifecycle=row["lifecycle"],
        health=row["health"],
        disabled_lock_kind=row["disabled_lock_kind"],
        disabled_reason=row["disabled_reason"],
        schedule=_json_loads(row["schedule_json"]),
        input=_json_loads(row["input_json"]),
        visibility_policy=row["visibility_policy"],
        notification_policy=_json_loads(row["notification_policy_json"]),
        approval_policy=_json_loads(row["approval_policy_json"]),
        preview_id=row["preview_id"],
        created_by=row["created_by"],
        updated_by=row["updated_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _audit_event_from_row(row: sqlite3.Row | None) -> AuditEventRow | None:
    if row is None:
        return None
    return AuditEventRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        definition_id=row["definition_id"],
        event_type=row["event_type"],
        actor=row["actor"],
        summary=row["summary"],
        before=_optional_json_loads(row["before_json"]),
        after=_optional_json_loads(row["after_json"]),
        created_at=row["created_at"],
        request_id=row["request_id"],
        idempotency_key=row["idempotency_key"],
    )


def _idempotency_record_from_row(row: sqlite3.Row | None) -> IdempotencyRecordRow | None:
    if row is None:
        return None
    return IdempotencyRecordRow(
        owner_id=int(row["owner_id"]),
        route=row["route"],
        key=row["key"],
        payload_hash=row["payload_hash"],
        response_ref=_json_loads(row["response_ref_json"]),
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )


class ScheduledTasksDatabase:
    """SQLite repository for one user's Scheduled Tasks automation database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    @classmethod
    def for_user(cls, user_id: int) -> ScheduledTasksDatabase:
        """Return the per-user Scheduled Tasks repository for ``user_id``."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return cls(user_dir / _SCHEDULED_TASKS_DB_NAME)

    def ensure_schema(self) -> None:
        """Create Scheduled Tasks automation tables and indexes if needed."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduled_task_previews (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    family TEXT NOT NULL,
                    definition_id TEXT,
                    definition_version INTEGER,
                    status TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    normalized_config_json TEXT NOT NULL,
                    validation_errors_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    risk_class TEXT,
                    visibility_policy TEXT NOT NULL,
                    schedule_preview_json TEXT NOT NULL,
                    redaction_policy_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    created_definition_id TEXT
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_definitions (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    lifecycle TEXT NOT NULL,
                    health TEXT NOT NULL,
                    disabled_lock_kind TEXT NOT NULL DEFAULT 'none'
                        CHECK (disabled_lock_kind IN ('none', 'admin', 'security', 'system')),
                    disabled_reason TEXT,
                    schedule_json TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    visibility_policy TEXT NOT NULL,
                    notification_policy_json TEXT NOT NULL,
                    approval_policy_json TEXT NOT NULL,
                    preview_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_audit_events (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    definition_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    created_at TEXT NOT NULL,
                    request_id TEXT,
                    idempotency_key TEXT
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_idempotency (
                    owner_id INTEGER NOT NULL,
                    route TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    response_ref_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (owner_id, route, key)
                );

                CREATE INDEX IF NOT EXISTS idx_scheduled_task_previews_owner_created
                    ON scheduled_task_previews(owner_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_previews_owner_definition
                    ON scheduled_task_previews(owner_id, definition_id);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_previews_owner_status
                    ON scheduled_task_previews(owner_id, status);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_family
                    ON scheduled_task_definitions(owner_id, family);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_lifecycle
                    ON scheduled_task_definitions(owner_id, lifecycle);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_health
                    ON scheduled_task_definitions(owner_id, health);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_updated
                    ON scheduled_task_definitions(owner_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_audit_definition_created
                    ON scheduled_task_audit_events(definition_id, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_task_idempotency_owner_route_key
                    ON scheduled_task_idempotency(owner_id, route, key);
                """
            )

    def create_preview(
        self,
        *,
        owner_id: int,
        mode: str,
        family: str,
        definition_id: str | None,
        definition_version: int | None,
        status: str,
        payload_hash: str,
        normalized_config: dict[str, Any],
        validation_errors: list[Any],
        warnings: list[Any],
        risk_class: str | None,
        visibility_policy: str,
        schedule_preview: dict[str, Any],
        redaction_policy: dict[str, Any],
        expires_at: str,
        created_by: str,
    ) -> PreviewRow:
        preview_id = _new_id()
        created_at = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_task_previews (
                    id, owner_id, mode, family, definition_id, definition_version,
                    status, payload_hash, normalized_config_json,
                    validation_errors_json, warnings_json, risk_class,
                    visibility_policy, schedule_preview_json, redaction_policy_json,
                    expires_at, created_by, created_at, consumed_at,
                    created_definition_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                [
                    preview_id,
                    owner_id,
                    mode,
                    family,
                    definition_id,
                    definition_version,
                    status,
                    payload_hash,
                    _json_dumps(normalized_config),
                    _json_dumps(validation_errors),
                    _json_dumps(warnings),
                    risk_class,
                    visibility_policy,
                    _json_dumps(schedule_preview),
                    _json_dumps(redaction_policy),
                    expires_at,
                    created_by,
                    created_at,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
                [owner_id, preview_id],
            ).fetchone()
        created = _preview_from_row(row)
        if created is None:
            raise RuntimeError("created preview could not be loaded")
        return created

    def get_preview(self, owner_id: int, preview_id: str) -> PreviewRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
                [owner_id, preview_id],
            ).fetchone()
        return _preview_from_row(row)

    def list_previews(
        self,
        owner_id: int,
        *,
        family: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        definition_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PreviewRow], int]:
        _validate_limit_offset(limit, offset)
        filter_params: list[Any] = [
            owner_id,
            family,
            family,
            mode,
            mode,
            status,
            status,
            definition_id,
            definition_id,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_previews
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR mode = ?)
                    AND (? IS NULL OR status = ?)
                    AND (? IS NULL OR definition_id = ?)
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_previews
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR mode = ?)
                    AND (? IS NULL OR status = ?)
                    AND (? IS NULL OR definition_id = ?)
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_preview_from_row(row) for row in rows) if row is not None], total

    def mark_preview_consumed(
        self,
        owner_id: int,
        preview_id: str,
        created_definition_id: str | None,
    ) -> PreviewRow:
        consumed_at = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_task_previews
                SET status = 'consumed',
                    consumed_at = ?,
                    created_definition_id = ?
                WHERE owner_id = ?
                    AND id = ?
                    AND consumed_at IS NULL
                    AND status != 'consumed'
                """,
                [consumed_at, created_definition_id, owner_id, preview_id],
            )
            updated_count = cursor.rowcount
            row = conn.execute(
                "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
                [owner_id, preview_id],
            ).fetchone()
        consumed = _preview_from_row(row)
        if consumed is None:
            raise KeyError(f"preview not found: {preview_id}")
        if updated_count == 0:
            raise ValueError("preview already consumed")
        return consumed

    def create_definition(
        self,
        *,
        owner_id: int,
        family: str,
        name: str,
        description: str | None,
        lifecycle: str,
        health: str,
        schedule: dict[str, Any],
        input: dict[str, Any],
        visibility_policy: str,
        notification_policy: dict[str, Any],
        approval_policy: dict[str, Any],
        preview_id: str,
        created_by: str,
        updated_by: str,
        disabled_lock_kind: str = "none",
        disabled_reason: str | None = None,
        version: int = 1,
    ) -> DefinitionRow:
        _validate_disabled_lock_kind(disabled_lock_kind)
        definition_id = _new_id()
        created_at = _utcnow_iso()
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            preview_exists = conn.execute(
                """
                SELECT 1
                FROM scheduled_task_previews
                WHERE owner_id = ? AND id = ?
                """,
                [owner_id, preview_id],
            ).fetchone()
            if preview_exists is None:
                raise KeyError(f"preview not found: {preview_id}")
            conn.execute(
                """
                INSERT INTO scheduled_task_definitions (
                    id, owner_id, version, family, name, description, lifecycle,
                    health, disabled_lock_kind, disabled_reason, schedule_json,
                    input_json, visibility_policy, notification_policy_json,
                    approval_policy_json, preview_id, created_by, updated_by,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    definition_id,
                    owner_id,
                    version,
                    family,
                    name,
                    description,
                    lifecycle,
                    health,
                    disabled_lock_kind,
                    disabled_reason,
                    _json_dumps(schedule),
                    _json_dumps(input),
                    visibility_policy,
                    _json_dumps(notification_policy),
                    _json_dumps(approval_policy),
                    preview_id,
                    created_by,
                    updated_by,
                    created_at,
                    created_at,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        created = _definition_from_row(row)
        if created is None:
            raise RuntimeError("created definition could not be loaded")
        return created

    def get_definition(self, owner_id: int, definition_id: str) -> DefinitionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        return _definition_from_row(row)

    def list_definitions(
        self,
        owner_id: int,
        *,
        family: str | None = None,
        lifecycle: str | None = None,
        health: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DefinitionRow], int]:
        _validate_limit_offset(limit, offset)
        pattern = f"%{query}%" if query else None
        filter_params: list[Any] = [
            owner_id,
            family,
            family,
            lifecycle,
            lifecycle,
            health,
            health,
            pattern,
            pattern,
            pattern,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_definitions
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR lifecycle = ?)
                    AND (? IS NULL OR health = ?)
                    AND (? IS NULL OR name LIKE ? OR description LIKE ?)
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_definitions
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR lifecycle = ?)
                    AND (? IS NULL OR health = ?)
                    AND (? IS NULL OR name LIKE ? OR description LIKE ?)
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_definition_from_row(row) for row in rows) if row is not None], total

    def update_definition(
        self,
        owner_id: int,
        definition_id: str,
        patch: dict[str, Any],
        expected_version: int | None = None,
    ) -> DefinitionRow:
        current = self.get_definition(owner_id=owner_id, definition_id=definition_id)
        if current is None:
            raise KeyError(f"definition not found: {definition_id}")
        if "disabled_lock_kind" in patch:
            _validate_disabled_lock_kind(str(patch["disabled_lock_kind"]))

        next_values = {
            "version": current.version + 1,
            "family": patch.get("family", current.family),
            "name": patch.get("name", current.name),
            "description": patch.get("description", current.description),
            "lifecycle": patch.get("lifecycle", current.lifecycle),
            "health": patch.get("health", current.health),
            "disabled_lock_kind": patch.get("disabled_lock_kind", current.disabled_lock_kind),
            "disabled_reason": patch.get("disabled_reason", current.disabled_reason),
            "schedule": patch.get("schedule", current.schedule),
            "input": patch.get("input", current.input),
            "visibility_policy": patch.get("visibility_policy", current.visibility_policy),
            "notification_policy": patch.get("notification_policy", current.notification_policy),
            "approval_policy": patch.get("approval_policy", current.approval_policy),
            "preview_id": patch.get("preview_id", current.preview_id),
            "updated_by": patch.get("updated_by", current.updated_by),
            "updated_at": _utcnow_iso(),
        }
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_task_definitions
                SET version = ?,
                    family = ?,
                    name = ?,
                    description = ?,
                    lifecycle = ?,
                    health = ?,
                    disabled_lock_kind = ?,
                    disabled_reason = ?,
                    schedule_json = ?,
                    input_json = ?,
                    visibility_policy = ?,
                    notification_policy_json = ?,
                    approval_policy_json = ?,
                    preview_id = ?,
                    updated_by = ?,
                    updated_at = ?
                WHERE owner_id = ? AND id = ?
                    AND (? IS NULL OR version = ?)
                """,
                [
                    next_values["version"],
                    next_values["family"],
                    next_values["name"],
                    next_values["description"],
                    next_values["lifecycle"],
                    next_values["health"],
                    next_values["disabled_lock_kind"],
                    next_values["disabled_reason"],
                    _json_dumps(next_values["schedule"]),
                    _json_dumps(next_values["input"]),
                    next_values["visibility_policy"],
                    _json_dumps(next_values["notification_policy"]),
                    _json_dumps(next_values["approval_policy"]),
                    next_values["preview_id"],
                    next_values["updated_by"],
                    next_values["updated_at"],
                    owner_id,
                    definition_id,
                    expected_version,
                    expected_version,
                ],
            )
            updated_count = cursor.rowcount
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        updated = _definition_from_row(row)
        if updated is None:
            raise KeyError(f"definition not found after update: {definition_id}")
        if updated_count == 0:
            if expected_version is not None:
                raise ValueError("definition version conflict")
            raise KeyError(f"definition not found after update: {definition_id}")
        return updated

    def create_audit_event(
        self,
        *,
        owner_id: int,
        definition_id: str,
        event_type: str,
        actor: str,
        summary: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AuditEventRow:
        event_id = _new_id()
        created_at = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_task_audit_events (
                    id, owner_id, definition_id, event_type, actor, summary,
                    before_json, after_json, created_at, request_id,
                    idempotency_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    event_id,
                    owner_id,
                    definition_id,
                    event_type,
                    actor,
                    summary,
                    _optional_json_dumps(before),
                    _optional_json_dumps(after),
                    created_at,
                    request_id,
                    idempotency_key,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_audit_events WHERE owner_id = ? AND id = ?",
                [owner_id, event_id],
            ).fetchone()
        created = _audit_event_from_row(row)
        if created is None:
            raise RuntimeError("created audit event could not be loaded")
        return created

    def list_audit_events(
        self,
        owner_id: int,
        definition_id: str,
        *,
        event_type: str | None = None,
        actor: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditEventRow], int]:
        _validate_limit_offset(limit, offset)
        filter_params: list[Any] = [
            owner_id,
            definition_id,
            event_type,
            event_type,
            actor,
            actor,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_audit_events
                WHERE owner_id = ?
                    AND definition_id = ?
                    AND (? IS NULL OR event_type = ?)
                    AND (? IS NULL OR actor = ?)
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_audit_events
                WHERE owner_id = ?
                    AND definition_id = ?
                    AND (? IS NULL OR event_type = ?)
                    AND (? IS NULL OR actor = ?)
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_audit_event_from_row(row) for row in rows) if row is not None], total

    def get_idempotency_record(
        self,
        owner_id: int,
        route: str,
        key: str,
    ) -> IdempotencyRecordRow | None:
        with self._connect() as conn:
            _prune_expired_idempotency_record(
                conn,
                owner_id=owner_id,
                route=route,
                key=key,
            )
            row = conn.execute(
                """
                SELECT * FROM scheduled_task_idempotency
                WHERE owner_id = ? AND route = ? AND key = ?
                """,
                [owner_id, route, key],
            ).fetchone()
        return _idempotency_record_from_row(row)

    def create_idempotency_record(
        self,
        *,
        owner_id: int,
        route: str,
        key: str,
        payload_hash: str,
        response_ref: dict[str, Any],
        expires_at: str,
    ) -> IdempotencyRecordRow:
        created_at = _utcnow_iso()
        with self._connect() as conn:
            _prune_expired_idempotency_record(
                conn,
                owner_id=owner_id,
                route=route,
                key=key,
            )
            conn.execute(
                """
                INSERT INTO scheduled_task_idempotency (
                    owner_id, route, key, payload_hash, response_ref_json,
                    created_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    owner_id,
                    route,
                    key,
                    payload_hash,
                    _json_dumps(response_ref),
                    created_at,
                    expires_at,
                ],
            )
            row = conn.execute(
                """
                SELECT * FROM scheduled_task_idempotency
                WHERE owner_id = ? AND route = ? AND key = ?
                """,
                [owner_id, route, key],
            ).fetchone()
        created = _idempotency_record_from_row(row)
        if created is None:
            raise RuntimeError("created idempotency record could not be loaded")
        return created

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        return conn
