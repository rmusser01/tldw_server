"""Repository for admin monitoring control-plane persistence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.AuthNZ.database import (
    DatabasePool,
    build_postgres_in_clause,
    build_sqlite_in_clause,
)

_UNSET = object()
_SCHEMA_ENSURE_LOCK = asyncio.Lock()
_SCHEMA_ENSURED_KEYS: set[str] = set()


@dataclass
class AuthnzAdminMonitoringRepo:
    """Repository for admin alert rules, overlay state, and alert events."""

    db_pool: DatabasePool

    def _schema_cache_key(self) -> str:
        """Return a stable process-local key for one-time schema initialization."""
        if self._is_postgres_backend():
            db_url = getattr(getattr(self.db_pool, "settings", None), "DATABASE_URL", None)
            return f"postgres:{db_url or id(getattr(self.db_pool, 'pool', None))}"
        db_fs_path = getattr(self.db_pool, "_sqlite_fs_path", None) or getattr(
            self.db_pool, "db_path", None
        )
        return f"sqlite:{db_fs_path}"

    def _is_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is using PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    async def ensure_schema_ready_once(self) -> None:
        """Ensure the admin-monitoring schema is initialized once per process/backend."""
        cache_key = self._schema_cache_key()
        async with _SCHEMA_ENSURE_LOCK:
            if cache_key in _SCHEMA_ENSURED_KEYS:
                return
            await self.ensure_schema()
            _SCHEMA_ENSURED_KEYS.add(cache_key)

    async def ensure_schema(self) -> None:
        """Ensure admin monitoring tables exist for the active backend."""
        if self._is_postgres_backend():
            from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                ensure_admin_monitoring_tables_pg,
            )
            from tldw_Server_API.app.core.AuthNZ.postgres_runtime_schema import (
                POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL,
            )

            async with self.db_pool.transaction() as conn:
                if not await conn.fetchval(POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL):
                    # Serving roles use owner-provisioned tables without replaying CREATE INDEX.
                    for query in (
                        "SELECT id, metric, operator, threshold, duration_minutes, severity, enabled, "
                        "created_by_user_id, updated_by_user_id, created_at, updated_at "
                        "FROM public.admin_alert_rules LIMIT 0",
                        "SELECT alert_identity, assigned_to_user_id, snoozed_until, escalated_severity, "
                        "acknowledged_at, dismissed_at, updated_by_user_id, updated_at "
                        "FROM public.admin_alert_state LIMIT 0",
                        "SELECT id, alert_identity, action, actor_user_id, details_json, created_at "
                        "FROM public.admin_alert_events LIMIT 0",
                    ):
                        await conn.fetch(query)
                    return

            ensured = await ensure_admin_monitoring_tables_pg(self.db_pool)
            if not ensured:
                raise RuntimeError("Failed to ensure PostgreSQL admin monitoring tables")
            return

        from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

        db_fs_path = getattr(self.db_pool, "_sqlite_fs_path", None) or getattr(
            self.db_pool, "db_path", None
        )
        if not db_fs_path:
            raise RuntimeError("SQLite admin monitoring schema requires a filesystem database path")
        if str(db_fs_path) == ":memory:":
            raise RuntimeError(
                "SQLite admin monitoring schema requires a filesystem database path; ':memory:' is not supported"
            )
        await asyncio.to_thread(ensure_authnz_tables, Path(str(db_fs_path)))

    @staticmethod
    def _normalize_timestamp(value: Any) -> Any:
        if not isinstance(value, datetime):
            return value
        if value.tzinfo is None:
            return value.isoformat()
        normalized = value.astimezone(timezone.utc).isoformat()
        return normalized.replace("+00:00", "Z")

    @staticmethod
    def _coerce_timestamp_input(value: Any) -> Any:
        if value is None or value is _UNSET:
            return value
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return value
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return value
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @classmethod
    def _normalize_row(cls, row: Any) -> dict[str, Any]:
        """Normalize backend-specific row objects into JSON-friendly dicts."""
        if row is None:
            return {}
        if isinstance(row, dict):
            record = dict(row)
        elif hasattr(row, "keys"):
            record = {str(key): row[key] for key in row.keys()}
        else:
            try:
                values = tuple(row)
            except TypeError as exc:
                raise TypeError(
                    f"Unsupported admin monitoring row type for normalization: {type(row)!r}"
                ) from exc
            record = {str(index): value for index, value in enumerate(values)}

        for field_name in (
            "id",
            "duration_minutes",
            "created_by_user_id",
            "updated_by_user_id",
            "assigned_to_user_id",
            "actor_user_id",
        ):
            if record.get(field_name) is not None:
                try:
                    record[field_name] = int(record[field_name])
                except (TypeError, ValueError):
                    _ = None

        if "enabled" in record:
            record["enabled"] = bool(record["enabled"])

        for field_name, field_value in list(record.items()):
            record[field_name] = cls._normalize_timestamp(field_value)

        return record

    async def create_rule(
        self,
        *,
        metric: str,
        operator: str,
        threshold: float,
        duration_minutes: int,
        severity: str,
        enabled: bool,
        created_by_user_id: int | None,
    ) -> dict[str, Any]:
        """Create a new admin alert rule and return the stored row."""
        async with self.db_pool.transaction() as conn:
            if self._is_postgres_backend():
                row = await conn.fetchrow(
                    """
                    INSERT INTO admin_alert_rules (
                        metric, operator, threshold, duration_minutes, severity, enabled,
                        created_by_user_id, updated_by_user_id, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING *
                    """,
                    metric,
                    operator,
                    float(threshold),
                    int(duration_minutes),
                    severity,
                    bool(enabled),
                    created_by_user_id,
                )
                return self._normalize_row(row)

            cursor = await conn.execute(
                """
                INSERT INTO admin_alert_rules (
                    metric, operator, threshold, duration_minutes, severity, enabled,
                    created_by_user_id, updated_by_user_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    metric,
                    operator,
                    float(threshold),
                    int(duration_minutes),
                    severity,
                    1 if enabled else 0,
                    created_by_user_id,
                    created_by_user_id,
                ),
            )
            rule_id = getattr(cursor, "lastrowid", None)
            if rule_id is None:
                raise RuntimeError("Failed to create admin alert rule")
            select_cursor = await conn.execute(
                "SELECT * FROM admin_alert_rules WHERE id = ?",
                (rule_id,),
            )
            row = await select_cursor.fetchone()
            return self._normalize_row(row)

    async def get_rule(self, rule_id: int) -> dict[str, Any] | None:
        """Return a single alert rule by id."""
        if self._is_postgres_backend():
            row = await self.db_pool.fetchone(
                "SELECT * FROM admin_alert_rules WHERE id = $1",
                int(rule_id),
            )
            return self._normalize_row(row) if row else None
        row = await self.db_pool.fetchone(
            "SELECT * FROM admin_alert_rules WHERE id = ?",
            int(rule_id),
        )
        return self._normalize_row(row) if row else None

    async def list_rules(self) -> list[dict[str, Any]]:
        """List alert rules newest first."""
        rows = await self.db_pool.fetchall(
            """
            SELECT *
            FROM admin_alert_rules
            ORDER BY created_at DESC, id DESC
            """
        )
        return [self._normalize_row(row) for row in rows]

    async def delete_rule(self, rule_id: int) -> bool:
        """Delete an alert rule and return whether a row was removed."""
        async with self.db_pool.transaction() as conn:
            if self._is_postgres_backend():
                result = await conn.execute(
                    "DELETE FROM admin_alert_rules WHERE id = $1",
                    int(rule_id),
                )
                return str(result).split()[-1] != "0"

            cursor = await conn.execute(
                "DELETE FROM admin_alert_rules WHERE id = ?",
                (int(rule_id),),
            )
            return bool(getattr(cursor, "rowcount", 0))

    async def clear_state_and_events(self) -> dict[str, int]:
        """Delete all alert overlay rows and history events."""
        async with self.db_pool.transaction() as conn:
            if self._is_postgres_backend():
                deleted_state_result = await conn.execute("DELETE FROM admin_alert_state")
                deleted_events_result = await conn.execute("DELETE FROM admin_alert_events")
                return {
                    "deleted_states": int(str(deleted_state_result).split()[-1]),
                    "deleted_events": int(str(deleted_events_result).split()[-1]),
                }

            deleted_state_cursor = await conn.execute("DELETE FROM admin_alert_state")
            deleted_events_cursor = await conn.execute("DELETE FROM admin_alert_events")
            return {
                "deleted_states": int(getattr(deleted_state_cursor, "rowcount", 0) or 0),
                "deleted_events": int(getattr(deleted_events_cursor, "rowcount", 0) or 0),
            }

    async def list_alert_states(self, alert_identities: list[str]) -> list[dict[str, Any]]:
        """List overlay rows for the supplied alert identities."""
        if not alert_identities:
            return []

        if self._is_postgres_backend():
            placeholders, params = build_postgres_in_clause(alert_identities)
            query = f"""
                SELECT *
                FROM admin_alert_state
                WHERE alert_identity IN ({placeholders})
                ORDER BY updated_at DESC, alert_identity ASC
                """  # nosec B608 - placeholders are generated by build_postgres_in_clause
            rows = await self.db_pool.fetchall(
                query,
                *params,
            )
            return [self._normalize_row(row) for row in rows]

        placeholders, params = build_sqlite_in_clause(alert_identities)
        query = f"""
            SELECT *
            FROM admin_alert_state
            WHERE alert_identity IN ({placeholders})
            ORDER BY updated_at DESC, alert_identity ASC
            """  # nosec B608 - placeholders are generated by build_sqlite_in_clause
        rows = await self.db_pool.fetchall(
            query,
            params,
        )
        return [self._normalize_row(row) for row in rows]

    async def upsert_alert_state(
        self,
        *,
        alert_identity: str,
        assigned_to_user_id: int | None | object = _UNSET,
        snoozed_until: str | None | object = _UNSET,
        escalated_severity: str | None | object = _UNSET,
        acknowledged_at: str | None | object = _UNSET,
        dismissed_at: str | None | object = _UNSET,
        updated_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Create or update overlay state for a runtime alert."""
        provided_values = {
            "assigned_to_user_id": assigned_to_user_id,
            "snoozed_until": snoozed_until,
            "escalated_severity": escalated_severity,
            "acknowledged_at": acknowledged_at,
            "dismissed_at": dismissed_at,
        }
        stored_values = {
            key: value for key, value in provided_values.items() if value is not _UNSET
        }

        if self._is_postgres_backend():
            insert_values = {
                "assigned_to_user_id": None if assigned_to_user_id is _UNSET else assigned_to_user_id,
                "snoozed_until": (
                    None if snoozed_until is _UNSET else self._coerce_timestamp_input(snoozed_until)
                ),
                "escalated_severity": None if escalated_severity is _UNSET else escalated_severity,
                "acknowledged_at": (
                    None if acknowledged_at is _UNSET else self._coerce_timestamp_input(acknowledged_at)
                ),
                "dismissed_at": (
                    None if dismissed_at is _UNSET else self._coerce_timestamp_input(dismissed_at)
                ),
            }
            update_flags = {
                "assigned_to_user_id": assigned_to_user_id is not _UNSET,
                "snoozed_until": snoozed_until is not _UNSET,
                "escalated_severity": escalated_severity is not _UNSET,
                "acknowledged_at": acknowledged_at is not _UNSET,
                "dismissed_at": dismissed_at is not _UNSET,
            }
            row = await self.db_pool.fetchone(
                """
                INSERT INTO admin_alert_state (
                    alert_identity,
                    assigned_to_user_id,
                    snoozed_until,
                    escalated_severity,
                    acknowledged_at,
                    dismissed_at,
                    updated_by_user_id,
                    updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                ON CONFLICT (alert_identity)
                DO UPDATE SET
                    assigned_to_user_id = CASE
                        WHEN $8 THEN EXCLUDED.assigned_to_user_id
                        ELSE admin_alert_state.assigned_to_user_id
                    END,
                    snoozed_until = CASE
                        WHEN $9 THEN EXCLUDED.snoozed_until
                        ELSE admin_alert_state.snoozed_until
                    END,
                    escalated_severity = CASE
                        WHEN $10 THEN EXCLUDED.escalated_severity
                        ELSE admin_alert_state.escalated_severity
                    END,
                    acknowledged_at = CASE
                        WHEN $11 THEN EXCLUDED.acknowledged_at
                        ELSE admin_alert_state.acknowledged_at
                    END,
                    dismissed_at = CASE
                        WHEN $12 THEN EXCLUDED.dismissed_at
                        ELSE admin_alert_state.dismissed_at
                    END,
                    updated_by_user_id = EXCLUDED.updated_by_user_id,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
                """,
                alert_identity,
                insert_values["assigned_to_user_id"],
                insert_values["snoozed_until"],
                insert_values["escalated_severity"],
                insert_values["acknowledged_at"],
                insert_values["dismissed_at"],
                updated_by_user_id,
                update_flags["assigned_to_user_id"],
                update_flags["snoozed_until"],
                update_flags["escalated_severity"],
                update_flags["acknowledged_at"],
                update_flags["dismissed_at"],
            )
            return self._normalize_row(row)

        existing = await self.db_pool.fetchone(
            "SELECT * FROM admin_alert_state WHERE alert_identity = ?",
            alert_identity,
        )
        merged_values: dict[str, Any] = {}
        if existing:
            merged_values.update(existing)
        merged_values.update(stored_values)
        merged_values["updated_by_user_id"] = updated_by_user_id

        async with self.db_pool.transaction() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO admin_alert_state (
                    alert_identity,
                    assigned_to_user_id,
                    snoozed_until,
                    escalated_severity,
                    acknowledged_at,
                    dismissed_at,
                    updated_by_user_id,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    alert_identity,
                    merged_values.get("assigned_to_user_id"),
                    merged_values.get("snoozed_until"),
                    merged_values.get("escalated_severity"),
                    merged_values.get("acknowledged_at"),
                    merged_values.get("dismissed_at"),
                    updated_by_user_id,
                ),
            )
            cursor = await conn.execute(
                "SELECT * FROM admin_alert_state WHERE alert_identity = ?",
                (alert_identity,),
            )
            row = await cursor.fetchone()
            return self._normalize_row(row)

    async def append_alert_event(
        self,
        *,
        alert_identity: str,
        action: str,
        actor_user_id: int | None,
        details_json: str | None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Append an alert-event history row and return the stored event."""
        created_at_value = created_at or datetime.now(timezone.utc).isoformat()

        async with self.db_pool.transaction() as conn:
            if self._is_postgres_backend():
                created_at_value = self._coerce_timestamp_input(created_at_value)
                row = await conn.fetchrow(
                    """
                    INSERT INTO admin_alert_events (
                        alert_identity, action, actor_user_id, details_json, created_at
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING *
                    """,
                    alert_identity,
                    action,
                    actor_user_id,
                    details_json,
                    created_at_value,
                )
                return self._normalize_row(row)

            cursor = await conn.execute(
                """
                INSERT INTO admin_alert_events (
                    alert_identity, action, actor_user_id, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    alert_identity,
                    action,
                    actor_user_id,
                    details_json,
                    created_at_value,
                ),
            )
            event_id = getattr(cursor, "lastrowid", None)
            if event_id is None:
                raise RuntimeError("Failed to append admin alert event")
            select_cursor = await conn.execute(
                "SELECT * FROM admin_alert_events WHERE id = ?",
                (event_id,),
            )
            row = await select_cursor.fetchone()
            return self._normalize_row(row)

    async def list_alert_events(
        self,
        *,
        alert_identity: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List alert events newest first, optionally scoped to one alert identity."""
        limit_value = max(1, int(limit))

        if self._is_postgres_backend():
            if alert_identity:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT *
                    FROM admin_alert_events
                    WHERE alert_identity = $1
                    ORDER BY created_at DESC, id DESC
                    LIMIT $2
                    """,
                    alert_identity,
                    limit_value,
                )
            else:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT *
                    FROM admin_alert_events
                    ORDER BY created_at DESC, id DESC
                    LIMIT $1
                    """,
                    limit_value,
                )
            return [self._normalize_row(row) for row in rows]

        if alert_identity:
            rows = await self.db_pool.fetchall(
                """
                SELECT *
                FROM admin_alert_events
                WHERE alert_identity = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                alert_identity,
                limit_value,
            )
        else:
            rows = await self.db_pool.fetchall(
                """
                SELECT *
                FROM admin_alert_events
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                limit_value,
            )
        return [self._normalize_row(row) for row in rows]
