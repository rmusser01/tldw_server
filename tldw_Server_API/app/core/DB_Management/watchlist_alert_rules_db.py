"""SQLite persistence helpers for watchlist alert rules."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    resolve_trusted_database_path,
)


@dataclass
class AlertRule:
    id: int
    user_id: str
    job_id: int | None
    name: str
    enabled: bool
    condition_type: str
    condition_value: str
    severity: str
    created_at: str
    updated_at: str


ALERT_RULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS watchlist_alert_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    job_id INTEGER,
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    condition_type TEXT NOT NULL,
    condition_value TEXT NOT NULL DEFAULT '{}',
    severity TEXT NOT NULL DEFAULT 'warning',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alert_rules_user_job
    ON watchlist_alert_rules(user_id, job_id);
"""


def _trusted_db_path(db_path: str) -> str:
    return str(resolve_trusted_database_path(db_path, label="watchlist alert rules db"))


def ensure_watchlist_alert_rules_table(db_path: str) -> None:
    """Create the watchlist alert rules table if it does not exist."""
    with sqlite3.connect(_trusted_db_path(db_path)) as conn:
        conn.executescript(ALERT_RULES_TABLE_SQL)


def list_watchlist_alert_rules(
    db_path: str,
    user_id: str,
    job_id: int | None = None,
) -> list[AlertRule]:
    with sqlite3.connect(_trusted_db_path(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        if job_id is not None:
            rows = conn.execute(
                """
                SELECT * FROM watchlist_alert_rules
                WHERE user_id = ? AND (job_id = ? OR job_id IS NULL)
                ORDER BY created_at DESC
                """,
                (user_id, job_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM watchlist_alert_rules
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
    return [_row_to_rule(row) for row in rows]


def create_watchlist_alert_rule(
    db_path: str,
    *,
    user_id: str,
    name: str,
    condition_type: str,
    condition_value: dict[str, Any] | str | None = None,
    job_id: int | None = None,
    severity: str = "warning",
) -> AlertRule:
    now = datetime.now(timezone.utc).isoformat()
    serialized_condition_value = _serialize_condition_value(condition_value)
    with sqlite3.connect(_trusted_db_path(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO watchlist_alert_rules
                (user_id, job_id, name, enabled, condition_type, condition_value, severity, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                job_id,
                name,
                condition_type,
                serialized_condition_value,
                severity,
                now,
                now,
            ),
        )
    return AlertRule(
        id=cur.lastrowid or 0,
        user_id=user_id,
        job_id=job_id,
        name=name,
        enabled=True,
        condition_type=condition_type,
        condition_value=serialized_condition_value,
        severity=severity,
        created_at=now,
        updated_at=now,
    )


def get_watchlist_alert_rule(db_path: str, rule_id: int, user_id: str) -> AlertRule | None:
    with sqlite3.connect(_trusted_db_path(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM watchlist_alert_rules WHERE id = ? AND user_id = ?",
            (rule_id, user_id),
        ).fetchone()
    if row is None:
        return None
    return _row_to_rule(row)


def update_watchlist_alert_rule(
    db_path: str,
    rule_id: int,
    user_id: str,
    **fields: Any,
) -> AlertRule | None:
    current_rule = get_watchlist_alert_rule(db_path, rule_id, user_id)
    if current_rule is None:
        return None

    updates = {key: value for key, value in fields.items() if value is not None}
    if not updates:
        return None

    updated_at = datetime.now(timezone.utc).isoformat()
    merged_values = {
        "name": updates.get("name", current_rule.name),
        "enabled": _serialize_enabled(updates.get("enabled", current_rule.enabled)),
        "condition_type": updates.get("condition_type", current_rule.condition_type),
        "condition_value": _serialize_condition_value(
            updates.get("condition_value", current_rule.condition_value)
        ),
        "severity": updates.get("severity", current_rule.severity),
        "job_id": updates.get("job_id", current_rule.job_id),
    }
    with sqlite3.connect(_trusted_db_path(db_path)) as conn:
        cur = conn.execute(
            """
            UPDATE watchlist_alert_rules
            SET name = ?, enabled = ?, condition_type = ?, condition_value = ?,
                severity = ?, job_id = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                merged_values["name"],
                merged_values["enabled"],
                merged_values["condition_type"],
                merged_values["condition_value"],
                merged_values["severity"],
                merged_values["job_id"],
                updated_at,
                rule_id,
                user_id,
            ),
        )
        if cur.rowcount <= 0:
            return None
    return get_watchlist_alert_rule(db_path, rule_id, user_id)


def delete_watchlist_alert_rule(db_path: str, rule_id: int, user_id: str) -> bool:
    with sqlite3.connect(_trusted_db_path(db_path)) as conn:
        cur = conn.execute(
            "DELETE FROM watchlist_alert_rules WHERE id = ? AND user_id = ?",
            (rule_id, user_id),
        )
    return cur.rowcount > 0


def _row_to_rule(row: sqlite3.Row) -> AlertRule:
    return AlertRule(
        id=row["id"],
        user_id=row["user_id"],
        job_id=row["job_id"],
        name=row["name"],
        enabled=bool(row["enabled"]),
        condition_type=row["condition_type"],
        condition_value=row["condition_value"],
        severity=row["severity"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _serialize_condition_value(value: dict[str, Any] | str | None) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _serialize_enabled(value: bool | int) -> int:
    return 1 if bool(value) else 0
