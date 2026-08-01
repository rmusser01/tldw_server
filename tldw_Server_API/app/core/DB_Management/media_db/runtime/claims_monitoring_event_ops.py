"""Package-owned claims monitoring event helpers."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def insert_claims_monitoring_event(
    self,
    *,
    user_id: str,
    event_type: str,
    severity: str | None = None,
    payload_json: str | None = None,
) -> dict[str, Any]:
    now = self._get_current_utc_timestamp_str()
    insert_sql = (
        "INSERT INTO claims_monitoring_events "
        "(user_id, event_type, severity, payload_json, created_at, delivered_at) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )
    if self.backend_type == BackendType.POSTGRESQL:
        insert_sql += " RETURNING id"
    cursor = self.execute_query(
        insert_sql,
        (
            str(user_id),
            str(event_type),
            severity,
            payload_json,
            now,
            None,
        ),
        commit=True,
    )
    if self.backend_type == BackendType.POSTGRESQL:
        row = cursor.fetchone()
        event_id = int(row["id"]) if row else 0
    else:
        event_id = int(getattr(cursor, "lastrowid", 0) or 0)
    return get_claims_monitoring_event(self, event_id) if event_id else {}


def get_claims_monitoring_event(self, event_id: int) -> dict[str, Any]:
    """Return one Claims monitoring event row by ID."""
    row = self.execute_query(
        "SELECT id, user_id, event_type, severity, payload_json, created_at, delivered_at "
        "FROM claims_monitoring_events WHERE id = ?",
        (int(event_id),),
    ).fetchone()
    return dict(row) if row else {}


def list_claims_monitoring_events(
    self,
    *,
    user_id: str,
    event_type: str | None = None,
    severity: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = ["user_id = ?"]
    params: list[Any] = [str(user_id)]
    if event_type:
        conditions.append("event_type = ?")
        params.append(str(event_type))
    if severity:
        conditions.append("severity = ?")
        params.append(str(severity))
    if start_time:
        conditions.append("created_at >= ?")
        params.append(str(start_time))
    if end_time:
        conditions.append("created_at <= ?")
        params.append(str(end_time))
    where_clause = " AND ".join(conditions)
    rows = self.execute_query(
        (
            "SELECT id, user_id, event_type, severity, payload_json, created_at, delivered_at "  # nosec B608
            "FROM claims_monitoring_events WHERE "
            + where_clause
            + " ORDER BY created_at ASC"
        ),
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def list_undelivered_claims_monitoring_events(
    self,
    *,
    user_id: str,
    event_type: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 500
    limit = max(1, min(5000, limit))
    conditions: list[str] = ["user_id = ?", "delivered_at IS NULL"]
    params: list[Any] = [str(user_id)]
    if event_type:
        conditions.append("event_type = ?")
        params.append(str(event_type))
    sql = (
        "SELECT id, user_id, event_type, severity, payload_json, created_at, delivered_at "  # nosec B608
        "FROM claims_monitoring_events WHERE "
        + " AND ".join(conditions)
        + " ORDER BY created_at ASC LIMIT ?"
    )
    params.append(limit)
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def mark_claims_monitoring_events_delivered(self, ids: list[int]) -> int:
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    now = self._get_current_utc_timestamp_str()
    sql = f"UPDATE claims_monitoring_events SET delivered_at = ? WHERE id IN ({placeholders})"  # nosec B608
    params: list[Any] = [str(now)]
    params.extend([int(i) for i in ids])
    cursor = self.execute_query(sql, tuple(params), commit=True)
    try:
        return int(getattr(cursor, "rowcount", 0) or 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return 0


def has_successful_claims_monitoring_event_delivery(
    self,
    *,
    user_id: str,
    event_id: int,
    alert_id: int,
    channel: str,
    limit: int = 1000,
) -> bool:
    """Check whether a monitoring event already has a successful delivery."""
    try:
        event_id = int(event_id)
        alert_id = int(alert_id)
        limit = int(limit)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return False
    if event_id <= 0 or alert_id <= 0:
        return False
    limit = max(1, min(5000, limit))
    rows = self.execute_query(
        (
            "SELECT payload_json FROM claims_monitoring_events "
            "WHERE user_id = ? AND event_type = ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?"
        ),
        (str(user_id), "webhook_delivery", limit),
    ).fetchall()
    wanted_channel = str(channel)
    for row in rows:
        try:
            raw_payload = row["payload_json"]
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            try:
                raw_payload = row[0]
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                continue
        try:
            payload = json.loads(str(raw_payload or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            if (
                str(payload.get("status")) == "success"
                and int(payload.get("event_id") or 0) == event_id
                and int(payload.get("alert_id") or 0) == alert_id
                and str(payload.get("channel") or "") == wanted_channel
            ):
                return True
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            continue
    return False


def get_latest_claims_monitoring_event_delivery(
    self,
    *,
    user_id: str,
    event_type: str | None = None,
) -> str | None:
    conditions: list[str] = ["user_id = ?", "delivered_at IS NOT NULL"]
    params: list[Any] = [str(user_id)]
    if event_type:
        conditions.append("event_type = ?")
        params.append(str(event_type))
    sql = (
        "SELECT MAX(delivered_at) AS delivered_at "  # nosec B608
        "FROM claims_monitoring_events WHERE "
        + " AND ".join(conditions)
    )
    row = self.execute_query(sql, tuple(params)).fetchone()
    if not row:
        return None
    try:
        return row.get("delivered_at")
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        try:
            return row[0]
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            return None
