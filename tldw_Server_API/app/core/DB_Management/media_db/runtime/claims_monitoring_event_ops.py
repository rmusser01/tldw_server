"""Package-owned claims monitoring event helpers."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS
_EVENT_ADVISORY_LOCK_PREFIX = "claims_monitoring_events:"


def _lock_postgres_event_owner(
    self,
    *,
    user_id: str,
    connection: Any,
    shared: bool,
) -> None:
    lock_sql = (
        "SELECT pg_advisory_xact_lock_shared(hashtextextended(?, 0))"
        if shared
        else "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))"
    )
    self.execute_query(
        lock_sql,
        (f"{_EVENT_ADVISORY_LOCK_PREFIX}{user_id}",),
        commit=False,
        connection=connection,
    )


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
    if self.backend_type == BackendType.POSTGRESQL:
        with self.transaction() as conn:
            _lock_postgres_event_owner(
                self,
                user_id=str(user_id),
                connection=conn,
                shared=True,
            )
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
                commit=False,
                connection=conn,
            )
            row = cursor.fetchone()
            event_id = int(row["id"]) if row else 0
    else:
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


def get_claims_monitoring_event_payload_bounded(
    self,
    *,
    user_id: str,
    event_id: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Return one owner-scoped payload only when it fits the caller's byte budget."""
    if type(event_id) is not int or event_id <= 0:
        raise ValueError("event_id must be a positive integer")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if self.backend_type == BackendType.POSTGRESQL:
        normalized_sql = (
            "CASE WHEN payload_json IS NULL OR payload_json = '' THEN '{}' "
            "WHEN payload_json IS JSON THEN json_serialize(payload_json::jsonb RETURNING text) "
            "ELSE '{}' END"
        )
        size_sql = f"octet_length({normalized_sql})"
    else:
        normalized_sql = (
            "CASE WHEN payload_json IS NULL OR payload_json = '' THEN '{}' "
            "WHEN json_valid(payload_json) THEN json(payload_json) ELSE '{}' END"
        )
        size_sql = f"length(CAST(({normalized_sql}) AS BLOB))"
    row = self.execute_query(
        (
            "SELECT CASE WHEN "  # nosec B608 - expression is selected by backend type.
            + size_sql
            + " <= ? THEN "
            + normalized_sql
            + " ELSE NULL END AS payload_json, "
            + size_sql
            + " AS payload_size_bytes FROM claims_monitoring_events "
            "WHERE id = ? AND user_id = ? LIMIT 1"
        ),
        (max_bytes, event_id, str(user_id)),
    ).fetchone()
    return dict(row) if row else {}


def get_claims_monitoring_event_high_water(self, *, user_id: str) -> int:
    """Return the owner's greatest persisted monitoring-event ID, or zero."""
    if self.backend_type == BackendType.POSTGRESQL:
        with self.transaction() as conn:
            _lock_postgres_event_owner(
                self,
                user_id=str(user_id),
                connection=conn,
                shared=False,
            )
            row = self.execute_query(
                "SELECT COALESCE(MAX(id), 0) AS event_id "
                "FROM claims_monitoring_events WHERE user_id = ?",
                (str(user_id),),
                commit=False,
                connection=conn,
            ).fetchone()
    else:
        row = self.execute_query(
            "SELECT COALESCE(MAX(id), 0) AS event_id "
            "FROM claims_monitoring_events WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()
    if not row:
        return 0
    try:
        return max(int(row["event_id"] or 0), 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        try:
            return max(int(row[0] or 0), 0)
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            return 0


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


def list_claims_monitoring_events_page(
    self,
    *,
    user_id: str,
    event_type: str | None = None,
    severity: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    after_created_at: Any = None,
    after_id: int | None = None,
    max_event_id: int | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    if (after_created_at is None) != (after_id is None):
        raise ValueError("after_created_at and after_id must be provided together")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 1000
    limit = max(1, min(1000, limit))

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
    if max_event_id is not None:
        if isinstance(max_event_id, bool) or not isinstance(max_event_id, int) or max_event_id < 0:
            raise ValueError("max_event_id must be a non-negative integer")
        conditions.append("id <= ?")
        params.append(max_event_id)
    if after_created_at is not None and after_id is not None:
        conditions.append("(created_at > ? OR (created_at = ? AND id > ?))")
        params.extend([after_created_at, after_created_at, int(after_id)])

    if self.backend_type == BackendType.POSTGRESQL:
        normalized_payload_sql = (
            "CASE WHEN payload_json IS NULL OR payload_json = '' THEN '{}' "
            "WHEN payload_json IS JSON THEN json_serialize(payload_json::jsonb RETURNING text) "
            "ELSE '{}' END"
        )
        payload_size_sql = f"octet_length({normalized_payload_sql})"
        payload_provider_sql = (
            "CASE WHEN payload_json IS JSON OBJECT THEN payload_json::jsonb ->> 'provider' END"
        )
        payload_model_sql = (
            "CASE WHEN payload_json IS JSON OBJECT THEN payload_json::jsonb ->> 'model' END"
        )
    else:
        normalized_payload_sql = (
            "CASE WHEN payload_json IS NULL OR payload_json = '' THEN '{}' "
            "WHEN json_valid(payload_json) THEN json(payload_json) ELSE '{}' END"
        )
        payload_size_sql = f"length(CAST(({normalized_payload_sql}) AS BLOB))"
        payload_provider_sql = (
            "CASE WHEN json_valid(payload_json) THEN "
            "CASE WHEN json_type(payload_json) = 'object' "
            "THEN CAST(json_extract(payload_json, '$.provider') AS TEXT) END END"
        )
        payload_model_sql = (
            "CASE WHEN json_valid(payload_json) THEN "
            "CASE WHEN json_type(payload_json) = 'object' "
            "THEN CAST(json_extract(payload_json, '$.model') AS TEXT) END END"
        )
    query = (
        "SELECT id, user_id, event_type, severity, created_at, "  # nosec B608
        + payload_size_sql
        + " AS payload_size_bytes, "
        + payload_provider_sql
        + " AS payload_provider, "
        + payload_model_sql
        + " AS payload_model "
        "FROM claims_monitoring_events WHERE "
        + " AND ".join(conditions)
        + " ORDER BY created_at ASC, id ASC LIMIT ?"
    )
    params.append(limit)
    rows = self.execute_query(query, tuple(params)).fetchall()
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
