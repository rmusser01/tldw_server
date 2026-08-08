"""Package-owned claims analytics export helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)


_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS

ALLOWED_EXPORT_TRANSITIONS = {
    ("queued", "processing"),
    ("queued", "failed"),
    ("processing", "ready"),
    ("processing", "failed"),
    ("failed", "processing"),
    ("ready", "ready"),
}

_SUPPORTED_EXPORT_FORMATS = {"json", "csv"}
_SUPPORTED_EXPORT_STATUSES = {"queued", "processing", "ready", "failed"}
_EXPORT_DELETE_CHUNK_SIZE = 400


def _affected_rows(cursor: Any) -> int:
    try:
        return max(int(getattr(cursor, "rowcount", 0) or 0), 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return 0


def _validate_job_id(job_id: Any) -> int:
    if type(job_id) is not int or job_id <= 0:
        raise ValueError("job_id must be a positive integer")
    return job_id


def _validate_claims_analytics_export_create(
    *,
    format: str,
    status: str,
    payload_json: str | None,
    payload_csv: str | None,
    error_code: str | None,
    error_message: str | None,
) -> None:
    if format not in _SUPPORTED_EXPORT_FORMATS:
        raise ValueError("format must be one of: json, csv")
    if status not in _SUPPORTED_EXPORT_STATUSES:
        raise ValueError("status must be one of: queued, processing, ready, failed")

    if status != "ready":
        if payload_json is not None or payload_csv is not None:
            raise ValueError("non-ready exports must not contain a payload")
        return

    payload_matches_format = (
        format == "json" and payload_json is not None and payload_csv is None
    ) or (
        format == "csv" and payload_csv is not None and payload_json is None
    )
    if not payload_matches_format:
        raise ValueError("ready export payload must match its format")
    if error_code is not None or error_message is not None:
        raise ValueError("ready exports must not contain error fields")


def create_claims_analytics_export(
    self,
    *,
    export_id: str,
    user_id: str,
    format: str,
    status: str,
    payload_json: str | None = None,
    payload_csv: str | None = None,
    filters_json: str | None = None,
    pagination_json: str | None = None,
    error_message: str | None = None,
    job_id: int | None = None,
    error_code: str | None = None,
    snapshot_at: str | None = None,
) -> dict[str, Any]:
    normalized_format = str(format)
    normalized_status = str(status)
    _validate_claims_analytics_export_create(
        format=normalized_format,
        status=normalized_status,
        payload_json=payload_json,
        payload_csv=payload_csv,
        error_code=error_code,
        error_message=error_message,
    )
    validated_job_id = None if job_id is None else _validate_job_id(job_id)
    now = self._get_current_utc_timestamp_str()
    self.execute_query(
        (
            "INSERT INTO claims_analytics_exports "
            "(export_id, user_id, format, status, payload_json, payload_csv, filters_json, "
            "pagination_json, error_message, job_id, error_code, snapshot_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            str(export_id),
            str(user_id),
            normalized_format,
            normalized_status,
            payload_json,
            payload_csv,
            filters_json,
            pagination_json,
            error_message,
            validated_job_id,
            error_code,
            snapshot_at,
            now,
            now,
        ),
        commit=True,
    )
    return get_claims_analytics_export(self, export_id, user_id=str(user_id))


def get_claims_analytics_export(
    self,
    export_id: str,
    *,
    user_id: str,
) -> dict[str, Any]:
    row = self.execute_query(
        (
            "SELECT export_id, user_id, format, status, payload_json, payload_csv, filters_json, "
            "pagination_json, error_message, job_id, error_code, snapshot_at, created_at, updated_at "
            "FROM claims_analytics_exports WHERE export_id = ? AND user_id = ? LIMIT 1"
        ),
        (str(export_id), str(user_id)),
    ).fetchone()
    return dict(row) if row else {}


def attach_claims_analytics_export_job(
    self,
    *,
    export_id: str,
    user_id: str,
    job_id: int,
) -> bool:
    validated_job_id = _validate_job_id(job_id)
    now = self._get_current_utc_timestamp_str()
    cursor = self.execute_query(
        (
            "UPDATE claims_analytics_exports SET job_id = ?, updated_at = ? "
            "WHERE export_id = ? AND user_id = ? AND (job_id IS NULL OR job_id = ?)"
        ),
        (
            validated_job_id,
            now,
            str(export_id),
            str(user_id),
            validated_job_id,
        ),
        commit=True,
    )
    return _affected_rows(cursor) > 0


def transition_claims_analytics_export_status(
    self,
    *,
    export_id: str,
    user_id: str,
    from_statuses: tuple[str, ...],
    to_status: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> bool:
    normalized_to = str(to_status)
    normalized_from = tuple(dict.fromkeys(str(status) for status in from_statuses))
    if not normalized_from or any(
        (from_status, normalized_to) not in ALLOWED_EXPORT_TRANSITIONS
        for from_status in normalized_from
    ):
        return False

    if normalized_to == "ready":
        if normalized_from != ("ready",):
            return False
        row = self.execute_query(
            (
                "SELECT 1 FROM claims_analytics_exports "
                "WHERE export_id = ? AND user_id = ? AND status = ? LIMIT 1"
            ),
            (str(export_id), str(user_id), "ready"),
        ).fetchone()
        return row is not None

    placeholders = ",".join("?" for _ in normalized_from)
    now = self._get_current_utc_timestamp_str()
    query = (
        "UPDATE claims_analytics_exports "
        "SET status = ?, error_code = ?, error_message = ?, updated_at = ? "
        "WHERE export_id = ? AND user_id = ? "
        f"AND status IN ({placeholders})"  # nosec B608
    )
    params: list[Any] = [
        normalized_to,
        error_code,
        error_message,
        now,
        str(export_id),
        str(user_id),
    ]
    params.extend(normalized_from)
    cursor = self.execute_query(query, tuple(params), commit=True)
    return _affected_rows(cursor) > 0


def mark_claims_analytics_export_ready(
    self,
    *,
    export_id: str,
    user_id: str,
    payload_json: str | None,
    payload_csv: str | None,
) -> bool:
    if (payload_json is None) == (payload_csv is None):
        return False
    expected_format = "json" if payload_json is not None else "csv"
    now = self._get_current_utc_timestamp_str()
    cursor = self.execute_query(
        (
            "UPDATE claims_analytics_exports "
            "SET status = ?, payload_json = ?, payload_csv = ?, error_code = NULL, "
            "error_message = NULL, updated_at = ? "
            "WHERE export_id = ? AND user_id = ? AND status = ? AND format = ?"
        ),
        (
            "ready",
            payload_json,
            payload_csv,
            now,
            str(export_id),
            str(user_id),
            "processing",
            expected_format,
        ),
        commit=True,
    )
    return _affected_rows(cursor) > 0


def list_claims_analytics_exports(
    self,
    user_id: str,
    *,
    status: str | None = None,
    format: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 100, 0
    limit = max(1, min(1000, limit))
    offset = max(0, offset)
    conditions = ["user_id = ?"]
    params: list[Any] = [str(user_id)]
    if status:
        conditions.append("status = ?")
        params.append(str(status))
    if format:
        conditions.append("format = ?")
        params.append(str(format))
    query = (
        "SELECT export_id, user_id, format, status, filters_json, pagination_json, error_message, "  # nosec B608
        "job_id, error_code, snapshot_at, created_at, updated_at "
        "FROM claims_analytics_exports WHERE "
        + " AND ".join(conditions)
        + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = self.execute_query(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def list_claims_analytics_exports_for_maintenance(
    self,
    *,
    user_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(1000, limit))
    rows = self.execute_query(
        (
            "SELECT export_id, user_id, format, status, filters_json, pagination_json, "
            "error_message, job_id, error_code, snapshot_at, created_at, updated_at "
            "FROM claims_analytics_exports WHERE user_id = ? "
            "ORDER BY updated_at ASC, export_id ASC LIMIT ?"
        ),
        (str(user_id), limit),
    ).fetchall()
    return [dict(row) for row in rows]


def count_claims_analytics_exports(
    self,
    user_id: str,
    *,
    status: str | None = None,
    format: str | None = None,
) -> int:
    conditions = ["user_id = ?"]
    params: list[Any] = [str(user_id)]
    if status:
        conditions.append("status = ?")
        params.append(str(status))
    if format:
        conditions.append("format = ?")
        params.append(str(format))
    row = self.execute_query(
        "SELECT COUNT(*) AS count FROM claims_analytics_exports WHERE " + " AND ".join(conditions),  # nosec B608
        tuple(params),
    ).fetchone()
    if not row:
        return 0
    try:
        return int(row["count"] or 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        try:
            return int(row[0] or 0)
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            return 0


def cleanup_claims_analytics_exports(
    self,
    *,
    user_id: str,
    retention_hours: float,
) -> int:
    try:
        retention_hours = float(retention_hours)
    except (TypeError, ValueError):
        return 0
    if retention_hours <= 0:
        return 0
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    ).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    cursor = self.execute_query(
        (
            "DELETE FROM claims_analytics_exports "
            "WHERE user_id = ? AND status = ? AND updated_at < ?"
        ),
        (str(user_id), "ready", cutoff),
        commit=True,
    )
    return _affected_rows(cursor)


def delete_claims_analytics_exports(
    self,
    *,
    user_id: str,
    export_ids: list[str],
    updated_before: str,
) -> int:
    normalized_ids = list(dict.fromkeys(str(export_id) for export_id in export_ids))
    if not normalized_ids:
        return 0
    deleted = 0
    for start in range(0, len(normalized_ids), _EXPORT_DELETE_CHUNK_SIZE):
        chunk = normalized_ids[start:start + _EXPORT_DELETE_CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "DELETE FROM claims_analytics_exports "
            "WHERE user_id = ? "
            f"AND export_id IN ({placeholders}) "  # nosec B608
            "AND updated_at < ?"
        )
        params: list[Any] = [str(user_id)]
        params.extend(chunk)
        params.append(str(updated_before))
        cursor = self.execute_query(query, tuple(params), commit=True)
        deleted += _affected_rows(cursor)
    return deleted
