"""Package-owned claims review metrics helpers."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _claims_review_window_bounds(self, start_date: date, end_date: date) -> tuple[Any, Any]:
    """Return half-open UTC parameters in the backend's stored timestamp format."""
    if type(start_date) is not date or type(end_date) is not date:
        raise ValueError("review metrics dates must be date values")
    if start_date > end_date or (end_date - start_date).days >= 366 or end_date == date.max:
        raise ValueError("review metrics window must contain 1..366 representable days")
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if self.backend_type == BackendType.POSTGRESQL:
        return start, end
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


def lock_claims_review_metrics_owner(self, *, owner_user_id: str) -> None:
    """Serialize PostgreSQL calculations for an owner until the window commits."""
    if self.backend_type != BackendType.POSTGRESQL:
        return
    connection = self._get_txn_conn()
    if connection is None:
        raise RuntimeError("review metrics owner lock requires an active transaction")
    self.execute_query(
        "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
        (f"claims-review-metrics:{owner_user_id}",),
        connection=connection,
    )


def get_claims_review_metrics_window_rows(
    self,
    *,
    owner_user_id: str,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Read metric and reason groups together from one source statement snapshot."""
    start, end = _claims_review_window_bounds(self, start_date, end_date)
    postgresql = self.backend_type == BackendType.POSTGRESQL
    day = "(l.created_at AT TIME ZONE 'UTC')::date" if postgresql else "DATE(l.created_at)"
    claims_table, media_table, _placeholder = _claims_review_tables(self)
    owner_predicate = ""
    params = [start, end]
    if postgresql:
        owner_predicate = " AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?"
        params.append(owner_user_id)

    # The SELECT wrapper also lets SQLite's cursor adapter collect CTE results.
    query = (
        "SELECT * FROM (WITH filtered AS (SELECT "  # nosec B608
        + day
        + " AS day, COALESCE(c.extractor, 'unknown') AS extractor, "
        "COALESCE(c.extractor_version, '') AS extractor_version, "
        "l.new_status, l.old_text, l.new_text, l.reason_code "
        "FROM claims_review_log l "
        f"LEFT JOIN {claims_table} c ON c.id = l.claim_id "
        f"LEFT JOIN {media_table} m ON m.id = c.media_id "
        "WHERE l.created_at >= ? AND l.created_at < ?" + owner_predicate + ") "
        "SELECT 'metrics' AS kind, day, extractor, extractor_version, COUNT(*) AS total_reviewed, "
        "SUM(CASE WHEN lower(new_status) = 'approved' THEN 1 ELSE 0 END) AS approved_count, "
        "SUM(CASE WHEN lower(new_status) = 'rejected' THEN 1 ELSE 0 END) AS rejected_count, "
        "SUM(CASE WHEN lower(new_status) = 'flagged' THEN 1 ELSE 0 END) AS flagged_count, "
        "SUM(CASE WHEN lower(new_status) = 'reassigned' THEN 1 ELSE 0 END) AS reassigned_count, "
        "SUM(CASE WHEN old_text IS NOT NULL AND new_text IS NOT NULL AND old_text <> new_text "
        "THEN 1 ELSE 0 END) AS edited_count, CAST(NULL AS TEXT) AS reason_code, "
        "CAST(0 AS BIGINT) AS reason_count FROM filtered GROUP BY day, extractor, extractor_version "
        "UNION ALL SELECT 'reason' AS kind, day, extractor, extractor_version, "
        "0, 0, 0, 0, 0, 0, reason_code, COUNT(*) AS reason_count "
        "FROM filtered GROUP BY day, extractor, extractor_version, reason_code) AS aggregates "
        "ORDER BY day, extractor, extractor_version, kind, reason_code"
    )
    return [dict(row) for row in self.execute_query(query, tuple(params)).fetchall()]


def list_claims_review_user_ids_page(
    self,
    *,
    start_date: date,
    end_date: date,
    after_user_id: str | None = None,
    limit: int = 100,
) -> list[str]:
    """Return a bounded, date-filtered PostgreSQL owner page in text key order."""
    if self.backend_type != BackendType.POSTGRESQL:
        return []
    start, end = _claims_review_window_bounds(self, start_date, end_date)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(100, limit))
    owner = "COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id)"
    cursor_predicate = f" AND {owner} > ?" if after_user_id is not None else ""
    params = [start, end]
    if after_user_id is not None:
        params.append(after_user_id)
    params.append(limit)
    query = (
        f"SELECT DISTINCT {owner} AS user_id FROM claims_review_log l "  # nosec B608
        "JOIN claims c ON c.id = l.claim_id JOIN media m ON m.id = c.media_id "
        "WHERE l.created_at >= ? AND l.created_at < ? "
        f"AND {owner} IS NOT NULL AND {owner} <> ''" + cursor_predicate + " ORDER BY user_id ASC LIMIT ?"
    )
    return [str(row["user_id"]) for row in self.execute_query(query, tuple(params)).fetchall()]


def _claims_review_tables(self) -> tuple[str, str, str]:
    """Return backend-specific Claims review metric table names."""
    if self.backend_type == BackendType.POSTGRESQL:
        return "claims", "media", "%s"
    return "Claims", "Media", "?"


def _claims_review_owner_join(
    self,
    owner_user_id: str | None,
) -> tuple[str, str, list[Any]]:
    """Build optional owner scoping SQL for review metric queries."""
    if not owner_user_id:
        return "", "", []
    _claims_table, media_table, placeholder = _claims_review_tables(self)
    return (
        f" JOIN {media_table} m ON m.id = c.media_id",  # nosec B608
        f" AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = {placeholder}",
        [str(owner_user_id)],
    )


def get_claims_review_latency_stats(
    self,
    *,
    owner_user_id: str | None = None,
) -> dict[str, float | None]:
    """Compute review latency aggregates for completed Claims reviews."""
    claims_table, _media_table, placeholder = _claims_review_tables(self)
    owner_join, owner_predicate, owner_params = _claims_review_owner_join(self, owner_user_id)
    if self.backend_type == BackendType.POSTGRESQL:
        latency_expr = "EXTRACT(EPOCH FROM (c.reviewed_at - c.created_at))"
    else:
        latency_expr = "(julianday(c.reviewed_at) - julianday(c.created_at)) * 86400.0"

    avg_row = self.execute_query(
        f"SELECT AVG({latency_expr}) AS avg_sec "  # nosec B608
        f"FROM {claims_table} c{owner_join} WHERE c.reviewed_at IS NOT NULL AND c.deleted = 0"
        + owner_predicate,
        tuple(owner_params),
    ).fetchone()
    avg_latency_sec = None
    if avg_row:
        try:
            avg_latency_sec = float(avg_row[0]) if avg_row[0] is not None else None
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            avg_latency_sec = None

    total_row = self.execute_query(
        f"SELECT COUNT(*) AS count FROM {claims_table} c{owner_join} "  # nosec B608
        "WHERE c.reviewed_at IS NOT NULL AND c.deleted = 0"
        + owner_predicate,
        tuple(owner_params),
    ).fetchone()
    try:
        total = int(total_row[0]) if total_row and total_row[0] is not None else 0
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        total = 0

    p95_latency_sec = None
    if total > 0:
        offset = max(0, int(math.ceil(total * 0.95)) - 1)
        row = self.execute_query(
            "SELECT "
            + latency_expr
            + f" AS latency FROM {claims_table} c{owner_join} "  # nosec B608
            "WHERE c.reviewed_at IS NOT NULL AND c.deleted = 0"
            + owner_predicate
            + f" ORDER BY {latency_expr} LIMIT 1 OFFSET {placeholder}",
            (*owner_params, offset),
        ).fetchone()
        if row:
            try:
                p95_latency_sec = float(row[0]) if row[0] is not None else None
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                p95_latency_sec = None

    return {
        "avg_review_latency_sec": avg_latency_sec,
        "p95_review_latency_sec": p95_latency_sec,
    }


def get_claims_review_extractor_metrics_daily(
    self,
    *,
    user_id: str,
    report_date: str,
    extractor: str,
    extractor_version: str | None = None,
) -> dict[str, Any]:
    version = "" if extractor_version is None else str(extractor_version)
    row = self.execute_query(
        (
            "SELECT id, user_id, report_date, extractor, extractor_version, total_reviewed, "
            "approved_count, rejected_count, flagged_count, reassigned_count, edited_count, "
            "reason_code_counts_json, created_at, updated_at "
            "FROM claims_review_extractor_metrics_daily "
            "WHERE user_id = ? AND report_date = ? AND extractor = ? AND extractor_version = ?"
        ),
        (
            str(user_id),
            str(report_date),
            str(extractor),
            version,
        ),
    ).fetchone()
    return dict(row) if row else {}


def upsert_claims_review_extractor_metrics_daily(
    self,
    *,
    user_id: str,
    report_date: str,
    extractor: str,
    extractor_version: str | None = None,
    total_reviewed: int = 0,
    approved_count: int = 0,
    rejected_count: int = 0,
    flagged_count: int = 0,
    reassigned_count: int = 0,
    edited_count: int = 0,
    reason_code_counts_json: str | None = None,
) -> dict[str, Any]:
    """Atomically insert/update one group, joining any caller-owned transaction."""
    version = "" if extractor_version is None else str(extractor_version)
    now = self._get_current_utc_timestamp_str()
    with self.transaction() as connection:
        row = self.execute_query(
            "INSERT INTO claims_review_extractor_metrics_daily "
            "(user_id, report_date, extractor, extractor_version, total_reviewed, approved_count, "
            "rejected_count, flagged_count, reassigned_count, edited_count, reason_code_counts_json, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, report_date, extractor, extractor_version) DO UPDATE SET "
            "total_reviewed = excluded.total_reviewed, approved_count = excluded.approved_count, "
            "rejected_count = excluded.rejected_count, flagged_count = excluded.flagged_count, "
            "reassigned_count = excluded.reassigned_count, edited_count = excluded.edited_count, "
            "reason_code_counts_json = excluded.reason_code_counts_json, updated_at = excluded.updated_at "
            "RETURNING *",
            (
                str(user_id),
                str(report_date),
                str(extractor),
                version,
                int(total_reviewed),
                int(approved_count),
                int(rejected_count),
                int(flagged_count),
                int(reassigned_count),
                int(edited_count),
                reason_code_counts_json,
                now,
                now,
            ),
            connection=connection,
        ).fetchone()
    return dict(row) if row else {}


def list_claims_review_extractor_metrics_daily(
    self,
    *,
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    extractor: str | None = None,
    extractor_version: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 500, 0
    limit = max(1, min(5000, limit))
    offset = max(0, offset)

    conditions: list[str] = ["user_id = ?"]
    params: list[Any] = [str(user_id)]
    if start_date:
        conditions.append("report_date >= ?")
        params.append(str(start_date))
    if end_date:
        conditions.append("report_date <= ?")
        params.append(str(end_date))
    if extractor:
        conditions.append("extractor = ?")
        params.append(str(extractor))
    if extractor_version is not None:
        conditions.append("extractor_version = ?")
        params.append(str(extractor_version))

    sql = (
        "SELECT id, user_id, report_date, extractor, extractor_version, total_reviewed, "  # nosec B608
        "approved_count, rejected_count, flagged_count, reassigned_count, edited_count, "
        "reason_code_counts_json, created_at, updated_at "
        "FROM claims_review_extractor_metrics_daily WHERE "
        + " AND ".join(conditions)
        + " ORDER BY report_date DESC, id DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def count_claims_review_extractor_metrics_daily(
    self,
    *,
    user_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    extractor: str | None = None,
    extractor_version: str | None = None,
) -> int:
    conditions: list[str] = ["user_id = ?"]
    params: list[Any] = [str(user_id)]
    if start_date:
        conditions.append("report_date >= ?")
        params.append(str(start_date))
    if end_date:
        conditions.append("report_date <= ?")
        params.append(str(end_date))
    if extractor:
        conditions.append("extractor = ?")
        params.append(str(extractor))
    if extractor_version is not None:
        conditions.append("extractor_version = ?")
        params.append(str(extractor_version))

    sql = (
        "SELECT COUNT(*) AS total FROM claims_review_extractor_metrics_daily WHERE "  # nosec B608
        + " AND ".join(conditions)
    )
    row = self.execute_query(sql, tuple(params)).fetchone()
    if not row:
        return 0
    value = row.get("total") if isinstance(row, dict) else row[0]
    return int(value or 0)


def list_claims_review_user_ids(self) -> list[str]:
    """Return distinct user IDs with review log activity (Postgres only)."""
    if self.backend_type != BackendType.POSTGRESQL:
        return []
    rows = self.execute_query(
        (
            "SELECT DISTINCT COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) AS user_id "
            "FROM claims_review_log l "
            "LEFT JOIN claims c ON c.id = l.claim_id "
            "LEFT JOIN media m ON m.id = c.media_id"
        ),
        (),
    ).fetchall()
    user_ids: list[str] = []
    for row in rows:
        try:
            user_id = row["user_id"]
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            try:
                user_id = row[0]
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                user_id = None
        if user_id is None:
            continue
        user_ids.append(str(user_id))
    return [uid for uid in user_ids if uid]
