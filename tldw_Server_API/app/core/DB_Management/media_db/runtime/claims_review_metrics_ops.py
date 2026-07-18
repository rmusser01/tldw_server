"""Package-owned claims review metrics helpers."""

from __future__ import annotations

import math
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _claims_review_tables(self) -> tuple[str, str, str]:
    if self.backend_type == BackendType.POSTGRESQL:
        return "claims", "media", "%s"
    return "Claims", "Media", "?"


def _claims_review_owner_join(
    self,
    owner_user_id: str | None,
) -> tuple[str, str, list[Any]]:
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
    version = "" if extractor_version is None else str(extractor_version)
    now = self._get_current_utc_timestamp_str()
    existing = self.execute_query(
        "SELECT id FROM claims_review_extractor_metrics_daily "
        "WHERE user_id = ? AND report_date = ? AND extractor = ? AND extractor_version = ?",
        (
            str(user_id),
            str(report_date),
            str(extractor),
            version,
        ),
    ).fetchone()
    existing_id: int | None = None
    if existing is not None:
        try:
            existing_id = int(existing["id"])
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            try:
                existing_id = int(existing[0])
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                existing_id = None

    if existing_id is None:
        insert_sql = (
            "INSERT INTO claims_review_extractor_metrics_daily "
            "(user_id, report_date, extractor, extractor_version, total_reviewed, approved_count, "
            "rejected_count, flagged_count, reassigned_count, edited_count, reason_code_counts_json, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        self.execute_query(
            insert_sql,
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
            commit=True,
        )
        return self.get_claims_review_extractor_metrics_daily(
            user_id=str(user_id),
            report_date=str(report_date),
            extractor=str(extractor),
            extractor_version=version,
        )

    self.execute_query(
        (
            "UPDATE claims_review_extractor_metrics_daily SET "
            "total_reviewed = ?, approved_count = ?, rejected_count = ?, flagged_count = ?, "
            "reassigned_count = ?, edited_count = ?, reason_code_counts_json = ?, updated_at = ? "
            "WHERE id = ?"
        ),
        (
            int(total_reviewed),
            int(approved_count),
            int(rejected_count),
            int(flagged_count),
            int(reassigned_count),
            int(edited_count),
            reason_code_counts_json,
            now,
            int(existing_id),
        ),
        commit=True,
    )
    return self.get_claims_review_extractor_metrics_daily(
        user_id=str(user_id),
        report_date=str(report_date),
        extractor=str(extractor),
        extractor_version=version,
    )


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
