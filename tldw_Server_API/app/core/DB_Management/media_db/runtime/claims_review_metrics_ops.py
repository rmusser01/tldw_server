"""Package-owned claims review metrics helpers."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)


_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


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
