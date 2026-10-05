"""Claims-owned aggregation of explicit UTC review-metrics windows."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.claims_analytics_export_contract import (
    is_routable_claims_owner_id_text,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def aggregate_claims_review_metrics_window(
    *,
    db: MediaDatabase,
    owner_user_id: str,
    start_date: date,
    end_date: date,
) -> int:
    """Persist every observed group atomically without deleting historical groups.

    The caller supplies the owner DB and any privileged PostgreSQL maintenance
    scope. All SQL, including owner filtering and serialization, stays in the DB
    layer; this function only validates and assembles domain metric records.

    Args:
        db: Open owner database session supporting transactional metric writes.
        owner_user_id: Canonical positive decimal owner identifier.
        start_date: Inclusive first UTC review date, supplied as a date value.
        end_date: Inclusive last UTC review date, supplied as a date value.

    Returns:
        Number of observed date/extractor/version groups upserted. Returns zero
        when no reviews match; historical groups absent from the window remain.

    Raises:
        ValueError: If the owner is not canonical, either date is not a date
            value, or the window is reversed, exceeds 366 days, or ends at
            date.max (the exclusive upper bound must be representable).
        Exception: Database read, lock, or write failures propagate to the
            caller after the database transaction rolls back.
    """
    if not is_routable_claims_owner_id_text(owner_user_id):
        raise ValueError("review metrics owner must be a canonical positive integer")
    if type(start_date) is not date or type(end_date) is not date:
        raise ValueError("review metrics dates must be date values")
    if start_date > end_date or (end_date - start_date).days >= 366 or end_date == date.max:
        raise ValueError("review metrics window must contain 1..366 representable days")

    with db.transaction():
        db.lock_claims_review_metrics_owner(owner_user_id=owner_user_id)
        rows = db.get_claims_review_metrics_window_rows(
            owner_user_id=owner_user_id,
            start_date=start_date,
            end_date=end_date,
        )
        reasons: dict[tuple[str, str, str], dict[str, int]] = {}
        groups = []
        for row in rows:
            day = row["day"]
            day_text = day.isoformat() if isinstance(day, date) else str(day)
            key = (day_text, str(row["extractor"] or "unknown"), str(row["extractor_version"] or ""))
            if row["kind"] == "metrics":
                groups.append((key, row))
            elif row["kind"] == "reason" and row["reason_code"] is not None:
                reason = str(row["reason_code"]).strip()
                if reason:
                    counts = reasons.setdefault(key, {})
                    counts[reason] = counts.get(reason, 0) + int(row["reason_count"])

        for (day, extractor, version), row in groups:
            counts = reasons.get((day, extractor, version))
            db.upsert_claims_review_extractor_metrics_daily(
                user_id=owner_user_id,
                report_date=day,
                extractor=extractor,
                extractor_version=version,
                total_reviewed=int(row["total_reviewed"]),
                approved_count=int(row["approved_count"]),
                rejected_count=int(row["rejected_count"]),
                flagged_count=int(row["flagged_count"]),
                reassigned_count=int(row["reassigned_count"]),
                edited_count=int(row["edited_count"]),
                reason_code_counts_json=json.dumps(counts, sort_keys=True, separators=(",", ":")) if counts else None,
            )
    return len(groups)
