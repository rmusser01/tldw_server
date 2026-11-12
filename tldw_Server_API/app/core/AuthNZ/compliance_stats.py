"""Aggregate counts behind the admin compliance-posture dashboard (TASK-13317).

Moved out of the admin endpoint. Each function returns ``(total, count)``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _pair(row: Any, first: str, second: str) -> tuple[int, int]:
    if not row:
        return 0, 0
    values = dict(row) if hasattr(row, "keys") or isinstance(row, dict) else {first: row[0], second: row[1]}
    return int(values.get(first) or 0), int(values.get(second) or 0)


async def mfa_adoption(pool: Any) -> tuple[int, int]:
    """(active users, active users with two-factor enabled)."""
    if getattr(pool, "pool", None):
        row = await pool.fetchone(
            "SELECT COUNT(*) AS total,"
            " COUNT(*) FILTER (WHERE COALESCE(two_factor_enabled, FALSE) = TRUE) AS mfa_on"
            " FROM users WHERE is_active = TRUE"
        )
    else:
        row = await pool.fetchone(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN COALESCE(two_factor_enabled, 0) = 1 THEN 1 ELSE 0 END) AS mfa_on"
            " FROM users WHERE is_active = 1"
        )
    return _pair(row, "total", "mfa_on")


async def api_key_rotation(pool: Any, threshold_days: int) -> tuple[int, int]:
    """(active API keys, those created within ``threshold_days``)."""
    threshold = datetime.now(timezone.utc) - timedelta(days=threshold_days)
    if getattr(pool, "pool", None):
        row = await pool.fetchone(
            "SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE created_at >= $1) AS compliant"
            " FROM api_keys WHERE status = 'active'",
            threshold,
        )
    else:
        row = await pool.fetchone(
            "SELECT COUNT(*) AS total, SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS compliant"
            " FROM api_keys WHERE status = 'active'",
            threshold.isoformat(),
        )
    return _pair(row, "total", "compliant")
