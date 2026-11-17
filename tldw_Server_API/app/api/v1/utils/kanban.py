from __future__ import annotations

"""
Shared utilities for Kanban API endpoints.
"""

from datetime import datetime, timezone


def to_db_timestamp(value: datetime | None) -> str | None:
    """Convert datetime to DB-friendly timestamp string."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def resolve_limit_offset(
    *,
    limit: int,
    offset: int,
    page: int | None = None,
    per_page: int | None = None,
) -> tuple[int, int]:
    """
    Resolve pagination params with compatibility support.

    Canonical pagination in Kanban endpoints is `limit` + `offset`.
    Legacy/PRD-compatible `page` + `per_page` can also be provided and will
    be translated to canonical values.
    """
    resolved_limit = per_page if per_page is not None else limit
    resolved_offset = offset
    if page is not None:
        resolved_offset = (page - 1) * resolved_limit
    return resolved_limit, resolved_offset


__all__ = ["to_db_timestamp", "resolve_limit_offset"]
