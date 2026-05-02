"""Shared pagination metadata builders for API routes and services."""

from __future__ import annotations

from tldw_Server_API.app.api.v1.schemas.pagination import (
    CursorPaginationMeta,
    OffsetPaginationMeta,
    PagePaginationMeta,
)


def build_offset_pagination_meta(
    *,
    limit: int,
    offset: int,
    total: int | None = None,
    count: int | None = None,
    has_more: bool | None = None,
) -> OffsetPaginationMeta:
    """Build canonical offset pagination metadata from route-local values."""
    normalized_count = max(count or 0, 0)

    if has_more is None:
        if total is not None:
            has_more = offset + normalized_count < total
        else:
            has_more = normalized_count >= limit

    next_offset = offset + limit if has_more else None

    return OffsetPaginationMeta(
        limit=limit,
        offset=offset,
        total=total,
        has_more=has_more,
        next_offset=next_offset,
    )


def build_cursor_pagination_meta(
    *,
    limit: int,
    cursor: str | None = None,
    next_cursor: str | None = None,
    has_more: bool | None = None,
) -> CursorPaginationMeta:
    """Build canonical cursor pagination metadata from route-local values."""
    normalized_has_more = bool(next_cursor) if has_more is None else bool(has_more)
    return CursorPaginationMeta(
        limit=limit,
        cursor=cursor,
        next_cursor=next_cursor,
        has_more=normalized_has_more,
    )


def build_page_pagination_meta(
    *,
    page: int,
    per_page: int,
    total: int | None = None,
    total_pages: int | None = None,
    has_more: bool | None = None,
) -> PagePaginationMeta:
    """Build canonical page-based pagination metadata from route-local values."""
    normalized_page = int(page)
    normalized_per_page = int(per_page)
    normalized_total = int(total) if total is not None else None
    normalized_total_pages = int(total_pages) if total_pages is not None else None
    if has_more is None:
        if normalized_total_pages is not None:
            normalized_has_more = normalized_page < normalized_total_pages
        elif normalized_total is not None and normalized_per_page > 0:
            expected_total_pages = (normalized_total + normalized_per_page - 1) // normalized_per_page
            normalized_has_more = normalized_page < expected_total_pages
        else:
            normalized_has_more = False
    else:
        normalized_has_more = bool(has_more)
    return PagePaginationMeta(
        page=normalized_page,
        per_page=normalized_per_page,
        total=normalized_total,
        total_pages=normalized_total_pages,
        has_more=normalized_has_more,
    )
