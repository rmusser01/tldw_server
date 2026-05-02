"""
Reusable helpers for building RFC5988 Link headers for paginated endpoints.

This module centralizes pagination metadata and Link header construction for
API endpoints while keeping dependencies limited to shared pagination schemas.
"""

from __future__ import annotations

import urllib.parse as _u

from tldw_Server_API.app.api.v1.schemas.pagination import (
    CursorPaginationMeta,
    OffsetPaginationMeta,
    PagePaginationMeta,
)


def build_link_header(
    base_path: str,
    common_params: list[tuple[str, str]] | None = None,
    *,
    next_cursor: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    has_more: bool | None = None,
    cursor_param: str = "cursor",
    include_first_last: bool = True,
) -> str | None:
    """Build an RFC5988 Link header string for pagination.

    - Cursor mode: when `next_cursor` is provided, returns a `rel="next"` link using
      the given `cursor_param`. If `limit` is provided it is included as a query param.
    - Offset mode: when `limit` and `offset` are provided, returns `rel="next"`,
      `rel="prev"`, and when `include_first_last=True`, `rel="first"` (offset=0)
      and a best-effort `rel="last"` when `has_more is False`.

    The helper is tolerant and will only include links it can build from provided inputs.
    Returns a comma-separated Link header value or None if no links are applicable.

    Example (cursor-based):
        >>> build_link_header(
        ...     base_path="/api/v1/workflows/runs",
        ...     common_params=[("status", "running"), ("order_by", "created_at")],
        ...     next_cursor="abc123",
        ...     limit=25,
        ... )
        '</api/v1/workflows/runs?status=running&order_by=created_at&limit=25&cursor=abc123>; rel="next"'

    Example (offset-based):
        >>> build_link_header(
        ...     base_path="/api/v1/workflows/runs",
        ...     common_params=[("status", "running")],
        ...     limit=25,
        ...     offset=50,
        ...     has_more=True,
        ... )
        '</api/v1/workflows/runs?status=running&limit=25&offset=75>; rel="next", '
        '</api/v1/workflows/runs?status=running&limit=25&offset=25>; rel="prev", '
        '</api/v1/workflows/runs?status=running&limit=25&offset=0>; rel="first"'
    """
    params_common: list[tuple[str, str]] = list(common_params or [])
    links: list[str] = []

    # Cursor-based next link
    if next_cursor:
        q = params_common + [("limit", str(limit))] if limit is not None else list(params_common)
        q.append((cursor_param, next_cursor))
        href = base_path + "?" + _u.urlencode(q, doseq=True)
        links.append(f"<{href}>; rel=\"next\"")

    # Offset-based links
    if limit is not None and offset is not None:
        # Next
        if has_more:
            qn = params_common + [("limit", str(limit)), ("offset", str(int(offset) + int(limit)))]
            hrefn = base_path + "?" + _u.urlencode(qn, doseq=True)
            links.append(f"<{hrefn}>; rel=\"next\"")
        # Prev
        if int(offset) > 0:
            prev_off = max(0, int(offset) - int(limit))
            qp = params_common + [("limit", str(limit)), ("offset", str(prev_off))]
            hrefp = base_path + "?" + _u.urlencode(qp, doseq=True)
            links.append(f"<{hrefp}>; rel=\"prev\"")
        # First/Last (best-effort)
        if include_first_last:
            qf = params_common + [("limit", str(limit)), ("offset", "0")]
            hreff = base_path + "?" + _u.urlencode(qf, doseq=True)
            links.append(f"<{hreff}>; rel=\"first\"")
            # We don't know total; treat current page as last when not has_more
            if has_more is False:
                ql = params_common + [("limit", str(limit)), ("offset", str(offset))]
                hrefl = base_path + "?" + _u.urlencode(ql, doseq=True)
                links.append(f"<{hrefl}>; rel=\"last\"")

    return ", ".join(links) if links else None


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
    normalized_total_pages = int(total_pages) if total_pages is not None else None
    normalized_has_more = (
        bool(normalized_total_pages and int(page) < normalized_total_pages)
        if has_more is None
        else bool(has_more)
    )
    return PagePaginationMeta(
        page=int(page),
        per_page=int(per_page),
        total=int(total) if total is not None else None,
        total_pages=normalized_total_pages,
        has_more=normalized_has_more,
    )


def build_pagination_link_header(
    base_path: str,
    common_params: list[tuple[str, str]] | None = None,
    *,
    pagination: OffsetPaginationMeta | CursorPaginationMeta,
    cursor_param: str = "cursor",
    include_first_last: bool = True,
) -> str | None:
    """Build an RFC5988 Link header from canonical pagination metadata."""
    if isinstance(pagination, OffsetPaginationMeta):
        return build_link_header(
            base_path=base_path,
            common_params=common_params,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=pagination.has_more,
            include_first_last=include_first_last,
        )

    return build_link_header(
        base_path=base_path,
        common_params=common_params,
        next_cursor=pagination.next_cursor,
        limit=pagination.limit,
        cursor_param=cursor_param,
    )


__all__ = [
    "build_cursor_pagination_meta",
    "build_link_header",
    "build_page_pagination_meta",
    "build_offset_pagination_meta",
    "build_pagination_link_header",
]
