"""
Reusable helpers for building RFC5988 Link headers for paginated endpoints.

This module centralizes pagination metadata and Link header construction for
API endpoints while keeping dependencies limited to shared pagination schemas.
"""

from __future__ import annotations

import urllib.parse as _u
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.api.v1.schemas.pagination import (
    CursorPaginationMeta,
    OffsetPaginationMeta,
)
from tldw_Server_API.app.api.v1.utils.pagination import (
    build_cursor_pagination_meta,
    build_offset_pagination_meta,
    build_page_pagination_meta,
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


def resolve_page_pagination_metadata(
    pagination_data: Mapping[str, Any] | None,
    *,
    page: int,
    per_page: int,
    item_count: int,
) -> dict[str, int]:
    """Normalize page pagination data from storage, defaulting missing totals safely."""
    pagination = pagination_data or {}

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    page_value = max(1, _as_int(pagination.get("page"), page))
    per_page_value = max(1, _as_int(pagination.get("per_page") or pagination.get("limit"), per_page))
    total_value = max(0, _as_int(pagination.get("total"), item_count))
    total_pages_value = pagination.get("total_pages")
    if total_pages_value is None:
        total_pages_value = (total_value + per_page_value - 1) // per_page_value if total_value else 0
    total_pages_value = max(0, _as_int(total_pages_value, 0))

    return {
        "page": page_value,
        "per_page": per_page_value,
        "total": total_value,
        "total_pages": total_pages_value,
    }


__all__ = [
    "build_cursor_pagination_meta",
    "build_link_header",
    "build_page_pagination_meta",
    "build_offset_pagination_meta",
    "build_pagination_link_header",
    "resolve_page_pagination_metadata",
]
