from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PaginationMode(str, Enum):
    """Canonical pagination mode discriminator."""

    OFFSET = "offset"
    CURSOR = "cursor"
    PAGE = "page"


class OffsetPaginationMeta(BaseModel):
    """Canonical metadata for offset-based pagination."""

    mode: Literal[PaginationMode.OFFSET] = Field(default=PaginationMode.OFFSET)
    limit: int = Field(..., ge=1, description="Canonical page size.")
    offset: int = Field(..., ge=0, description="Canonical zero-based item offset.")
    total: int | None = Field(default=None, ge=0, description="Total number of matching items, when known.")
    has_more: bool = Field(..., description="Whether another page is available.")
    next_offset: int | None = Field(default=None, ge=0, description="Next canonical offset, or null when exhausted.")


class CursorPaginationMeta(BaseModel):
    """Canonical metadata for cursor-based pagination."""

    mode: Literal[PaginationMode.CURSOR] = Field(default=PaginationMode.CURSOR)
    limit: int = Field(..., ge=1, description="Canonical page size.")
    cursor: str | None = Field(default=None, description="Canonical input cursor.")
    next_cursor: str | None = Field(default=None, description="Next cursor, or null when exhausted.")
    has_more: bool = Field(..., description="Whether another page is available.")


class PagePaginationMeta(BaseModel):
    """Canonical metadata for page-number-based pagination."""

    mode: Literal[PaginationMode.PAGE] = Field(default=PaginationMode.PAGE)
    page: int = Field(..., ge=1, description="Canonical 1-indexed page number.")
    per_page: int = Field(..., ge=1, description="Canonical page size.")
    total: int | None = Field(default=None, ge=0, description="Total number of matching items, when known.")
    total_pages: int | None = Field(default=None, ge=0, description="Total number of available pages, when known.")
    has_more: bool = Field(..., description="Whether another page is available.")


def _declares_field(response: Any, field_name: str) -> bool:
    """Return whether a response object exposes a Pydantic field or attribute."""
    model_fields = getattr(type(response), "model_fields", None)
    if isinstance(model_fields, dict):
        return field_name in model_fields
    return hasattr(response, field_name)


def _default_pagination_aliases(response: Any, aliases: tuple[tuple[str, str], ...]) -> Any:
    """Populate declared top-level aliases from canonical pagination metadata."""
    pagination = getattr(response, "pagination", None)
    if pagination is None:
        return response
    for alias_name, metadata_name in aliases:
        if not _declares_field(response, alias_name):
            continue
        if getattr(response, alias_name, None) is None:
            setattr(response, alias_name, getattr(pagination, metadata_name, None))
    return response


def _validate_pagination_aliases(response: Any, aliases: tuple[tuple[str, str], ...]) -> Any:
    """Populate missing aliases and reject aliases that drift from metadata."""
    pagination = getattr(response, "pagination", None)
    if pagination is None:
        return response
    for alias_name, metadata_name in aliases:
        if not _declares_field(response, alias_name):
            continue
        expected = getattr(pagination, metadata_name, None)
        actual = getattr(response, alias_name, None)
        if actual is None:
            setattr(response, alias_name, expected)
            continue
        if actual != expected:
            raise ValueError(
                f"{alias_name} alias mismatch: {alias_name}={actual} "
                f"pagination.{metadata_name}={expected}"
            )
    return response


_OFFSET_ALIAS_FIELDS: tuple[tuple[str, str], ...] = (
    ("limit", "limit"),
    ("offset", "offset"),
    ("total", "total"),
    ("has_more", "has_more"),
    ("next_offset", "next_offset"),
)
_PAGE_ALIAS_FIELDS: tuple[tuple[str, str], ...] = (
    ("page", "page"),
    ("per_page", "per_page"),
    ("total", "total"),
    ("total_pages", "total_pages"),
    ("has_more", "has_more"),
)
_CURSOR_ALIAS_FIELDS: tuple[tuple[str, str], ...] = (
    ("limit", "limit"),
    ("cursor", "cursor"),
    ("next_cursor", "next_cursor"),
    ("has_more", "has_more"),
)


def default_offset_pagination_aliases(response: Any) -> Any:
    """Populate legacy top-level offset aliases from canonical metadata."""
    return _default_pagination_aliases(response, _OFFSET_ALIAS_FIELDS)


def validate_offset_pagination_aliases(response: Any) -> Any:
    """Populate missing offset aliases and reject contradictions."""
    return _validate_pagination_aliases(response, _OFFSET_ALIAS_FIELDS)


def default_page_pagination_aliases(response: Any) -> Any:
    """Populate legacy top-level page aliases from canonical metadata."""
    return _default_pagination_aliases(response, _PAGE_ALIAS_FIELDS)


def default_cursor_pagination_aliases(response: Any) -> Any:
    """Populate legacy top-level cursor aliases from canonical metadata."""
    return _default_pagination_aliases(response, _CURSOR_ALIAS_FIELDS)
