from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PaginationMode(str, Enum):
    """Canonical pagination mode discriminator."""

    OFFSET = "offset"
    CURSOR = "cursor"


class OffsetPaginationMeta(BaseModel):
    """Canonical metadata for offset-based pagination."""

    mode: PaginationMode = Field(default=PaginationMode.OFFSET)
    limit: int = Field(..., ge=1, description="Canonical page size.")
    offset: int = Field(..., ge=0, description="Canonical zero-based item offset.")
    total: int | None = Field(default=None, ge=0, description="Total number of matching items, when known.")
    has_more: bool = Field(..., description="Whether another page is available.")
    next_offset: int | None = Field(default=None, ge=0, description="Next canonical offset, or null when exhausted.")


class CursorPaginationMeta(BaseModel):
    """Canonical metadata for cursor-based pagination."""

    mode: PaginationMode = Field(default=PaginationMode.CURSOR)
    limit: int = Field(..., ge=1, description="Canonical page size.")
    cursor: str | None = Field(default=None, description="Canonical input cursor.")
    next_cursor: str | None = Field(default=None, description="Next cursor, or null when exhausted.")
    has_more: bool = Field(..., description="Whether another page is available.")
