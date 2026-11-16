# Document Outline/TOC Schemas
# Schemas for document outline/table of contents extraction
#
from __future__ import annotations

from pydantic import BaseModel, Field


class OutlineEntry(BaseModel):
    """A single entry in the document outline/table of contents."""

    level: int = Field(..., ge=1, le=6, description="Heading depth (1-6)")
    title: str = Field(..., description="Title of the outline entry")
    page: int = Field(..., ge=1, description="1-indexed page number")


class DocumentOutlineResponse(BaseModel):
    """Response containing the document's table of contents."""

    media_id: int = Field(..., description="ID of the media item")
    has_outline: bool = Field(..., description="Whether the document has an outline")
    entries: list[OutlineEntry] = Field(
        default_factory=list, description="List of outline entries"
    )
    total_pages: int = Field(..., ge=0, description="Total number of pages")


__all__ = ["OutlineEntry", "DocumentOutlineResponse"]
