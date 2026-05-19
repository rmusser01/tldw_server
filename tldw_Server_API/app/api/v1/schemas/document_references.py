# Document References Schemas
# Schemas for bibliography/reference extraction from documents
#
from __future__ import annotations

from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


class ReferenceEntry(BaseModel):
    """A single reference/citation extracted from the document."""

    raw_text: str = Field(..., description="Original reference text as found in document")
    title: str | None = Field(None, description="Parsed title of the referenced work")
    authors: str | None = Field(None, description="Authors of the referenced work")
    year: int | None = Field(None, ge=1000, le=2100, description="Publication year")
    venue: str | None = Field(None, description="Journal, conference, or publisher")
    doi: str | None = Field(None, description="Digital Object Identifier")
    arxiv_id: str | None = Field(None, description="arXiv paper ID (e.g., 2301.12345)")
    url: str | None = Field(None, description="URL to the referenced work")
    # Enriched fields from external APIs
    citation_count: int | None = Field(None, ge=0, description="Citation count from external API")
    semantic_scholar_id: str | None = Field(None, description="Semantic Scholar paper ID")
    open_access_pdf: str | None = Field(None, description="URL to open access PDF")


class DocumentReferencesResponse(BaseModel):
    """Response containing references extracted from a document."""

    media_id: int = Field(..., description="ID of the media item")
    has_references: bool = Field(..., description="Whether references were found")
    references: list[ReferenceEntry] = Field(
        default_factory=list, description="List of extracted references"
    )
    enrichment_source: str | None = Field(
        None,
        description="External API used for enrichment (semantic_scholar, crossref, arxiv)",
    )
    enriched_count: int = Field(
        0,
        ge=0,
        description="Number of references modified by external enrichment during this request",
    )
    enrichment_limited: bool = Field(
        False,
        description=(
            "True when enrichment was intentionally capped (for example, first N references only)"
        ),
    )
    total_detected: int = Field(
        0,
        ge=0,
        description="Total references detected before response limits are applied",
    )
    truncated: bool = Field(
        False,
        description="True when detected references exceeded the response cap",
    )
    offset: int = Field(
        0,
        ge=0,
        description="Start offset of the current page in the post-cap reference list",
    )
    limit: int = Field(
        0,
        ge=0,
        description="Requested page size for this response",
    )
    returned_count: int = Field(
        0,
        ge=0,
        description="Number of references returned in this page",
    )
    total_available: int = Field(
        0,
        ge=0,
        description="Total references available after applying parse cap (if any)",
    )
    has_more: bool = Field(
        False,
        description="Whether more references are available beyond this page",
    )
    next_offset: int | None = Field(
        None,
        ge=0,
        description="Offset to request for the next page when has_more=true",
    )
    pagination: OffsetPaginationMeta


__all__ = ["ReferenceEntry", "DocumentReferencesResponse"]
