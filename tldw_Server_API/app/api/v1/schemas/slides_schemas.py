"""Pydantic models for Slides/Presentation module."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class SlideLayout(str, Enum):
    """Supported slide layout identifiers."""

    TITLE = "title"
    CONTENT = "content"
    TWO_COLUMN = "two_column"
    QUOTE = "quote"
    SECTION = "section"
    BLANK = "blank"


class Slide(BaseModel):
    """Slide payload for presentation content."""

    order: int
    layout: SlideLayout
    title: str | None = None
    content: str = ""
    speaker_notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _validate_visual_style_selection_pair(
    *,
    visual_style_id: str | None,
    visual_style_scope: str | None,
) -> None:
    """Require visual-style id and scope to be set or cleared together."""

    if (visual_style_id is None) != (visual_style_scope is None):
        raise ValueError("visual_style_id and visual_style_scope must be provided together")


class VisualStyleSelectionMixin(BaseModel):
    """Shared validation for presentation-level visual-style selection."""

    visual_style_id: str | None = None
    visual_style_scope: str | None = None

    @model_validator(mode="after")
    def _validate_visual_style_selection(self) -> VisualStyleSelectionMixin:
        _validate_visual_style_selection_pair(
            visual_style_id=self.visual_style_id,
            visual_style_scope=self.visual_style_scope,
        )
        return self


class PresentationBase(VisualStyleSelectionMixin):
    """Shared fields for presentation create/update payloads."""

    title: str
    description: str | None = None
    theme: str = "black"
    marp_theme: str | None = None
    template_id: str | None = None
    settings: dict[str, Any] | None = None
    studio_data: dict[str, Any] | None = None
    slides: list[Slide] = Field(default_factory=list)
    custom_css: str | None = None


class PresentationCreateRequest(PresentationBase):
    """Request payload for creating a presentation."""

    pass


class PresentationUpdateRequest(PresentationBase):
    """Request payload for updating a presentation."""

    pass


class PresentationPatchRequest(VisualStyleSelectionMixin):
    """Request payload for patching a presentation."""

    title: str | None = None
    description: str | None = None
    theme: str | None = None
    marp_theme: str | None = None
    template_id: str | None = None
    settings: dict[str, Any] | None = None
    studio_data: dict[str, Any] | None = None
    slides: list[Slide] | None = None
    custom_css: str | None = None


class PresentationReorderRequest(BaseModel):
    """Request payload for reordering presentation slides."""

    order: list[int] = Field(..., min_items=1)


class PresentationResponse(PresentationBase):
    """Presentation response model."""

    id: str
    visual_style_name: str | None = None
    visual_style_version: int | None = None
    visual_style_snapshot: dict[str, Any] | None = None
    source_type: str | None = None
    source_ref: Any | None = None
    source_query: str | None = None
    created_at: datetime
    last_modified: datetime
    deleted: bool
    client_id: str
    version: int


class PresentationVersionSummary(BaseModel):
    """Summary for a presentation version entry."""

    presentation_id: str
    version: int
    created_at: datetime
    title: str | None = None
    deleted: bool | None = None


class PresentationVersionListResponse(BaseModel):
    """Paginated list of presentation versions."""

    versions: list[PresentationVersionSummary]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class SlidesTemplateResponse(BaseModel):
    """Template details for slide generation."""

    id: str
    name: str
    theme: str
    marp_theme: str | None = None
    settings: dict[str, Any] | None = None
    default_slides: list[Slide] | None = None
    custom_css: str | None = None


class SlidesTemplateListResponse(BaseModel):
    """List response for slide templates."""

    templates: list[SlidesTemplateResponse]


class VisualStyleBase(BaseModel):
    """Structured visual style payload fields."""

    name: str
    description: str | None = None
    generation_rules: dict[str, Any] = Field(default_factory=dict)
    artifact_preferences: list[str] = Field(default_factory=list)
    appearance_defaults: dict[str, Any] = Field(default_factory=dict)
    fallback_policy: dict[str, Any] = Field(default_factory=dict)


class VisualStyleCreateRequest(VisualStyleBase):
    """Request payload for creating a user visual style."""

    pass


class VisualStylePatchRequest(BaseModel):
    """Request payload for patching a user visual style."""

    name: str | None = None
    description: str | None = None
    generation_rules: dict[str, Any] | None = None
    artifact_preferences: list[str] | None = None
    appearance_defaults: dict[str, Any] | None = None
    fallback_policy: dict[str, Any] | None = None


class VisualStyleResponse(VisualStyleBase):
    """Visual style response model for built-in and user presets."""

    id: str
    scope: str
    category: str | None = None
    guide_number: int | None = None
    tags: list[str] | None = None
    best_for: list[str] | None = None
    version: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class VisualStyleListResponse(BaseModel):
    """List response for visual styles."""

    styles: list[VisualStyleResponse]
    total_count: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class PresentationSummary(BaseModel):
    """Summary item for presentation listings."""

    id: str
    title: str
    description: str | None = None
    theme: str
    created_at: datetime
    last_modified: datetime
    deleted: bool
    version: int


class PresentationListResponse(BaseModel):
    """Paginated list response for presentations."""

    presentations: list[PresentationSummary]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class PresentationSearchResponse(BaseModel):
    """Paginated list response for presentation search results."""

    presentations: list[PresentationSummary]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class SlideGenerationBase(VisualStyleSelectionMixin):
    """Shared settings for slide generation requests."""

    title_hint: str | None = None
    theme: str | None = None
    marp_theme: str | None = None
    template_id: str | None = None
    settings: dict[str, Any] | None = None
    custom_css: str | None = None
    max_source_tokens: int | None = Field(default=None, ge=1)
    max_source_chars: int | None = Field(default=None, ge=1)
    enable_chunking: bool = False
    chunk_size_tokens: int | None = Field(default=None, ge=1)
    summary_tokens: int | None = Field(default=None, ge=1)
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class GenerateFromPromptRequest(SlideGenerationBase):
    """Request payload for generating slides from a prompt."""

    prompt: str


class GenerateFromChatRequest(SlideGenerationBase):
    """Request payload for generating slides from chat history."""

    conversation_id: str


class GenerateFromNotesRequest(SlideGenerationBase):
    """Request payload for generating slides from notes."""

    note_ids: list[str]


class GenerateFromMediaRequest(SlideGenerationBase):
    """Request payload for generating slides from media."""

    media_id: int = Field(..., ge=1)


class GenerateFromRagRequest(SlideGenerationBase):
    """Request payload for generating slides from a RAG query."""

    query: str
    top_k: int | None = Field(default=8, ge=1)


class ExportFormat(str, Enum):
    """Supported presentation export formats."""

    REVEAL = "revealjs"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"


class PresentationRenderFormat(str, Enum):
    """Supported presentation video render formats."""

    MP4 = "mp4"
    WEBM = "webm"


class PresentationRenderRequest(BaseModel):
    """Request payload for starting a presentation render job."""

    format: PresentationRenderFormat


class PresentationRenderJobResponse(BaseModel):
    """Render job creation response."""

    job_id: int
    status: str
    job_type: str
    presentation_id: str
    presentation_version: int
    format: PresentationRenderFormat


class PresentationRenderJobStatusResponse(BaseModel):
    """Render job status response."""

    job_id: int
    status: str
    job_type: str
    presentation_id: str | None = None
    presentation_version: int | None = None
    format: PresentationRenderFormat | None = None
    output_id: int | None = None
    download_url: str | None = None
    error: str | None = None


class PresentationRenderArtifactInfo(BaseModel):
    """Output artifact summary for presentation renders."""

    output_id: int
    format: PresentationRenderFormat
    title: str | None = None
    download_url: str
    presentation_version: int | None = None
    created_at: datetime | None = None


class PresentationRenderArtifactListResponse(BaseModel):
    """List of render artifacts for a presentation."""

    presentation_id: str
    artifacts: list[PresentationRenderArtifactInfo]


class SlidesHealthResponse(BaseModel):
    """Health status response for the slides service."""

    service: str
    status: str
    detail: str | None = None
