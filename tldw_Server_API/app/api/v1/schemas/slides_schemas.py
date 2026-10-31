"""Pydantic models for Slides/Presentation module."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    content_kind: str | None = "structured_slides"
    html_document: str | None = None
    html_sha256: str | None = None
    html_bytes: int | None = None
    html_slide_count: int | None = None
    generation_job_uuid: str | None = None
    generation_provenance: dict[str, Any] | None = None


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
    content_kind: str | None = None
    html_document: str | None = None
    html_sha256: str | None = None
    html_bytes: int | None = None
    html_slide_count: int | None = None
    generation_job_uuid: str | None = None
    generation_provenance: dict[str, Any] | None = None


class PresentationReorderRequest(BaseModel):
    """Request payload for reordering presentation slides."""

    order: list[int] = Field(..., min_items=1)


class PresentationResponseBase(BaseModel):
    """Fields shared by every authenticated detail representation."""

    id: str
    title: str
    description: str | None = None
    theme: str = "black"
    source_type: str | None = None
    source_ref: Any | None = None
    source_query: str | None = None
    created_at: datetime
    last_modified: datetime
    deleted: bool
    client_id: str
    version: int


class LegacyPresentationResponse(PresentationResponseBase):
    """Exact pre-negotiation structured Slides detail shape."""

    model_config = ConfigDict(extra="forbid")

    marp_theme: str | None = None
    template_id: str | None = None
    visual_style_id: str | None = None
    visual_style_scope: str | None = None
    visual_style_name: str | None = None
    visual_style_version: int | None = None
    visual_style_snapshot: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None
    studio_data: dict[str, Any] | None = None
    slides: list[Slide] = Field(default_factory=list)
    custom_css: str | None = None


class StructuredPresentationResponse(LegacyPresentationResponse):
    """Opted-in structured Slides detail."""

    content_kind: Literal["structured_slides"]


class StandaloneHtmlPresentationResponse(PresentationResponseBase):
    """Standalone source detail; never rendered or executed by tldw."""

    content_kind: Literal["standalone_html"]
    html_document: str
    html_sha256: str
    html_bytes: int
    html_slide_count: int
    generation_provenance: dict[str, Any]


DiscriminatedPresentationResponse = Annotated[
    StructuredPresentationResponse | StandaloneHtmlPresentationResponse,
    Field(discriminator="content_kind"),
]
PresentationResponse = LegacyPresentationResponse | DiscriminatedPresentationResponse


class StandaloneHtmlTombstone(BaseModel):
    """Source-free delete result for standalone HTML."""

    id: str
    content_kind: Literal["standalone_html"]
    deleted_at: datetime


PresentationDeleteResponse = PresentationResponse | StandaloneHtmlTombstone


class LegacyPresentationVersionSummary(BaseModel):
    """Exact pre-negotiation version-list entry."""

    model_config = ConfigDict(extra="forbid")

    presentation_id: str
    version: int
    created_at: datetime
    title: str | None = None
    deleted: bool | None = None


class AdditivePresentationVersionSummary(LegacyPresentationVersionSummary):
    """Opted-in source-free version-list entry."""

    content_kind: Literal["structured_slides", "standalone_html"]


PresentationVersionSummary = LegacyPresentationVersionSummary | AdditivePresentationVersionSummary


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


class PresentationProvenanceSummary(BaseModel):
    """Bounded generation provenance safe for list and search responses."""

    source_kind: str | None = None
    provider: str | None = None
    model: str | None = None


class PresentationSummaryBase(BaseModel):
    """Source-free fields common to presentation summaries."""

    id: str
    title: str
    description: str | None = None
    theme: str
    created_at: datetime
    last_modified: datetime
    deleted: bool
    version: int
    provenance: PresentationProvenanceSummary


class LegacyPresentationSummary(BaseModel):
    """Exact pre-negotiation structured list/search item."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str | None = None
    theme: str
    created_at: datetime
    last_modified: datetime
    deleted: bool
    version: int


class StructuredPresentationSummary(PresentationSummaryBase):
    content_kind: Literal["structured_slides"]
    slide_count: int


class StandaloneHtmlPresentationSummary(PresentationSummaryBase):
    content_kind: Literal["standalone_html"]
    html_slide_count: int
    html_bytes: int


DiscriminatedPresentationSummary = Annotated[
    StructuredPresentationSummary | StandaloneHtmlPresentationSummary,
    Field(discriminator="content_kind"),
]
PresentationSummary = LegacyPresentationSummary | DiscriminatedPresentationSummary


class StructuredPresentationMetadata(StructuredPresentationSummary):
    """Closed source-free metadata response for a structured presentation."""

    model_config = ConfigDict(extra="forbid")


class StandaloneHtmlPresentationMetadata(StandaloneHtmlPresentationSummary):
    """Closed source-free metadata response for a standalone HTML presentation."""

    model_config = ConfigDict(extra="forbid")


PresentationMetadataResponse = Annotated[
    StructuredPresentationMetadata | StandaloneHtmlPresentationMetadata,
    Field(discriminator="content_kind"),
]


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
    claims_verification_provider: str | None = None
    claims_verification_model: str | None = None
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


class _ClosedStandaloneModel(BaseModel):
    """Closed public schema used by the standalone HTML transport."""

    model_config = ConfigDict(extra="forbid")


class StandaloneHtmlPromptSource(_ClosedStandaloneModel):
    kind: Literal["prompt"]
    prompt: str


class StandaloneHtmlChatSource(_ClosedStandaloneModel):
    kind: Literal["chat"]
    conversation_id: str


class StandaloneHtmlMediaSource(_ClosedStandaloneModel):
    kind: Literal["media"]
    media_id: int = Field(ge=1, le=9_223_372_036_854_775_807)


class StandaloneHtmlNotesSource(_ClosedStandaloneModel):
    kind: Literal["notes"]
    note_ids: list[str] = Field(min_length=1, max_length=100)


class StandaloneHtmlRagSource(_ClosedStandaloneModel):
    kind: Literal["rag"]
    query: str = Field(max_length=20_000)
    top_k: int = Field(default=8, ge=1, le=100)


StandaloneHtmlGenerationSource = Annotated[
    StandaloneHtmlPromptSource
    | StandaloneHtmlChatSource
    | StandaloneHtmlMediaSource
    | StandaloneHtmlNotesSource
    | StandaloneHtmlRagSource,
    Field(discriminator="kind"),
]


class StandaloneHtmlOptions(_ClosedStandaloneModel):
    presentation_type: Literal[
        "pitch-deck",
        "tech-sharing",
        "product-launch",
        "weekly-report",
        "course-module",
        "keynote",
        "data-report",
        "training",
        "social-media",
        "case-study",
        "comparison",
        "roadmap",
    ]
    audience: str = Field(max_length=500)
    slide_count: int = Field(ge=1, le=30)
    visual_direction: Literal[
        "auto",
        "dark-technical",
        "minimal-light",
        "editorial",
        "corporate",
        "soft-pastel",
        "bold-creative",
        "neo-brutalist",
    ]
    delivery_style: Literal["speaker-led", "self-guided"]


class StandaloneHtmlGenerationRequest(_ClosedStandaloneModel):
    generation_mode: Literal["standalone_html"]
    generation_config_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source: StandaloneHtmlGenerationSource
    html_options: StandaloneHtmlOptions


class StandaloneHtmlGenerationPendingResponse(_ClosedStandaloneModel):
    generation_id: str
    status: Literal["queued", "running"]
    status_url: str
    presentation_id: None
    progress_text: str | None = None


class StandaloneHtmlGenerationCompletedResponse(_ClosedStandaloneModel):
    generation_id: str
    status: Literal["completed"]
    status_url: str
    presentation_id: str
    content_kind: Literal["standalone_html"]


class StandaloneHtmlGenerationFailedResponse(_ClosedStandaloneModel):
    generation_id: str
    status: Literal["failed"]
    status_url: str
    presentation_id: None
    error_code: str
    error_message: str


class StandaloneHtmlGenerationCancelledResponse(_ClosedStandaloneModel):
    generation_id: str
    status: Literal["cancelled"]
    status_url: str
    presentation_id: None
    error_code: Literal["generation_cancelled"]


StandaloneHtmlGenerationResponse = Annotated[
    StandaloneHtmlGenerationPendingResponse
    | StandaloneHtmlGenerationCompletedResponse
    | StandaloneHtmlGenerationFailedResponse
    | StandaloneHtmlGenerationCancelledResponse,
    Field(discriminator="status"),
]


class StandaloneHtmlContentLimits(_ClosedStandaloneModel):
    max_document_bytes: int
    max_source_write_bytes: int
    max_draft_attachment_bytes: int
    max_slides: int
    max_nesting_depth: int


class StructuredSlidesContentCapability(_ClosedStandaloneModel):
    read: Literal[True]
    edit: Literal[True]


StandaloneHtmlContentCapabilityReason = Literal["validator_unavailable"]
StandaloneHtmlGenerationCapabilityReason = Literal[
    "feature_disabled",
    "egress_disabled",
    "default_model_not_configured",
    "default_model_not_allowed",
    "default_endpoint_not_allowed",
    "prompt_asset_unavailable",
    "digest_key_unavailable",
    "generation_worker_unavailable",
    "generation_reconciler_overloaded",
    "validator_unavailable",
]


class StandaloneHtmlContentCapability(_ClosedStandaloneModel):
    read: Literal[True]
    edit: bool
    export_attachment: bool
    draft_attachment: Literal[True]
    reason: StandaloneHtmlContentCapabilityReason | None
    limits: StandaloneHtmlContentLimits


class SlidesContentKindsCapability(_ClosedStandaloneModel):
    structured_slides: StructuredSlidesContentCapability
    standalone_html: StandaloneHtmlContentCapability


class StructuredSlidesGenerationCapability(_ClosedStandaloneModel):
    enabled: Literal[True]
    transport: Literal["existing_source_endpoints"]


class StandaloneHtmlInputLimitsResponse(_ClosedStandaloneModel):
    max_request_bytes: int
    max_source_chars: int
    max_source_tokens: int
    max_audience_chars: int
    max_source_identifier_bytes: int
    max_note_ids: int
    max_rag_query_chars: int
    max_rag_top_k: int


class StandaloneHtmlOutputLimitsResponse(_ClosedStandaloneModel):
    max_provider_response_bytes: int
    max_document_bytes: int


class StandaloneHtmlGenerationCapability(_ClosedStandaloneModel):
    enabled: bool
    reason: StandaloneHtmlGenerationCapabilityReason | None
    transport: Literal["slides_generation_job"]
    source_kinds: tuple[Literal["prompt", "chat", "media", "notes", "rag"], ...]
    provider: str | None
    model: str | None
    adapter_id: str | None
    endpoint_identity: str | None
    generation_config_revision: str | None
    input_limits: StandaloneHtmlInputLimitsResponse
    output_limits: StandaloneHtmlOutputLimitsResponse


class SlidesGenerationModesCapability(_ClosedStandaloneModel):
    structured_slides: StructuredSlidesGenerationCapability
    standalone_html: StandaloneHtmlGenerationCapability


class SlidesCapabilitiesResponse(_ClosedStandaloneModel):
    schema_version: Literal[1]
    content_kind_request_header: Literal["X-Slides-Accept-Content-Kinds"]
    content_kinds: SlidesContentKindsCapability
    generation_modes: SlidesGenerationModesCapability


class ExportFormat(str, Enum):
    """Supported presentation export formats."""

    REVEAL = "revealjs"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"
    HTML = "html"


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
