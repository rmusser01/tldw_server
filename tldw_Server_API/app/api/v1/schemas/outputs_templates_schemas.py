from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta, default_offset_pagination_aliases

TemplateType = Literal[
    "newsletter_markdown",
    "briefing_markdown",
    "mece_markdown",
    "newsletter_html",
    "tts_audio",
]

TemplateFormat = Literal["md", "html", "mp3"]


class OutputTemplateCreate(BaseModel):
    """Create an output template used for rendering collections (items or runs)."""

    name: str = Field(..., min_length=1, max_length=200)
    type: TemplateType
    format: TemplateFormat
    body: str = Field(..., description="Template body (e.g., Jinja2/Markdown/HTML)")
    description: str | None = Field(None, max_length=500)
    is_default: bool = False
    metadata: dict[str, Any] | None = Field(default=None, description="Arbitrary template metadata. For tts, supports keys tts_default_model, tts_default_voice, tts_default_speed.")

    @model_validator(mode="after")
    def validate_format_matches_type(self) -> "OutputTemplateCreate":
        t = self.type
        v = self.format
        if t in ("newsletter_markdown", "briefing_markdown", "mece_markdown") and v != "md":
            raise ValueError("Markdown-type templates must use format 'md'.")
        if t == "newsletter_html" and v != "html":
            raise ValueError("newsletter_html templates must use format 'html'.")
        if t == "tts_audio" and v != "mp3":
            raise ValueError("tts_audio templates must use format 'mp3'.")
        return self


class OutputTemplateUpdate(BaseModel):
    """Partial update for an existing template."""

    name: str | None = Field(None, min_length=1, max_length=200)
    type: TemplateType | None = None
    format: TemplateFormat | None = None
    body: str | None = None
    description: str | None = Field(None, max_length=500)
    is_default: bool | None = None
    metadata: dict[str, Any] | None = None


class OutputTemplate(BaseModel):
    """Template model used in responses."""

    id: int
    user_id: str | None = None
    name: str
    type: TemplateType
    format: TemplateFormat
    body: str
    description: str | None = None
    is_default: bool = False
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] | None = None


class OutputTemplateList(BaseModel):
    items: list[OutputTemplate]
    total: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> "OutputTemplateList":
        return default_offset_pagination_aliases(self)


class TemplatePreviewRequest(BaseModel):
    """Preview rendering without persisting an output artifact.

    Advanced users can pass inline `data` to dry-run rendering with a custom
    context. When `data` is provided, `item_ids`/`run_id` are optional.
    """

    template_id: int
    item_ids: list[int] | None = Field(default=None, description="Items to render.")
    run_id: int | None = Field(default=None, description="Use items from this run.")
    limit: int = Field(default=50, ge=1, le=1000)
    data: dict[str, object] | None = Field(
        default=None,
        description="Inline context for preview. Example: { 'items': [...], 'date': '...', 'job': {...} }",
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "TemplatePreviewRequest":
        # Allow inline data to satisfy preview requirements
        if not self.item_ids and not self.run_id and not self.data:
            raise ValueError("Provide item_ids, run_id, or inline data for preview.")
        return self


class TemplatePreviewResponse(BaseModel):
    rendered: str
    format: Literal["md", "html"]
