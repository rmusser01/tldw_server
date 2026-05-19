from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class OutputCreateRequest(BaseModel):
    template_id: int
    item_ids: list[int] | None = Field(default=None, description="Items to render")
    run_id: int | None = Field(default=None, description="Run to select items from (future)")
    title: str | None = None
    data: dict[str, object] | None = Field(default=None, description="Inline context override (advanced)")
    workspace_tag: str | None = Field(
        default=None,
        description="Optional workspace tag (e.g., 'workspace:<slug-or-id>') to associate this output.",
    )
    generate_mece: bool = Field(default=False, description="Generate a MECE variant output")
    mece_template_id: int | None = Field(default=None, description="Override template id for MECE output")
    generate_tts: bool = Field(default=False, description="Generate a TTS audio variant output")
    tts_template_id: int | None = Field(default=None, description="Override template id for TTS output")
    ingest_to_media_db: bool = Field(default=False, description="Ingest outputs into Media DB")
    tts_model: str | None = Field(default=None, description="TTS model id, e.g., 'kokoro', 'tts-1'")
    tts_voice: str | None = Field(default=None, description="TTS voice id, e.g., 'af_heart'")
    tts_speed: float | None = Field(default=None, ge=0.25, le=4.0, description="TTS speed override")


class OutputArtifact(BaseModel):
    id: int
    title: str
    type: str
    format: Literal["md", "html", "mp3"]
    storage_path: str
    media_item_id: int | None = None
    created_at: datetime
    workspace_tag: str | None = None


class OutputListResponse(BaseModel):
    items: list[OutputArtifact]
    total: int
    page: int
    size: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class OutputUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    retention_until: str | None = Field(default=None, description="ISO timestamp when this output can be purged")
    format: Literal["md", "html"] | None = Field(default=None, description="Change text format and re-encode (md/html only)")
