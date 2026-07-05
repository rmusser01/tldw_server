"""Schemas for Research Workspace media output jobs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceArtifactResponse

ResearchWorkspaceOutputArtifactType = Literal["video_overview", "infographic"]
ResearchWorkspaceOutputStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]


class ResearchWorkspaceOutputSettings(BaseModel):
    """Optional provider, style, voice, and image settings for output generation."""

    provider: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=256)
    title_hint: str | None = Field(default=None, max_length=256)
    slides_visual_style_id: str | None = Field(default=None, max_length=128)
    tts_provider: str | None = Field(default=None, max_length=64)
    tts_model: str | None = Field(default=None, max_length=128)
    tts_voice: str | None = Field(default=None, max_length=128)
    image_backend: str | None = Field(default=None, max_length=128)
    image_width: int | None = Field(default=None, ge=256, le=2048)
    image_height: int | None = Field(default=None, ge=256, le=2048)


class ResearchWorkspaceOutputSubmitRequest(BaseModel):
    """Request body for submitting a Research Workspace media output job."""

    artifact_type: ResearchWorkspaceOutputArtifactType
    source_ids: list[str] = Field(..., min_length=1, max_length=50)
    settings: ResearchWorkspaceOutputSettings = Field(default_factory=ResearchWorkspaceOutputSettings)

    @field_validator("source_ids")
    @classmethod
    def _source_ids_must_be_non_empty_strings(cls, value: list[str]) -> list[str]:
        """Normalize source ids and reject empty source selections."""
        normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not normalized:
            raise ValueError("source_ids must include at least one source id")
        return list(dict.fromkeys(normalized))


class ResearchWorkspaceOutputSubmitResponse(BaseModel):
    """Response returned after a media output job is queued."""

    job_id: int
    status: ResearchWorkspaceOutputStatus
    workspace_id: str
    artifact_id: str
    artifact_type: ResearchWorkspaceOutputArtifactType


class ResearchWorkspaceOutputStatusResponse(BaseModel):
    """Current job state plus the linked workspace artifact when available."""

    job_id: int
    status: ResearchWorkspaceOutputStatus
    progress_percent: float | None = None
    progress_message: str | None = None
    workspace_id: str
    artifact_id: str
    artifact_type: ResearchWorkspaceOutputArtifactType
    artifact: WorkspaceArtifactResponse | None = None
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
