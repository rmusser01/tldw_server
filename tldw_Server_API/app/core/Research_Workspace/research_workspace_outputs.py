"""Core-owned Research Workspace media output models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tldw_Server_API.app.core.Workspaces.workspace_schemas import WorkspaceArtifactResponse

ResearchWorkspaceOutputArtifactType = Literal["video_overview", "infographic", "summary", "compare_sources", "report"]
ResearchWorkspaceOutputStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]


class ResearchWorkspaceSectionSelection(BaseModel):
    """Reviewed outline identity, resolved again under the requesting owner."""

    source_id: str = Field(min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=256)
    navigation_version: str = Field(min_length=1, max_length=128)


class ResearchWorkspaceOutputSettings(BaseModel):
    """Optional provider, style, voice, and image settings for output generation."""

    sections: list[ResearchWorkspaceSectionSelection] = Field(default_factory=list, max_length=500)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    summary_instruction: str = Field(default="Summarize the major supported claims and disagreements.", max_length=8000)
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

    @model_validator(mode="before")
    @classmethod
    def _provider_output_budget(cls, value: Any) -> Any:
        """Honor explicit budgets; DeepSeek reasoning shares the output allowance."""
        if (
            isinstance(value, dict)
            and "max_tokens" not in value
            and str(value.get("provider") or "").strip().lower() == "deepseek"
        ):
            return {**value, "max_tokens": 8192}
        return value


class ResearchWorkspaceOutputSubmitRequest(BaseModel):
    """Request body for submitting a Research Workspace media output job."""

    artifact_type: ResearchWorkspaceOutputArtifactType
    source_ids: list[str] = Field(..., min_length=1, max_length=50)
    settings: ResearchWorkspaceOutputSettings = Field(default_factory=ResearchWorkspaceOutputSettings)
    resume_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_sections(self):
        identities = [(section.source_id, section.node_id) for section in self.settings.sections]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate selected sections")
        if any(section.source_id not in self.source_ids for section in self.settings.sections):
            raise ValueError("Sections must belong to selected sources")
        if self.artifact_type == "compare_sources" and (len(identities) if identities else len(self.source_ids)) < 2:
            raise ValueError("Select at least two sources or sections to compare")
        if self.resume_artifact_id and self.artifact_type not in {"summary", "compare_sources", "report"}:
            raise ValueError("Only section outputs support resume")
        return self

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
