from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ResearchWorkspaceArtifactType = Literal["audio_overview", "data_table", "mindmap"]


class ResearchWorkspaceArtifactGenerateRequest(BaseModel):
    artifact_type: ResearchWorkspaceArtifactType
    media_ids: list[int] = Field(..., min_length=1, max_length=50)
    model: str = Field(..., min_length=1)
    api_provider: str | None = None
    claims_verification_provider: str | None = Field(None, description="Optional Claims verification LLM provider override")
    claims_verification_model: str | None = Field(None, description="Optional Claims verification LLM model override")
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    max_tokens: int = Field(1200, ge=1, le=12000)

    @model_validator(mode="after")
    def clean_media_ids(self) -> "ResearchWorkspaceArtifactGenerateRequest":
        self.media_ids = list(dict.fromkeys(int(media_id) for media_id in self.media_ids if int(media_id) > 0))
        if not self.media_ids:
            raise ValueError("At least one positive media_id is required")
        return self


class ResearchWorkspaceArtifactGenerateResponse(BaseModel):
    artifact_type: ResearchWorkspaceArtifactType
    content: str
    data: dict[str, Any] = Field(default_factory=dict)
    claim_verification: dict[str, Any] | None = None
