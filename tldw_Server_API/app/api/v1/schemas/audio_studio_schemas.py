"""Pydantic schemas for Audio Studio APIs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class AudioStudioWorkflow(str, Enum):
    """Supported Audio Studio workflow types."""

    NARRATION = "narration"
    PODCAST = "podcast"
    BRIEFING = "briefing"
    MUSIC = "music"


class AudioStudioResourceKind(str, Enum):
    """Project resource kinds that can participate in revisions and jobs."""

    SECTION = "section"
    TRACK = "track"
    CLIP = "clip"
    ARTIFACT = "artifact"
    RENDER = "render"
    EXPORT = "export"


class AudioStudioTrackKind(str, Enum):
    """Timeline track categories."""

    SPEECH = "speech"
    MUSIC = "music"
    SFX = "sfx"
    AMBIENCE = "ambience"
    MIXED = "mixed"


class AudioStudioClipType(str, Enum):
    """Timeline clip categories."""

    SPEECH = "speech"
    MUSIC = "music"
    SFX = "sfx"
    AMBIENCE = "ambience"
    IMPORTED = "imported"
    RENDER = "render"


_FORBIDDEN_CLIENT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "client_secret",
    "external_url",
    "password",
    "secret",
    "token",
}


def _reject_secret_payload(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _FORBIDDEN_CLIENT_KEYS or normalized.endswith("_secret"):
                raise ValueError(f"{path} must not include secret or external_url fields")
            _reject_secret_payload(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secret_payload(nested, path=f"{path}[{index}]")


class _BaseAudioStudioModel(BaseModel):
    model_config = {"use_enum_values": False}


class _SecretFreePayloadMixin(_BaseAudioStudioModel):
    @field_validator("provider", "options", "settings", mode="after", check_fields=False)
    @classmethod
    def _validate_secret_free_payload(cls, value: Any) -> Any:
        _reject_secret_payload(value)
        return value

    @model_validator(mode="after")
    def _validate_no_forbidden_client_payload(self):
        _reject_secret_payload(self.model_dump(mode="python"))
        return self


class AudioStudioProjectCreate(_SecretFreePayloadMixin):
    """Create a server-backed Audio Studio project."""

    title: str = Field(..., min_length=1, max_length=200)
    workflow: AudioStudioWorkflow
    description: str | None = Field(None, max_length=2000)
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioStudioProjectUpdate(_SecretFreePayloadMixin):
    """Update project metadata/settings with optimistic concurrency."""

    base_revision_id: str = Field(..., min_length=1, max_length=120)
    title: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    status: str | None = Field(None, max_length=40)
    settings: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class AudioStudioProjectResponse(_BaseAudioStudioModel):
    """Project response returned by Audio Studio endpoints."""

    project_id: str
    title: str
    description: str | None = None
    workflow: AudioStudioWorkflow
    status: str
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_revision_id: str | None = None
    created_at: str
    updated_at: str
    archived_at: str | None = None


class AudioStudioProjectListResponse(_BaseAudioStudioModel):
    """List of visible Audio Studio projects."""

    projects: list[AudioStudioProjectResponse]
    limit: int
    offset: int


class AudioStudioSectionUpsert(_SecretFreePayloadMixin):
    """Create or update a workflow section."""

    base_revision_id: str = Field(..., min_length=1, max_length=120)
    title: str | None = Field(None, max_length=200)
    body_text: str | None = None
    speaker_id: str | None = Field(None, max_length=120)
    order_index: int = Field(0, ge=0)
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioStudioSectionResponse(_BaseAudioStudioModel):
    section_id: str
    workflow: AudioStudioWorkflow
    title: str | None
    body_text: str | None
    speaker_id: str | None
    order_index: int
    settings: dict[str, Any] = Field(default_factory=dict)
    current_revision_id: str | None = None
    archived_at: str | None = None


class AudioStudioTrackUpsert(_SecretFreePayloadMixin):
    """Create or update a timeline track."""

    base_revision_id: str = Field(..., min_length=1, max_length=120)
    name: str = Field(..., min_length=1, max_length=160)
    kind: AudioStudioTrackKind
    order_index: int = Field(0, ge=0)
    muted: bool = False
    solo: bool = False
    volume: float = Field(1.0, ge=0.0, le=4.0)
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioStudioTrackResponse(_BaseAudioStudioModel):
    track_id: str
    name: str
    kind: AudioStudioTrackKind
    order_index: int
    muted: bool
    solo: bool
    volume: float
    settings: dict[str, Any] = Field(default_factory=dict)
    current_revision_id: str | None = None
    archived_at: str | None = None


class AudioStudioClipUpsert(_SecretFreePayloadMixin):
    """Create or update a timeline clip."""

    base_revision_id: str = Field(..., min_length=1, max_length=120)
    section_id: str | None = Field(None, max_length=120)
    track_id: str = Field(..., min_length=1, max_length=120)
    title: str | None = Field(None, max_length=200)
    clip_type: AudioStudioClipType
    start_ms: int = Field(0, ge=0)
    duration_ms: int | None = Field(None, ge=0)
    volume: float = Field(1.0, ge=0.0, le=4.0)
    fade_in_ms: int = Field(0, ge=0)
    fade_out_ms: int = Field(0, ge=0)
    muted: bool = False
    artifact_id: str | None = Field(None, max_length=120)
    settings: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioStudioClipResponse(_BaseAudioStudioModel):
    clip_id: str
    section_id: str | None
    track_id: str
    title: str | None
    clip_type: AudioStudioClipType
    start_ms: int
    duration_ms: int | None
    volume: float
    fade_in_ms: int
    fade_out_ms: int
    muted: bool
    artifact_id: str | None
    settings: dict[str, Any] = Field(default_factory=dict)
    current_revision_id: str | None = None
    archived_at: str | None = None


class _AudioStudioJobCreate(_SecretFreePayloadMixin):
    idempotency_key: str = Field(..., min_length=16, max_length=200)
    target_resource_kind: AudioStudioResourceKind
    target_resource_id: str = Field(..., min_length=1, max_length=120)
    target_revision_id: str = Field(..., min_length=1, max_length=120)
    options: dict[str, Any] = Field(default_factory=dict)


class AudioStudioGenerationCreate(_AudioStudioJobCreate):
    """Schema-only contract for future generation job creation."""

    kind: str = Field(..., min_length=1, max_length=80)
    provider: str | dict[str, Any] = Field(..., min_length=1)


class AudioStudioRenderCreate(_AudioStudioJobCreate):
    """Schema-only contract for future render job creation."""

    render_type: str = Field(..., min_length=1, max_length=80)
    settings: dict[str, Any] = Field(default_factory=dict)


class AudioStudioExportCreate(_AudioStudioJobCreate):
    """Schema-only contract for future export job creation."""

    export_type: str = Field(..., min_length=1, max_length=80)
    source_render_id: str | None = Field(None, max_length=120)
    settings: dict[str, Any] = Field(default_factory=dict)


class AudioStudioMigrationPreview(_SecretFreePayloadMixin):
    """Schema-only contract for legacy Audiobook Studio migration preview."""

    legacy_project_id: str | None = Field(None, max_length=200)
    project_payload: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class AudioStudioMigrationCommit(_SecretFreePayloadMixin):
    """Schema-only contract for legacy Audiobook Studio migration commit."""

    preview_id: str | None = Field(None, max_length=200)
    base_revision_id: str | None = Field(None, max_length=120)
    project_payload: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_commit_target(self) -> AudioStudioMigrationCommit:
        if not self.preview_id and not self.project_payload:
            raise ValueError("migration commit requires preview_id or project_payload")
        return self
