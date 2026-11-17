"""Core-owned research run models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResearchRunResponse(BaseModel):
    """Current state returned for research session operations."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    phase: str
    control_state: str = "running"
    progress_percent: float | None = None
    progress_message: str | None = None
    active_job_id: str | None = None
    latest_checkpoint_id: str | None = None
    completed_at: str | None = None
    chat_id: str | None = None


class ResearchCheckpointSummary(BaseModel):
    """Current checkpoint summary included in research stream snapshots."""

    checkpoint_id: str
    checkpoint_type: str
    status: str
    proposed_payload: dict[str, Any]
    resolution: str | None = None


class ResearchArtifactManifestEntry(BaseModel):
    """Latest-version artifact metadata for research stream snapshots."""

    artifact_name: str
    artifact_version: int
    content_type: str
    phase: str
    job_id: str | None = None


class ResearchRunSnapshotResponse(BaseModel):
    """Reconnect-safe snapshot for live research progress streams."""

    run: ResearchRunResponse
    latest_event_id: int = 0
    checkpoint: ResearchCheckpointSummary | None = None
    artifacts: list[ResearchArtifactManifestEntry] = Field(default_factory=list)
