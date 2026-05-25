from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ResearchWorkspaceCapabilityId = Literal[
    "source_browse",
    "chat",
    "artifact_text_generation",
    "slides_generation",
    "audio_summary",
    "export_download",
    "sync_share",
]
ResearchWorkspaceCapabilityStatus = Literal["ready", "degraded", "unavailable", "unknown"]
ResearchWorkspaceCapabilityMode = Literal["allow", "warn", "block"]
ResearchWorkspaceOverallStatus = Literal["ready", "degraded", "unavailable", "unknown"]


class ResearchWorkspaceCapability(BaseModel):
    """User-safe readiness for one Research Workspace action boundary."""

    status: ResearchWorkspaceCapabilityStatus
    mode: ResearchWorkspaceCapabilityMode
    dependencies: list[str] = Field(default_factory=list)
    reason_code: str | None = None


class ResearchWorkspaceCapabilitiesResponse(BaseModel):
    """Capability contract consumed by the Research Workspace frontend."""

    status: ResearchWorkspaceOverallStatus
    ttl_seconds: int = 30
    capabilities: dict[str, ResearchWorkspaceCapability]
    timestamp: datetime
