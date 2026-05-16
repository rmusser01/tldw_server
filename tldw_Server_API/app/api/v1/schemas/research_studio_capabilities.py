from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ResearchStudioCapabilityId = Literal[
    "source_browse",
    "chat",
    "artifact_text_generation",
    "slides_generation",
    "audio_summary",
    "export_download",
    "sync_share",
]
ResearchStudioCapabilityStatus = Literal["ready", "degraded", "unavailable", "unknown"]
ResearchStudioCapabilityMode = Literal["allow", "warn", "block"]
ResearchStudioOverallStatus = Literal["ready", "degraded", "unavailable", "unknown"]


class ResearchStudioCapability(BaseModel):
    """User-safe readiness for one Research Studio action boundary."""

    status: ResearchStudioCapabilityStatus
    mode: ResearchStudioCapabilityMode
    dependencies: list[str] = Field(default_factory=list)
    reason_code: str | None = None


class ResearchStudioCapabilitiesResponse(BaseModel):
    """Capability contract consumed by the Research Studio frontend."""

    status: ResearchStudioOverallStatus
    ttl_seconds: int = 30
    capabilities: dict[str, ResearchStudioCapability]
    timestamp: datetime
