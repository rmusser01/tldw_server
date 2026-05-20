"""Pydantic schemas for audio health and capability metadata."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SttAvailability = Literal["ready", "on_demand", "unavailable", "unknown"]
SttCapabilityValue = Literal["supported", "unsupported", "unknown"]
AudioMetadataSource = Literal["health", "static_catalog", "provider", "response_schema", "unknown"]


class SttCapabilityModelResponse(BaseModel):
    """Capability summary for one STT model."""

    id: str
    label: str
    description: Optional[str] = None
    category: Optional[str] = None
    provider: str
    availability: SttAvailability
    availability_source: AudioMetadataSource
    capabilities: dict[str, SttCapabilityValue] = Field(default_factory=dict)
    sources: dict[str, AudioMetadataSource] = Field(default_factory=dict)
    message: Optional[str] = None


class SttCapabilitiesResponse(BaseModel):
    """Read-only STT capability summary for model-selection UIs."""

    models: list[SttCapabilityModelResponse]
    timestamp: str

