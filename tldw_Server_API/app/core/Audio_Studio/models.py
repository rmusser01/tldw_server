"""Core models shared by Audio Studio provider adapters and Jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AudioGenerationRequest:
    """Provider-neutral request for generated Audio Studio assets."""

    workflow: str
    kind: str
    prompt: str | None
    text: str | None
    provider_options: dict[str, Any]
    target_resource_kind: str
    target_resource_id: str
    target_revision_id: str


@dataclass(frozen=True)
class AudioGenerationResult:
    """Provider-neutral generated audio result."""

    mime_type: str
    content_bytes: bytes
    provider: str
    metadata: dict[str, Any]


class AudioStudioProviderAdapter(Protocol):
    """Provider adapter contract for Audio Studio generation."""

    provider_id: str
    supported_kinds: frozenset[str]

    async def generate(self, request: AudioGenerationRequest, **kwargs: Any) -> AudioGenerationResult:
        """Generate audio for an Audio Studio request."""
