"""Shared interfaces for Fish S2 backend implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


FishS2SynthesisResult = bytes | AsyncIterator[bytes]


@runtime_checkable
class FishS2Backend(Protocol):
    """Transport/backend contract for the Fish S2 provider."""

    async def health_check(self) -> bool:
        """Return whether the backend appears reachable/usable."""

    async def synthesize(
        self,
        *,
        text: str,
        response_format: str,
        streaming: bool,
        reference_id: str | None,
        extra_params: dict[str, Any] | None,
    ) -> FishS2SynthesisResult:
        """Generate speech or return a byte stream for a request."""

    async def add_reference(
        self,
        *,
        reference_id: str,
        audio_b64: str,
        reference_text: str,
    ) -> dict[str, Any]:
        """Create or update a managed Fish reference."""

    async def delete_reference(self, *, reference_id: str) -> bool:
        """Delete a managed Fish reference."""
