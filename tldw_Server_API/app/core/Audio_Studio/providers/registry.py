"""Provider registry for Audio Studio generation adapters."""

from __future__ import annotations

from typing import Iterable

from loguru import logger

from tldw_Server_API.app.core.Audio_Studio.models import AudioStudioProviderAdapter

from .ace_step import AceStepHttpAdapter
from .speech import SpeechTtsAdapter


class AudioStudioProviderRegistry:
    """Lookup and describe configured Audio Studio provider adapters."""

    def __init__(self, adapters: Iterable[AudioStudioProviderAdapter]) -> None:
        self._adapters = {adapter.provider_id: adapter for adapter in adapters}

    def list_providers(self) -> list[dict[str, object]]:
        """Return secret-free provider catalog entries."""

        return [
            {
                "provider_id": adapter.provider_id,
                "supported_kinds": sorted(adapter.supported_kinds),
            }
            for adapter in self._adapters.values()
        ]

    def get_adapter(self, provider_id: str, kind: str) -> AudioStudioProviderAdapter:
        """Return a configured adapter that supports the requested generation kind."""

        adapter = self._adapters.get(str(provider_id or "").strip())
        if adapter is None:
            raise KeyError("audio_studio_provider_not_found")
        if str(kind or "").strip() not in adapter.supported_kinds:
            raise ValueError("unsupported_audio_generation_kind")
        return adapter


def build_audio_studio_provider_registry() -> AudioStudioProviderRegistry:
    """Build the default provider registry from runtime configuration."""

    adapters: list[AudioStudioProviderAdapter] = [SpeechTtsAdapter()]
    try:
        if AceStepHttpAdapter.is_configured():
            adapters.append(AceStepHttpAdapter())
    except ValueError as exc:
        logger.warning("ACE-Step Audio Studio provider disabled by configuration: {}", exc)
    return AudioStudioProviderRegistry(adapters)
