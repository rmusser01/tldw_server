"""Placeholder OmniVoice TTS adapter.

This module intentionally provides a minimal importable adapter surface so the
registry can resolve OmniVoice without requiring the sidecar implementation yet.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from ..tts_exceptions import TTSProviderNotConfiguredError
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse


class OmniVoiceAdapter(TTSAdapter):
    """Placeholder adapter for the future OmniVoice sidecar runtime."""

    PROVIDER_KEY = "omnivoice"
    SUPPORTED_FORMATS = {AudioFormat.PCM, AudioFormat.WAV, AudioFormat.MP3}
    SUPPORTED_LANGUAGES = {"en"}
    DEFAULT_SAMPLE_RATE = 24000
    MAX_TEXT_LENGTH = 8192

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.sample_rate = int((config or {}).get("sample_rate") or self.DEFAULT_SAMPLE_RATE)

    async def initialize(self) -> bool:
        self._status = ProviderStatus.NOT_CONFIGURED
        logger.info("OmniVoice adapter is a placeholder and is not configured yet")
        return False

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="OmniVoice",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=[],
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.MAX_TEXT_LENGTH,
            supports_streaming=False,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            supports_speech_rate=False,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=None,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.PCM,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        raise TTSProviderNotConfiguredError(
            "OmniVoice sidecar adapter is not implemented yet",
            provider=self.PROVIDER_KEY,
        )
