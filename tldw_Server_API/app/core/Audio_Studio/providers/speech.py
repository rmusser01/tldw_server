"""Speech generation adapter backed by the existing TTS service."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest, AudioGenerationResult
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

_MIME_BY_FORMAT = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "pcm": "audio/L16",
    "ulaw": "audio/basic",
    "wav": "audio/wav",
    "webm": "audio/webm",
}


class SpeechTtsAdapter:
    """Generate speech using the existing OpenAI-compatible TTS path."""

    provider_id = "tts"
    supported_kinds = frozenset({"speech"})

    def __init__(self, *, tts_service_factory: Callable[[], Any] = get_tts_service_v2) -> None:
        self._tts_service_factory = tts_service_factory

    async def generate(
        self,
        request: AudioGenerationRequest,
        *,
        user_id: int | None = None,
        provider_hint: str | None = None,
        **_: Any,
    ) -> AudioGenerationResult:
        """Generate speech bytes for an Audio Studio request."""

        if request.kind != "speech":
            raise ValueError("unsupported_audio_generation_kind")
        text = str(request.text or "").strip()
        if not text:
            raise ValueError("audio_studio_generation_text_required")

        options = dict(request.provider_options or {})
        response_format = str(options.get("format") or options.get("response_format") or "mp3").strip().lower()
        model = str(options.get("model") or "tts-1")
        voice = str(options.get("voice") or "af_heart")
        speed = float(options.get("speed") or 1.0)
        speech_request = OpenAISpeechRequest(
            model=model,
            input=text,
            voice=voice,
            response_format=response_format,  # type: ignore[arg-type]
            speed=speed,
            stream=False,
            extra_params=options.get("extra_params") if isinstance(options.get("extra_params"), dict) else None,
        )
        service = self._tts_service_factory()
        if inspect.isawaitable(service):
            service = await service
        audio_iter = service.generate_speech(
            speech_request,
            provider=provider_hint or options.get("provider_hint"),
            fallback=True,
            user_id=user_id,
        )
        chunks = b""
        async for chunk in audio_iter:
            chunks += chunk
        metadata = getattr(speech_request, "_tts_metadata", None)
        metadata_payload = dict(metadata) if isinstance(metadata, dict) else {}
        metadata_payload.setdefault("format", response_format)
        metadata_payload.setdefault("voice", voice)
        metadata_payload.setdefault("model", model)
        return AudioGenerationResult(
            mime_type=_MIME_BY_FORMAT.get(response_format, "application/octet-stream"),
            content_bytes=chunks,
            provider=self.provider_id,
            metadata=metadata_payload,
        )
