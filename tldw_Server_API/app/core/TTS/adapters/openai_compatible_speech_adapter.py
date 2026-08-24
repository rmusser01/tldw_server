"""Generic, server-configured OpenAI-compatible speech gateway adapter."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from copy import deepcopy
from typing import Any

from tldw_Server_API.app.core.exceptions import NetworkError as CoreNetworkError
from tldw_Server_API.app.core.http_client import RetryPolicy, astream_bytes

from ..gateway_config import build_gateway_url, copy_gateway_extra_params
from ..tts_exceptions import (
    TTSAudioQualityError,
    TTSAuthenticationError,
    TTSError,
    TTSModelNotFoundError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderNotConfiguredError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSTextTooLongError,
    TTSTimeoutError,
    TTSValidationError,
)
from .base import (
    AudioFormat,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    VoiceInfo,
)

_HEADER_NAME_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_SNIFF_LIMIT = 65536
_MIME_ALIASES: dict[AudioFormat, frozenset[str]] = {
    AudioFormat.MP3: frozenset({"audio/mpeg", "audio/mp3"}),
    AudioFormat.WAV: frozenset({"audio/wav", "audio/x-wav", "audio/wave"}),
    AudioFormat.FLAC: frozenset({"audio/flac", "audio/x-flac"}),
    AudioFormat.OGG: frozenset({"audio/ogg", "application/ogg"}),
    AudioFormat.OPUS: frozenset({"audio/ogg", "audio/opus", "application/ogg"}),
    AudioFormat.AAC: frozenset({"audio/aac", "audio/mp4"}),
    AudioFormat.WEBM: frozenset({"audio/webm", "video/webm"}),
    AudioFormat.PCM: frozenset({"audio/pcm", "audio/l16", "audio/raw"}),
    AudioFormat.ULAW: frozenset({"audio/basic", "audio/ulaw", "audio/x-mulaw"}),
}
_SIGNATURE_BYTES: dict[AudioFormat, int] = {
    AudioFormat.MP3: 3,
    AudioFormat.WAV: 12,
    AudioFormat.FLAC: 4,
    AudioFormat.OGG: 4,
    AudioFormat.AAC: 8,
    AudioFormat.WEBM: 4,
}


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    folded = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == folded:
            return str(value)
    return None


class OpenAICompatibleSpeechAdapter(TTSAdapter):
    """One OpenAI-compatible speech attempt using only server-owned routing data."""

    handles_text_chunking = True

    def __init__(self, config: dict[str, Any] | None = None):
        raw = deepcopy(config or {})
        super().__init__({})
        self._backend_id = str(raw.get("backend_id") or "").strip()
        self._api_key = raw.get("api_key") if isinstance(raw.get("api_key"), str) else None
        self._default_voice = (
            raw.get("default_voice") if isinstance(raw.get("default_voice"), str) else None
        )
        self._allowed_request_options = frozenset(raw.get("allowed_request_options") or ())
        self._conversion_needed = bool(raw.get("conversion_needed", False))
        self._timeout_seconds = float(raw.get("timeout_seconds", 30.0))

        try:
            self._speech_url = build_gateway_url(
                str(raw.get("base_url") or ""),
                str(raw.get("speech_path") or ""),
            )
            self._source_format = AudioFormat(str(raw.get("source_format") or "").lower())
        except (TypeError, ValueError) as exc:
            raise TTSValidationError(
                "Gateway adapter routing configuration is invalid",
                provider=self._backend_id or None,
            ) from exc

        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise TTSValidationError(
                "Gateway adapter capabilities are invalid",
                provider=self._backend_id or None,
            )
        self._supports_speed = bool(capabilities.get("supports_speed", False))
        self._supports_language = bool(capabilities.get("supports_language", False))
        self._supports_target_sample_rate = bool(
            capabilities.get("supports_target_sample_rate", False)
        )
        self._allow_octet_stream = bool(capabilities.get("allow_octet_stream", False))
        self._max_input_characters = int(capabilities.get("max_input_characters", 12000))
        self._max_response_bytes = int(capabilities.get("max_response_bytes", 26214400))
        pcm = capabilities.get("pcm")
        pcm = pcm if isinstance(pcm, Mapping) else {}
        self._sample_rate = int(pcm.get("sample_rate", 24000))
        self._channels = int(pcm.get("channels", 1))
        self._sample_width_bits = int(pcm.get("sample_width_bits", 16))
        if (
            self._max_input_characters <= 0
            or self._max_response_bytes <= 0
            or self._sample_rate <= 0
            or self._channels <= 0
            or self._sample_width_bits <= 0
            or self._sample_width_bits % 8
        ):
            raise TTSValidationError(
                "Gateway adapter capability bounds are invalid",
                provider=self._backend_id or None,
            )

        configured_formats = capabilities.get("formats", (self._source_format.value,))
        try:
            self._formats = frozenset(AudioFormat(str(value).lower()) for value in configured_formats)
        except (TypeError, ValueError) as exc:
            raise TTSValidationError(
                "Gateway adapter formats are invalid",
                provider=self._backend_id or None,
            ) from exc
        if self._source_format not in self._formats:
            raise TTSValidationError(
                "Gateway source format is not enabled",
                provider=self._backend_id or None,
            )

        raw_headers = raw.get("headers", ())
        header_items = raw_headers.items() if isinstance(raw_headers, Mapping) else raw_headers
        headers: dict[str, str] = {}
        seen: set[str] = set()
        try:
            for name, value in header_items:
                if not isinstance(name, str) or not isinstance(value, str):
                    raise ValueError
                folded = name.casefold()
                if (
                    not _HEADER_NAME_RE.fullmatch(name)
                    or folded == "authorization"
                    or folded in seen
                    or "\r" in value
                    or "\n" in value
                ):
                    raise ValueError
                seen.add(folded)
                headers[name] = value
        except (TypeError, ValueError) as exc:
            raise TTSValidationError(
                "Gateway server header configuration is invalid",
                provider=self._backend_id or None,
            ) from exc
        self._server_headers = headers

    @property
    def provider_key(self) -> str:
        return self._backend_id

    async def initialize(self) -> bool:
        """Validate local attempt configuration without probing the gateway."""
        return bool(self._backend_id and self._api_key and self._api_key.strip())

    async def get_capabilities(self) -> TTSCapabilities:
        """Return the configured model capabilities without remote discovery."""
        voices = []
        if self._default_voice:
            voices.append(VoiceInfo(id=self._default_voice, name=self._default_voice))
        return TTSCapabilities(
            provider_name=self._backend_id,
            supported_languages={"*"} if self._supports_language else set(),
            supported_voices=voices,
            supported_formats=set(self._formats),
            max_text_length=self._max_input_characters,
            supports_streaming=True,
            supports_speech_rate=self._supports_speed,
            sample_rate=self._sample_rate,
            default_format=self._source_format,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Return a validated async audio iterator for exactly one upstream POST."""
        payload, model, voice = self._build_payload(request)
        if not self._api_key or not self._api_key.strip():
            raise TTSProviderNotConfiguredError(
                "Gateway credential is not configured",
                provider=self._backend_id,
            )
        headers = dict(self._server_headers)
        headers["Authorization"] = f"Bearer {self._api_key}"
        headers["Content-Type"] = "application/json"
        metadata: dict[str, Any] = {
            "backend_id": self._backend_id,
            "model": model,
            "voice": voice,
            "source_format": self._source_format.value,
            "declared_content_type": None,
            "conversion_needed": self._conversion_needed,
        }

        async def audio_stream() -> AsyncIterator[bytes]:
            upstream: AsyncIterator[bytes] | None = None
            try:
                upstream = astream_bytes(
                    method="POST",
                    url=self._speech_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                    retry=RetryPolicy(attempts=1),
                    chunk_size=_SNIFF_LIMIT,
                    on_response=self._response_validator(metadata),
                )
                async for chunk in self._validated_chunks(upstream):
                    yield chunk
            except asyncio.CancelledError:
                raise
            except TTSError:
                raise
            except (CoreNetworkError, OSError, RuntimeError, TypeError, ValueError) as exc:
                self._raise_transport_error(exc)
            finally:
                if upstream is not None:
                    closer = getattr(upstream, "aclose", None)
                    if callable(closer):
                        with suppress(Exception):
                            result = closer()
                            if inspect.isawaitable(result):
                                await result

        return TTSResponse(
            audio_stream=audio_stream(),
            format=self._source_format,
            sample_rate=self._sample_rate,
            channels=self._channels,
            voice_used=voice,
            provider=self._backend_id,
            model=model,
            metadata=metadata,
        )

    def _build_payload(self, request: TTSRequest) -> tuple[dict[str, Any], str, str]:
        text = request.text
        if not isinstance(text, str) or not text:
            raise TTSValidationError("Gateway input text must be non-empty", provider=self._backend_id)
        if len(text) > self._max_input_characters:
            raise TTSTextTooLongError(
                "Gateway input exceeds the configured character limit",
                provider=self._backend_id,
                details={"max_input_characters": self._max_input_characters},
            )
        model = request.model
        if not isinstance(model, str) or not model.strip():
            raise TTSValidationError("Gateway model is required", provider=self._backend_id)
        voice = request.voice or self._default_voice
        if not isinstance(voice, str) or not voice.strip():
            raise TTSValidationError("Gateway voice is required", provider=self._backend_id)

        supplied_fields = getattr(request, "supplied_fields", frozenset()) or frozenset()
        speed_supplied = "speed" in supplied_fields
        lang_code_supplied = "lang_code" in supplied_fields
        language_supplied = "language" in supplied_fields
        supplied_values = getattr(request, "supplied_field_values", None) or {}
        lang_code = supplied_values.get("lang_code", getattr(request, "lang_code", None))
        language = supplied_values.get("language", getattr(request, "language", None))
        if lang_code_supplied and language_supplied and lang_code != language:
            raise TTSValidationError(
                "Gateway lang_code and language values conflict",
                provider=self._backend_id,
            )

        payload: dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": self._source_format.value,
        }
        if speed_supplied:
            if not self._supports_speed:
                raise TTSValidationError(
                    "Gateway does not support speed",
                    provider=self._backend_id,
                )
            payload["speed"] = request.speed
        if lang_code_supplied or language_supplied:
            unsupported_field = "lang_code" if lang_code_supplied else "language"
            if not self._supports_language:
                raise TTSValidationError(
                    f"Gateway does not support {unsupported_field}",
                    provider=self._backend_id,
                )
            effective_language = lang_code if lang_code_supplied else language
            if effective_language is not None:
                payload["language"] = effective_language
        target_sample_rate = request.target_sample_rate
        if target_sample_rate is not None:
            if not self._supports_target_sample_rate:
                raise TTSValidationError(
                    "Gateway does not support target_sample_rate",
                    provider=self._backend_id,
                )
            payload["target_sample_rate"] = target_sample_rate

        try:
            options = copy_gateway_extra_params(
                request.extra_params or {},
                self._allowed_request_options,
            )
        except ValueError as exc:
            raise TTSValidationError(
                "Gateway extra_params validation failed",
                provider=self._backend_id,
            ) from exc
        payload.update(options)
        return payload, model, voice

    def _response_validator(self, metadata: dict[str, Any]):
        def validate(status: int, headers: Mapping[str, str]) -> None:
            self._raise_for_status(int(status))
            raw_content_type = _header_value(headers, "content-type")
            if not raw_content_type or "," in raw_content_type:
                raise TTSAudioQualityError(
                    "Gateway response content type is missing or ambiguous",
                    provider=self._backend_id,
                )
            content_type = raw_content_type.split(";", 1)[0].strip().lower()
            expected = _MIME_ALIASES[self._source_format]
            if content_type == "application/octet-stream":
                accepted = self._allow_octet_stream
            else:
                accepted = content_type in expected
            if not accepted:
                raise TTSAudioQualityError(
                    "Gateway response content type does not match the source format",
                    provider=self._backend_id,
                )
            metadata["declared_content_type"] = content_type
            raw_length = _header_value(headers, "content-length")
            if raw_length is not None:
                try:
                    declared_length = int(raw_length)
                except (TypeError, ValueError) as exc:
                    raise TTSAudioQualityError(
                        "Gateway response size declaration is invalid",
                        provider=self._backend_id,
                    ) from exc
                if declared_length < 0 or declared_length > self._max_response_bytes:
                    raise TTSAudioQualityError(
                        "Gateway response size exceeds the configured limit",
                        provider=self._backend_id,
                        error_code="RESPONSE_SIZE_EXCEEDED",
                    )
                if self._source_format is AudioFormat.PCM and declared_length % self._frame_bytes:
                    raise TTSAudioQualityError(
                        "Gateway PCM response is not frame aligned",
                        provider=self._backend_id,
                    )

        return validate

    @property
    def _frame_bytes(self) -> int:
        return self._channels * (self._sample_width_bits // 8)

    async def _validated_chunks(self, upstream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        if self._source_format is AudioFormat.PCM:
            async for chunk in self._validated_pcm_chunks(upstream):
                yield chunk
            return
        if self._source_format is AudioFormat.OPUS:
            async for chunk in self._validated_opus_chunks(upstream):
                yield chunk
            return

        minimum = _SIGNATURE_BYTES.get(self._source_format)
        prefix = bytearray()
        total = 0
        validated = minimum is None
        async for chunk in upstream:
            if not chunk:
                continue
            if not isinstance(chunk, bytes):
                raise TTSAudioQualityError(
                    "Gateway returned an invalid audio chunk",
                    provider=self._backend_id,
                )
            total += len(chunk)
            if total > self._max_response_bytes:
                raise TTSAudioQualityError(
                    "Gateway response size exceeds the configured limit",
                    provider=self._backend_id,
                    error_code="RESPONSE_SIZE_EXCEEDED",
                )
            if validated:
                yield chunk
                continue
            needed = min(minimum - len(prefix), _SNIFF_LIMIT - len(prefix))
            prefix.extend(chunk[:needed])
            remainder = chunk[needed:]
            if len(prefix) < minimum:
                continue
            if not self._has_expected_signature(bytes(prefix)):
                raise TTSAudioQualityError(
                    "Gateway audio signature does not match the source format",
                    provider=self._backend_id,
                )
            validated = True
            yield bytes(prefix)
            if remainder:
                yield remainder
        if total == 0:
            raise TTSAudioQualityError("Gateway returned empty audio", provider=self._backend_id)
        if not validated:
            raise TTSAudioQualityError(
                "Gateway audio signature does not match the source format",
                provider=self._backend_id,
            )

    async def _validated_opus_chunks(
        self,
        upstream: AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        sniffed = bytearray()
        total = 0
        validated = False
        async for chunk in upstream:
            if not chunk:
                continue
            if not isinstance(chunk, bytes):
                raise TTSAudioQualityError(
                    "Gateway returned an invalid audio chunk",
                    provider=self._backend_id,
                )
            total += len(chunk)
            if total > self._max_response_bytes:
                raise TTSAudioQualityError(
                    "Gateway response size exceeds the configured limit",
                    provider=self._backend_id,
                    error_code="RESPONSE_SIZE_EXCEEDED",
                )
            if validated:
                yield chunk
                continue
            remaining = _SNIFF_LIMIT - len(sniffed)
            accepted = min(len(chunk), remaining)
            sniffed.extend(chunk[:accepted])
            remainder = chunk[accepted:]
            if len(sniffed) >= 4 and not sniffed.startswith(b"OggS"):
                raise TTSAudioQualityError(
                    "Gateway audio signature does not match the source format",
                    provider=self._backend_id,
                )
            if b"OpusHead" in sniffed:
                validated = True
                yield bytes(sniffed)
                if remainder:
                    yield remainder
            elif len(sniffed) == _SNIFF_LIMIT or remainder:
                raise TTSAudioQualityError(
                    "Gateway audio signature does not match the source format",
                    provider=self._backend_id,
                )
        if total == 0:
            raise TTSAudioQualityError("Gateway returned empty audio", provider=self._backend_id)
        if not validated:
            raise TTSAudioQualityError(
                "Gateway audio signature does not match the source format",
                provider=self._backend_id,
            )

    async def _validated_pcm_chunks(self, upstream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        tail = b""
        total = 0
        async for chunk in upstream:
            if not chunk:
                continue
            if not isinstance(chunk, bytes):
                raise TTSAudioQualityError(
                    "Gateway returned an invalid PCM chunk",
                    provider=self._backend_id,
                )
            total += len(chunk)
            if total > self._max_response_bytes:
                raise TTSAudioQualityError(
                    "Gateway response size exceeds the configured limit",
                    provider=self._backend_id,
                    error_code="RESPONSE_SIZE_EXCEEDED",
                )
            framed = tail + chunk
            complete = len(framed) - (len(framed) % self._frame_bytes)
            if complete:
                yield framed[:complete]
            tail = framed[complete:]
        if total == 0:
            raise TTSAudioQualityError("Gateway returned empty audio", provider=self._backend_id)
        if tail:
            raise TTSAudioQualityError(
                "Gateway PCM response ended with a partial frame",
                provider=self._backend_id,
            )

    def _has_expected_signature(self, data: bytes) -> bool:
        if self._source_format is AudioFormat.MP3:
            return data.startswith(b"ID3") or (
                len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
            )
        if self._source_format is AudioFormat.WAV:
            return data.startswith(b"RIFF") and data[8:12] == b"WAVE"
        if self._source_format is AudioFormat.FLAC:
            return data.startswith(b"fLaC")
        if self._source_format is AudioFormat.OGG:
            return data.startswith(b"OggS")
        if self._source_format is AudioFormat.AAC:
            return (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xF0 == 0xF0) or (
                len(data) >= 8 and data[4:8] == b"ftyp"
            )
        if self._source_format is AudioFormat.WEBM:
            return data.startswith(b"\x1aE\xdf\xa3")
        return True

    def _raise_for_status(self, status: int) -> None:
        if 200 <= status < 300:
            return
        details = {"status": status}
        if status in {401, 403}:
            raise TTSAuthenticationError(
                "Gateway authentication failed",
                provider=self._backend_id,
                error_code="AUTH_ERROR",
                details=details,
            )
        if status == 402:
            raise TTSQuotaExceededError(
                "Gateway quota is exhausted",
                provider=self._backend_id,
                error_code="QUOTA_EXCEEDED",
                details=details,
            )
        if status == 429:
            raise TTSRateLimitError(
                "Gateway rate limit exceeded",
                provider=self._backend_id,
                error_code="RATE_LIMIT",
                details=details,
            )
        if status in {408, 504}:
            raise TTSTimeoutError(
                "Gateway request timed out",
                provider=self._backend_id,
                error_code="TIMEOUT",
                details=details,
            )
        if status == 404:
            raise TTSModelNotFoundError(
                "Gateway model was not found",
                provider=self._backend_id,
                error_code="MODEL_NOT_FOUND",
                details=details,
            )
        if 400 <= status < 500:
            raise TTSValidationError(
                "Gateway rejected the speech request",
                provider=self._backend_id,
                error_code=str(status),
                details=details,
            )
        raise TTSProviderError(
            "Gateway upstream request failed",
            provider=self._backend_id,
            error_code=str(status),
            details=details,
        )

    def _raise_transport_error(self, exc: Exception) -> None:
        name = exc.__class__.__name__.casefold()
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
            raise TTSTimeoutError(
                "Gateway request timed out",
                provider=self._backend_id,
                error_code="TIMEOUT",
            ) from exc
        if isinstance(exc, (CoreNetworkError, OSError, ConnectionError)) or any(
            token in name for token in ("network", "connect", "socket", "dns", "ssl", "tls")
        ):
            raise TTSNetworkError(
                "Gateway network request failed",
                provider=self._backend_id,
                error_code="NETWORK_ERROR",
            ) from exc
        raise TTSProviderError(
            "Gateway request failed",
            provider=self._backend_id,
        ) from exc
