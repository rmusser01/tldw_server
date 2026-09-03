"""HTTP client for audiocpp_server."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..tts_exceptions import (
    TTSGenerationError,
    TTSNetworkError,
    TTSProviderError,
    TTSTimeoutError,
)
from .audio_cpp_config import PROVIDER_KEY, validate_base_url

_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\r\n\t]+")
_POSIX_PATH_RE = re.compile(r"(?<!\w)/(?:[^\s/:;,)\]}]+/)+[^\s:;,)\]}]+")
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|authorization|bearer)\s*[:=]\s*[^,\s]+"
)


@dataclass(frozen=True)
class AudioCppSpeechResult:
    """Decoded audio.cpp speech response."""

    audio_bytes: bytes
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _sanitize_error_text(text: str | None, *, redaction_terms: list[str] | None = None) -> str:
    sanitized = str(text or "").strip()
    for term in redaction_terms or []:
        if term:
            sanitized = sanitized.replace(str(term), "[redacted-input]")
    sanitized = _WINDOWS_PATH_RE.sub("[redacted-path]", sanitized)
    sanitized = _POSIX_PATH_RE.sub("[redacted-path]", sanitized)
    sanitized = _SECRET_RE.sub(lambda match: f"{match.group(1)}=[redacted-secret]", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > 300:
        return f"{sanitized[:300]}..."
    return sanitized


class AudioCppClient:
    """Small async client for audiocpp_server routes used by the TTS adapter."""

    def __init__(
        self,
        *,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
        allow_remote_base_url: bool = False,
    ) -> None:
        self.base_url = validate_base_url(
            base_url,
            allow_remote_base_url=allow_remote_base_url,
        )
        self.timeout = timeout
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._owns_client = http_client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._http_client.aclose()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _content_type(self, response: httpx.Response) -> str:
        return str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http_client.request(
                method,
                self._url(path),
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise TTSTimeoutError(
                "Timeout waiting for audio.cpp server",
                provider=PROVIDER_KEY,
                error_code="TIMEOUT",
            ) from exc
        except httpx.RequestError as exc:
            raise TTSNetworkError(
                "Network error communicating with audio.cpp server",
                provider=PROVIDER_KEY,
                error_code="NETWORK_ERROR",
                details={"error_type": type(exc).__name__},
            ) from exc
        return response

    def _raise_for_status(self, response: httpx.Response, *, payload: dict[str, Any] | None = None) -> None:
        if response.status_code < 400:
            return
        request_text = ""
        if isinstance(payload, dict):
            request_text = str(payload.get("input") or "")
        raise TTSProviderError(
            f"audio.cpp server returned HTTP {response.status_code}",
            provider=PROVIDER_KEY,
            error_code=f"HTTP_{response.status_code}",
            details={
                "status_code": response.status_code,
                "response_text": _sanitize_error_text(
                    response.text,
                    redaction_terms=[request_text],
                ),
            },
        )

    async def health(self) -> dict[str, Any]:
        response = await self._request("GET", "/health")
        self._raise_for_status(response)
        if not response.content:
            return {"status": "ok"}
        try:
            return dict(response.json())
        except ValueError:
            return {"status": response.text.strip() or "ok"}

    async def list_models(self) -> list[str]:
        response = await self._request("GET", "/v1/models")
        self._raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise TTSProviderError(
                "audio.cpp /v1/models returned invalid JSON",
                provider=PROVIDER_KEY,
                error_code="INVALID_MODELS_RESPONSE",
            ) from exc

        models = payload.get("data") if isinstance(payload, dict) else None
        if models is None and isinstance(payload, dict):
            models = payload.get("models")
        if not isinstance(models, list):
            return []

        model_ids: list[str] = []
        for model in models:
            if isinstance(model, dict):
                model_id = str(model.get("id") or "").strip()
            else:
                model_id = str(model or "").strip()
            if model_id:
                model_ids.append(model_id)
        return model_ids

    async def speech(self, payload: dict[str, Any]) -> AudioCppSpeechResult:
        response = await self._request("POST", "/v1/audio/speech", json=payload)
        self._raise_for_status(response, payload=payload)
        content_type = self._content_type(response) or "application/octet-stream"
        if content_type != "application/json":
            return AudioCppSpeechResult(
                audio_bytes=response.content,
                content_type=content_type,
                metadata={"upstream_response_format": content_type},
            )
        return self._decode_json_speech_response(response)

    def _decode_json_speech_response(self, response: httpx.Response) -> AudioCppSpeechResult:
        try:
            payload = response.json()
        except ValueError as exc:
            raise TTSGenerationError(
                "audio.cpp speech JSON response was invalid",
                provider=PROVIDER_KEY,
                error_code="INVALID_SPEECH_RESPONSE",
            ) from exc
        if not isinstance(payload, dict):
            raise TTSGenerationError(
                "audio.cpp speech JSON response must be an object",
                provider=PROVIDER_KEY,
                error_code="INVALID_SPEECH_RESPONSE",
            )

        encoded_audio = None
        for key in ("audio", "audio_base64", "audio_data", "data"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                encoded_audio = value
                break
        if not encoded_audio:
            raise TTSGenerationError(
                "audio.cpp speech JSON response did not include audio",
                provider=PROVIDER_KEY,
                error_code="MISSING_AUDIO",
            )
        try:
            audio_bytes = base64.b64decode(encoded_audio, validate=True)
        except ValueError as exc:
            raise TTSGenerationError(
                "audio.cpp speech JSON audio was not valid base64",
                provider=PROVIDER_KEY,
                error_code="INVALID_AUDIO_BASE64",
            ) from exc
        return AudioCppSpeechResult(
            audio_bytes=audio_bytes,
            content_type="application/json",
            metadata={
                "upstream_response_format": "application/json",
                "json_format": payload.get("format") or payload.get("response_format"),
            },
        )
