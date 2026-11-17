"""HTTP client for audiocpp_server.

Every request goes through ``core.http_client.afetch``: one attempt, no redirects, and the
egress policy checked on the address the connection uses. A loopback ``base_url`` (the
default, and the managed sidecar) is admitted by a ``ConfiguredEndpointScope`` for its exact
origin. A remote one (``allow_remote_base_url``) gets the default egress policy, so a private
address is refused unless the operator's egress settings allow it, or unless the operator sets
``allow_private_network``: then the remote origin gets the same exact-origin scope, which admits
private and LAN addresses there but never cloud metadata or link-local ones.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch, create_async_client
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

from ..tts_exceptions import (
    TTSConfigurationError,
    TTSGenerationError,
    TTSNetworkError,
    TTSProviderError,
    TTSTimeoutError,
    TTSValidationError,
)
from .audio_cpp_config import PROVIDER_KEY, _is_loopback_host, validate_base_url


def is_healthy_response(payload: Any) -> bool:
    """Require an explicit positive health status from the upstream server."""
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or payload.get("state") or "").strip().lower()
    return status in {"ok", "ready", "healthy"}


@dataclass(frozen=True)
class AudioCppSpeechResult:
    """Decoded audio.cpp speech response."""

    audio_bytes: bytes
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AudioCppClient:
    """Small async client for audiocpp_server routes used by the TTS adapter."""

    def __init__(
        self,
        *,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
        allow_remote_base_url: bool = False,
        allow_private_network: bool = False,
    ) -> None:
        """Use an injected client or own a proxy-independent HTTP client."""
        self.base_url = validate_base_url(
            base_url,
            allow_remote_base_url=allow_remote_base_url,
        )
        self.timeout = timeout
        # allow_private_network is operator provider config only, never request input.
        parsed = urlparse(self.base_url)
        self._endpoint_scope: ConfiguredEndpointScope | None = None
        if allow_private_network or _is_loopback_host(parsed.hostname):
            # The scoped egress path refuses userinfo. An unscoped remote base_url keeps it
            # (sent as Basic auth, e.g. to a reverse proxy).
            if parsed.username is not None or parsed.password is not None:
                raise TTSValidationError(
                    "audio.cpp credentials in base_url are not supported for loopback or private-network endpoints",
                    provider=PROVIDER_KEY,
                    error_code="invalid_base_url",
                )
            self._endpoint_scope = ConfiguredEndpointScope.from_url(self.base_url)
        self._http_client = http_client or create_async_client(timeout=timeout, trust_env=False)
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
            response: httpx.Response = await afetch(
                method=method,
                url=self._url(path),
                client=self._http_client,
                timeout=self.timeout,
                allow_redirects=False,
                retry=RetryPolicy(attempts=1),
                configured_endpoint=self._endpoint_scope,
                **kwargs,
            )
        except EgressPolicyError as exc:
            raise TTSConfigurationError(
                "audio.cpp base_url refused by the egress policy",
                provider=PROVIDER_KEY,
                error_code="EGRESS_DENIED",
                details={"reason_code": exc.reason_code},
            ) from exc
        except NetworkError as exc:
            if exc.classification == "timeout":
                raise TTSTimeoutError(
                    "Timeout waiting for audio.cpp server",
                    provider=PROVIDER_KEY,
                    error_code="TIMEOUT",
                ) from exc
            raise TTSNetworkError(
                "Network error communicating with audio.cpp server",
                provider=PROVIDER_KEY,
                error_code="NETWORK_ERROR",
                details={"error_type": str(exc)},
            ) from exc
        return response

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Expose status only; arbitrary upstream bodies can contain private data."""
        if 200 <= response.status_code < 300:
            return
        raise TTSProviderError(
            f"audio.cpp server returned HTTP {response.status_code}",
            provider=PROVIDER_KEY,
            error_code=f"HTTP_{response.status_code}",
            details={
                "status_code": response.status_code,
                "response_text": "Upstream error response omitted",
            },
        )

    async def health(self) -> dict[str, Any]:
        """Read health without treating empty or unrelated responses as ready."""
        response = await self._request("GET", "/health")
        self._raise_for_status(response)
        try:
            payload = response.json()
        except ValueError:
            payload = {"status": response.text.strip()}
        if not is_healthy_response(payload):
            raise TTSProviderError(
                "audio.cpp /health did not report a ready status",
                provider=PROVIDER_KEY,
                error_code="INVALID_HEALTH_RESPONSE",
            )
        return payload

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
        self._raise_for_status(response)
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
