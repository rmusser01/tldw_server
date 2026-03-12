"""Native Fish Audio HTTP backend for the Fish S2 provider."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.http_client import afetch, astream_bytes

from ..tts_exceptions import (
    TTSAuthenticationError,
    TTSNetworkError,
    TTSProviderError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSValidationError,
    auth_error,
    network_error,
    provider_error,
    rate_limit_error,
    timeout_error,
)
from .fish_s2_base import FishS2Backend, FishS2SynthesisResult


_PASSTHROUGH_PARAMS = {
    "chunk_length",
    "normalize",
    "seed",
    "top_p",
    "temperature",
    "repetition_penalty",
    "use_memory_cache",
    "references",
}


class FishS2NativeHttpBackend(FishS2Backend):
    """HTTP transport wrapper for Fish Speech's `/v1/tts` API."""

    PROVIDER_KEY = "fish_s2"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.base_url = str(self.config.get("base_url", "")).rstrip("/")
        self.api_key = self._normalize_api_key(self.config.get("api_key"))
        self.timeout = self.config.get("timeout", 60)

    async def health_check(self) -> bool:
        return bool(self.base_url)

    async def synthesize(
        self,
        *,
        text: str,
        response_format: str,
        streaming: bool,
        reference_id: str | None,
        extra_params: dict[str, Any] | None,
    ) -> FishS2SynthesisResult:
        payload = self._build_tts_payload(
            text=text,
            response_format=response_format,
            streaming=streaming,
            reference_id=reference_id,
            extra_params=extra_params,
        )

        if streaming:
            return self._stream_audio(payload)

        try:
            response = await afetch(
                method="POST",
                url=self._tts_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:
            self._raise_transport_error(exc)

        self._raise_for_response_error(response)
        return getattr(response, "content", b"") or b""

    async def add_reference(
        self,
        *,
        reference_id: str,
        audio_b64: str,
        reference_text: str,
    ) -> dict[str, Any]:
        payload = {
            "reference_id": reference_id,
            "audio_b64": audio_b64,
            "text": reference_text,
        }
        try:
            response = await afetch(
                method="POST",
                url=f"{self.base_url}/v1/references/add",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:
            self._raise_transport_error(exc)

        self._raise_for_response_error(response)
        data = self._response_json(response)
        if isinstance(data, dict):
            return data
        return {"reference_id": reference_id}

    async def delete_reference(self, *, reference_id: str) -> bool:
        try:
            response = await afetch(
                method="DELETE",
                url=f"{self.base_url}/v1/references/delete",
                headers=self._headers(),
                json={"reference_id": reference_id},
                timeout=self.timeout,
            )
        except Exception as exc:
            self._raise_transport_error(exc)

        self._raise_for_response_error(response)
        return True

    @property
    def _tts_url(self) -> str:
        return f"{self.base_url}/v1/tts"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _build_tts_payload(
        self,
        *,
        text: str,
        response_format: str,
        streaming: bool,
        reference_id: str | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "format": self._normalize_format(response_format),
            "streaming": streaming,
        }
        if reference_id:
            payload["reference_id"] = reference_id
        if extra_params:
            for key, value in extra_params.items():
                if key in _PASSTHROUGH_PARAMS and value is not None:
                    payload[key] = value
        return payload

    async def _stream_audio(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        try:
            async for chunk in astream_bytes(
                method="POST",
                url=self._tts_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            ):
                if chunk:
                    yield chunk
        except Exception as exc:
            self._raise_transport_error(exc)

    def _raise_for_response_error(self, response: Any) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code is None or status_code < 400:
            return

        body = getattr(response, "text", "") or ""
        headers = getattr(response, "headers", {}) or {}

        logger.error(
            "%s upstream returned %s: %s",
            self.PROVIDER_KEY,
            status_code,
            body,
        )
        self._raise_status_error(status_code=status_code, body=body, headers=headers)

    def _raise_status_error(
        self,
        *,
        status_code: int,
        body: str,
        headers: dict[str, Any],
    ) -> None:
        if status_code in (401, 403):
            raise auth_error(self.PROVIDER_KEY, "Fish S2 authentication failed")
        if status_code == 429:
            retry_after = headers.get("retry-after") if isinstance(headers, dict) else None
            raise rate_limit_error(
                self.PROVIDER_KEY,
                retry_after=int(retry_after) if retry_after else None,
            )
        if status_code in (400, 404):
            raise TTSValidationError(
                f"Fish S2 request failed ({status_code})",
                provider=self.PROVIDER_KEY,
                details={"status": status_code, "body": body},
            )
        if status_code in (408, 504):
            raise timeout_error(self.PROVIDER_KEY, timeout_seconds=self.timeout)
        if 500 <= status_code < 600:
            raise provider_error(
                "Fish S2 upstream error",
                provider=self.PROVIDER_KEY,
                error_code=str(status_code),
                details={"status": status_code, "body": body},
            )
        raise TTSProviderError(
            f"Fish S2 request failed ({status_code})",
            provider=self.PROVIDER_KEY,
            details={"status": status_code, "body": body},
        )

    def _raise_transport_error(self, exc: Exception) -> None:
        if isinstance(exc, (TTSAuthenticationError, TTSRateLimitError, TTSTimeoutError, TTSValidationError, TTSProviderError)):
            raise exc
        if self._is_timeout_error(exc):
            raise timeout_error(self.PROVIDER_KEY, timeout_seconds=self.timeout) from exc
        raise network_error(self.PROVIDER_KEY, exc) from exc

    @staticmethod
    def _response_json(response: Any) -> Any:
        try:
            json_fn = getattr(response, "json", None)
            if callable(json_fn):
                return json_fn()
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_api_key(value: Any) -> str | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw or raw.lower() in {"none", "null"}:
            return None
        return raw

    @staticmethod
    def _normalize_format(response_format: Any) -> str:
        value = getattr(response_format, "value", response_format)
        return str(value).lower()

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return True
        exc_name = exc.__class__.__name__.lower()
        return "timeout" in exc_name or "timeout" in str(exc).lower()
