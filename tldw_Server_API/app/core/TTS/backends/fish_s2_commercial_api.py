"""Commercial Fish Audio HTTP backend for the Fish S2 provider."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
from typing import Any

from tldw_Server_API.app.core.exceptions import (
    NetworkError as CoreNetworkError,
)
from tldw_Server_API.app.core.exceptions import (
    raise_detached_error,
)
from tldw_Server_API.app.core.http_client import afetch, astream_bytes

from ..tts_exceptions import (
    TTSError,
    TTSNetworkError,
    TTSProviderError,
    TTSValidationError,
    auth_error,
    provider_error,
    rate_limit_error,
    timeout_error,
)
from .fish_s2_base import FishS2SynthesisResult

_PASSTHROUGH_PARAMS = {
    "chunk_length",
    "condition_on_previous_chunks",
    "early_stop_threshold",
    "latency",
    "max_new_tokens",
    "min_chunk_length",
    "mp3_bitrate",
    "normalize",
    "opus_bitrate",
    "prosody",
    "reference_id",
    "references",
    "repetition_penalty",
    "sample_rate",
    "temperature",
    "top_p",
}


def _rebuild_typed_tts_error(exc: TTSError) -> TTSError:
    """Recreate a local TTS error category without copying provider-owned state."""

    error_class: type[TTSError] = type(exc)
    if error_class.__module__ != TTSError.__module__:
        error_class = TTSError
    return error_class(
        "Fish Audio request failed",
        provider="fish_s2",
        details={"error_type": error_class.__name__},
    )


class FishS2CommercialApiBackend:
    """HTTP transport wrapper for Fish Audio's hosted `/v1/tts` API."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.base_url = str(self.config.get("base_url") or "https://api.fish.audio").rstrip("/")
        self.api_key = self._normalize_api_key(self.config.get("api_key"))
        self.model = str(self.config.get("model") or "s2-pro").strip() or "s2-pro"
        self.timeout = self.config.get("timeout", 60)

    async def health_check(self) -> bool:
        return bool(self.base_url and self.api_key)

    async def synthesize(
        self,
        *,
        text: str,
        response_format: str,
        streaming: bool,
        reference_id: str | list[str] | None,
        extra_params: dict[str, Any] | None,
    ) -> FishS2SynthesisResult:
        payload = self._build_tts_payload(
            text=text,
            response_format=response_format,
            reference_id=reference_id,
            extra_params=extra_params,
        )
        if streaming:
            return self._stream_audio(payload)

        safe_error: Exception | None = None
        try:
            response = await afetch(
                method="POST",
                url=f"{self.base_url}/v1/tts",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                sensitive_observability=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize the backend boundary
            safe_error = self._normalize_transport_error(exc)
        if safe_error is not None:
            raise_detached_error(safe_error)

        self._raise_for_response_error(response)
        return getattr(response, "content", b"") or b""

    async def _stream_audio(self, payload: dict[str, Any]):
        safe_error: Exception | None = None
        try:
            async with contextlib.aclosing(
                astream_bytes(
                    method="POST",
                    url=f"{self.base_url}/v1/tts",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                    sensitive_observability=True,
                )
            ) as stream:
                async for chunk in stream:
                    if chunk:
                        yield chunk
        except Exception as exc:  # noqa: BLE001 - normalize the backend boundary
            safe_error = self._normalize_transport_error(exc)
        if safe_error is not None:
            raise_detached_error(safe_error)

    async def add_reference(
        self,
        *,
        reference_id: str,
        audio_b64: str,
        reference_text: str,
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError):
            audio_bytes = None
        if audio_bytes is None:
            del audio_b64
            raise_detached_error(
                TTSValidationError(
                    "Fish Audio reference audio must be valid base64",
                    provider="fish_s2",
                    details={"reference_id": reference_id},
                )
            )

        safe_error: Exception | None = None
        try:
            response = await afetch(
                method="POST",
                url=f"{self.base_url}/model",
                headers=self._auth_headers(),
                data={
                    "type": "tts",
                    "title": title or reference_id,
                    "description": description or "",
                    "train_mode": "fast",
                    "visibility": "private",
                    "texts": reference_text,
                    "enhance_audio_quality": "true",
                    "generate_sample": "false",
                },
                files={
                    "voices": ("reference.wav", audio_bytes, "audio/wav"),
                },
                timeout=self.timeout,
                sensitive_observability=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize the backend boundary
            safe_error = self._normalize_transport_error(exc)
        if safe_error is not None:
            raise_detached_error(safe_error)

        self._raise_for_response_error(response)
        data = self._response_json(response)
        remote_id = data.get("_id") if isinstance(data, dict) else None
        return {
            "reference_id": remote_id or reference_id,
            "remote_reference_id": remote_id or reference_id,
            "state": data.get("state") if isinstance(data, dict) else None,
        }

    async def delete_reference(self, *, reference_id: str) -> bool:
        safe_error: Exception | None = None
        try:
            response = await afetch(
                method="DELETE",
                url=f"{self.base_url}/model/{reference_id}",
                headers=self._auth_headers(),
                timeout=self.timeout,
                sensitive_observability=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize the backend boundary
            safe_error = self._normalize_transport_error(exc)
        if safe_error is not None:
            raise_detached_error(safe_error)

        self._raise_for_response_error(response)
        return True

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "model": self.model,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _raise_for_response_error(self, response: Any) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code is None or status_code < 400:
            return

        headers = getattr(response, "headers", {}) or {}
        raise_detached_error(
            self._normalized_status_error(
                status_code=status_code,
                headers=headers,
            )
        )

    def _normalized_status_error(
        self,
        *,
        status_code: int,
        headers: Any,
    ) -> TTSError:
        """Build a sanitized domain error for one upstream HTTP status."""

        if status_code in (401, 403):
            return auth_error("fish_s2", "Fish Audio authentication failed")
        if status_code == 429:
            retry_after = headers.get("retry-after") if isinstance(headers, dict) else None
            return rate_limit_error(
                "fish_s2",
                retry_after=self._parse_retry_after(retry_after),
            )
        if status_code in (400, 404, 422):
            return TTSValidationError(
                f"Fish Audio request failed ({status_code})",
                provider="fish_s2",
                details={"status": status_code},
            )
        if status_code in (408, 504):
            return timeout_error("fish_s2", timeout_seconds=self.timeout)
        if status_code == 402:
            return TTSProviderError(
                "Fish Audio payment required",
                provider="fish_s2",
                error_code=str(status_code),
                details={"status": status_code},
            )
        if 500 <= status_code < 600:
            return provider_error(
                "Fish Audio upstream error",
                provider="fish_s2",
                error_code=str(status_code),
                details={"status": status_code},
            )
        return TTSProviderError(
            f"Fish Audio request failed ({status_code})",
            provider="fish_s2",
            details={"status": status_code},
        )

    def _normalize_transport_error(self, exc: Exception) -> Exception:
        """Create a sanitized domain error from a transport failure."""

        if isinstance(exc, TTSError):
            return _rebuild_typed_tts_error(exc)
        if isinstance(exc, CoreNetworkError) and exc.status_code is not None:
            return self._normalized_status_error(
                status_code=exc.status_code,
                headers={},
            )
        if self._is_timeout_error(exc):
            return timeout_error("fish_s2", timeout_seconds=self.timeout)
        return TTSNetworkError(
            "Network request to fish_s2 failed",
            provider="fish_s2",
            error_code="NETWORK_ERROR",
            details={"error_type": type(exc).__name__},
        )

    def _raise_transport_error(self, exc: Exception) -> None:
        """Raise a sanitized transport error for compatibility with direct callers."""

        safe_error = self._normalize_transport_error(exc)
        raise_detached_error(safe_error)

    def _build_tts_payload(
        self,
        *,
        text: str,
        response_format: str,
        reference_id: str | list[str] | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "format": self._normalize_format(response_format),
        }
        if reference_id:
            payload["reference_id"] = reference_id
        if extra_params:
            for key, value in extra_params.items():
                if key in _PASSTHROUGH_PARAMS and value is not None:
                    payload[key] = value
        return payload

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
    def _response_json(response: Any) -> dict[str, Any]:
        try:
            json_fn = getattr(response, "json", None)
            if callable(json_fn):
                data = json_fn()
                if isinstance(data, dict):
                    return data
        except Exception:  # noqa: BLE001 - third-party response implementations vary
            return {}
        return {}

    @staticmethod
    def _parse_retry_after(value: Any) -> int | None:
        try:
            return int(value) if value else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return True
        exc_name = exc.__class__.__name__.lower()
        if "timeout" in exc_name:
            return True
        return type(exc) is CoreNetworkError and any(
            isinstance(arg, str) and "timeout" in arg.lower()
            for arg in exc.args
        )
