"""Native Fish Audio HTTP backend for the Fish S2 provider."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

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


def _rebuild_typed_tts_error(exc: TTSError) -> TTSError:
    """Recreate a local TTS error category without copying provider-owned state."""

    error_class: type[TTSError] = type(exc)
    if error_class.__module__ != TTSError.__module__:
        error_class = TTSError
    return error_class(
        "Fish S2 request failed",
        provider="fish_s2",
        details={"error_type": error_class.__name__},
    )


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
        reference_id: str | list[str] | None,
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

        safe_error: Exception | None = None
        try:
            response = await afetch(
                method="POST",
                url=self._tts_url,
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
                    "Fish S2 reference audio must be valid base64",
                    provider=self.PROVIDER_KEY,
                    details={"reference_id": reference_id},
                )
            )

        form_data = {
            "id": reference_id,
            "text": reference_text,
        }
        files = {
            "audio": ("reference.wav", audio_bytes, "audio/wav"),
        }
        safe_error: Exception | None = None
        try:
            response = await afetch(
                method="POST",
                url=f"{self.base_url}/v1/references/add",
                headers=self._headers(),
                data=form_data,
                files=files,
                timeout=self.timeout,
                sensitive_observability=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize the backend boundary
            safe_error = self._normalize_transport_error(exc)
        if safe_error is not None:
            raise_detached_error(safe_error)

        self._raise_for_response_error(response)
        data = self._response_json(response)
        if isinstance(data, dict):
            return data
        return {"reference_id": reference_id}

    async def delete_reference(self, *, reference_id: str) -> bool:
        safe_error: Exception | None = None
        try:
            response = await afetch(
                method="DELETE",
                url=f"{self.base_url}/v1/references/delete",
                headers=self._headers(),
                json={"reference_id": reference_id},
                timeout=self.timeout,
                sensitive_observability=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize the backend boundary
            safe_error = self._normalize_transport_error(exc)
        if safe_error is not None:
            raise_detached_error(safe_error)

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
        reference_id: str | list[str] | None,
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
        safe_error: Exception | None = None
        try:
            async with contextlib.aclosing(
                astream_bytes(
                    method="POST",
                    url=self._tts_url,
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

    def _raise_for_response_error(self, response: Any) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code is None or status_code < 400:
            return

        headers = getattr(response, "headers", {}) or {}

        logger.error(
            "{} upstream returned status={}",
            self.PROVIDER_KEY,
            status_code,
        )
        self._raise_status_error(status_code=status_code, headers=headers)

    def _raise_status_error(
        self,
        *,
        status_code: int,
        headers: dict[str, Any],
    ) -> None:
        if status_code in (401, 403):
            raise_detached_error(
                auth_error(self.PROVIDER_KEY, "Fish S2 authentication failed")
            )
        if status_code == 429:
            retry_after = headers.get("retry-after") if isinstance(headers, dict) else None
            raise_detached_error(
                rate_limit_error(
                    self.PROVIDER_KEY,
                    retry_after=self._parse_retry_after(retry_after),
                )
            )
        if status_code in (400, 404):
            raise_detached_error(
                TTSValidationError(
                    f"Fish S2 request failed ({status_code})",
                    provider=self.PROVIDER_KEY,
                    details={"status": status_code},
                )
            )
        if status_code in (408, 504):
            raise_detached_error(
                timeout_error(self.PROVIDER_KEY, timeout_seconds=self.timeout)
            )
        if 500 <= status_code < 600:
            raise_detached_error(
                provider_error(
                    "Fish S2 upstream error",
                    provider=self.PROVIDER_KEY,
                    error_code=str(status_code),
                    details={"status": status_code},
                )
            )
        raise_detached_error(
            TTSProviderError(
                f"Fish S2 request failed ({status_code})",
                provider=self.PROVIDER_KEY,
                details={"status": status_code},
            )
        )

    def _normalize_transport_error(self, exc: Exception) -> Exception:
        """Create a sanitized domain error from a transport failure."""

        if isinstance(exc, TTSError):
            return _rebuild_typed_tts_error(exc)
        if self._is_timeout_error(exc):
            return timeout_error(self.PROVIDER_KEY, timeout_seconds=self.timeout)
        return TTSNetworkError(
            f"Network request to {self.PROVIDER_KEY} failed",
            provider=self.PROVIDER_KEY,
            error_code="NETWORK_ERROR",
            details={"error_type": type(exc).__name__},
        )

    def _raise_transport_error(self, exc: Exception) -> None:
        """Raise a sanitized transport error for compatibility with direct callers."""

        safe_error = self._normalize_transport_error(exc)
        raise_detached_error(safe_error)

    @staticmethod
    def _response_json(response: Any) -> Any:
        try:
            json_fn = getattr(response, "json", None)
            if callable(json_fn):
                return json_fn()
        except (TypeError, ValueError) as exc:
            logger.debug("Fish S2 response JSON parse failed: {}", type(exc).__name__)
            return None
        return None

    @staticmethod
    def _parse_retry_after(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
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
        if "timeout" in exc_name:
            return True
        return type(exc) is CoreNetworkError and any(
            isinstance(arg, str) and "timeout" in arg.lower()
            for arg in exc.args
        )
