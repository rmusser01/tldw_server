"""
Base interfaces and helpers for LLM provider adapters.

Adapters implement a unified ChatProvider interface and are responsible for:
- Auth + base URL resolution
- Request payload shaping (OpenAI-like input)
- Streaming normalization via shared SSE helpers
- Error mapping to Chat*Error types

Adapters should return OpenAI-compatible chat completion JSON for non-streaming
and yield OpenAI-compatible SSE lines for streaming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Any, NoReturn

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError


def raise_if_in_band_provider_error(
    provider: str,
    value: Any,
    *,
    phase: str,
) -> None:
    """Reject structured HTTP-200 and SSE error envelopes safely."""

    from tldw_Server_API.app.core.Chat.streaming_utils import (
        normalize_provider_stream_error,
    )
    from tldw_Server_API.app.core.exceptions import raise_detached_error
    from tldw_Server_API.app.core.LLM_Calls.error_utils import (
        build_sanitized_chat_error,
        log_provider_failure,
    )

    normalized = normalize_provider_stream_error(value)
    if normalized is None:
        return
    error = build_sanitized_chat_error(
        provider,
        status_code=normalized.status_code,
    )
    log_provider_failure(
        provider,
        error,
        phase=phase,
        status_code=normalized.status_code,
    )
    raise_detached_error(error)


class ChatProvider(ABC):
    """Abstract base for LLM chat providers."""

    name: str = "provider"
    async_chat_is_native: bool = False

    def _bind_request_credentials(
        self,
        request: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Consume one authentic runtime capability before payload validation."""

        bound, _credentials = self._bind_request_credentials_with_handle(request)
        return bound

    def _bind_request_credentials_with_handle(
        self,
        request: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], Any | None]:
        """Bind credentials while retaining the handle for scoped transports."""

        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
            bind_provider_call_credentials,
        )

        return bind_provider_call_credentials(self.name, request, consume=True)

    def _raise_sanitized_provider_failure(
        self,
        exc: Exception,
        *,
        phase: str,
        credential_refresh_retry_safe: bool = False,
    ) -> NoReturn:
        """Raise one detached typed error without upstream body, URL, or cause."""

        from tldw_Server_API.app.core.exceptions import raise_detached_error
        from tldw_Server_API.app.core.LLM_Calls.error_utils import (
            build_sanitized_chat_error,
            get_http_status_from_exception,
            log_provider_failure,
        )

        status = get_http_status_from_exception(exc)
        log_provider_failure(self.name, exc, phase=phase, status_code=status)
        error = build_sanitized_chat_error(self.name, status_code=status)
        if status is not None:
            error.upstream_status_code = status
        if credential_refresh_retry_safe and status == 401:
            error.credential_refresh_retry_safe = True
        raise_detached_error(error)

    def _raise_if_in_band_provider_error(
        self,
        value: Any,
        *,
        phase: str,
    ) -> None:
        """Reject structured HTTP-200 and SSE error envelopes safely."""

        raise_if_in_band_provider_error(self.name, value, phase=phase)

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return provider capability flags and hints.

        Example keys:
        - supports_streaming: bool
        - supports_tools: bool
        - json_mode: bool
        - default_timeout_seconds: int
        - max_output_tokens_default: Optional[int]
        """

    @abstractmethod
    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Non-streaming chat completion (OpenAI-compatible response)."""

    @abstractmethod
    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Streaming chat completion.

        Yields OpenAI-compatible SSE lines. Callers are responsible for emitting a
        final [DONE] using sse.finalize_stream() to avoid duplicates.
        """

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Async variant; adapters may override for native async paths.

        Native async implementations must also set ``async_chat_is_native`` to
        true. The default raises instead of silently running sync work inline.
        """
        raise NotImplementedError("Async chat not implemented for this provider")

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        """Async streaming variant; adapters may override for native async paths."""
        raise NotImplementedError("Async stream not implemented for this provider")

    def normalize_error(self, exc: Exception) -> ChatAPIError:
        """Map arbitrary exceptions to project Chat*Error classes.

        Adapters may override for provider-specific error shapes. This default
        provides a conservative mapping for common HTTP exceptions if available,
        falling back to ChatProviderError.
        """
        from tldw_Server_API.app.core.LLM_Calls.error_utils import (
            build_sanitized_chat_error,
            get_http_status_from_exception,
            is_http_status_error,
            log_provider_failure,
            log_provider_failure_metadata,
            privacy_safe_chat_error,
            provider_errors_are_privacy_safe,
        )

        if provider_errors_are_privacy_safe():
            log_provider_failure_metadata(self.name, exc, phase="normalization")
            return privacy_safe_chat_error(self.name, exc)

        if is_http_status_error(exc):
            status = get_http_status_from_exception(exc)
            log_provider_failure(
                self.name,
                exc,
                phase="normalize_http_error",
                status_code=status,
            )
            return build_sanitized_chat_error(self.name, status_code=status)

        # Fallback
        log_provider_failure(
            self.name,
            exc,
            phase="normalize_generic_error",
        )
        return build_sanitized_chat_error(self.name)


def apply_tool_choice(payload: dict[str, Any], tools: list | None, tool_choice: Any | None) -> None:
    """Safely set tool_choice only when supported.

    - Always honor explicit "none" to disable tools.
    - Apply tool_choice only if provided and tools list is present.
    """
    try:
        if tool_choice == "none":
            payload["tool_choice"] = "none"
        elif tool_choice is not None and tools:
            payload["tool_choice"] = tool_choice
    except Exception as payload_error:
        # Never fail due to helper
        logger.debug(
            "Provider payload helper failed while attaching tool metadata error_type={}",
            type(payload_error).__name__,
        )


class EmbeddingsAdapterUnavailableError(NotImplementedError):
    """Signal that an embedding adapter declined before any provider dispatch."""


class EmbeddingsProvider(ABC):
    """Abstract base for embeddings providers.

    Implementations should return OpenAI-compatible embeddings responses or
    a plain list/array of floats when used as a library.
    """

    name: str = "embeddings_provider"

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return provider capability flags and hints.

        Example keys:
        - dimensions_default: Optional[int]
        - max_batch_size: Optional[int]
        - default_timeout_seconds: int
        """

    @abstractmethod
    def embed(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Create embeddings for given input(s).

        Request shape should accept keys similar to OpenAI's API:
        - input: Union[str, List[str]]
        - model: str
        - api_key: Optional[str]
        - user: Optional[str]
        - encoding_format: Optional[str]
        """
