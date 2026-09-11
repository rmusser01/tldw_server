"""Cancellation-safe provider calls bound to one credential runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    is_runtime_issued_provider_call_credentials,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker
from tldw_Server_API.app.core.Chat.streaming_utils import (
    invoke_stream_close_bounded,
    normalize_provider_stream_error,
)

_T = TypeVar("_T")


def attach_runtime_provider_credentials(
    call_kwargs: dict[str, Any],
    credential_handle: ProviderCallCredentials | None,
) -> None:
    """Attach one authentic runtime handle to an in-memory provider request."""

    if credential_handle is None:
        return
    provider = call_kwargs.get("api_provider") or call_kwargs.get("api_endpoint")
    if not is_runtime_issued_provider_call_credentials(
        credential_handle,
        provider=str(provider or ""),
    ):
        raise RuntimeError("Provider credentials were not issued for this RAG call")
    call_kwargs[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = credential_handle


def provider_result_succeeded(result: Any) -> bool:
    """Return whether a completed adapter result represents provider success."""

    if normalize_provider_stream_error(result) is not None:
        return False
    if isinstance(result, str):
        return bool(result.strip()) and not result.lstrip().lower().startswith("error:")
    if not isinstance(result, dict):
        return False

    choices = result.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return True

    return any(
        isinstance(result.get(field), str) and result[field].strip()
        for field in ("content", "text")
    )


async def await_runtime_bound_provider_call(
    awaitable: Awaitable[_T],
    *,
    credential_runtime: Any,
    credential_handle: Any,
    mark_success: bool = True,
) -> _T:
    """Drain one provider call and its usage mark before cancellation escapes."""

    if credential_runtime is None or credential_handle is None:
        return await awaitable

    async def _call_and_mark() -> _T:
        result = await awaitable
        if mark_success and provider_result_succeeded(result):
            await credential_runtime.mark_used(credential_handle)
        return result

    return await await_owned_worker(
        _call_and_mark(),
        on_cancel_result=close_provider_stream,
    )


async def close_provider_stream(iterator: Any) -> None:
    """Close one provider-backed iterator within the shared cleanup bound."""

    close = getattr(iterator, "aclose", None)
    if not callable(close):
        close = getattr(iterator, "close", None)
    if not callable(close):
        return
    try:
        await invoke_stream_close_bounded(close)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - cleanup cannot replace the stream error
        # Cleanup is best-effort and must not replace the original stream error.
        return


__all__ = [
    "attach_runtime_provider_credentials",
    "await_runtime_bound_provider_call",
    "close_provider_stream",
    "provider_result_succeeded",
]
