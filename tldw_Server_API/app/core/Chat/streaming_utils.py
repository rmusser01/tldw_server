# streaming_utils.py
# Description: Utilities for handling streaming responses safely
#
# Imports
import asyncio
import concurrent.futures
import contextlib
import inspect
import json
import math
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Union

from loguru import logger

from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    DaemonCapacityError,
    await_bounded_daemon_with_timeout,
    daemon_capacity_from_env,
    start_bounded_stream_cleanup_daemon,
    start_bounded_stream_daemon,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatConfigurationError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.testing import is_truthy

#######################################################################################################################
#
# Constants:
# Load configuration values

_STREAMING_NONCRITICAL_EXCEPTIONS = (
    ChatAPIError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
)

_SSE_CONTROL_PREFIXES = (":", "event:")
_SSE_FRAMED_CONTROL_PREFIXES = ("id:", "retry:")
_OPENAI_STREAM_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "function_call",
    "content_filter",
}
STREAM_TASK_CANCEL_DRAIN_SECONDS = 0.05
STREAM_CLEANUP_TIMEOUT_SECONDS = 0.05
STREAM_TASK_MAX_ACTIVE = daemon_capacity_from_env(
    "CHAT_STREAM_ASYNC_MAX_TASKS",
    default=256,
)
STREAM_CLEANUP_TASK_MAX_ACTIVE = daemon_capacity_from_env(
    "CHAT_STREAM_ASYNC_CLEANUP_MAX_TASKS",
    default=32,
)

_STREAM_TASK_CAPACITY_LOCK = threading.Lock()
_STREAM_TASK_ACTIVE_COUNT = 0
_STREAM_CLEANUP_TASK_ACTIVE_COUNT = 0


class StreamTaskCapacityError(RuntimeError):
    """Raised before async stream work can exceed its process-wide bound."""


_TRUSTED_LOCAL_STREAM_FRAME_PROVENANCE = object()
_TRUSTED_LOCAL_STREAM_ERROR_MESSAGES = {
    "unsupported_multi_choice_tool_autoexec": (
        "Local tool auto-execution supports one assistant choice per request."
    ),
}


class _TrustedLocalStreamFrame(str):
    """Service-created SSE control frame carrying private provenance."""


def _trusted_local_stream_frame(value: str) -> str:
    frame = _TrustedLocalStreamFrame(value)
    frame._tldw_local_stream_provenance = _TRUSTED_LOCAL_STREAM_FRAME_PROVENANCE
    return frame


def trusted_local_stream_error_frame(code: str) -> str:
    """Build an allowlisted local SSE error frame with private provenance."""

    message = _TRUSTED_LOCAL_STREAM_ERROR_MESSAGES.get(code)
    if message is None:
        raise ValueError("Unsupported local stream error code")
    return _trusted_local_stream_frame(
        f"data: {json.dumps({'error': {'code': code, 'type': code, 'message': message}})}\n\n"
    )


def is_trusted_local_stream_frame(value: Any) -> bool:
    """Return whether a frame was created by the local stream controller."""

    return (
        isinstance(value, _TrustedLocalStreamFrame)
        and getattr(value, "_tldw_local_stream_provenance", None)
        is _TRUSTED_LOCAL_STREAM_FRAME_PROVENANCE
    )


def _release_stream_task_capacity(
    _task: asyncio.Future[Any] | None = None,
    *,
    cleanup: bool = False,
) -> None:
    """Release one async stream-task lease after the task actually exits."""

    global _STREAM_CLEANUP_TASK_ACTIVE_COUNT, _STREAM_TASK_ACTIVE_COUNT
    with _STREAM_TASK_CAPACITY_LOCK:
        if cleanup:
            _STREAM_CLEANUP_TASK_ACTIVE_COUNT -= 1
        else:
            _STREAM_TASK_ACTIVE_COUNT -= 1


def create_bounded_stream_task(
    awaitable: Any,
    *,
    cleanup: bool = False,
) -> asyncio.Future[Any]:
    """Schedule stream work only while process-wide async capacity is available."""

    global _STREAM_CLEANUP_TASK_ACTIVE_COUNT, _STREAM_TASK_ACTIVE_COUNT
    with _STREAM_TASK_CAPACITY_LOCK:
        if cleanup:
            if _STREAM_CLEANUP_TASK_ACTIVE_COUNT >= STREAM_CLEANUP_TASK_MAX_ACTIVE:
                admitted = False
            else:
                _STREAM_CLEANUP_TASK_ACTIVE_COUNT += 1
                admitted = True
        elif _STREAM_TASK_ACTIVE_COUNT >= STREAM_TASK_MAX_ACTIVE:
            admitted = False
        else:
            _STREAM_TASK_ACTIVE_COUNT += 1
            admitted = True
    if not admitted:
        _discard_unawaited_close_result(awaitable)
        raise StreamTaskCapacityError("Async streaming task capacity is exhausted")

    try:
        task = asyncio.ensure_future(awaitable)
    except BaseException:
        _release_stream_task_capacity(cleanup=cleanup)
        _discard_unawaited_close_result(awaitable)
        raise
    task.add_done_callback(
        lambda completed: _release_stream_task_capacity(
            completed,
            cleanup=cleanup,
        )
    )
    return task


async def await_bounded_owned_operation(
    awaitable: Any,
    *,
    timeout_seconds: float,
    timeout_message: str,
    on_abandoned: Callable[[], Any],
    released_event: threading.Event | None = None,
    cleanup_claimed: threading.Event | None = None,
    on_cancel_success: Callable[[], Any] | None = None,
    on_cancel_result: Callable[[Any], Any] | None = None,
    on_abandoned_success: Callable[[], Any] | None = None,
) -> Any:
    """Apply a hard deadline while retaining late work and its resources.

    Cleanup capacity is reserved before the operation starts.  Timeout and
    caller cancellation transfer cleanup to a detached bounded task without
    cancelling the operation; the callback runs only after the operation and
    any associated daemon worker have really exited.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        _discard_unawaited_close_result(awaitable)
        raise ValueError("Owned operation timeout must be a positive finite number")

    loop = asyncio.get_running_loop()
    operation_ready: asyncio.Future[asyncio.Future[Any]] = loop.create_future()
    cleanup_decision: asyncio.Future[str | None] = loop.create_future()

    async def cleanup_late_operation() -> None:
        operation = await operation_ready
        abandonment_reason = await cleanup_decision
        if abandonment_reason is None:
            return

        operation_succeeded = False
        operation_result: Any = None
        try:
            operation_result = await operation
        except BaseException:  # noqa: BLE001 - late provider outcomes are consumed
            pass
        else:
            operation_succeeded = True

        while released_event is not None and not released_event.is_set():
            await asyncio.sleep(0.01)

        if operation_succeeded and on_abandoned_success is not None:
            try:
                late_success_result = on_abandoned_success()
                if inspect.isawaitable(late_success_result):
                    await late_success_result
            except BaseException as exc:  # noqa: BLE001 - usage marking is best effort
                logger.warning(
                    "Late owned-operation success callback failed (type={})",
                    type(exc).__name__,
                )

        if (
            abandonment_reason == "caller_cancel"
            and operation_succeeded
            and on_cancel_result is not None
        ):
            try:
                cancel_result = on_cancel_result(operation_result)
                if inspect.isawaitable(cancel_result):
                    await cancel_result
            except BaseException as exc:  # noqa: BLE001 - usage marking is best effort
                logger.warning(
                    "Late owned-operation result callback failed (type={})",
                    type(exc).__name__,
                )

        if (
            abandonment_reason == "caller_cancel"
            and operation_succeeded
            and on_cancel_success is not None
        ):
            try:
                cancel_result = on_cancel_success()
                if inspect.isawaitable(cancel_result):
                    await cancel_result
            except BaseException as exc:  # noqa: BLE001 - usage marking is best effort
                logger.warning(
                    "Late owned-operation success callback failed (type={})",
                    type(exc).__name__,
                )

        try:
            cleanup_result = on_abandoned()
            if inspect.isawaitable(cleanup_result):
                await cleanup_result
        except BaseException as exc:  # noqa: BLE001 - cleanup is best effort and detached
            logger.warning(
                "Late owned-operation cleanup failed (type={})",
                type(exc).__name__,
            )

    cleanup_task: asyncio.Future[Any]
    try:
        cleanup_task = create_bounded_stream_task(
            cleanup_late_operation(),
            cleanup=True,
        )
    except BaseException:
        _discard_unawaited_close_result(awaitable)
        raise
    cleanup_task.add_done_callback(_observe_stream_task)

    try:
        operation_task = create_bounded_stream_task(awaitable)
    except BaseException:
        completed: asyncio.Future[Any] = loop.create_future()
        completed.set_result(None)
        operation_ready.set_result(completed)
        cleanup_decision.set_result(None)
        await asyncio.shield(cleanup_task)
        raise
    operation_ready.set_result(operation_task)

    def abandon(reason: str) -> None:
        if cleanup_claimed is not None:
            cleanup_claimed.set()
        if not cleanup_decision.done():
            cleanup_decision.set_result(reason)

    try:
        done, _pending = await asyncio.wait(
            {operation_task},
            timeout=float(timeout_seconds),
        )
    except asyncio.CancelledError:
        abandon("caller_cancel")
        raise

    if operation_task not in done:
        abandon("timeout")
        raise TimeoutError(timeout_message)

    if operation_task.cancelled():
        cleanup_decision.set_result(None)
        await asyncio.shield(cleanup_task)
        return operation_task.result()

    if (
        released_event is not None
        and not released_event.is_set()
        and isinstance(operation_task.exception(), TimeoutError)
    ):
        abandon("timeout")
    else:
        cleanup_decision.set_result(None)
        await asyncio.shield(cleanup_task)
    return operation_task.result()

PROVIDER_STREAM_ERROR_MESSAGES = {
    "provider_authentication_failed": "The selected provider credentials could not be authenticated.",
    "invalid_provider_credentials": "The selected provider credentials are invalid.",
    "missing_provider_credentials": "The selected provider credentials are not configured.",
    "credential_store_unavailable": "Provider credential storage is temporarily unavailable.",
    "credential_scope_revoked": "The selected provider credential scope is no longer available.",
    "provider_disabled": "The selected provider is disabled by administrator policy.",
    "model_not_allowed": "The selected model is not allowed for this provider.",
    "provider_configuration_invalid": "The selected provider configuration is invalid.",
    "provider_unavailable": "The chat service provider is currently unavailable.",
}
_PROVIDER_STREAM_ERROR_STATUS = {
    "provider_authentication_failed": 502,
    "provider_unavailable": 502,
    "provider_disabled": 403,
    "model_not_allowed": 403,
}
_PROVIDER_STREAM_ERROR_TYPE_CODES = {
    "chatauthenticationerror": "provider_authentication_failed",
    "chatconfigurationerror": "provider_configuration_invalid",
    "chatprovidererror": "provider_unavailable",
    "chatapierror": "provider_unavailable",
}


@dataclass(frozen=True, slots=True)
class NormalizedProviderStreamError:
    """Bounded public provider error and its explicit replay capability."""

    code: str
    message: str
    status_code: int
    replay_certified: bool = False
    credential_refresh_retry_certified: bool = False


def _provider_stream_code(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized in PROVIDER_STREAM_ERROR_MESSAGES:
        return normalized
    return _PROVIDER_STREAM_ERROR_TYPE_CODES.get(normalized.lower())


def _provider_stream_error_candidate(value: Any) -> tuple[Any, bool]:
    """Return an error candidate and whether an error envelope was present."""
    if isinstance(value, BaseException):
        return value, True
    if isinstance(value, dict):
        if "error" in value and value.get("error") is not None:
            return value.get("error"), True
        if value.get("type") == "error" or value.get("error_code") is not None:
            return value, True
        return None, False
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
    else:
        return None, False

    for raw_line in text.splitlines() or [text]:
        candidate_text = raw_line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
        if not candidate_text:
            continue
        if not candidate_text.startswith("data:"):
            continue
        candidate_text = candidate_text[len("data:") :].strip()
        try:
            decoded = json.loads(candidate_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        candidate, present = _provider_stream_error_candidate(decoded)
        if present:
            return candidate, True
    return None, False


def _normalized_provider_error_for_code(value: Any) -> Optional[NormalizedProviderStreamError]:
    """Build a canonical error from one explicit internal code or type name."""

    code = _provider_stream_code(value)
    if code is None:
        return None
    return NormalizedProviderStreamError(
        code=code,
        message=PROVIDER_STREAM_ERROR_MESSAGES[code],
        status_code=_PROVIDER_STREAM_ERROR_STATUS.get(code, 503),
    )


def normalize_provider_stream_error(value: Any) -> Optional[NormalizedProviderStreamError]:
    """Normalize raised or in-band provider failures without retaining raw detail."""
    if isinstance(value, NormalizedProviderStreamError):
        return value

    if isinstance(value, bytes):
        explicit_text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        explicit_text = value
    else:
        explicit_text = None
    if explicit_text is not None:
        explicit_error = _normalized_provider_error_for_code(explicit_text)
        explicit_error = explicit_error or _normalize_unframed_provider_stream_error(
            explicit_text
        )
        if explicit_error is not None:
            return explicit_error

    candidate, present = _provider_stream_error_candidate(value)
    if not present:
        return None

    code: Optional[str] = None
    if isinstance(candidate, ChatAuthenticationError):
        code = "provider_authentication_failed"
    elif isinstance(candidate, ChatConfigurationError):
        code = _provider_stream_code(getattr(candidate, "error_code", None))
        code = code or "provider_configuration_invalid"
    elif isinstance(candidate, BaseException):
        code = _provider_stream_code(getattr(candidate, "code", None))
        code = code or _provider_stream_code(getattr(candidate, "error_code", None))
        detail = getattr(candidate, "detail", None)
        if code is None and isinstance(detail, dict):
            code = _provider_stream_code(detail.get("error_code") or detail.get("code"))
        if code is None:
            code = _provider_stream_code(type(candidate).__name__)
    elif isinstance(candidate, dict):
        code = _provider_stream_code(
            candidate.get("error_code") or candidate.get("code") or candidate.get("type")
        )
    else:
        code = _provider_stream_code(candidate)

    if code not in PROVIDER_STREAM_ERROR_MESSAGES:
        code = "provider_unavailable"
    replay_certified = bool(
        isinstance(value, ChatAPIError)
        and getattr(value, "upstream_dispatched", None) is False
        and getattr(value, "output_emitted", None) is False
        and getattr(value, "allow_non_stream_fallback", None) is True
    )
    credential_refresh_retry_certified = bool(
        isinstance(candidate, BaseException)
        and getattr(candidate, "credential_refresh_retry_safe", None) is True
    )
    return NormalizedProviderStreamError(
        code=code,
        message=PROVIDER_STREAM_ERROR_MESSAGES[code],
        status_code=_PROVIDER_STREAM_ERROR_STATUS.get(code, 503),
        replay_certified=replay_certified,
        credential_refresh_retry_certified=credential_refresh_retry_certified,
    )


def _normalize_unframed_provider_stream_error(
    value: str,
) -> Optional[NormalizedProviderStreamError]:
    """Accept only an explicit canonical code from an unframed JSON string.

    A provider may legitimately stream assistant-authored JSON containing an
    ``error`` field.  Arbitrary error envelopes are therefore authoritative
    only when carried by an SSE ``data:`` frame (or as a structured adapter
    object before string conversion).  The raw-string compatibility path is
    intentionally narrower and recognizes only our bounded internal codes.
    """

    try:
        decoded = json.loads(value.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    candidate, present = _provider_stream_error_candidate(decoded)
    if not present:
        return None
    if isinstance(candidate, dict):
        code = _provider_stream_code(
            candidate.get("error_code") or candidate.get("code") or candidate.get("type")
        )
    else:
        code = _provider_stream_code(candidate)
    return _normalized_provider_error_for_code(code)


def provider_result_contains_error(
    value: Any,
    *,
    legacy_error_prefix: bool = False,
) -> bool:
    """Return whether a supported response structure contains a provider error.

    Only response-bearing fields are traversed. Tool inputs and function
    arguments are deliberately excluded so domain objects containing an
    ``error`` key remain valid tool data.
    """

    seen: set[int] = set()

    def visit(item: Any) -> bool:
        if normalize_provider_stream_error(item) is not None:
            return True
        if isinstance(item, str):
            if not legacy_error_prefix:
                return False
            if item.lstrip().lower().startswith("error:"):
                return True
            for raw_line in item.splitlines():
                line = raw_line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
                if line.startswith("data:") and line[len("data:") :].lstrip().lower().startswith(
                    "error:"
                ):
                    return True
            return False
        if isinstance(item, bytes):
            return False
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return False
            seen.add(identity)
            return any(visit(nested) for nested in item)
        if not isinstance(item, dict):
            return False

        identity = id(item)
        if identity in seen:
            return False
        seen.add(identity)

        for field in ("choices", "message", "delta", "content"):
            if field in item and visit(item[field]):
                return True

        block_type = item.get("type")
        if block_type in {"text", "output_text"} and "text" in item:
            return visit(item["text"])
        return False

    return visit(value)


def provider_payload_structural_error_code(value: Any) -> str | None:
    """Return a bounded error code from protocol-owned response envelopes."""

    seen: set[int] = set()

    def visit(item: Any) -> str | None:
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            for nested in item:
                code = visit(nested)
                if code is not None:
                    return code
            return None
        if not isinstance(item, dict):
            return None

        identity = id(item)
        if identity in seen:
            return None
        seen.add(identity)
        normalized = normalize_provider_stream_error(item)
        if normalized is not None:
            return normalized.code

        for field in ("choices", "message", "delta"):
            if field in item:
                code = visit(item[field])
                if code is not None:
                    return code

        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    code = visit(block)
                    if code is not None:
                        return code
        return None

    return visit(value)


def provider_payload_has_structural_error(value: Any) -> bool:
    """Inspect protocol-owned response envelopes without classifying output text."""

    return provider_payload_structural_error_code(value) is not None


def provider_stream_error_payload(value: Any) -> dict[str, dict[str, str]]:
    """Build the canonical SSE/HTTP-safe provider error payload."""
    normalized = _normalized_provider_error_for_code(value)
    normalized = normalized or normalize_provider_stream_error(value)
    if normalized is None:
        normalized = NormalizedProviderStreamError(
            code="provider_unavailable",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"],
            status_code=502,
        )
    return {
        "error": {
            "code": normalized.code,
            "type": normalized.code,
            "message": normalized.message,
        }
    }


def sanitized_provider_stream_exception(value: Any) -> SanitizedProviderStreamError:
    """Return a core safe exception for one raised provider failure."""
    normalized = _normalized_provider_error_for_code(value)
    normalized = normalized or normalize_provider_stream_error(value)
    if normalized is None:
        normalized = NormalizedProviderStreamError(
            code="provider_unavailable",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"],
            status_code=502,
        )
    return SanitizedProviderStreamError(
        code=normalized.code,
        message=normalized.message,
        status_code=normalized.status_code,
        replay_certified=normalized.replay_certified,
        credential_refresh_retry_certified=(
            normalized.credential_refresh_retry_certified
        ),
    )


def provider_stream_error_allows_replay(value: Any) -> bool:
    """Return True only for a literal trusted pre-dispatch certificate."""
    normalized = normalize_provider_stream_error(value)
    return normalized is not None and normalized.replay_certified is True


def _observe_stream_task(task: asyncio.Future[Any]) -> None:
    """Consume a detached task result so late completion stays silent."""

    try:
        task.exception()
    except asyncio.CancelledError:
        return


async def cancel_stream_tasks_bounded(
    tasks: list[asyncio.Future[Any]] | tuple[asyncio.Future[Any], ...],
    timeout: float | None = None,
) -> None:
    """Cancel and briefly drain tasks without trusting cooperative cancellation."""

    pending = [task for task in tasks if task is not None and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        done: set[asyncio.Future[Any]] = set()
        still_pending = set(pending)
        try:
            done, still_pending = await asyncio.wait(
                still_pending,
                timeout=max(
                    0.0,
                    STREAM_TASK_CANCEL_DRAIN_SECONDS if timeout is None else timeout,
                ),
            )
        finally:
            for task in done:
                _observe_stream_task(task)
            for task in still_pending:
                task.add_done_callback(_observe_stream_task)
    for task in tasks:
        if task is not None and task.done():
            _observe_stream_task(task)


async def await_stream_operation_bounded(
    awaitable: Any,
    timeout: float | None = None,
    *,
    cleanup: bool = False,
) -> Any:
    """Await an operation with a hard wall-clock bound, then detach if resistant."""

    budget = STREAM_CLEANUP_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    if budget <= 0:
        _discard_unawaited_close_result(awaitable)
        raise asyncio.TimeoutError
    task = create_bounded_stream_task(awaitable, cleanup=cleanup)
    try:
        done, _ = await asyncio.wait(
            {task},
            timeout=max(
                0.0,
                budget,
            ),
        )
    except asyncio.CancelledError:
        await cancel_stream_tasks_bounded([task])
        raise
    if task not in done:
        await cancel_stream_tasks_bounded([task])
        raise asyncio.TimeoutError
    return task.result()


def _discard_unawaited_close_result(value: Any) -> None:
    """Dispose of a late close awaitable without executing it on the wrong loop."""

    if inspect.iscoroutine(value):
        value.close()
    elif isinstance(value, asyncio.Future):
        value.cancel()


async def _invoke_sync_close_bounded(close: Callable[[], Any], timeout: float) -> Any:
    """Invoke a regular-def close off-loop and wait for its daemon lease release."""

    if timeout <= 0:
        raise asyncio.TimeoutError
    deadline = time.monotonic() + timeout
    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[Any] = loop.create_future()
    abandoned = threading.Event()
    delivered = threading.Event()
    worker_released = threading.Event()

    def deliver(value: Any = None, error: BaseException | None = None) -> None:
        try:
            if abandoned.is_set() or result_future.done():
                if error is None:
                    _discard_unawaited_close_result(value)
                return
            if error is not None:
                result_future.set_exception(error)
            else:
                result_future.set_result(value)
        finally:
            delivered.set()

    def worker() -> None:
        try:
            value = close()
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                loop.call_soon_threadsafe(deliver, None, error)
            except RuntimeError:
                return
        else:
            try:
                loop.call_soon_threadsafe(deliver, value, None)
            except RuntimeError:
                _discard_unawaited_close_result(value)
                return
        while not delivered.wait(0.05):
            if loop.is_closed():
                if "value" in locals():
                    _discard_unawaited_close_result(value)
                return

    start_bounded_stream_cleanup_daemon(
        worker,
        name="async-stream-close-invocation",
        released_event=worker_released,
    )
    value: Any = None
    value_delivered = False
    try:
        value = await await_stream_operation_bounded(
            result_future,
            max(0.0, deadline - time.monotonic()),
            cleanup=True,
        )
        value_delivered = True
        while not worker_released.is_set():
            if time.monotonic() >= deadline:
                abandoned.set()
                raise asyncio.TimeoutError
            await asyncio.sleep(0)
        return value
    except (asyncio.TimeoutError, asyncio.CancelledError):
        abandoned.set()
        if value_delivered:
            _discard_unawaited_close_result(value)
        raise
    except BaseException:
        while not worker_released.is_set():
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0)
        raise


async def invoke_stream_close_bounded(
    close: Callable[[], Any],
    timeout: float | None = None,
) -> Any:
    """Invoke and await a stream close without blocking the event-loop thread."""

    budget = STREAM_CLEANUP_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    if budget <= 0:
        raise asyncio.TimeoutError
    deadline = time.monotonic() + budget
    close_owner = getattr(close, "__self__", None)
    if inspect.iscoroutinefunction(close) or inspect.isasyncgen(close_owner):
        result = close()
    else:
        result = await _invoke_sync_close_bounded(close, budget)
    if not inspect.isawaitable(result):
        return result
    return await await_stream_operation_bounded(
        result,
        max(0.0, deadline - time.monotonic()),
        cleanup=True,
    )


async def invoke_owned_stream_close(
    close: Callable[[], Any],
    timeout: float | None = None,
) -> Any:
    """Close a late-owned stream without abandoning a resistant close worker.

    The caller must already hold bounded detached-cleanup capacity.  A sync
    close still receives a diagnostic deadline, but this helper retains that
    existing lease until its daemon really exits instead of nesting another
    cleanup-task reservation.
    """

    async def finish_close() -> Any:
        budget = STREAM_CLEANUP_TIMEOUT_SECONDS if timeout is None else timeout
        close_owner = getattr(close, "__self__", None)
        if inspect.iscoroutinefunction(close) or inspect.isasyncgen(close_owner):
            result = close()
        else:
            result = await await_bounded_daemon_with_timeout(
                close,
                pool=bounded_daemon_module.STREAM_CLEANUP_DAEMON_POOL,
                name="owned-stream-close",
                timeout_seconds=budget,
                timeout_message="owned-stream-close timed out",
                retain_result_after_timeout=True,
            )
        if inspect.isawaitable(result):
            return await result
        return result

    close_task = asyncio.create_task(finish_close())
    try:
        return await asyncio.shield(close_task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if close_task.cancelled() or current is None or current.cancelling() == 0:
            raise
        while True:
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                if close_task.cancelled():
                    break
                continue
            except BaseException as exc:  # noqa: BLE001 - cancellation remains authoritative
                logger.debug(
                    "Cancelled owned stream close failed error_type={}",
                    type(exc).__name__,
                )
            break
        raise

_config = load_comprehensive_config()
# ConfigParser uses sections, check if Chat-Module section exists
_chat_config = {}
if _config and _config.has_section('Chat-Module'):
    _chat_config = dict(_config.items('Chat-Module'))


def _parse_int(value: Any, default: int, *, min_value: Optional[int] = None) -> int:
    try:
        if value is None:
            return default
        parsed = int(str(value).strip())
    except _STREAMING_NONCRITICAL_EXCEPTIONS:
        return default
    if min_value is not None and parsed < min_value:
        return min_value
    return parsed

# Timeout for idle connections (seconds)
STREAMING_IDLE_TIMEOUT = _parse_int(
    os.getenv('STREAMING_IDLE_TIMEOUT_SECONDS') or
    _chat_config.get('streaming_idle_timeout_seconds', 300),
    300,
    min_value=1,
)  # Default 5 minutes

# Heartbeat interval for long-running streams (seconds)
HEARTBEAT_INTERVAL = _parse_int(
    os.getenv('STREAMING_HEARTBEAT_INTERVAL_SECONDS') or
    _chat_config.get('streaming_heartbeat_interval_seconds', 30),
    30,
    min_value=0,
)

# Maximum response size in bytes (default 10MB) - configurable via env or config
MAX_RESPONSE_SIZE_BYTES = _parse_int(
    os.getenv('STREAMING_MAX_RESPONSE_SIZE_BYTES') or
    _chat_config.get('streaming_max_response_size_bytes', 10 * 1024 * 1024),
    10 * 1024 * 1024,
    min_value=1,
)

# Tool call accumulator max index to prevent memory exhaustion - configurable
MAX_TOOL_CALL_INDEX = _parse_int(
    os.getenv('STREAMING_MAX_TOOL_CALL_INDEX') or
    _chat_config.get('streaming_max_tool_call_index', 1000),
    1000,
    min_value=0,
)

# Maximum length for accumulated tool call arguments (in characters)
# This prevents OOM attacks from malicious streams with unbounded arguments
MAX_TOOL_ARGUMENT_LENGTH = _parse_int(
    os.getenv('STREAMING_MAX_TOOL_ARGUMENT_LENGTH') or
    _chat_config.get('streaming_max_tool_argument_length', 50_000),
    50_000,
    min_value=0,
)

# Maximum number of items in the full_response list to prevent unbounded growth
MAX_RESPONSE_LIST_LENGTH = _parse_int(
    os.getenv('STREAMING_MAX_RESPONSE_LIST_LENGTH') or
    _chat_config.get('streaming_max_response_list_length', 100_000),
    100_000,
    min_value=1,
)

# Offload sync iterators to a background thread to avoid blocking the event loop
try:
    STREAMING_SYNC_BRIDGE_ENABLED = is_truthy(str(
        os.getenv('STREAMING_SYNC_BRIDGE_ENABLED') or
        _chat_config.get('streaming_sync_bridge_enabled', 'true')
    ).lower())
except (ValueError, TypeError) as exc:
    logger.debug(f"Failed to parse STREAMING_SYNC_BRIDGE_ENABLED, using default: {exc}")
    STREAMING_SYNC_BRIDGE_ENABLED = True

try:
    STREAMING_SYNC_BRIDGE_MAX_QUEUE = int(
        os.getenv('STREAMING_SYNC_BRIDGE_MAX_QUEUE') or
        _chat_config.get('streaming_sync_bridge_max_queue', 32)
    )
except (ValueError, TypeError) as exc:
    logger.debug(f"Failed to parse STREAMING_SYNC_BRIDGE_MAX_QUEUE, using default: {exc}")
    STREAMING_SYNC_BRIDGE_MAX_QUEUE = 32
if STREAMING_SYNC_BRIDGE_MAX_QUEUE <= 0:
    STREAMING_SYNC_BRIDGE_MAX_QUEUE = 32

try:
    _include_meta_raw = (
        os.getenv("CHAT_STREAM_INCLUDE_METADATA")
        or _chat_config.get("chat_stream_include_metadata")
        or "true"
    )
    CHAT_STREAM_INCLUDE_METADATA = is_truthy(_include_meta_raw)
except (ValueError, TypeError) as exc:
    logger.debug(f"Failed to parse CHAT_STREAM_INCLUDE_METADATA, using default: {exc}")
    CHAT_STREAM_INCLUDE_METADATA = True

#######################################################################################################################
#
# Functions:

def _extract_text_from_upstream_sse(chunk_str: str) -> tuple[Optional[str], Optional[dict[str, Any]], bool]:
    """
    Normalize provider-emitted SSE frames to plain text content.

    Accepts a string that may be:
      - a raw text fragment (returns it as text_content)
      - an SSE line like "data: {...}" (extracts JSON and returns delta.content if present)
      - an SSE DONE line "data: [DONE]" (signals completion via is_done=True)

    Returns: (text_content, error_payload, is_done)
      - text_content: extracted textual delta (or original text) or None
      - error_payload: if upstream provided an error object, return it for direct emission
      - is_done: True if upstream indicated [DONE]
    """
    if not chunk_str:
        return None, None, False

    # Normalize common invisible prefixes (BOM, zero-width spaces) and trim whitespace
    s = chunk_str.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
    is_sse_framed = "\n\n" in chunk_str or "\r\n\r\n" in chunk_str or s.startswith((":", "event:", "data:"))

    # Ignore SSE control-only lines from upstream
    if (
        s.startswith(_SSE_CONTROL_PREFIXES) or (is_sse_framed and s.startswith(_SSE_FRAMED_CONTROL_PREFIXES))
    ) and "data:" not in s:
        return None, None, False

    # If any 'data:' line exists, try to parse; some providers send 'event:' + 'data:' pairs or multiple frames
    if s.startswith("data:") or ("\ndata:" in s or s.startswith("event:") or "data:" in s):
        saw_done = False
        first_error = None
        # Process by lines to handle possible multi-line chunks
        for line in s.splitlines():
            ls = line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
            if not ls:
                continue
            if ls.startswith(_SSE_CONTROL_PREFIXES) or ls.startswith(_SSE_FRAMED_CONTROL_PREFIXES):
                # Skip SSE control fields
                continue
            if not ls.startswith("data:"):
                continue
            payload_str = ls[len("data:") :].strip()
            if payload_str == "[DONE]":
                saw_done = True
                continue
            try:
                data = json.loads(payload_str)
            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                # Try next line if present
                continue

            if (
                isinstance(data, dict)
                and data.get("error") is not None
                and first_error is None
            ):
                first_error = provider_stream_error_payload(data)
                continue

            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0] or {}
                    delta = first.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        return str(content), None, False
                    # Fallback to message.content (non-stream case)
                    message = first.get("message") or {}
                    msg_content = message.get("content")
                    if msg_content:
                        return str(msg_content), None, False
        # If no content found but DONE or error encountered, reflect that
        if first_error is not None:
            return None, first_error, False
        if saw_done:
            return None, None, True
        return None, None, False

    # Not an SSE frame; treat as plain text chunk
    return chunk_str, None, False


async def _async_iter_sync_stream(
    stream: Iterator[Any],
    *,
    queue_maxsize: int = STREAMING_SYNC_BRIDGE_MAX_QUEUE,
    admission_event: threading.Event | None = None,
    cleanup_claimed: threading.Event | None = None,
) -> AsyncIterator[Any]:
    """Bridge a sync iterator onto the event loop without blocking it.

    Spawns a daemon thread to consume the sync iterator, passing chunks
    through an asyncio.Queue for non-blocking async consumption.

    Args:
        stream: A synchronous iterator to bridge.
        queue_maxsize: Maximum queue depth for backpressure (default: 32).

    Yields:
        Items from the sync iterator, now available asynchronously.

    Raises:
        Exception: Re-raises any exception that occurred in the sync iterator.
    """
    loop = asyncio.get_running_loop()
    maxsize = max(int(queue_maxsize or 0), 1)
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=maxsize)
    stop_event = threading.Event()
    worker_released = threading.Event()

    def _queue_put(item: tuple[str, Any]) -> None:
        if loop.is_closed():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        except (RuntimeError, asyncio.InvalidStateError) as exc:
            logger.debug("Failed to schedule sync stream enqueue error_type={}", type(exc).__name__)
            return
        while True:
            try:
                fut.result(timeout=1.0)
            except (concurrent.futures.TimeoutError, TimeoutError):
                if stop_event.is_set() or loop.is_closed():
                    try:
                        fut.cancel()
                    except (RuntimeError, asyncio.InvalidStateError) as cancel_err:
                        logger.debug(
                            "Failed to cancel sync stream enqueue error_type={}",
                            type(cancel_err).__name__,
                        )
                    return
            except (RuntimeError, concurrent.futures.CancelledError) as exc:
                logger.debug("Failed to enqueue sync stream chunk error_type={}", type(exc).__name__)
                return
            else:
                return

    def _safe_stream_failure(error: BaseException) -> SanitizedProviderStreamError:
        try:
            return sanitized_provider_stream_exception(error)
        except BaseException as sanitize_error:  # noqa: BLE001 - untrusted exception metadata
            logger.debug(
                "Sync stream failure normalization failed error_type={}",
                type(sanitize_error).__name__,
            )
            return sanitized_provider_stream_exception("provider_unavailable")

    def _consume_stream() -> None:
        failure: SanitizedProviderStreamError | None = None
        try:
            for chunk in stream:
                if stop_event.is_set():
                    break
                _queue_put(("data", chunk))
        except BaseException as exc:  # noqa: BLE001 - arbitrary adapters are untrusted
            failure = _safe_stream_failure(exc)

        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        except BaseException as exc:  # noqa: BLE001 - close is an adapter boundary
            if failure is None:
                failure = _safe_stream_failure(exc)
            logger.debug(
                "Exception while closing bridged sync stream stream_type={} error_type={}",
                type(stream).__name__,
                type(exc).__name__,
            )

        try:
            if failure is not None:
                _queue_put(("error", failure))
            _queue_put(("done", None))
        except BaseException as exc:  # noqa: BLE001 - daemon exceptions must be consumed
            logger.debug(
                "Sync stream bridge finalization failed error_type={}",
                type(exc).__name__,
            )

    def _worker() -> None:
        try:
            _consume_stream()
        except BaseException as exc:  # noqa: BLE001 - never leak daemon failures
            logger.debug(
                "Sync stream bridge worker failed error_type={}",
                type(exc).__name__,
            )
            with contextlib.suppress(BaseException):
                _queue_put(
                    (
                        "error",
                        sanitized_provider_stream_exception("provider_unavailable"),
                    )
                )
                _queue_put(("done", None))
    try:
        start_bounded_stream_daemon(
            _worker,
            name="sync-stream-bridge",
            released_event=worker_released,
        )
    except DaemonCapacityError:
        if cleanup_claimed is not None:
            cleanup_claimed.set()
        await _close_unstarted_sync_stream_bounded(stream)
        raise
    if admission_event is not None:
        admission_event.set()

    try:
        while True:
            kind, payload = await queue.get()
            if kind == "data":
                yield payload
            elif kind == "error":
                # The worker queues a terminal error only after provider
                # iteration and close have finished. Preserve that ownership
                # boundary through the pool's lease-release epilogue so an
                # immediate retry cannot fail admission on the prior attempt.
                while not worker_released.is_set():
                    await asyncio.sleep(0)
                raise payload
            elif kind == "done":
                break
    finally:
        # Never join or close synchronously on the event-loop thread. A provider
        # may block forever in next() or close(); the daemon worker owns cleanup
        # once it observes this signal and regains control.
        stop_event.set()
        deadline = time.monotonic() + STREAM_CLEANUP_TIMEOUT_SECONDS
        while not worker_released.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0)


async def _close_unstarted_sync_stream_bounded(stream: Any) -> None:
    """Close an unstarted sync stream off-loop when bounded capacity is available."""

    close_fn = getattr(stream, "close", None)
    if not callable(close_fn):
        return
    deadline = time.monotonic() + STREAM_CLEANUP_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Unstarted sync stream close skipped: daemon capacity remained exhausted"
            )
            return
        try:
            await invoke_stream_close_bounded(close_fn, remaining)
            return
        except DaemonCapacityError:
            await asyncio.sleep(0)
            continue
        except asyncio.TimeoutError:
            logger.debug("Unstarted sync stream close exceeded bounded timeout")
            return
        except _STREAMING_NONCRITICAL_EXCEPTIONS as close_error:
            logger.debug(
                "Unstarted sync stream close failed stream_type={} error_type={}",
                type(stream).__name__,
                type(close_error).__name__,
            )
            return

class StreamingResponseHandler:
    """
    Handles streaming responses with proper error handling, cleanup, and timeouts.

    This class is designed to be thread-safe for concurrent state access through
    the use of an asyncio.Lock for state modifications.
    """

    def __init__(
        self,
        conversation_id: str,
        model_name: str,
        idle_timeout: int = STREAMING_IDLE_TIMEOUT,
        heartbeat_interval: int = HEARTBEAT_INTERVAL,
        max_response_size: int = MAX_RESPONSE_SIZE_BYTES,
        text_transform: Optional[callable] = None,
    ):
        """
        Initialize the streaming response handler.

        Args:
            conversation_id: ID of the conversation
            model_name: Name of the model being used
            idle_timeout: Timeout for idle connections in seconds
            heartbeat_interval: Interval for sending heartbeat messages
            max_response_size: Maximum response size in bytes
        """
        self.conversation_id = conversation_id
        self.model_name = model_name
        self.idle_timeout = idle_timeout
        self.heartbeat_interval = heartbeat_interval
        self.max_response_size = max_response_size
        self.last_activity = time.time()
        self.is_cancelled = False
        self.full_response: list[str] = []
        self.response_size = 0
        self.error_occurred = False
        # Optional transform to apply to textual deltas before emission (e.g., moderation redaction)
        self.text_transform = text_transform
        # Track whether a terminal [DONE] was already sent (directly or via transform-combined payload)
        self.done_sent = False
        # Track upstream DONE so we can defer the terminal sentinel until after metadata.
        self.upstream_done_received = False
        # OpenAI-compatible SSE is successful only after semantic output and a
        # valid finish reason. These fields are request-local by construction.
        self.openai_sse_seen = False
        self.semantic_output_seen = False
        self.valid_finish_received = False
        # Accumulate tool/function call deltas for persistence once the stream completes
        self.tool_call_accumulator: dict[int, dict[str, Any]] = {}
        self.tool_call_order: list[int] = []
        self.function_call_accumulator: Optional[dict[str, Any]] = None
        self.saved_message_id: Optional[str] = None
        self.system_message_id: Optional[str] = None
        self.continuation_metadata: Optional[dict[str, Any]] = None
        # Lock for thread-safe state modifications
        self._state_lock = asyncio.Lock()

    def _attach_stream_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        if not CHAT_STREAM_INCLUDE_METADATA:
            return payload
        if self.conversation_id:
            payload.setdefault("conversation_id", self.conversation_id)
            payload.setdefault("tldw_conversation_id", self.conversation_id)
        if self.system_message_id:
            payload.setdefault("tldw_system_message_id", self.system_message_id)
        if self.saved_message_id:
            payload.setdefault("tldw_message_id", self.saved_message_id)
        if self.continuation_metadata:
            payload.setdefault("tldw_continuation", self.continuation_metadata)
        return payload

    def _parse_save_callback_result(self, save_result: Any) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Normalize save-callback return payload.

        Supported return values:
        - `"message_id"` string (legacy)
        - `{"saved_message_id": "...", "events": [{"event": "...", "data": {...}}]}`
        """
        saved_message_id: Optional[str] = None
        extra_events: list[dict[str, Any]] = []

        if isinstance(save_result, str):
            normalized = save_result.strip()
            if normalized:
                saved_message_id = normalized
            return saved_message_id, extra_events

        if not isinstance(save_result, dict):
            return saved_message_id, extra_events

        raw_id = save_result.get("saved_message_id")
        if raw_id is None:
            raw_id = save_result.get("message_id")
        if isinstance(raw_id, str):
            normalized = raw_id.strip()
            if normalized:
                saved_message_id = normalized

        raw_events = save_result.get("events")
        if raw_events is None:
            raw_events = save_result.get("extra_events")
        if raw_events is None:
            raw_events = save_result.get("loop_events")
        if not isinstance(raw_events, list):
            return saved_message_id, extra_events

        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            raw_name = raw_event.get("event")
            event_name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not event_name:
                continue
            if "data" not in raw_event:
                continue
            extra_events.append({"event": event_name, "data": raw_event.get("data")})

        return saved_message_id, extra_events

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = time.time()

    def is_timed_out(self) -> bool:
        """Check if the stream has timed out due to inactivity."""
        return (time.time() - self.last_activity) > self.idle_timeout

    def cancel(self):
        """Mark the stream as cancelled."""
        self.is_cancelled = True
        logger.info(f"Stream cancelled for conversation {self.conversation_id}")

    def _accumulate_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Merge incremental tool call deltas into a final structure.

        This method includes bounds checking to prevent memory exhaustion from
        malformed tool call indices.
        """
        if not isinstance(tool_calls, list):
            return
        for idx, entry in enumerate(tool_calls):
            if not isinstance(entry, dict):
                continue
            call_index = entry.get("index")
            if call_index is None:
                call_index = idx
            try:
                call_index = int(call_index)
            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                call_index = idx

            # Bounds check to prevent memory exhaustion
            if call_index < 0 or call_index > MAX_TOOL_CALL_INDEX:
                logger.warning(
                    f"Tool call index {call_index} out of bounds (0-{MAX_TOOL_CALL_INDEX}), skipping"
                )
                continue

            if call_index not in self.tool_call_accumulator:
                self.tool_call_accumulator[call_index] = {
                    "id": None,
                    "type": None,
                    "function": {"name": None, "arguments": ""},
                }
                self.tool_call_order.append(call_index)
            accumulator = self.tool_call_accumulator[call_index]
            if entry.get("id"):
                accumulator["id"] = entry["id"]
            if entry.get("type"):
                accumulator["type"] = entry["type"]
            function_delta = entry.get("function") or {}
            if function_delta.get("name"):
                accumulator["function"]["name"] = function_delta["name"]
            if function_delta.get("arguments"):
                new_args = function_delta["arguments"]
                current_len = len(accumulator["function"]["arguments"])
                # Enforce bounds on accumulated argument length to prevent OOM
                if current_len + len(new_args) > MAX_TOOL_ARGUMENT_LENGTH:
                    logger.warning(
                        f"Tool call arguments exceeded max length ({MAX_TOOL_ARGUMENT_LENGTH}), truncating"
                    )
                    # Truncate to fit within bounds
                    remaining = MAX_TOOL_ARGUMENT_LENGTH - current_len
                    if remaining > 0:
                        accumulator["function"]["arguments"] += new_args[:remaining]
                    # Skip further argument accumulation for this tool call
                else:
                    accumulator["function"]["arguments"] += new_args

    def _accumulate_function_call(self, function_delta: dict[str, Any]) -> None:
        """Merge incremental function call deltas into a final structure."""
        if not isinstance(function_delta, dict):
            return
        if self.function_call_accumulator is None:
            self.function_call_accumulator = {"name": None, "arguments": ""}
        if function_delta.get("name"):
            self.function_call_accumulator["name"] = function_delta["name"]
        if function_delta.get("arguments"):
            new_args = function_delta["arguments"]
            current_len = len(self.function_call_accumulator["arguments"])
            # Enforce bounds on accumulated argument length to prevent OOM
            if current_len + len(new_args) > MAX_TOOL_ARGUMENT_LENGTH:
                logger.warning(
                    f"Function call arguments exceeded max length ({MAX_TOOL_ARGUMENT_LENGTH}), truncating"
                )
                remaining = MAX_TOOL_ARGUMENT_LENGTH - current_len
                if remaining > 0:
                    self.function_call_accumulator["arguments"] += new_args[:remaining]
            else:
                self.function_call_accumulator["arguments"] += new_args

    def get_accumulated_tool_calls(self) -> Optional[list[dict[str, Any]]]:
        """Return the finalized list of tool calls, if any were streamed."""
        if not self.tool_call_accumulator:
            return None
        ordered_indices = sorted(set(self.tool_call_order))
        results: list[dict[str, Any]] = []
        for index in ordered_indices:
            data = self.tool_call_accumulator.get(index)
            if not data:
                continue
            function_block = data.get("function") or {}
            results.append(
                {
                    "id": data.get("id"),
                    "type": data.get("type"),
                    "function": {
                        "name": function_block.get("name"),
                        "arguments": function_block.get("arguments", ""),
                    },
                }
            )
        return results or None

    def get_accumulated_function_call(self) -> Optional[dict[str, Any]]:
        """Return the finalized function call payload, if one was streamed."""
        if not self.function_call_accumulator:
            return None
        name = self.function_call_accumulator.get("name")
        arguments = self.function_call_accumulator.get("arguments", "")
        if not name and not arguments:
            return None
        return {"name": name, "arguments": arguments}

    def has_accumulated_output(self) -> bool:
        """Return True when any provider-owned semantic output was gathered."""
        return bool(
            self.semantic_output_seen
            or self.full_response
            or self.tool_call_accumulator
            or self.function_call_accumulator
        )

    async def heartbeat_generator(self) -> AsyncIterator[str]:
        """
        Generate heartbeat messages to keep the connection alive.

        Yields:
            SSE heartbeat messages
        """
        while not self.is_cancelled and not self.error_occurred:
            await asyncio.sleep(self.heartbeat_interval)
            if self.is_timed_out():
                logger.warning(f"Stream timeout for conversation {self.conversation_id}")
                self.cancel()
                payload = provider_stream_error_payload("provider_unavailable")
                self._attach_stream_metadata(payload)
                yield f"data: {json.dumps(payload)}\n\n"
                break
            yield f": heartbeat {datetime.now(timezone.utc).isoformat()}\n\n"

    async def safe_stream_generator(
        self,
        stream: Union[Iterator, AsyncIterator],
        save_callback: Optional[callable] = None,
        finalize_callback: Optional[callable] = None,
        before_success_callback: Optional[Callable[[], Any]] = None,
        on_first_output: Optional[Callable[[], Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Safely generate streaming responses with error handling and cleanup.

        Args:
            stream: The stream to process (sync or async iterator)
            save_callback: Optional callback to save the full response
            finalize_callback: Optional callback invoked on error/cancel to finalize state
            on_first_output: Optional callback invoked once after the first valid provider output

        Yields:
            SSE formatted messages
        """
        bridged_stream: AsyncIterator[Any] | None = None
        sync_bridge_started = False
        sync_bridge_admitted = threading.Event()
        sync_bridge_cleanup_claimed = threading.Event()
        try:
            # Send initial metadata
            start_payload = {
                "conversation_id": self.conversation_id,
                "model": self.model_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._attach_stream_metadata(start_payload)
            yield f"event: stream_start\ndata: {json.dumps(start_payload)}\n\n"
            self.update_activity()

            def iter_logical_lines(raw_chunk: str) -> list[str]:
                return raw_chunk.splitlines() if ("\n" in raw_chunk or raw_chunk.count("data:") > 1) else [raw_chunk]

            def is_sse_framed(raw_chunk: str) -> bool:
                stripped_chunk = raw_chunk.lstrip("\ufeff\u200b\u200c\u200d\u2060")
                return (
                    "\n\n" in raw_chunk
                    or "\r\n\r\n" in raw_chunk
                    or stripped_chunk.startswith((":", "event:", "data:"))
                )

            def append_content(text_piece: str) -> bool:
                if not text_piece:
                    return True
                chunk_size = len(text_piece.encode("utf-8"))
                if self.response_size + chunk_size > self.max_response_size:
                    return False
                # Also check list length to prevent unbounded item count
                if len(self.full_response) >= MAX_RESPONSE_LIST_LENGTH:
                    logger.warning(
                        f"Response list length exceeded max ({MAX_RESPONSE_LIST_LENGTH}) for {self.conversation_id}"
                    )
                    return False
                self.full_response.append(text_piece)
                self.response_size += chunk_size
                return True

            def canonical_provider_error(value: Any) -> str:
                err_payload = provider_stream_error_payload(value)
                self._attach_stream_metadata(err_payload)
                self.error_occurred = True
                return f"data: {json.dumps(err_payload)}\n\n"

            def structural_error_code(value: Any) -> str | None:
                nested_code = provider_payload_structural_error_code(value)
                if nested_code is not None:
                    return nested_code
                if not isinstance(value, (str, bytes)):
                    normalized = normalize_provider_stream_error(value)
                    return normalized.code if normalized is not None else None
                raw_value = (
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else value
                )
                for raw_line in raw_value.splitlines():
                    candidate = raw_line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
                    if not candidate.startswith("data:"):
                        continue
                    try:
                        data = json.loads(candidate[len("data:") :].strip())
                    except _STREAMING_NONCRITICAL_EXCEPTIONS:
                        continue
                    nested_code = provider_payload_structural_error_code(data)
                    if nested_code is not None:
                        return nested_code
                return None

            first_output_pending = False
            first_output_notified = False

            async def notify_first_output() -> None:
                nonlocal first_output_pending, first_output_notified
                if (
                    not first_output_pending
                    or not self.semantic_output_seen
                    or first_output_notified
                ):
                    return
                first_output_pending = False
                first_output_notified = True
                if not callable(on_first_output):
                    return
                try:
                    maybe_result = on_first_output()
                    if hasattr(maybe_result, "__await__"):
                        await maybe_result
                except _STREAMING_NONCRITICAL_EXCEPTIONS as callback_error:
                    logger.debug(
                        "First-output callback failed for {} error_type={}",
                        self.conversation_id,
                        type(callback_error).__name__,
                    )

            def process_line(raw_line: str, *, sse_framed: bool) -> tuple[list[str], bool]:
                nonlocal first_output_pending
                outputs: list[str] = []
                stripped_leading = raw_line.lstrip("\ufeff\u200b\u200c\u200d\u2060")
                candidate = stripped_leading.strip()
                if not candidate and not stripped_leading:
                    return outputs, False
                if candidate.startswith(_SSE_CONTROL_PREFIXES) or (
                    sse_framed and candidate.startswith(_SSE_FRAMED_CONTROL_PREFIXES)
                ):
                    return outputs, False
                if candidate.startswith("data:"):
                    payload_str = candidate[len("data:") :].strip()
                    if payload_str == "[DONE]":
                        # Defer terminal DONE until after stream_end metadata is emitted.
                        self.upstream_done_received = True
                        self.update_activity()
                        return outputs, True
                    try:
                        data = json.loads(payload_str)
                    except _STREAMING_NONCRITICAL_EXCEPTIONS:
                        if payload_str:
                            self.semantic_output_seen = True
                            first_output_pending = True
                        outputs.append(f"data: {payload_str}\n\n")
                        self.update_activity()
                        return outputs, False
                    error_code = structural_error_code(data)
                    if error_code is not None:
                        outputs.append(canonical_provider_error(error_code))
                        return outputs, True
                    if isinstance(data, dict):
                        choices = data.get("choices")
                        if isinstance(choices, list) and choices:
                            self.openai_sse_seen = True
                            for choice in choices:
                                if not isinstance(choice, dict):
                                    outputs.append(
                                        canonical_provider_error("provider_unavailable")
                                    )
                                    return outputs, True
                                finish_reason = choice.get("finish_reason")
                                if finish_reason is not None:
                                    if (
                                        not isinstance(finish_reason, str)
                                        or finish_reason not in _OPENAI_STREAM_FINISH_REASONS
                                    ):
                                        outputs.append(
                                            canonical_provider_error("provider_unavailable")
                                        )
                                        return outputs, True
                                    self.valid_finish_received = True
                                    if finish_reason == "content_filter":
                                        self.semantic_output_seen = True
                                        first_output_pending = True
                                delta = choice.get("delta")
                                # Be tolerant: providers/tests may send a plain string delta
                                if isinstance(delta, str):
                                    delta = {"content": delta}
                                # Guard against unexpected delta types
                                if not isinstance(delta, dict):
                                    delta = {}

                                tool_calls_delta = delta.get("tool_calls")
                                if tool_calls_delta:
                                    self.semantic_output_seen = True
                                    first_output_pending = True
                                    self._accumulate_tool_calls(tool_calls_delta)
                                function_call_delta = delta.get("function_call")
                                if function_call_delta:
                                    self.semantic_output_seen = True
                                    first_output_pending = True
                                    self._accumulate_function_call(function_call_delta)
                                if any(
                                    delta.get(field) not in (None, "", [], {})
                                    for field in (
                                        "reasoning_content",
                                        "reasoning",
                                        "reasoning_details",
                                        "thinking",
                                        "analysis",
                                    )
                                ):
                                    self.semantic_output_seen = True
                                    first_output_pending = True
                                refusal = delta.get("refusal")
                                if isinstance(refusal, str) and refusal:
                                    self.semantic_output_seen = True
                                    first_output_pending = True
                                if "content" in delta and delta["content"] is not None:
                                    text_piece = str(delta["content"])
                                    if text_piece:
                                        self.semantic_output_seen = True
                                        first_output_pending = True
                                    try:
                                        if self.text_transform:
                                            text_piece = self.text_transform(text_piece)
                                    except StopStreamWithError as stopper:
                                        err_payload = {
                                            "error": {
                                                "message": str(stopper) or "Stream blocked by policy",
                                                "type": stopper.error_type,
                                            }
                                        }
                                        self._attach_stream_metadata(err_payload)
                                        outputs.append(
                                            _trusted_local_stream_frame(
                                                f"data: {json.dumps(err_payload)}\n\n"
                                            )
                                        )
                                        self.error_occurred = True
                                        return outputs, True
                                    except StopIteration:
                                        return outputs, True
                                    except _STREAMING_NONCRITICAL_EXCEPTIONS as transform_err:
                                        logger.error(
                                            "text_transform failed for {} error_type={}",
                                            self.conversation_id,
                                            type(transform_err).__name__,
                                        )
                                        err_payload = {
                                            "error": {
                                                "message": "Stream transform failed",
                                                "type": "stream_transform_error",
                                            }
                                        }
                                        self._attach_stream_metadata(err_payload)
                                        outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                                        self.error_occurred = True
                                        return outputs, True
                                    if text_piece and not append_content(text_piece):
                                        err_payload = {"error": {"message": "Response size limit exceeded"}}
                                        self._attach_stream_metadata(err_payload)
                                        outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                                        self.error_occurred = True
                                        return outputs, True
                                    delta["content"] = text_piece
                            self._attach_stream_metadata(data)
                            outputs.append(f"data: {json.dumps(data)}\n\n")
                            self.update_activity()
                            return outputs, False
                    if isinstance(data, dict):
                        self._attach_stream_metadata(data)
                    elif data not in (None, "", [], {}):
                        self.semantic_output_seen = True
                        first_output_pending = True
                    outputs.append(f"data: {json.dumps(data)}\n\n")
                    self.update_activity()
                    return outputs, False
                if not sse_framed:
                    normalized_error = _normalize_unframed_provider_stream_error(raw_line)
                    if normalized_error is not None:
                        outputs.append(canonical_provider_error(normalized_error.code))
                        return outputs, True
                # Non-SSE chunk: preserve spaces (avoid stripping)
                text_piece = stripped_leading
                with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                    text_piece = str(text_piece)
                if text_piece:
                    self.semantic_output_seen = True
                    first_output_pending = True
                try:
                    if self.text_transform:
                        text_piece = self.text_transform(text_piece)
                except StopStreamWithError as stopper:
                    err_payload = {
                        "error": {
                            "message": str(stopper) or "Stream blocked by policy",
                            "type": stopper.error_type,
                        }
                    }
                    self._attach_stream_metadata(err_payload)
                    outputs.append(
                        _trusted_local_stream_frame(
                            f"data: {json.dumps(err_payload)}\n\n"
                        )
                    )
                    self.error_occurred = True
                    return outputs, True
                except StopIteration:
                    return outputs, True
                except _STREAMING_NONCRITICAL_EXCEPTIONS as transform_err:
                    logger.error(
                        "text_transform failed for {} error_type={}",
                        self.conversation_id,
                        type(transform_err).__name__,
                    )
                    err_payload = {
                        "error": {
                            "message": "Stream transform failed",
                            "type": "stream_transform_error",
                        }
                    }
                    self._attach_stream_metadata(err_payload)
                    outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                    self.error_occurred = True
                    return outputs, True
                if text_piece and not append_content(text_piece):
                    err_payload = {"error": {"message": "Response size limit exceeded"}}
                    self._attach_stream_metadata(err_payload)
                    outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                    self.error_occurred = True
                    return outputs, True
                if text_piece:
                    content_payload = {"choices": [{"delta": {"content": text_piece}}]}
                    self._attach_stream_metadata(content_payload)
                    outputs.append(f"data: {json.dumps(content_payload)}\n\n")
                    self.update_activity()
                return outputs, False

            # Process the stream
            # Hard timeouts require every synchronous provider iterator to stay
            # off the event loop. The legacy flag is retained for configuration
            # compatibility but can no longer disable this safety invariant.
            if hasattr(stream, "__aiter__"):
                async_stream = stream
            else:
                bridged_stream = _async_iter_sync_stream(
                    stream,
                    admission_event=sync_bridge_admitted,
                    cleanup_claimed=sync_bridge_cleanup_claimed,
                )
                async_stream = bridged_stream
                sync_bridge_started = True

            if async_stream is not None:
                # Async iterator (native or bridged from sync)
                async for chunk in async_stream:
                    if self.is_cancelled:
                        logger.info(f"Stream processing cancelled for {self.conversation_id}")
                        break

                    if self.is_timed_out():
                        logger.warning(f"Stream timeout during processing for {self.conversation_id}")
                        yield canonical_provider_error("provider_unavailable")
                        break

                    if is_trusted_local_stream_frame(chunk):
                        self.error_occurred = True
                        yield chunk
                        break

                    if self.upstream_done_received:
                        error_code = structural_error_code(chunk)
                        if error_code is not None:
                            yield canonical_provider_error(error_code)
                            break
                        continue

                    try:
                        if not isinstance(chunk, (str, bytes)):
                            error_code = structural_error_code(chunk)
                            if error_code is not None:
                                yield canonical_provider_error(error_code)
                                break
                        if isinstance(chunk, bytes):
                            raw_str = chunk.decode("utf-8", errors="replace")
                        elif isinstance(chunk, str):
                            raw_str = chunk
                        else:
                            raw_str = json.dumps(chunk, default=str)
                        stop_stream = False
                        chunk_outputs: list[str] = []
                        sse_framed = is_sse_framed(raw_str)
                        for logical_line in iter_logical_lines(raw_str):
                            if self.upstream_done_received:
                                error_code = structural_error_code(logical_line)
                                if error_code is not None:
                                    chunk_outputs.append(
                                        canonical_provider_error(error_code)
                                    )
                                    stop_stream = True
                                    break
                                continue
                            outputs, should_stop = process_line(logical_line, sse_framed=sse_framed)
                            chunk_outputs.extend(outputs)
                            # Provider-use accounting belongs to the first
                            # validated semantic frame, not clean stream
                            # completion.  Keep the callback ahead of a later
                            # frame-level failure while rejecting lines that
                            # failed validation or output transformation.
                            if first_output_pending and not self.error_occurred:
                                await notify_first_output()
                            if should_stop:
                                if self.upstream_done_received and not self.error_occurred:
                                    continue
                                stop_stream = True
                                break
                        for out in chunk_outputs:
                            yield out
                        if stop_stream:
                            if self.upstream_done_received and not self.error_occurred:
                                continue
                            break
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
                        normalized_error = normalize_provider_stream_error(e)
                        logger.error(
                            "Provider stream chunk processing failed for {} code={} error_type={}",
                            self.conversation_id,
                            normalized_error.code if normalized_error else "provider_unavailable",
                            type(e).__name__,
                        )
                        yield canonical_provider_error(e)
                        break
                else:
                    if (
                        not self.error_occurred
                        and (
                            (
                                self.openai_sse_seen
                                and (
                                    not self.semantic_output_seen
                                    or not (
                                        self.valid_finish_received
                                        or self.upstream_done_received
                                    )
                                )
                            )
                            or (
                                self.upstream_done_received
                                and not self.semantic_output_seen
                            )
                        )
                    ):
                        yield canonical_provider_error("provider_unavailable")
        except asyncio.CancelledError:
            # Client disconnected
            logger.info(f"Client disconnected from stream for {self.conversation_id}")
            self.cancel()
            raise
        except GeneratorExit:
            # Generator is being closed; do not yield anything here
            logger.info(f"Stream generator closed for {self.conversation_id}")
            self.cancel()
            # Re-raise to ensure proper generator closure semantics
            raise
        except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
            # Unexpected error
            normalized_error = normalize_provider_stream_error(e)
            logger.error(
                "Provider stream failed for {} code={} error_type={}",
                self.conversation_id,
                normalized_error.code if normalized_error else "provider_unavailable",
                type(e).__name__,
            )
            # Best-effort: flush any buffered tail before emitting the error frame.
            # This preserves earlier valid chunks when the upstream fails mid-stream.
            if self.text_transform:
                flush_fn = getattr(self.text_transform, "flush", None)
                if callable(flush_fn):
                    try:
                        flush_text = flush_fn()
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as flush_err:
                        logger.debug(
                            "text_transform flush on error ignored error_type={}",
                            type(flush_err).__name__,
                        )
                        flush_text = None
                    if flush_text:
                        try:
                            flush_text = str(flush_text)
                        except _STREAMING_NONCRITICAL_EXCEPTIONS:
                            flush_text = ""
                        if flush_text:
                            if append_content(flush_text):
                                content_payload = {"choices": [{"delta": {"content": flush_text}}]}
                                self._attach_stream_metadata(content_payload)
                                yield f"data: {json.dumps(content_payload)}\n\n"
                                self.update_activity()
                            else:
                                size_err = {"error": {"message": "Response size limit exceeded"}}
                                self._attach_stream_metadata(size_err)
                                self.error_occurred = True
                                yield f"data: {json.dumps(size_err)}\n\n"
                                return
            yield canonical_provider_error(e)

        finally:
            # Cleanup and final message
            try:
                # Async cleanup is safe on the loop. Raw synchronous streams are
                # closed by the daemon bridge worker; never call close() here.
                try:
                    cleanup_stream = bridged_stream if sync_bridge_started else stream
                    if hasattr(cleanup_stream, "aclose") and callable(cleanup_stream.aclose):
                        await invoke_stream_close_bounded(  # type: ignore[attr-defined]
                            cleanup_stream.aclose,
                        )
                    if (
                        not hasattr(stream, "__aiter__")
                        and not sync_bridge_admitted.is_set()
                        and not sync_bridge_cleanup_claimed.is_set()
                    ):
                        sync_bridge_cleanup_claimed.set()
                        await _close_unstarted_sync_stream_bounded(stream)
                except asyncio.TimeoutError:
                    logger.debug(
                        "Stream cleanup exceeded bounded timeout for {}",
                        self.conversation_id,
                    )
                except _STREAMING_NONCRITICAL_EXCEPTIONS as cleanup_err:
                    # Log cleanup errors for debugging, but don't propagate
                    logger.debug(
                        "Stream cleanup warning for {} error_type={}",
                        self.conversation_id,
                        type(cleanup_err).__name__,
                    )

                # If cancelled (e.g., client disconnect or generator close), do not yield or await further
                if self.is_cancelled:
                    if finalize_callback and (self.is_cancelled or self.error_occurred):
                        try:
                            maybe_result = finalize_callback(
                                success=False,
                                cancelled=True,
                                error=self.error_occurred,
                            )
                            if hasattr(maybe_result, "__await__"):
                                await maybe_result
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as finalize_err:
                            logger.debug(
                                "Finalize callback error after cancel error_type={}",
                                type(finalize_err).__name__,
                            )

                # Flush any pending tail from text_transform (e.g., moderation holdback)
                if not self.is_cancelled and not self.error_occurred and self.text_transform:
                    flush_fn = getattr(self.text_transform, "flush", None)
                    if callable(flush_fn):
                        try:
                            flush_text = flush_fn()
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as flush_err:
                            logger.debug(
                                "text_transform flush error ignored error_type={}",
                                type(flush_err).__name__,
                            )
                            flush_text = None
                        if flush_text:
                            try:
                                flush_text = str(flush_text)
                            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                                flush_text = ""
                            if flush_text:
                                if not append_content(flush_text):
                                    err_payload = {"error": {"message": "Response size limit exceeded"}}
                                    self._attach_stream_metadata(err_payload)
                                    yield f"data: {json.dumps(err_payload)}\n\n"
                                    self.error_occurred = True
                                else:
                                    content_payload = {"choices": [{"delta": {"content": flush_text}}]}
                                    self._attach_stream_metadata(content_payload)
                                    yield f"data: {json.dumps(content_payload)}\n\n"
                                    self.update_activity()

                # Save the full response/tool calls if callback provided (only when not cancelled)
                has_output = self.has_accumulated_output()
                if (
                    not self.is_cancelled
                    and not self.error_occurred
                ):
                    if before_success_callback and not self.is_cancelled:
                        try:
                            maybe_before_success = before_success_callback()
                            if hasattr(maybe_before_success, "__await__"):
                                await maybe_before_success
                        except StopStreamWithError as stopper:
                            err_payload = {
                                "error": {
                                    "message": str(stopper) or "Stream blocked by policy",
                                    "type": stopper.error_type,
                                }
                            }
                            self._attach_stream_metadata(err_payload)
                            self.error_occurred = True
                            yield _trusted_local_stream_frame(
                                f"data: {json.dumps(err_payload)}\n\n"
                            )
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as before_success_err:
                            logger.error(
                                "Before-success callback error for {} error_type={}",
                                self.conversation_id,
                                type(before_success_err).__name__,
                            )
                            self.error_occurred = True
                            err_payload = provider_stream_error_payload("provider_unavailable")
                            self._attach_stream_metadata(err_payload)
                            yield f"data: {json.dumps(err_payload)}\n\n"

                if (
                    not self.is_cancelled
                    and save_callback
                    and not self.error_occurred
                    and has_output
                ):
                    full_text = "".join(self.full_response)
                    aggregated_tool_calls = self.get_accumulated_tool_calls()
                    aggregated_function_call = self.get_accumulated_function_call()
                    extra_events: list[dict[str, Any]] = []
                    try:
                        # Support flexible callback signatures (text only or extended)
                        maybe_result = None
                        save_result = None
                        try:
                            maybe_result = save_callback(
                                full_text,
                                aggregated_tool_calls,
                                aggregated_function_call,
                            )
                        except TypeError:
                            maybe_result = save_callback(full_text)
                        if hasattr(maybe_result, "__await__"):
                            save_result = await maybe_result
                        else:
                            save_result = maybe_result
                        parsed_message_id, parsed_events = self._parse_save_callback_result(save_result)
                        if parsed_message_id:
                            self.saved_message_id = parsed_message_id
                        extra_events = parsed_events
                        logger.info(
                            'Saved streaming response for {} (text_len={}, tool_calls={}, function_call={}, events={})',
                            self.conversation_id,
                            len(full_text),
                            len(aggregated_tool_calls or []),
                            "yes" if aggregated_function_call else "no",
                            len(extra_events),
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to save streaming response for {} error_type={}",
                            self.conversation_id,
                            type(e).__name__,
                        )
                        extra_events = []

                    for event_entry in extra_events:
                        event_name = str(event_entry.get("event") or "").strip()
                        if not event_name:
                            continue
                        payload_obj = event_entry.get("data")
                        if payload_obj is None:
                            continue
                        try:
                            if isinstance(payload_obj, dict):
                                payload = dict(payload_obj)
                                self._attach_stream_metadata(payload)
                                payload_json = json.dumps(payload)
                            else:
                                payload_json = json.dumps(payload_obj, default=str)
                            yield f"event: {event_name}\ndata: {payload_json}\n\n"
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as event_err:
                            logger.debug(
                                "Skipping extra stream event {} for {} error_type={}",
                                event_name,
                                self.conversation_id,
                                type(event_err).__name__,
                            )

                # Send completion marker(s) after save so metadata includes IDs.
                if not self.is_cancelled and not self.error_occurred:
                    done_payload = {
                        "id": f"chatcmpl-{datetime.now(timezone.utc).timestamp()}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "model": self.model_name,
                        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                    }
                    self._attach_stream_metadata(done_payload)
                    yield f"data: {json.dumps(done_payload)}\n\n"

                if not self.is_cancelled and finalize_callback and self.error_occurred:
                    try:
                        maybe_result = finalize_callback(
                            success=False,
                            cancelled=False,
                            error=True,
                        )
                        if hasattr(maybe_result, "__await__"):
                            await maybe_result
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as finalize_err:
                        logger.debug(
                            "Finalize callback error after stream error error_type={}",
                            type(finalize_err).__name__,
                        )

                # Send stream end event (only when not cancelled)
                if not self.is_cancelled:
                    end_payload = {
                        "conversation_id": self.conversation_id,
                        "success": not self.error_occurred,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self._attach_stream_metadata(end_payload)
                    yield f"event: stream_end\ndata: {json.dumps(end_payload)}\n\n"
                # Ensure final [DONE] sentinel for client compatibility (unless already sent).
                # If upstream already sent [DONE], defer emission until after stream_end.
                if not self.is_cancelled:
                    if (self.upstream_done_received and not self.done_sent) or not self.done_sent:
                        yield "data: [DONE]\n\n"
                        self.done_sent = True
                    self.upstream_done_received = False

            except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
                logger.error(
                    "Error in stream cleanup for {} error_type={}",
                    self.conversation_id,
                    type(e).__name__,
                )


async def create_streaming_response_with_timeout(
    stream: Union[Iterator, AsyncIterator],
    conversation_id: str,
    model_name: str,
    save_callback: Optional[callable] = None,
    finalize_callback: Optional[callable] = None,
    before_success_callback: Optional[Callable[[], Any]] = None,
    on_first_output: Optional[Callable[[], Any]] = None,
    idle_timeout: int = STREAMING_IDLE_TIMEOUT,
    heartbeat_interval: int = HEARTBEAT_INTERVAL,
    text_transform: Optional[callable] = None,
    system_message_id: Optional[str] = None,
    continuation_metadata: Optional[dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """
    Create a streaming response with timeout and error handling.

    Args:
        stream: The stream to process
        conversation_id: ID of the conversation
        model_name: Name of the model
        save_callback: Optional callback to save the response
        finalize_callback: Optional callback invoked on error/cancel to finalize state
        on_first_output: Optional callback invoked once after the first valid provider output
        idle_timeout: Timeout for idle connections
        heartbeat_interval: Interval for heartbeat messages
        system_message_id: Optional system message ID to echo in stream_end payload
        continuation_metadata: Optional continuation metadata to attach to stream payloads

    Yields:
        SSE formatted messages
    """
    handler = StreamingResponseHandler(
        conversation_id=conversation_id,
        model_name=model_name,
        idle_timeout=idle_timeout,
        heartbeat_interval=heartbeat_interval,
        text_transform=text_transform,
    )
    handler.system_message_id = system_message_id
    if isinstance(continuation_metadata, dict) and continuation_metadata:
        handler.continuation_metadata = dict(continuation_metadata)

    # Create tasks for streaming and optional heartbeat using persistent generator instances
    async def stream_with_heartbeat():
        finalize_invoked = False

        async def finalize_once(
            *,
            success: bool,
            cancelled: bool,
            error: bool,
        ) -> None:
            nonlocal finalize_invoked
            if finalize_invoked or not callable(finalize_callback):
                return
            finalize_invoked = True
            maybe_result = finalize_callback(
                success=success,
                cancelled=cancelled,
                error=error,
            )
            if inspect.isawaitable(maybe_result):
                await maybe_result

        guarded_finalize = finalize_once if callable(finalize_callback) else None
        stream_gen = handler.safe_stream_generator(
            stream,
            save_callback,
            guarded_finalize,
            before_success_callback,
            on_first_output,
        )
        heartbeats_enabled = isinstance(heartbeat_interval, (int, float)) and heartbeat_interval > 0
        heartbeat_gen = handler.heartbeat_generator() if heartbeats_enabled else None

        stream_task: Optional[asyncio.Future[Any]] = None
        heartbeat_task: Optional[asyncio.Task[Any]] = None

        def capacity_error_frame() -> str:
            handler.error_occurred = True
            payload = provider_stream_error_payload("provider_unavailable")
            handler._attach_stream_metadata(payload)
            return f"data: {json.dumps(payload)}\n\n"

        async def close_and_finalize_initial_capacity_rejection() -> None:
            close = getattr(stream, "aclose", None)
            if not callable(close):
                close = getattr(stream, "close", None)
            if callable(close):
                try:
                    await invoke_stream_close_bounded(close)
                except _STREAMING_NONCRITICAL_EXCEPTIONS as close_error:
                    logger.debug(
                        "Capacity-rejected stream close failed error_type={}",
                        type(close_error).__name__,
                    )
            if guarded_finalize is not None:
                try:
                    await guarded_finalize(
                        success=False,
                        cancelled=False,
                        error=True,
                    )
                except _STREAMING_NONCRITICAL_EXCEPTIONS as finalize_error:
                    logger.debug(
                        "Capacity-rejected stream finalize failed error_type={}",
                        type(finalize_error).__name__,
                    )

        try:
            stream_task = create_bounded_stream_task(stream_gen.__anext__())
            heartbeat_task = (
                asyncio.create_task(heartbeat_gen.__anext__())
                if heartbeats_enabled and heartbeat_gen is not None
                else None
            )
            while not handler.is_cancelled and (stream_task is not None or heartbeat_task is not None):
                if handler.error_occurred and heartbeat_task is not None:
                    if not heartbeat_task.done():
                        heartbeat_task.cancel()
                    heartbeat_task = None
                wait_set = {t for t in (stream_task, heartbeat_task) if t is not None}
                if not wait_set:
                    break
                done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                should_exit = False
                for task in done:
                    try:
                        result = task.result()
                        if task is stream_task:
                            # Stream chunk
                            if result is not None:
                                yield result
                            # Schedule next chunk
                            stream_task = create_bounded_stream_task(stream_gen.__anext__())
                        elif heartbeat_task is not None and task is heartbeat_task:
                            # Heartbeat
                            if result is not None:
                                yield result
                            # Schedule next heartbeat
                            if heartbeats_enabled and heartbeat_gen is not None:
                                heartbeat_task = asyncio.create_task(heartbeat_gen.__anext__())
                    except StopAsyncIteration:
                        # A generator ended naturally; exit the loop without flagging cancel
                        should_exit = True
                    except asyncio.CancelledError:
                        # Task was cancelled (likely due to shutdown); exit loop
                        should_exit = True
                    except StreamTaskCapacityError:
                        error_frame = capacity_error_frame()
                        if guarded_finalize is not None:
                            try:
                                await guarded_finalize(
                                    success=False,
                                    cancelled=False,
                                    error=True,
                                )
                            except _STREAMING_NONCRITICAL_EXCEPTIONS as finalize_error:
                                logger.debug(
                                    "Capacity-exhausted stream finalize failed error_type={}",
                                    type(finalize_error).__name__,
                                )
                        yield error_frame
                        should_exit = True
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
                        normalized_error = normalize_provider_stream_error(e)
                        logger.error(
                            "Streaming task failed code={} error_type={}",
                            normalized_error.code if normalized_error else "provider_unavailable",
                            type(e).__name__,
                        )
                        handler.error_occurred = True
                        should_exit = True

                # Do not cancel pending tasks on normal loop progression; keep them running

                if should_exit:
                    gather_targets = tuple(filter(None, (stream_task, heartbeat_task)))
                    if gather_targets:
                        try:
                            await cancel_stream_tasks_bounded(gather_targets)
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as gather_err:
                            logger.debug(
                                "Task gather cleanup error_type={}",
                                type(gather_err).__name__,
                            )
                    # As a safety net, emit a final [DONE] only if it hasn't been sent yet
                    try:
                        if not handler.done_sent and not handler.is_cancelled:
                            yield "data: [DONE]\n\n"
                            handler.done_sent = True
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as done_err:
                        logger.debug(
                            "Final DONE emission error_type={}",
                            type(done_err).__name__,
                        )
                    break
        except StreamTaskCapacityError:
            if stream_task is None:
                await close_and_finalize_initial_capacity_rejection()
            yield capacity_error_frame()
            if not handler.done_sent and not handler.is_cancelled:
                yield "data: [DONE]\n\n"
                handler.done_sent = True
        finally:
            # Provider iterators may ignore cancellation. Drain briefly, then
            # detach with result observers so one stream cannot pin this loop.
            remaining_tasks = [t for t in (stream_task, heartbeat_task) if t is not None]
            if remaining_tasks:
                try:
                    await cancel_stream_tasks_bounded(remaining_tasks)
                except _STREAMING_NONCRITICAL_EXCEPTIONS as final_gather_err:
                    logger.debug("Final task cleanup error_type={}", type(final_gather_err).__name__)
            # Ensure generators are properly closed; avoid yielding here
            try:
                await await_stream_operation_bounded(
                    stream_gen.aclose(),
                    cleanup=True,
                )
            except asyncio.TimeoutError:
                logger.debug("Stream generator close exceeded bounded timeout")
            except _STREAMING_NONCRITICAL_EXCEPTIONS as stream_close_err:
                logger.debug("Stream generator close error_type={}", type(stream_close_err).__name__)
            if heartbeat_gen is not None:
                try:
                    await await_stream_operation_bounded(
                        heartbeat_gen.aclose(),
                        cleanup=True,
                    )
                except asyncio.TimeoutError:
                    logger.debug("Heartbeat generator close exceeded bounded timeout")
                except _STREAMING_NONCRITICAL_EXCEPTIONS as heartbeat_close_err:
                    logger.debug(
                        "Heartbeat generator close error_type={}",
                        type(heartbeat_close_err).__name__,
                    )

    async for message in stream_with_heartbeat():
        yield message


class StopStreamWithError(Exception):
    """Signal the streaming handler to stop after emitting an SSE error payload."""
    def __init__(self, message: str = "Stream blocked by policy", error_type: str = "stream_error"):
        super().__init__(message)
        self.error_type = error_type


#
# End of streaming_utils.py
#######################################################################################################################
