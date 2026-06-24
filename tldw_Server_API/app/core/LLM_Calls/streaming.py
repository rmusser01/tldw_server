"""
Common streaming helpers for LLM providers.

These utilities standardize how we iterate provider streams and normalize
their output into OpenAI-compatible SSE chunks. They intentionally suppress
forwarding a provider's own [DONE] line; callers should append a single
final sentinel using sse_done()/finalize_stream to avoid duplicates.
"""

import asyncio
import concurrent.futures
import os
import threading
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.http_client import RetryPolicy, astream_sse
from tldw_Server_API.app.core.LLM_Calls.error_utils import is_chunked_encoding_error

from .sse import is_done_line, normalize_provider_line, sse_data


_DEFAULT_SYNC_STREAM_QUEUE_SIZE = 16
_SYNC_STREAM_QUEUE_TIMEOUT_SECONDS = 0.05


def iter_sse_lines_requests(
    response: Any,
    *,
    decode_unicode: bool = True,
    provider: str = "provider",
    provider_control_passthru: Optional[bool] = None,
    control_filter: Optional[Callable[[str, str], Optional[tuple[str, str]]]] = None,
) -> Iterator[str]:
    """Yield normalized SSE lines from a Response-like stream.

    - Skips blank/control lines and suppresses provider [DONE] frames
      (caller should append a final sentinel once).
    - Wraps unexpected payloads as OpenAI delta chunks.
    - Converts common transport errors into SSE error payloads rather than
      raising mid-stream.
    """
    try:
        for raw_line in response.iter_lines(decode_unicode=decode_unicode):
            if not raw_line:
                continue
            # raw_line can be bytes when decode_unicode=False
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, (bytes, bytearray)) else str(raw_line)
            if not line:
                continue
            if is_done_line(line):
                # Suppress forwarding provider's [DONE]; caller will append one.
                continue
            passthru = (
                provider_control_passthru
                if provider_control_passthru is not None
                else os.getenv("STREAM_PROVIDER_CONTROL_PASSTHRU", "0") == "1"
            )
            normalized = normalize_provider_line(
                line,
                provider_control_passthru=passthru,
                control_filter=control_filter,
            )
            if normalized is None:
                continue
            yield normalized
    except Exception as e_stream:
        # Surface as an SSE error frame so the client can handle gracefully
        if is_chunked_encoding_error(e_stream):
            message = f"Stream connection error: {str(e_stream)}"
        else:
            message = f"Stream iteration error: {str(e_stream)}"
        yield sse_data({"error": {"message": message, "type": f"{provider}_stream_error"}})


async def aiter_sse_lines_httpx(
    resp: Any,
    *,
    provider: str = "provider",
    provider_control_passthru: Optional[bool] = None,
    control_filter: Optional[Callable[[str, str], Optional[tuple[str, str]]]] = None,
) -> AsyncIterator[str]:
    """Async iterator of normalized SSE lines for an httpx streaming response.

    - Skips provider [DONE] frames; callers should append one final sentinel.
    - Wraps unexpected payloads as OpenAI delta chunks.
    - Converts transport errors during iteration into SSE error payloads.
    """
    try:
        async for line in resp.aiter_lines():
            if not line:
                continue
            if is_done_line(line):
                continue
            passthru = (
                provider_control_passthru
                if provider_control_passthru is not None
                else os.getenv("STREAM_PROVIDER_CONTROL_PASSTHRU", "0") == "1"
            )
            normalized = normalize_provider_line(
                line,
                provider_control_passthru=passthru,
                control_filter=control_filter,
            )
            if normalized is None:
                continue
            yield normalized
    except Exception as e_stream:
        yield sse_data({"error": {"message": f"Stream iteration error: {str(e_stream)}", "type": f"{provider}_stream_error"}})


async def wrap_sync_stream(
    sync_iter: Iterable[str],
    *,
    max_queue_size: int = _DEFAULT_SYNC_STREAM_QUEUE_SIZE,
) -> AsyncIterator[str]:
    """Bridge a sync generator into an async iterator without blocking the event loop."""
    max_queue_size = max(1, int(max_queue_size or _DEFAULT_SYNC_STREAM_QUEUE_SIZE))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_queue_size)
    sentinel = object()
    stop_event = threading.Event()
    close_lock = threading.Lock()
    closed = False

    def _close_sync_iter() -> None:
        nonlocal closed
        with close_lock:
            if closed:
                return
            closed = True
        try:
            close_fn = getattr(sync_iter, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as close_error:
            logger.debug("Sync stream iterator close failed during cleanup: {}", close_error)

    def _put_item(item: Any) -> bool:
        if stop_event.is_set():
            return False
        try:
            put_future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        except RuntimeError as queue_error:
            if not stop_event.is_set():
                logger.debug("Sync stream queue put failed before scheduling: {}", queue_error)
            return False

        while True:
            if stop_event.is_set():
                put_future.cancel()
                return False
            try:
                put_future.result(timeout=_SYNC_STREAM_QUEUE_TIMEOUT_SECONDS)
                return True
            except concurrent.futures.TimeoutError:
                continue
            except concurrent.futures.CancelledError:
                return False
            except Exception as put_error:
                if not stop_event.is_set():
                    logger.debug("Sync stream queue put failed: {}", put_error)
                return False
        return False

    def _put_sentinel() -> None:
        _put_item(sentinel)

    def _worker() -> None:
        try:
            for item in sync_iter:
                if stop_event.is_set():
                    break
                if not _put_item(item):
                    break
        except Exception as exc:
            _put_item(exc)
        finally:
            _close_sync_iter()
            _put_sentinel()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        _close_sync_iter()


async def aiter_normalized_sse(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    json: Optional[dict] = None,
    data: Optional[dict] = None,
    retry: Optional[RetryPolicy] = None,
    provider: str = "provider",
    provider_control_passthru: Optional[bool] = None,
    control_filter: Optional[Callable[[str, str], Optional[tuple[str, str]]]] = None,
) -> AsyncIterator[str]:
    """Standardized SSE iterator built on the centralized astream_sse helper.

    - Enforces egress policy and retries per PRD defaults.
    - Normalizes provider lines using existing helpers.
    """
    passthru = (
        provider_control_passthru
        if provider_control_passthru is not None
        else os.getenv("STREAM_PROVIDER_CONTROL_PASSTHRU", "0") == "1"
    )
    async for ev in astream_sse(url=url, method=method, headers=headers, params=params, json=json, data=data, retry=retry):
        if not ev or not ev.data:
            continue
        # Normalize SSE payload as if it were a provider line
        normalized = normalize_provider_line(
            ev.data,
            provider_control_passthru=passthru,
            control_filter=control_filter,
        )
        if normalized is None:
            continue
        yield normalized
