import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.LLM_Calls.sse import (
    ensure_sse_control_line,
    ensure_sse_line,
    sse_data,
    sse_done,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)

_STREAM_METRICS_REGISTERED = False

_STREAMING_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _ensure_stream_metrics_registered() -> None:
    global _STREAM_METRICS_REGISTERED
    if _STREAM_METRICS_REGISTERED:
        return
    reg = get_metrics_registry()
    try:
        reg.register_metric(
            MetricDefinition(
                name="sse_enqueue_to_yield_ms",
                type=MetricType.HISTOGRAM,
                description="Time from SSE enqueue to yield (ms)",
                unit="ms",
                labels=["component", "endpoint", "transport"],
                buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
            )
        )
        reg.register_metric(
            MetricDefinition(
                name="sse_queue_high_watermark",
                type=MetricType.GAUGE,
                description="Max SSE queue size observed",
                unit="items",
                labels=["component", "endpoint", "transport"],
            )
        )
        reg.register_metric(
            MetricDefinition(
                name="ws_send_latency_ms",
                type=MetricType.HISTOGRAM,
                description="WebSocket send_json latency (ms)",
                unit="ms",
                labels=["component", "endpoint", "transport", "kind"],
                buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
            )
        )
        reg.register_metric(
            MetricDefinition(
                name="ws_pings_total",
                type=MetricType.COUNTER,
                description="Total WS ping frames sent",
                labels=["component", "endpoint", "transport"],
            )
        )
        reg.register_metric(
            MetricDefinition(
                name="ws_ping_failures_total",
                type=MetricType.COUNTER,
                description="Total WS ping send failures",
                labels=["component", "endpoint", "transport"],
            )
        )
        reg.register_metric(
            MetricDefinition(
                name="ws_idle_timeouts_total",
                type=MetricType.COUNTER,
                description="Total WS idle timeouts",
                labels=["component", "endpoint", "transport"],
            )
        )
        _STREAM_METRICS_REGISTERED = True
    except _STREAMING_NONCRITICAL_EXCEPTIONS:
        logger.debug("Stream metrics registration failed or already registered")


class SSEStream:
    """
    SSE stream helper with queue, heartbeats, and optional idle/max enforcement.

    Features:
    - Bounded queue (default maxsize 256; block on full)
    - Heartbeats (comment or data mode)
    - Provider control pass-through toggle for upstream normalization flows
    - Idle timeout and max duration enforcement (idle tracks last emitted line for connection liveness)
    - Optional labels dict for metrics tagging (not enforced here)
    """

    def __init__(
        self,
        *,
        heartbeat_interval_s: Optional[float] = None,
        heartbeat_mode: Optional[str] = None,  # "comment" or "data"; env-driven by default
        queue_maxsize: Optional[int] = None,
        close_on_error: bool = True,
        idle_timeout_s: Optional[float] = None,
        max_duration_s: Optional[float] = None,
        provider_control_passthru: Optional[bool] = None,
        control_filter: Optional[Callable[[str, str], Optional[tuple[str, str]]]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        self.heartbeat_interval_s = (
            heartbeat_interval_s
            if heartbeat_interval_s is not None
            else float(os.getenv("STREAM_HEARTBEAT_INTERVAL_S", "10"))
        )
        self.heartbeat_mode = (
            heartbeat_mode if heartbeat_mode is not None else os.getenv("STREAM_HEARTBEAT_MODE", "comment")
        )
        self.queue_maxsize = (
            queue_maxsize if queue_maxsize is not None else int(os.getenv("STREAM_QUEUE_MAXSIZE", "256"))
        )
        self.close_on_error = close_on_error
        self.idle_timeout_s = (
            idle_timeout_s
            if idle_timeout_s is not None
            else _parse_float_env("STREAM_IDLE_TIMEOUT_S")
        )
        self.max_duration_s = (
            max_duration_s
            if max_duration_s is not None
            else _parse_float_env("STREAM_MAX_DURATION_S")
        )
        self.provider_control_passthru = (
            provider_control_passthru
            if provider_control_passthru is not None
            else (os.getenv("STREAM_PROVIDER_CONTROL_PASSTHRU", "0") == "1")
        )
        self.control_filter = control_filter
        self.labels = labels or {}

        _ensure_stream_metrics_registered()
        self._queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue(maxsize=self.queue_maxsize)
        self._closed = False
        self._done_enqueued = False
        self._high_watermark = 0
        self._labels = {"transport": "sse"}
        self._labels.update(self.labels)

    async def send_event(
        self,
        event: str,
        data: Any | None = None,
        *,
        event_id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> None:
        # Compose a single SSE frame: optional id/retry, event line, then data or a blank line.
        if event_id is not None:
            await self._enqueue(ensure_sse_control_line(f"id: {event_id}"))
        if retry is not None:
            await self._enqueue(ensure_sse_control_line(f"retry: {retry}"))
        await self._enqueue(ensure_sse_control_line(f"event: {event}"))
        if data is not None:
            await self.send_json(data)
        else:
            # SSE requires a blank line to dispatch event
            await self._enqueue("\n")

    async def send_json(self, payload: dict[str, Any], *, force: bool = False) -> None:
        await self._enqueue(sse_data(payload), force=force)

    async def send_raw_sse_line(self, line: str) -> None:
        if "\n" in line:
            lower = line.lower()
            if "data:" in lower:
                await self._enqueue(ensure_sse_line(line))
            else:
                await self._enqueue(ensure_sse_control_line(line))
            return
        stripped = line.lstrip()
        lower = stripped.lower()
        if lower.startswith(("event:", "id:", "retry:", ":")):
            await self._enqueue(ensure_sse_control_line(line))
        else:
            await self._enqueue(ensure_sse_line(line))

    async def error(
        self,
        code: str,
        message: str,
        *,
        data: Optional[dict[str, Any]] = None,
        close: Optional[bool] = None,
        force: bool = False,
    ) -> None:
        payload: dict[str, Any] = {"error": {"code": code, "message": message}}
        if data is not None:
            payload["error"]["data"] = data
        await self.send_json(payload, force=force)
        should_close = self.close_on_error if close is None else bool(close)
        if should_close:
            await self.done(force=force)

    async def done(self, *, force: bool = False) -> None:
        if not self._done_enqueued:
            self._done_enqueued = True
            await self._enqueue(sse_done(), force=force)
        self._closed = True

    async def iter_sse(self) -> AsyncIterator[str]:
        start_ts = time.monotonic()
        last_emit_ts = start_ts
        last_hb_ts = start_ts

        while not self._closed:
            now = time.monotonic()
            # Enforce max duration proactively even when data continues flowing
            if self.max_duration_s and self.max_duration_s > 0 and now >= start_ts + self.max_duration_s:
                await self.error(
                    "max_duration_exceeded",
                    "stream exceeded maximum duration",
                    force=True,
                )
                break
            # Compute deadlines
            next_heartbeat_delta = None
            if self.heartbeat_interval_s and self.heartbeat_interval_s > 0:
                hb_due_at = last_hb_ts + self.heartbeat_interval_s
                next_heartbeat_delta = max(0.0, hb_due_at - now)

            # Idle timeout is based on last emitted line (including heartbeats) for liveness tracking.
            idle_delta = None
            if self.idle_timeout_s and self.idle_timeout_s > 0:
                idle_due_at = last_emit_ts + self.idle_timeout_s
                idle_delta = max(0.0, idle_due_at - now)

            max_delta = None
            if self.max_duration_s and self.max_duration_s > 0:
                max_due_at = start_ts + self.max_duration_s
                max_delta = max(0.0, max_due_at - now)

            timeouts = [d for d in (next_heartbeat_delta, idle_delta, max_delta) if d is not None]
            timeout = min(timeouts) if timeouts else None

            try:
                if timeout is not None and timeout <= 0:
                    # Avoid wait_for(timeout=0) which can time out even with queued data.
                    try:
                        line, enq_ts = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        raise asyncio.TimeoutError from None
                elif timeout is not None:
                    line, enq_ts = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                else:
                    line, enq_ts = await self._queue.get()
                last_emit_ts = time.monotonic()
                try:
                    dt_ms = max(0.0, (last_emit_ts - enq_ts) * 1000.0)
                    get_metrics_registry().observe("sse_enqueue_to_yield_ms", dt_ms, self._labels)
                except _STREAMING_NONCRITICAL_EXCEPTIONS:
                    pass
                with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                    logger.debug(f"SSEStream yielding line: {line.strip()[:120]}")
                yield line
                continue
            except asyncio.TimeoutError:
                now = time.monotonic()

                # Check terminal conditions first
                if self.idle_timeout_s and self.idle_timeout_s > 0:
                    if now >= last_emit_ts + self.idle_timeout_s:
                        await self.error("idle_timeout", "idle timeout", force=True)
                        # error() enqueues DONE when close_on_error
                        break
                # Max duration is also enforced above to cover active-stream cases

                # Heartbeat (suppressed once DONE is enqueued)
                if self.heartbeat_interval_s and self.heartbeat_interval_s > 0 and not self._done_enqueued:
                    if now >= last_hb_ts + self.heartbeat_interval_s:
                        with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                            logger.debug(
                                f"SSEStream heartbeat emit mode={self.heartbeat_mode} interval_s={self.heartbeat_interval_s}"
                            )
                        if self.heartbeat_mode == "data":
                            await self._enqueue(sse_data({"heartbeat": True}))
                        else:
                            # Comment heartbeat
                            await self._enqueue(ensure_sse_line(":"))
                        last_hb_ts = time.monotonic()
                        # Drain immediately
                        continue

        # Drain any remaining items (e.g., DONE) if the loop was closed by error/done
        while not self._queue.empty():
            try:
                line, enq_ts = self._queue.get_nowait()
                try:
                    dt_ms = max(0.0, (time.monotonic() - enq_ts) * 1000.0)
                    get_metrics_registry().observe("sse_enqueue_to_yield_ms", dt_ms, self._labels)
                except _STREAMING_NONCRITICAL_EXCEPTIONS:
                    pass
                yield line
            except asyncio.QueueEmpty:
                break

    async def _enqueue(self, line: str, *, force: bool = False) -> None:
        # Blocking (default) backpressure policy; force=True drops oldest to ensure termination frames.
        if self._closed and not force:
            return
        enq_ts = time.monotonic()
        if not force:
            await self._queue.put((line, enq_ts))
        else:
            try:
                self._queue.put_nowait((line, enq_ts))
            except asyncio.QueueFull:
                with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                    _ = self._queue.get_nowait()
                try:
                    self._queue.put_nowait((line, enq_ts))
                except _STREAMING_NONCRITICAL_EXCEPTIONS:
                    return
        try:
            qsize = self._queue.qsize()
            if qsize > self._high_watermark:
                self._high_watermark = qsize
                get_metrics_registry().set_gauge("sse_queue_high_watermark", float(self._high_watermark), self._labels)
        except _STREAMING_NONCRITICAL_EXCEPTIONS:
            pass


class WebSocketStream:
    """
    WebSocket stream helper providing standardized lifecycle frames, optional ping loop,
    close code mapping, and metrics.
    """

    def __init__(
        self,
        websocket: Any,
        *,
        heartbeat_interval_s: Optional[float] = None,
        close_on_done: bool = True,
        compat_error_type: bool = False,
        idle_timeout_s: Optional[float] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        _ensure_stream_metrics_registered()
        self.ws = websocket
        self.heartbeat_interval_s = heartbeat_interval_s if heartbeat_interval_s is not None else float(
            os.getenv("STREAM_HEARTBEAT_INTERVAL_S", "10")
        )
        self.close_on_done = close_on_done
        self.compat_error_type = compat_error_type
        self.idle_timeout_s = idle_timeout_s
        self.labels = labels or {}
        self._labels = {"transport": "ws"}
        self._labels.update(self.labels)

        self._running = False
        self._ping_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._last_activity = time.monotonic()

    async def start(self) -> None:
        self._running = True
        # Accept the connection if not already accepted
        try:
            already_accepted = False
            try:
                # Starlette exposes application_state when available
                state = getattr(self.ws, "application_state", None)
                # Avoid importing starlette if not present in tests
                if state is not None:
                    # Compare string form to avoid importing WebSocketState enum
                    if str(state).upper().endswith("CONNECTED"):
                        already_accepted = True
            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                already_accepted = False
            if hasattr(self.ws, "accept") and not already_accepted:
                await maybe_await(self.ws.accept())
        except _STREAMING_NONCRITICAL_EXCEPTIONS:
            pass
        if self.heartbeat_interval_s and self.heartbeat_interval_s > 0:
            self._ping_task = asyncio.create_task(self._ping_loop())
        if self.idle_timeout_s and self.idle_timeout_s > 0:
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def stop(self) -> None:
        self._running = False
        for task in (self._ping_task, self._idle_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Python 3.11+ raises CancelledError as BaseException; ignore on shutdown
                    pass
                except _STREAMING_NONCRITICAL_EXCEPTIONS:
                    pass

    def mark_activity(self) -> None:
        self._last_activity = time.monotonic()

    async def receive_text(self) -> str:
        """Receive a text frame and mark activity for idle tracking."""
        text = await maybe_await(self.ws.receive_text())
        with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
            self.mark_activity()
        return text

    async def receive_json(self) -> Any:
        """Receive a JSON frame and mark activity for idle tracking."""
        data = await maybe_await(self.ws.receive_json())
        with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
            self.mark_activity()
        return data

    async def send_event(self, event: str, data: Any | None = None) -> None:
        # Optional WS event frame
        payload = {"type": "event", "event": event}
        if data is not None:
            payload["data"] = data
        await self._send_json_with_metrics(payload, kind="event")

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._send_json_with_metrics(payload, kind="json")

    async def done(self, *, close_code: int = 1000) -> None:
        await self._send_json_with_metrics({"type": "done"}, kind="done")
        if self.close_on_done:
            with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                await maybe_await(self.ws.close(code=close_code))

    async def error(self, code: str, message: str, *, data: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {"type": "error", "code": code, "message": message}
        if data is not None:
            payload["data"] = data
        if self.compat_error_type:
            payload["error_type"] = code
            # Compatibility shim: surface certain data fields at top-level
            try:
                if isinstance(data, dict) and "quota" in data:
                    payload["quota"] = data.get("quota")
            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                pass
        await self._send_json_with_metrics(payload, kind="error")
        close_code = self._map_close_code(code)
        with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
            await maybe_await(self.ws.close(code=close_code))

    async def _send_json_with_metrics(self, payload: dict[str, Any], *, kind: str) -> None:
        t0 = time.monotonic()
        sent = False
        try:
            await maybe_await(self.ws.send_json(payload))
            sent = True
        except _STREAMING_NONCRITICAL_EXCEPTIONS:
            self._running = False
            raise
        finally:
            dt_ms = max(0.0, (time.monotonic() - t0) * 1000.0)
            with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                get_metrics_registry().observe("ws_send_latency_ms", dt_ms, {**self._labels, "kind": kind})
            if sent:
                self.mark_activity()

    async def _ping_loop(self) -> None:
        reg = get_metrics_registry()
        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval_s)
                try:
                    await self._send_json_with_metrics({"type": "ping"}, kind="ping")
                    reg.increment("ws_pings_total", 1, self._labels)
                except _STREAMING_NONCRITICAL_EXCEPTIONS:
                    reg.increment("ws_ping_failures_total", 1, self._labels)
                    self._running = False
                    break
        except asyncio.CancelledError:
            return

    async def _idle_loop(self) -> None:
        reg = get_metrics_registry()
        try:
            while self._running:
                await asyncio.sleep(max(0.05, min(self.idle_timeout_s or 60.0, 1.0)))
                now = time.monotonic()
                if self.idle_timeout_s and now - self._last_activity >= self.idle_timeout_s:
                    # Close with 1001 and increment counter
                    reg.increment("ws_idle_timeouts_total", 1, self._labels)
                    with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                        await maybe_await(self.ws.close(code=1001))
                    self._running = False
                    break
        except asyncio.CancelledError:
            return

    @staticmethod
    def _map_close_code(code: str) -> int:
        lower = (code or "").lower()
        if lower == "quota_exceeded":
            return 1008
        if lower == "idle_timeout":
            return 1001
        if lower in ("internal_error", "transport_error", "provider_error"):
            return 1011
        return 1000


async def maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


def _parse_float_env(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return float(raw)
    except _STREAMING_NONCRITICAL_EXCEPTIONS:
        logger.debug("Invalid float in stream env")
        return None
