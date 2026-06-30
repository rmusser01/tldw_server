from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import threading
import time
import uuid

from loguru import logger

from tldw_Server_API.app.core.Infrastructure.redis_factory import create_sync_redis_client
from tldw_Server_API.app.core.testing import is_truthy

try:
    import redis as _redis_lib
    _REDIS_EXCEPTIONS: tuple[type[Exception], ...] = (_redis_lib.RedisError,)
except ImportError:
    _REDIS_EXCEPTIONS = ()

_SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
) + _REDIS_EXCEPTIONS


class RunStreamHub:
    """In-memory pub/sub for run log/event streaming with caps and backpressure."""

    def __init__(self) -> None:
        # Map run_id -> list of (loop, subscriber queue) pairs (fan-out to all subscribers)
        self._queues: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._buffers: dict[str, list[dict]] = {}
        self._log_bytes: dict[str, int] = {}
        self._truncated: set[str] = set()
        self._ended: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._max_queue = 1000
        self._max_log_bytes_default = 10 * 1024 * 1024
        self._seq: dict[str, int] = {}
        # Per-run serialized dispatcher
        self._dispatch: dict[str, list[dict]] = {}
        self._dispatching: set[str] = set()
        # Interactive stdin caps and state per run
        self._stdin_cfg: dict[str, dict] = {}
        self._stdin_state: dict[str, dict] = {}
        # Inbound stdin data queues per run (producer: WS handler; consumer: runner)
        import queue as _queue  # local import to avoid global in non-stdin contexts
        self._stdin_queues: dict[str, _queue.Queue] = {}
        # Optional Redis fan-out (cross-worker broadcast)
        self._redis_enabled: bool = False
        self._redis_client = None
        self._redis_thread: threading.Thread | None = None
        self._redis_channel: str = str(os.getenv("SANDBOX_WS_REDIS_CHANNEL") or "tldw:sandbox:streams:v1")
        self._instance_id: str = uuid.uuid4().hex
        self._maybe_enable_redis_fanout()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        # Always update to the current running loop. In test environments,
        # TestClient may create transient loops per connection; using the
        # latest loop ensures call_soon_threadsafe targets a live loop.
        with self._lock:
            self._loop = loop
            pending = list(self._dispatching)
        # If frames were queued while no live loop was available (or a prior loop
        # was closed), re-schedule dispatch now that we have a fresh loop.
        for run_id in pending:
            try:
                self._schedule_dispatch(run_id)
            except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                continue

    def _get_queue(self, run_id: str) -> asyncio.Queue:
        """Create a new subscriber queue for the given run_id and register it.

        Each call returns a distinct asyncio.Queue so multiple subscribers
        receive the same frames independently and in the same order.
        """
        with self._lock:
            q = asyncio.Queue(self._max_queue)
            loop = self._resolve_subscriber_loop()
            self._queues.setdefault(run_id, []).append((loop, q))
            return q

    def subscribe(self, run_id: str) -> asyncio.Queue:
        return self._get_queue(run_id)

    def subscribe_with_buffer(self, run_id: str) -> asyncio.Queue:
        """Subscribe a new consumer and pre-fill its queue with buffered frames.

        Ensures buffered frames have sequence numbers assigned before enqueueing
        them for this subscriber, avoiding races where live dispatch could stamp
        seq later and interleave frames. The subscriber is only registered after
        the buffered frames are enqueued, so it will start receiving new frames
        from the dispatcher afterwards while still seeing the historical frames
        first on its own queue.
        """
        with self._lock:
            q = asyncio.Queue(self._max_queue)
            loop = self._resolve_subscriber_loop()
            # Stamp seq on buffered frames if missing, then copy into this queue
            buf = self._buffers.get(run_id) or []
            import copy as _copy
            for frame in buf[-100:]:
                if isinstance(frame, dict) and "seq" not in frame:
                    frame["seq"] = self._next_seq(run_id)
                try:
                    q.put_nowait(_copy.deepcopy(frame))
                except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                    break
            # Finally register this subscriber for future live frames
            self._queues.setdefault(run_id, []).append((loop, q))
            return q

    def subscribe_with_buffer_from_seq(self, run_id: str, from_seq: int) -> asyncio.Queue:
        """Subscribe and pre-fill only buffered frames with seq >= from_seq.

        Stamps sequence numbers on buffered frames first (if missing) to ensure
        consistent numbering across subscribers, then enqueues only those with
        seq >= from_seq for this subscriber. Live frames are delivered as usual.
        """
        if from_seq is None or int(from_seq) <= 0:
            return self.subscribe_with_buffer(run_id)
        with self._lock:
            q = asyncio.Queue(self._max_queue)
            loop = self._resolve_subscriber_loop()
            buf = self._buffers.get(run_id) or []
            import copy as _copy
            for frame in buf[-100:]:
                if isinstance(frame, dict) and "seq" not in frame:
                    frame["seq"] = self._next_seq(run_id)
                try:
                    if isinstance(frame, dict) and int(frame.get("seq", 0)) >= int(from_seq):
                        q.put_nowait(_copy.deepcopy(frame))
                except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                    break
            self._queues.setdefault(run_id, []).append((loop, q))
            return q

    def _resolve_subscriber_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass
        if self._loop is not None:
            return self._loop
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # Python 3.11+ may not create a default loop for sync callers.
            # A private loop is sufficient for unit tests that drain buffers directly.
            return asyncio.new_event_loop()

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        """Remove a subscriber queue for a run.

        This is best-effort and safe to call multiple times.
        """
        with self._lock:
            subs = self._queues.get(run_id) or []
            if not subs:
                return
            remaining = [(lp, sq) for (lp, sq) in subs if sq is not q]
            if remaining:
                self._queues[run_id] = remaining
            else:
                self._queues.pop(run_id, None)

    def _next_seq(self, run_id: str) -> int:
        with self._lock:
            cur = self._seq.get(run_id, 0) + 1
            self._seq[run_id] = cur
            return cur

    def _publish(self, run_id: str, frame: dict) -> None:
        # Local dispatch first
        self._publish_local(run_id, frame)
        # Redis relay for cross-worker subscribers if enabled
        try:
            if self._redis_enabled and self._redis_client is not None:
                payload = {"origin": self._instance_id, "run_id": run_id, "frame": frame}
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                # fire-and-forget; swallow errors
                self._redis_client.publish(self._redis_channel, data)
        except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"redis publish failed: {e}")

    def _publish_local(self, run_id: str, frame: dict) -> None:
        with self._lock:
            # Buffer non-heartbeat frames for reconnects (seq is assigned at dispatch time)
            if not (isinstance(frame, dict) and frame.get("type") == "heartbeat"):
                self._buffers.setdefault(run_id, []).append(frame)
                buf = self._buffers.get(run_id)
                if buf is not None and len(buf) > 100:
                    del buf[:-100]
            # Enqueue for serialized dispatch (seq will be stamped in dispatcher)
            self._dispatch.setdefault(run_id, []).append(frame)
            if run_id not in self._dispatching:
                self._dispatching.add(run_id)
                self._schedule_dispatch(run_id)

    def reset_loop(self) -> None:
        """Clear cached event loop reference.  Useful in test fixtures to avoid stale loops."""
        with self._lock:
            self._loop = None

    def _schedule_dispatch(self, run_id: str) -> None:
        # Choose a loop to trigger the dispatcher; prefer last known loop or any subscriber loop
        with self._lock:
            loop = self._loop
            try:
                if loop is not None and loop.is_closed():
                    self._loop = None
                    loop = None
            except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                pass
            if not loop:
                subs = self._queues.get(run_id) or []
                loop = subs[0][0] if subs else None
        if loop is None:
            # No loop available yet; retry shortly from a timer thread
            try:
                t = threading.Timer(0.005, lambda: self._schedule_dispatch(run_id))
                t.daemon = True
                t.start()
            except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                pass
            return
        try:
            loop.call_soon_threadsafe(self._do_dispatch, run_id)
        except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"dispatch schedule failed: {e}")

    def _do_dispatch(self, run_id: str) -> None:
        # Drain queued frames and fan-out to all subscribers in arrival order
        while True:
            with self._lock:
                queue = self._dispatch.get(run_id) or []
                if not queue:
                    self._dispatching.discard(run_id)
                    return
                frame = queue.pop(0)
                # Stamp sequence centrally here to ensure strict ordering for all subscribers.
                # Update in-place so the buffered frame also carries the seq for future drains.
                if isinstance(frame, dict) and "seq" not in frame:
                    frame["seq"] = self._next_seq(run_id)
                subs = list(self._queues.get(run_id) or [])
            for (lp, q) in subs:
                try:
                    import copy as _copy
                    lp.call_soon_threadsafe(self._queue_put_nowait, q, _copy.deepcopy(frame))
                except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                    # Swallow delivery errors to individual subscribers
                    pass

    @staticmethod
    def _queue_put_nowait(q: asyncio.Queue, item: dict) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Drop oldest by draining one, then put
            with contextlib.suppress(_SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS):
                _ = q.get_nowait()
            # Metrics: queue overflow/drop
            try:
                from tldw_Server_API.app.core.Metrics import increment_counter
                increment_counter(
                    "sandbox_ws_queue_drops_total",
                    labels={"component": "sandbox", "reason": "drop_oldest"},
                )
            except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                pass
            with contextlib.suppress(_SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS):
                q.put_nowait(item)

    def publish_event(self, run_id: str, event: str, data: dict | None = None) -> None:
        # Deduplicate final end event to avoid double-emission from runner and service
        if event == "end":
            with self._lock:
                # Protect _ended access with the lock and avoid duplicates
                if run_id in self._ended:
                    return
                self._ended.add(run_id)
        frame = {"type": "event", "event": event, "data": data or {}}
        self._publish(run_id, frame)

    def has_ended(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._ended

    def publish_heartbeat(self, run_id: str) -> bool:
        """Publish a heartbeat frame unless the run is already ended."""
        with self._lock:
            if run_id in self._ended:
                return False
        self._publish(run_id, {"type": "heartbeat"})
        return True

    def publish_truncated(self, run_id: str, reason: str) -> None:
        """Publish a truncated frame with a reason code."""
        self._publish(run_id, {"type": "truncated", "reason": str(reason)})

    def mark_log_truncated(self, run_id: str) -> None:
        """Mark a run as log-truncated when a runner caps file-backed output."""
        with self._lock:
            should_emit = self._mark_log_truncated_locked(run_id)
        if should_emit:
            self._emit_log_truncated(run_id)

    def publish_stdout(self, run_id: str, chunk: bytes, max_log_bytes: int | None = None) -> None:
        self._publish_stream(run_id, "stdout", chunk, max_log_bytes=max_log_bytes)

    def publish_stderr(self, run_id: str, chunk: bytes, max_log_bytes: int | None = None) -> None:
        self._publish_stream(run_id, "stderr", chunk, max_log_bytes=max_log_bytes)

    def _publish_stream(self, run_id: str, kind: str, chunk: bytes, *, max_log_bytes: int | None) -> None:
        cap = max_log_bytes or self._max_log_bytes_default
        with self._lock:
            used = self._log_bytes.get(run_id, 0)
            if used >= cap:
                should_emit = self._mark_log_truncated_locked(run_id)
                remaining = 0
            else:
                should_emit = False
                remaining = cap - used
        if used >= cap:
            if should_emit:
                self._emit_log_truncated(run_id)
            return
        data = chunk[:remaining]
        truncated_chunk = len(chunk) > len(data)
        try:
            text = data.decode("utf-8")
            frame = {"type": kind, "encoding": "utf8", "data": text}
        except UnicodeDecodeError:
            b64 = base64.b64encode(data).decode("ascii")
            frame = {"type": kind, "encoding": "base64", "data": b64}
        with self._lock:
            self._log_bytes[run_id] = self._log_bytes.get(run_id, 0) + len(data)
        self._publish(run_id, frame)
        if truncated_chunk:
            with self._lock:
                should_emit = self._mark_log_truncated_locked(run_id)
            if should_emit:
                self._emit_log_truncated(run_id)

    def _mark_log_truncated_locked(self, run_id: str) -> bool:
        if run_id in self._truncated:
            return False
        self._truncated.add(run_id)
        return True

    def _emit_log_truncated(self, run_id: str) -> None:
        self._publish(run_id, {"type": "truncated", "reason": "log_cap"})
        try:
            from tldw_Server_API.app.core.Metrics import increment_counter
            increment_counter(
                "sandbox_log_truncations_total",
                labels={"component": "sandbox", "reason": "log_cap"},
            )
        except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
            pass

    def drain_buffer(self, run_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            buf = self._buffers.get(run_id) or []
            if not buf:
                return
            # Emit up to the last 100 buffered frames with seq stamped
            for frame in buf[-100:]:
                try:
                    if isinstance(frame, dict) and "seq" not in frame:
                        frame["seq"] = self._next_seq(run_id)
                    import copy as _copy
                    q.put_nowait(_copy.deepcopy(frame))
                except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                    break

    def get_buffer_snapshot(self, run_id: str) -> list[dict]:
        """Return a deep-copied snapshot of the buffered frames for a run.

        The snapshot contains at most the last 100 frames and preserves any
        assigned sequence numbers. Heartbeats are not buffered and therefore
        are not included in this snapshot.
        """
        with self._lock:
            import copy as _copy
            buf = self._buffers.get(run_id) or []
            return [_copy.deepcopy(f) for f in buf[-100:]]

    # -----------------
    # Interactive stdin
    # -----------------
    def configure_stdin(self, run_id: str, *, interactive: bool,
                         stdin_max_bytes: int | None = None,
                         stdin_max_frame_bytes: int | None = None,
                         stdin_bps: int | None = None,
                         stdin_idle_timeout_sec: int | None = None) -> None:
        """Configure stdin caps for a run. If interactive is False, clears any config."""
        with self._lock:
            if not interactive:
                self._stdin_cfg.pop(run_id, None)
                self._stdin_state.pop(run_id, None)
                return
            cfg = {
                "interactive": True,
                "stdin_max_bytes": int(stdin_max_bytes) if stdin_max_bytes is not None else None,
                "stdin_max_frame_bytes": int(stdin_max_frame_bytes) if stdin_max_frame_bytes is not None else None,
                "stdin_bps": int(stdin_bps) if stdin_bps is not None else None,
                "stdin_idle_timeout_sec": int(stdin_idle_timeout_sec) if stdin_idle_timeout_sec is not None else None,
            }
            self._stdin_cfg[run_id] = cfg
            st = self._stdin_state.get(run_id) or {}
            # Initialize token bucket and counters
            import time as _time
            st.setdefault("bytes_total", 0)
            st.setdefault("last_refill", float(_time.time()))
            # bucket capacity equals 1 second worth of tokens
            rate = cfg.get("stdin_bps") or 0
            st.setdefault("tokens", int(rate))
            st["rate"] = int(rate)
            st.setdefault("last_input", float(_time.time()))
            self._stdin_state[run_id] = st

    def get_stdin_config(self, run_id: str) -> dict | None:
        with self._lock:
            cfg = self._stdin_cfg.get(run_id)
            return dict(cfg) if cfg else None

    def _refill_tokens(self, st: dict) -> None:
        try:
            import time as _time
            now = float(_time.time())
            last = float(st.get("last_refill", now))
            rate = int(st.get("rate", 0))
            if rate <= 0:
                st["last_refill"] = now
                return
            delta = max(0.0, now - last)
            add = int(delta * rate)
            cap = rate  # 1s burst
            tokens = int(st.get("tokens", 0))
            tokens = min(cap, tokens + add)
            st["tokens"] = tokens
            st["last_refill"] = now
        except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
            return

    def consume_stdin(self, run_id: str, data_len: int) -> tuple[int, str | None]:
        """Consume stdin bytes for a run according to configured caps.

        Returns a tuple of (allowed_bytes, reason_if_truncated). If allowed_bytes is 0
        and a reason is provided, the caller may drop the frame or retry later.
        """
        with self._lock:
            cfg = self._stdin_cfg.get(run_id)
            if not cfg or not cfg.get("interactive"):
                return (0, None)
            st = self._stdin_state.setdefault(run_id, {})
            # refill tokens for rate limiting
            self._refill_tokens(st)
            allowed = int(data_len)
            reason: str | None = None
            # Per-frame cap
            if cfg.get("stdin_max_frame_bytes") is not None:
                mfb = int(cfg["stdin_max_frame_bytes"])
                if allowed > mfb:
                    allowed = mfb
                    reason = reason or "stdin_frame_cap"
            # Rate limit
            tokens = int(st.get("tokens", 0))
            if cfg.get("stdin_bps") is not None:
                if tokens <= 0:
                    allowed = 0
                    reason = reason or "stdin_rate"
                else:
                    if allowed > tokens:
                        allowed = tokens
                        reason = reason or "stdin_rate"
                    st["tokens"] = max(0, tokens - allowed)
            # Total cap
            if cfg.get("stdin_max_bytes") is not None and allowed > 0:
                used = int(st.get("bytes_total", 0))
                remain = max(0, int(cfg["stdin_max_bytes"]) - used)
                if remain <= 0:
                    allowed = 0
                    reason = reason or "stdin_cap"
                else:
                    if allowed > remain:
                        allowed = remain
                        reason = reason or "stdin_cap"
            # Update counters
            if allowed > 0:
                st["bytes_total"] = int(st.get("bytes_total", 0)) + int(allowed)
                import time as _time
                st["last_input"] = float(_time.time())
            return (int(allowed), reason)

    def get_stdin_idle_timeout(self, run_id: str) -> int | None:
        with self._lock:
            cfg = self._stdin_cfg.get(run_id) or {}
            val = cfg.get("stdin_idle_timeout_sec")
            return int(val) if val is not None else None

    def get_last_stdin_input_time(self, run_id: str) -> float | None:
        with self._lock:
            st = self._stdin_state.get(run_id) or {}
            return float(st.get("last_input")) if st.get("last_input") is not None else None

    # -----------------
    # Stdin data piping
    # -----------------
    def push_stdin(self, run_id: str, data: bytes) -> None:
        """Queue raw stdin bytes for a run.

        The WebSocket handler should call consume_stdin() to enforce caps before
        calling push_stdin(); this method simply enqueues the (possibly truncated)
        bytes for the runner-side pump to write to the process.
        """
        if not data:
            return
        with self._lock:
            try:
                import queue as _queue
                q = self._stdin_queues.get(run_id)
                if q is None:
                    q = _queue.Queue()
                    self._stdin_queues[run_id] = q
            except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                return
            try:
                q.put_nowait(bytes(data))
            except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                # Best-effort; drop on overflow
                pass

    def get_stdin_queue(self, run_id: str):
        """Return the thread-safe Queue for stdin bytes for a run (create if absent)."""
        with self._lock:
            import queue as _queue
            q = self._stdin_queues.get(run_id)
            if q is None:
                q = _queue.Queue()
                self._stdin_queues[run_id] = q
            return q

    def close(self, run_id: str) -> None:
        # Use publish_event so end-event deduplication applies
        self.publish_event(run_id, "end", {})
        # Cleanup all per-run state to prevent memory leaks
        self.cleanup_run(run_id)

    def get_log_bytes(self, run_id: str) -> int:
        """Return the total number of bytes published to stdout/stderr for a run."""
        with self._lock:
            return int(self._log_bytes.get(run_id, 0))

    def is_log_truncated(self, run_id: str) -> bool:
        """Return whether stdout/stderr streaming hit the configured byte cap."""
        with self._lock:
            return run_id in self._truncated

    def cleanup_run(self, run_id: str) -> None:
        """Remove all references for a run to avoid memory leaks.

        Thread-safe: acquires the internal lock before mutating any structures.
        Removes entries from queues, buffers, log counters, truncation/end markers,
        and sequence tracking.
        """
        with self._lock:
            # Remove any queue reference (the queue object may still be held by subscribers)
            self._queues.pop(run_id, None)
            # Drop buffered frames and counters
            self._buffers.pop(run_id, None)
            self._log_bytes.pop(run_id, None)
            # Clear truncation and end flags
            self._truncated.discard(run_id)
            self._ended.discard(run_id)
            # Clear sequence counter
            self._seq.pop(run_id, None)
            # Drop stdin queues and state
            with contextlib.suppress(_SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS):
                self._stdin_queues.pop(run_id, None)

    # -----------------
    # Redis fan-out (optional)
    # -----------------
    def _maybe_enable_redis_fanout(self) -> None:
        try:
            # Redis fan-out must be explicitly enabled for sandbox streams.
            toggle_env = str(os.getenv("SANDBOX_WS_REDIS_FANOUT") or "").strip().lower()
            if not is_truthy(toggle_env):
                return
            # Resolve URL
            url = os.getenv("SANDBOX_REDIS_URL") or os.getenv("REDIS_URL")
            if not url:
                # Try host/port/db
                try:
                    host = str(os.getenv("REDIS_HOST", "127.0.0.1"))
                    port = int(os.getenv("REDIS_PORT", "6379"))
                    db = int(os.getenv("REDIS_DB", "0"))
                    url = f"redis://{host}:{port}/{db}"
                except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                    url = None
            if not url:
                return
            try:
                client = create_sync_redis_client(
                    preferred_url=url,
                    context="sandbox_streams",
                    fallback_to_fake=False,
                    decode_responses=True,
                )
                self._redis_client = client
                self._redis_enabled = True
                # Start background subscriber
                th = threading.Thread(target=self._redis_listen_loop, name="sandbox-redis-fanout", daemon=True)
                th.start()
                self._redis_thread = th
                logger.debug("Sandbox WS Redis fan-out enabled")
            except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS as e:
                self._redis_enabled = False
                self._redis_client = None
                logger.debug(f"Sandbox WS Redis fan-out unavailable: {e}")
        except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
            # Never break hub init on redis issues
            self._redis_enabled = False
            self._redis_client = None

    def _redis_listen_loop(self) -> None:
        try:
            if not (self._redis_enabled and self._redis_client is not None):
                return
            pubsub = self._redis_client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(self._redis_channel)
            for msg in pubsub.listen():
                try:
                    if msg is None:
                        continue
                    if msg.get("type") != "message":
                        continue
                    data = msg.get("data")
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8", "ignore")
                    payload = json.loads(data)
                    if payload.get("origin") == self._instance_id:
                        continue
                    run_id = payload.get("run_id")
                    frame = payload.get("frame")
                    if isinstance(run_id, str) and isinstance(frame, dict):
                        self._publish_local(run_id, frame)
                except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS:
                    # Keep listening on individual message errors
                    continue
        except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"redis listen loop ended: {e}")

    def get_redis_status(self) -> dict:
        return {
            "enabled": bool(self._redis_enabled),
            "channel": self._redis_channel,
            "connected": bool(self._redis_enabled and self._redis_client is not None),
        }

    def ping_redis(self) -> dict:
        """Ping Redis and return timing information.

        Returns a dict with keys: ok (bool), ms (float|None), error (str|None).
        If Redis is disabled or no client is present, returns ok=False.
        """
        if not (self._redis_enabled and self._redis_client is not None):
            return {"ok": False, "ms": None, "error": None}
        try:
            t0 = time.perf_counter()
            self._redis_client.ping()
            dt = (time.perf_counter() - t0) * 1000.0
            return {"ok": True, "ms": float(dt), "error": None}
        except _SANDBOX_STREAMS_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover (network flake)
            return {"ok": False, "ms": None, "error": str(e)}


_HUB = RunStreamHub()


def get_hub() -> RunStreamHub:
    return _HUB
