import asyncio
import os
import time
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.LLM_Calls.streaming import iter_sse_lines_requests
from tldw_Server_API.app.core.Streaming import streams
from tldw_Server_API.app.core.Streaming.streams import SSEStream, WebSocketStream
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):

        for ln in self._lines:
            # Simulate raw provider bytes when decode_unicode=False
            yield ln


@contextmanager
def env_override(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def test_iter_normalization_passthru_off_drops_control_lines():


    lines = [
        b"event: chunk",
        b"id: 42",
        b"retry: 1000",
        b": heartbeat",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
        b"data: [DONE]",
    ]
    resp = _FakeResponse(lines)
    out = list(iter_sse_lines_requests(resp, decode_unicode=False, provider="test"))
    # Control lines dropped; data line preserved; DONE suppressed
    assert len(out) == 1
    assert out[0].startswith("data:")


def test_iter_normalization_passthru_on_preserves_control_lines():


    lines = [
        b"event: chunk",
        b"id: 42",
        b"retry: 1000",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
        b"data: [DONE]",
    ]
    resp = _FakeResponse(lines)
    with env_override("STREAM_PROVIDER_CONTROL_PASSTHRU", "1"):
        out = list(iter_sse_lines_requests(resp, decode_unicode=False, provider="test"))
    # Control lines preserved along with data line; DONE suppressed
    assert any(x.startswith("event:") for x in out)
    assert any(x.startswith("id:") for x in out)
    assert any(x.startswith("retry:") for x in out)
    assert any(x.startswith("data:") and "[DONE]" not in x for x in out)


def test_iter_normalization_control_filter_maps_and_drops():


    lines = [
        b"event: original",
        b"id: 99",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}",
    ]
    resp = _FakeResponse(lines)

    def filt(name: str, value: str):
        if name.lower() == "event":
            return ("event", "renamed")
        if name.lower() == "id":
            return None  # drop id
        return (name, value)

    out = list(
        iter_sse_lines_requests(
            resp,
            decode_unicode=False,
            provider="test",
            provider_control_passthru=True,
            control_filter=filt,
        )
    )
    assert any(x.startswith("event: renamed") for x in out)
    assert not any(x.startswith("id:") for x in out)
    assert any(x.startswith("data:") for x in out)


@pytest.mark.asyncio
async def test_sse_stream_idle_timeout_triggers_error_then_done():
    # Make heartbeat longer than idle to avoid masking
    stream = SSEStream(heartbeat_interval_s=1.0, idle_timeout_s=0.2)

    async def collect_first_n(n):
        out = []
        async for ln in stream.iter_sse():
            out.append(ln)
            if len(out) >= n:
                break
        return out

    # Expect two lines: error + DONE
    t0 = time.monotonic()
    out = await asyncio.wait_for(collect_first_n(2), timeout=2.0)
    assert any("\"idle_timeout\"" in x for x in out)
    assert any(x.strip().lower() == "data: [done]" for x in out)
    assert (time.monotonic() - t0) >= 0.2


@pytest.mark.asyncio
async def test_sse_stream_max_duration_triggers_error_then_done():
    # Disable heartbeat by making it longer than max duration
    stream = SSEStream(heartbeat_interval_s=10.0, max_duration_s=0.2)

    async def collect_first_n(n):
        out = []
        async for ln in stream.iter_sse():
            out.append(ln)
            if len(out) >= n:
                break
        return out

    out = await asyncio.wait_for(collect_first_n(2), timeout=2.0)
    assert any("\"max_duration_exceeded\"" in x for x in out)
    assert any(x.strip().lower() == "data: [done]" for x in out)


@pytest.mark.asyncio
async def test_sse_stream_send_json_and_raw_line():
    stream = SSEStream(heartbeat_interval_s=0.5)

    async def producer():
        await stream.send_json({"hello": "world"})
        await stream.send_raw_sse_line("event: summary")
        await stream.send_json({"summary": True})
        await stream.done()

    async def collect():
        out = []
        async def gen():
            async for ln in stream.iter_sse():
                out.append(ln)
        await asyncio.gather(gen(), producer())
        return out

    out = await asyncio.wait_for(collect(), timeout=2.0)
    assert any("data: {\"hello\": \"world\"}" in x for x in out)
    assert any(x.startswith("event: summary") for x in out)
    assert any("data: {\"summary\": true}" in x.lower() for x in out)
    assert out[-1].strip().lower() == "data: [done]"


@pytest.mark.asyncio
async def test_sse_stream_send_event_without_data_dispatches_blank():
    stream = SSEStream(heartbeat_interval_s=10.0)  # suppress heartbeats

    async def producer():
        await stream.send_event("summary")
        await stream.done()

    lines = []

    async def consumer():
        async for ln in stream.iter_sse():
            lines.append(ln)

    await asyncio.gather(producer(), consumer())
    # Expect an event line followed by a blank line then DONE at end
    assert any(x.startswith("event: summary") for x in lines)
    # Find the event line index and assert next line is blank
    idx = next(i for i, v in enumerate(lines) if v.startswith("event: summary"))
    assert lines[idx + 1] == "\n"
    assert lines[-1].strip().lower() == "data: [done]"


@pytest.mark.asyncio
async def test_sse_stream_send_event_with_data_is_single_frame():
    stream = SSEStream(heartbeat_interval_s=10.0)  # suppress heartbeats

    async def producer():
        await stream.send_event("job", {"ok": True})
        await stream.done()

    out = []

    async def consumer():
        async for ln in stream.iter_sse():
            out.append(ln)

    await asyncio.gather(producer(), consumer())

    idx = next(i for i, v in enumerate(out) if v.startswith("event: job"))
    assert not out[idx].endswith("\n\n")
    assert out[idx + 1].startswith("data: ")
    assert "\"ok\": true" in out[idx + 1].lower()
    assert out[idx + 1].endswith("\n\n")
    assert out[idx + 1] != "\n"


@pytest.mark.asyncio
async def test_sse_stream_send_event_with_id_and_data_compose():
    stream = SSEStream(heartbeat_interval_s=10.0)  # suppress heartbeats

    async def producer():
        await stream.send_event("job", {"ok": True}, event_id="42")
        await stream.done()

    out = []

    async def consumer():
        async for ln in stream.iter_sse():
            out.append(ln)

    await asyncio.gather(producer(), consumer())

    idx = next(i for i, v in enumerate(out) if v.startswith("id: 42"))
    assert not out[idx].endswith("\n\n")
    assert out[idx + 1].startswith("event: job")
    assert not out[idx + 1].endswith("\n\n")
    assert out[idx + 2].startswith("data: ")
    assert out[idx + 2].endswith("\n\n")


@pytest.mark.asyncio
async def test_sse_stream_idle_timeout_env_vars(monkeypatch):
    # Drive idle timeout via env; ensure heartbeat longer than idle
    monkeypatch.setenv("STREAM_IDLE_TIMEOUT_S", "0.2")
    stream = SSEStream(heartbeat_interval_s=1.0)

    async def collect_first_n(n):
        out = []
        async for ln in stream.iter_sse():
            out.append(ln)
            if len(out) >= n:
                break
        return out

    out = await asyncio.wait_for(collect_first_n(2), timeout=2.0)
    assert any("\"idle_timeout\"" in x for x in out)
    assert any(x.strip().lower() == "data: [done]" for x in out)


@pytest.mark.asyncio
async def test_sse_stream_max_duration_env_vars(monkeypatch):
    # Drive max duration via env and suppress heartbeat
    monkeypatch.setenv("STREAM_MAX_DURATION_S", "0.2")
    stream = SSEStream(heartbeat_interval_s=10.0)

    async def collect_first_n(n):
        out = []
        async for ln in stream.iter_sse():
            out.append(ln)
            if len(out) >= n:
                break
        return out

    out = await asyncio.wait_for(collect_first_n(2), timeout=2.0)
    assert any("\"max_duration_exceeded\"" in x for x in out)
    assert any(x.strip().lower() == "data: [done]" for x in out)


@pytest.mark.asyncio
async def test_sse_stream_comment_heartbeat_mode(monkeypatch):
    # Force comment-mode heartbeats with a short interval
    monkeypatch.setenv("STREAM_HEARTBEAT_MODE", "comment")
    stream = SSEStream(heartbeat_interval_s=0.05)

    async def collect_first_heartbeat():
        async for ln in stream.iter_sse():
            if ln.startswith(":"):
                return ln

    hb = await asyncio.wait_for(collect_first_heartbeat(), timeout=1.0)
    assert hb.startswith(":")


def test_stream_metrics_registration_failure_debug_log_is_sanitized(monkeypatch):
    class FailingRegistry:
        def register_metric(self, metric_definition):
            raise RuntimeError("secret metric path /private/metrics.db")

    class FakeLogger:
        def __init__(self):
            self.debug_messages = []

        def debug(self, message):
            self.debug_messages.append(message)

    fake_logger = FakeLogger()
    monkeypatch.setattr(streams, "_STREAM_METRICS_REGISTERED", False)
    monkeypatch.setattr(streams, "get_metrics_registry", lambda: FailingRegistry())
    monkeypatch.setattr(streams, "logger", fake_logger)

    streams._ensure_stream_metrics_registered()

    assert fake_logger.debug_messages == ["Stream metrics registration failed or already registered"]
    assert "secret metric path" not in fake_logger.debug_messages[0]
    assert "/private/metrics.db" not in fake_logger.debug_messages[0]
    assert streams._STREAM_METRICS_REGISTERED is False


def test_parse_float_env_invalid_value_debug_log_is_sanitized(monkeypatch):
    class FakeLogger:
        def __init__(self):
            self.debug_messages = []

        def debug(self, message):
            self.debug_messages.append(message)

    fake_logger = FakeLogger()
    monkeypatch.setenv("STREAM_IDLE_TIMEOUT_S", "super-secret-token")
    monkeypatch.setattr(streams, "logger", fake_logger)

    assert streams._parse_float_env("STREAM_IDLE_TIMEOUT_S") is None
    assert fake_logger.debug_messages == ["Invalid float in stream env"]
    assert "STREAM_IDLE_TIMEOUT_S" not in fake_logger.debug_messages[0]
    assert "super-secret-token" not in fake_logger.debug_messages[0]


class _StubWebSocket:
    def __init__(self):
        self.sent = []
        self.accepted = False
        self.closed = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code: int = 1000):
        self.closed = True
        self.close_code = code


@pytest.mark.asyncio
async def test_ws_stream_send_done_and_ping_metrics():
    ws = _StubWebSocket()
    reg = get_metrics_registry()
    stream = WebSocketStream(ws, heartbeat_interval_s=0.05, labels={"component": "test", "endpoint": "ws"})
    await stream.start()
    await stream.send_json({"hello": "world"})
    await stream.done()
    # Allow a couple of pings
    await asyncio.sleep(0.12)
    await stream.stop()

    assert ws.accepted is True
    assert any(msg.get("type") == "done" for msg in ws.sent)

    # Metrics assertions
    ws_latency_stats = reg.get_metric_stats("ws_send_latency_ms")
    assert ws_latency_stats.get("count", 0) >= 2
    pings_stats = reg.get_metric_stats("ws_pings_total")
    assert pings_stats.get("count", 0) >= 1
    ping_fail_stats = reg.get_metric_stats("ws_ping_failures_total")
    assert ping_fail_stats.get("count", 0) == 0


@pytest.mark.asyncio
async def test_ws_stream_error_compat_and_close_code():
    ws = _StubWebSocket()
    stream = WebSocketStream(ws, heartbeat_interval_s=0, compat_error_type=True)
    await stream.start()
    await stream.error("quota_exceeded", "limit reached", data={"limit": 5})
    assert ws.closed is True
    assert ws.close_code == 1008
    # Last sent payload is error
    assert ws.sent[-1]["type"] == "error"
    assert ws.sent[-1]["code"] == "quota_exceeded"
    assert ws.sent[-1]["error_type"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_ws_stream_idle_timeout_counter_and_close():
    ws = _StubWebSocket()
    reg = get_metrics_registry()
    # Disable pings, set short idle timeout
    stream = WebSocketStream(ws, heartbeat_interval_s=0, idle_timeout_s=0.1)
    await stream.start()
    # Wait for idle loop to trigger
    await asyncio.sleep(0.2)
    assert ws.closed is True
    assert ws.close_code == 1001
    idle_stats = reg.get_metric_stats("ws_idle_timeouts_total")
    assert idle_stats.get("count", 0) >= 1


@pytest.mark.asyncio
async def test_ws_stream_idle_edge_and_mark_activity():
    # Disable pings; set an idle timeout and simulate client activity before threshold
    ws = _StubWebSocket()
    stream = WebSocketStream(ws, heartbeat_interval_s=0, idle_timeout_s=0.12)
    await stream.start()
    # Nearly hit the threshold, then mark activity
    await asyncio.sleep(0.06)
    stream.mark_activity()
    await asyncio.sleep(0.07)
    # Should not be closed yet
    assert ws.closed is False
    # Now let it cross the threshold
    await asyncio.sleep(0.12)
    assert ws.closed is True
    assert ws.close_code == 1001


@pytest.mark.asyncio
async def test_ws_stream_close_code_transport_and_done_without_close():
    ws = _StubWebSocket()
    # close_on_done=False should not close on done()
    stream = WebSocketStream(ws, heartbeat_interval_s=0, close_on_done=False)
    await stream.start()
    await stream.done()
    assert ws.closed is False
    assert any(msg.get("type") == "done" for msg in ws.sent)

    # transport_error should map to 1011
    await stream.error("transport_error", "network failure")
    assert ws.closed is True
    assert ws.close_code == 1011


@pytest.mark.asyncio
async def test_sse_metrics_enqueue_to_yield_and_high_watermark():
    reg = get_metrics_registry()
    stream = SSEStream(heartbeat_interval_s=10.0)  # avoid heartbeat noise

    async def producer():
        # Enqueue a couple of lines
        await stream.send_json({"a": 1})
        await stream.send_json({"b": 2})
        await stream.done()

    out = []

    async def consumer():
        async for ln in stream.iter_sse():
            out.append(ln)

    await asyncio.gather(producer(), consumer())
    assert len(out) >= 3

    e2y_stats = reg.get_metric_stats("sse_enqueue_to_yield_ms")
    assert e2y_stats.get("count", 0) >= 3
    hwm_stats = reg.get_metric_stats("sse_queue_high_watermark")
    assert hwm_stats.get("latest", 0) >= 1


@pytest.mark.asyncio
async def test_sse_backpressure_heartbeats_under_load():
    """Under heavy producer pressure, ensure a heartbeat is eventually emitted once idle.

    We stress the queue with a small max size so producer backpressure engages, then
    verify that a heartbeat (comment mode) appears shortly after the producer stops.
    """
    stream = SSEStream(heartbeat_interval_s=0.05, heartbeat_mode="comment", queue_maxsize=5)

    producer_done = asyncio.Event()
    heartbeat_seen = asyncio.Event()

    async def producer():
        for i in range(100):
            await stream.send_json({"i": i})
        producer_done.set()

    async def closer():
        await producer_done.wait()
        try:
            await asyncio.wait_for(
                heartbeat_seen.wait(),
                timeout=stream.heartbeat_interval_s + 0.1,
            )
        except TimeoutError:
            pass
        await stream.done(force=True)

    async def consumer():
        async for ln in stream.iter_sse():
            if ln.startswith(":"):
                heartbeat_seen.set()
            # If producer is done and queue drains, the next emission should be a heartbeat within interval
            if producer_done.is_set():
                # Keep looping until heartbeat is encountered
                continue

    # Run both concurrently with a timeout guard
    await asyncio.wait_for(asyncio.gather(producer(), consumer(), closer()), timeout=2.0)
    assert heartbeat_seen.is_set(), "heartbeat not observed under backpressure after producer finished"


@pytest.mark.asyncio
async def test_sse_event_without_data_emits_blank_line():
    """send_event without data should produce an event line and a blank line."""
    stream = SSEStream(heartbeat_interval_s=10.0)  # avoid heartbeat noise

    async def producer():
        await stream.send_event("summary")  # no data
        await stream.done()

    out = []

    async def consumer():
        async for ln in stream.iter_sse():
            out.append(ln)
            if len(out) >= 3:
                break

    await asyncio.gather(producer(), consumer())

    # Expect: event line, a blank line (separator), and DONE
    assert out[0].startswith("event: summary")
    # Second line should be exactly a blank line (single newline) or a double-terminated event line
    assert out[1] in {"\n", "\r\n"}
    assert out[-1].strip().lower() == "data: [done]"
