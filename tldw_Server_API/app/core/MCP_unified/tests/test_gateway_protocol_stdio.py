"""Portable byte-stdio tests for the strict MCP protocol connection."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import json
import os
import re
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mcp_unified.gateway.protocol_stdio as protocol_stdio
import mcp_unified.gateway.protocol_stdio_adapters as stdio_adapters
import pytest
import yaml
from mcp_unified.gateway import (
    GatewayAsyncByteReader,
    GatewayAsyncByteWriter,
    GatewayLimits,
    GatewayProtocolStdioServer,
    GatewayStdioServer,
    handle_stdio_line,
    serve_stdio,
)
from mcp_unified.gateway.protocol_stdio_adapters import (
    _open_native_stdio,
    _open_threaded_stdio,
    _OwnedStdioAdapters,
)

pytestmark = pytest.mark.unit

_MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}
_REPO_ROOT = Path(__file__).resolve().parents[5]


def _request(
    request_id: str | int,
    method: str = "ping",
    params: dict[str, Any] | None = None,
) -> bytes:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**(params or {}), "_meta": _MODERN_META},
    }
    return json.dumps(payload, separators=(",", ":")).encode() + b"\n"


def _cancel(request_id: str | int) -> bytes:
    payload = {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": request_id},
    }
    return json.dumps(payload, separators=(",", ":")).encode() + b"\n"


class _CoreRuntime:
    """Small core-only runtime whose resource read can be held for races."""

    name = "stdio-runtime"
    version = "1.0"

    def __init__(self) -> None:
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()
        self.read_release.set()
        self.read_cancelled = asyncio.Event()
        self.contexts: list[Any] = []
        self.active_reads = 0

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> Any:
        return {"ok": True}

    async def list_resources(self, context: Any) -> list[dict[str, Any]]:
        return []

    async def read_resource(self, uri: str, context: Any) -> dict[str, Any]:
        self.contexts.append(context)
        self.active_reads += 1
        self.read_started.set()
        try:
            await self.read_release.wait()
            return {"contents": [{"uri": uri, "text": "resource"}]}
        except asyncio.CancelledError:
            self.read_cancelled.set()
            raise
        finally:
            self.active_reads -= 1

    async def list_prompts(self, context: Any) -> list[dict[str, Any]]:
        return []

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        return {"messages": []}

    def block_reads(self) -> None:
        self.read_started.clear()
        self.read_release.clear()


class _CompatRuntime(_CoreRuntime):
    """Legacy protocol shape retained for characterization tests."""

    async def list_modules(self, context: Any) -> list[dict[str, Any]]:
        return []

    async def get_modules_health(self, context: Any) -> dict[str, Any]:
        return {}


class _QueueReader:
    """Caller-owned async reader with deterministic admission control."""

    def __init__(self, *lines: bytes | BaseException) -> None:
        self._queue: asyncio.Queue[bytes | BaseException] = asyncio.Queue()
        for line in lines:
            self._queue.put_nowait(line)
        self.read_calls = 0
        self.closed = False

    async def readline(self) -> bytes:
        self.read_calls += 1
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def feed(self, line: bytes) -> None:
        self._queue.put_nowait(line)


class _MemoryWriter:
    """Caller-owned byte writer that records complete write calls."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.drain_calls = 0
        self.active_drains = 0
        self.max_active_drains = 0
        self.closed = False
        self.fail_write = False

    def write(self, data: bytes) -> None:
        if self.fail_write:
            raise OSError("write failed")
        assert isinstance(data, bytes)
        self.chunks.append(data)

    async def drain(self) -> None:
        self.drain_calls += 1
        self.active_drains += 1
        self.max_active_drains = max(self.max_active_drains, self.active_drains)
        try:
            await asyncio.sleep(0)
        finally:
            self.active_drains -= 1

    @property
    def values(self) -> list[Any]:
        return [json.loads(chunk) for chunk in self.chunks]


class _BlockingWriter(_MemoryWriter):
    """Writer whose drain exposes the connection's final write race."""

    def __init__(self, *, resist_cancellation: bool = False) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.resist_cancellation = resist_cancellation

    async def drain(self) -> None:
        self.drain_calls += 1
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.resist_cancellation:
                raise
            await self.release.wait()


class _CountingTextReader(io.StringIO):
    def __init__(self, value: str) -> None:
        super().__init__(value)
        self.read_calls = 0

    def readline(self, *args: Any, **kwargs: Any) -> str:
        self.read_calls += 1
        return super().readline(*args, **kwargs)


async def _eventually(predicate: Callable[[], bool], *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition did not become true")


def _ids(writer: _MemoryWriter) -> list[str | int | None]:
    return [value.get("id") for value in writer.values]


def test_strict_stdio_public_contract_is_exact() -> None:
    """Removing an export or widening serve_stdio must break consumer imports."""

    assert inspect.isclass(GatewayAsyncByteReader)
    assert inspect.isclass(GatewayAsyncByteWriter)
    assert inspect.isclass(GatewayProtocolStdioServer)
    assert str(inspect.signature(serve_stdio)) == (
        "(runtime: 'GatewayCoreRuntime', *, "
        "input_stream: 'GatewayAsyncByteReader | None' = None, "
        "output_stream: 'GatewayAsyncByteWriter | None' = None, "
        "limits: 'GatewayLimits' = GatewayLimits(max_input_line_bytes=1048576, "
        "max_output_line_bytes=1048576, max_result_bytes=786432, max_json_depth=64, "
        "max_in_flight=16, default_catalog_page_size=50, max_catalog_page_size=100, "
        "max_catalog_items=10000, max_batch_items=100, max_requests_per_minute=600, "
        "request_burst=32, max_schema_bytes=262144, max_schema_depth=32, "
        "max_schema_subschemas=1024, max_schema_refs=256, max_schema_pattern_chars=4096, "
        "max_schema_validation_processes=4, schema_validation_timeout_seconds=1.0, "
        "graceful_shutdown_timeout_seconds=5.0), "
        "metadata: 'Mapping[str, Any] | None' = None) -> 'int'"
    )


def test_text_streams_are_rejected_before_input_is_read() -> None:
    """Accepting text I/O would permit locale decoding before byte limits apply."""

    reader = _CountingTextReader("{}\n")
    with pytest.raises(ValueError, match="binary"):
        GatewayProtocolStdioServer(
            _CoreRuntime(),
            input_stream=reader,  # type: ignore[arg-type]
            output_stream=_MemoryWriter(),
        )

    assert reader.read_calls == 0


@pytest.mark.parametrize(
    ("reader", "writer"),
    [
        (object(), _MemoryWriter()),
        (_QueueReader(b""), object()),
        (_QueueReader(b""), io.BytesIO()),
    ],
)
def test_invalid_injected_stream_shapes_fail_locally(reader: Any, writer: Any) -> None:
    """Missing async byte methods must fail before a serving task is created."""

    with pytest.raises(ValueError):
        GatewayProtocolStdioServer(
            _CoreRuntime(),
            input_stream=reader,
            output_stream=writer,
        )


def test_stdio_validation_helpers_are_documented() -> None:
    """Private transport validators still require maintainable API documentation."""

    assert inspect.getdoc(protocol_stdio._validate_reader)  # nosec B101
    assert inspect.getdoc(protocol_stdio._validate_writer)  # nosec B101


def test_stderr_diagnostic_does_not_swallow_process_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A best-effort diagnostic must not consume KeyboardInterrupt or SystemExit."""

    class _InterruptingStderr:
        def write(self, message: str) -> int:
            del message
            raise KeyboardInterrupt

    monkeypatch.setattr(protocol_stdio.sys, "stderr", _InterruptingStderr())

    with pytest.raises(KeyboardInterrupt):
        protocol_stdio._write_stderr_diagnostic("fixed diagnostic")


@pytest.mark.asyncio
async def test_injected_streams_emit_one_line_and_remain_caller_owned() -> None:
    """Closing or coalescing injected streams would violate embedder ownership."""

    reader = _QueueReader(b"\n", b"  \t\r\n", _request("ping"), b"")
    writer = _MemoryWriter()

    result = await serve_stdio(
        _CoreRuntime(),
        input_stream=reader,
        output_stream=writer,
    )

    assert result == 0
    assert _ids(writer) == ["ping"]
    assert all(chunk.endswith(b"\n") and not chunk.endswith(b"\n\n") for chunk in writer.chunks)
    assert all(len(chunk.splitlines()) == 1 for chunk in writer.chunks)
    assert writer.drain_calls == 2
    assert reader.closed is False
    assert writer.closed is False


@pytest.mark.asyncio
async def test_protocol_input_failures_are_safe_and_keep_exit_zero() -> None:
    """Malformed, incomplete, and oversized lines are peer errors, not fatal transport errors."""

    limits = GatewayLimits(max_input_line_bytes=256)
    reader = _QueueReader(b"not-json\n", b'{"jsonrpc":"2.0"', b"x" * 257 + b"\n", b"")
    writer = _MemoryWriter()

    result = await serve_stdio(
        _CoreRuntime(),
        input_stream=reader,
        output_stream=writer,
        limits=limits,
    )

    assert result == 0
    assert [value["error"]["code"] for value in writer.values] == [-32700, -32700, -32700]
    assert all(value["id"] is None for value in writer.values)
    assert all(len(chunk) <= limits.max_output_line_bytes for chunk in writer.chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(("output_limit", "expected_chunks"), [(79, 1), (78, 0)])
async def test_output_overflow_uses_task3_minimal_error_boundary(
    output_limit: int,
    expected_chunks: int,
) -> None:
    """The stdio layer must not truncate or invent a smaller Task 3 overflow result."""

    limits = GatewayLimits(
        max_output_line_bytes=output_limit,
        max_result_bytes=output_limit,
    )
    writer = _MemoryWriter()

    result = await serve_stdio(
        _CoreRuntime(),
        input_stream=_QueueReader(_request("bounded"), b""),
        output_stream=writer,
        limits=limits,
    )

    assert result == 0
    assert len(writer.chunks) == expected_chunks
    if expected_chunks:
        assert writer.chunks == [b'{"error":{"code":-32603,"message":"Internal error"},"id":null,"jsonrpc":"2.0"}\n']
        assert len(writer.chunks[0]) == 79


@pytest.mark.asyncio
async def test_reader_admits_ping_while_runtime_request_is_blocked() -> None:
    """Awaiting each runtime request in the read loop would prevent cancellation and concurrency."""

    runtime = _CoreRuntime()
    runtime.block_reads()
    reader = _QueueReader()
    writer = _MemoryWriter()
    task = asyncio.create_task(serve_stdio(runtime, input_stream=reader, output_stream=writer))

    reader.feed(_request("slow", "resources/read", {"uri": "file:///slow"}))
    await runtime.read_started.wait()
    reader.feed(_request("fast"))
    await _eventually(lambda: "fast" in _ids(writer))

    assert runtime.active_reads == 1
    runtime.read_release.set()
    reader.feed(b"")
    assert await task == 0
    assert set(_ids(writer)) == {"slow", "fast"}


@pytest.mark.asyncio
async def test_concurrent_responses_never_interleave_writer_drains() -> None:
    """Dropping serialized output would permit concurrent drain calls and corrupt framing."""

    writer = _MemoryWriter()
    reader = _QueueReader(*(_request(index) for index in range(8)))
    task = asyncio.create_task(serve_stdio(_CoreRuntime(), input_stream=reader, output_stream=writer))
    await _eventually(lambda: len(writer.chunks) == 8)
    reader.feed(b"")

    assert await task == 0

    assert writer.max_active_drains == 1
    assert sorted(_ids(writer)) == list(range(8))
    assert all(json.loads(chunk) for chunk in writer.chunks)


@pytest.mark.asyncio
async def test_cancellation_at_writer_race_suppresses_late_result() -> None:
    """A cancelled request queued behind the writer lock must never emit its finished result."""

    runtime = _CoreRuntime()
    runtime.block_reads()
    reader = _QueueReader()
    writer = _BlockingWriter()
    task = asyncio.create_task(serve_stdio(runtime, input_stream=reader, output_stream=writer))

    reader.feed(_request("slow", "resources/read", {"uri": "file:///slow"}))
    await runtime.read_started.wait()
    reader.feed(_request("lock-holder"))
    await writer.entered.wait()
    runtime.read_release.set()
    reader.feed(_cancel("slow"))
    await _eventually(lambda: bool(runtime.contexts[0].cancellation.cancelled))
    writer.release.set()
    reader.feed(b"")

    assert await task == 0
    assert _ids(writer) == ["lock-holder"]


@pytest.mark.asyncio
async def test_eof_cancels_and_reaps_admitted_work() -> None:
    """Returning at EOF with live request tasks would leak runtime and validation work."""

    runtime = _CoreRuntime()
    runtime.block_reads()
    reader = _QueueReader()
    task = asyncio.create_task(serve_stdio(runtime, input_stream=reader, output_stream=_MemoryWriter()))

    reader.feed(_request("slow", "resources/read", {"uri": "file:///slow"}))
    await runtime.read_started.wait()
    reader.feed(b"")

    assert await task == 0
    assert runtime.read_cancelled.is_set()
    assert runtime.active_reads == 0


@pytest.mark.asyncio
async def test_serving_task_cancellation_cleans_then_reraises() -> None:
    """Swallowing serving cancellation would make process supervisors misclassify shutdown."""

    runtime = _CoreRuntime()
    runtime.block_reads()
    reader = _QueueReader()
    task = asyncio.create_task(serve_stdio(runtime, input_stream=reader, output_stream=_MemoryWriter()))
    reader.feed(_request("slow", "resources/read", {"uri": "file:///slow"}))
    await runtime.read_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runtime.read_cancelled.is_set()
    assert runtime.active_reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_side", ["read", "write"])
async def test_fatal_transport_failures_return_one(failure_side: str) -> None:
    """Transport loss must be distinguishable from peer-level protocol errors."""

    writer = _MemoryWriter()
    if failure_side == "read":
        reader = _QueueReader(OSError("read failed"))
    else:
        reader = _QueueReader(_request("ping"), b"")
        writer.fail_write = True

    assert await serve_stdio(_CoreRuntime(), input_stream=reader, output_stream=writer) == 1


@pytest.mark.asyncio
async def test_writer_drain_shutdown_is_bounded() -> None:
    """A drain that ignores cancellation must not hold server shutdown indefinitely."""

    limits = GatewayLimits(graceful_shutdown_timeout_seconds=0.05)
    reader = _QueueReader()
    writer = _BlockingWriter(resist_cancellation=True)
    task = asyncio.create_task(
        serve_stdio(
            _CoreRuntime(),
            input_stream=reader,
            output_stream=writer,
            limits=limits,
        )
    )
    reader.feed(_request("ping"))
    await writer.entered.wait()
    reader.feed(b"")

    try:
        assert await asyncio.wait_for(task, timeout=0.5) == 1
    finally:
        writer.release.set()
        await asyncio.sleep(0)


async def _pipe_server(
    opener: Callable[[Any, Any, GatewayLimits], Awaitable[_OwnedStdioAdapters]],
    input_bytes: bytes,
    *,
    limits: GatewayLimits = GatewayLimits(),
) -> tuple[int, bytes, _OwnedStdioAdapters, list[threading.Thread]]:
    input_read_fd, input_write_fd = os.pipe()
    output_read_fd, output_write_fd = os.pipe()
    os.write(input_write_fd, input_bytes)
    os.close(input_write_fd)
    input_file = os.fdopen(input_read_fd, "rb", buffering=0)
    output_file = os.fdopen(output_write_fd, "wb", buffering=0)
    captured: list[_OwnedStdioAdapters] = []
    owned_threads: list[threading.Thread] = []
    prior_threads = set(threading.enumerate())

    async def selector(
        direction: str,
        selected_limits: GatewayLimits,
    ) -> _OwnedStdioAdapters:
        del direction
        if not captured:
            adapters = await opener(input_file, output_file, selected_limits)
            captured.append(adapters)
            owned_threads.extend(thread for thread in threading.enumerate() if thread not in prior_threads)
            input_file.close()
            output_file.close()
        return captured[0]

    server = GatewayProtocolStdioServer(
        _CoreRuntime(),
        limits=limits,
        _adapter_selector=selector,
    )
    result = await server.serve()
    chunks: list[bytes] = []
    while chunk := os.read(output_read_fd, 65_536):
        chunks.append(chunk)
    os.close(output_read_fd)
    return result, b"".join(chunks), captured[0], owned_threads


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="native add_reader path is POSIX-only")
async def test_native_posix_adapter_serves_duplicated_binary_pipes() -> None:
    """Regressing to mocked async streams would leave the native pipe claim unproved."""

    result, output, adapters, owned_threads = await _pipe_server(
        _open_native_stdio,
        _request("native"),
    )

    assert result == 0
    assert json.loads(output)["id"] == "native"
    assert adapters.closed is True
    assert owned_threads == []


@pytest.mark.asyncio
async def test_threaded_fallback_bounds_reads_serializes_writes_and_joins() -> None:
    """An unbounded or shared-executor fallback would not satisfy the Windows contract."""

    limits = GatewayLimits(max_input_line_bytes=256)
    result, output, adapters, owned_threads = await _pipe_server(
        _open_threaded_stdio,
        b"x" * 300 + b"\n" + _request("fallback"),
        limits=limits,
    )
    values = [json.loads(line) for line in output.splitlines()]

    assert result == 0
    assert [value.get("id") for value in values] == [None, "fallback"]
    assert values[0]["error"]["code"] == -32700
    assert len(owned_threads) == 2
    assert all(thread.name.startswith("mcp-stdio-") for thread in owned_threads)
    assert all(not thread.is_alive() for thread in owned_threads)


@pytest.mark.asyncio
async def test_threaded_fallback_propagates_cancellation_and_joins_threads() -> None:
    """Cancellation must leave no fallback worker running after cooperative pipe release."""

    input_read_fd, input_write_fd = os.pipe()
    output_read_fd, output_write_fd = os.pipe()
    input_file = os.fdopen(input_read_fd, "rb", buffering=0)
    output_file = os.fdopen(output_write_fd, "wb", buffering=0)
    captured: list[_OwnedStdioAdapters] = []
    owned_threads: list[threading.Thread] = []
    prior_threads = set(threading.enumerate())

    async def selector(direction: str, limits: GatewayLimits) -> _OwnedStdioAdapters:
        del direction
        if not captured:
            adapters = await _open_threaded_stdio(input_file, output_file, limits)
            captured.append(adapters)
            owned_threads.extend(thread for thread in threading.enumerate() if thread not in prior_threads)
            input_file.close()
            output_file.close()
        return captured[0]

    task = asyncio.create_task(
        GatewayProtocolStdioServer(
            _CoreRuntime(),
            _adapter_selector=selector,
        ).serve()
    )
    await _eventually(lambda: bool(captured))
    task.cancel()
    os.close(input_write_fd)
    with pytest.raises(asyncio.CancelledError):
        await task

    os.close(output_read_fd)
    assert len(owned_threads) == 2
    assert all(not thread.is_alive() for thread in owned_threads)
    assert captured[0].closed is True


@pytest.mark.asyncio
async def test_threaded_fallback_fatal_write_reaps_workers_and_fds() -> None:
    """A broken output pipe must return fatal only after fallback ownership is reaped."""

    input_read_fd, input_write_fd = os.pipe()
    output_read_fd, output_write_fd = os.pipe()
    os.write(input_write_fd, _request("broken-output"))
    os.close(input_write_fd)
    os.close(output_read_fd)
    input_file = os.fdopen(input_read_fd, "rb", buffering=0)
    output_file = os.fdopen(output_write_fd, "wb", buffering=0)
    captured: list[_OwnedStdioAdapters] = []
    owned_threads: list[threading.Thread] = []
    prior_threads = set(threading.enumerate())

    async def selector(direction: str, limits: GatewayLimits) -> _OwnedStdioAdapters:
        del direction
        if not captured:
            adapters = await _open_threaded_stdio(input_file, output_file, limits)
            captured.append(adapters)
            owned_threads.extend(thread for thread in threading.enumerate() if thread not in prior_threads)
            input_file.close()
            output_file.close()
        return captured[0]

    result = await GatewayProtocolStdioServer(
        _CoreRuntime(),
        _adapter_selector=selector,
    ).serve()

    assert result == 1
    assert len(owned_threads) == 2
    assert all(not thread.is_alive() for thread in owned_threads)
    assert captured[0].closed is True


@pytest.mark.asyncio
async def test_default_adapter_preserves_globals_and_closes_only_duplicated_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process globals and caller descriptors must survive owned-adapter shutdown."""

    input_read_fd, input_write_fd = os.pipe()
    output_read_fd, output_write_fd = os.pipe()
    os.close(input_write_fd)
    input_file = os.fdopen(input_read_fd, "rb", buffering=0)
    output_file = os.fdopen(output_write_fd, "wb", buffering=0)
    fake_stdin = SimpleNamespace(buffer=input_file)
    fake_stdout = SimpleNamespace(buffer=output_file)
    fake_sys = SimpleNamespace(stdin=fake_stdin, stdout=fake_stdout)
    monkeypatch.setattr(stdio_adapters, "sys", fake_sys)

    input_adapters = await stdio_adapters._select_process_stdio("input", GatewayLimits())
    output_adapters = await stdio_adapters._select_process_stdio("output", GatewayLimits())
    adapters = _OwnedStdioAdapters(
        input_adapters.reader,
        output_adapters.writer,
        input_adapters.workers + output_adapters.workers,
    )
    owned_reader_fd = getattr(adapters.reader, "_fd", None)
    if owned_reader_fd is None:
        owned_reader_fd = adapters.reader._stream.fileno()  # type: ignore[attr-defined]
    owned_writer_fd = getattr(adapters.writer, "_fd", None)
    if owned_writer_fd is None:
        owned_writer_fd = adapters.writer._stream.fileno()  # type: ignore[attr-defined]
    await adapters.shutdown(0.5)

    assert fake_sys.stdin is fake_stdin
    assert fake_sys.stdout is fake_stdout
    assert input_file.closed is False
    assert output_file.closed is False
    os.fstat(input_file.fileno())
    os.fstat(output_file.fileno())
    with pytest.raises(OSError):
        os.fstat(owned_reader_fd)
    with pytest.raises(OSError):
        os.fstat(owned_writer_fd)

    input_file.close()
    output_file.close()
    os.close(output_read_fd)


@pytest.mark.asyncio
async def test_compatibility_stdio_keeps_independent_message_semantics() -> None:
    """Delegating compatibility helpers to strict lifecycle would reject standalone pings."""

    runtime = _CompatRuntime()
    server = GatewayStdioServer(runtime=runtime)

    first = await server.handle_line('{"jsonrpc":"2.0","method":"ping","id":1}\n')
    second = await handle_stdio_line(
        runtime,
        b'{"jsonrpc":"2.0","method":"ping","id":2}\n',
    )

    assert json.loads(first or "null") == {"jsonrpc": "2.0", "result": {"pong": True}, "id": 1}
    assert json.loads(second or "null") == {"jsonrpc": "2.0", "result": {"pong": True}, "id": 2}
    assert await server.handle_line(b" \t\r\n") is None


@pytest.mark.asyncio
async def test_compatibility_stdio_keeps_parse_error_and_notification_behavior() -> None:
    """Strict additions must not change legacy parse errors or notification suppression."""

    runtime = _CompatRuntime()
    malformed = await handle_stdio_line(runtime, b"not-json\n")
    notification = await handle_stdio_line(
        runtime,
        b'{"jsonrpc":"2.0","method":"ping"}\n',
    )

    assert json.loads(malformed or "null")["error"]["code"] == -32700
    assert notification is None


def test_rc_workflow_runs_installed_stdio_contracts_on_linux_and_windows() -> None:
    """The protected matrix must prove each supported Python artifact contract."""

    workflow = yaml.load(
        (_REPO_ROOT / ".github/workflows/mcp-unified-rc.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["portable-stdio"]
    assert job["name"] == "Portable stdio (${{ matrix.os }}, Python ${{ matrix.python }})"
    assert job["strategy"]["matrix"] == {
        "include": [
            {"os": "ubuntu-latest", "python": "3.10"},
            {"os": "ubuntu-latest", "python": "3.11"},
            {"os": "ubuntu-latest", "python": "3.12"},
            {"os": "ubuntu-latest", "python": "3.13"},
            {"os": "windows-latest", "python": "3.11"},
        ]
    }
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job["needs"] == "admission"
    assert workflow["jobs"]["admission"]["uses"] == "./.github/workflows/license-first-admission.yml"
    checkout = next(step for step in job["steps"] if step["name"] == "Checkout")
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkout["uses"])
    setup_python = next(step for step in job["steps"] if step["name"] == "Setup Python")
    assert setup_python["with"]["python-version"] == "${{ matrix.python }}"
    run_blocks = "\n".join(step["run"] for step in job["steps"] if "run" in step)
    assert 'python -m pip install "./apps/mcp-unified[dev]"' not in run_blocks
    assert "python -m pip install build twine setuptools wheel pytest pytest-asyncio bandit" in run_blocks
    assert '"jsonschema>=4.23,<5"' in run_blocks
    assert "python Helper_Scripts/mcp_unified_rc.py portable-gate" in run_blocks
    assert "tldw_Server_API/app/core/MCP_unified/tests/" not in run_blocks
    upload = next(step for step in job["steps"] if step["name"] == "Upload portable RC evidence")
    assert upload["if"] == "always()"
    assert re.fullmatch(r"actions/upload-artifact@[0-9a-f]{40}", upload["uses"])
    assert upload["with"]["name"] == "mcp-unified-portable-${{ matrix.os }}-py${{ matrix.python }}"
    assert upload["with"]["path"] == ".artifacts/mcp-unified-rc/**"
    assert upload["with"]["if-no-files-found"] == "error"
    assert (
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_artifact_consumer.py"
        in workflow["on"]["pull_request"]["paths"]
    )


def test_rc_workflow_and_license_admission_route_every_release_input() -> None:
    """No installed suite, fixture, or release helper change may bypass the RC."""

    workflow = yaml.load(
        (_REPO_ROOT / ".github/workflows/mcp-unified-rc.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    routes = json.loads(
        (_REPO_ROOT / ".github/license-first-paths.json").read_text(encoding="utf-8")
    )
    required_paths = [
        "apps/mcp-unified/**",
        "Makefile",
        "Helper_Scripts/mcp_unified_rc.py",
        "Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py",
        "Helper_Scripts/Testing-related/mcp_official_sdk_stdio_smoke.py",
        ".github/tests/test_mcp_unified_artifact_gate.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_contracts.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_validation.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_projection.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_connection.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_stdio.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_protocol_artifact_consumer.py",
        "tldw_Server_API/app/core/MCP_unified/tests/mcp_unified_artifact_test_utils.py",
        "tldw_Server_API/app/core/MCP_unified/tests/fixtures/mcp_protocol/**",
        "tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py",
        "tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py",
        ".github/workflows/mcp-unified-publish.yml",
        ".github/license-first-paths.json",
        ".github/workflows/mcp-unified-rc.yml",
    ]

    workflow_paths = workflow["on"]["pull_request"]["paths"]
    admission = routes["mcp-unified-rc.yml"]
    assert workflow_paths == required_paths
    assert admission == {"mode": "paths", "patterns": required_paths}


def _pipe_files() -> tuple[Any, Any, list[int]]:
    input_read_fd, input_write_fd = os.pipe()
    output_read_fd, output_write_fd = os.pipe()
    os.close(input_write_fd)
    os.close(output_read_fd)
    return (
        os.fdopen(input_read_fd, "rb", buffering=0),
        os.fdopen(output_write_fd, "wb", buffering=0),
        [input_read_fd, output_write_fd],
    )


def _fd_is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="native pipe registration is POSIX-only")
async def test_native_second_dup_failure_closes_first_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second native dup failure must not leak the first owned descriptor."""

    input_file, output_file, _ = _pipe_files()
    real_dup = os.dup
    duplicates: list[int] = []

    def failing_second_dup(fd: int) -> int:
        if duplicates:
            raise OSError("second dup failed")
        duplicate = real_dup(fd)
        duplicates.append(duplicate)
        return duplicate

    monkeypatch.setattr(stdio_adapters.os, "dup", failing_second_dup)
    try:
        with pytest.raises(OSError, match="second dup failed"):
            await _open_native_stdio(input_file, output_file, GatewayLimits())
        assert duplicates and not _fd_is_open(duplicates[0])
    finally:
        for duplicate in duplicates:
            if _fd_is_open(duplicate):
                os.close(duplicate)
        input_file.close()
        output_file.close()


@pytest.mark.asyncio
async def test_threaded_second_fdopen_failure_closes_every_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later fdopen failure must close both its raw fd and the first file object."""

    input_file, output_file, _ = _pipe_files()
    real_dup = os.dup
    real_fdopen = os.fdopen
    duplicates: list[int] = []
    fdopen_calls = 0

    def tracking_dup(fd: int) -> int:
        duplicate = real_dup(fd)
        duplicates.append(duplicate)
        return duplicate

    def failing_second_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        nonlocal fdopen_calls
        fdopen_calls += 1
        if fdopen_calls == 2:
            raise OSError("second fdopen failed")
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(stdio_adapters.os, "dup", tracking_dup)
    monkeypatch.setattr(stdio_adapters.os, "fdopen", failing_second_fdopen)
    try:
        with pytest.raises(OSError, match="second fdopen failed"):
            await _open_threaded_stdio(input_file, output_file, GatewayLimits())
        assert len(duplicates) == 2
        assert all(not _fd_is_open(fd) for fd in duplicates)
    finally:
        for duplicate in duplicates:
            if _fd_is_open(duplicate):
                os.close(duplicate)
        input_file.close()
        output_file.close()


@pytest.mark.asyncio
async def test_thread_start_failure_closes_prepared_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure to start the first dedicated worker must close all prepared files."""

    input_file, output_file, _ = _pipe_files()
    real_dup = os.dup
    real_fdopen = os.fdopen
    duplicates: list[int] = []
    prepared_files: list[Any] = []

    def tracking_dup(fd: int) -> int:
        duplicate = real_dup(fd)
        duplicates.append(duplicate)
        return duplicate

    def fail_start(thread: Any) -> None:
        raise RuntimeError("thread start failed")

    def tracking_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        prepared = real_fdopen(fd, *args, **kwargs)
        prepared_files.append(prepared)
        return prepared

    monkeypatch.setattr(stdio_adapters.os, "dup", tracking_dup)
    monkeypatch.setattr(stdio_adapters.os, "fdopen", tracking_fdopen)
    monkeypatch.setattr(stdio_adapters.threading.Thread, "start", fail_start)
    try:
        with pytest.raises(RuntimeError, match="thread start failed"):
            await _open_threaded_stdio(input_file, output_file, GatewayLimits())
        assert len(duplicates) == 2
        assert all(prepared.closed for prepared in prepared_files)
        assert all(not _fd_is_open(fd) for fd in duplicates)
    finally:
        for duplicate in duplicates:
            if _fd_is_open(duplicate):
                os.close(duplicate)
        input_file.close()
        output_file.close()


@pytest.mark.asyncio
async def test_second_thread_start_failure_stops_first_worker_and_closes_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure to start worker two must stop worker one and close both files."""

    input_file, output_file, _ = _pipe_files()
    real_fdopen = os.fdopen
    prepared_files: list[Any] = []
    workers: list[Any] = []

    class StartWorker:
        def __init__(self, name: str) -> None:
            self.name = name
            self.started = False
            self.shutdown_calls = 0
            workers.append(self)

        @property
        def alive(self) -> bool:
            return self.started

        def start(self) -> None:
            if len([worker for worker in workers if worker.started]) == 1:
                raise RuntimeError("second thread start failed")
            self.started = True

        async def shutdown(self, timeout: float) -> bool:
            self.shutdown_calls += 1
            self.started = False
            return True

    def tracking_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        prepared = real_fdopen(fd, *args, **kwargs)
        prepared_files.append(prepared)
        return prepared

    monkeypatch.setattr(stdio_adapters, "_BlockingIOWorker", StartWorker)
    monkeypatch.setattr(stdio_adapters.os, "fdopen", tracking_fdopen)
    try:
        with pytest.raises(RuntimeError, match="second thread start failed"):
            await _open_threaded_stdio(input_file, output_file, GatewayLimits())
        assert len(workers) == 2
        assert workers[0].shutdown_calls == 1
        assert workers[0].alive is False
        assert workers[1].shutdown_calls == 0
        assert all(prepared.closed for prepared in prepared_files)
    finally:
        for prepared in prepared_files:
            if not prepared.closed:
                prepared.close()
        input_file.close()
        output_file.close()


@pytest.mark.asyncio
async def test_second_worker_construction_failure_closes_prepared_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure to construct the second worker must close both prepared files."""

    input_file, output_file, _ = _pipe_files()
    real_fdopen = os.fdopen
    prepared_files: list[Any] = []
    worker_calls = 0

    class PreparedWorker:
        def __init__(self) -> None:
            self.start_calls = 0

        @property
        def alive(self) -> bool:
            return False

        def start(self) -> None:
            self.start_calls += 1

    prepared_worker = PreparedWorker()

    def failing_second_worker(name: str) -> Any:
        nonlocal worker_calls
        worker_calls += 1
        if worker_calls == 2:
            raise RuntimeError("second worker construction failed")
        return prepared_worker

    def tracking_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        prepared = real_fdopen(fd, *args, **kwargs)
        prepared_files.append(prepared)
        return prepared

    monkeypatch.setattr(stdio_adapters, "_BlockingIOWorker", failing_second_worker)
    monkeypatch.setattr(stdio_adapters.os, "fdopen", tracking_fdopen)
    try:
        with pytest.raises(RuntimeError, match="second worker construction failed"):
            await _open_threaded_stdio(input_file, output_file, GatewayLimits())
        assert len(prepared_files) == 2
        assert all(prepared.closed for prepared in prepared_files)
        assert prepared_worker.start_calls == 0
    finally:
        for prepared in prepared_files:
            if not prepared.closed:
                prepared.close()
        input_file.close()
        output_file.close()


class _CloseSpy:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.close_calls = 0

    async def readline(self) -> bytes:
        return b""

    def write(self, data: bytes) -> None:
        return None

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


class _ShutdownWorkerSpy:
    def __init__(self, *, stops: bool) -> None:
        self.stops = stops
        self.shutdown_calls = 0

    @property
    def alive(self) -> bool:
        return not self.stops

    async def shutdown(self, timeout: float) -> bool:
        self.shutdown_calls += 1
        return self.stops


@pytest.mark.asyncio
async def test_owned_shutdown_closes_writer_when_reader_close_raises() -> None:
    """One close exception must not prevent its sibling descriptor from closing."""

    reader = _CloseSpy(failure=OSError("reader close failed"))
    writer = _CloseSpy()
    adapters = _OwnedStdioAdapters(reader, writer)

    with pytest.raises(Exception, match="stdio adapter shutdown failed"):
        await adapters.shutdown(0.1)

    assert reader.close_calls == 1
    assert writer.close_calls == 1


@pytest.mark.asyncio
async def test_owned_shutdown_accounts_all_workers_and_discloses_residual_safely() -> None:
    """A residual worker must not hide sibling cleanup or leak payload data to diagnostics."""

    reader = _CloseSpy()
    writer = _CloseSpy()
    residual = _ShutdownWorkerSpy(stops=False)
    sibling = _ShutdownWorkerSpy(stops=True)
    diagnostics: list[str] = []
    adapters = _OwnedStdioAdapters(
        reader,
        writer,
        (residual, sibling),  # type: ignore[arg-type]
    )

    with pytest.raises(Exception, match="stdio adapter shutdown failed"):
        await adapters.shutdown(0.1, diagnostic=diagnostics.append)

    assert residual.shutdown_calls == 1
    assert sibling.shutdown_calls == 1
    assert reader.close_calls == 1
    assert writer.close_calls == 1
    assert diagnostics == [
        "MCP stdio shutdown incomplete: residual blocking I/O worker; process termination may be required"
    ]
    assert "reader close failed" not in diagnostics[0]


class _BlockingCleanupConnection:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.shutdown_calls = 0

    async def receive(self, payload: Any) -> None:
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.started.set()
        await self.release.wait()


class _CancellationAwareReader:
    def __init__(self, *, resist_cancellation: bool = False) -> None:
        self.resist_cancellation = resist_cancellation
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.active_reads = 0

    async def readline(self) -> bytes:
        self.active_reads += 1
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            if self.resist_cancellation:
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        continue
            raise
        finally:
            self.active_reads -= 1


@pytest.mark.asyncio
async def test_cancellation_inside_read_wait_accounts_both_child_tasks() -> None:
    """Cancelling the outer wait must reap its read and transport-failure children."""

    reader = _CancellationAwareReader()
    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        serve_stdio(
            _CoreRuntime(),
            input_stream=reader,
            output_stream=_MemoryWriter(),
        )
    )
    await reader.entered.wait()
    await asyncio.sleep(0)
    task.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert reader.cancelled.is_set()
        assert reader.active_reads == 0
        assert [child for child in asyncio.all_tasks() if child not in baseline] == []
    finally:
        leaked = [child for child in asyncio.all_tasks() if child not in baseline]
        for child in leaked:
            child.cancel()
        await asyncio.gather(*leaked, return_exceptions=True)


@pytest.mark.asyncio
async def test_resistant_read_cancellation_is_bounded_and_disclosed_safely(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-returning injected read is disclosed without delaying cancellation."""

    reader = _CancellationAwareReader(resist_cancellation=True)
    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        serve_stdio(
            _CoreRuntime(),
            input_stream=reader,
            output_stream=_MemoryWriter(),
            limits=GatewayLimits(graceful_shutdown_timeout_seconds=0.05),
        )
    )
    await reader.entered.wait()
    await asyncio.sleep(0)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=0.4)

    try:
        assert task in done
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reader.cancelled.is_set()
        assert reader.active_reads == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            "MCP stdio shutdown incomplete: residual input read; process termination may be required\n"
        )
    finally:
        leaked = [child for child in asyncio.all_tasks() if child not in baseline]
        for child in leaked:
            child.cancel()
        reader.release.set()
        await _eventually(lambda: reader.active_reads == 0)
        await asyncio.sleep(0)
        await asyncio.gather(*leaked, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_the_same_cleanup_task() -> None:
    """A second cancellation must not detach connection and adapter cleanup."""

    baseline = set(asyncio.all_tasks())
    server = GatewayProtocolStdioServer(
        _CoreRuntime(),
        input_stream=(reader := _QueueReader()),
        output_stream=_MemoryWriter(),
        limits=GatewayLimits(graceful_shutdown_timeout_seconds=0.2),
    )
    connection = _BlockingCleanupConnection()
    server._connection = connection  # type: ignore[assignment]
    task = asyncio.create_task(server.serve())
    await _eventually(lambda: reader.read_calls > 0)
    task.cancel()
    await connection.started.wait()
    task.cancel()
    await asyncio.sleep(0)

    try:
        assert task.done() is False
        assert [
            child
            for child in asyncio.all_tasks()
            if child not in baseline
            and child is not task
            and getattr(child.get_coro(), "__qualname__", "") == "Event.wait"
        ] == []
        connection.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert connection.shutdown_calls == 1
        await asyncio.sleep(0)
        assert [child for child in asyncio.all_tasks() if child not in baseline] == []
    finally:
        connection.release.set()
        if not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        leaked = [child for child in asyncio.all_tasks() if child not in baseline]
        for child in leaked:
            child.cancel()
        await asyncio.gather(*leaked, return_exceptions=True)


class _PrefixResistantWriter:
    def __init__(self) -> None:
        self.write_calls = 0
        self.drain_calls = 0
        self.prefix = bytearray()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active_drains = 0

    def write(self, data: bytes) -> None:
        self.write_calls += 1
        self.prefix.extend(data[:8])

    async def drain(self) -> None:
        self.drain_calls += 1
        self.active_drains += 1
        self.entered.set()
        try:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
        finally:
            self.active_drains -= 1


@pytest.mark.asyncio
async def test_drain_timeout_poison_is_terminal_and_never_redrains(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A prefix-emitting resistant drain must terminally poison the byte transport."""

    writer = _PrefixResistantWriter()
    reader = _QueueReader(_request("first"))
    server = GatewayProtocolStdioServer(
        _CoreRuntime(),
        input_stream=reader,
        output_stream=writer,
        limits=GatewayLimits(graceful_shutdown_timeout_seconds=0.05),
    )
    task = asyncio.create_task(server.serve())
    await writer.entered.wait()
    reader.feed(_request("must-not-write"))
    done, _ = await asyncio.wait({task}, timeout=0.4)

    try:
        assert task in done
        assert await task == 1
        assert writer.write_calls == 1
        assert writer.drain_calls == 1
        assert writer.active_drains == 1
        assert bytes(writer.prefix) and b"\n" not in writer.prefix
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "MCP stdio shutdown incomplete: residual output drain" in captured.err
    finally:
        writer.release.set()
        reader.feed(b"")
        await _eventually(lambda: writer.active_drains == 0)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


class _ExplodingGlobal:
    @property
    def buffer(self) -> Any:
        raise AssertionError("irrelevant process global was accessed")


@pytest.mark.asyncio
async def test_injected_reader_only_resolves_process_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected reader must make an invalid process stdin irrelevant."""

    output_read_fd, output_write_fd = os.pipe()
    output_file = os.fdopen(output_write_fd, "wb", buffering=0)
    fake_sys = SimpleNamespace(
        stdin=_ExplodingGlobal(),
        stdout=SimpleNamespace(buffer=output_file),
    )
    monkeypatch.setattr(stdio_adapters, "sys", fake_sys)

    try:
        assert await serve_stdio(_CoreRuntime(), input_stream=_QueueReader(b"")) == 0
        assert output_file.closed is False
    finally:
        output_file.close()
        os.close(output_read_fd)


@pytest.mark.asyncio
async def test_injected_writer_only_resolves_process_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected writer must make an invalid process stdout irrelevant."""

    input_read_fd, input_write_fd = os.pipe()
    os.close(input_write_fd)
    input_file = os.fdopen(input_read_fd, "rb", buffering=0)
    fake_sys = SimpleNamespace(
        stdin=SimpleNamespace(buffer=input_file),
        stdout=_ExplodingGlobal(),
    )
    monkeypatch.setattr(stdio_adapters, "sys", fake_sys)
    writer = _MemoryWriter()

    try:
        assert await serve_stdio(_CoreRuntime(), output_stream=writer) == 0
        assert input_file.closed is False
        assert writer.closed is False
    finally:
        input_file.close()
