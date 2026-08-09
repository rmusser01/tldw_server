"""Portable byte-stdio tests for the strict MCP protocol connection."""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mcp_unified.gateway.protocol_stdio as protocol_stdio
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
from mcp_unified.gateway.protocol_stdio import (
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
) -> tuple[int, bytes, _OwnedStdioAdapters]:
    input_read_fd, input_write_fd = os.pipe()
    output_read_fd, output_write_fd = os.pipe()
    os.write(input_write_fd, input_bytes)
    os.close(input_write_fd)
    input_file = os.fdopen(input_read_fd, "rb", buffering=0)
    output_file = os.fdopen(output_write_fd, "wb", buffering=0)
    captured: list[_OwnedStdioAdapters] = []

    async def selector(selected_limits: GatewayLimits) -> _OwnedStdioAdapters:
        adapters = await opener(input_file, output_file, selected_limits)
        captured.append(adapters)
        input_file.close()
        output_file.close()
        return adapters

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
    return result, b"".join(chunks), captured[0]


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="native add_reader path is POSIX-only")
async def test_native_posix_adapter_serves_duplicated_binary_pipes() -> None:
    """Regressing to mocked async streams would leave the native pipe claim unproved."""

    result, output, adapters = await _pipe_server(
        _open_native_stdio,
        _request("native"),
    )

    assert result == 0
    assert json.loads(output)["id"] == "native"
    assert adapters.closed is True
    assert adapters.thread_count == 0


@pytest.mark.asyncio
async def test_threaded_fallback_bounds_reads_serializes_writes_and_joins() -> None:
    """An unbounded or shared-executor fallback would not satisfy the Windows contract."""

    limits = GatewayLimits(max_input_line_bytes=256)
    result, output, adapters = await _pipe_server(
        _open_threaded_stdio,
        b"x" * 300 + b"\n" + _request("fallback"),
        limits=limits,
    )
    values = [json.loads(line) for line in output.splitlines()]

    assert result == 0
    assert [value.get("id") for value in values] == [None, "fallback"]
    assert values[0]["error"]["code"] == -32700
    assert adapters.thread_count == 2
    assert adapters.threads_alive == 0
    assert adapters.max_pending_per_thread <= 1


@pytest.mark.asyncio
async def test_threaded_fallback_propagates_cancellation_and_joins_threads() -> None:
    """Cancellation must leave no fallback worker running after cooperative pipe release."""

    input_read_fd, input_write_fd = os.pipe()
    output_read_fd, output_write_fd = os.pipe()
    input_file = os.fdopen(input_read_fd, "rb", buffering=0)
    output_file = os.fdopen(output_write_fd, "wb", buffering=0)
    captured: list[_OwnedStdioAdapters] = []

    async def selector(limits: GatewayLimits) -> _OwnedStdioAdapters:
        adapters = await _open_threaded_stdio(input_file, output_file, limits)
        captured.append(adapters)
        input_file.close()
        output_file.close()
        return adapters

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
    assert captured[0].threads_alive == 0
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

    async def selector(limits: GatewayLimits) -> _OwnedStdioAdapters:
        adapters = await _open_threaded_stdio(input_file, output_file, limits)
        captured.append(adapters)
        input_file.close()
        output_file.close()
        return adapters

    result = await GatewayProtocolStdioServer(
        _CoreRuntime(),
        _adapter_selector=selector,
    ).serve()

    assert result == 1
    assert captured[0].threads_alive == 0
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
    monkeypatch.setattr(protocol_stdio, "sys", fake_sys)

    adapters = await protocol_stdio._select_process_stdio(GatewayLimits())
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
    """Removing either real platform job would invalidate the portable package claim."""

    workflow = yaml.load(
        (_REPO_ROOT / ".github/workflows/mcp-unified-rc.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["portable-stdio"]
    assert job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert job["runs-on"] == "${{ matrix.os }}"
    run_blocks = "\n".join(step["run"] for step in job["steps"] if "run" in step)
    assert 'python -m pip install "./apps/mcp-unified[dev]"' in run_blocks
    assert "test_gateway_protocol_contracts.py" in run_blocks
    assert "test_gateway_protocol_connection.py" in run_blocks
    assert "test_gateway_protocol_stdio.py" in run_blocks
    assert "--noconftest" in run_blocks
    assert "-c apps/mcp-unified/pyproject.toml" in run_blocks
