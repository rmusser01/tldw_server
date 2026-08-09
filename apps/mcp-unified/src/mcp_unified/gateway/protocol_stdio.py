"""Portable binary stdio engine for the strict MCP protocol connection."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import io
import json
import os
import queue
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Protocol, TypeAlias, runtime_checkable

from .protocol_connection import GatewayProtocolConnection
from .protocol_limits import GatewayLimits
from .runtime import GatewayCoreRuntime, GatewayJSONValue

_PARSE_ERROR: dict[str, GatewayJSONValue] = {
    "jsonrpc": "2.0",
    "id": None,
    "error": {"code": -32700, "message": "Parse error"},
}


@runtime_checkable
class GatewayAsyncByteReader(Protocol):
    """Minimal caller-owned asynchronous byte reader accepted by strict stdio."""

    async def readline(self) -> bytes:
        """Return one newline-terminated byte line, or ``b\"\"`` at EOF."""


@runtime_checkable
class GatewayAsyncByteWriter(Protocol):
    """Minimal caller-owned asynchronous byte writer accepted by strict stdio."""

    def write(self, data: bytes) -> None:
        """Buffer one complete protocol line."""

    async def drain(self) -> None:
        """Flush buffered bytes to the underlying transport."""


class _OwnedByteReader(GatewayAsyncByteReader, Protocol):
    def close(self) -> None: ...


class _OwnedByteWriter(GatewayAsyncByteWriter, Protocol):
    def close(self) -> None: ...


@dataclass(slots=True)
class _WorkItem:
    operation: Callable[[], Any]
    future: concurrent.futures.Future[Any]


class _BlockingIOWorker:
    """One bounded dedicated worker for a single blocking byte-stream direction."""

    def __init__(self, name: str) -> None:
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._closed = False
        self._pending = 0
        self.max_pending = 0
        self._thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    async def submit(self, operation: Callable[[], Any]) -> Any:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("blocking I/O worker is closed")
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            self._pending += 1
            self.max_pending = max(self.max_pending, self._pending)
        try:
            self._queue.put_nowait(_WorkItem(operation, future))
        except queue.Full as exc:
            with self._state_lock:
                self._pending -= 1
            raise RuntimeError("blocking I/O worker queue is full") from exc
        return await asyncio.wrap_future(future)

    async def shutdown(self, timeout: float) -> bool:
        with self._state_lock:
            self._closed = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        sentinel_sent = False
        while self._thread.is_alive():
            if not sentinel_sent:
                try:
                    self._queue.put_nowait(None)
                    sentinel_sent = True
                except queue.Full:
                    pass
            remaining = max(0.0, deadline - loop.time())
            if remaining <= 0:
                break
            await asyncio.to_thread(self._thread.join, min(0.05, remaining))
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if item.future.set_running_or_notify_cancel():
                    try:
                        item.future.set_result(item.operation())
                    except BaseException as exc:  # noqa: BLE001 - cross-thread propagation
                        item.future.set_exception(exc)
            finally:
                with self._state_lock:
                    self._pending -= 1


class _ThreadedByteReader:
    def __init__(self, stream: BinaryIO, limit: int, worker: _BlockingIOWorker) -> None:
        self._stream = stream
        self._limit = limit
        self._worker = worker

    async def readline(self) -> bytes:
        value = await self._worker.submit(lambda: self._stream.readline(self._limit + 1))
        if not isinstance(value, bytes):
            raise TypeError("binary reader returned non-bytes")
        return value

    def close(self) -> None:
        self._stream.close()


class _ThreadedByteWriter:
    def __init__(self, stream: BinaryIO, limit: int, worker: _BlockingIOWorker) -> None:
        self._stream = stream
        self._limit = limit
        self._worker = worker
        self._buffer: bytes | None = None

    def write(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("binary writer requires bytes")
        if len(data) > self._limit:
            raise ValueError("output line exceeds configured limit")
        if self._buffer is not None:
            raise RuntimeError("previous write has not been drained")
        self._buffer = data

    async def drain(self) -> None:
        data = self._buffer
        self._buffer = None
        if data is None:
            await self._worker.submit(self._stream.flush)
            return
        await self._worker.submit(lambda: self._write_all(data))

    def close(self) -> None:
        self._stream.close()

    def _write_all(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = self._stream.write(view)
            if not isinstance(written, int) or written <= 0:
                raise OSError("binary output stream did not make progress")
            view = view[written:]
        self._stream.flush()


class _NativePosixByteReader:
    def __init__(self, fd: int, limit: int) -> None:
        self._fd = fd
        self._limit = limit
        self._buffer = bytearray()
        self._closed = False
        self._waiter: asyncio.Future[bytes] | None = None

    async def readline(self) -> bytes:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                end = newline + 1
                value = bytes(self._buffer[:end])
                del self._buffer[:end]
                return value
            if len(self._buffer) >= self._limit + 1:
                value = bytes(self._buffer[: self._limit + 1])
                del self._buffer[: self._limit + 1]
                return value
            chunk = await self._read_ready()
            if not chunk:
                value = bytes(self._buffer)
                self._buffer.clear()
                return value
            self._buffer.extend(chunk)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        loop.remove_reader(self._fd)
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(b"")
        os.close(self._fd)

    async def _read_ready(self) -> bytes:
        if self._closed:
            return b""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[bytes] = loop.create_future()
        self._waiter = waiter

        def readable() -> None:
            loop.remove_reader(self._fd)
            if waiter.done():
                return
            try:
                remaining = self._limit + 1 - len(self._buffer)
                waiter.set_result(os.read(self._fd, max(1, min(65_536, remaining))))
            except BlockingIOError:
                loop.add_reader(self._fd, readable)
            except BaseException as exc:  # noqa: BLE001 - transport propagation
                waiter.set_exception(exc)

        loop.add_reader(self._fd, readable)
        try:
            return await waiter
        finally:
            loop.remove_reader(self._fd)
            self._waiter = None


class _NativePosixByteWriter:
    def __init__(self, fd: int, limit: int) -> None:
        self._fd = fd
        self._limit = limit
        self._buffer: bytes | None = None
        self._closed = False

    def write(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("binary writer requires bytes")
        if len(data) > self._limit:
            raise ValueError("output line exceeds configured limit")
        if self._buffer is not None:
            raise RuntimeError("previous write has not been drained")
        self._buffer = data

    async def drain(self) -> None:
        data = self._buffer
        self._buffer = None
        if data is None:
            return
        offset = 0
        while offset < len(data):
            await self._writable()
            try:
                written = os.write(self._fd, data[offset : offset + 65_536])
            except BlockingIOError:
                continue
            if written <= 0:
                raise OSError("binary output pipe did not make progress")
            offset += written

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        asyncio.get_running_loop().remove_writer(self._fd)
        os.close(self._fd)

    async def _writable(self) -> None:
        if self._closed:
            raise OSError("binary output pipe is closed")
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()

        def writable() -> None:
            loop.remove_writer(self._fd)
            if not waiter.done():
                waiter.set_result(None)

        loop.add_writer(self._fd, writable)
        try:
            await waiter
        finally:
            loop.remove_writer(self._fd)


@dataclass(slots=True)
class _OwnedStdioAdapters:
    """Process-stream duplicates and any dedicated workers owned by the server."""

    reader: _OwnedByteReader
    writer: _OwnedByteWriter
    workers: tuple[_BlockingIOWorker, ...] = ()
    closed: bool = False
    _thread_count: int = field(init=False)

    def __post_init__(self) -> None:
        self._thread_count = len(self.workers)

    @property
    def thread_count(self) -> int:
        return self._thread_count

    @property
    def threads_alive(self) -> int:
        return sum(worker.alive for worker in self.workers)

    @property
    def max_pending_per_thread(self) -> int:
        return max((worker.max_pending for worker in self.workers), default=0)

    async def shutdown(self, timeout: float) -> None:
        if self.closed:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        joined = True
        for worker in self.workers:
            joined = await worker.shutdown(max(0.0, deadline - loop.time())) and joined
        if not joined:
            raise RuntimeError("blocking stdio worker did not stop; Python threads cannot be force-killed")
        self.reader.close()
        self.writer.close()
        self.closed = True


_AdapterSelector: TypeAlias = Callable[[GatewayLimits], Awaitable[_OwnedStdioAdapters]]


def _validate_binary_process_stream(stream: object, direction: str) -> BinaryIO:
    if isinstance(stream, io.TextIOBase):
        raise ValueError(f"process {direction} must be a binary stream")
    if not callable(getattr(stream, "fileno", None)):
        raise ValueError(f"process {direction} must expose a file descriptor")
    if direction == "input" and not callable(getattr(stream, "readline", None)):
        raise ValueError("process input must be a binary readable stream")
    if direction == "output" and not callable(getattr(stream, "write", None)):
        raise ValueError("process output must be a binary writable stream")
    return stream  # type: ignore[return-value]


def _standard_binary_streams() -> tuple[BinaryIO, BinaryIO]:
    input_stream = getattr(sys.stdin, "buffer", None)
    output_stream = getattr(sys.stdout, "buffer", None)
    return (
        _validate_binary_process_stream(input_stream, "input"),
        _validate_binary_process_stream(output_stream, "output"),
    )


async def _open_native_stdio(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    limits: GatewayLimits,
) -> _OwnedStdioAdapters:
    """Duplicate and register POSIX binary pipes without owning process globals."""

    if os.name != "posix":
        raise NotImplementedError("native pipe registration is POSIX-only")
    input_fd = os.dup(input_stream.fileno())
    output_fd = os.dup(output_stream.fileno())
    try:
        os.set_blocking(input_fd, False)
        os.set_blocking(output_fd, False)
        loop = asyncio.get_running_loop()
        loop.add_reader(input_fd, lambda: None)
        loop.remove_reader(input_fd)
        loop.add_writer(output_fd, lambda: None)
        loop.remove_writer(output_fd)
        return _OwnedStdioAdapters(
            _NativePosixByteReader(input_fd, limits.max_input_line_bytes),
            _NativePosixByteWriter(output_fd, limits.max_output_line_bytes),
        )
    except BaseException:
        os.close(input_fd)
        os.close(output_fd)
        raise


async def _open_threaded_stdio(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    limits: GatewayLimits,
) -> _OwnedStdioAdapters:
    """Duplicate binary streams and bind them to two bounded dedicated workers."""

    input_file = os.fdopen(os.dup(input_stream.fileno()), "rb", buffering=0)
    output_file = os.fdopen(os.dup(output_stream.fileno()), "wb", buffering=0)
    read_worker = _BlockingIOWorker("mcp-stdio-reader")
    write_worker = _BlockingIOWorker("mcp-stdio-writer")
    return _OwnedStdioAdapters(
        _ThreadedByteReader(input_file, limits.max_input_line_bytes, read_worker),
        _ThreadedByteWriter(output_file, limits.max_output_line_bytes, write_worker),
        (read_worker, write_worker),
    )


async def _select_process_stdio(limits: GatewayLimits) -> _OwnedStdioAdapters:
    input_stream, output_stream = _standard_binary_streams()
    if os.name == "posix":
        try:
            return await _open_native_stdio(input_stream, output_stream, limits)
        except (NotImplementedError, OSError, ValueError):
            pass
    return await _open_threaded_stdio(input_stream, output_stream, limits)


def _validate_reader(reader: object) -> GatewayAsyncByteReader:
    if isinstance(reader, io.TextIOBase):
        raise ValueError("input_stream must be a binary async reader")
    readline = getattr(reader, "readline", None)
    if not callable(readline) or not inspect.iscoroutinefunction(readline):
        raise ValueError("input_stream must define async readline() -> bytes")
    return reader  # type: ignore[return-value]


def _validate_writer(writer: object) -> GatewayAsyncByteWriter:
    if isinstance(writer, io.TextIOBase):
        raise ValueError("output_stream must be a binary async writer")
    write = getattr(writer, "write", None)
    drain = getattr(writer, "drain", None)
    if (
        not callable(write)
        or inspect.iscoroutinefunction(write)
        or not callable(drain)
        or not inspect.iscoroutinefunction(drain)
    ):
        raise ValueError("output_stream must define sync write(bytes) and async drain()")
    return writer  # type: ignore[return-value]


class GatewayProtocolStdioServer:
    """Own one strict connection and its portable line-delimited byte transport."""

    def __init__(
        self,
        runtime: GatewayCoreRuntime,
        *,
        input_stream: GatewayAsyncByteReader | None = None,
        output_stream: GatewayAsyncByteWriter | None = None,
        limits: GatewayLimits = GatewayLimits(),
        metadata: Mapping[str, Any] | None = None,
        _adapter_selector: _AdapterSelector = _select_process_stdio,
    ) -> None:
        if not callable(_adapter_selector):
            raise ValueError("adapter selector must be callable")
        self._limits = limits
        self._input = _validate_reader(input_stream) if input_stream is not None else None
        self._output = _validate_writer(output_stream) if output_stream is not None else None
        self._adapter_selector = _adapter_selector
        self._owned_adapters: _OwnedStdioAdapters | None = None
        self._write_lock = asyncio.Lock()
        self._drain_tasks: set[asyncio.Task[None]] = set()
        self._connection = GatewayProtocolConnection(
            runtime,
            self._write_protocol_value,
            limits=limits,
            metadata=metadata,
        )

    async def serve(self) -> int:
        """Serve until EOF and return a process-style status code."""

        fatal = False
        cancelled = False
        try:
            await self._resolve_streams()
            await self._read_loop()
        except asyncio.CancelledError:
            cancelled = True
        except Exception:  # noqa: BLE001 - fatal transport/internal boundary
            fatal = True

        try:
            cleanup = asyncio.create_task(self._shutdown())
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:  # noqa: BLE001 - cleanup failure is a fatal exit
            fatal = True

        if cancelled:
            raise asyncio.CancelledError
        return 1 if fatal else 0

    async def _resolve_streams(self) -> None:
        if self._input is not None and self._output is not None:
            return
        adapters = await self._adapter_selector(self._limits)
        if not isinstance(adapters, _OwnedStdioAdapters):
            raise ValueError("adapter selector returned an invalid adapter bundle")
        self._owned_adapters = adapters
        if self._input is None:
            self._input = _validate_reader(adapters.reader)
        if self._output is None:
            self._output = _validate_writer(adapters.writer)

    async def _read_loop(self) -> None:
        reader = self._require_reader()
        while True:
            line = await reader.readline()
            if not isinstance(line, bytes):
                raise TypeError("input_stream.readline() must return bytes")
            if line == b"":
                return
            if len(line) > self._limits.max_input_line_bytes:
                await self._write_direct(_PARSE_ERROR)
                if not line.endswith(b"\n"):
                    await self._discard_oversized_line(reader)
                continue
            if not line.strip():
                continue
            if not line.endswith(b"\n") or b"\n" in line[:-1]:
                await self._write_direct(_PARSE_ERROR)
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                await self._write_direct(_PARSE_ERROR)
                continue
            await self._connection.receive(payload)
            await asyncio.sleep(0)

    async def _discard_oversized_line(self, reader: GatewayAsyncByteReader) -> None:
        while True:
            chunk = await reader.readline()
            if not isinstance(chunk, bytes):
                raise TypeError("input_stream.readline() must return bytes")
            if not chunk or chunk.endswith(b"\n"):
                return

    async def _write_protocol_value(self, value: GatewayJSONValue) -> None:
        data = self._serialize(value)
        if len(data) > self._limits.max_output_line_bytes:
            raise RuntimeError("protocol connection emitted an oversized line")
        await self._write_bytes(data)

    async def _write_direct(self, value: GatewayJSONValue) -> None:
        data = self._serialize(value)
        if len(data) <= self._limits.max_output_line_bytes:
            await self._write_bytes(data)

    async def _write_bytes(self, data: bytes) -> None:
        async with self._write_lock:
            writer = self._require_writer()
            writer.write(data)
            await self._bounded_drain(writer)

    async def _bounded_drain(self, writer: GatewayAsyncByteWriter) -> None:
        task = asyncio.create_task(writer.drain())
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)
        done, _ = await asyncio.wait(
            {task},
            timeout=self._limits.graceful_shutdown_timeout_seconds,
        )
        if task not in done:
            task.cancel()
            raise TimeoutError("output drain timed out")
        await task

    async def _shutdown(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._limits.graceful_shutdown_timeout_seconds
        errors: list[BaseException] = []

        try:
            await asyncio.wait_for(
                self._connection.shutdown(),
                timeout=max(0.0, deadline - loop.time()),
            )
        except BaseException as exc:  # noqa: BLE001 - finish all owned cleanup
            errors.append(exc)

        output = self._output
        if output is not None and loop.time() < deadline:
            try:
                await self._drain_with_deadline(output, deadline)
            except BaseException as exc:  # noqa: BLE001 - finish adapter cleanup
                errors.append(exc)

        adapters = self._owned_adapters
        if adapters is not None:
            try:
                await adapters.shutdown(max(0.0, deadline - loop.time()))
            except BaseException as exc:  # noqa: BLE001 - report bounded join failure
                errors.append(exc)

        for task in tuple(self._drain_tasks):
            task.cancel()
        if errors:
            raise errors[0]

    async def _drain_with_deadline(
        self,
        writer: GatewayAsyncByteWriter,
        deadline: float,
    ) -> None:
        task = asyncio.create_task(writer.drain())
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)
        done, _ = await asyncio.wait(
            {task},
            timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
        )
        if task not in done:
            task.cancel()
            raise TimeoutError("output shutdown drain timed out")
        await task

    def _require_reader(self) -> GatewayAsyncByteReader:
        if self._input is None:
            raise RuntimeError("stdio input was not initialized")
        return self._input

    def _require_writer(self) -> GatewayAsyncByteWriter:
        if self._output is None:
            raise RuntimeError("stdio output was not initialized")
        return self._output

    @staticmethod
    def _serialize(value: GatewayJSONValue) -> bytes:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )


async def serve_stdio(
    runtime: GatewayCoreRuntime,
    *,
    input_stream: GatewayAsyncByteReader | None = None,
    output_stream: GatewayAsyncByteWriter | None = None,
    limits: GatewayLimits = GatewayLimits(),
    metadata: Mapping[str, Any] | None = None,
) -> int:
    """Serve one strict MCP connection over binary stdio streams."""

    server = GatewayProtocolStdioServer(
        runtime=runtime,
        input_stream=input_stream,
        output_stream=output_stream,
        limits=limits,
        metadata=metadata,
    )
    return await server.serve()


__all__ = [
    "GatewayAsyncByteReader",
    "GatewayAsyncByteWriter",
    "GatewayProtocolStdioServer",
    "serve_stdio",
]
