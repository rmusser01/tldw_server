"""Private native and dedicated-thread adapters for MCP binary stdio."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import os
import queue
import sys
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Literal, Protocol, TypeAlias

from .protocol_limits import GatewayLimits

_RESIDUAL_WORKER_MESSAGE = (
    "MCP stdio shutdown incomplete: residual blocking I/O worker; process termination may be required"
)


class _OwnedByteReader(Protocol):
    async def readline(self) -> bytes: ...

    def close(self) -> None: ...


class _OwnedByteWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def close(self) -> None: ...


class _StdioAdapterShutdownError(RuntimeError):
    """Report that all cleanup was attempted but at least one operation failed."""


@dataclass(slots=True)
class _WorkItem:
    operation: Callable[[], Any]
    future: concurrent.futures.Future[Any]


class _BlockingIOWorker:
    """One bounded dedicated worker for one blocking byte-stream direction."""

    def __init__(self, name: str) -> None:
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._closed = False
        self._started = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        """Start only after every resource for the adapter has been prepared."""

        try:
            self._thread.start()
        finally:
            self._started = self._thread.ident is not None

    async def submit(self, operation: Callable[[], Any]) -> Any:
        with self._state_lock:
            if self._closed or not self._started:
                raise RuntimeError("blocking I/O worker is not available")
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            self._queue.put_nowait(_WorkItem(operation, future))
        except queue.Full as exc:
            raise RuntimeError("blocking I/O worker queue is full") from exc
        return await asyncio.wrap_future(future)

    async def shutdown(self, timeout: float) -> bool:
        with self._state_lock:
            self._closed = True
        if not self._started:
            return True
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
            if item.future.set_running_or_notify_cancel():
                try:
                    item.future.set_result(item.operation())
                except BaseException as exc:  # noqa: BLE001 - cross-thread boundary
                    item.future.set_exception(exc)


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
        try:
            loop.remove_reader(self._fd)
            waiter = self._waiter
            if waiter is not None and not waiter.done():
                waiter.set_result(b"")
        finally:
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
            except BaseException as exc:  # noqa: BLE001 - transport boundary
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
        try:
            asyncio.get_running_loop().remove_writer(self._fd)
        finally:
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
    """One or both duplicated process directions and their dedicated workers."""

    reader: _OwnedByteReader | None = None
    writer: _OwnedByteWriter | None = None
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

    async def shutdown(
        self,
        timeout: float,
        *,
        diagnostic: Callable[[str], None] | None = None,
    ) -> None:
        """Attempt every close and worker join within one shared deadline."""

        if self.closed:
            return
        self.closed = True
        errors: list[BaseException] = []
        for stream in (self.reader, self.writer):
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as exc:  # noqa: BLE001 - continue sibling cleanup
                errors.append(exc)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        for worker in self.workers:
            try:
                stopped = await worker.shutdown(max(0.0, deadline - loop.time()))
            except BaseException as exc:  # noqa: BLE001 - continue sibling cleanup
                errors.append(exc)
                stopped = False
            if not stopped:
                errors.append(RuntimeError("residual blocking I/O worker"))
                if diagnostic is not None:
                    try:
                        diagnostic(_RESIDUAL_WORKER_MESSAGE)
                    except BaseException:  # noqa: BLE001 - continue cleanup
                        pass

        if errors:
            raise _StdioAdapterShutdownError("stdio adapter shutdown failed")


_AdapterDirection: TypeAlias = Literal["input", "output"]
_AdapterSelector: TypeAlias = Callable[[_AdapterDirection, GatewayLimits], Awaitable[_OwnedStdioAdapters]]


def _validate_binary_process_stream(stream: object, direction: _AdapterDirection) -> BinaryIO:
    if isinstance(stream, io.TextIOBase):
        raise ValueError(f"process {direction} must be a binary stream")
    if not callable(getattr(stream, "fileno", None)):
        raise ValueError(f"process {direction} must expose a file descriptor")
    required = "readline" if direction == "input" else "write"
    if not callable(getattr(stream, required, None)):
        raise ValueError(f"process {direction} must be a binary {required} stream")
    return stream  # type: ignore[return-value]


def _standard_binary_stream(direction: _AdapterDirection) -> BinaryIO:
    process_stream = sys.stdin if direction == "input" else sys.stdout
    return _validate_binary_process_stream(getattr(process_stream, "buffer", None), direction)


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _close_file(stream: BinaryIO | None) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except BaseException:  # noqa: BLE001 - preserve acquisition failure
        pass


def _duplicate_file(stream: BinaryIO, mode: Literal["rb", "wb"]) -> BinaryIO:
    fd = os.dup(stream.fileno())
    try:
        return os.fdopen(fd, mode, buffering=0)
    except BaseException:
        _close_fd(fd)
        raise


async def _open_native_direction(
    direction: _AdapterDirection,
    stream: BinaryIO,
    limits: GatewayLimits,
) -> _OwnedStdioAdapters:
    if os.name != "posix":
        raise NotImplementedError("native pipe registration is POSIX-only")
    fd: int | None = None
    registered = False
    try:
        fd = os.dup(stream.fileno())
        os.set_blocking(fd, False)
        loop = asyncio.get_running_loop()
        if direction == "input":
            loop.add_reader(fd, lambda: None)
            registered = True
            loop.remove_reader(fd)
            registered = False
            reader = _NativePosixByteReader(fd, limits.max_input_line_bytes)
            fd = None
            return _OwnedStdioAdapters(reader=reader)
        loop.add_writer(fd, lambda: None)
        registered = True
        loop.remove_writer(fd)
        registered = False
        writer = _NativePosixByteWriter(fd, limits.max_output_line_bytes)
        fd = None
        return _OwnedStdioAdapters(writer=writer)
    except BaseException:
        if fd is not None and registered:
            try:
                if direction == "input":
                    asyncio.get_running_loop().remove_reader(fd)
                else:
                    asyncio.get_running_loop().remove_writer(fd)
            except BaseException:  # noqa: BLE001 - preserve acquisition failure
                pass
        _close_fd(fd)
        raise


async def _open_threaded_direction(
    direction: _AdapterDirection,
    stream: BinaryIO,
    limits: GatewayLimits,
) -> _OwnedStdioAdapters:
    prepared: BinaryIO | None = None
    worker: _BlockingIOWorker | None = None
    try:
        prepared = _duplicate_file(stream, "rb" if direction == "input" else "wb")
        worker = _BlockingIOWorker(f"mcp-stdio-{direction}")
        if direction == "input":
            reader = _ThreadedByteReader(prepared, limits.max_input_line_bytes, worker)
            bundle = _OwnedStdioAdapters(reader=reader, workers=(worker,))
        else:
            writer = _ThreadedByteWriter(prepared, limits.max_output_line_bytes, worker)
            bundle = _OwnedStdioAdapters(writer=writer, workers=(worker,))
        worker.start()
        prepared = None
        return bundle
    except BaseException:
        _close_file(prepared)
        if worker is not None and worker.alive:
            await worker.shutdown(limits.graceful_shutdown_timeout_seconds)
        raise


async def _open_native_stdio(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    limits: GatewayLimits,
) -> _OwnedStdioAdapters:
    """Open both native directions with rollback if either stage fails."""

    input_bundle = await _open_native_direction("input", input_stream, limits)
    try:
        output_bundle = await _open_native_direction("output", output_stream, limits)
    except BaseException:
        await input_bundle.shutdown(limits.graceful_shutdown_timeout_seconds)
        raise
    return _OwnedStdioAdapters(
        reader=input_bundle.reader,
        writer=output_bundle.writer,
        workers=input_bundle.workers + output_bundle.workers,
    )


async def _open_threaded_stdio(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    limits: GatewayLimits,
) -> _OwnedStdioAdapters:
    """Prepare both fallback directions before starting either worker."""

    input_file: BinaryIO | None = None
    output_file: BinaryIO | None = None
    read_worker: _BlockingIOWorker | None = None
    write_worker: _BlockingIOWorker | None = None
    try:
        input_file = _duplicate_file(input_stream, "rb")
        output_file = _duplicate_file(output_stream, "wb")
        read_worker = _BlockingIOWorker("mcp-stdio-input")
        write_worker = _BlockingIOWorker("mcp-stdio-output")
        reader = _ThreadedByteReader(input_file, limits.max_input_line_bytes, read_worker)
        writer = _ThreadedByteWriter(output_file, limits.max_output_line_bytes, write_worker)
        read_worker.start()
        write_worker.start()
        input_file = None
        output_file = None
        return _OwnedStdioAdapters(reader, writer, (read_worker, write_worker))
    except BaseException:
        _close_file(input_file)
        _close_file(output_file)
        for worker in (read_worker, write_worker):
            if worker is not None and worker.alive:
                await worker.shutdown(limits.graceful_shutdown_timeout_seconds)
        raise


async def _select_process_stdio(
    direction: _AdapterDirection,
    limits: GatewayLimits,
) -> _OwnedStdioAdapters:
    """Select one process direction without inspecting the irrelevant global."""

    stream = _standard_binary_stream(direction)
    if os.name == "posix":
        try:
            return await _open_native_direction(direction, stream, limits)
        except (NotImplementedError, OSError, ValueError):
            pass
    return await _open_threaded_direction(direction, stream, limits)
