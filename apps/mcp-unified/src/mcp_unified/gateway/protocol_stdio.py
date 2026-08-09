"""Portable binary stdio engine for the strict MCP protocol connection."""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import sys
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from .protocol_connection import GatewayProtocolConnection
from .protocol_limits import GatewayLimits
from .protocol_stdio_adapters import (
    _AdapterDirection,
    _AdapterSelector,
    _OwnedStdioAdapters,
    _select_process_stdio,
)
from .runtime import GatewayCoreRuntime, GatewayJSONValue

_PARSE_ERROR: dict[str, GatewayJSONValue] = {
    "jsonrpc": "2.0",
    "id": None,
    "error": {"code": -32700, "message": "Parse error"},
}
_RESIDUAL_DRAIN_MESSAGE = "MCP stdio shutdown incomplete: residual output drain; process termination may be required"
_RESIDUAL_READ_MESSAGE = "MCP stdio shutdown incomplete: residual input read; process termination may be required"
_RESIDUAL_CONNECTION_MESSAGE = (
    "MCP stdio shutdown incomplete: residual protocol cleanup; process termination may be required"
)


def _write_stderr_diagnostic(message: str) -> None:
    """Best-effort fixed diagnostics that can never corrupt protocol stdout."""

    try:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
    except BaseException:  # noqa: BLE001 - diagnostics must not block cleanup
        pass


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
        _diagnostic: Callable[[str], None] = _write_stderr_diagnostic,
    ) -> None:
        if not callable(_adapter_selector) or not callable(_diagnostic):
            raise ValueError("adapter selector must be callable")
        self._limits = limits
        self._input = _validate_reader(input_stream) if input_stream is not None else None
        self._output = _validate_writer(output_stream) if output_stream is not None else None
        self._adapter_selector = _adapter_selector
        self._diagnostic = _diagnostic
        self._owned_adapters: list[_OwnedStdioAdapters] = []
        self._write_lock = asyncio.Lock()
        self._transport_failed = asyncio.Event()
        self._transport_poisoned = False
        self._active_read_task: asyncio.Task[bytes] | None = None
        self._active_failure_task: asyncio.Task[bool] | None = None
        self._active_drain_task: asyncio.Task[None] | None = None
        self._residual_read_disclosed = False
        self._residual_drain_disclosed = False
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

        cleanup = asyncio.create_task(self._shutdown())
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                cancelled = True
                continue
            except Exception:  # noqa: BLE001 - cleanup failure is a fatal exit
                fatal = True
                break

        current_task = asyncio.current_task()
        cancelling = getattr(current_task, "cancelling", lambda: 0)
        if cancelled or cancelling():
            raise asyncio.CancelledError
        return 1 if fatal else 0

    async def _resolve_streams(self) -> None:
        if self._input is None:
            adapters = await self._resolve_direction("input")
            if adapters.reader is None:
                raise ValueError("input adapter did not provide a reader")
            self._input = _validate_reader(adapters.reader)
        if self._output is None:
            adapters = await self._resolve_direction("output")
            if adapters.writer is None:
                raise ValueError("output adapter did not provide a writer")
            self._output = _validate_writer(adapters.writer)

    async def _resolve_direction(self, direction: _AdapterDirection) -> _OwnedStdioAdapters:
        adapters = await self._adapter_selector(direction, self._limits)
        if not isinstance(adapters, _OwnedStdioAdapters):
            raise ValueError("adapter selector returned an invalid adapter bundle")
        if not any(owned is adapters for owned in self._owned_adapters):
            self._owned_adapters.append(adapters)
        return adapters

    async def _read_loop(self) -> None:
        reader = self._require_reader()
        while True:
            line = await self._readline_or_transport_failure(reader)
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
            chunk = await self._readline_or_transport_failure(reader)
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
            if self._transport_poisoned:
                raise RuntimeError("stdio output transport is unavailable")
            writer = self._require_writer()
            try:
                writer.write(data)
            except BaseException:
                self._poison_transport()
                raise
            await self._bounded_drain(writer)

    async def _bounded_drain(self, writer: GatewayAsyncByteWriter) -> None:
        task = asyncio.create_task(writer.drain())
        self._active_drain_task = task
        task.add_done_callback(self._consume_drain_result)
        try:
            done, _ = await asyncio.wait(
                {task},
                timeout=self._limits.graceful_shutdown_timeout_seconds,
            )
            if task not in done:
                self._poison_transport()
                task.cancel()
                raise TimeoutError("output drain timed out")
            await task
        except asyncio.CancelledError:
            self._poison_transport()
            task.cancel()
            raise
        except BaseException:
            self._poison_transport()
            raise

    async def _shutdown(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._limits.graceful_shutdown_timeout_seconds
        errors: list[BaseException] = []

        connection_task = asyncio.create_task(self._connection.shutdown())
        if not await self._account_input_waiters(deadline):
            errors.append(TimeoutError("input read did not stop"))
        connection_done, _ = await asyncio.wait({connection_task}, timeout=max(0.0, deadline - loop.time()))
        if connection_task in connection_done:
            try:
                await connection_task
            except BaseException as exc:  # noqa: BLE001 - finish all owned cleanup
                errors.append(exc)
        else:
            connection_task.cancel()
            await asyncio.sleep(0)
            if connection_task.done():
                await asyncio.gather(connection_task, return_exceptions=True)
            else:
                self._emit_diagnostic(_RESIDUAL_CONNECTION_MESSAGE)
            errors.append(TimeoutError("protocol cleanup timed out"))

        output = self._output
        active_drain = self._active_drain_task
        if active_drain is not None and not active_drain.done():
            if not await self._account_drain(active_drain, deadline):
                self._disclose_residual_drain()
                errors.append(TimeoutError("output drain did not stop"))
        elif output is not None and not self._transport_poisoned and loop.time() < deadline:
            try:
                await self._drain_with_deadline(output, deadline)
            except BaseException as exc:  # noqa: BLE001 - finish adapter cleanup
                errors.append(exc)

        for adapters in self._owned_adapters:
            try:
                await adapters.shutdown(
                    max(0.0, deadline - loop.time()),
                    diagnostic=self._diagnostic,
                )
            except BaseException as exc:  # noqa: BLE001 - report bounded join failure
                errors.append(exc)

        if errors:
            raise RuntimeError("stdio shutdown failed")

    async def _drain_with_deadline(
        self,
        writer: GatewayAsyncByteWriter,
        deadline: float,
    ) -> None:
        task = asyncio.create_task(writer.drain())
        self._active_drain_task = task
        task.add_done_callback(self._consume_drain_result)
        done, _ = await asyncio.wait(
            {task},
            timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
        )
        if task not in done:
            self._poison_transport()
            task.cancel()
            if not await self._account_drain(task, deadline):
                self._disclose_residual_drain()
            raise TimeoutError("output shutdown drain timed out")
        await task

    async def _readline_or_transport_failure(
        self,
        reader: GatewayAsyncByteReader,
    ) -> bytes:
        if self._transport_poisoned:
            raise RuntimeError("stdio output transport is unavailable")
        read_task = asyncio.create_task(reader.readline())
        failure_task = asyncio.create_task(self._transport_failed.wait())
        self._active_read_task = read_task
        self._active_failure_task = failure_task
        read_task.add_done_callback(self._consume_read_result)
        try:
            done, _ = await asyncio.wait({read_task, failure_task}, return_when=asyncio.FIRST_COMPLETED)
            if failure_task in done and self._transport_poisoned:
                read_task.cancel()
                failure_task.cancel()
                raise RuntimeError("stdio output transport is unavailable")
            failure_task.cancel()
            await asyncio.gather(failure_task, return_exceptions=True)
            if self._active_failure_task is failure_task:
                self._active_failure_task = None
            try:
                return await read_task
            finally:
                if self._active_read_task is read_task:
                    self._active_read_task = None
        except BaseException:
            read_task.cancel()
            failure_task.cancel()
            raise

    async def _account_input_waiters(self, deadline: float) -> bool:
        failure_task = self._active_failure_task
        if failure_task is not None:
            failure_task.cancel()
            await asyncio.gather(failure_task, return_exceptions=True)
            if self._active_failure_task is failure_task:
                self._active_failure_task = None

        read_task = self._active_read_task
        if read_task is None:
            return True
        read_task.cancel()
        done, _ = await asyncio.wait({read_task}, timeout=max(0.0, deadline - asyncio.get_running_loop().time()))
        if read_task not in done:
            self._disclose_residual_read()
            return False
        await asyncio.gather(read_task, return_exceptions=True)
        if self._active_read_task is read_task:
            self._active_read_task = None
        return True

    async def _account_drain(self, task: asyncio.Task[None], deadline: float) -> bool:
        if task.done():
            await asyncio.gather(task, return_exceptions=True)
            return True
        done, _ = await asyncio.wait({task}, timeout=max(0.0, deadline - asyncio.get_running_loop().time()))
        if task in done:
            await asyncio.gather(task, return_exceptions=True)
            return True
        return False

    def _consume_drain_result(self, task: asyncio.Task[None]) -> None:
        if self._active_drain_task is task and task.done():
            self._active_drain_task = None
        if not task.cancelled():
            task.exception()

    def _consume_read_result(self, task: asyncio.Task[bytes]) -> None:
        if self._active_read_task is task:
            self._active_read_task = None
        if not task.cancelled():
            task.exception()

    def _poison_transport(self) -> None:
        self._transport_poisoned = True
        self._transport_failed.set()

    def _disclose_residual_drain(self) -> None:
        if not self._residual_drain_disclosed:
            self._residual_drain_disclosed = True
            self._emit_diagnostic(_RESIDUAL_DRAIN_MESSAGE)

    def _disclose_residual_read(self) -> None:
        if not self._residual_read_disclosed:
            self._residual_read_disclosed = True
            self._emit_diagnostic(_RESIDUAL_READ_MESSAGE)

    def _emit_diagnostic(self, message: str) -> None:
        try:
            self._diagnostic(message)
        except BaseException:  # noqa: BLE001 - diagnostics cannot block cleanup
            pass

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
