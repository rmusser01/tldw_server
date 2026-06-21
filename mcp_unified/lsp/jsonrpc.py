"""Async LSP JSON-RPC client over stdio Content-Length framing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import DEFAULT_LSP_CONFIG, LspRuntimeConfig
from .errors import LspToolError, redact_lsp_detail

JsonRpcPayload = dict[str, Any]


class LspJsonRpcClient:
    """Manage one stdio JSON-RPC subprocess using LSP framing."""

    def __init__(
        self,
        *,
        argv: Sequence[str],
        workspace_root: Path,
        config: LspRuntimeConfig = DEFAULT_LSP_CONFIG,
    ):
        if isinstance(argv, str) or not argv:
            raise ValueError("argv must be a non-empty sequence of executable arguments")
        if not all(isinstance(part, str) and part for part in argv):
            raise TypeError("argv must contain only non-empty strings")
        self.argv = tuple(argv)
        self.workspace_root = workspace_root
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[JsonRpcPayload]] = {}
        self._next_id = 1
        self._stderr = bytearray()
        self._close_errors: list[str] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_errors(self) -> tuple[str, ...]:
        return tuple(self._close_errors)

    async def start(self) -> None:
        """Start the stdio subprocess and background readers."""

        if self._process is not None:
            return
        try:
            self._process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *self.argv,
                    cwd=str(self.workspace_root),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self.config.startup_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise LspToolError("backend_timeout", "LSP backend startup timed out") from exc
        except OSError as exc:
            raise LspToolError(
                "backend_missing",
                "failed to start LSP backend",
                detail=f"{exc.__class__.__name__}: {exc}",
            ) from exc

        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr_loop())

    async def request(self, method: str, params: object | None = None) -> object:
        """Send a JSON-RPC request and return its result."""

        if self._closed:
            raise LspToolError("backend_unhealthy", "LSP backend client is closed")
        await self.start()
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[JsonRpcPayload] = loop.create_future()
        self._pending[request_id] = future
        payload: JsonRpcPayload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        try:
            await self._write_payload(payload)
            response = await asyncio.wait_for(future, timeout=self.config.request_timeout_seconds)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise LspToolError("backend_timeout", f"LSP request timed out: {method}") from exc
        except LspToolError:
            self._pending.pop(request_id, None)
            raise
        except Exception as exc:
            self._pending.pop(request_id, None)
            raise LspToolError(
                "backend_unhealthy",
                f"LSP request failed: {method}",
                detail=f"{exc.__class__.__name__}: {exc}",
            ) from exc

        if "error" in response:
            raise LspToolError("backend_unhealthy", f"LSP request failed: {method}", detail=str(response["error"]))
        return response.get("result")

    async def notify(self, method: str, params: object | None = None) -> None:
        """Send a JSON-RPC notification."""

        if self._closed:
            return
        await self.start()
        payload: JsonRpcPayload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._write_payload(payload)

    def stderr_text(self, *, workspace_root: Path | None = None) -> str:
        """Return bounded, redacted stderr text captured from the subprocess."""

        raw_text = bytes(self._stderr).decode("utf-8", errors="replace")
        return redact_lsp_detail(raw_text, workspace_root=workspace_root, max_length=self.config.max_stderr_bytes) or ""

    async def close(self) -> None:
        """Shutdown the subprocess and cancel background readers without raising."""

        if self._closed:
            return
        process = self._process
        if process is not None and process.returncode is None:
            try:
                await self.request("shutdown")
            except (LspToolError, BrokenPipeError, ConnectionError, OSError, RuntimeError) as exc:
                self._close_errors.append(f"shutdown:{exc.__class__.__name__}")
            try:
                await self.notify("exit")
            except (LspToolError, BrokenPipeError, ConnectionError, OSError, RuntimeError) as exc:
                self._close_errors.append(f"exit:{exc.__class__.__name__}")
            await self._close_process(process)
        await self._cancel_reader_tasks()
        self._fail_pending(LspToolError("backend_unhealthy", "LSP backend client closed"))
        self._closed = True

    async def _write_payload(self, payload: JsonRpcPayload) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise LspToolError("backend_unhealthy", "LSP backend process is not started")
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        await process.stdin.drain()

    async def _read_stdout_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                payload = await _read_lsp_payload(process.stdout)
                if payload is None:
                    break
                request_id = payload.get("id")
                if isinstance(request_id, int):
                    future = self._pending.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_result(payload)
        except (LspToolError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._fail_pending(
                LspToolError(
                    "backend_unhealthy",
                    "LSP backend stdout reader failed",
                    detail=f"{exc.__class__.__name__}: {exc}",
                )
            )
        finally:
            self._fail_pending(LspToolError("backend_unhealthy", "LSP backend closed stdout"))

    async def _read_stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        raw_limit = max(self.config.max_stderr_bytes * 4, self.config.max_stderr_bytes)
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                return
            remaining = raw_limit - len(self._stderr)
            if remaining > 0:
                self._stderr.extend(chunk[:remaining])

    def _fail_pending(self, exc: LspToolError) -> None:
        for request_id, future in list(self._pending.items()):
            self._pending.pop(request_id, None)
            if not future.done():
                future.set_exception(exc)

    async def _close_process(self, process: asyncio.subprocess.Process) -> None:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, ConnectionError, OSError, RuntimeError) as exc:
                self._close_errors.append(f"stdin_close:{exc.__class__.__name__}")
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
            return
        except asyncio.TimeoutError:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
            return
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _cancel_reader_tasks(self) -> None:
        tasks = [task for task in (self._reader_task, self._stderr_task) if task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
            except (LspToolError, OSError, RuntimeError, ValueError) as exc:
                self._close_errors.append(f"reader_task:{exc.__class__.__name__}")


async def _read_lsp_payload(reader: asyncio.StreamReader) -> JsonRpcPayload | None:
    try:
        header_bytes = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as exc:
        if exc.partial:
            raise LspToolError("backend_unhealthy", "incomplete LSP header") from exc
        return None
    content_length = _parse_content_length(header_bytes)
    try:
        body = await reader.readexactly(content_length)
    except asyncio.IncompleteReadError as exc:
        raise LspToolError("backend_unhealthy", "incomplete LSP body") from exc
    return json.loads(body.decode("utf-8"))


def _parse_content_length(header_bytes: bytes) -> int:
    header_text = header_bytes.decode("ascii", errors="replace")
    for line in header_text.splitlines():
        name, _, value = line.partition(":")
        if name.lower() == "content-length":
            content_length = int(value.strip())
            if content_length <= 0:
                raise LspToolError("backend_unhealthy", "invalid LSP Content-Length")
            return content_length
    raise LspToolError("backend_unhealthy", "missing LSP Content-Length")
