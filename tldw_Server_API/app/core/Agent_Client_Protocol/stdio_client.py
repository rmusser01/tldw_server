from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.hardening import (
    ACP_DEFAULT_RPC_TIMEOUT_SECONDS,
    redact_agent_output,
)


class ACPResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error = error or {}


@dataclass
class ACPMessage:
    jsonrpc: str
    id: Any | None = None
    method: str | None = None
    params: Any | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


RequestHandler = Callable[[ACPMessage], Awaitable[ACPMessage]]
NotificationHandler = Callable[[ACPMessage], Awaitable[None]]


class ACPStdioClient:
    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        rpc_timeout_sec: float | None = None,
    ) -> None:
        self.command = command
        self.args = args
        self.env = env or {}
        self.cwd = cwd
        self.rpc_timeout_sec = ACP_DEFAULT_RPC_TIMEOUT_SECONDS if rpc_timeout_sec is None else float(rpc_timeout_sec)
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._request_handler: RequestHandler | None = None
        self._notification_handler: NotificationHandler | None = None

    def set_request_handler(self, handler: RequestHandler) -> None:
        self._request_handler = handler

    def set_notification_handler(self, handler: NotificationHandler) -> None:
        self._notification_handler = handler

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        if not self.command:
            raise ValueError("ACP runner command is required")
        if self._proc is not None:
            return

        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env={**os.environ, **self.env},
        )

        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())

    async def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        if self._reader_task:
            self._reader_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()
        self._proc = None

    async def call(self, method: str, params: Any | None = None) -> ACPMessage:
        if self._proc is None:
            raise ACPResponseError("ACP process not started")

        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[str(request_id)] = future

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            await self._send(payload)
            if self.rpc_timeout_sec > 0:
                resp = await asyncio.wait_for(future, timeout=self.rpc_timeout_sec)
            else:
                resp = await future
        except asyncio.TimeoutError as exc:
            raise ACPResponseError(f"ACP call timed out waiting for {method}", code=-32000) from exc
        finally:
            self._pending.pop(str(request_id), None)
        if resp.error:
            raise ACPResponseError(
                resp.error.get("message", "ACP error"),
                code=resp.error.get("code"),
                error=resp.error,
            )
        return resp

    async def notify(self, method: str, params: Any | None = None) -> None:
        if self._proc is None:
            raise ACPResponseError("ACP process not started")
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send(payload)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise ACPResponseError("ACP stdin not available")
        data = json.dumps(payload, separators=(",", ":"))
        if "\n" in data:
            raise ACPResponseError("ACP message contains newline")

        async with self._write_lock:
            self._proc.stdin.write((data + "\n").encode("utf-8"))
            await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._drain_pending("ACP stdout not available")
            return
        reader = proc.stdout
        while True:
            line = await reader.readline()
            if not line:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("ACP client received invalid JSON")
                continue
            msg = ACPMessage(
                jsonrpc=payload.get("jsonrpc", "2.0"),
                id=payload.get("id"),
                method=payload.get("method"),
                params=payload.get("params"),
                result=payload.get("result"),
                error=payload.get("error"),
            )

            if msg.method and msg.id is None:
                await self._handle_notification(msg)
                continue
            if msg.method and msg.id is not None:
                await self._handle_request(msg)
                continue

            if msg.id is not None:
                future = self._pending.pop(str(msg.id), None)
                if future and not future.done():
                    future.set_result(msg)
        self._drain_pending("ACP connection closed")

    async def _handle_request(self, msg: ACPMessage) -> None:
        if self._request_handler is None:
            await self._send_error(msg.id, -32601, "method not found")
            return
        try:
            response = await self._request_handler(msg)
        except Exception as exc:  # pragma: no cover - defensive
            await self._send_error(msg.id, -32603, str(exc))
            return
        if response is None:
            await self._send_error(msg.id, -32603, "empty response")
            return
        payload = {
            "jsonrpc": "2.0",
            "id": msg.id,
            "result": response.result,
        }
        if response.error:
            payload["error"] = response.error
            payload.pop("result", None)
        await self._send(payload)

    async def _handle_notification(self, msg: ACPMessage) -> None:
        if self._notification_handler is None:
            return
        await self._notification_handler(msg)

    async def _send_error(self, request_id: Any, code: int, message: str) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        await self._send(payload)

    async def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            logger.debug("ACP runner stderr: {}", redact_agent_output(line).rstrip())

    def _drain_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ACPResponseError(reason))
        self._pending.clear()
