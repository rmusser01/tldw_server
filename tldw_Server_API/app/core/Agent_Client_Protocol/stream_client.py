from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import (
    ACPMessage,
    ACPResponseError,
    NotificationHandler,
    RequestHandler,
)


@dataclass
class ACPStreamClient:
    """ACP client that reads/writes newline-delimited JSON over an abstract byte stream."""

    send_bytes: Callable[[bytes], Awaitable[None]]
    _request_handler: RequestHandler | None = None
    _notification_handler: NotificationHandler | None = None

    def __post_init__(self) -> None:
        self._pending: dict[str, asyncio.Future] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._buffer = bytearray()
        self._running = True

    def set_request_handler(self, handler: RequestHandler) -> None:
        self._request_handler = handler

    def set_notification_handler(self, handler: NotificationHandler) -> None:
        self._notification_handler = handler

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        # No-op; transport is external
        self._running = True

    async def close(self) -> None:
        self._running = False
        self._drain_pending("ACP connection closed")

    async def call(self, method: str, params: Any | None = None) -> ACPMessage:
        if not self._running:
            raise ACPResponseError("ACP stream not running")

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
        await self._send(payload)
        resp = await future
        if resp.error:
            raise ACPResponseError(
                resp.error.get("message", "ACP error"),
                code=resp.error.get("code"),
                error=resp.error,
            )
        return resp

    async def notify(self, method: str, params: Any | None = None) -> None:
        if not self._running:
            raise ACPResponseError("ACP stream not running")
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send(payload)

    async def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"))
        if "\n" in data:
            raise ACPResponseError("ACP message contains newline")
        async with self._write_lock:
            await self.send_bytes((data + "\n").encode("utf-8"))

    async def feed_bytes(self, data: bytes) -> None:
        if not data:
            return
        self._buffer.extend(data)
        while True:
            try:
                idx = self._buffer.index(b"\n")
            except ValueError:
                return
            line = bytes(self._buffer[:idx]).strip()
            del self._buffer[: idx + 1]
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("ACP stream received invalid JSON")
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

    def _drain_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ACPResponseError(reason))
        self._pending.clear()
