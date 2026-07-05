"""WebSocket transport adapter for external MCP federation."""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import suppress
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from ..config_schema import ExternalMCPServerConfig
from .base import (
    BrokeredExternalCredential,
    ExternalMCPTransportAdapter,
    ExternalToolCallResult,
    ExternalToolDefinition,
    call_tool_with_ephemeral_adapter,
)

_MCP_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "tldw_external_federation", "version": "0.1.0"}


class WebSocketExternalMCPAdapter(ExternalMCPTransportAdapter):
    """WebSocket adapter for external MCP servers using JSON-RPC 2.0."""

    def __init__(
        self,
        server_config: ExternalMCPServerConfig,
        ws_connector: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> None:
        super().__init__(server_id=server_config.id)
        self.server_config = server_config
        self._ws_connector = ws_connector or self._default_ws_connector
        self._ws: Any | None = None
        self._connected = False
        self._initialized = False
        self._reader_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._next_request_id = 1

    @property
    def transport_name(self) -> str:
        return "websocket"

    async def connect(self) -> None:
        """Connect and initialize an MCP session over websocket."""

        if self.server_config.websocket is None:
            raise ValueError(f"Missing websocket config for server '{self.server_id}'")

        if self._connected and self._initialized and self._ws is not None:
            return

        async with self._connect_lock:
            if self._connected and self._initialized and self._ws is not None:
                return

            ws_cfg = self.server_config.websocket
            headers = dict(self.server_config.auth.resolve_headers())
            headers.update(dict(ws_cfg.headers or {}))
            subprotocols = list(ws_cfg.subprotocols or [])
            connect_timeout = float(self.server_config.timeouts.connect_seconds)

            logger.debug(f"External websocket adapter connect requested for {self.server_id}")
            self._ws = await self._ws_connector(
                url=ws_cfg.url,
                subprotocols=subprotocols,
                headers=headers,
                connect_timeout=connect_timeout,
            )
            self._connected = True
            self._initialized = False
            self._reader_task = asyncio.create_task(self._reader_loop())

            try:
                initialize_response = await self._jsonrpc_request(
                    method="initialize",
                    params={
                        "protocolVersion": _MCP_PROTOCOL_VERSION,
                        "clientInfo": _CLIENT_INFO,
                    },
                )
                if initialize_response.get("error"):
                    error = initialize_response["error"]
                    raise RuntimeError(
                        f"External MCP initialize failed for '{self.server_id}': "
                        f"{self._error_message(error)}"
                    )
                self._initialized = True
            except Exception:
                await self.close()
                raise

    async def close(self) -> None:
        """Close websocket/session and fail any pending requests."""

        reader_task = self._reader_task
        self._reader_task = None
        if reader_task is not None:
            reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await reader_task

        ws = self._ws
        self._ws = None
        if ws is not None:
            with suppress(Exception):
                await ws.close()

        self._connected = False
        self._initialized = False
        self._fail_pending(RuntimeError(f"External websocket adapter '{self.server_id}' closed"))

    async def health_check(self) -> dict[str, bool]:
        return {
            "configured": self.server_config.websocket is not None,
            "connected": self._connected,
            "initialized": self._initialized,
        }

    async def list_tools(self) -> list[ExternalToolDefinition]:
        await self._ensure_connected()
        response = await self._jsonrpc_request("tools/list", {})

        if response.get("error"):
            raise RuntimeError(
                f"External MCP tools/list failed for '{self.server_id}': "
                f"{self._error_message(response['error'])}"
            )

        result = response.get("result") or {}
        raw_tools: list[Any]
        if isinstance(result, dict):
            raw_tools = result.get("tools") or []
        elif isinstance(result, list):
            raw_tools = result
        else:
            raw_tools = []

        tools: list[ExternalToolDefinition] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            input_schema = item.get("inputSchema")
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object"}
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            description = item.get("description")
            if not isinstance(description, str):
                description = ""
            tools.append(
                ExternalToolDefinition(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    metadata=metadata,
                )
            )

        return tools

    async def list_resources(self) -> list[dict[str, Any]]:
        """Discover external resources and return normalized descriptors."""
        await self._ensure_connected()
        response = await self._jsonrpc_request("resources/list", {})

        if response.get("error"):
            raise RuntimeError(
                f"External MCP resources/list failed for '{self.server_id}': "
                f"{self._error_message(response['error'])}"
            )

        result = response.get("result") or {}
        if isinstance(result, dict):
            raw_resources = result.get("resources") or []
        elif isinstance(result, list):
            raw_resources = result
        else:
            raw_resources = []

        resources: list[dict[str, Any]] = []
        for item in raw_resources:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            if not isinstance(uri, str) or not uri.strip():
                continue
            name = item.get("name")
            if not isinstance(name, str):
                name = ""
            description = item.get("description")
            if not isinstance(description, str):
                description = ""
            mime_type = item.get("mimeType")
            if not isinstance(mime_type, str):
                mime_type = None
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            resources.append(
                {
                    "uri": uri,
                    "name": name,
                    "description": description,
                    "mimeType": mime_type,
                    "metadata": dict(metadata),
                }
            )
        return resources

    async def read_resource(
        self,
        uri: str,
        context: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Read one external resource."""
        del context
        await self._ensure_connected()
        response = await self._jsonrpc_request("resources/read", {"uri": uri})

        if response.get("error"):
            raise RuntimeError(
                f"External MCP resources/read failed for '{self.server_id}': "
                f"{self._error_message(response['error'])}"
            )

        result = response.get("result")
        return dict(result) if isinstance(result, dict) else {"contents": []}

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Optional[Any] = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        del context  # context is reserved for future adapter-aware policy hooks
        if runtime_auth is not None and runtime_auth.headers:
            return await self._call_tool_with_ephemeral_headers(
                tool_name=tool_name,
                arguments=arguments,
                runtime_auth=runtime_auth,
            )
        await self._ensure_connected()
        response = await self._jsonrpc_request(
            method="tools/call",
            params={"name": tool_name, "arguments": arguments or {}},
        )

        if response.get("error"):
            error = response["error"]
            message = self._error_message(error)
            return ExternalToolCallResult(
                content=[{"type": "text", "text": message}],
                is_error=True,
                metadata={
                    "server_id": self.server_id,
                    "tool_name": tool_name,
                    "upstream_error": error,
                },
            )

        result = response.get("result")
        upstream_is_error = bool(result.get("isError")) if isinstance(result, dict) else False
        if isinstance(result, dict) and "content" in result:
            content = result.get("content")
        else:
            content = result

        return ExternalToolCallResult(
            content=content,
            is_error=upstream_is_error,
            metadata={"server_id": self.server_id, "tool_name": tool_name},
        )

    async def _default_ws_connector(
        self,
        *,
        url: str,
        subprotocols: list[str],
        headers: dict[str, str],
        connect_timeout: float,
    ) -> Any:
        """Open websocket connection while handling websockets API variation."""

        try:
            import websockets  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "websockets dependency is required for external websocket transport"
            ) from exc

        connect_fn = websockets.connect
        kwargs: dict[str, Any] = {"open_timeout": connect_timeout}
        if subprotocols:
            kwargs["subprotocols"] = subprotocols

        params = inspect.signature(connect_fn).parameters
        if headers:
            if "additional_headers" in params:
                kwargs["additional_headers"] = headers
            elif "extra_headers" in params:
                kwargs["extra_headers"] = headers

        return await connect_fn(url, **kwargs)

    async def _call_tool_with_ephemeral_headers(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        runtime_auth: BrokeredExternalCredential,
    ) -> ExternalToolCallResult:
        def _prepare_config(ephemeral_config: ExternalMCPServerConfig) -> None:
            websocket_cfg = ephemeral_config.websocket
            if websocket_cfg is None:
                raise ValueError(f"Missing websocket config for server '{self.server_id}'")
            merged_headers = dict(websocket_cfg.headers or {})
            merged_headers.update(dict(runtime_auth.headers or {}))
            websocket_cfg.headers = merged_headers

        return await call_tool_with_ephemeral_adapter(
            server_config=self.server_config,
            adapter_factory=lambda config: WebSocketExternalMCPAdapter(
                config,
                ws_connector=self._ws_connector,
            ),
            prepare_config=_prepare_config,
            tool_name=tool_name,
            arguments=arguments,
        )

    async def _ensure_connected(self) -> None:
        if not self._connected or not self._initialized or self._ws is None:
            await self.connect()

    async def _reader_loop(self) -> None:
        """Receive websocket messages and dispatch JSON-RPC responses by id."""

        ws = self._ws
        if ws is None:
            return

        failure: Exception | None = None
        try:
            while True:
                raw_message = await ws.recv()
                payload = self._decode_payload(raw_message)
                if payload is None:
                    continue
                await self._dispatch_payload(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = exc
            logger.warning(
                "External websocket reader loop ended for '{}': {}",
                self.server_id,
                exc,
            )
        finally:
            self._connected = False
            self._initialized = False
            self._ws = None
            if failure is None:
                failure = RuntimeError(f"External websocket reader closed for '{self.server_id}'")
            self._fail_pending(failure)

    @staticmethod
    def _decode_payload(raw_message: Any) -> dict[str, Any] | None:
        if isinstance(raw_message, bytes):
            text = raw_message.decode("utf-8", errors="replace")
        elif isinstance(raw_message, str):
            text = raw_message
        else:
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None

    async def _dispatch_payload(self, payload: dict[str, Any]) -> None:
        response_id = payload.get("id")
        if response_id is None:
            return

        pending = self._pending.pop(str(response_id), None)
        if pending is not None and not pending.done():
            pending.set_result(payload)

    async def _jsonrpc_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        ws = self._ws
        if ws is None or not self._connected:
            raise RuntimeError(
                f"External websocket adapter for '{self.server_id}' is not connected"
            )

        request_id = self._next_request_id
        self._next_request_id += 1
        key = str(request_id)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[key] = future

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        try:
            async with self._send_lock:
                await ws.send(json.dumps(payload))
            timeout = float(self.server_config.timeouts.request_seconds)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(key, None)
            raise TimeoutError(
                f"External MCP request timed out for server '{self.server_id}' method '{method}'"
            ) from exc
        except Exception:
            self._pending.pop(key, None)
            raise

    def _fail_pending(self, exc: Exception) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(exc)

    @staticmethod
    def _error_message(error: Any) -> str:
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message
            code = error.get("code")
            return f"External MCP error ({code})"
        return str(error)
