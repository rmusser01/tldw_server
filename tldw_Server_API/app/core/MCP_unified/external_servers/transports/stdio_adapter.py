"""Stdio transport adapter for external MCP federation."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from loguru import logger
from mcp_unified.federation.resource_payloads import (
    normalize_external_resource_list,
    normalize_external_resource_read,
)

from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import (
    ACPResponseError,
    ACPStdioClient,
)

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


class StdioExternalMCPAdapter(ExternalMCPTransportAdapter):
    """Stdio adapter for external MCP servers using JSON-RPC 2.0 over newline framing."""

    def __init__(
        self,
        server_config: ExternalMCPServerConfig,
        client_factory: Optional[Callable[[ExternalMCPServerConfig], Awaitable[Any] | Any]] = None,
    ) -> None:
        super().__init__(server_id=server_config.id)
        self.server_config = server_config
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None
        self._connected = False
        self._initialized = False
        self._connect_lock = asyncio.Lock()

    @property
    def transport_name(self) -> str:
        return "stdio"

    async def connect(self) -> None:
        """Start stdio process and initialize MCP session."""

        if self._connected and self._initialized and self._is_client_running():
            return
        if self.server_config.stdio is None:
            raise ValueError(f"Missing stdio config for server '{self.server_id}'")

        async with self._connect_lock:
            if self._connected and self._initialized and self._is_client_running():
                return

            logger.debug(f"External stdio adapter connect requested for {self.server_id}")

            self._client = await self._build_client()
            connect_timeout = float(self.server_config.timeouts.connect_seconds)
            await asyncio.wait_for(self._client.start(), timeout=connect_timeout)
            self._connected = True
            self._initialized = False

            try:
                await self._request(
                    "initialize",
                    {
                        "protocolVersion": _MCP_PROTOCOL_VERSION,
                        "clientInfo": _CLIENT_INFO,
                    },
                )
                self._initialized = True
            except Exception:
                await self.close()
                raise

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.close()
            except Exception as close_error:
                logger.debug("MCP stdio transport client close failed", exc_info=close_error)
        self._connected = False
        self._initialized = False

    async def health_check(self) -> dict[str, bool]:
        return {
            "configured": self.server_config.stdio is not None,
            "connected": self._connected and self._is_client_running(),
            "initialized": self._initialized,
        }

    async def list_tools(self) -> list[ExternalToolDefinition]:
        await self._ensure_connected()
        response = await self._request("tools/list", {})
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
        response = await self._request("resources/list", {})
        return normalize_external_resource_list(response.get("result"))

    async def read_resource(
        self,
        uri: str,
        context: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Read one external resource."""
        del context
        await self._ensure_connected()
        response = await self._request("resources/read", {"uri": uri})
        return normalize_external_resource_read(response.get("result"))

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Optional[Any] = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        del context  # Reserved for future policy hooks
        if runtime_auth is not None and runtime_auth.env:
            return await self._call_tool_with_ephemeral_env(
                tool_name=tool_name,
                arguments=arguments,
                runtime_auth=runtime_auth,
            )
        await self._ensure_connected()
        try:
            response = await self._request(
                "tools/call",
                {"name": tool_name, "arguments": arguments or {}},
            )
        except ACPResponseError as exc:
            message = str(exc) or "External MCP call failed"
            return ExternalToolCallResult(
                content=[{"type": "text", "text": message}],
                is_error=True,
                metadata={"server_id": self.server_id, "tool_name": tool_name},
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

    def _default_client_factory(self, server_config: ExternalMCPServerConfig) -> ACPStdioClient:
        stdio_cfg = server_config.stdio
        if stdio_cfg is None:
            raise ValueError(f"Missing stdio config for server '{self.server_id}'")
        return ACPStdioClient(
            command=stdio_cfg.command,
            args=list(stdio_cfg.args or []),
            env=dict(stdio_cfg.env or {}),
            cwd=stdio_cfg.cwd,
        )

    async def _build_client(self) -> Any:
        candidate = self._client_factory(self.server_config)
        if asyncio.iscoroutine(candidate):
            return await candidate
        return candidate

    async def _ensure_connected(self) -> None:
        if not self._connected or not self._initialized or not self._is_client_running():
            await self.connect()

    def _is_client_running(self) -> bool:
        client = self._client
        if client is None:
            return False
        return bool(getattr(client, "is_running", False))

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise RuntimeError(f"External stdio adapter for '{self.server_id}' is not connected")
        timeout = float(self.server_config.timeouts.request_seconds)
        try:
            message = await asyncio.wait_for(client.call(method, params), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"External MCP request timed out for server '{self.server_id}' method '{method}'"
            ) from exc
        return {
            "jsonrpc": getattr(message, "jsonrpc", "2.0"),
            "id": getattr(message, "id", None),
            "result": getattr(message, "result", None),
            "error": getattr(message, "error", None),
        }

    async def _call_tool_with_ephemeral_env(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        runtime_auth: BrokeredExternalCredential,
    ) -> ExternalToolCallResult:
        def _prepare_config(ephemeral_config: ExternalMCPServerConfig) -> None:
            stdio_cfg = ephemeral_config.stdio
            if stdio_cfg is None:
                raise ValueError(f"Missing stdio config for server '{self.server_id}'")
            merged_env = dict(stdio_cfg.env or {})
            merged_env.update(dict(runtime_auth.env or {}))
            stdio_cfg.env = merged_env

        return await call_tool_with_ephemeral_adapter(
            server_config=self.server_config,
            adapter_factory=lambda config: StdioExternalMCPAdapter(
                config,
                client_factory=self._client_factory,
            ),
            prepare_config=_prepare_config,
            tool_name=tool_name,
            arguments=arguments,
        )
