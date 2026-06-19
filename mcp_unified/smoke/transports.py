"""Transport adapters for MCP smoke JSON-RPC flows."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from mcp_unified.gateway.jsonrpc import (
    GATEWAY_RESPONSE_TYPES,
    GatewayNoResponse,
    handle_jsonrpc,
    response_to_json,
)
from mcp_unified.gateway.runtime import GatewayRuntime

JsonObject = dict[str, object]
JsonRpcPayload = JsonObject | list[object]


class McpSmokeTransport(Protocol):
    """Async transport protocol consumed by MCP smoke scenarios."""

    async def start(self) -> None:
        """Open any resources needed by this transport."""

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """Send a JSON-RPC request or batch payload and return the decoded response."""

    async def notify(self, payload: JsonObject) -> None:
        """Send a JSON-RPC notification payload."""

    async def close(self) -> None:
        """Release any resources opened by this transport."""


class InProcessGatewayTransport:
    """Call the standalone gateway JSON-RPC handler directly in-process."""

    def __init__(
        self,
        runtime: GatewayRuntime,
        *,
        path: str = "inprocess://mcp/request",
        client_host: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.path = path
        self.client_host = client_host
        self.metadata = dict(metadata or {})
        self._started = False

    async def start(self) -> None:
        """Mark the no-op in-process transport as started."""

        self._started = True

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """Dispatch one JSON-RPC payload through `handle_jsonrpc`."""

        response = await handle_jsonrpc(
            self.runtime,
            payload,
            path=self.path,
            client_host=self.client_host,
            metadata=self.metadata,
        )
        return _gateway_response_to_plain_json(response)

    async def notify(self, payload: JsonObject) -> None:
        """Dispatch a notification payload and ignore the no-response sentinel."""

        await self.request(payload)

    async def close(self) -> None:
        """Mark the no-op in-process transport as closed."""

        self._started = False


class InProcessFastApiTransport:
    """Call a FastAPI MCP JSON-RPC route through `httpx.ASGITransport`."""

    def __init__(
        self,
        app: Any,
        *,
        request_path: str = "/api/v1/mcp/request",
        base_url: str = "http://mcp-smoke.local",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.app = app
        self.request_path = request_path
        self.base_url = base_url
        self.headers = dict(headers or {})
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create the ASGI-backed async HTTP client."""

        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url=self.base_url,
            headers=self.headers,
        )

    async def request(self, payload: JsonRpcPayload) -> object | None:
        """POST one JSON-RPC request or batch to the configured ASGI route."""

        client = await self._started_client()
        response = await client.post(self.request_path, json=payload)
        if response.status_code == 204:
            return None
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    async def notify(self, payload: JsonObject) -> None:
        """POST one notification to the configured ASGI route."""

        await self.request(payload)

    async def close(self) -> None:
        """Close the ASGI-backed async HTTP client."""

        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    async def _started_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.start()
        if self._client is None:  # pragma: no cover - defensive guard
            raise RuntimeError("In-process FastAPI transport failed to start")
        return self._client


def _gateway_response_to_plain_json(response: object) -> object | None:
    if isinstance(response, GatewayNoResponse):
        return None
    if isinstance(response, list):
        return [_gateway_response_to_plain_json(item) for item in response]
    if isinstance(response, GATEWAY_RESPONSE_TYPES):
        return response_to_json(response)
    return response


__all__ = [
    "InProcessFastApiTransport",
    "InProcessGatewayTransport",
    "JsonObject",
    "JsonRpcPayload",
    "McpSmokeTransport",
]
