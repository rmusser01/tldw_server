"""MCPStreamableHTTPTransport — streamable HTTP MCP transport.

Single HTTP endpoint.  Client POSTs JSON-RPC requests.  Server responds with
either ``application/json`` (single response) or ``text/event-stream`` (SSE
stream of JSON-RPC response events).  This transport handles both transparently.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transport import (
    MCPTransport,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.hardening import validate_mcp_http_url


class MCPStreamableHTTPTransport(MCPTransport):
    """MCP transport over the *Streamable HTTP* protocol.

    Connection flow:
    1. Create an HTTP client.
    2. Perform the MCP ``initialize`` handshake via POST.
    3. Send the ``initialized`` notification.
    4. Mark connected.
    """

    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout_sec: int = 30,
        allow_private_network: bool | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._headers = headers or {}
        self._timeout_sec = timeout_sec
        self._allow_private_network = allow_private_network
        self._http_client: httpx.AsyncClient | None = None
        self._next_id = 1
        self._connected = False

    # -- MCPTransport interface ------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Establish a connection: create HTTP client, run MCP handshake."""
        validate_mcp_http_url(
            self._endpoint,
            allow_private_network=self._allow_private_network,
            label="MCP streamable HTTP endpoint",
        )
        self._http_client = self._create_http_client()
        try:
            await self._json_rpc_call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "tldw_acp_harness", "version": "0.1.0"},
                    "capabilities": {},
                },
            )
            # Send the ``initialized`` notification (no response expected).
            await self._json_rpc_notify("initialized", {})
            self._connected = True
        except Exception:
            if self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None
            self._connected = False
            raise

    async def close(self) -> None:
        """Close the underlying HTTP client and mark disconnected."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        self._connected = False

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("Not connected")
        result = await self._json_rpc_call("tools/list", {})
        return result.get("tools", [])

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("Not connected")
        return await self._json_rpc_call(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )

    async def health_check(self) -> bool:
        return self._connected

    # -- Internal helpers ------------------------------------------------------

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create the async HTTP client. Extracted so tests can replace it."""
        return httpx.AsyncClient(
            headers=self._headers,
            timeout=self._timeout_sec,
        )

    async def _json_rpc_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC *request* (has ``id``) and return the result."""
        request_id = str(self._next_id)
        self._next_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        if self._http_client is None:
            raise RuntimeError("Not connected")
        validate_mcp_http_url(
            self._endpoint,
            allow_private_network=self._allow_private_network,
            label="MCP streamable HTTP endpoint",
        )
        resp = await self._http_client.post(self._endpoint, json=payload)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse_response(resp.text, request_id)

        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message", "RPC error"))
        return data.get("result", {})

    async def _json_rpc_notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC *notification* (no ``id``, no response expected)."""
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if self._http_client is None:
            raise RuntimeError("Not connected")
        validate_mcp_http_url(
            self._endpoint,
            allow_private_network=self._allow_private_network,
            label="MCP streamable HTTP endpoint",
        )
        resp = await self._http_client.post(self._endpoint, json=payload)
        resp.raise_for_status()

    def _parse_sse_response(self, text: str, expected_id: str) -> dict[str, Any]:
        """Parse SSE text body to extract the JSON-RPC response for *expected_id*."""
        for line in text.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if str(data.get("id")) == expected_id:
                    if "error" in data:
                        raise RuntimeError(
                            data["error"].get("message", "RPC error")
                        )
                    return data.get("result", {})
        raise RuntimeError("No matching response in SSE stream")
