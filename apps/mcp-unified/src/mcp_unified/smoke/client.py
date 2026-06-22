"""Small JSON-RPC client facade for MCP smoke tests."""

from __future__ import annotations

from mcp_unified.smoke.exceptions import McpSmokeClientError
from mcp_unified.smoke.types import JsonObject, McpSmokeTransport


class McpSmokeClient:
    """Minimal async MCP JSON-RPC client with stable smoke request ids."""

    def __init__(self, transport: McpSmokeTransport) -> None:
        self._transport = transport
        self._next_request_number = 1

    async def request(self, method: str, params: JsonObject | None = None) -> object:
        """Send a request and return its JSON-RPC result."""

        request_id = self._next_id()
        payload = self._build_payload(method=method, request_id=request_id, params=params)
        response = await self._transport.request(payload)
        return self._extract_result(response, request_id=request_id)

    async def notify(self, method: str, params: JsonObject | None = None) -> object | None:
        """Send a notification without a JSON-RPC id and return any response."""

        payload = self._build_payload(method=method, request_id=None, params=params)
        return await self._transport.notify(payload)

    async def initialize(self, client_name: str = "mcp-smoke-client") -> object:
        """Initialize an MCP session."""

        return await self.request(
            "initialize",
            {
                "clientInfo": {"name": client_name},
                "capabilities": {},
            },
        )

    async def ping(self) -> object:
        """Send a JSON-RPC ping request."""

        return await self.request("ping")

    async def list_tools(self) -> object:
        """List server tools."""

        return await self.request("tools/list")

    async def call_tool(self, name: str, arguments: JsonObject) -> object:
        """Call one MCP tool with caller-supplied arguments."""

        return await self.request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )

    async def list_resources(self) -> object:
        """List server resources."""

        return await self.request("resources/list")

    async def read_resource(self, uri: str) -> object:
        """Read one MCP resource by URI."""

        return await self.request("resources/read", {"uri": uri})

    async def list_prompts(self) -> object:
        """List server prompts."""

        return await self.request("prompts/list")

    async def get_prompt(self, name: str, arguments: JsonObject | None = None) -> object:
        """Get one MCP prompt by name."""

        params: JsonObject = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return await self.request("prompts/get", params)

    def _next_id(self) -> str:
        request_id = f"smoke-{self._next_request_number}"
        self._next_request_number += 1
        return request_id

    @staticmethod
    def _build_payload(
        *,
        method: str,
        request_id: str | None,
        params: JsonObject | None,
    ) -> JsonObject:
        payload: JsonObject = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if request_id is not None:
            payload["id"] = request_id
        if params is not None:
            payload["params"] = params
        return payload

    @staticmethod
    def _extract_result(response: object | None, *, request_id: str) -> object:
        if not isinstance(response, dict):
            raise McpSmokeClientError(
                "JSON-RPC response must be an object",
                response=response,
            )
        if response.get("jsonrpc") != "2.0":
            raise McpSmokeClientError(
                "JSON-RPC response has an invalid version",
                response=response,
            )
        if response.get("id") != request_id:
            raise McpSmokeClientError(
                "JSON-RPC response id does not match the request id",
                response=response,
            )

        has_result = "result" in response
        has_error = "error" in response
        if has_result == has_error:
            raise McpSmokeClientError(
                "JSON-RPC response must contain exactly one of result or error",
                response=response,
            )
        if has_error:
            raise McpSmokeClientError(
                "JSON-RPC error response",
                response=response,
                error=response.get("error"),
            )
        return response["result"]
