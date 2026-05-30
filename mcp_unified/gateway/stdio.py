"""Client-facing stdio transport helpers for standalone MCP gateway runtimes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .jsonrpc import (
    GATEWAY_RESPONSE_TYPES as _GATEWAY_RESPONSE_TYPES,
)
from .jsonrpc import (
    GatewayJSONRPCResult,
    GatewayNoResponse,
    handle_jsonrpc,
)
from .jsonrpc import (
    parse_json_payload as _parse_json_payload,
)
from .jsonrpc import (
    response_to_json as _response_to_json,
)
from .runtime import GatewayRuntime

STDIO_PATH = "stdio://stdin"


@dataclass(slots=True)
class GatewayStdioServer:
    """Line-delimited JSON-RPC stdio transport for a gateway runtime."""

    runtime: GatewayRuntime
    path: str = STDIO_PATH
    metadata: dict[str, Any] = field(default_factory=dict)

    async def handle_line(self, line: str | bytes) -> str | None:
        """Handle one JSON-RPC input line and return one output line when needed."""

        payload = _parse_json_payload(line)
        if isinstance(payload, _GATEWAY_RESPONSE_TYPES):
            response: GatewayJSONRPCResult = payload
        else:
            response = await handle_jsonrpc(
                self.runtime,
                payload,
                path=self.path,
                metadata={"transport": "stdio", **self.metadata},
            )
        if isinstance(response, GatewayNoResponse):
            return None
        return _serialize_stdio_response(response)


async def handle_stdio_line(
    runtime: GatewayRuntime,
    line: str | bytes,
    *,
    path: str = STDIO_PATH,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Handle one stdin-style JSON-RPC line for a gateway runtime."""

    server = GatewayStdioServer(runtime=runtime, path=path, metadata=metadata or {})
    return await server.handle_line(line)


def _serialize_stdio_response(response: GatewayJSONRPCResult) -> str:
    """Serialize one stdio JSON-RPC response line."""

    if isinstance(response, list):
        payload: Any = [_response_to_json(item) for item in response]
    else:
        payload = _response_to_json(response)
    return json.dumps(payload, separators=(",", ":")) + "\n"


__all__ = [
    "STDIO_PATH",
    "GatewayStdioServer",
    "handle_stdio_line",
]
