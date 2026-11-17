"""FastAPI dependency that hands the MCP HTTP routes the container's MCP server."""

from __future__ import annotations

from tldw_Server_API.app.core.MCP_unified import MCPServer, get_mcp_server


async def get_mcp_server_dep() -> MCPServer:
    # async so a first-time MCPServer() is built on the event loop, not in the threadpool
    return get_mcp_server()
