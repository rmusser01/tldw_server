"""Chrome DevTools Protocol helpers for MCP browser inspection tools."""

from .client import CDPBrowserClient, CDPClientConfig, CDPClientError, CDPPageTarget

__all__ = [
    "CDPBrowserClient",
    "CDPClientConfig",
    "CDPClientError",
    "CDPPageTarget",
]
