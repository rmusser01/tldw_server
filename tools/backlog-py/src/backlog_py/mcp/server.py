from __future__ import annotations

import importlib.util


def is_mcp_sdk_available() -> bool:
    """Return whether the optional MCP SDK can be imported in this environment."""
    return importlib.util.find_spec("mcp") is not None


def main() -> None:
    """Start the MCP server when a future task adds the optional SDK adapter."""
    if not is_mcp_sdk_available():
        raise RuntimeError(
            "MCP SDK is not installed. Task 6 provides only pure read-only "
            "registry functions; install and wire the SDK in a later task."
        )
    raise RuntimeError("MCP SDK adapter is not implemented in Task 6.")
