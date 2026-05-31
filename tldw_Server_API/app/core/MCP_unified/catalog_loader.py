"""
Compatibility wrapper for the standalone MCP catalog loader.

The implementation lives in :mod:`mcp_unified.federation.catalog_loader` so
standalone MCP hosts can use the same YAML catalog loading behavior without
importing the tldw_server API package.
"""
from __future__ import annotations

from mcp_unified.federation.catalog_loader import (
    _CATALOG_CACHE,
    get_catalog_entry,
    list_catalog_entries,
    load_mcp_catalog,
)

__all__ = [
    "_CATALOG_CACHE",
    "get_catalog_entry",
    "list_catalog_entries",
    "load_mcp_catalog",
]
