"""
MCP server catalog loader.

Loads curated external MCP server catalog entries from a YAML file,
validates them with Pydantic, and caches them in memory for fast access
by front-end setup flows and standalone MCP server hosts.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import MCPCatalogEntry

logger = logging.getLogger(__name__)

# Module-level cache
_CATALOG_CACHE: list[MCPCatalogEntry] = []


def _replace_cache(entries: list[MCPCatalogEntry]) -> list[MCPCatalogEntry]:
    """Replace the catalog cache in place and return a caller-owned list."""
    _CATALOG_CACHE.clear()
    _CATALOG_CACHE.extend(entries)
    return list(_CATALOG_CACHE)


def load_mcp_catalog(path: str | Path) -> list[MCPCatalogEntry]:
    """Load catalog entries from a YAML file, validate via Pydantic, and cache.

    The YAML file must contain a top-level ``catalog`` key whose value is
    a list of mapping objects. Each mapping is validated as an
    :class:`MCPCatalogEntry`. Malformed entries are logged and skipped so
    that one bad entry does not prevent the rest from loading.

    The cache is replaced in place so host compatibility wrappers keep a
    shared cache object. This function is intended to be called once at
    startup; it is not designed for concurrent hot-reload.
    """
    file_path = Path(path)
    if not file_path.is_file():
        logger.warning("MCP catalog file does not exist: %s", file_path)
        return _replace_cache([])

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError):
        logger.warning(
            "Failed to read/parse MCP catalog file: %s",
            file_path,
            exc_info=True,
        )
        return _replace_cache([])

    if not isinstance(data, dict) or "catalog" not in data:
        logger.warning("Skipping %s: missing top-level 'catalog' key", file_path.name)
        return _replace_cache([])

    entries = data["catalog"]
    if not isinstance(entries, list):
        logger.warning("Skipping %s: 'catalog' value is not a list", file_path.name)
        return _replace_cache([])

    new_cache: list[MCPCatalogEntry] = []
    for idx, entry_data in enumerate(entries):
        try:
            entry = MCPCatalogEntry(**entry_data)
            new_cache.append(entry)
            logger.debug("Loaded MCP catalog entry '%s'", entry.key)
        except ValidationError:
            logger.warning(
                "Skipping malformed MCP catalog entry at index %s",
                idx,
                exc_info=True,
            )

    _replace_cache(new_cache)
    logger.info("Loaded %s MCP catalog entry/entries from %s", len(_CATALOG_CACHE), file_path)
    return list(_CATALOG_CACHE)


def list_catalog_entries(
    archetype_key: str | None = None,
) -> list[MCPCatalogEntry]:
    """Return cached catalog entries, optionally filtered by archetype key."""
    if archetype_key is None:
        return list(_CATALOG_CACHE)
    return [e for e in _CATALOG_CACHE if archetype_key in e.suggested_for]


def get_catalog_entry(key: str) -> MCPCatalogEntry | None:
    """Return the cached catalog entry identified by *key*, or ``None``."""
    for entry in _CATALOG_CACHE:
        if entry.key == key:
            return entry
    return None
