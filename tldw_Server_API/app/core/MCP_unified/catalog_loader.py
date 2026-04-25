"""
MCP Server Catalog Loader.

Loads curated external MCP server catalog entries from a YAML file,
validates them with Pydantic, and caches them in memory for fast access
by API endpoints and the setup wizard.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.archetype_schemas import MCPCatalogEntry

# Module-level cache
_CATALOG_CACHE: list[MCPCatalogEntry] = []


def load_mcp_catalog(path: str | Path) -> list[MCPCatalogEntry]:
    """Load catalog entries from a YAML file, validate via Pydantic, and cache.

    The YAML file must contain a top-level ``catalog`` key whose value is
    a list of mapping objects.  Each mapping is validated as an
    :class:`MCPCatalogEntry`.  Malformed entries are logged with loguru
    and skipped so that one bad entry does not prevent the rest from
    loading.

    The cache is replaced atomically so that concurrent readers never see
    a partially-populated state.  This function is intended to be called
    once at startup; it is **not** designed for concurrent hot-reload.

    Parameters
    ----------
    path:
        Path to the YAML catalog file.

    Returns
    -------
    list[MCPCatalogEntry]
        The validated (and now cached) catalog entries.
    """
    global _CATALOG_CACHE

    file_path = Path(path)
    if not file_path.is_file():
        logger.warning("MCP catalog file does not exist: {}", file_path)
        _CATALOG_CACHE = []
        return []

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError):
        logger.opt(exception=True).warning(
            "Failed to read/parse MCP catalog file: {}", file_path
        )
        _CATALOG_CACHE = []
        return []

    if not isinstance(data, dict) or "catalog" not in data:
        logger.warning(
            "Skipping {}: missing top-level 'catalog' key", file_path.name
        )
        _CATALOG_CACHE = []
        return _CATALOG_CACHE

    entries = data["catalog"]
    if not isinstance(entries, list):
        logger.warning(
            "Skipping {}: 'catalog' value is not a list", file_path.name
        )
        _CATALOG_CACHE = []
        return _CATALOG_CACHE

    new_cache: list[MCPCatalogEntry] = []
    for idx, entry_data in enumerate(entries):
        try:
            entry = MCPCatalogEntry(**entry_data)
            new_cache.append(entry)
            logger.debug("Loaded MCP catalog entry '{}'", entry.key)
        except ValidationError:
            logger.opt(exception=True).warning(
                "Skipping malformed MCP catalog entry at index {}", idx
            )

    # Atomic replacement — readers never see a half-populated cache.
    _CATALOG_CACHE = new_cache
    logger.info(
        "Loaded {} MCP catalog entry/entries from {}", len(_CATALOG_CACHE), file_path
    )
    return list(_CATALOG_CACHE)


def list_catalog_entries(
    archetype_key: str | None = None,
) -> list[MCPCatalogEntry]:
    """Return cached catalog entries, optionally filtered by archetype key.

    Parameters
    ----------
    archetype_key:
        If provided, only entries whose ``suggested_for`` list contains
        this value are returned.  If ``None``, all entries are returned.

    Returns
    -------
    list[MCPCatalogEntry]
    """
    if archetype_key is None:
        return list(_CATALOG_CACHE)
    return [e for e in _CATALOG_CACHE if archetype_key in e.suggested_for]


def get_catalog_entry(key: str) -> MCPCatalogEntry | None:
    """Return the cached catalog entry identified by *key*, or ``None``."""
    for entry in _CATALOG_CACHE:
        if entry.key == key:
            return entry
    return None
