"""
Startup catalog loading extracted from the application lifespan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_startup_catalogs(
    *,
    module_file: str,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        config_dir = _resolve_config_dir(module_file)
        _load_archetypes_from_directory(config_dir / "persona_archetypes")
        _load_mcp_catalog(config_dir / "mcp_server_catalog.yaml")
    except startup_guard_exceptions + import_exceptions as exc:
        logger.debug("Archetype/catalog loading skipped: {}", exc)


def _resolve_config_dir(module_file: str) -> Path:
    return Path(module_file).resolve().parent.parent / "Config_Files"


def _load_archetypes_from_directory(path: Path) -> None:
    from tldw_Server_API.app.core.Persona.archetype_loader import load_archetypes_from_directory

    load_archetypes_from_directory(path)


def _load_mcp_catalog(path: Path) -> None:
    from tldw_Server_API.app.core.MCP_unified.catalog_loader import load_mcp_catalog

    load_mcp_catalog(path)
