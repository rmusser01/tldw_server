"""
Archetype YAML Loader Service.

Loads persona archetype templates from YAML files, validates them with
Pydantic, and caches them in memory for fast access by API endpoints.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.archetype_schemas import (
    ArchetypeSummary,
    ArchetypeTemplate,
)

# Module-level cache
_CACHE: dict[str, ArchetypeTemplate] = {}


def load_archetypes_from_directory(
    directory: str | Path,
) -> dict[str, ArchetypeTemplate]:
    """Load all ``*.yaml`` files from *directory*, validate via Pydantic, and cache.

    Malformed files are logged with loguru and skipped so that one bad
    file does not prevent the rest from loading.  The glob results are
    sorted to guarantee deterministic ordering across platforms.

    The cache is replaced atomically so that concurrent readers never see
    a partially-populated state.  This function is intended to be called
    once at startup; it is **not** designed for concurrent hot-reload.

    Parameters
    ----------
    directory:
        Path to a directory containing archetype YAML files.

    Returns
    -------
    dict[str, ArchetypeTemplate]
        Mapping of archetype *key* to its validated template.
    """
    global _CACHE

    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning("Archetype directory does not exist: {}", dir_path)
        _CACHE = {}
        return {}

    new_cache: dict[str, ArchetypeTemplate] = {}
    yaml_files = sorted(dir_path.glob("*.yaml"))
    for yaml_file in yaml_files:
        try:
            raw = yaml_file.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
            if not isinstance(data, dict) or "archetype" not in data:
                logger.warning(
                    "Skipping {}: missing top-level 'archetype' key", yaml_file.name
                )
                continue
            template = ArchetypeTemplate(**data["archetype"])
            if template.key in new_cache:
                logger.warning(
                    "Skipping duplicate archetype '{}' from {}; entry already loaded",
                    template.key,
                    yaml_file.name,
                )
                continue
            new_cache[template.key] = template
            logger.debug("Loaded archetype '{}' from {}", template.key, yaml_file.name)
        except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError):
            logger.opt(exception=True).warning(
                "Skipping malformed archetype file: {}", yaml_file.name
            )

    # Atomic replacement — readers never see a half-populated cache.
    _CACHE = new_cache
    logger.info("Loaded {} archetype(s) from {}", len(_CACHE), dir_path)
    return dict(_CACHE)


def list_archetypes() -> list[ArchetypeSummary]:
    """Return a summary (key, label, tagline, icon) of all cached archetypes."""
    return [
        ArchetypeSummary(
            key=t.key,
            label=t.label,
            tagline=t.tagline,
            icon=t.icon,
        )
        for t in _CACHE.values()
    ]


def get_archetype(key: str) -> ArchetypeTemplate | None:
    """Return the cached archetype identified by *key*, or ``None``."""
    return _CACHE.get(key)
