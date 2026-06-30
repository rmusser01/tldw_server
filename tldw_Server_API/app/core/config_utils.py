# config_utils.py
"""Shared helpers for module configuration loading and merging."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from tldw_Server_API.app.core.config_paths import resolve_module_yaml


def _parse_scalar(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except Exception:
        return value
    return value if parsed is None else parsed


def section_to_nested_dict(section: dict[str, str]) -> dict[str, Any]:
    """Convert a flat config section into a nested dict using dot notation."""
    nested: dict[str, Any] = {}
    for raw_key, raw_value in section.items():
        key = raw_key.strip()
        value = _parse_scalar(str(raw_value))
        parts = [part for part in key.split(".") if part]
        if not parts:
            continue
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def load_module_yaml(
    module_name: str,
    filename_override: str | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Load module YAML config using the shared config root."""
    path = resolve_module_yaml(module_name, filename_override=filename_override)
    if path is None:
        return {}, None

    if not path.exists():
        logger.debug(f"Module YAML not found for {module_name}: {path}")
        return {}, path

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            logger.debug(f"Loaded module YAML for {module_name}: {path}")
            return data, path
        logger.warning(f"Module YAML for {module_name} is not a mapping: {path}")
        return {}, path
    except Exception:
        logger.warning(f"Failed to load module YAML for {module_name}")
        return {}, path


def _merge_layer(
    base: dict[str, Any],
    layer: dict[str, Any],
    source: str,
    sources: dict[str, Any],
) -> None:
    for key, value in layer.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            nested_sources = sources.get(key)
            if not isinstance(nested_sources, dict):
                nested_sources = {}
            _merge_layer(base[key], value, source, nested_sources)
            sources[key] = nested_sources
        else:
            base[key] = value
            sources[key] = source


def merge_config_layers(
    layers: Iterable[tuple[str, dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge config layers and return merged config + source map."""
    merged: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for source, layer in layers:
        if not layer:
            continue
        _merge_layer(merged, layer, source, sources)
    return merged, sources


def apply_default_sources(
    values: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing source tags with 'default'."""
    for key, value in values.items():
        if key not in sources:
            if isinstance(value, dict):
                sources[key] = apply_default_sources(value, {})
            else:
                sources[key] = "default"
            continue

        if isinstance(value, dict) and isinstance(sources.get(key), dict):
            apply_default_sources(value, sources[key])
    return sources
