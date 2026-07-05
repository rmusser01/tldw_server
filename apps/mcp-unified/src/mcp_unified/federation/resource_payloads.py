"""Shared normalization helpers for external MCP resource payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_external_resource_list(result: Any) -> list[dict[str, Any]]:
    """Return normalized resource descriptors from an MCP resources/list result."""

    if isinstance(result, dict):
        raw_resources = result.get("resources") or []
    elif isinstance(result, list):
        raw_resources = result
    else:
        raw_resources = []

    resources: list[dict[str, Any]] = []
    for item in raw_resources:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            continue
        name = item.get("name")
        description = item.get("description")
        mime_type = item.get("mimeType")
        metadata = item.get("metadata")
        resources.append(
            {
                "uri": uri,
                "name": name if isinstance(name, str) else "",
                "description": description if isinstance(description, str) else "",
                "mimeType": mime_type if isinstance(mime_type, str) else None,
                "metadata": deepcopy(metadata) if isinstance(metadata, dict) else {},
            }
        )
    return resources


def normalize_external_resource_read(result: Any) -> dict[str, Any]:
    """Return a caller-owned resources/read result payload."""

    return deepcopy(result) if isinstance(result, dict) else {"contents": []}
