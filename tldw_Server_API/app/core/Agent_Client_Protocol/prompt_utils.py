"""Prompt preprocessing utilities for ACP.

Handles @mention resolution for MCP tools.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPResponseError


def normalize_prompt_content(prompt: Any) -> list[dict[str, Any]]:
    """Validate authored/native prompts and return a detached ACP v1 wire copy.

    Authored roles and extra fields remain namespaced metadata; their text is
    unchanged. Native content blocks retain every field and their exact order.
    Callers must keep the authored input for governance, storage and bootstrap.
    """
    invalid = "ACP prompt requires valid content blocks or role/content messages"
    if not isinstance(prompt, list) or not prompt:
        raise ACPResponseError(invalid, code=-32602)
    try:
        encoded = json.dumps(prompt, allow_nan=False)
        if json.loads(encoded) != prompt:
            raise ValueError("Prompt contains values requiring JSON coercion")
    except (TypeError, ValueError) as exc:
        raise ACPResponseError(invalid, code=-32602) from exc

    blocks: list[dict[str, Any]] = []
    for item in prompt:
        if not isinstance(item, dict):
            raise ACPResponseError(invalid, code=-32602)
        if "type" not in item:
            if item.get("role") not in ("system", "user", "assistant") or not isinstance(item.get("content"), str):
                raise ACPResponseError(invalid, code=-32602)
            blocks.append({
                "type": "text",
                "text": item["content"],
                "_meta": {"tldw-server/legacy-message": copy.deepcopy({
                    key: value for key, value in item.items() if key != "content"
                })},
            })
            continue

        if "role" in item or "content" in item:
            raise ACPResponseError(invalid, code=-32602)
        if any(key in item and item[key] is not None and not isinstance(item[key], dict) for key in ("_meta", "annotations")):
            raise ACPResponseError(invalid, code=-32602)
        kind = item["type"]
        if kind == "text":
            valid = isinstance(item.get("text"), str)
        elif kind in ("image", "audio"):
            valid = isinstance(item.get("data"), str) and isinstance(item.get("mimeType"), str)
        elif kind == "resource_link":
            valid = isinstance(item.get("uri"), str) and isinstance(item.get("name"), str)
        elif kind == "resource":
            resource = item.get("resource")
            valid = (
                isinstance(resource, dict)
                and isinstance(resource.get("uri"), str)
                and (("text" in resource) != ("blob" in resource))
                and isinstance(resource.get("text", resource.get("blob")), str)
            )
        else:
            valid = False
        if not valid:
            raise ACPResponseError(invalid, code=-32602)
        blocks.append(copy.deepcopy(item))
    return blocks


# Match @tool_name patterns (alphanumeric, hyphens, underscores, dots).
# The negative lookbehind ``(?<!\w)`` prevents matching email addresses
# such as ``user@example.com`` because the ``@`` is preceded by a word
# character.
_AT_MENTION_RE = re.compile(r"(?<!\w)@([\w.-]+)")


async def preprocess_mentions(
    messages: list[dict[str, Any]],
    tool_registry: Any | None = None,
    cache: dict[str, bool] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse ``@tool_name`` patterns in *messages* and resolve them.

    Parameters
    ----------
    messages:
        List of message dicts (``role`` / ``content``).
    tool_registry:
        Object with an async ``tool_exists(name: str) -> bool`` method.
        When *None*, all ``@mentions`` are accepted without validation.
    cache:
        Optional per-session cache mapping tool names to their resolved
        status (``True`` = exists, ``False`` = unresolved).  Supplying
        the same dict across turns avoids redundant registry lookups.

    Returns
    -------
    tuple[list[dict[str, Any]], list[str]]
        The *messages* list (unchanged) and a **sorted** list of resolved
        tool hint names.
    """
    if cache is None:
        cache = {}

    tool_hints: set[str] = set()
    unresolved: set[str] = set()

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for match in _AT_MENTION_RE.finditer(content):
            name = match.group(1)

            # Fast path: already resolved in this session.
            if name in cache:
                if cache[name]:
                    tool_hints.add(name)
                else:
                    unresolved.add(name)
                continue

            # Resolve against the registry when available.
            if tool_registry is not None:
                try:
                    exists = await tool_registry.tool_exists(name)
                except Exception:
                    logger.debug("Failed to resolve @mention: {}", name)
                    exists = False
                cache[name] = exists
            else:
                # No registry — accept all mentions.
                cache[name] = True
                exists = True

            if exists:
                tool_hints.add(name)
            else:
                unresolved.add(name)

    if unresolved:
        logger.debug("Unresolved @mentions: {}", unresolved)

    return messages, sorted(tool_hints)
