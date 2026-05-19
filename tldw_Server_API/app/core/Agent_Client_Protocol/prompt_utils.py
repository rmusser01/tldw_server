"""Prompt preprocessing utilities for ACP.

Handles @mention resolution for MCP tools.
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger


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
