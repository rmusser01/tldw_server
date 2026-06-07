"""Tool-use recorder contracts for MCP reporting sinks."""

from __future__ import annotations

import asyncio
from typing import Protocol

from loguru import logger

from mcp_unified.tool_use_reporting.models import ToolUseEvent
from mcp_unified.tool_use_reporting.store import ToolUseEventStore


class ToolUseRecorder(Protocol):
    """Async sink for metadata-only MCP tool-use events."""

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        """Persist or forward one metadata-only tool-use event."""


class NoopToolUseRecorder:
    """Recorder implementation used when a host has not configured reporting."""

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        """Accept an event without persisting it."""

        del event


class StoreBackedToolUseRecorder:
    """Recorder that writes tool-use events to an injected async store."""

    def __init__(
        self,
        store: ToolUseEventStore,
        *,
        timeout_seconds: float | None = 2.0,
    ) -> None:
        self._store = store
        self._timeout_seconds = timeout_seconds

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        """Record an event without letting sink failures break tool execution."""

        try:
            if self._timeout_seconds is None:
                await self._store.append_event(event)
                return
            await asyncio.wait_for(
                self._store.append_event(event),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - exercised through host sinks.
            logger.warning(
                "MCP tool-use recorder failed; event dropped. error_class={}",
                exc.__class__.__name__,
            )


async def record_tool_use_safely(
    recorder: ToolUseRecorder,
    event: ToolUseEvent,
    *,
    timeout_seconds: float | None = 2.0,
) -> None:
    """Record an event without allowing recorder failure to affect tool behavior."""

    try:
        if timeout_seconds is None:
            await recorder.record_tool_use(event)
            return
        await asyncio.wait_for(
            recorder.record_tool_use(event),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        logger.warning(
            "MCP tool-use recorder failed; event dropped. error_class={}",
            exc.__class__.__name__,
        )
