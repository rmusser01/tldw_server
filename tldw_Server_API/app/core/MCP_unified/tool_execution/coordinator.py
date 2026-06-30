"""Coordinator for the MCP tools/call execution path."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..protocol_types import RequestContext
from .reporting import ToolExecutionReporter


@dataclass(slots=True)
class ToolExecutionCoordinator:
    """Coordinate prepare, execution, and prepare-failure reporting for tools/call."""

    prepare_tool_call_impl: Callable[..., Awaitable[Any]]
    execute_prepared_tool_call_impl: Callable[[Any], Awaitable[dict[str, Any]]]
    reporter: ToolExecutionReporter

    async def handle_tools_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
    ) -> dict[str, Any]:
        """Prepare and execute a tools/call request, reporting prepare failures."""

        start_ts = time.time()
        try:
            prepared = await self.prepare_tool_call(params=params, context=context)
        except Exception as exc:
            try:
                await self.reporter.record_prepare_failure(
                    context=context,
                    params=params,
                    exc=exc,
                    start_ts=start_ts,
                )
            except Exception as record_exc:
                logger.warning(
                    "Failed to build or record prepare-failure tool-use event: {}",
                    record_exc.__class__.__name__,
                )
            raise
        return await self.execute_prepared_tool_call(prepared)

    async def prepare_tool_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> Any:
        """Delegate prepare-time validation and policy checks to the configured implementation."""

        return await self.prepare_tool_call_impl(
            params=params,
            context=context,
            idempotency_key=idempotency_key,
        )

    async def execute_prepared_tool_call(self, prepared: Any) -> dict[str, Any]:
        """Delegate runtime execution for an already prepared tool call."""

        return await self.execute_prepared_tool_call_impl(prepared)
