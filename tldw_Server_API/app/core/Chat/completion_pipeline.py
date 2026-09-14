# completion_pipeline.py
# Description: Coordinator boundary for Chat completion execution.
"""Coordinator object for Chat completion execution paths."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ChatCompletionPipeline:
    """Delegates Chat completion execution to focused executors."""

    non_stream_executor: Callable[..., Awaitable[dict[str, Any]]]
    streaming_executor: Callable[..., Any]

    async def execute_non_stream(self, **kwargs: Any) -> dict[str, Any]:
        return await self.non_stream_executor(**kwargs)

    def execute_streaming(self, **kwargs: Any) -> Any:
        return self.streaming_executor(**kwargs)


__all__ = ["ChatCompletionPipeline"]
