"""
Usage-aggregator shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageAggregatorShutdownHandles:
    """Updated usage-related task handles after shutdown processing."""

    usage_task: Any | None = None
    llm_usage_task: Any | None = None


async def stop_usage_aggregators(
    *,
    coordinated_legacy_component_names: set[str],
    stopped_background_worker_names: set[str] | None = None,
    usage_task: Any | None,
    llm_usage_task: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> UsageAggregatorShutdownHandles:
    """Stop usage and LLM usage aggregators while preserving legacy fallback semantics."""
    stopped_background_worker_names = stopped_background_worker_names or set()

    if "usage_aggregator" in stopped_background_worker_names:
        usage_task = None
    elif "usage_aggregator" not in coordinated_legacy_component_names and usage_task:
        try:
            await _stop_usage_aggregator_service(usage_task)
            usage_task = None
        except guard_exceptions:
            try:
                usage_task.cancel()
            except guard_exceptions:
                pass
            usage_task = None

    if "llm_usage_aggregator" in stopped_background_worker_names:
        llm_usage_task = None
    elif "llm_usage_aggregator" not in coordinated_legacy_component_names and llm_usage_task:
        try:
            await _stop_llm_usage_aggregator_service(llm_usage_task)
            llm_usage_task = None
        except guard_exceptions:
            try:
                llm_usage_task.cancel()
            except guard_exceptions:
                pass
            llm_usage_task = None

    return UsageAggregatorShutdownHandles(
        usage_task=usage_task,
        llm_usage_task=llm_usage_task,
    )


async def _stop_usage_aggregator_service(task: Any) -> None:
    from tldw_Server_API.app.services.usage_aggregator import stop_usage_aggregator

    await stop_usage_aggregator(task)


async def _stop_llm_usage_aggregator_service(task: Any) -> None:
    from tldw_Server_API.app.services.llm_usage_aggregator import stop_llm_usage_aggregator

    await stop_llm_usage_aggregator(task)
