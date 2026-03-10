"""
Managed vLLM jobs worker shutdown helper.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class VLLMManagementShutdownHandles:
    """Updated managed vLLM jobs worker handles after shutdown processing."""

    vllm_management_task: Any | None = None
    vllm_management_stop_event: Any | None = None


async def shutdown_vllm_management_worker(
    *,
    vllm_management_task: Any | None,
    vllm_management_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> VLLMManagementShutdownHandles:
    """Stop the managed vLLM jobs worker while preserving late-stop semantics."""
    await _shutdown_vllm_management_worker(
        task=vllm_management_task,
        stop_event=vllm_management_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return VLLMManagementShutdownHandles(
        vllm_management_task=vllm_management_task,
        vllm_management_stop_event=vllm_management_stop_event,
    )


async def _shutdown_vllm_management_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if task is None:
        return
    if not should_run_late_stop("vllm_management_task", task):
        return
    fallback_exceptions = (asyncio.TimeoutError,) + guard_exceptions
    if stop_event is not None:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info("Managed vLLM Jobs worker stopped via stop_event")
        except fallback_exceptions:
            _safe_cancel_task(task, guard_exceptions=guard_exceptions)
    else:
        _safe_cancel_task(task, guard_exceptions=guard_exceptions)


def _safe_cancel_task(
    task: Any,
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        task.cancel()
    except guard_exceptions:
        pass


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)
