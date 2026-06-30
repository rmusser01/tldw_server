"""Batch execution utilities with semaphore-based rate limiting.

Provides reusable patterns for running batch operations concurrently
with configurable concurrency limits, cancel-on-error support, and
ordered result collection.

Key patterns:
- ``run_batch``: Execute async callables with semaphore-based concurrency
- ``run_batch_indexed``: Same but preserves original ordering
- Cancel event for fail_fast early termination

Ported from RAGnarok-AI's concurrent evaluation pattern, adapted for tldw_server2.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class BatchResult:
    """Result of a batch operation.

    Attributes:
        results: Successfully completed results (in original order if indexed).
        results_by_index: Mapping of original index to result for successful items.
        errors: List of (index, exception) tuples for failed items.
        total: Total items attempted.
        completed: Number successfully completed.
        cancelled: Whether the batch was cancelled early.
    """

    results: list[Any] = field(default_factory=list)
    results_by_index: dict[int, Any] = field(default_factory=dict)
    errors: list[tuple[int, Exception]] = field(default_factory=list)
    total: int = 0
    completed: int = 0
    cancelled: bool = False

    @property
    def success_rate(self) -> float:
        """Fraction of items that completed successfully."""
        return self.completed / self.total if self.total > 0 else 0.0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def ordered_results_with_errors(self, default: Any = None) -> list[Any]:
        """Return results in original order with errors slotted in.

        For each index 0..total-1, returns the successful result if
        available, the Exception if the item failed, or *default* if
        the item was skipped (e.g. due to cancellation).
        """
        err_map = dict(self.errors)
        out: list[Any] = []
        if self.results_by_index:
            for idx in range(self.total):
                if idx in err_map:
                    out.append(err_map[idx])
                elif idx in self.results_by_index:
                    out.append(self.results_by_index[idx])
                else:
                    out.append(default)
            return out

        success_iter = iter(self.results)
        for idx in range(self.total):
            if idx in err_map:
                out.append(err_map[idx])
            else:
                out.append(next(success_iter, default))
        return out


async def run_batch(
    items: list[Any],
    func: Callable[[Any], Awaitable[T]],
    max_concurrency: int = 5,
    fail_fast: bool = False,
    on_progress: Callable[[int, int], Any] | None = None,
) -> BatchResult:
    """Execute an async function over a batch of items with concurrency control.

    Args:
        items: List of items to process.
        func: Async function to apply to each item.
        max_concurrency: Maximum concurrent tasks. Defaults to 5.
        fail_fast: If True, cancel remaining tasks on first error.
        on_progress: Optional callback(completed, total) called after each item.

    Returns:
        BatchResult with ordered results and any errors.

    Example::

        async def embed_doc(doc):
            return await embedding_model.embed(doc.content)

        result = await run_batch(
            items=documents,
            func=embed_doc,
            max_concurrency=10,
            fail_fast=False,
        )
        print(f"Embedded {result.completed}/{result.total} documents")
    """
    if max_concurrency < 1:
        max_concurrency = 1

    total = len(items)
    if total == 0:
        return BatchResult(total=0, completed=0)

    semaphore = asyncio.Semaphore(max_concurrency)
    cancel_event = asyncio.Event()
    results_dict: dict[int, Any] = {}
    errors: list[tuple[int, Exception]] = []
    completed_count = 0
    lock = asyncio.Lock()

    async def process_item(index: int, item: Any) -> None:
        nonlocal completed_count

        if cancel_event.is_set():
            return

        async with semaphore:
            if cancel_event.is_set():
                return

            try:
                result = await func(item)
                results_dict[index] = result

                async with lock:
                    completed_count += 1
                    current = completed_count

                if on_progress:
                    callback_result = on_progress(current, total)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result

            except Exception as e:
                async with lock:
                    errors.append((index, e))

                if fail_fast:
                    cancel_event.set()
                    logger.warning(
                        f"Batch item {index} failed (fail_fast=True), "
                        "cancelling remaining"
                    )
                else:
                    logger.warning(f"Batch item {index} failed")

    # Create and run tasks
    tasks = [
        asyncio.create_task(process_item(i, item))
        for i, item in enumerate(items)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Build ordered results
    ordered_results = [
        results_dict[i]
        for i in range(total)
        if i in results_dict
    ]

    return BatchResult(
        results=ordered_results,
        results_by_index=results_dict,
        errors=errors,
        total=total,
        completed=completed_count,
        cancelled=cancel_event.is_set(),
    )


async def run_batch_indexed(
    items: list[Any],
    func: Callable[[int, Any], Awaitable[T]],
    max_concurrency: int = 5,
    fail_fast: bool = False,
    on_progress: Callable[[int, int], Any] | None = None,
) -> BatchResult:
    """Like ``run_batch`` but the function receives (index, item).

    Useful when the processing function needs to know the item's position
    (e.g., for checkpoint tracking).

    Args:
        items: List of items to process.
        func: Async function(index, item) to apply.
        max_concurrency: Maximum concurrent tasks.
        fail_fast: If True, cancel remaining tasks on first error.
        on_progress: Optional callback(completed, total).

    Returns:
        BatchResult with ordered results and any errors.
    """
    if max_concurrency < 1:
        max_concurrency = 1

    total = len(items)
    if total == 0:
        return BatchResult(total=0, completed=0)

    semaphore = asyncio.Semaphore(max_concurrency)
    cancel_event = asyncio.Event()
    results_dict: dict[int, Any] = {}
    errors: list[tuple[int, Exception]] = []
    completed_count = 0
    lock = asyncio.Lock()

    async def process_item(index: int, item: Any) -> None:
        nonlocal completed_count

        if cancel_event.is_set():
            return

        async with semaphore:
            if cancel_event.is_set():
                return

            try:
                result = await func(index, item)
                results_dict[index] = result

                async with lock:
                    completed_count += 1
                    current = completed_count

                if on_progress:
                    callback_result = on_progress(current, total)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result

            except Exception as e:
                async with lock:
                    errors.append((index, e))

                if fail_fast:
                    cancel_event.set()
                else:
                    logger.warning(f"Batch item {index} failed")

    tasks = [
        asyncio.create_task(process_item(i, item))
        for i, item in enumerate(items)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    ordered_results = [
        results_dict[i]
        for i in range(total)
        if i in results_dict
    ]

    return BatchResult(
        results=ordered_results,
        results_by_index=results_dict,
        errors=errors,
        total=total,
        completed=completed_count,
        cancelled=cancel_event.is_set(),
    )
