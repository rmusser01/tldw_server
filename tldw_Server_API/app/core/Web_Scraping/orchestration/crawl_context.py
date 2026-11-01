"""Process-local atomic parent accounting and terminal crawl publication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import Lock
from typing import Any

from ..contracts.execution import CrawlOutcome
from ..crawl.models import CrawlLimits
from ..runtime.controls import (
    COUNTERS,
    BudgetSnapshot,
    Counter,
    ParentStop,
    Reservation,
    WorkHandle,
    positive_duration,
    positive_integer,
)


class CrawlContext:
    """One execution owner; cumulative usage, stop, start, and publish share a lock."""

    def __init__(self, limits: CrawlLimits, *, clock: Callable[[], float]) -> None:
        self._limits = limits
        self._clock = clock
        self._deadline = clock() + limits.execution_seconds
        self._lock = Lock()
        self._counts = dict.fromkeys(COUNTERS, 0)
        self._cancelled = False
        self._stop_reason: str | None = None
        self._outcome: CrawlOutcome | None = None
        self._successful_result_present = False
        self._handles: dict[int, WorkHandle[Any]] = {}

    def _check_locked(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError
        if self._outcome is not None:
            raise RuntimeError("crawl already published")
        if self._stop_reason is None and self._clock() >= self._deadline:
            self._stop_reason = "deadline_exhausted"
        if self._stop_reason is not None:
            raise ParentStop(self._stop_reason)

    def check(self) -> None:
        """Raise parent cancellation or latched exhaustion before starting work."""
        with self._lock:
            self._check_locked()

    def reserve(self, counter: Counter, amount: int = 1) -> Reservation:
        """Admit cumulative work atomically; an unused admission can roll back once."""
        positive_integer(amount, "amount")
        if counter not in COUNTERS:
            raise ValueError("unknown cumulative counter")
        with self._lock:
            self._check_locked()
            self._charge_locked(counter, amount)
        return Reservation(self._lock, self._check_locked, lambda: self._refund_locked(counter, amount))

    def _charge_locked(self, counter: Counter, amount: int) -> None:
        if self._counts[counter] + amount > getattr(self._limits, counter):
            self._stop_reason = "budget_exhausted"
            raise ParentStop(self._stop_reason)
        self._counts[counter] += amount

    def _refund_locked(self, counter: Counter, amount: int) -> None:
        self._counts[counter] -= amount

    def charge_bytes(self, amount: int) -> None:
        """Charge observed chunks atomically; no over-allowance chunk is retained."""
        positive_integer(amount, "amount")
        with self._lock:
            self._check_locked()
            self._charge_locked("transfer_bytes", amount)

    def cap_timeout(self, requested: float | None) -> float:
        """Cap per-operation timeouts by the active execution deadline."""
        if requested is not None:
            positive_duration(requested, "requested")
        with self._lock:
            self._check_locked()
            remaining = self._deadline - self._clock()
            if remaining <= 0:
                self._stop_reason = "deadline_exhausted"
                raise ParentStop(self._stop_reason)
            return remaining if requested is None else min(requested, remaining)

    def cancel(self) -> None:
        """Cancel only attached owned handles, never mutate their result/drain futures."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            handles = tuple(self._handles.values())
            self._handles.clear()
        for handle in handles:
            handle.cancel()

    def attach(self, handle: WorkHandle[Any]) -> None:
        """Register owned work, or cancel it if cancellation already won the race."""
        with self._lock:
            cancelled = self._cancelled
            if not cancelled:
                # Prune only physically drained ownership, not logical completion.
                self._handles = {key: item for key, item in self._handles.items() if not item.drained.done()}
                self._handles[id(handle)] = handle
        if cancelled:
            handle.cancel()

    def snapshot(self) -> BudgetSnapshot:
        """Return an immutable known-accounting snapshot without suppressing cleanup."""
        with self._lock:
            return BudgetSnapshot(**self._counts, remaining_seconds=max(0.0, self._deadline - self._clock()))

    def note_successful_result(self) -> None:
        """Mark successful logical commit presence, rejecting stopped/terminal work.

        The engine must commit article/frontier disposition without an intervening
        await. This bit is neither a page counter nor an article payload copy.
        """
        with self._lock:
            self._check_locked()
            self._successful_result_present = True

    def publish(self, outcome: CrawlOutcome) -> CrawlOutcome:
        """Publish once after finalization; earlier cancellation wins atomically."""
        with self._lock:
            if self._outcome is not None:
                return self._outcome
            has_partial_results = self._successful_result_present or outcome.has_partial_results
            if self._cancelled:
                outcome = CrawlOutcome("cancelled", "cancelled", has_partial_results)
            elif self._stop_reason is not None:
                outcome = CrawlOutcome(
                    "partial" if has_partial_results else "failed",
                    self._stop_reason,
                    has_partial_results,
                )
            self._outcome = outcome
            return outcome
