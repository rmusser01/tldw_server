"""Lower-level parent controls; no crawl engine or orchestration dependencies."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Generic, Literal, Protocol, TypeVar

Counter = Literal[
    "page_attempts", "network_requests", "browser_attempts", "active_probes", "transfer_bytes", "discovered_candidates"
]
COUNTERS: tuple[Counter, ...] = (
    "page_attempts",
    "network_requests",
    "browser_attempts",
    "active_probes",
    "transfer_bytes",
    "discovered_candidates",
)
T = TypeVar("T")


def positive_integer(value: int, name: str) -> None:
    """Reject non-integral and non-positive trusted safety allowances."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def positive_duration(value: float, name: str) -> None:
    """Reject booleans, non-finite numbers, and disabled time bounds."""
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")


class ParentStop(Exception):
    """Non-retryable parent deadline or resource exhaustion (not cancellation)."""

    def __init__(self, reason: str) -> None:
        if reason not in {"deadline_exhausted", "budget_exhausted"}:
            raise ValueError("invalid parent stop reason")
        self.reason = reason
        super().__init__(reason)


class Reservation:
    """One cumulative reservation, charged at admission and refundable until start.

    The callbacks run under the owner's lock. Commit must check parent stop state
    before marking physical work started; close refunds unused admission once.
    This is not an active-capacity lease: committed work is never refunded.
    """

    def __init__(self, lock: Lock, check: Callable[[], None], rollback: Callable[[], None]) -> None:
        self._lock = lock
        self._check = check
        self._rollback = rollback
        self._state = "reserved"

    def commit(self) -> None:
        """Atomically mark work started unless the parent stopped first."""
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("reservation already closed")
            if self._state == "reserved":
                self._check()
                self._state = "committed"

    def close(self) -> None:
        """Refund an unused reservation once; started work stays charged."""
        with self._lock:
            if self._state == "reserved":
                self._rollback()
                self._state = "closed"


class ParentControls(Protocol):
    """Controls carried downward into governed acquisition and preflight."""

    def check(self) -> None: ...
    def reserve(self, counter: Counter, amount: int = 1) -> Reservation: ...
    def charge_bytes(self, amount: int) -> None: ...
    def cap_timeout(self, requested: float | None) -> float: ...


@dataclass(frozen=True)
class WorkHandle(Generic[T]):
    """Owned logical completion and independent physical drain notification.

    The owner supplies a thread-safe cancel callback. Consumers must never cancel
    the shared futures to claim that the underlying physical work has drained.
    """

    result: asyncio.Future[T]
    drained: asyncio.Future[None]
    cancel: Callable[[], None]


@dataclass(frozen=True)
class BudgetSnapshot:
    """Known admitted usage (including uncommitted reservations), not wire/RSS use."""

    page_attempts: int
    network_requests: int
    browser_attempts: int
    active_probes: int
    transfer_bytes: int
    discovered_candidates: int
    remaining_seconds: float
    accounting_complete: bool = True
