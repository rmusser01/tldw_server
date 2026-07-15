"""Request-scoped deadlines, budgets, cleanup, and injected probe dependencies."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol

from loguru import logger

from ..runtime.policy import OutboundPolicyChecker
from ..runtime.requests import RuntimeRequestContext
from .probes import (
    BrowserProbe,
    ExternalToolProbe,
    HttpProbe,
    ProbeBudgetExhausted,
)

if TYPE_CHECKING:
    from ..runtime.policy import ProbeEgressGuard


BudgetKind = Literal["request", "browser", "active_probe"]


@dataclass(frozen=True, slots=True)
class PreflightLimits:
    """Optional request-scoped limits; ``None`` leaves a budget unbounded."""

    requests: int | None = None
    browsers: int | None = None
    active_probes: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("requests", "browsers", "active_probes"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class PreflightConsumed:
    """Immutable snapshot of consumed preflight budgets."""

    requests: int = 0
    browsers: int = 0
    active_probes: int = 0


class _CleanupHandle(Protocol):
    async def close(self) -> None:
        raise NotImplementedError

    async def force_close(self) -> None:
        raise NotImplementedError


class PreflightDeadlineExceeded(Exception):
    """Internal signal for the overall preflight monotonic deadline."""

    def __init__(self) -> None:
        super().__init__("Preflight deadline exceeded.")


class PreflightRuntimeControls:
    """Atomic budgets, deadline helpers, and one bounded cleanup stack."""

    def __init__(
        self,
        request_context: RuntimeRequestContext,
        limits: PreflightLimits | None = None,
        deadline: float | None = None,
        clock: Callable[[], float] = monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if deadline is not None:
            if isinstance(deadline, bool):
                raise ValueError("deadline must be finite")
            deadline = float(deadline)
            if not math.isfinite(deadline):
                raise ValueError("deadline must be finite")
        self.request_context = request_context
        self.limits = limits or PreflightLimits()
        self.deadline = deadline
        self._clock = clock
        self._sleep = sleep
        self._consumed = PreflightConsumed()
        self._budget_lock = asyncio.Lock()
        self._cleanup_handles: list[_CleanupHandle] = []
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def consumed(self) -> PreflightConsumed:
        """Return the current immutable consumed-budget snapshot."""
        return self._consumed

    async def reserve(self, kind: BudgetKind, amount: int = 1) -> None:
        """Atomically reserve capacity from one budget."""
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1:
            raise ValueError("amount must be a positive integer")
        async with self._budget_lock:
            field_name = {
                "request": "requests",
                "browser": "browsers",
                "active_probe": "active_probes",
            }[kind]
            current = getattr(self._consumed, field_name)
            limit = getattr(self.limits, field_name)
            if limit is not None and current + amount > limit:
                raise ProbeBudgetExhausted()
            self._consumed = replace(
                self._consumed,
                **{field_name: current + amount},
            )

    def remaining_seconds(self) -> float | None:
        """Return non-negative time remaining on the overall deadline."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - self._clock())

    def cap_timeout(self, requested_s: float | None) -> float | None:
        """Cap a local timeout against the remaining overall deadline."""
        remaining = self.remaining_seconds()
        values = [value for value in (requested_s, remaining) if value is not None]
        if values and min(values) <= 0:
            raise PreflightDeadlineExceeded()
        return min(values) if values else None

    def deadline_exhausted(self) -> bool:
        """Return whether the overall preflight deadline has expired."""
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= 0

    async def sleep(self, delay_s: float) -> None:
        """Sleep no longer than the overall deadline permits."""
        effective = self.cap_timeout(delay_s)
        if effective is None:
            effective = delay_s
        await self._sleep(effective)
        if effective < delay_s:
            raise PreflightDeadlineExceeded()

    def register_cleanup(self, handle: _CleanupHandle) -> None:
        """Register a resource for reverse-order graceful and forced cleanup."""
        if self._cleanup_task is not None:
            raise RuntimeError("preflight cleanup has already started")
        self._cleanup_handles.append(handle)

    @staticmethod
    def _cleanup_label(handle: _CleanupHandle) -> str:
        raw_label = type(handle).__name__
        label = "".join(character if character.isalnum() or character == "_" else "_" for character in raw_label)[:64]
        return label or "cleanup_handle"

    @classmethod
    def _log_cleanup_failure(cls, handle: _CleanupHandle) -> None:
        logger.warning(f"Preflight cleanup failed for {cls._cleanup_label(handle)}.")

    async def _graceful_cleanup(self) -> None:
        while self._cleanup_handles:
            handle = self._cleanup_handles[-1]
            try:
                await handle.close()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self._log_cleanup_failure(handle)
                try:
                    await handle.force_close()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    self._log_cleanup_failure(handle)
            self._cleanup_handles.pop()

    async def _force_cleanup(self) -> None:
        remaining = tuple(reversed(self._cleanup_handles))
        self._cleanup_handles.clear()
        for handle in remaining:
            try:
                await handle.force_close()
            except asyncio.CancelledError:
                self._log_cleanup_failure(handle)
            except Exception:  # noqa: BLE001
                self._log_cleanup_failure(handle)

    async def _cleanup_with_grace(self, grace_s: float) -> None:
        cleanup_task = asyncio.current_task()
        if cleanup_task is None:  # pragma: no cover - asyncio always supplies one
            raise RuntimeError("cleanup must run in an asyncio task")
        loop = asyncio.get_running_loop()
        grace_expired = False

        def _expire_grace() -> None:
            if not cleanup_task.done():
                cleanup_task.cancel()

        timeout_handle = loop.call_later(grace_s, _expire_grace)
        try:
            await self._graceful_cleanup()
        except asyncio.CancelledError:
            grace_expired = True
        finally:
            timeout_handle.cancel()

        if grace_expired:
            await self._force_cleanup()

    async def close(self, grace_s: float = 2.0) -> None:
        """Close all resources within one grace while preserving caller cancellation."""
        if isinstance(grace_s, bool):
            raise ValueError("grace_s must be a non-negative finite number")
        grace_s = float(grace_s)
        if not math.isfinite(grace_s) or grace_s < 0:
            raise ValueError("grace_s must be a non-negative finite number")

        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_with_grace(grace_s),
                name="preflight-cleanup",
            )

        pending_cancellation: asyncio.CancelledError | None = None
        while not self._cleanup_task.done():
            try:
                await asyncio.shield(self._cleanup_task)
            except asyncio.CancelledError as exc:
                if pending_cancellation is None:
                    pending_cancellation = exc

        self._cleanup_task.result()
        if pending_cancellation is not None:
            raise pending_cancellation


@dataclass(slots=True)
class PreflightExecutionContext:
    """Injected analyzer-facing dependencies for one preflight execution."""

    request_context: RuntimeRequestContext
    policy_checker: OutboundPolicyChecker
    egress_guard: ProbeEgressGuard
    controls: PreflightRuntimeControls
    http: HttpProbe
    browser: BrowserProbe
    external_tools: ExternalToolProbe
    identity_selector: Callable[[], Mapping[str, str]]
    _selected_identity: dict[str, str] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def browser_identity(self) -> dict[str, str]:
        """Return a copy of one lazily selected identity for this context."""
        if self._selected_identity is None:
            self._selected_identity = {str(key): str(value) for key, value in self.identity_selector().items()}
        return dict(self._selected_identity)

    async def close(self) -> None:
        """Delegate request-scoped resource cleanup to runtime controls."""
        await self.controls.close()
