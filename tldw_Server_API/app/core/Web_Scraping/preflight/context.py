"""Request-scoped deadlines, budgets, cleanup, and injected probe dependencies."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
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
_CLEANUP_FORCE_RESERVE_FRACTION = 0.5


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


@dataclass(slots=True)
class _CleanupEntry:
    handle: _CleanupHandle
    graceful_complete: bool = False
    force_started: bool = False


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
        self._cleanup_grace_remaining_s: float | None = None
        self._cleanup_episode_lock = asyncio.Lock()

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
        if delay_s == 0:
            await self._sleep(0)
            return
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

    async def _graceful_cleanup(self, entries: tuple[_CleanupEntry, ...]) -> None:
        for entry in entries:
            if entry.force_started:
                continue
            try:
                await entry.handle.close()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self._log_cleanup_failure(entry.handle)
            else:
                entry.graceful_complete = True

    async def _force_cleanup_entry(self, entry: _CleanupEntry) -> None:
        try:
            await entry.handle.force_close()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self._log_cleanup_failure(entry.handle)

    def _start_force_cleanup(
        self,
        entries: tuple[_CleanupEntry, ...],
    ) -> set[asyncio.Task[None]]:
        tasks: set[asyncio.Task[None]] = set()
        for index, entry in enumerate(entries):
            if entry.graceful_complete or entry.force_started:
                continue
            entry.force_started = True
            tasks.add(
                asyncio.create_task(
                    self._force_cleanup_entry(entry),
                    name=f"preflight-cleanup-force-{index}",
                )
            )
        return tasks

    @staticmethod
    def _consume_cleanup_task(task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    @classmethod
    async def _consume_cleanup_tasks_by_deadline(
        cls,
        tasks: set[asyncio.Task[None]],
        *,
        deadline: float,
    ) -> None:
        if not tasks:
            return

        loop = asyncio.get_running_loop()
        remaining_s = max(0.0, deadline - loop.time())
        if remaining_s > 0:
            await asyncio.wait(tasks, timeout=remaining_s)
        else:
            # Give newly-created force tasks and released graceful work a turn.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        pending = {task for task in tasks if not task.done()}
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        for task in tasks:
            if task.done():
                cls._consume_cleanup_task(task)
            else:
                task.add_done_callback(cls._consume_cleanup_task)

    async def _cleanup_with_grace(
        self,
        handles: tuple[_CleanupHandle, ...],
        grace_s: float,
    ) -> None:
        entries = tuple(_CleanupEntry(handle=handle) for handle in handles)
        loop = asyncio.get_running_loop()
        grace_deadline = loop.time() + grace_s
        graceful_deadline = grace_deadline - (grace_s * _CLEANUP_FORCE_RESERVE_FRACTION)
        graceful_task = asyncio.create_task(
            self._graceful_cleanup(entries),
            name="preflight-cleanup-graceful",
        )
        deadline_task = asyncio.create_task(
            asyncio.sleep(max(0.0, graceful_deadline - loop.time())),
            name="preflight-cleanup-deadline",
        )

        done, _ = await asyncio.wait(
            {graceful_task, deadline_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if graceful_task in done:
            self._consume_cleanup_task(graceful_task)
            deadline_task.cancel()
        else:
            graceful_task.cancel()

        force_tasks = self._start_force_cleanup(entries)
        cleanup_tasks = force_tasks | {graceful_task, deadline_task}
        await self._consume_cleanup_tasks_by_deadline(
            cleanup_tasks,
            deadline=grace_deadline,
        )

    @staticmethod
    def _validate_cleanup_grace(grace_s: float) -> float:
        if isinstance(grace_s, bool):
            raise ValueError("grace_s must be a non-negative finite number")
        normalized = float(grace_s)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError("grace_s must be a non-negative finite number")
        return normalized

    async def _run_cleanup_episode(
        self,
        handles: tuple[_CleanupHandle, ...],
        grace_s: float,
    ) -> None:
        async with self._cleanup_episode_lock:
            if self._cleanup_grace_remaining_s is None:
                self._cleanup_grace_remaining_s = grace_s
            else:
                self._cleanup_grace_remaining_s = min(
                    self._cleanup_grace_remaining_s,
                    grace_s,
                )
            episode_budget_s = self._cleanup_grace_remaining_s
            loop = asyncio.get_running_loop()
            started_at = loop.time()
            try:
                await self._cleanup_with_grace(handles, episode_budget_s)
            finally:
                elapsed_s = max(0.0, loop.time() - started_at)
                self._cleanup_grace_remaining_s = max(
                    0.0,
                    self._cleanup_grace_remaining_s - elapsed_s,
                )

    @staticmethod
    async def _await_cleanup_supervisor(task: asyncio.Task[None]) -> None:
        pending_cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if pending_cancellation is None:
                    pending_cancellation = exc

        try:
            task.result()
        except BaseException:
            if pending_cancellation is not None:
                raise pending_cancellation from None
            raise
        if pending_cancellation is not None:
            raise pending_cancellation

    async def cleanup_handles(
        self,
        handles: Iterable[_CleanupHandle],
        grace_s: float = 2.0,
    ) -> None:
        """Clean one owned handle graph without closing unrelated resources."""
        normalized_grace = self._validate_cleanup_grace(grace_s)
        owned_handles = tuple(handles)
        if not owned_handles:
            return
        supervisor = asyncio.create_task(
            self._run_cleanup_episode(
                tuple(reversed(owned_handles)),
                normalized_grace,
            ),
            name="preflight-cleanup-subset-supervisor",
        )
        await self._await_cleanup_supervisor(supervisor)

    async def close(self, grace_s: float = 2.0) -> None:
        """Close all resources within one grace while preserving caller cancellation."""
        normalized_grace = self._validate_cleanup_grace(grace_s)

        if self._cleanup_task is None:
            handles = tuple(reversed(self._cleanup_handles))
            self._cleanup_handles.clear()
            self._cleanup_task = asyncio.create_task(
                self._run_cleanup_episode(handles, normalized_grace),
                name="preflight-cleanup-supervisor",
            )
        await self._await_cleanup_supervisor(self._cleanup_task)


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
