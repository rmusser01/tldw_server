from __future__ import annotations

import asyncio
import contextlib
from copy import deepcopy
import inspect
from collections import OrderedDict
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from tldw_Server_API.app.services.shutdown_models import (
    ShutdownComponent,
    ShutdownComponentSummary,
    ShutdownPhase,
    ShutdownPhaseSummary,
    ShutdownPolicy,
    ShutdownResult,
    ShutdownSummary,
)

ORDERED_PHASES: tuple[ShutdownPhase, ...] = (
    ShutdownPhase.TRANSITION,
    ShutdownPhase.ACCEPTORS,
    ShutdownPhase.PRODUCERS,
    ShutdownPhase.WORKERS,
    ShutdownPhase.RESOURCES,
    ShutdownPhase.FINALIZERS,
)


@dataclass(slots=True)
class _ShutdownProfile:
    deadline_ms: int
    soft_overrun_ms: int


_PROFILE_DEFAULTS: dict[str, _ShutdownProfile] = {
    "dev_fast": _ShutdownProfile(deadline_ms=5000, soft_overrun_ms=0),
    "prod_drain": _ShutdownProfile(deadline_ms=15000, soft_overrun_ms=2000),
}


class ShutdownCoordinator:
    def __init__(
        self,
        profile: str = "dev_fast",
        *,
        clock: Callable[[], float] | None = None,
        deadline_ms: int | None = None,
        soft_overrun_ms: int | None = None,
    ) -> None:
        if profile not in _PROFILE_DEFAULTS and profile != "custom":
            raise ValueError(f"Unknown shutdown profile: {profile!r}")

        config = _PROFILE_DEFAULTS.get(profile, _PROFILE_DEFAULTS["dev_fast"])
        self.profile = profile
        self._clock = clock or time.monotonic
        self._deadline_ms = config.deadline_ms if deadline_ms is None else max(0, int(deadline_ms))
        self._soft_overrun_ms = (
            config.soft_overrun_ms if soft_overrun_ms is None else max(0, int(soft_overrun_ms))
        )
        self._components: OrderedDict[str, ShutdownComponent] = OrderedDict()
        self._summary: ShutdownSummary | None = None
        self._shutdown_task: asyncio.Task[ShutdownSummary] | None = None
        self._state_lock = threading.Lock()
        self._registration_closed = False
        self._draining = False

    def register(self, component: ShutdownComponent) -> ShutdownComponent:
        with self._state_lock:
            if self._registration_closed or self._summary is not None or self._shutdown_task is not None:
                raise RuntimeError("Shutdown registration is closed")
            if component.name in self._components:
                raise ValueError(f"Shutdown component already registered: {component.name!r}")
            self._components[component.name] = component
        return component

    async def shutdown(self) -> ShutdownSummary:
        with self._state_lock:
            if self._summary is not None:
                return self._clone_summary(self._summary, idempotent=True)
            task = self._shutdown_task
            created_task = task is None
            if task is None:
                self._registration_closed = True
                task = asyncio.create_task(self._shutdown_impl())
                task.add_done_callback(self._clear_shutdown_task)
                self._shutdown_task = task

        summary = await asyncio.shield(task)
        if created_task:
            return self._clone_summary(summary, idempotent=False)
        return self._clone_summary(summary, idempotent=True)

    def _clear_shutdown_task(self, task: asyncio.Task[ShutdownSummary]) -> None:
        with self._state_lock:
            if self._shutdown_task is task:
                self._shutdown_task = None

    @staticmethod
    def _clone_summary(summary: ShutdownSummary, *, idempotent: bool) -> ShutdownSummary:
        return replace(
            summary,
            idempotent=idempotent,
            components=deepcopy(summary.components),
            phases=deepcopy(summary.phases),
        )

    async def _shutdown_impl(self) -> ShutdownSummary:
        if self._summary is not None:
            return self._summary

        started_at = float(self._clock())
        deadline_at = started_at + (self._deadline_ms / 1000.0)
        hard_cutoff_at = deadline_at + (self._soft_overrun_ms / 1000.0)
        summary = ShutdownSummary(
            profile=self.profile,
            started_at=started_at,
            finished_at=started_at,
            deadline_at=deadline_at,
            hard_cutoff_at=hard_cutoff_at,
            wall_time_ms=0,
            soft_overrun_used_ms=0,
        )
        components_snapshot = OrderedDict(self._components)

        self._draining = True
        try:
            await self._run_phase(ShutdownPhase.TRANSITION, summary, components_snapshot)

            for phase in ORDERED_PHASES[1:]:
                await self._run_phase(phase, summary, components_snapshot)

            finished_at = float(self._clock())
            summary.finished_at = finished_at
            summary.wall_time_ms = max(0, int((finished_at - started_at) * 1000))
            summary.soft_overrun_used_ms = max(
                0,
                min(
                    self._soft_overrun_ms,
                    int(max(0.0, (finished_at - deadline_at) * 1000)),
                ),
            )

            self._summary = summary
            return summary
        finally:
            self._draining = False

    async def _run_phase(
        self,
        phase: ShutdownPhase,
        summary: ShutdownSummary,
        components_snapshot: OrderedDict[str, ShutdownComponent],
    ) -> None:
        components = [component for component in components_snapshot.values() if component.phase == phase]
        phase_started = float(self._clock())
        phase_budget_ms = self._phase_budget_ms(
            summary,
            current_phase=phase,
            components_snapshot=components_snapshot,
        )
        phase_summary = ShutdownPhaseSummary(
            phase=phase,
            started_at=phase_started,
            finished_at=phase_started,
            duration_ms=0,
            budget_ms=phase_budget_ms,
            component_names=[component.name for component in components],
        )
        summary.phases[phase] = phase_summary

        if not components:
            return

        outcomes = await asyncio.gather(
            *(
                self._run_component(component, phase_budget_ms=phase_budget_ms, summary=summary)
                for component in components
            ),
            return_exceptions=True,
        )

        for component, outcome in zip(components, outcomes):
            component_summary = self._coerce_component_outcome(
                component,
                outcome,
                phase_started_at=phase_started,
            )
            summary.components[component_summary.name] = component_summary

        phase_finished = float(self._clock())
        phase_summary.finished_at = phase_finished
        phase_summary.duration_ms = max(0, int((phase_finished - phase_started) * 1000))

    def _coerce_component_outcome(
        self,
        component: ShutdownComponent,
        outcome: object,
        *,
        phase_started_at: float,
    ) -> ShutdownComponentSummary:
        if isinstance(outcome, ShutdownComponentSummary):
            return outcome

        finished_at = float(self._clock())
        duration_ms = max(0, int((finished_at - phase_started_at) * 1000))
        if isinstance(outcome, asyncio.CancelledError):
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result="cancelled",
                started_at=phase_started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                timeout_ms=component.default_timeout_ms,
                error=str(outcome) or outcome.__class__.__name__,
            )
        if isinstance(outcome, Exception):
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result="failed",
                started_at=phase_started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                timeout_ms=component.default_timeout_ms,
                error=str(outcome) or outcome.__class__.__name__,
            )
        raise TypeError(f"Unexpected shutdown component outcome for {component.name!r}: {outcome!r}")

    def _phase_budget_ms(
        self,
        summary: ShutdownSummary,
        *,
        current_phase: ShutdownPhase,
        components_snapshot: OrderedDict[str, ShutdownComponent],
    ) -> int:
        now = float(self._clock())
        try:
            phase_index = ORDERED_PHASES.index(current_phase)
        except ValueError:
            return 0

        runnable_phases = self._remaining_runnable_phase_count(
            current_phase=current_phase,
            components_snapshot=components_snapshot,
        )
        if runnable_phases <= 0:
            return 0

        window_end = summary.deadline_at
        if (
            self._phase_allows_soft_overrun(current_phase)
            and self._soft_overrun_ms > 0
            and any(
                self._component_allows_soft_overrun(component)
                for component in components_snapshot.values()
                if component.phase == current_phase
            )
        ):
            window_end = summary.hard_cutoff_at
        remaining_ms = max(0, int((window_end - now) * 1000))
        if remaining_ms <= 0:
            return 0
        return max(1, remaining_ms // runnable_phases)

    def _remaining_runnable_phase_count(
        self,
        *,
        current_phase: ShutdownPhase,
        components_snapshot: OrderedDict[str, ShutdownComponent],
    ) -> int:
        current_index = ORDERED_PHASES.index(current_phase)
        runnable_phases = 0
        for phase in ORDERED_PHASES[current_index:]:
            phase_components = [component for component in components_snapshot.values() if component.phase == phase]
            if not phase_components:
                continue
            if phase == current_phase or any(
                component.policy != ShutdownPolicy.BEST_EFFORT for component in phase_components
            ):
                runnable_phases += 1
        return runnable_phases

    def _phase_allows_soft_overrun(self, phase: ShutdownPhase) -> bool:
        return phase in {
            ShutdownPhase.WORKERS,
            ShutdownPhase.RESOURCES,
            ShutdownPhase.FINALIZERS,
        }

    def _component_allows_soft_overrun(self, component: ShutdownComponent) -> bool:
        return (
            self.profile in {"prod_drain", "custom"}
            and component.policy == ShutdownPolicy.PROD_DRAIN
            and self._phase_allows_soft_overrun(component.phase)
        )

    async def _run_component(
        self,
        component: ShutdownComponent,
        *,
        phase_budget_ms: int,
        summary: ShutdownSummary,
    ) -> ShutdownComponentSummary:
        started_at = float(self._clock())
        is_best_effort = component.policy == ShutdownPolicy.BEST_EFFORT

        timeout_ms = max(
            0,
            min(
                component.default_timeout_ms,
                phase_budget_ms,
                self._component_budget_ms(component, summary, started_at),
            ),
        )

        if is_best_effort and timeout_ms <= 0:
            finished_at = float(self._clock())
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result="skipped",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=max(0, int((finished_at - started_at) * 1000)),
                timeout_ms=0,
                budget_exhausted=True,
            )

        if started_at >= summary.hard_cutoff_at:
            result: ShutdownResult = "cancelled"
            finished_at = started_at
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result=result,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=0,
                timeout_ms=timeout_ms,
                budget_exhausted=True,
            )

        if timeout_ms <= 0:
            result = "cancelled"
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result=result,
                started_at=started_at,
                finished_at=started_at,
                duration_ms=0,
                timeout_ms=timeout_ms,
                budget_exhausted=True,
            )

        stop_task = asyncio.create_task(self._invoke_stop(component, timeout_ms=timeout_ms))
        stop_task.add_done_callback(self._consume_stop_task_result)

        try:
            # Keep the stop task shielded so wait_for() can time out promptly even if
            # the underlying stop path swallows CancelledError during hard cutoff.
            await asyncio.wait_for(asyncio.shield(stop_task), timeout=timeout_ms / 1000.0)
        except TimeoutError:
            stop_task.cancel()
            await self._wait_for_task_quiescence(stop_task, deadline_at=summary.hard_cutoff_at)
            finished_at = float(self._clock())
            result = "timed_out" if finished_at < summary.hard_cutoff_at else "cancelled"
            error: str | None = None
            if stop_task.done():
                if stop_task.cancelled():
                    result = "cancelled"
                else:
                    try:
                        stop_task.result()
                    except Exception as exc:
                        result = "failed"
                        error = str(exc) or exc.__class__.__name__
                    else:
                        result = "stopped"
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result=result,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=max(0, int((finished_at - started_at) * 1000)),
                timeout_ms=timeout_ms,
                error=error,
                budget_exhausted=True,
            )
        except asyncio.CancelledError:
            stop_task.cancel()
            finished_at = float(self._clock())
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result="cancelled",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=max(0, int((finished_at - started_at) * 1000)),
                timeout_ms=timeout_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            stop_task.cancel()
            finished_at = float(self._clock())
            return ShutdownComponentSummary(
                name=component.name,
                phase=component.phase,
                policy=component.policy,
                result="failed",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=max(0, int((finished_at - started_at) * 1000)),
                timeout_ms=timeout_ms,
                error=str(exc),
            )

        finished_at = float(self._clock())
        return ShutdownComponentSummary(
            name=component.name,
            phase=component.phase,
            policy=component.policy,
            result="stopped",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=max(0, int((finished_at - started_at) * 1000)),
            timeout_ms=timeout_ms,
        )

    def _component_budget_ms(
        self,
        component: ShutdownComponent,
        summary: ShutdownSummary,
        started_at: float,
    ) -> int:
        window_end = summary.hard_cutoff_at if self._component_allows_soft_overrun(component) else summary.deadline_at
        return max(0, int((window_end - started_at) * 1000))

    @staticmethod
    async def _invoke_stop(component: ShutdownComponent, *, timeout_ms: int) -> None:
        stop_callable = component.stop
        accepts_timeout = ShutdownCoordinator._stop_accepts_timeout(stop_callable)
        if inspect.iscoroutinefunction(stop_callable) or inspect.iscoroutinefunction(
            getattr(stop_callable, "__call__", None)
        ):
            outcome = stop_callable(timeout_ms) if accepts_timeout else stop_callable()
        else:
            if accepts_timeout:
                outcome = await asyncio.to_thread(stop_callable, timeout_ms)
            else:
                outcome = await asyncio.to_thread(stop_callable)
        if inspect.isawaitable(outcome):
            await outcome

    @staticmethod
    def _stop_accepts_timeout(stop_callable: Callable[..., object]) -> bool:
        try:
            signature = inspect.signature(stop_callable)
        except (TypeError, ValueError):
            return False
        return bool(signature.parameters)

    async def _wait_for_task_quiescence(self, task: asyncio.Task[None], *, deadline_at: float) -> None:
        remaining_s = max(0.0, deadline_at - float(self._clock()))
        if remaining_s <= 0:
            return
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait({task}, timeout=remaining_s)

    @staticmethod
    def _consume_stop_task_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            return
