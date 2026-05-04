from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, replace
from typing import Any

from loguru import logger

from tldw_Server_API.app.services.app_lifecycle import mark_lifecycle_shutdown
from tldw_Server_API.app.services.shutdown_models import (
    ShutdownComponent,
    ShutdownPhase,
    ShutdownPolicy,
    ShutdownSummary,
)

StopCallable = Callable[[], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class LegacyShutdownContext:
    readiness_state: MutableMapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _LegacyShutdownSpec:
    name: str
    phase: ShutdownPhase
    policy: ShutdownPolicy
    default_timeout_ms: int
    enabled: Callable[[LegacyShutdownContext], bool]
    stop: Callable[[Any, LegacyShutdownContext], StopCallable]


def _plan_visibility(plan: list[ShutdownComponent]) -> dict[str, Any]:
    phase_groups: dict[str, list[str]] = {}
    for component in plan:
        phase_groups.setdefault(component.phase.value, []).append(component.name)
    return {
        "visible": bool(plan),
        "component_names": [component.name for component in plan],
        "phase_groups": phase_groups,
    }


def _store_legacy_inventory(app: Any, plan: list[ShutdownComponent]) -> None:
    state = getattr(app, "state", None)
    if state is None:
        return
    inventory = _plan_visibility(plan)
    try:
        state._tldw_shutdown_legacy_plan = plan
        state._tldw_shutdown_legacy_inventory = inventory
        state._tldw_shutdown_legacy_phase_groups = inventory["phase_groups"]
        state._tldw_shutdown_legacy_inventory_visible = inventory["visible"]
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.debug(f"Legacy shutdown inventory storage skipped: {exc}")


def register_legacy_shutdown_components(
    coordinator: Any,
    plan: list[ShutdownComponent],
    *,
    component_names: tuple[str, ...] = (),
    phase_overrides: Mapping[str, ShutdownPhase | str] | None = None,
) -> list[ShutdownComponent]:
    """Register the selected legacy shutdown components with a coordinator."""
    selected_components = {component.name: component for component in plan}
    overrides = phase_overrides or {}
    registered_components: list[ShutdownComponent] = []

    for component_name in component_names:
        component = selected_components.get(component_name)
        if component is None:
            continue
        phase = overrides.get(component_name)
        if phase is not None:
            component = replace(component, phase=phase)
        if component.phase == ShutdownPhase.TRANSITION:
            continue
        coordinator.register(component)
        registered_components.append(component)

    return registered_components


_COORDINATED_DIRECT_STOP_RESULTS = {"stopped"}


def get_legacy_shutdown_suppressed_component_names(
    summary: ShutdownSummary | None,
) -> set[str]:
    """Return migrated legacy components that own the direct stop path.

    Only coordinator outcomes that actually completed the component (`stopped`)
    suppress the old direct teardown. Best-effort `skipped` results mean the
    coordinator never invoked the stop callable because budget was exhausted,
    so the legacy direct stop path must remain enabled as fallback. Failed,
    timed out, and cancelled results also leave the legacy direct stop path
    enabled so shutdown can still fall back to the original behavior.
    """
    if summary is None:
        return set()

    return {
        name
        for name, component_summary in summary.components.items()
        if component_summary.result in _COORDINATED_DIRECT_STOP_RESULTS
    }


def _transition_enabled(_: LegacyShutdownContext) -> bool:
    return True


def _transition_stop(app: Any, context: LegacyShutdownContext) -> StopCallable:
    readiness_state = context.readiness_state

    async def _stop() -> None:
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _JobManager

        mark_lifecycle_shutdown(
            app,
            readiness_state if isinstance(readiness_state, MutableMapping) else None,
        )
        _JobManager.set_acquire_gate(True)

    return _stop


_LEGACY_SHUTDOWN_SPECS: tuple[_LegacyShutdownSpec, ...] = (
    _LegacyShutdownSpec(
        name="lifecycle_gate",
        phase=ShutdownPhase.TRANSITION,
        policy=ShutdownPolicy.DEV_FAST,
        default_timeout_ms=1000,
        enabled=_transition_enabled,
        stop=_transition_stop,
    ),
)


def build_legacy_shutdown_plan(
    app: Any,
    context: LegacyShutdownContext | None = None,
) -> list[ShutdownComponent]:
    """Build the legacy shutdown inventory without changing the teardown order yet."""
    shutdown_context = context or LegacyShutdownContext()
    plan: list[ShutdownComponent] = []
    for spec in _LEGACY_SHUTDOWN_SPECS:
        if not spec.enabled(shutdown_context):
            continue
        plan.append(
            ShutdownComponent(
                name=spec.name,
                phase=spec.phase,
                policy=spec.policy,
                default_timeout_ms=spec.default_timeout_ms,
                stop=spec.stop(app, shutdown_context),
            )
        )

    _store_legacy_inventory(app, plan)
    return plan
