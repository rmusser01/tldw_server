"""
Transition-handoff shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class TransitionHandoffHandles:
    """Updated transition-handoff outputs after shutdown processing."""

    legacy_shutdown_plan: list[Any] = field(default_factory=list)
    transition_gate_applied: bool = False


async def shutdown_transition_handoff(
    *,
    app: Any,
    readiness_state: Any | None,
    build_legacy_shutdown_context: Callable[..., Any],
    apply_shutdown_transition_gate: Callable[[Any, Any | None], None],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> TransitionHandoffHandles:
    """Run the legacy transition-handoff slice and preserve the fallback drain path."""
    transition_gate_applied = False
    legacy_shutdown_plan: list[Any] = []

    try:
        shutdown_context = build_legacy_shutdown_context(
            readiness_state=readiness_state,
        )
        legacy_shutdown_plan = _build_legacy_shutdown_plan(app, shutdown_context)
        legacy_phase_groups = _legacy_phase_groups(legacy_shutdown_plan)
        _store_legacy_shutdown_inventory(
            app,
            legacy_shutdown_plan,
            legacy_phase_groups,
            guard_exceptions=startup_guard_exceptions,
        )
        logger.info(
            "App Shutdown: legacy inventory visible={} phase_groups={}",
            bool(legacy_shutdown_plan),
            legacy_phase_groups,
        )

        transition_coordinator = _create_transition_coordinator(profile="dev_fast")
        for component in legacy_shutdown_plan:
            if _is_transition_component(component):
                transition_coordinator.register(component)
        if legacy_shutdown_plan:
            transition_summary = await transition_coordinator.shutdown()
            transition_gate_summary = transition_summary.components.get("lifecycle_gate")
            transition_gate_applied = bool(
                transition_gate_summary is not None and transition_gate_summary.result == "stopped"
            )
            if transition_gate_applied:
                logger.info("App Shutdown: legacy transition gate handoff executed via coordinator")
            else:
                logger.warning(
                    "App Shutdown: legacy transition gate handoff did not complete cleanly; "
                    "falling back to direct drain",
                )
    except (startup_guard_exceptions + import_exceptions) as exc:
        logger.debug(f"Legacy shutdown inventory skipped: {exc}")
    finally:
        if not transition_gate_applied:
            apply_shutdown_transition_gate(app, readiness_state)

    return TransitionHandoffHandles(
        legacy_shutdown_plan=legacy_shutdown_plan,
        transition_gate_applied=transition_gate_applied,
    )


def _legacy_phase_groups(legacy_shutdown_plan: list[Any]) -> dict[str, list[str]]:
    phase_groups: dict[str, list[str]] = {}
    for component in legacy_shutdown_plan:
        phase_value = getattr(getattr(component, "phase", None), "value", None)
        if phase_value is None:
            phase_value = str(getattr(component, "phase", "unknown"))
        phase_groups.setdefault(str(phase_value), []).append(str(getattr(component, "name", "unknown")))
    return phase_groups


def _store_legacy_shutdown_inventory(
    app: Any,
    legacy_shutdown_plan: list[Any],
    legacy_phase_groups: dict[str, list[str]],
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        app.state._tldw_shutdown_legacy_plan = legacy_shutdown_plan
        app.state._tldw_shutdown_legacy_phase_groups = legacy_phase_groups
        app.state._tldw_shutdown_legacy_inventory_visible = bool(legacy_shutdown_plan)
    except guard_exceptions:
        pass


def _is_transition_component(component: Any) -> bool:
    return getattr(getattr(component, "phase", None), "value", None) == "transition"


def _build_legacy_shutdown_plan(app: Any, shutdown_context: Any) -> list[Any]:
    from tldw_Server_API.app.services.shutdown_legacy_adapters import (
        build_legacy_shutdown_plan,
    )

    return build_legacy_shutdown_plan(app, shutdown_context)


def _create_transition_coordinator(*, profile: str) -> Any:
    from tldw_Server_API.app.services.shutdown_coordinator import ShutdownCoordinator

    return ShutdownCoordinator(profile=profile)
