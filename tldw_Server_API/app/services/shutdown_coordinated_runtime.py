"""
Coordinated shutdown runtime helpers extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, Callable


def apply_shutdown_transition_gate(
    app: Any,
    readiness_state: Any | None,
    *,
    get_or_create_lifecycle_state: Callable[[Any], Any],
    mark_lifecycle_shutdown: Callable[[Any, Any | None], None],
    set_job_acquire_gate: Callable[[bool], None],
    logger_obj: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    """Move the app into draining mode and gate new jobs."""
    try:
        lifecycle_state = get_or_create_lifecycle_state(app)
    except startup_guard_exceptions as exc:
        lifecycle_state = None
        logger_obj.debug(f"Shutdown transition gate: lifecycle state lookup skipped: {exc}")

    try:
        if lifecycle_state is None or lifecycle_state.phase != "draining" or not lifecycle_state.draining:
            mark_lifecycle_shutdown(app, readiness_state)
    except startup_guard_exceptions as exc:
        logger_obj.warning(f"Shutdown transition gate: failed to mark lifecycle shutdown: {exc}")

    try:
        set_job_acquire_gate(True)
    except import_exceptions as exc:
        logger_obj.debug(f"Shutdown transition gate: job acquire gate unavailable: {exc}")


def build_coordinated_shutdown_coordinator(
    app: Any,
    legacy_shutdown_plan: list[Any],
    *,
    transport_registry: Any | None = None,
    coordinator_factory: Callable[..., Any],
    register_legacy_shutdown_components: Callable[[Any, list[Any]], list[Any]],
    build_shutdown_components: Callable[[Any | None], list[Any]],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> tuple[Any, list[Any], list[Any]]:
    """Assemble the production drain coordinator with legacy and transport owners."""
    coordinator = coordinator_factory(profile="prod_drain")
    legacy_components: list[Any] = []
    try:
        legacy_components = register_legacy_shutdown_components(
            coordinator,
            legacy_shutdown_plan,
        )
    except (startup_guard_exceptions + import_exceptions):
        legacy_components = []
    transport_components = build_shutdown_components(transport_registry)
    for component in transport_components:
        coordinator.register(component)

    try:
        app.state._tldw_shutdown_transport_component_names = [
            component.name for component in transport_components
        ]
    except startup_guard_exceptions:
        pass

    return coordinator, legacy_components, transport_components


async def run_coordinated_shutdown(
    app: Any,
    legacy_shutdown_plan: list[Any],
    *,
    transport_registry: Any | None = None,
    build_coordinated_shutdown_coordinator: Callable[..., tuple[Any, list[Any], list[Any]]],
    get_legacy_shutdown_suppressed_component_names: Callable[[Any], set[str]],
    logger_obj: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> set[str]:
    """Run the coordinated shutdown slice used by the real lifespan teardown."""
    del import_exceptions

    (
        coordinated_legacy_coordinator,
        coordinated_legacy_components,
        coordinated_transport_components,
    ) = build_coordinated_shutdown_coordinator(
        app,
        legacy_shutdown_plan,
        transport_registry=transport_registry,
    )
    all_components = list(coordinated_legacy_components) + list(coordinated_transport_components)
    if not all_components:
        return set()

    coordinated_legacy_summary = await coordinated_legacy_coordinator.shutdown()
    legacy_component_name_set = {component.name for component in coordinated_legacy_components}
    coordinated_legacy_component_names = {
        name
        for name in get_legacy_shutdown_suppressed_component_names(coordinated_legacy_summary)
        if name in legacy_component_name_set
    }
    phase_groups = {
        phase.value: phase_summary.component_names
        for phase, phase_summary in coordinated_legacy_summary.phases.items()
    }
    try:
        app.state._tldw_shutdown_legacy_coordinator_summary = coordinated_legacy_summary
        app.state._tldw_shutdown_legacy_coordinator_component_names = [
            component.name for component in all_components
        ]
        app.state._tldw_shutdown_legacy_coordinator_phase_groups = phase_groups
    except startup_guard_exceptions:
        pass
    logger_obj.info(
        "App Shutdown: legacy coordinator summary components={} phase_groups={} wall_time_ms={}",
        [component.name for component in all_components],
        phase_groups,
        coordinated_legacy_summary.wall_time_ms,
    )
    return coordinated_legacy_component_names
