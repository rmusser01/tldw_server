"""
Coordinated legacy shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger


@dataclass
class CoordinatedLegacyShutdownHandles:
    """Updated coordinated-legacy shutdown outputs after processing."""

    coordinated_legacy_component_names: set[str] = field(default_factory=set)


async def shutdown_coordinated_legacy_components(
    *,
    app: Any,
    legacy_shutdown_plan: list[Any],
    run_coordinated_shutdown: Callable[[Any, list[Any]], Awaitable[set[str]]],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> CoordinatedLegacyShutdownHandles:
    """Run the non-transition legacy components through the coordinated shutdown path."""
    coordinated_legacy_component_names = await _shutdown_coordinated_legacy_components(
        app=app,
        legacy_shutdown_plan=legacy_shutdown_plan,
        run_coordinated_shutdown=run_coordinated_shutdown,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
        stopped_background_worker_names=stopped_background_worker_names,
    )
    return CoordinatedLegacyShutdownHandles(
        coordinated_legacy_component_names=coordinated_legacy_component_names,
    )


async def run_shutdown_coordinated_legacy_components(
    *,
    app: Any,
    legacy_shutdown_plan: list[Any],
    run_coordinated_shutdown: Callable[[Any, list[Any]], Awaitable[set[str]]],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> CoordinatedLegacyShutdownHandles:
    """Run coordinated legacy shutdown with main-lifespan fallback behavior."""
    try:
        return await shutdown_coordinated_legacy_components(
            app=app,
            legacy_shutdown_plan=legacy_shutdown_plan,
            run_coordinated_shutdown=run_coordinated_shutdown,
            startup_guard_exceptions=startup_guard_exceptions,
            import_exceptions=import_exceptions,
            stopped_background_worker_names=stopped_background_worker_names,
        )
    except (startup_guard_exceptions + import_exceptions) as exc:
        logger.debug(f"Legacy coordinator shutdown skipped: {exc}")
        return CoordinatedLegacyShutdownHandles()


async def _shutdown_coordinated_legacy_components(
    *,
    app: Any,
    legacy_shutdown_plan: list[Any],
    run_coordinated_shutdown: Callable[[Any, list[Any]], Awaitable[set[str]]],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
    stopped_background_worker_names: set[str] | None = None,
) -> set[str]:
    del startup_guard_exceptions, import_exceptions
    stopped_background_worker_names = stopped_background_worker_names or set()

    non_transition_legacy_shutdown_plan = [
        component
        for component in legacy_shutdown_plan
        if getattr(getattr(component, "phase", None), "value", None) != "transition"
        and getattr(component, "name", None) not in stopped_background_worker_names
    ]
    return await run_coordinated_shutdown(
        app,
        non_transition_legacy_shutdown_plan,
    )
