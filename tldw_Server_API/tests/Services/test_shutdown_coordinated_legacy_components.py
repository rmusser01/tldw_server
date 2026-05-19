from __future__ import annotations

import importlib
import sys

import pytest

from tldw_Server_API.app.services.shutdown_models import (
    ShutdownComponent,
    ShutdownPhase,
    ShutdownPolicy,
)


pytestmark = pytest.mark.unit


def _import_shutdown_coordinated_legacy_components():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_coordinated_legacy_components", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_coordinated_legacy_components")


def _component(
    name: str,
    *,
    phase: ShutdownPhase,
    policy: ShutdownPolicy,
) -> ShutdownComponent:
    return ShutdownComponent(
        name=name,
        phase=phase,
        policy=policy,
        default_timeout_ms=1000,
        stop=lambda: None,
    )


@pytest.mark.asyncio
async def test_shutdown_coordinated_legacy_components_returns_handles_and_filters_transition_components() -> None:
    shutdown_legacy = _import_shutdown_coordinated_legacy_components()
    calls: list[tuple[object, list[ShutdownComponent]]] = []
    legacy_shutdown_plan = [
        _component(
            "lifecycle_gate",
            phase=ShutdownPhase.TRANSITION,
            policy=ShutdownPolicy.DEV_FAST,
        ),
        _component(
            "usage_aggregator",
            phase=ShutdownPhase.RESOURCES,
            policy=ShutdownPolicy.BEST_EFFORT,
        ),
        _component(
            "storage_cleanup_service",
            phase=ShutdownPhase.FINALIZERS,
            policy=ShutdownPolicy.PROD_DRAIN,
        ),
    ]

    async def _fake_run_coordinated_shutdown(app_obj, non_transition_plan):
        calls.append((app_obj, non_transition_plan))
        return {"usage_aggregator"}

    handles = await shutdown_legacy.shutdown_coordinated_legacy_components(
        app="app",
        legacy_shutdown_plan=legacy_shutdown_plan,
        run_coordinated_shutdown=_fake_run_coordinated_shutdown,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert len(calls) == 1
    assert calls[0][0] == "app"
    assert [component.name for component in calls[0][1]] == [
        "usage_aggregator",
        "storage_cleanup_service",
    ]
    assert handles.coordinated_legacy_component_names == {"usage_aggregator"}


@pytest.mark.asyncio
async def test_shutdown_coordinated_legacy_components_filters_stopped_background_workers() -> None:
    shutdown_legacy = _import_shutdown_coordinated_legacy_components()
    calls: list[list[ShutdownComponent]] = []
    legacy_shutdown_plan = [
        _component(
            "chatbooks_cleanup",
            phase=ShutdownPhase.WORKERS,
            policy=ShutdownPolicy.PROD_DRAIN,
        ),
        _component(
            "usage_aggregator",
            phase=ShutdownPhase.RESOURCES,
            policy=ShutdownPolicy.BEST_EFFORT,
        ),
    ]

    async def _fake_run_coordinated_shutdown(_app_obj, non_transition_plan):
        calls.append(non_transition_plan)
        return {component.name for component in non_transition_plan}

    handles = await shutdown_legacy.shutdown_coordinated_legacy_components(
        app="app",
        legacy_shutdown_plan=legacy_shutdown_plan,
        run_coordinated_shutdown=_fake_run_coordinated_shutdown,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
        stopped_background_worker_names={"chatbooks_cleanup"},
    )

    assert len(calls) == 1
    assert [component.name for component in calls[0]] == ["usage_aggregator"]
    assert handles.coordinated_legacy_component_names == {"usage_aggregator"}


@pytest.mark.asyncio
async def test_shutdown_coordinated_legacy_components_propagates_guard_exception() -> None:
    shutdown_legacy = _import_shutdown_coordinated_legacy_components()

    async def _failing_run_coordinated_shutdown(_app_obj, _non_transition_plan):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await shutdown_legacy.shutdown_coordinated_legacy_components(
            app="app",
            legacy_shutdown_plan=[],
            run_coordinated_shutdown=_failing_run_coordinated_shutdown,
            startup_guard_exceptions=(RuntimeError,),
            import_exceptions=(ImportError,),
        )


@pytest.mark.asyncio
async def test_shutdown_coordinated_legacy_components_propagates_import_exception() -> None:
    shutdown_legacy = _import_shutdown_coordinated_legacy_components()

    async def _failing_run_coordinated_shutdown(_app_obj, _non_transition_plan):
        raise ImportError("boom")

    with pytest.raises(ImportError, match="boom"):
        await shutdown_legacy.shutdown_coordinated_legacy_components(
            app="app",
            legacy_shutdown_plan=[],
            run_coordinated_shutdown=_failing_run_coordinated_shutdown,
            startup_guard_exceptions=(RuntimeError,),
            import_exceptions=(ImportError,),
        )


@pytest.mark.asyncio
async def test_run_shutdown_coordinated_legacy_components_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_legacy = _import_shutdown_coordinated_legacy_components()
    recorded_calls: list[dict[str, object]] = []
    expected_handles = shutdown_legacy.CoordinatedLegacyShutdownHandles(
        coordinated_legacy_component_names={"usage_aggregator"},
    )

    async def _fake_shutdown_coordinated_legacy_components(**kwargs):
        recorded_calls.append(kwargs)
        return expected_handles

    monkeypatch.setattr(
        shutdown_legacy,
        "shutdown_coordinated_legacy_components",
        _fake_shutdown_coordinated_legacy_components,
    )

    handles = await shutdown_legacy.run_shutdown_coordinated_legacy_components(
        app="app",
        legacy_shutdown_plan=["component"],
        run_coordinated_shutdown=object(),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles is expected_handles
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] == "app"
    assert recorded_calls[0]["legacy_shutdown_plan"] == ["component"]


@pytest.mark.asyncio
async def test_run_shutdown_coordinated_legacy_components_logs_and_returns_empty_handles_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_legacy = _import_shutdown_coordinated_legacy_components()
    debug_messages: list[str] = []

    async def _raise_guard_failure(**_kwargs):
        raise RuntimeError("coordinator unavailable")

    monkeypatch.setattr(
        shutdown_legacy,
        "shutdown_coordinated_legacy_components",
        _raise_guard_failure,
    )
    monkeypatch.setattr(
        shutdown_legacy.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )

    handles = await shutdown_legacy.run_shutdown_coordinated_legacy_components(
        app="app",
        legacy_shutdown_plan=[],
        run_coordinated_shutdown=object(),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles.coordinated_legacy_component_names == set()
    assert any("Legacy coordinator shutdown skipped" in message for message in debug_messages)
