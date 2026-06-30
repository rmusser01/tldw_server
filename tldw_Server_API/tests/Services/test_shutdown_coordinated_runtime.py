from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI


pytestmark = pytest.mark.unit


def test_apply_shutdown_transition_gate_marks_shutdown_and_sets_gate() -> None:
    from tldw_Server_API.app.services.shutdown_coordinated_runtime import (
        apply_shutdown_transition_gate,
    )

    app = FastAPI()
    readiness_state = {"ready": True}
    calls: list[tuple[str, object]] = []

    def _mark_lifecycle_shutdown(app_obj: FastAPI, readiness_obj: object) -> None:
        calls.append(("mark", app_obj))
        assert readiness_obj is readiness_state

    def _set_acquire_gate(enabled: bool) -> None:
        calls.append(("gate", enabled))

    apply_shutdown_transition_gate(
        app,
        readiness_state,
        get_or_create_lifecycle_state=lambda _app: SimpleNamespace(phase="ready", draining=False),
        mark_lifecycle_shutdown=_mark_lifecycle_shutdown,
        set_job_acquire_gate=_set_acquire_gate,
        logger_obj=SimpleNamespace(debug=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError, ModuleNotFoundError),
    )

    assert calls == [("mark", app), ("gate", True)]


def test_build_coordinated_shutdown_coordinator_registers_legacy_and_transport_components() -> None:
    from tldw_Server_API.app.services.shutdown_coordinated_runtime import (
        build_coordinated_shutdown_coordinator,
    )

    app = FastAPI()
    legacy_component = SimpleNamespace(name="legacy:usage")
    transport_component = SimpleNamespace(name="transport:mcp.websocket")

    class _Coordinator:
        def __init__(self, *, profile: str) -> None:
            self.profile = profile
            self.registered: list[object] = []

        def register(self, component: object) -> None:
            self.registered.append(component)

    def _register_legacy_shutdown_components(coordinator: _Coordinator, plan: list[object]) -> list[object]:
        assert plan == ["legacy-plan"]
        coordinator.register(legacy_component)
        return [legacy_component]

    coordinator, legacy_components, transport_components = build_coordinated_shutdown_coordinator(
        app,
        ["legacy-plan"],
        transport_registry="transport-registry",
        coordinator_factory=_Coordinator,
        register_legacy_shutdown_components=_register_legacy_shutdown_components,
        build_shutdown_components=lambda registry: [transport_component] if registry == "transport-registry" else [],
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError, ModuleNotFoundError),
    )

    assert coordinator.profile == "prod_drain"
    assert coordinator.registered == [legacy_component, transport_component]
    assert legacy_components == [legacy_component]
    assert transport_components == [transport_component]
    assert app.state._tldw_shutdown_transport_component_names == ["transport:mcp.websocket"]


@pytest.mark.asyncio
async def test_run_coordinated_shutdown_filters_suppressed_names_to_legacy_components() -> None:
    from tldw_Server_API.app.services.shutdown_coordinated_runtime import (
        run_coordinated_shutdown,
    )

    app = FastAPI()
    legacy_component = SimpleNamespace(name="legacy:usage")
    transport_component = SimpleNamespace(name="transport:mcp.websocket")

    class _Phase:
        value = "resources"

    summary = SimpleNamespace(
        phases={_Phase(): SimpleNamespace(component_names=["legacy:usage", "transport:mcp.websocket"])},
        wall_time_ms=42,
    )

    class _Coordinator:
        async def shutdown(self) -> object:
            return summary

    def _build_coordinator(*_args, **_kwargs):
        return _Coordinator(), [legacy_component], [transport_component]

    suppressed = await run_coordinated_shutdown(
        app,
        ["legacy-plan"],
        transport_registry="transport-registry",
        build_coordinated_shutdown_coordinator=_build_coordinator,
        get_legacy_shutdown_suppressed_component_names=lambda _summary: {
            "legacy:usage",
            "transport:mcp.websocket",
        },
        logger_obj=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError, ModuleNotFoundError),
    )

    assert suppressed == {"legacy:usage"}
    assert app.state._tldw_shutdown_legacy_coordinator_component_names == [
        "legacy:usage",
        "transport:mcp.websocket",
    ]
    assert app.state._tldw_shutdown_legacy_coordinator_phase_groups == {
        "resources": ["legacy:usage", "transport:mcp.websocket"]
    }
