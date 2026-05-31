from __future__ import annotations

import asyncio
import importlib
import sys
import types
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable

import pytest

from tldw_Server_API.app.services.shutdown_coordinator import ShutdownCoordinator
from tldw_Server_API.app.services.shutdown_models import (
    ShutdownComponent,
    ShutdownPhase,
    ShutdownPolicy,
)

pytestmark = pytest.mark.unit


@dataclass
class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return float(self.now)

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def component(
    name: str,
    *,
    phase: ShutdownPhase | str,
    stop: Callable[[], Awaitable[None] | None],
    policy: ShutdownPolicy | str = ShutdownPolicy.PROD_DRAIN,
    default_timeout_ms: int = 250,
) -> ShutdownComponent:
    return ShutdownComponent(
        name=name,
        phase=phase,
        policy=policy,
        default_timeout_ms=default_timeout_ms,
        stop=stop,
    )


def test_build_legacy_shutdown_plan_includes_known_legacy_components() -> None:
    from tldw_Server_API.app.services.shutdown_legacy_adapters import (
        LegacyShutdownContext,
        build_legacy_shutdown_plan,
    )

    class _App:
        pass

    app = _App()
    plan = build_legacy_shutdown_plan(
        app,
        LegacyShutdownContext(),
    )

    assert [component.name for component in plan] == ["lifecycle_gate"]
    assert plan[0].phase == ShutdownPhase.TRANSITION
    assert plan[0].policy == ShutdownPolicy.DEV_FAST


def test_build_legacy_shutdown_plan_records_visibility_on_app_state() -> None:
    from tldw_Server_API.app.services.shutdown_legacy_adapters import (
        LegacyShutdownContext,
        build_legacy_shutdown_plan,
    )

    class _State:
        pass

    class _App:
        def __init__(self) -> None:
            self.state = _State()

    app = _App()

    plan = build_legacy_shutdown_plan(app, LegacyShutdownContext())

    assert app.state._tldw_shutdown_legacy_inventory_visible is True
    assert app.state._tldw_shutdown_legacy_phase_groups == {
        "transition": ["lifecycle_gate"],
    }
    assert app.state._tldw_shutdown_legacy_inventory["component_names"] == ["lifecycle_gate"]
    assert plan[0].name == "lifecycle_gate"


def test_build_legacy_shutdown_plan_omits_removed_usage_stop_components() -> None:
    from tldw_Server_API.app.services.shutdown_legacy_adapters import (
        LegacyShutdownContext,
        build_legacy_shutdown_plan,
    )

    class _App:
        pass

    plan = build_legacy_shutdown_plan(
        _App(),
        LegacyShutdownContext(),
    )

    assert [component.name for component in plan] == ["lifecycle_gate"]
    assert [component.phase for component in plan] == [ShutdownPhase.TRANSITION]


def test_shutdown_legacy_adapters_import_is_lazy_for_optional_stop_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tldw_Server_API.app.services.shutdown_legacy_adapters"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.AuthNZ.scheduler",
        types.ModuleType("tldw_Server_API.app.core.AuthNZ.scheduler"),
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Jobs.manager",
        types.ModuleType("tldw_Server_API.app.core.Jobs.manager"),
    )

    module = importlib.import_module(module_name)
    LegacyShutdownContext = module.LegacyShutdownContext

    class _App:
        pass

    plan = module.build_legacy_shutdown_plan(
        _App(),
        LegacyShutdownContext(),
    )

    assert [component.name for component in plan] == ["lifecycle_gate"]


def test_get_legacy_shutdown_suppressed_component_names_uses_summary_results() -> None:
    from tldw_Server_API.app.services.shutdown_legacy_adapters import (
        get_legacy_shutdown_suppressed_component_names,
    )
    from tldw_Server_API.app.services.shutdown_models import (
        ShutdownComponentSummary,
        ShutdownPhase,
        ShutdownPolicy,
        ShutdownSummary,
    )

    summary = ShutdownSummary(
        profile="prod_drain",
        started_at=0.0,
        finished_at=0.0,
        deadline_at=0.0,
        hard_cutoff_at=0.0,
        wall_time_ms=0,
        soft_overrun_used_ms=0,
        components={
            "chatbooks_cleanup": ShutdownComponentSummary(
                name="chatbooks_cleanup",
                phase=ShutdownPhase.WORKERS,
                policy=ShutdownPolicy.PROD_DRAIN,
                result="stopped",
                started_at=0.0,
                finished_at=0.0,
                duration_ms=0,
                timeout_ms=1000,
            ),
            "usage_aggregator": ShutdownComponentSummary(
                name="usage_aggregator",
                phase=ShutdownPhase.RESOURCES,
                policy=ShutdownPolicy.BEST_EFFORT,
                result="skipped",
                started_at=0.0,
                finished_at=0.0,
                duration_ms=0,
                timeout_ms=0,
            ),
            "llm_usage_aggregator": ShutdownComponentSummary(
                name="llm_usage_aggregator",
                phase=ShutdownPhase.RESOURCES,
                policy=ShutdownPolicy.BEST_EFFORT,
                result="timed_out",
                started_at=0.0,
                finished_at=0.0,
                duration_ms=0,
                timeout_ms=0,
            ),
            "storage_cleanup_service": ShutdownComponentSummary(
                name="storage_cleanup_service",
                phase=ShutdownPhase.FINALIZERS,
                policy=ShutdownPolicy.PROD_DRAIN,
                result="cancelled",
                started_at=0.0,
                finished_at=0.0,
                duration_ms=0,
                timeout_ms=0,
            ),
        },
    )

    assert get_legacy_shutdown_suppressed_component_names(summary) == {"chatbooks_cleanup"}


@pytest.mark.asyncio
async def test_register_legacy_shutdown_components_default_omits_removed_usage_components() -> None:
    from tldw_Server_API.app.services.shutdown_legacy_adapters import (
        register_legacy_shutdown_components,
    )

    events: list[str] = []
    plan = [
        component(
            "usage_aggregator",
            phase=ShutdownPhase.RESOURCES,
            stop=lambda: events.append("usage"),
        ),
        component(
            "llm_usage_aggregator",
            phase=ShutdownPhase.RESOURCES,
            stop=lambda: events.append("llm"),
        ),
        component(
            "authnz_scheduler",
            phase=ShutdownPhase.FINALIZERS,
            stop=lambda: events.append("authnz"),
        ),
    ]

    coordinator = ShutdownCoordinator(profile="dev_fast")
    registered = register_legacy_shutdown_components(
        coordinator,
        plan,
    )

    summary = await coordinator.shutdown()

    assert registered == []
    assert plan[-1].phase == ShutdownPhase.FINALIZERS
    assert events == []
    assert summary.phases[ShutdownPhase.PRODUCERS].component_names == []
    assert summary.phases[ShutdownPhase.WORKERS].component_names == []
    assert summary.phases[ShutdownPhase.RESOURCES].component_names == []
    assert summary.phases[ShutdownPhase.FINALIZERS].component_names == []


def test_register_legacy_shutdown_components_skips_effective_transition_overrides() -> None:
    from tldw_Server_API.app.services.shutdown_legacy_adapters import (
        register_legacy_shutdown_components,
    )

    class _Coordinator:
        def __init__(self) -> None:
            self.registered: list[ShutdownComponent] = []

        def register(self, component: ShutdownComponent) -> ShutdownComponent:
            self.registered.append(component)
            return component

    coordinator = _Coordinator()
    registered = register_legacy_shutdown_components(
        coordinator,
        [
            component("chatbooks_cleanup", phase=ShutdownPhase.WORKERS, stop=lambda: None),
        ],
        component_names=("chatbooks_cleanup",),
        phase_overrides={"chatbooks_cleanup": ShutdownPhase.TRANSITION},
    )

    assert registered == []
    assert coordinator.registered == []


@pytest.mark.asyncio
async def test_coordinator_runs_same_phase_components_in_parallel() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    started = asyncio.Event()
    release = asyncio.Event()
    running = 0

    async def _stop_a() -> None:
        nonlocal running
        running += 1
        if running == 2:
            started.set()
        await release.wait()

    async def _stop_b() -> None:
        nonlocal running
        running += 1
        if running == 2:
            started.set()
        await release.wait()

    coordinator.register(component("worker-a", phase=ShutdownPhase.WORKERS, stop=_stop_a))
    coordinator.register(component("worker-b", phase=ShutdownPhase.WORKERS, stop=_stop_b))

    shutdown_task = asyncio.create_task(coordinator.shutdown())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert running == 2
    release.set()

    summary = await shutdown_task

    assert summary.phases[ShutdownPhase.WORKERS].component_names == ["worker-a", "worker-b"]
    assert summary.components["worker-a"].result == "stopped"
    assert summary.components["worker-b"].result == "stopped"


@pytest.mark.asyncio
async def test_coordinator_offloads_sync_stop_callables_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    to_thread_calls: list[Callable[[], Awaitable[None] | None]] = []
    events: list[str] = []

    async def _fake_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr("tldw_Server_API.app.services.shutdown_coordinator.asyncio.to_thread", _fake_to_thread)

    coordinator = ShutdownCoordinator(profile="dev_fast")

    def _sync_stop() -> None:
        events.append("sync")

    coordinator.register(component("producer-a", phase="producers", stop=_sync_stop))

    summary = await coordinator.shutdown()

    assert to_thread_calls == [_sync_stop]
    assert summary.components["producer-a"].result == "stopped"
    assert events == ["sync"]


@pytest.mark.asyncio
async def test_transport_registry_components_are_visible_to_shutdown_coordinator() -> None:
    from tldw_Server_API.app.services.shutdown_transport_registry import (
        ShutdownTransportRegistry,
        build_shutdown_components,
    )

    events: list[str] = []
    async def _drain_mcp(timeout_s: float | None = None) -> None:
        del timeout_s
        events.append("mcp")

    async def _drain_prompt_studio(timeout_s: float | None = None) -> None:
        del timeout_s
        events.append("prompt_studio")

    registry = ShutdownTransportRegistry()
    registry.register_family(
        "mcp.websocket",
        active_count=lambda: 2,
        drain=_drain_mcp,
    )
    registry.register_family(
        "prompt_studio.websocket",
        active_count=lambda: 1,
        drain=_drain_prompt_studio,
    )

    coordinator = ShutdownCoordinator(profile="dev_fast")
    for shutdown_component in build_shutdown_components(registry):
        coordinator.register(shutdown_component)

    summary = await coordinator.shutdown()

    assert summary.phases[ShutdownPhase.ACCEPTORS].component_names == [
        "transport:mcp.websocket",
        "transport:prompt_studio.websocket",
    ]
    assert summary.components["transport:mcp.websocket"].result == "stopped"
    assert summary.components["transport:prompt_studio.websocket"].result == "stopped"
    assert sorted(events) == ["mcp", "prompt_studio"]


def test_main_shutdown_path_registers_transport_registry_components() -> None:
    from fastapi import FastAPI

    from tldw_Server_API.app.main import _build_coordinated_shutdown_coordinator
    from tldw_Server_API.app.services.shutdown_transport_registry import ShutdownTransportRegistry

    registry = ShutdownTransportRegistry()

    async def _drain_mcp(timeout_s: float | None = None) -> None:
        del timeout_s

    async def _drain_prompt_studio(timeout_s: float | None = None) -> None:
        del timeout_s

    registry.register_family(
        "mcp.websocket",
        active_count=lambda: 2,
        drain=_drain_mcp,
    )
    registry.register_family(
        "prompt_studio.websocket",
        active_count=lambda: 1,
        drain=_drain_prompt_studio,
    )

    coordinator, legacy_components, transport_components = _build_coordinated_shutdown_coordinator(
        FastAPI(),
        [],
        transport_registry=registry,
    )

    assert legacy_components == []
    assert [component.name for component in transport_components] == [
        "transport:mcp.websocket",
        "transport:prompt_studio.websocket",
    ]
    assert list(coordinator._components) == [
        "transport:mcp.websocket",
        "transport:prompt_studio.websocket",
    ]


@pytest.mark.asyncio
async def test_main_shutdown_path_drains_transport_components_with_empty_legacy_plan() -> None:
    from fastapi import FastAPI

    from tldw_Server_API.app.main import _run_coordinated_shutdown
    from tldw_Server_API.app.services.shutdown_transport_registry import ShutdownTransportRegistry

    app = FastAPI()
    events: list[str] = []
    registry = ShutdownTransportRegistry()

    async def _drain_mcp(timeout_s: float | None = None) -> None:
        del timeout_s
        events.append("mcp")

    async def _drain_prompt_studio(timeout_s: float | None = None) -> None:
        del timeout_s
        events.append("prompt_studio")

    registry.register_family(
        "mcp.websocket",
        active_count=lambda: 2,
        drain=_drain_mcp,
    )
    registry.register_family(
        "prompt_studio.websocket",
        active_count=lambda: 1,
        drain=_drain_prompt_studio,
    )

    suppressed = await _run_coordinated_shutdown(
        app,
        [],
        transport_registry=registry,
    )

    assert suppressed == set()
    assert app.state._tldw_shutdown_transport_component_names == [
        "transport:mcp.websocket",
        "transport:prompt_studio.websocket",
    ]
    assert app.state._tldw_shutdown_legacy_coordinator_component_names == [
        "transport:mcp.websocket",
        "transport:prompt_studio.websocket",
    ]
    assert sorted(events) == ["mcp", "prompt_studio"]


@pytest.mark.asyncio
async def test_coordinator_synthesizes_failed_summary_for_unexpected_component_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    coordinator.register(component("producer-a", phase="producers", stop=lambda: None))

    async def _boom(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(coordinator, "_run_component", _boom)

    summary = await coordinator.shutdown()

    assert summary.components["producer-a"].result == "failed"
    assert summary.components["producer-a"].error == "boom"


@pytest.mark.asyncio
async def test_coordinator_runs_components_by_phase_and_records_summary() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    events: list[str] = []

    coordinator.register(
        component("producer-a", phase="producers", stop=lambda: events.append("producer"))
    )
    coordinator.register(
        component("resource-a", phase="resources", stop=lambda: events.append("resource"))
    )

    summary = await coordinator.shutdown()

    assert events == ["producer", "resource"]
    assert summary.components["producer-a"].result == "stopped"


@pytest.mark.asyncio
async def test_coordinator_runs_transition_components_and_records_summary() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    events: list[str] = []

    coordinator.register(
        component("transition-a", phase="transition", stop=lambda: events.append("transition"))
    )

    summary = await coordinator.shutdown()

    assert events == ["transition"]
    assert summary.components["transition-a"].result == "stopped"
    assert summary.phases[ShutdownPhase.TRANSITION].component_names == ["transition-a"]


@pytest.mark.asyncio
async def test_coordinator_allocates_remaining_time_across_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    timeout_calls: list[float] = []

    async def _fake_wait_for(awaitable, timeout):
        timeout_calls.append(timeout)
        return await awaitable

    async def _slow_producer() -> None:
        clock.advance(4.0)

    async def _resource() -> None:
        return None

    monkeypatch.setattr("tldw_Server_API.app.services.shutdown_coordinator.asyncio.wait_for", _fake_wait_for)

    coordinator = ShutdownCoordinator(profile="dev_fast", clock=clock)
    coordinator.register(
        component("producer-a", phase="producers", stop=_slow_producer, default_timeout_ms=10_000)
    )
    coordinator.register(
        component("resource-a", phase="resources", stop=_resource, default_timeout_ms=10_000)
    )

    summary = await coordinator.shutdown()

    assert timeout_calls[0] > timeout_calls[1]
    assert summary.components["producer-a"].result == "stopped"
    assert summary.components["resource-a"].result == "stopped"


@pytest.mark.asyncio
async def test_coordinator_hard_cutoff_after_soft_overrun(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(now=0.009)
    timeout_calls: list[float] = []

    async def _fake_wait_for(awaitable, timeout):
        timeout_calls.append(timeout)
        return await awaitable

    async def _drain_eligible_resource() -> None:
        clock.advance(0.8)

    monkeypatch.setattr("tldw_Server_API.app.services.shutdown_coordinator.asyncio.wait_for", _fake_wait_for)

    coordinator = ShutdownCoordinator(
        profile="prod_drain",
        clock=clock,
        deadline_ms=10,
        soft_overrun_ms=1000,
    )
    coordinator.register(
        component("worker-a", phase="workers", policy="dev_fast", stop=lambda: None, default_timeout_ms=5_000)
    )
    coordinator.register(
        component("resource-a", phase="resources", stop=_drain_eligible_resource, default_timeout_ms=5_000)
    )

    summary = await coordinator.shutdown()

    assert timeout_calls[0] <= 0.01
    assert timeout_calls[1] >= 0.5
    assert summary.components["resource-a"].result == "stopped"
    assert summary.soft_overrun_used_ms > 0


@pytest.mark.asyncio
async def test_coordinator_custom_profile_honors_configured_soft_overrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(now=0.009)
    timeout_calls: list[float] = []

    async def _fake_wait_for(awaitable, timeout):
        timeout_calls.append(timeout)
        return await awaitable

    async def _resource() -> None:
        clock.advance(0.8)

    monkeypatch.setattr("tldw_Server_API.app.services.shutdown_coordinator.asyncio.wait_for", _fake_wait_for)

    coordinator = ShutdownCoordinator(
        profile="custom",
        clock=clock,
        deadline_ms=10,
        soft_overrun_ms=1000,
    )
    coordinator.register(
        component("resource-a", phase="resources", stop=_resource, default_timeout_ms=5_000)
    )

    summary = await coordinator.shutdown()

    assert timeout_calls[0] >= 0.5
    assert summary.components["resource-a"].result == "stopped"


@pytest.mark.asyncio
async def test_coordinator_uses_full_budget_when_only_one_runnable_phase_remains(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(now=0.009)
    timeout_calls: list[float] = []

    async def _fake_wait_for(awaitable, timeout):
        timeout_calls.append(timeout)
        return await awaitable

    async def _resource() -> None:
        clock.advance(0.2)

    monkeypatch.setattr("tldw_Server_API.app.services.shutdown_coordinator.asyncio.wait_for", _fake_wait_for)

    coordinator = ShutdownCoordinator(
        profile="prod_drain",
        clock=clock,
        deadline_ms=10,
        soft_overrun_ms=1000,
    )
    coordinator.register(
        component("resource-a", phase="resources", stop=_resource, default_timeout_ms=5_000)
    )

    summary = await coordinator.shutdown()

    assert timeout_calls[0] >= 0.9
    assert summary.components["resource-a"].result == "stopped"


@pytest.mark.asyncio
async def test_coordinator_best_effort_components_do_not_block_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    events: list[str] = []

    async def _fake_wait_for(awaitable, timeout):
        if timeout <= 0:
            raise TimeoutError
        return await awaitable

    async def _producer_stop() -> None:
        events.append("producer")
        clock.advance(6.0)

    async def _best_effort_stop() -> None:
        events.append("best_effort")
        raise AssertionError("best-effort stop should not have been awaited")

    monkeypatch.setattr("tldw_Server_API.app.services.shutdown_coordinator.asyncio.wait_for", _fake_wait_for)

    coordinator = ShutdownCoordinator(profile="dev_fast", clock=clock)
    coordinator.register(component("producer-a", phase="producers", stop=_producer_stop))
    coordinator.register(
        ShutdownComponent(
            name="resource-a",
            phase="resources",
            policy="best_effort",
            default_timeout_ms=50,
            stop=_best_effort_stop,
        )
    )

    summary = await coordinator.shutdown()

    assert events == ["producer"]
    assert summary.components["resource-a"].result == "skipped"
    assert summary.components["resource-a"].timeout_ms == 0


@pytest.mark.asyncio
async def test_coordinator_best_effort_only_later_phases_do_not_reduce_mandatory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    timeout_calls: list[float] = []
    events: list[str] = []

    async def _fake_wait_for(awaitable, timeout):
        timeout_calls.append(timeout)
        return await awaitable

    async def _producer_stop() -> None:
        events.append("producer")

    async def _best_effort_stop() -> None:
        events.append("best_effort")

    monkeypatch.setattr("tldw_Server_API.app.services.shutdown_coordinator.asyncio.wait_for", _fake_wait_for)

    coordinator = ShutdownCoordinator(profile="dev_fast", clock=clock)
    coordinator.register(component("producer-a", phase="producers", stop=_producer_stop, default_timeout_ms=10_000))
    coordinator.register(
        ShutdownComponent(
            name="resource-a",
            phase="resources",
            policy="best_effort",
            default_timeout_ms=10_000,
            stop=_best_effort_stop,
        )
    )

    summary = await coordinator.shutdown()

    assert timeout_calls[0] >= 4.9
    assert summary.components["producer-a"].result == "stopped"
    assert summary.components["resource-a"].result == "stopped"
    assert events == ["producer", "best_effort"]


@pytest.mark.asyncio
async def test_coordinator_records_timeout_when_stop_swallows_cancellation() -> None:
    coordinator = ShutdownCoordinator(
        profile="custom",
        deadline_ms=10,
        soft_overrun_ms=0,
    )
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def _stop() -> None:
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    coordinator.register(
        component(
            "producer-a",
            phase="producers",
            stop=_stop,
            policy=ShutdownPolicy.DEV_FAST,
            default_timeout_ms=1_000,
        )
    )

    summary = await asyncio.wait_for(coordinator.shutdown(), timeout=1.0)
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    release.set()

    assert summary.components["producer-a"].result in {"timed_out", "cancelled"}
    assert summary.components["producer-a"].result != "stopped"


@pytest.mark.asyncio
async def test_coordinator_preserves_completed_result_after_timeout_quiescence() -> None:
    coordinator = ShutdownCoordinator(
        profile="custom",
        deadline_ms=10,
        soft_overrun_ms=100,
    )
    cancelled = asyncio.Event()

    async def _stop() -> None:
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            cancelled.set()
            return

    coordinator.register(
        component(
            "producer-a",
            phase="producers",
            stop=_stop,
            policy=ShutdownPolicy.DEV_FAST,
            default_timeout_ms=1_000,
        )
    )

    summary = await coordinator.shutdown()

    assert cancelled.is_set()
    assert summary.components["producer-a"].result == "stopped"
    assert summary.components["producer-a"].budget_exhausted is True


@pytest.mark.asyncio
async def test_coordinator_records_failed_result_when_stop_raises() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")

    def _stop() -> None:
        raise RuntimeError("boom")

    coordinator.register(component("producer-a", phase="producers", stop=_stop))

    summary = await coordinator.shutdown()

    assert summary.components["producer-a"].result == "failed"
    assert summary.components["producer-a"].error == "boom"


@pytest.mark.asyncio
async def test_coordinator_concurrent_shutdown_calls_share_in_progress_run() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    started = asyncio.Event()
    release = asyncio.Event()
    stop_count = 0

    async def _stop() -> None:
        nonlocal stop_count
        stop_count += 1
        started.set()
        await release.wait()

    coordinator.register(
        component("producer-a", phase="producers", stop=_stop, default_timeout_ms=5_000)
    )

    first = asyncio.create_task(coordinator.shutdown())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    second = asyncio.create_task(coordinator.shutdown())
    await asyncio.sleep(0)
    release.set()

    first_summary, second_summary = await asyncio.gather(first, second)

    assert stop_count == 1
    assert first_summary is not second_summary
    assert first_summary.idempotent is False
    assert second_summary.idempotent is True


@pytest.mark.asyncio
async def test_coordinator_cancelled_waiter_does_not_cancel_shared_shutdown_run() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    started = asyncio.Event()
    release = asyncio.Event()
    stop_count = 0

    async def _stop() -> None:
        nonlocal stop_count
        stop_count += 1
        started.set()
        await release.wait()

    coordinator.register(
        component("producer-a", phase="producers", stop=_stop, default_timeout_ms=5_000)
    )

    waiter = asyncio.create_task(coordinator.shutdown())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    second = asyncio.create_task(coordinator.shutdown())
    release.set()
    summary = await second

    assert stop_count == 1
    assert summary.components["producer-a"].result == "stopped"


@pytest.mark.asyncio
async def test_coordinator_rejects_registration_during_and_after_shutdown() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    events: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def _stop() -> None:
        events.append("producer")
        started.set()
        await release.wait()

    coordinator.register(component("producer-a", phase="producers", stop=_stop))

    task = asyncio.create_task(coordinator.shutdown())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    with pytest.raises(RuntimeError):
        coordinator.register(component("late-a", phase="resources", stop=lambda: None))

    release.set()
    await task

    with pytest.raises(RuntimeError):
        coordinator.register(component("late-b", phase="resources", stop=lambda: None))


@pytest.mark.asyncio
async def test_coordinator_second_shutdown_call_returns_copy_without_mutating_first_summary() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    events: list[str] = []

    coordinator.register(component("producer-a", phase="producers", stop=lambda: events.append("producer")))

    first = await coordinator.shutdown()
    second = await coordinator.shutdown()

    assert first is not second
    assert first.idempotent is False
    assert second.idempotent is True
    second.components["producer-a"].result = "failed"
    second.phases[ShutdownPhase.PRODUCERS].component_names.append("mutated")
    assert first.components["producer-a"].result == "stopped"
    assert first.phases[ShutdownPhase.PRODUCERS].component_names == ["producer-a"]
    assert events == ["producer"]


@pytest.mark.asyncio
async def test_coordinator_first_shutdown_return_does_not_alias_cached_summary() -> None:
    coordinator = ShutdownCoordinator(profile="dev_fast")
    events: list[str] = []

    coordinator.register(component("producer-a", phase="producers", stop=lambda: events.append("producer")))

    first = await coordinator.shutdown()
    first.components["producer-a"].result = "failed"
    first.phases[ShutdownPhase.PRODUCERS].component_names.append("mutated")

    second = await coordinator.shutdown()

    assert first is not second
    assert second.idempotent is True
    assert second.components["producer-a"].result == "stopped"
    assert second.phases[ShutdownPhase.PRODUCERS].component_names == ["producer-a"]
    assert events == ["producer"]
