from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services.shutdown_models import (
    ShutdownComponent,
    ShutdownComponentSummary,
    ShutdownPhase,
    ShutdownPhaseSummary,
    ShutdownPolicy,
    ShutdownSummary,
)

pytestmark = pytest.mark.unit


def _import_shutdown_transition_handoff():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_transition_handoff", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_transition_handoff")


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


def _summary_for_components(
    components: list[ShutdownComponent],
    *,
    results: dict[str, str],
) -> ShutdownSummary:
    phase_groups: dict[ShutdownPhase, list[str]] = {}
    component_summaries: dict[str, ShutdownComponentSummary] = {}

    for component in components:
        phase_groups.setdefault(component.phase, []).append(component.name)
        component_summaries[component.name] = ShutdownComponentSummary(
            name=component.name,
            phase=component.phase,
            policy=component.policy,
            result=results[component.name],
            started_at=0.0,
            finished_at=0.0,
            duration_ms=0,
            timeout_ms=component.default_timeout_ms,
        )

    phase_summaries = {
        phase: ShutdownPhaseSummary(
            phase=phase,
            started_at=0.0,
            finished_at=0.0,
            duration_ms=0,
            budget_ms=0,
            component_names=component_names,
        )
        for phase, component_names in phase_groups.items()
    }
    return ShutdownSummary(
        profile="dev_fast",
        started_at=0.0,
        finished_at=0.0,
        deadline_at=0.0,
        hard_cutoff_at=0.0,
        wall_time_ms=0,
        soft_overrun_used_ms=0,
        components=component_summaries,
        phases=phase_summaries,
    )


class _FakeCoordinator:
    def __init__(self, summary: ShutdownSummary) -> None:
        self.summary = summary
        self.registered: list[ShutdownComponent] = []
        self.shutdown_calls = 0

    def register(self, component: ShutdownComponent) -> None:
        self.registered.append(component)

    async def shutdown(self) -> ShutdownSummary:
        self.shutdown_calls += 1
        return self.summary


@pytest.mark.asyncio
async def test_shutdown_transition_handoff_returns_plan_and_skips_direct_drain_when_transition_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_handoff = _import_shutdown_transition_handoff()
    app = SimpleNamespace(state=SimpleNamespace())
    build_context_calls: list[dict[str, object]] = []
    apply_calls: list[tuple[object, object]] = []
    created_profiles: list[str] = []

    plan = [
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
    ]
    coordinator = _FakeCoordinator(
        _summary_for_components(
            [plan[0]],
            results={"lifecycle_gate": "stopped"},
        )
    )

    def _fake_build_context(**kwargs):
        build_context_calls.append(kwargs)
        return "shutdown-context"

    def _fake_apply_transition_gate(app_obj, readiness_state):
        apply_calls.append((app_obj, readiness_state))

    monkeypatch.setattr(
        shutdown_handoff,
        "_build_legacy_shutdown_plan",
        lambda app_obj, context: plan if app_obj is app and context == "shutdown-context" else [],
    )
    monkeypatch.setattr(
        shutdown_handoff,
        "_create_transition_coordinator",
        lambda *, profile: created_profiles.append(profile) or coordinator,
    )

    handles = await shutdown_handoff.shutdown_transition_handoff(
        app=app,
        readiness_state="ready-state",
        build_legacy_shutdown_context=_fake_build_context,
        apply_shutdown_transition_gate=_fake_apply_transition_gate,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert len(build_context_calls) == 1
    assert build_context_calls[0]["readiness_state"] == "ready-state"
    assert "usage_task" not in build_context_calls[0]
    assert "llm_usage_task" not in build_context_calls[0]
    assert "authnz_scheduler_started" not in build_context_calls[0]
    assert created_profiles == ["dev_fast"]
    assert [component.name for component in coordinator.registered] == ["lifecycle_gate"]
    assert coordinator.shutdown_calls == 1
    assert handles.legacy_shutdown_plan == plan
    assert handles.transition_gate_applied is True
    assert app.state._tldw_shutdown_legacy_plan == plan
    assert app.state._tldw_shutdown_legacy_phase_groups == {
        "transition": ["lifecycle_gate"],
        "resources": ["usage_aggregator"],
    }
    assert app.state._tldw_shutdown_legacy_inventory_visible is True
    assert apply_calls == []


@pytest.mark.asyncio
async def test_shutdown_transition_handoff_applies_direct_drain_when_transition_not_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_handoff = _import_shutdown_transition_handoff()
    app = SimpleNamespace(state=SimpleNamespace())
    apply_calls: list[tuple[object, object]] = []

    plan = [
        _component(
            "lifecycle_gate",
            phase=ShutdownPhase.TRANSITION,
            policy=ShutdownPolicy.DEV_FAST,
        ),
    ]
    coordinator = _FakeCoordinator(
        _summary_for_components(
            plan,
            results={"lifecycle_gate": "skipped"},
        )
    )

    monkeypatch.setattr(
        shutdown_handoff,
        "_build_legacy_shutdown_plan",
        lambda _app, _context: plan,
    )
    monkeypatch.setattr(
        shutdown_handoff,
        "_create_transition_coordinator",
        lambda *, profile: coordinator,
    )

    handles = await shutdown_handoff.shutdown_transition_handoff(
        app=app,
        readiness_state="ready-state",
        build_legacy_shutdown_context=lambda **kwargs: kwargs,
        apply_shutdown_transition_gate=lambda app_obj, readiness_state: apply_calls.append((app_obj, readiness_state)),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles.legacy_shutdown_plan == plan
    assert handles.transition_gate_applied is False
    assert apply_calls == [(app, "ready-state")]


@pytest.mark.asyncio
async def test_shutdown_transition_handoff_applies_direct_drain_when_plan_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_handoff = _import_shutdown_transition_handoff()
    app = SimpleNamespace(state=SimpleNamespace())
    apply_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        shutdown_handoff,
        "_build_legacy_shutdown_plan",
        lambda _app, _context: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        shutdown_handoff,
        "_create_transition_coordinator",
        lambda *, profile: (_ for _ in ()).throw(AssertionError("coordinator should not be created")),
    )

    handles = await shutdown_handoff.shutdown_transition_handoff(
        app=app,
        readiness_state="ready-state",
        build_legacy_shutdown_context=lambda **kwargs: kwargs,
        apply_shutdown_transition_gate=lambda app_obj, readiness_state: apply_calls.append((app_obj, readiness_state)),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert handles.legacy_shutdown_plan == []
    assert handles.transition_gate_applied is False
    assert apply_calls == [(app, "ready-state")]


@pytest.mark.asyncio
async def test_shutdown_transition_handoff_applies_direct_drain_without_running_empty_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_handoff = _import_shutdown_transition_handoff()
    app = SimpleNamespace(state=SimpleNamespace())
    apply_calls: list[tuple[object, object]] = []
    created_profiles: list[str] = []
    coordinator = _FakeCoordinator(
        _summary_for_components(
            [],
            results={},
        )
    )

    monkeypatch.setattr(
        shutdown_handoff,
        "_build_legacy_shutdown_plan",
        lambda _app, _context: [],
    )
    monkeypatch.setattr(
        shutdown_handoff,
        "_create_transition_coordinator",
        lambda *, profile: created_profiles.append(profile) or coordinator,
    )

    handles = await shutdown_handoff.shutdown_transition_handoff(
        app=app,
        readiness_state="ready-state",
        build_legacy_shutdown_context=lambda **kwargs: kwargs,
        apply_shutdown_transition_gate=lambda app_obj, readiness_state: apply_calls.append((app_obj, readiness_state)),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert created_profiles == ["dev_fast"]
    assert coordinator.registered == []
    assert coordinator.shutdown_calls == 0
    assert handles.legacy_shutdown_plan == []
    assert handles.transition_gate_applied is False
    assert apply_calls == [(app, "ready-state")]
