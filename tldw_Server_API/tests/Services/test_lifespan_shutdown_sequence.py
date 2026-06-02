from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_lifespan_shutdown_sequence_stops_lifecycle_phases_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import (
        lifespan_shutdown_sequence,
        shutdown_coordinated_legacy_components,
        shutdown_final_cleanup_tail,
        shutdown_job_poller_handoff,
        shutdown_post_worker_services,
        shutdown_pre_worker_cleanup,
        shutdown_transition_handoff,
    )
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    app = FastAPI()

    class _FakeLifecycleSession:
        def __init__(self) -> None:
            self.name = "worker-session"
            self.stopped_names_by_phase: dict[ShutdownPhase, list[str]] = {}

        def mark_stopped(self, name: str, phase: ShutdownPhase) -> None:
            self.stopped_names_by_phase.setdefault(phase, []).append(name)

        def publish_stopped_names(self, phase: ShutdownPhase) -> None:
            attr_by_phase = {
                ShutdownPhase.JOB_POLLER_QUIESCE: "_tldw_shutdown_quiesced_job_poller_names",
                ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN: (
                    "_tldw_shutdown_stopped_background_worker_names"
                ),
                ShutdownPhase.POST_WORKER_SHUTDOWN: "_tldw_shutdown_stopped_post_worker_names",
            }
            setattr(
                app.state,
                attr_by_phase[phase],
                self.stopped_names_by_phase.get(phase, []),
            )

    worker_lifecycle_session = _FakeLifecycleSession()
    worker_runtime = LifespanWorkerRuntimeState(
        worker_lifecycle_session=worker_lifecycle_session,
    )
    calls: list[tuple[str, dict[str, object]]] = []
    recorded_totals: list[int] = []
    monotonic_values = iter((10.0, 10.25))

    @contextmanager
    def _timed_shutdown_segment(_app: FastAPI, segment_name: str):
        calls.append(("segment", {"segment_name": segment_name}))
        yield

    async def _fake_transition_handoff(**kwargs):
        calls.append(("transition", kwargs))
        return SimpleNamespace(legacy_shutdown_plan=["transition-plan"])

    async def _fake_stop_phase(session: object, phase: ShutdownPhase) -> None:
        calls.append(("engine", {"session": session, "phase": phase}))
        if phase is ShutdownPhase.JOB_POLLER_QUIESCE:
            session.mark_stopped("core_jobs_task", phase)
        if phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN:
            session.mark_stopped("jobs_metrics_task", phase)
        if phase is ShutdownPhase.POST_WORKER_SHUTDOWN:
            session.mark_stopped("jobs_webhooks_task", phase)

    lifecycle_worker_engine = SimpleNamespace(stop_phase=_fake_stop_phase)

    async def _fake_job_poller_handoff(**kwargs):
        calls.append(("pollers", kwargs))
        await kwargs["lifecycle_worker_engine"].stop_phase(
            kwargs["worker_lifecycle_session"],
            ShutdownPhase.JOB_POLLER_QUIESCE,
        )
        kwargs["worker_lifecycle_session"].publish_stopped_names(
            ShutdownPhase.JOB_POLLER_QUIESCE
        )
        return SimpleNamespace(
            should_run_late_stop=lambda task_name, task: bool(task)
            and task_name != "core_jobs_task",
        )

    async def _fake_coordinated_shutdown(**kwargs):
        calls.append(("coordinated", kwargs))
        return SimpleNamespace(
            coordinated_legacy_component_names={"usage_aggregator"},
        )

    async def _fake_pre_worker_cleanup(**kwargs):
        calls.append(("pre", kwargs))
        return SimpleNamespace()

    async def _fake_post_worker_non_worker_cleanup(**kwargs):
        calls.append(("post-cleanup", kwargs))
        return SimpleNamespace()

    async def _fake_final_cleanup(**kwargs):
        calls.append(("final", kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(
        shutdown_transition_handoff,
        "shutdown_transition_handoff",
        _fake_transition_handoff,
    )
    monkeypatch.setattr(
        lifespan_shutdown_sequence,
        "LifecycleWorkerEngine",
        lambda: lifecycle_worker_engine,
    )
    monkeypatch.setattr(
        shutdown_job_poller_handoff,
        "run_shutdown_job_poller_handoff",
        _fake_job_poller_handoff,
    )
    monkeypatch.setattr(
        shutdown_coordinated_legacy_components,
        "run_shutdown_coordinated_legacy_components",
        _fake_coordinated_shutdown,
    )
    monkeypatch.setattr(
        shutdown_pre_worker_cleanup,
        "run_shutdown_pre_worker_cleanup",
        _fake_pre_worker_cleanup,
    )
    monkeypatch.setattr(
        shutdown_post_worker_services,
        "run_shutdown_post_worker_non_worker_cleanup",
        _fake_post_worker_non_worker_cleanup,
    )
    monkeypatch.setattr(
        shutdown_final_cleanup_tail,
        "shutdown_final_cleanup_tail",
        _fake_final_cleanup,
    )

    await lifespan_shutdown_sequence.run_lifespan_shutdown_sequence(
        app=app,
        worker_runtime=worker_runtime,
        readiness_state={"ready": True},
        db_pool="db-pool",
        session_manager="session-manager",
        heavy_startup_handles="heavy-handles",
        build_legacy_shutdown_context=lambda **kwargs: kwargs,
        apply_shutdown_transition_gate=lambda *_args, **_kwargs: None,
        quiesce_owned_job_pollers_for_shutdown=lambda *_args, **_kwargs: None,
        run_coordinated_shutdown=lambda **_kwargs: None,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
        in_pytest_runtime=True,
        test_db_instance_ref="test-db-ref",
        timed_shutdown_segment=_timed_shutdown_segment,
        record_shutdown_timing_total=lambda _app, total_ms: recorded_totals.append(total_ms),
        monotonic=lambda: next(monotonic_values),
    )

    assert [name for name, _ in calls] == [
        "segment",
        "transition",
        "pollers",
        "engine",
        "segment",
        "engine",
        "coordinated",
        "pre",
        "engine",
        "post-cleanup",
        "final",
    ]
    assert calls[0][1]["segment_name"] == "transition_handoff"
    assert calls[2][1]["worker_lifecycle_session"] is worker_lifecycle_session
    assert calls[2][1]["lifecycle_worker_engine"] is lifecycle_worker_engine
    assert "owned_job_pollers" not in calls[2][1]
    assert calls[3][1] == {
        "session": worker_lifecycle_session,
        "phase": ShutdownPhase.JOB_POLLER_QUIESCE,
    }
    assert calls[4][1]["segment_name"] == "background_worker_shutdown"
    assert calls[5][1] == {
        "session": worker_lifecycle_session,
        "phase": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    }
    assert calls[6][1]["legacy_shutdown_plan"] == ["transition-plan"]
    assert calls[6][1]["stopped_background_worker_names"] == {"jobs_metrics_task"}
    assert calls[8][1] == {
        "session": worker_lifecycle_session,
        "phase": ShutdownPhase.POST_WORKER_SHUTDOWN,
    }
    assert "jobs_metrics_task" not in calls[9][1]
    assert "media_ingest_jobs_task" not in calls[9][1]
    assert "stopped_background_worker_names" not in calls[9][1]
    assert calls[10][1]["db_pool"] == "db-pool"
    assert calls[10][1]["session_manager"] == "session-manager"
    assert calls[10][1]["heavy_startup_handles"] == "heavy-handles"
    assert calls[10][1]["in_pytest_for_db_pool_shutdown"] is True
    assert calls[10][1]["in_pytest_for_tts_shutdown"] is True
    assert app.state._tldw_shutdown_quiesced_job_poller_names == ["core_jobs_task"]
    assert app.state._tldw_shutdown_stopped_background_worker_names == [
        "jobs_metrics_task"
    ]
    assert app.state._tldw_shutdown_stopped_post_worker_names == ["jobs_webhooks_task"]
    assert app.state._tldw_shutdown_timing_segments == []
    assert recorded_totals == [250]


@pytest.mark.asyncio
async def test_run_lifespan_shutdown_sequence_skips_managed_phase_stops_without_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import (
        lifespan_shutdown_sequence,
        shutdown_coordinated_legacy_components,
        shutdown_final_cleanup_tail,
        shutdown_job_poller_handoff,
        shutdown_post_worker_services,
        shutdown_pre_worker_cleanup,
        shutdown_transition_handoff,
    )
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    app = FastAPI()
    app.state._tldw_shutdown_stopped_background_worker_names = ["stale_worker"]
    calls: list[str] = []
    coordinated_stopped_names: list[set[str]] = []

    @contextmanager
    def _timed_shutdown_segment(_app: FastAPI, _segment_name: str):
        yield

    async def _fake_transition_handoff(**_kwargs):
        calls.append("transition")
        return SimpleNamespace(legacy_shutdown_plan=[])

    async def _fake_job_poller_handoff(**kwargs):
        calls.append("pollers")
        assert kwargs["worker_lifecycle_session"] is None
        return SimpleNamespace(should_run_late_stop=lambda *_args: False)

    async def _fake_stop_phase(*_args, **_kwargs) -> None:
        calls.append("engine")

    async def _fake_coordinated_shutdown(**kwargs):
        calls.append("coordinated")
        coordinated_stopped_names.append(kwargs["stopped_background_worker_names"])

    async def _fake_pre_worker_cleanup(**_kwargs):
        calls.append("pre")
        return SimpleNamespace()

    async def _fake_post_worker_non_worker_cleanup(**_kwargs):
        calls.append("post-cleanup")
        return SimpleNamespace()

    async def _fake_final_cleanup(**_kwargs):
        calls.append("final")
        return SimpleNamespace()

    monkeypatch.setattr(
        shutdown_transition_handoff,
        "shutdown_transition_handoff",
        _fake_transition_handoff,
    )
    monkeypatch.setattr(
        lifespan_shutdown_sequence,
        "LifecycleWorkerEngine",
        lambda: SimpleNamespace(stop_phase=_fake_stop_phase),
    )
    monkeypatch.setattr(
        shutdown_job_poller_handoff,
        "run_shutdown_job_poller_handoff",
        _fake_job_poller_handoff,
    )
    monkeypatch.setattr(
        shutdown_coordinated_legacy_components,
        "run_shutdown_coordinated_legacy_components",
        _fake_coordinated_shutdown,
    )
    monkeypatch.setattr(
        shutdown_pre_worker_cleanup,
        "run_shutdown_pre_worker_cleanup",
        _fake_pre_worker_cleanup,
    )
    monkeypatch.setattr(
        shutdown_post_worker_services,
        "run_shutdown_post_worker_non_worker_cleanup",
        _fake_post_worker_non_worker_cleanup,
    )
    monkeypatch.setattr(
        shutdown_final_cleanup_tail,
        "shutdown_final_cleanup_tail",
        _fake_final_cleanup,
    )

    await lifespan_shutdown_sequence.run_lifespan_shutdown_sequence(
        app=app,
        worker_runtime=LifespanWorkerRuntimeState(),
        readiness_state={},
        db_pool=None,
        session_manager=None,
        heavy_startup_handles=None,
        build_legacy_shutdown_context=lambda **kwargs: kwargs,
        apply_shutdown_transition_gate=lambda *_args, **_kwargs: None,
        quiesce_owned_job_pollers_for_shutdown=lambda *_args, **_kwargs: None,
        run_coordinated_shutdown=lambda **_kwargs: None,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
        in_pytest_runtime=True,
        test_db_instance_ref=None,
        timed_shutdown_segment=_timed_shutdown_segment,
        record_shutdown_timing_total=lambda *_args, **_kwargs: None,
        monotonic=lambda: 1.0,
    )

    assert calls == [
        "transition",
        "pollers",
        "coordinated",
        "pre",
        "post-cleanup",
        "final",
    ]
    assert coordinated_stopped_names == [set()]
    assert app.state._tldw_shutdown_stopped_background_worker_names == []
