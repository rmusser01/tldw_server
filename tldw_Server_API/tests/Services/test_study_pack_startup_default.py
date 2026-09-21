"""The active Study worker catalog preserves route defaults and worker ownership."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services.lifecycle_worker_engine import LifecycleWorkerEngine
from tldw_Server_API.app.services.lifecycle_worker_specs import ShutdownPhase, WorkerLifecycleContext

pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[
        ("STUDY_PACK_JOBS_WORKER_ENABLED", "study_pack_jobs_task", "flashcards", "_run_study_pack_jobs_worker_service"),
        (
            "STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED",
            "study_suggestions_jobs_task",
            "study-suggestions",
            "_run_study_suggestions_jobs_worker_service",
        ),
    ],
    ids=["study-pack", "study-suggestions"],
)
def study_worker(request):
    return request.param


def _context(*, route_key: str, route: bool, sidecar: bool = False, test_mode: bool = False):
    return WorkerLifecycleContext(
        app=SimpleNamespace(state=SimpleNamespace()),
        settings={},
        test_mode=test_mode,
        route_enabled=lambda key, **_kwargs: route if key == route_key else False,
        logger=None,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
        sidecar_mode=sidecar,
    )


@pytest.mark.parametrize(
    ("flag", "route", "sidecar", "test_mode", "expected"),
    [
        (None, True, False, False, True),
        ("", True, False, False, True),
        ("true", True, False, False, True),
        ("false", True, False, False, False),
        (None, False, False, False, False),
        ("true", False, False, False, False),
        (None, True, True, False, False),
        ("true", True, True, False, False),
        (None, True, False, True, False),
        ("true", True, False, True, True),
    ],
)
def test_study_worker_spec_respects_route_flag_and_runtime_mode(
    monkeypatch, study_worker, flag, route, sidecar, test_mode, expected
):
    flag_key, name, route_key, _service = study_worker
    pollers = importlib.import_module("tldw_Server_API.app.services.startup_study_privilege_jobs_pollers")
    if flag is None:
        monkeypatch.delenv(flag_key, raising=False)
    else:
        monkeypatch.setenv(flag_key, flag)
    spec = next(spec for spec in pollers.provide_study_privilege_jobs_worker_specs() if spec.name == name)
    assert spec.enabled(_context(route_key=route_key, route=route, sidecar=sidecar, test_mode=test_mode)) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flag", "route", "sidecar", "expected"),
    [
        (None, True, False, True),
        ("true", True, False, True),
        ("false", True, False, False),
        (None, False, False, False),
        ("true", True, True, False),
    ],
)
async def test_active_catalog_bootstrap_registers_and_quiesces_study_worker(
    monkeypatch, study_worker, flag, route, sidecar, expected
):
    flag_key, name, route_key, service = study_worker
    bootstrap = importlib.import_module("tldw_Server_API.app.services.startup_worker_bootstrap")
    pollers = importlib.import_module("tldw_Server_API.app.services.startup_study_privilege_jobs_pollers")
    if flag is None:
        monkeypatch.delenv(flag_key, raising=False)
    else:
        monkeypatch.setenv(flag_key, flag)
    monkeypatch.setenv("TLDW_WORKERS_SIDECAR_MODE", str(sidecar).lower())
    context = _context(route_key=route_key, route=route)
    started = []
    stopped = []

    async def worker_body(stop_event):
        started.append(stop_event)
        try:
            await stop_event.wait()
        finally:
            stopped.append(stop_event)

    monkeypatch.setattr(pollers, service, worker_body)
    actual_collect = bootstrap._collect_startup_worker_specs
    collected = []

    def only_study_worker(actual_context):
        # Keep the real active provider catalog and this spec unchanged. Other
        # worker bodies are outside this bounded lifecycle test.
        specs = actual_collect(actual_context)
        matching = tuple(spec for spec in specs if spec.name == name)
        assert len(matching) == 1
        collected.extend(matching)
        return matching

    async def no_non_worker_tail(**_kwargs):
        return None

    monkeypatch.setattr(bootstrap, "_collect_startup_worker_specs", only_study_worker)
    monkeypatch.setattr(bootstrap, "_load_app_settings", lambda: {})
    monkeypatch.setattr(bootstrap, "_run_startup_non_worker_tail", no_non_worker_tail)
    handles = await bootstrap.initialize_startup_worker_bootstrap(
        app=context.app,
        test_mode=False,
        route_enabled=context.route_enabled,
        run_pg_rls_auto_ensure=None,
        register_owned_job_poller=None,
        replace_owned_job_poller_inventory=None,
        logger=None,
        startup_api_key_log_value=None,
        shared_is_truthy=None,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )
    session = handles.worker_lifecycle_session
    assert session is not None
    engine = LifecycleWorkerEngine()
    try:
        assert len(collected) == 1
        assert not session.startup_failures
        assert (name in session.handles_by_name) is expected
        assert (name in session.disabled_names) is not expected
        if expected:
            handle = session.handles_by_name[name]
            assert started == [handle.stop_event]
            assert handle.task.get_name() == name
            assert not handle.task.done()
        else:
            assert started == []
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)
    if expected:
        handle = session.handles_by_name[name]
        assert handle.stop_event.is_set()
        assert handle.task.done() and not handle.task.cancelled()
        assert stopped == [handle.stop_event]
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)
        assert stopped == [handle.stop_event]
