"""The active StudyPack catalog preserves route defaults and worker ownership."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services.lifecycle_worker_engine import LifecycleWorkerEngine
from tldw_Server_API.app.services.lifecycle_worker_specs import ShutdownPhase, WorkerLifecycleContext

pytestmark = pytest.mark.unit
FLAG = "STUDY_PACK_JOBS_WORKER_ENABLED"
NAME = "study_pack_jobs_task"


def _context(*, route: bool, sidecar: bool = False, test_mode: bool = False):
    return WorkerLifecycleContext(
        app=SimpleNamespace(state=SimpleNamespace()),
        settings={},
        test_mode=test_mode,
        route_enabled=lambda key, **_kwargs: route if key == "flashcards" else False,
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
def test_study_pack_spec_respects_route_flag_and_runtime_mode(monkeypatch, flag, route, sidecar, test_mode, expected):
    pollers = importlib.import_module("tldw_Server_API.app.services.startup_study_privilege_jobs_pollers")
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    spec = next(spec for spec in pollers.provide_study_privilege_jobs_worker_specs() if spec.name == NAME)
    assert spec.enabled(_context(route=route, sidecar=sidecar, test_mode=test_mode)) is expected


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
async def test_active_catalog_bootstrap_registers_and_quiesces_study_pack(monkeypatch, flag, route, sidecar, expected):
    bootstrap = importlib.import_module("tldw_Server_API.app.services.startup_worker_bootstrap")
    pollers = importlib.import_module("tldw_Server_API.app.services.startup_study_privilege_jobs_pollers")
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv("TLDW_WORKERS_SIDECAR_MODE", str(sidecar).lower())
    context = _context(route=route)
    started = []
    stopped = []

    async def worker_body(stop_event):
        started.append(stop_event)
        try:
            await stop_event.wait()
        finally:
            stopped.append(stop_event)

    monkeypatch.setattr(pollers, "_run_study_pack_jobs_worker_service", worker_body)
    actual_collect = bootstrap._collect_startup_worker_specs
    collected = []

    def only_study_pack(actual_context):
        # Keep the real active provider catalog and this spec unchanged. Other
        # worker bodies are outside this bounded lifecycle test.
        specs = actual_collect(actual_context)
        matching = tuple(spec for spec in specs if spec.name == NAME)
        assert len(matching) == 1
        collected.extend(matching)
        return matching

    async def no_non_worker_tail(**_kwargs):
        return None

    monkeypatch.setattr(bootstrap, "_collect_startup_worker_specs", only_study_pack)
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
        assert (NAME in session.handles_by_name) is expected
        assert (NAME in session.disabled_names) is not expected
        if expected:
            handle = session.handles_by_name[NAME]
            assert started == [handle.stop_event]
            assert handle.task.get_name() == NAME
            assert not handle.task.done()
        else:
            assert started == []
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)
    if expected:
        handle = session.handles_by_name[NAME]
        assert handle.stop_event.is_set()
        assert handle.task.done() and not handle.task.cancelled()
        assert stopped == [handle.stop_event]
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)
        assert stopped == [handle.stop_event]
