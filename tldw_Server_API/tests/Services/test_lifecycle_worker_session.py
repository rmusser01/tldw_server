import asyncio

import pytest
from fastapi import FastAPI


async def _wait_for_stop(stop_event: asyncio.Event) -> None:
    await stop_event.wait()


async def _noop_callback() -> None:
    return None


def _worker_spec(**overrides: object):
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        ShutdownPhase,
        WorkerSpec,
    )

    values = {
        "name": "worker_a",
        "task_name": "worker-a-task",
        "category": "jobs",
        "phase": ShutdownPhase.JOB_POLLER_QUIESCE,
        "factory": lambda _context, stop_event: stop_event.wait(),
    }
    values.update(overrides)
    return WorkerSpec(**values)


def _worker_graph(*specs: object):
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        validate_worker_spec_graph,
    )

    return validate_worker_spec_graph(list(specs))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_publishes_full_and_job_poller_compatibility_inventory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_session import (
        WorkerLifecycleSession,
    )
    from tldw_Server_API.app.services.lifecycle_worker_specs import ShutdownPhase
    from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker

    app = FastAPI()
    job_stop_event = asyncio.Event()
    background_stop_event = asyncio.Event()
    job_task = asyncio.create_task(_wait_for_stop(job_stop_event), name="job-task")
    background_task = asyncio.create_task(
        _wait_for_stop(background_stop_event),
        name="background-task",
    )
    job_spec = _worker_spec(
        name="job_worker",
        task_name="job-task",
        category="jobs",
        phase=ShutdownPhase.JOB_POLLER_QUIESCE,
        timeout_sec=3.0,
    )
    background_spec = _worker_spec(
        name="background_worker",
        task_name="background-task",
        category="maintenance",
        phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        timeout_sec=7.0,
    )
    session = WorkerLifecycleSession(
        app=app,
        graph=_worker_graph(job_spec, background_spec),
    )

    try:
        session.register_handle(
            job_spec,
            ManagedWorker(
                name="job_worker",
                task=job_task,
                stop_event=job_stop_event,
                timeout_sec=3.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            ),
        )
        session.register_handle(
            background_spec,
            ManagedWorker(
                name="background_worker",
                task=background_task,
                stop_event=background_stop_event,
                timeout_sec=7.0,
                category="maintenance",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            ),
        )

        session.publish_inventory()

        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "job_worker",
                "task_name": "job-task",
                "has_stop_event": True,
                "timeout_sec": 3.0,
                "category": "jobs",
                "shutdown_phase": "job_poller_quiesce",
            },
            {
                "name": "background_worker",
                "task_name": "background-task",
                "has_stop_event": True,
                "timeout_sec": 7.0,
                "category": "maintenance",
                "shutdown_phase": "background_worker_shutdown",
            },
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "job_worker",
                "task_name": "job-task",
                "has_stop_event": True,
                "timeout_sec": 3.0,
            }
        ]
    finally:
        job_stop_event.set()
        background_stop_event.set()
        await asyncio.wait_for(asyncio.gather(job_task, background_task), timeout=1)


@pytest.mark.unit
def test_session_publishes_callback_only_worker_without_stop_event() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_session import (
        WorkerLifecycleSession,
    )
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        ShutdownPhase,
        WorkerStrategy,
    )
    from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker

    app = FastAPI()
    spec = _worker_spec(
        name="callback_worker",
        task_name="callback-worker-task",
        category="recurring-scheduler",
        phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        strategy=WorkerStrategy.CALLBACK_ONLY,
        factory=None,
        shutdown_callback_factory=lambda _context: _noop_callback,
    )
    session = WorkerLifecycleSession(app=app, graph=_worker_graph(spec))

    session.register_handle(
        spec,
        ManagedWorker(
            name="callback_worker",
            task=None,
            stop_event=None,
            shutdown_callback=_noop_callback,
            timeout_sec=4.0,
            category="recurring-scheduler",
            shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        ),
    )
    session.publish_inventory()

    assert app.state._tldw_shutdown_worker_inventory == [
        {
            "name": "callback_worker",
            "task_name": "callback-worker-task",
            "has_stop_event": False,
            "timeout_sec": 4.0,
            "category": "recurring-scheduler",
            "shutdown_phase": "background_worker_shutdown",
        }
    ]
    assert app.state._tldw_shutdown_job_poller_inventory == []


@pytest.mark.unit
def test_session_records_stopped_and_quiesced_names_across_phases() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_session import (
        WorkerLifecycleSession,
    )
    from tldw_Server_API.app.services.lifecycle_worker_specs import ShutdownPhase

    app = FastAPI()
    session = WorkerLifecycleSession(app=app, graph=_worker_graph())

    session.mark_stopped("job_worker", ShutdownPhase.JOB_POLLER_QUIESCE)
    session.mark_stopped("background_worker", ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)
    session.mark_stopped("post_worker", ShutdownPhase.POST_WORKER_SHUTDOWN)
    session.publish_stopped_names(ShutdownPhase.JOB_POLLER_QUIESCE)
    session.publish_stopped_names(ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)
    session.publish_stopped_names(ShutdownPhase.POST_WORKER_SHUTDOWN)

    assert session.stopped_names_by_phase == {
        ShutdownPhase.JOB_POLLER_QUIESCE: {"job_worker"},
        ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN: {"background_worker"},
        ShutdownPhase.POST_WORKER_SHUTDOWN: {"post_worker"},
    }
    assert session.stopped_or_quiesced_names == {
        "job_worker",
        "background_worker",
        "post_worker",
    }
    assert app.state._tldw_shutdown_quiesced_job_poller_names == ["job_worker"]
    assert app.state._tldw_shutdown_stopped_background_worker_names == ["background_worker"]
    assert app.state._tldw_shutdown_stopped_post_worker_names == ["post_worker"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_later_phase_lookups_exclude_already_stopped_or_quiesced_names() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_session import (
        WorkerLifecycleSession,
    )
    from tldw_Server_API.app.services.lifecycle_worker_specs import ShutdownPhase
    from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker

    app = FastAPI()
    post_stop_event = asyncio.Event()
    post_task = asyncio.create_task(_wait_for_stop(post_stop_event), name="post-task")
    post_spec = _worker_spec(
        name="shared_worker",
        task_name="post-task",
        phase=ShutdownPhase.POST_WORKER_SHUTDOWN,
    )
    session = WorkerLifecycleSession(app=app, graph=_worker_graph(post_spec))

    try:
        session.register_handle(
            post_spec,
            ManagedWorker(
                name="shared_worker",
                task=post_task,
                stop_event=post_stop_event,
                shutdown_phase=ShutdownPhase.POST_WORKER_SHUTDOWN,
            ),
        )

        session.mark_stopped("shared_worker", ShutdownPhase.JOB_POLLER_QUIESCE)

        assert session.handles_for_phase(ShutdownPhase.POST_WORKER_SHUTDOWN) == []
    finally:
        post_stop_event.set()
        await asyncio.wait_for(post_task, timeout=1)


@pytest.mark.unit
def test_session_post_worker_stopped_names_do_not_replace_compatibility_fields() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_session import (
        WorkerLifecycleSession,
    )
    from tldw_Server_API.app.services.lifecycle_worker_specs import ShutdownPhase

    app = FastAPI()
    app.state._tldw_shutdown_quiesced_job_poller_names = ["existing_job_worker"]
    app.state._tldw_shutdown_stopped_background_worker_names = ["existing_background_worker"]
    session = WorkerLifecycleSession(app=app, graph=_worker_graph())

    session.mark_stopped("post_worker", ShutdownPhase.POST_WORKER_SHUTDOWN)
    session.publish_stopped_names(ShutdownPhase.POST_WORKER_SHUTDOWN)

    assert app.state._tldw_shutdown_quiesced_job_poller_names == ["existing_job_worker"]
    assert app.state._tldw_shutdown_stopped_background_worker_names == ["existing_background_worker"]
    assert app.state._tldw_shutdown_stopped_post_worker_names == ["post_worker"]
