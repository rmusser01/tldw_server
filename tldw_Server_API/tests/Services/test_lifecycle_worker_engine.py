import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerFailurePolicy,
    WorkerLifecycleContext,
    WorkerSpec,
    WorkerSpecValidationError,
    WorkerStrategy,
)


async def _noop_callback() -> None:
    return None


def _context(app: FastAPI | None = None) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=app or FastAPI(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _stop_event_spec(**overrides: Any) -> WorkerSpec:
    values: dict[str, Any] = {
        "name": "worker_a",
        "task_name": "worker-a-task",
        "category": "jobs",
        "phase": ShutdownPhase.JOB_POLLER_QUIESCE,
        "factory": lambda _context, stop_event: stop_event.wait(),
    }
    values.update(overrides)
    return WorkerSpec(**values)


def _callback_spec(
    *,
    name: str,
    callback: Callable[[], Awaitable[None]],
    **overrides: Any,
) -> WorkerSpec:
    values: dict[str, Any] = {
        "name": name,
        "task_name": f"{name}-task",
        "category": "background",
        "phase": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        "strategy": WorkerStrategy.CALLBACK_ONLY,
        "factory": None,
        "shutdown_callback_factory": lambda _context: callback,
    }
    values.update(overrides)
    return WorkerSpec(**values)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_startup_order_follows_dependencies() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started: list[str] = []

    def factory(name: str) -> Callable[[WorkerLifecycleContext, asyncio.Event], Awaitable[None]]:
        async def wait_for_stop(stop_event: asyncio.Event) -> None:
            await stop_event.wait()

        def create(_context: WorkerLifecycleContext, stop_event: asyncio.Event) -> Awaitable[None]:
            started.append(name)
            return wait_for_stop(stop_event)

        return create

    parent = _stop_event_spec(name="parent", task_name="parent-task", factory=factory("parent"))
    child = _stop_event_spec(
        name="child",
        task_name="child-task",
        depends_on=("parent",),
        factory=factory("child"),
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [child, parent])

    try:
        assert started == ["parent", "child"]
        assert list(session.handles_by_name) == ["parent", "child"]
        assert session.handles_by_name["parent"].task is not None
        assert session.handles_by_name["parent"].task.get_name() == "parent-task"
        assert session.handles_by_name["parent"].stop_event is not None
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_disabled_predicates_skip_workers_deterministically() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    predicate_calls: list[str] = []
    factory_calls: list[str] = []

    def enabled(name: str, value: bool) -> Callable[[WorkerLifecycleContext], bool]:
        def evaluate(_context: WorkerLifecycleContext) -> bool:
            predicate_calls.append(name)
            return value

        return evaluate

    def factory(name: str) -> Callable[[WorkerLifecycleContext, asyncio.Event], Awaitable[None]]:
        def create(_context: WorkerLifecycleContext, stop_event: asyncio.Event) -> Awaitable[None]:
            factory_calls.append(name)
            return stop_event.wait()

        return create

    disabled = _stop_event_spec(
        name="disabled_worker",
        task_name="disabled-task",
        enabled=enabled("disabled_worker", False),
        factory=factory("disabled_worker"),
    )
    enabled_worker = _stop_event_spec(
        name="enabled_worker",
        task_name="enabled-task",
        enabled=enabled("enabled_worker", True),
        factory=factory("enabled_worker"),
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [disabled, enabled_worker])

    try:
        assert predicate_calls == ["disabled_worker", "enabled_worker"]
        assert factory_calls == ["enabled_worker"]
        assert session.disabled_names == {"disabled_worker"}
        assert session.enabled_names == {"enabled_worker"}
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_rejects_non_boolean_enabled_predicate_results() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    skipped = _stop_event_spec(
        name="skipped_non_bool",
        task_name="skipped-non-bool-task",
        enabled=lambda _context: "false",
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    healthy = _stop_event_spec(
        name="healthy",
        task_name="healthy-task",
        enabled=lambda _context: True,
    )

    session = await LifecycleWorkerEngine().start(_context(), [skipped, healthy])
    try:
        assert session.enabled_names == {"healthy"}
        assert "must return bool" in session.startup_failures["skipped_non_bool"]
    finally:
        await LifecycleWorkerEngine().stop_phase(
            session,
            ShutdownPhase.JOB_POLLER_QUIESCE,
        )

    aborting = _stop_event_spec(
        name="aborting_non_bool",
        task_name="aborting-non-bool-task",
        enabled=lambda _context: 1,
        failure_policy=WorkerFailurePolicy.ABORT,
    )

    with pytest.raises(TypeError, match="must return bool"):
        await LifecycleWorkerEngine().start(_context(), [aborting])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_enabled_worker_depending_on_disabled_worker_fails_after_predicates() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    predicate_calls: list[str] = []

    def enabled(name: str, value: bool) -> Callable[[WorkerLifecycleContext], bool]:
        def evaluate(_context: WorkerLifecycleContext) -> bool:
            predicate_calls.append(name)
            return value

        return evaluate

    dependency = _stop_event_spec(
        name="dependency",
        enabled=enabled("dependency", False),
    )
    child = _stop_event_spec(
        name="child",
        enabled=enabled("child", True),
        depends_on=("dependency",),
    )

    with pytest.raises(WorkerSpecValidationError, match="child.*dependency"):
        await LifecycleWorkerEngine().start(_context(), [child, dependency])

    assert predicate_calls == ["child", "dependency"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_startup_failures_follow_failure_policy() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started: list[str] = []

    def failing_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        raise RuntimeError("factory unavailable")

    def successful_factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        started.append("healthy")
        return stop_event.wait()

    skipped = _stop_event_spec(
        name="skipped_failure",
        task_name="skipped-failure-task",
        factory=failing_factory,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    healthy = _stop_event_spec(
        name="healthy",
        task_name="healthy-task",
        factory=successful_factory,
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [skipped, healthy])

    try:
        assert "factory unavailable" in session.startup_failures["skipped_failure"]
        assert started == ["healthy"]
        assert session.enabled_names == {"healthy"}
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)

    aborting = _stop_event_spec(
        name="aborting_failure",
        task_name="aborting-failure-task",
        factory=failing_factory,
        failure_policy=WorkerFailurePolicy.ABORT,
    )
    with pytest.raises(RuntimeError, match="factory unavailable"):
        await engine.start(_context(), [aborting])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_rejects_callback_only_non_callable_shutdown_hook() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    invalid_callback_spec = _callback_spec(
        name="invalid_callback",
        callback=_noop_callback,
        shutdown_callback_factory=lambda _context: None,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    healthy = _stop_event_spec(name="healthy")

    session = await LifecycleWorkerEngine().start(_context(), [invalid_callback_spec, healthy])
    try:
        assert session.enabled_names == {"healthy"}
        assert "non-callable result" in session.startup_failures["invalid_callback"]
    finally:
        await LifecycleWorkerEngine().stop_phase(
            session,
            ShutdownPhase.JOB_POLLER_QUIESCE,
        )

    aborting = _callback_spec(
        name="aborting_callback",
        callback=_noop_callback,
        shutdown_callback_factory=lambda _context: "not-callable",
        failure_policy=WorkerFailurePolicy.ABORT,
    )

    with pytest.raises(RuntimeError, match="non-callable result"):
        await LifecycleWorkerEngine().start(_context(), [aborting])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_immediate_coroutine_startup_failure_skips_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started: list[str] = []

    async def raises_before_first_await() -> None:
        raise RuntimeError("coroutine unavailable")

    def failing_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        return raises_before_first_await()

    def successful_factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        started.append("healthy")
        return stop_event.wait()

    failing_worker = _stop_event_spec(
        name="immediate_failure",
        task_name="immediate-failure-task",
        factory=failing_factory,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    healthy_worker = _stop_event_spec(
        name="healthy",
        task_name="healthy-task",
        factory=successful_factory,
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [failing_worker, healthy_worker])

    try:
        assert "coroutine unavailable" in session.startup_failures["immediate_failure"]
        assert set(session.handles_by_name) == {"healthy"}
        assert session.enabled_names == {"healthy"}
        assert started == ["healthy"]
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_immediate_coroutine_startup_failure_aborts_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    async def raises_before_first_await() -> None:
        raise RuntimeError("coroutine unavailable")

    def failing_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        return raises_before_first_await()

    failing_worker = _stop_event_spec(
        name="immediate_failure",
        task_name="immediate-failure-task",
        factory=failing_factory,
        failure_policy=WorkerFailurePolicy.ABORT,
    )

    with pytest.raises(RuntimeError, match="coroutine unavailable"):
        await LifecycleWorkerEngine().start(_context(), [failing_worker])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_immediate_cancelled_startup_task_skips_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started: list[str] = []

    async def cancels_before_first_await() -> None:
        raise asyncio.CancelledError()

    def cancelling_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        return cancels_before_first_await()

    def successful_factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        started.append("healthy")
        return stop_event.wait()

    cancelled_worker = _stop_event_spec(
        name="cancelled_worker",
        task_name="cancelled-worker-task",
        factory=cancelling_factory,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    healthy_worker = _stop_event_spec(
        name="healthy",
        task_name="healthy-task",
        factory=successful_factory,
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [cancelled_worker, healthy_worker])

    try:
        assert "cancelled_worker" in session.startup_failures
        assert "cancelled" in session.startup_failures["cancelled_worker"]
        assert set(session.handles_by_name) == {"healthy"}
        assert session.enabled_names == {"healthy"}
        assert started == ["healthy"]
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_immediate_cancelled_startup_task_aborts_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    async def cancels_before_first_await() -> None:
        raise asyncio.CancelledError()

    def cancelling_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        return cancels_before_first_await()

    cancelled_worker = _stop_event_spec(
        name="cancelled_worker",
        task_name="cancelled-worker-task",
        factory=cancelling_factory,
        failure_policy=WorkerFailurePolicy.ABORT,
    )

    with pytest.raises(RuntimeError, match="cancelled_worker.*cancelled"):
        await LifecycleWorkerEngine().start(_context(), [cancelled_worker])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_skips_worker_when_started_dependency_failed() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started: list[str] = []

    async def raises_before_first_await() -> None:
        raise RuntimeError("dependency unavailable")

    def failing_dependency_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        return raises_before_first_await()

    def child_factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        started.append("child")
        return stop_event.wait()

    dependency = _stop_event_spec(
        name="dependency",
        task_name="dependency-task",
        factory=failing_dependency_factory,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    child = _stop_event_spec(
        name="child",
        task_name="child-task",
        depends_on=("dependency",),
        factory=child_factory,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [child, dependency])

    assert started == []
    assert session.handles_by_name == {}
    assert "dependency unavailable" in session.startup_failures["dependency"]
    assert "dependency" in session.startup_failures["child"]
    assert session.enabled_names == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_aborts_worker_when_started_dependency_failed() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started: list[str] = []

    async def raises_before_first_await() -> None:
        raise RuntimeError("dependency unavailable")

    def failing_dependency_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        return raises_before_first_await()

    def child_factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        started.append("child")
        return stop_event.wait()

    dependency = _stop_event_spec(
        name="dependency",
        task_name="dependency-task",
        factory=failing_dependency_factory,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    child = _stop_event_spec(
        name="child",
        task_name="child-task",
        depends_on=("dependency",),
        factory=child_factory,
        failure_policy=WorkerFailurePolicy.ABORT,
    )

    with pytest.raises(RuntimeError, match="dependency.*did not start"):
        await LifecycleWorkerEngine().start(_context(), [child, dependency])
    assert started == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_startup_abort_stops_already_started_workers() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started_stop_events: list[asyncio.Event] = []
    started_tasks: list[asyncio.Task[Any]] = []

    async def wait_for_stop(stop_event: asyncio.Event) -> None:
        started_tasks.append(asyncio.current_task())
        await stop_event.wait()

    def successful_factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        started_stop_events.append(stop_event)
        return wait_for_stop(stop_event)

    def aborting_factory(
        _context: WorkerLifecycleContext,
        _stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        raise RuntimeError("startup abort")

    started_worker = _stop_event_spec(
        name="started_worker",
        task_name="started-worker-task",
        factory=successful_factory,
    )
    aborting_worker = _stop_event_spec(
        name="aborting_worker",
        task_name="aborting-worker-task",
        factory=aborting_factory,
        failure_policy=WorkerFailurePolicy.ABORT,
    )

    with pytest.raises(RuntimeError, match="startup abort"):
        await LifecycleWorkerEngine().start(_context(), [started_worker, aborting_worker])

    assert len(started_stop_events) == 1
    assert started_stop_events[0].is_set()
    assert len(started_tasks) == 1
    assert started_tasks[0].done()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_inventory_publishes_diagnostic_names_without_rekeying_session() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    spec = _stop_event_spec(
        name="internal_job_worker",
        diagnostic_name="legacy_job_worker",
        task_name="legacy-job-task",
        phase=ShutdownPhase.JOB_POLLER_QUIESCE,
    )
    engine = LifecycleWorkerEngine()
    app = FastAPI()
    session = await engine.start(_context(app), [spec])

    try:
        assert set(session.handles_by_name) == {"internal_job_worker"}
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "legacy_job_worker",
                "task_name": "legacy-job-task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
                "category": "jobs",
                "shutdown_phase": "job_poller_quiesce",
            }
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "legacy_job_worker",
                "task_name": "legacy-job-task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
            }
        ]
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_stopped_name_diagnostics_publish_diagnostic_names() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    async def stop_worker() -> None:
        return None

    spec = _callback_spec(
        name="internal_background_worker",
        diagnostic_name="legacy_background_worker",
        callback=stop_worker,
        phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )
    engine = LifecycleWorkerEngine()
    app = FastAPI()
    session = await engine.start(_context(app), [spec])

    await engine.stop_phase(session, ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)

    assert session.stopped_names_by_phase[ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN] == {
        "internal_background_worker"
    }
    assert session.stopped_or_quiesced_names == {"internal_background_worker"}
    assert app.state._tldw_shutdown_stopped_background_worker_names == [
        "legacy_background_worker"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_enabled_predicate_failures_follow_failure_policy() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    started: list[str] = []

    def raising_enabled(_context: WorkerLifecycleContext) -> bool:
        raise RuntimeError("predicate unavailable")

    def successful_factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[None]:
        started.append("healthy")
        return stop_event.wait()

    skipped = _stop_event_spec(
        name="predicate_skip",
        task_name="predicate-skip-task",
        enabled=raising_enabled,
        failure_policy=WorkerFailurePolicy.SKIP,
    )
    healthy = _stop_event_spec(
        name="healthy",
        task_name="healthy-task",
        factory=successful_factory,
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [skipped, healthy])

    try:
        assert "predicate unavailable" in session.startup_failures["predicate_skip"]
        assert started == ["healthy"]
        assert session.enabled_names == {"healthy"}
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)

    aborting = _stop_event_spec(
        name="predicate_abort",
        task_name="predicate-abort-task",
        enabled=raising_enabled,
        failure_policy=WorkerFailurePolicy.ABORT,
    )
    with pytest.raises(RuntimeError, match="predicate unavailable"):
        await engine.start(_context(), [aborting])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_shutdown_reverses_dependencies_inside_phase() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    stopped: list[str] = []

    async def stop_parent() -> None:
        stopped.append("parent")

    async def stop_child() -> None:
        stopped.append("child")

    parent = _callback_spec(name="parent", callback=stop_parent)
    child = _callback_spec(name="child", callback=stop_child, depends_on=("parent",))
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [parent, child])

    await engine.stop_phase(session, ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)

    assert stopped == ["child", "parent"]
    assert session.stopped_names_by_phase[ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN] == {
        "child",
        "parent",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_shutdown_stops_independent_workers_concurrently() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    events: list[str] = []

    def callback(name: str) -> Callable[[], Awaitable[None]]:
        async def stop() -> None:
            events.append(f"{name}:start")
            await asyncio.sleep(0)
            events.append(f"{name}:finish")

        return stop

    worker_a = _callback_spec(name="worker_a", callback=callback("worker_a"))
    worker_b = _callback_spec(name="worker_b", callback=callback("worker_b"))
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [worker_a, worker_b])

    await engine.stop_phase(session, ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)

    assert events[:2] == ["worker_a:start", "worker_b:start"]
    assert sorted(events[2:]) == ["worker_a:finish", "worker_b:finish"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_shutdown_timeout_or_failure_does_not_block_remaining_workers() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    stopped: list[str] = []

    async def raises() -> None:
        raise RuntimeError("shutdown failed")

    async def times_out() -> None:
        await asyncio.sleep(1)

    async def succeeds() -> None:
        stopped.append("healthy")

    failing_worker = _callback_spec(
        name="failing_worker",
        callback=raises,
        timeout_sec=0.05,
    )
    slow_worker = _callback_spec(
        name="slow_worker",
        callback=times_out,
        timeout_sec=0.01,
    )
    healthy_worker = _callback_spec(name="healthy_worker", callback=succeeds)
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [failing_worker, slow_worker, healthy_worker])

    await engine.stop_phase(session, ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)

    assert stopped == ["healthy"]
    assert session.stopped_names_by_phase[ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN] == {
        "healthy_worker"
    }
    assert session.stopped_or_quiesced_names == {"healthy_worker"}
    assert session.app.state._tldw_shutdown_stopped_background_worker_names == ["healthy_worker"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_skips_worker_already_quiesced_in_earlier_phase() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    stop_calls: list[str] = []

    async def stop_shared() -> None:
        stop_calls.append("shared_worker")

    shared_worker = _callback_spec(
        name="shared_worker",
        callback=stop_shared,
        phase=ShutdownPhase.POST_WORKER_SHUTDOWN,
    )
    engine = LifecycleWorkerEngine()
    session = await engine.start(_context(), [shared_worker])
    session.mark_stopped("shared_worker", ShutdownPhase.JOB_POLLER_QUIESCE)

    await engine.stop_phase(session, ShutdownPhase.POST_WORKER_SHUTDOWN)

    assert stop_calls == []
    assert not session.stopped_names_by_phase.get(ShutdownPhase.POST_WORKER_SHUTDOWN)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_job_poller_phase_publishes_quiesced_names() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    async def stop_worker() -> None:
        return None

    worker = _callback_spec(
        name="job_poller",
        callback=stop_worker,
        phase=ShutdownPhase.JOB_POLLER_QUIESCE,
    )
    engine = LifecycleWorkerEngine()
    app = FastAPI()
    session = await engine.start(_context(app), [worker])

    await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)

    assert app.state._tldw_shutdown_quiesced_job_poller_names == ["job_poller"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_background_phase_publishes_stopped_names() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_engine import (
        LifecycleWorkerEngine,
    )

    async def stop_worker() -> None:
        return None

    worker = _callback_spec(
        name="background_worker",
        callback=stop_worker,
        phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )
    engine = LifecycleWorkerEngine()
    app = FastAPI()
    session = await engine.start(_context(app), [worker])

    await engine.stop_phase(session, ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)

    assert app.state._tldw_shutdown_stopped_background_worker_names == ["background_worker"]
