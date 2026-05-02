import asyncio

import pytest
from fastapi import FastAPI


async def _wait_for_stop(stop_event: asyncio.Event) -> None:
    await stop_event.wait()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_worker_registry_register_custom_starts_and_registers_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ShutdownPhase,
        WorkerRegistry,
    )

    app = FastAPI()
    registry = WorkerRegistry(app)
    observed_stop_event: asyncio.Event | None = None

    async def _worker(stop_event: asyncio.Event) -> None:
        nonlocal observed_stop_event
        observed_stop_event = stop_event
        await stop_event.wait()

    task, stop_event = await registry.register_custom(
        name="custom_worker",
        task_name="custom-worker-task",
        coroutine_factory=_worker,
        timeout_sec=1.5,
        category="jobs",
        shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )

    try:
        await asyncio.sleep(0)

        assert observed_stop_event is stop_event
        assert task.get_name() == "custom-worker-task"
        assert registry.handles[0].name == "custom_worker"
        assert registry.handles[0].shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "custom_worker",
                "task_name": "custom-worker-task",
                "has_stop_event": True,
                "timeout_sec": 1.5,
                "category": "jobs",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publish_worker_inventory_logs_guarded_state_publication_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import lifecycle_workers
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        publish_worker_inventory,
    )

    class _FailingState:
        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError(f"{name} unavailable")

    app = type("App", (), {"state": _FailingState()})()
    stop_event = asyncio.Event()
    task = asyncio.create_task(_wait_for_stop(stop_event), name="state-guard-task")
    debug_calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        lifecycle_workers.logger,
        "debug",
        lambda *args, **kwargs: debug_calls.append(args),
    )

    try:
        publish_worker_inventory(
            app,
            [
                ManagedWorker(
                    name="guarded_worker",
                    task=task,
                    stop_event=stop_event,
                )
            ],
        )
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)

    assert any("_tldw_shutdown_worker_inventory" in str(args) for args in debug_calls)
    assert any("_tldw_shutdown_job_poller_inventory" in str(args) for args in debug_calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_worker_inventory_publishes_full_and_filtered_views() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
        publish_worker_inventory,
    )

    app = FastAPI()
    job_stop_event = asyncio.Event()
    background_stop_event = asyncio.Event()
    job_task = asyncio.create_task(_wait_for_stop(job_stop_event), name="job-task")
    background_task = asyncio.create_task(
        _wait_for_stop(background_stop_event),
        name="background-task",
    )

    try:
        publish_worker_inventory(
            app,
            [
                ManagedWorker(
                    name="job_worker",
                    task=job_task,
                    stop_event=job_stop_event,
                    timeout_sec=5.0,
                    category="jobs",
                    shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
                ),
                ManagedWorker(
                    name="background_worker",
                    task=background_task,
                    stop_event=background_stop_event,
                    timeout_sec=2.0,
                    category="jobs",
                    shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                ),
            ],
        )

        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "job_worker",
                "task_name": "job-task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
                "category": "jobs",
                "shutdown_phase": "job_poller_quiesce",
            },
            {
                "name": "background_worker",
                "task_name": "background-task",
                "has_stop_event": True,
                "timeout_sec": 2.0,
                "category": "jobs",
                "shutdown_phase": "background_worker_shutdown",
            },
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "job_worker",
                "task_name": "job-task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
            }
        ]
    finally:
        job_stop_event.set()
        background_stop_event.set()
        await asyncio.wait_for(asyncio.gather(job_task, background_task), timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_worker_inventory_publishes_callback_only_background_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
        WorkerRegistry,
    )

    app = FastAPI()
    registry = WorkerRegistry(app)

    async def _shutdown_callback() -> None:
        return None

    handle = registry.register(
        ManagedWorker(
            name="authnz_scheduler",
            task=None,
            stop_event=None,
            shutdown_callback=_shutdown_callback,
            category="recurring-scheduler",
            shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        )
    )

    assert handle.task is None
    assert app.state._tldw_shutdown_worker_inventory == [
        {
            "name": "authnz_scheduler",
            "task_name": None,
            "has_stop_event": False,
            "timeout_sec": 5.0,
            "category": "recurring-scheduler",
            "shutdown_phase": "background_worker_shutdown",
        }
    ]
    assert app.state._tldw_shutdown_job_poller_inventory == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_stop_event_worker_registers_named_task_and_stop_event() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ShutdownPhase,
        WorkerInventory,
        start_stop_event_worker,
    )

    app = FastAPI()
    inventory = WorkerInventory(app)
    observed_stop_event: asyncio.Event | None = None

    async def _worker(stop_event: asyncio.Event) -> None:
        nonlocal observed_stop_event
        observed_stop_event = stop_event
        await stop_event.wait()

    task, stop_event = await start_stop_event_worker(
        inventory,
        name="background_worker",
        task_name="stable-background-task",
        coroutine_factory=_worker,
        timeout_sec=2.5,
        category="jobs",
        shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )

    try:
        await asyncio.sleep(0)

        assert isinstance(stop_event, asyncio.Event)
        assert observed_stop_event is stop_event
        assert task.get_name() == "stable-background-task"
        assert len(inventory.handles) == 1

        handle = inventory.handles[0]
        assert handle.name == "background_worker"
        assert handle.task is task
        assert handle.stop_event is stop_event
        assert handle.timeout_sec == 2.5
        assert handle.category == "jobs"
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "background_worker",
                "task_name": "stable-background-task",
                "has_stop_event": True,
                "timeout_sec": 2.5,
                "category": "jobs",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_worker_inventory_phase_helpers_accept_raw_string_phases() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
        WorkerInventory,
    )

    app = FastAPI()
    job_stop_event = asyncio.Event()
    background_stop_event = asyncio.Event()
    replacement_stop_event = asyncio.Event()
    job_task = asyncio.create_task(_wait_for_stop(job_stop_event), name="job-task")
    background_task = asyncio.create_task(
        _wait_for_stop(background_stop_event),
        name="background-task",
    )
    replacement_task = asyncio.create_task(
        _wait_for_stop(replacement_stop_event),
        name="replacement-task",
    )

    try:
        inventory = WorkerInventory(
            app,
            [
                ManagedWorker(
                    name="job_worker",
                    task=job_task,
                    stop_event=job_stop_event,
                    shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
                ),
                ManagedWorker(
                    name="background_worker",
                    task=background_task,
                    stop_event=background_stop_event,
                    shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                ),
            ],
        )

        assert [
            handle.name
            for handle in inventory.handles_for_phase("job_poller_quiesce")
        ] == ["job_worker"]

        inventory.replace_phase(
            "job_poller_quiesce",
            [
                ManagedWorker(
                    name="replacement_job_worker",
                    task=replacement_task,
                    stop_event=replacement_stop_event,
                    shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
                )
            ],
        )

        assert [handle.name for handle in inventory.handles] == [
            "background_worker",
            "replacement_job_worker",
        ]
        assert [
            handle.name
            for handle in inventory.handles_for_phase("job_poller_quiesce")
        ] == ["replacement_job_worker"]
        assert app.state._tldw_shutdown_job_poller_inventory == [
            {
                "name": "replacement_job_worker",
                "task_name": "replacement-task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
            }
        ]
    finally:
        job_stop_event.set()
        background_stop_event.set()
        replacement_stop_event.set()
        await asyncio.wait_for(
            asyncio.gather(job_task, background_task, replacement_task),
            timeout=1,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_sets_events_and_waits_concurrently() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()

    async def _delayed_shutdown(stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        await asyncio.sleep(0.12)

    stop_a = asyncio.Event()
    stop_b = asyncio.Event()
    task_a = asyncio.create_task(_delayed_shutdown(stop_a), name="worker-a-task")
    task_b = asyncio.create_task(_delayed_shutdown(stop_b), name="worker-b-task")

    started = asyncio.get_running_loop().time()
    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="worker_a",
                task=task_a,
                stop_event=stop_a,
                timeout_sec=1.0,
            ),
            ManagedWorker(
                name="worker_b",
                task=task_b,
                stop_event=stop_b,
                timeout_sec=1.0,
            ),
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert stop_a.is_set() is True
    assert stop_b.is_set() is True
    assert elapsed < 0.2
    assert app.state._tldw_stopped_worker_names == ["worker_a", "worker_b"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_awaits_custom_shutdown_callback() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()
    shutdown_calls = 0

    async def _worker() -> None:
        await asyncio.Future()

    task = asyncio.create_task(_worker(), name="callback-task")

    async def _shutdown_callback() -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await asyncio.sleep(0)

    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="callback_worker",
                task=task,
                stop_event=None,
                shutdown_callback=_shutdown_callback,
                timeout_sec=1.0,
            )
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )

    assert shutdown_calls == 1
    assert task.cancelled() is True
    assert app.state._tldw_stopped_worker_names == ["callback_worker"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_awaits_callback_only_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()
    shutdown_calls = 0

    async def _shutdown_callback() -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1

    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="authnz_scheduler",
                task=None,
                stop_event=None,
                shutdown_callback=_shutdown_callback,
                timeout_sec=1.0,
            )
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )

    assert shutdown_calls == 1
    assert app.state._tldw_stopped_worker_names == ["authnz_scheduler"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_handles_cancelled_shutdown_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import lifecycle_workers
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()
    warnings: list[tuple[object, ...]] = []

    async def _shutdown_callback() -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        lifecycle_workers.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="cancelled_callback_worker",
                task=None,
                stop_event=None,
                shutdown_callback=_shutdown_callback,
                timeout_sec=1.0,
            )
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )

    assert any("shutdown callback was cancelled" in str(args[0]) for args in warnings)
    assert app.state._tldw_stopped_worker_names == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_bounds_custom_shutdown_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import lifecycle_workers
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()
    warnings: list[tuple[object, ...]] = []

    async def _worker() -> None:
        await asyncio.Future()

    async def _shutdown_callback() -> None:
        await asyncio.Future()

    monkeypatch.setattr(
        lifecycle_workers.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    task = asyncio.create_task(_worker(), name="hung-callback-task")
    await asyncio.sleep(0)

    try:
        await asyncio.wait_for(
            stop_registered_workers(
                app,
                [
                    ManagedWorker(
                        name="hung_callback_worker",
                        task=task,
                        stop_event=None,
                        shutdown_callback=_shutdown_callback,
                        timeout_sec=0.01,
                    )
                ],
                stopped_names_attr="_tldw_stopped_worker_names",
                log_label="test worker",
            ),
            timeout=0.5,
        )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert any("shutdown callback" in str(args[0]) for args in warnings)
    assert task.cancelled() is True
    assert app.state._tldw_stopped_worker_names == ["hung_callback_worker"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_logs_stopped_names_publication_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import lifecycle_workers
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    class _FailingState:
        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError(f"{name} unavailable")

    app = type("App", (), {"state": _FailingState()})()
    stop_event = asyncio.Event()
    task = asyncio.create_task(_wait_for_stop(stop_event), name="publish-guard-task")
    debug_calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        lifecycle_workers.logger,
        "debug",
        lambda *args, **kwargs: debug_calls.append(args),
    )

    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="guarded_worker",
                task=task,
                stop_event=stop_event,
            )
        ],
        stopped_names_attr="_tldw_guarded_stopped_names",
        log_label="test worker",
    )

    assert any("_tldw_guarded_stopped_names" in str(args) for args in debug_calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_logs_runtime_error_after_timeout_as_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import lifecycle_workers
    from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker

    app = FastAPI()
    stop_event = asyncio.Event()
    warnings: list[tuple[object, ...]] = []
    debug_calls: list[tuple[object, ...]] = []

    async def _raises_after_cancel(stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as exc:
            raise RuntimeError("cancel failure") from exc

    monkeypatch.setattr(
        lifecycle_workers.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )
    monkeypatch.setattr(
        lifecycle_workers.logger,
        "debug",
        lambda *args, **kwargs: debug_calls.append(args),
    )

    task = asyncio.create_task(_raises_after_cancel(stop_event), name="raising-task")
    await asyncio.sleep(0)

    await lifecycle_workers.stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="raising_worker",
                task=task,
                stop_event=stop_event,
                timeout_sec=0.01,
            )
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )

    assert any("raised after cancellation" in str(args[0]) for args in warnings)
    assert not any("guard triggered" in str(args[0]) for args in debug_calls)
    assert app.state._tldw_stopped_worker_names == ["raising_worker"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_workers_cancels_timeout_without_blocking_cooperative_worker() -> None:
    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        stop_registered_workers,
    )

    app = FastAPI()
    cooperative_stop_event = asyncio.Event()
    stubborn_stop_event = asyncio.Event()

    async def _cooperative_worker(stop_event: asyncio.Event) -> None:
        await stop_event.wait()

    async def _stubborn_worker(stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        await asyncio.Future()

    cooperative_task = asyncio.create_task(
        _cooperative_worker(cooperative_stop_event),
        name="cooperative-task",
    )
    stubborn_task = asyncio.create_task(
        _stubborn_worker(stubborn_stop_event),
        name="stubborn-task",
    )
    await asyncio.sleep(0)

    await stop_registered_workers(
        app,
        [
            ManagedWorker(
                name="cooperative_worker",
                task=cooperative_task,
                stop_event=cooperative_stop_event,
                timeout_sec=1.0,
            ),
            ManagedWorker(
                name="stubborn_worker",
                task=stubborn_task,
                stop_event=stubborn_stop_event,
                timeout_sec=0.01,
            ),
        ],
        stopped_names_attr="_tldw_stopped_worker_names",
        log_label="test worker",
    )

    assert cooperative_stop_event.is_set() is True
    assert stubborn_stop_event.is_set() is True
    assert cooperative_task.done() is True
    assert stubborn_task.cancelled() is True
    assert app.state._tldw_stopped_worker_names == [
        "cooperative_worker",
        "stubborn_worker",
    ]
