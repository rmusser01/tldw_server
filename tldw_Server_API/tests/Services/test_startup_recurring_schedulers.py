from __future__ import annotations

import asyncio
import importlib
import sys

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_startup_recurring_schedulers():
    sys.modules.pop("tldw_Server_API.app.services.startup_recurring_schedulers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_recurring_schedulers")


async def _wait_forever() -> None:
    await asyncio.Event().wait()


def _context(*, test_mode: bool = False) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=FastAPI(),
        settings={},
        test_mode=test_mode,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _specs_by_name(startup_recurring):
    return {
        spec.name: spec
        for spec in startup_recurring.provide_recurring_scheduler_worker_specs()
    }


def test_recurring_scheduler_worker_specs_match_ownership_matrix() -> None:
    startup_recurring = _import_startup_recurring_schedulers()

    specs = _specs_by_name(startup_recurring)

    expected_task_names = {
        "authnz_scheduler": "authnz_scheduler",
        "workflows_sched_task": "workflows_recurring_scheduler",
        "reading_digest_sched_task": "reading_digest_scheduler",
        "admin_backup_sched_task": "admin_backup_scheduler",
        "companion_reflection_sched_task": "companion_reflection_scheduler",
        "reminders_sched_task": "reminders_scheduler",
        "connectors_sync_sched_task": "connectors_sync_scheduler",
    }
    assert set(specs) == set(expected_task_names)
    for name, task_name in expected_task_names.items():
        spec = specs[name]
        assert spec.task_name == task_name
        assert spec.category == "recurring-scheduler"
        assert spec.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert spec.timeout_sec == 5.0
        assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
        assert spec.factory is not None


def test_recurring_scheduler_worker_spec_predicates_match_legacy_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    monkeypatch.setattr(startup_recurring, "_env_flag_enabled", lambda key: key != "DISABLE_AUTHNZ_SCHEDULER")
    monkeypatch.setattr(
        startup_recurring,
        "_env_flag",
        lambda key, default: key == "READING_DIGEST_SCHEDULER_ENABLED",
    )

    specs = _specs_by_name(startup_recurring)

    assert specs["authnz_scheduler"].enabled(_context()) is True
    assert specs["reading_digest_sched_task"].enabled(_context(test_mode=False)) is True


@pytest.mark.asyncio
async def test_authnz_scheduler_worker_spec_factory_starts_and_stops_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    calls: list[str] = []

    async def _fake_start() -> None:
        calls.append("start")

    async def _fake_stop() -> None:
        calls.append("stop")

    monkeypatch.setattr(startup_recurring, "_start_authnz_scheduler_service", _fake_start)
    monkeypatch.setattr(startup_recurring, "_stop_authnz_scheduler_service", _fake_stop)

    spec = _specs_by_name(startup_recurring)["authnz_scheduler"]
    stop_event = asyncio.Event()
    assert spec.factory is not None
    lifecycle_task = asyncio.create_task(spec.factory(_context(), stop_event))

    await asyncio.sleep(0)
    assert calls == ["start"]
    assert lifecycle_task.done() is False

    stop_event.set()
    await asyncio.wait_for(lifecycle_task, timeout=1)
    assert calls == ["start", "stop"]


@pytest.mark.asyncio
async def test_recurring_scheduler_worker_spec_factory_starts_and_stops_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    scheduler_started = asyncio.Event()
    stopped: list[str] = []

    async def _scheduler_loop() -> None:
        scheduler_started.set()
        await asyncio.Event().wait()

    async def _fake_start() -> asyncio.Task[None]:
        return asyncio.create_task(_scheduler_loop(), name="workflows_recurring_scheduler")

    async def _fake_stop(task: asyncio.Task[None]) -> None:
        stopped.append(task.get_name())
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    monkeypatch.setattr(startup_recurring, "_start_workflows_scheduler_service", _fake_start)
    monkeypatch.setattr(startup_recurring, "_stop_workflows_scheduler_service", _fake_stop)

    spec = _specs_by_name(startup_recurring)["workflows_sched_task"]
    stop_event = asyncio.Event()
    assert spec.factory is not None
    lifecycle_task = asyncio.create_task(spec.factory(_context(), stop_event))

    await asyncio.wait_for(scheduler_started.wait(), timeout=1)
    assert lifecycle_task.done() is False

    stop_event.set()
    await asyncio.wait_for(lifecycle_task, timeout=1)
    assert stopped == ["workflows_recurring_scheduler"]


@pytest.mark.asyncio
async def test_start_recurring_schedulers_combines_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    calls: list[str] = []

    async def _fake_authnz(*, worker_inventory: object | None = None) -> bool:
        assert worker_inventory is None
        calls.append("authnz")
        return True

    async def _fake_workflows():
        calls.append("workflows")
        return "workflows-task"

    async def _fake_reading_digest(*, test_mode: bool):
        calls.append(f"reading-digest:{test_mode}")
        return "reading-digest-task"

    async def _fake_admin_backup():
        calls.append("admin-backup")
        return "admin-backup-task"

    async def _fake_companion_reflection():
        calls.append("companion-reflection")
        return "companion-reflection-task"

    async def _fake_reminders():
        calls.append("reminders")
        return "reminders-task"

    async def _fake_connectors_sync():
        calls.append("connectors-sync")
        return "connectors-sync-task"

    monkeypatch.setattr(startup_recurring, "_start_authnz_scheduler", _fake_authnz)
    monkeypatch.setattr(startup_recurring, "_start_workflows_scheduler", _fake_workflows)
    monkeypatch.setattr(startup_recurring, "_start_reading_digest_scheduler", _fake_reading_digest)
    monkeypatch.setattr(startup_recurring, "_start_admin_backup_scheduler", _fake_admin_backup)
    monkeypatch.setattr(startup_recurring, "_start_companion_reflection_scheduler", _fake_companion_reflection)
    monkeypatch.setattr(startup_recurring, "_start_reminders_scheduler", _fake_reminders)
    monkeypatch.setattr(startup_recurring, "_start_connectors_sync_scheduler", _fake_connectors_sync)

    handles = await startup_recurring.start_recurring_schedulers(test_mode=False)

    assert calls == [
        "authnz",
        "workflows",
        "reading-digest:False",
        "admin-backup",
        "companion-reflection",
        "reminders",
        "connectors-sync",
    ]
    assert handles.authnz_scheduler_started is True
    assert handles.workflows_sched_task == "workflows-task"
    assert handles.reading_digest_sched_task == "reading-digest-task"
    assert handles.admin_backup_sched_task == "admin-backup-task"
    assert handles.companion_reflection_sched_task == "companion-reflection-task"
    assert handles.reminders_sched_task == "reminders-task"
    assert handles.connectors_sync_sched_task == "connectors-sync-task"


@pytest.mark.asyncio
async def test_start_authnz_scheduler_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()

    monkeypatch.setattr(startup_recurring, "_env_flag_enabled", lambda key: True)

    started = await startup_recurring._start_authnz_scheduler()

    assert started is False


@pytest.mark.asyncio
async def test_start_reading_digest_scheduler_defaults_off_in_test_mode_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    calls: list[bool] = []

    async def _fake_start(*, enabled: bool):
        calls.append(enabled)
        return "reading-digest-task"

    monkeypatch.delenv("READING_DIGEST_SCHEDULER_ENABLED", raising=False)
    monkeypatch.setattr(startup_recurring, "_start_reading_digest_scheduler_service", _fake_start)

    task = await startup_recurring._start_reading_digest_scheduler(test_mode=True)

    assert task is None
    assert calls == []


@pytest.mark.asyncio
async def test_start_companion_reflection_scheduler_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()

    async def _fake_start(*, enabled: bool):
        assert enabled is True
        return "companion-reflection-task"

    monkeypatch.setattr(startup_recurring, "_env_flag", lambda key, default: True)
    monkeypatch.setattr(startup_recurring, "_start_companion_reflection_scheduler_service", _fake_start)

    task = await startup_recurring._start_companion_reflection_scheduler()

    assert task == "companion-reflection-task"


@pytest.mark.asyncio
async def test_start_recurring_schedulers_registers_background_inventory_with_shutdown_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    created_tasks: list[asyncio.Task[None]] = []
    stopped: list[str] = []
    service_calls: list[str] = []

    def _make_task(name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(_wait_forever(), name=name)
        created_tasks.append(task)
        return task

    async def _fake_authnz_start() -> None:
        service_calls.append("start-authnz")

    async def _fake_authnz_stop() -> None:
        stopped.append("authnz")

    async def _fake_workflows():
        return _make_task("workflows_recurring_scheduler")

    async def _fake_reading_digest(*, enabled: bool):
        assert enabled is True
        return _make_task("reading_digest_scheduler")

    async def _fake_admin_backup():
        return _make_task("admin_backup_scheduler")

    async def _fake_companion_reflection(*, enabled: bool):
        assert enabled is True
        return _make_task("companion_reflection_scheduler")

    async def _fake_reminders():
        return _make_task("reminders_scheduler")

    async def _fake_connectors_sync():
        return _make_task("connectors_sync_scheduler")

    def _fake_stop(label: str):
        async def _stop(task: asyncio.Task[None]) -> None:
            stopped.append(label)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return _stop

    monkeypatch.setattr(startup_recurring, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(startup_recurring, "_start_authnz_scheduler_service", _fake_authnz_start)
    monkeypatch.setattr(startup_recurring, "_stop_authnz_scheduler_service", _fake_authnz_stop)
    monkeypatch.setattr(startup_recurring, "_start_workflows_scheduler_service", _fake_workflows)
    monkeypatch.setattr(startup_recurring, "_start_reading_digest_scheduler_service", _fake_reading_digest)
    monkeypatch.setattr(startup_recurring, "_start_admin_backup_scheduler_service", _fake_admin_backup)
    monkeypatch.setattr(
        startup_recurring,
        "_start_companion_reflection_scheduler_service",
        _fake_companion_reflection,
    )
    monkeypatch.setattr(startup_recurring, "_start_reminders_scheduler_service", _fake_reminders)
    monkeypatch.setattr(startup_recurring, "_start_connectors_sync_scheduler_service", _fake_connectors_sync)
    monkeypatch.setattr(startup_recurring, "_env_flag", lambda key, default: True)
    monkeypatch.setattr(startup_recurring, "_stop_workflows_scheduler_service", _fake_stop("workflows"))
    monkeypatch.setattr(
        startup_recurring,
        "_stop_reading_digest_scheduler_service",
        _fake_stop("reading-digest"),
    )
    monkeypatch.setattr(startup_recurring, "_stop_admin_backup_scheduler_service", _fake_stop("admin-backup"))
    monkeypatch.setattr(
        startup_recurring,
        "_stop_companion_reflection_scheduler_service",
        _fake_stop("companion-reflection"),
    )
    monkeypatch.setattr(startup_recurring, "_stop_reminders_scheduler_service", _fake_stop("reminders"))
    monkeypatch.setattr(startup_recurring, "_stop_connectors_sync_scheduler_service", _fake_stop("connectors-sync"))

    try:
        handles = await startup_recurring.start_recurring_schedulers(
            test_mode=False,
            worker_inventory=worker_inventory,
        )

        assert handles.authnz_scheduler_started is True
        assert service_calls == ["start-authnz"]
        assert handles.workflows_sched_task in created_tasks
        assert handles.reading_digest_sched_task in created_tasks
        assert handles.admin_backup_sched_task in created_tasks
        assert handles.companion_reflection_sched_task in created_tasks
        assert handles.reminders_sched_task in created_tasks
        assert handles.connectors_sync_sched_task in created_tasks
        assert {
            handle.name: handle.shutdown_phase
            for handle in worker_inventory.handles
        } == {
            "authnz_scheduler": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "workflows_sched_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "reading_digest_sched_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "admin_backup_sched_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "companion_reflection_sched_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "reminders_sched_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "connectors_sync_sched_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        }
        assert all(handle.stop_event is None for handle in worker_inventory.handles)
        assert all(handle.shutdown_callback is not None for handle in worker_inventory.handles)
        assert {handle.category for handle in worker_inventory.handles} == {"recurring-scheduler"}
        assert app.state._tldw_shutdown_job_poller_inventory == []

        for handle in worker_inventory.handles:
            await handle.shutdown_callback()

        assert stopped == [
            "authnz",
            "workflows",
            "reading-digest",
            "admin-backup",
            "companion-reflection",
            "reminders",
            "connectors-sync",
        ]
    finally:
        for task in created_tasks:
            task.cancel()
        await asyncio.gather(*created_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_start_authnz_scheduler_registers_callback_only_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    service_calls: list[str] = []

    async def _fake_start() -> None:
        service_calls.append("start")

    async def _fake_stop() -> None:
        service_calls.append("stop")

    monkeypatch.setattr(startup_recurring, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(startup_recurring, "_start_authnz_scheduler_service", _fake_start)
    monkeypatch.setattr(startup_recurring, "_stop_authnz_scheduler_service", _fake_stop)

    started = await startup_recurring._start_authnz_scheduler(
        worker_inventory=worker_inventory,
    )

    assert started is True
    assert service_calls == ["start"]
    assert len(worker_inventory.handles) == 1
    handle = worker_inventory.handles[0]
    assert handle.name == "authnz_scheduler"
    assert handle.task is None
    assert handle.stop_event is None
    assert handle.shutdown_callback is not None
    assert handle.category == "recurring-scheduler"
    assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
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

    await handle.shutdown_callback()
    assert service_calls == ["start", "stop"]


@pytest.mark.asyncio
async def test_register_recurring_scheduler_task_warns_on_incomplete_inventory_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    from tldw_Server_API.app.services.lifecycle_workers import WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    warnings: list[tuple[object, ...]] = []

    async def _stop(task: asyncio.Task[None]) -> None:
        task.cancel()

    monkeypatch.setattr(
        startup_recurring.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    task = asyncio.create_task(_wait_forever(), name="metadata-missing-scheduler")
    try:
        await startup_recurring._register_recurring_scheduler_task(
            worker_inventory=worker_inventory,
            task=task,
            worker_name=None,
            stopper=_stop,
        )

        assert worker_inventory.handles == []
        assert any("registration skipped" in str(args[0]) for args in warnings)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_register_recurring_scheduler_task_bounds_failed_registration_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_recurring = _import_startup_recurring_schedulers()
    warnings: list[tuple[object, ...]] = []

    class _FailingInventory:
        def register(self, worker: object) -> None:
            raise RuntimeError("inventory unavailable")

    allow_cancel = False
    entered = asyncio.Event()

    async def _stubborn_task() -> None:
        nonlocal allow_cancel
        while True:
            entered.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                if allow_cancel:
                    raise
                continue

    async def _stop(task: asyncio.Task[None]) -> None:
        task.cancel()

    monkeypatch.setattr(startup_recurring, "_SCHEDULER_ROLLBACK_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(
        startup_recurring.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    task = asyncio.create_task(_stubborn_task(), name="stubborn-rollback-scheduler")
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        with pytest.raises(RuntimeError, match="inventory unavailable"):
            await asyncio.wait_for(
                startup_recurring._register_recurring_scheduler_task(
                    worker_inventory=_FailingInventory(),
                    task=task,
                    worker_name="stubborn_sched_task",
                    stopper=_stop,
                ),
                timeout=0.5,
            )

        assert any("startup rollback timed out" in str(args[0]) for args in warnings)
    finally:
        allow_cancel = True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
