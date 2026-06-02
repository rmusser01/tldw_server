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


def _import_startup_auxiliary_services():
    sys.modules.pop("tldw_Server_API.app.services.startup_auxiliary_services", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_auxiliary_services")


def _context() -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=FastAPI(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _specs_by_name(startup_aux):
    return {
        spec.name: spec
        for spec in startup_aux.provide_auxiliary_worker_specs()
    }


def test_auxiliary_worker_specs_match_legacy_scheduler_contract() -> None:
    startup_aux = _import_startup_auxiliary_services()

    specs = _specs_by_name(startup_aux)

    expected = {
        "claims_alerts_task": "claims_alerts_scheduler",
        "claims_review_metrics_task": "claims_review_metrics_scheduler",
    }
    assert set(specs) == set(expected)
    for name, task_name in expected.items():
        spec = specs[name]
        assert spec.task_name == task_name
        assert spec.category == "auxiliary"
        assert spec.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert spec.timeout_sec == 5.0
        assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
        assert spec.factory is not None


@pytest.mark.asyncio
async def test_auxiliary_worker_spec_factory_starts_and_cancels_scheduler_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    scheduler_started = asyncio.Event()
    cancelled: list[str] = []

    async def _scheduler_loop() -> None:
        scheduler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("claims-alerts")
            raise

    async def _fake_start() -> asyncio.Task[None]:
        return asyncio.create_task(_scheduler_loop(), name="claims_alerts_scheduler")

    monkeypatch.setattr(startup_aux, "_start_claims_alerts_scheduler_service", _fake_start)

    spec = _specs_by_name(startup_aux)["claims_alerts_task"]
    stop_event = asyncio.Event()
    assert spec.factory is not None
    lifecycle_task = asyncio.create_task(spec.factory(_context(), stop_event))

    await asyncio.wait_for(scheduler_started.wait(), timeout=1)
    assert lifecycle_task.done() is False

    stop_event.set()
    await asyncio.wait_for(lifecycle_task, timeout=1)

    assert cancelled == ["claims-alerts"]


@pytest.mark.asyncio
async def test_start_auxiliary_services_combines_handles_and_starts_personalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    calls: list[str] = []

    async def _fake_claims_alerts(**kwargs: object) -> str:
        assert kwargs == {"worker_inventory": None}
        calls.append("claims-alerts")
        return "claims-alerts-task"

    async def _fake_claims_review(**kwargs: object) -> str:
        assert kwargs == {"worker_inventory": None}
        calls.append("claims-review")
        return "claims-review-task"

    async def _fake_usage(**kwargs: object) -> str:
        assert kwargs == {"worker_inventory": None}
        calls.append("usage")
        return "usage-task"

    async def _fake_llm_usage(**kwargs: object) -> str:
        assert kwargs == {"worker_inventory": None}
        calls.append("llm-usage")
        return "llm-usage-task"

    async def _fake_personalization(app_settings):
        calls.append("personalization")
        assert app_settings["PERSONALIZATION_ENABLED"] is True

    monkeypatch.setattr(startup_aux, "_start_claims_alerts_scheduler", _fake_claims_alerts)
    monkeypatch.setattr(startup_aux, "_start_claims_review_metrics_scheduler", _fake_claims_review)
    monkeypatch.setattr(startup_aux, "_start_usage_aggregator", _fake_usage)
    monkeypatch.setattr(startup_aux, "_start_llm_usage_aggregator", _fake_llm_usage)
    monkeypatch.setattr(startup_aux, "_start_personalization_consolidation", _fake_personalization)

    handles = await startup_aux.start_auxiliary_services(
        {"PERSONALIZATION_ENABLED": True},
    )

    assert calls == [
        "claims-alerts",
        "claims-review",
        "usage",
        "llm-usage",
        "personalization",
    ]
    assert handles.claims_alerts_task == "claims-alerts-task"
    assert handles.claims_review_metrics_task == "claims-review-task"
    assert handles.usage_task == "usage-task"
    assert handles.llm_usage_task == "llm-usage-task"


@pytest.mark.asyncio
async def test_start_auxiliary_services_passes_worker_inventory_to_usage_aggregators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    worker_inventory = object()
    seen_inventory: list[object] = []

    async def _fake_claims_alerts(**kwargs: object) -> None:
        assert kwargs == {"worker_inventory": worker_inventory}
        return None

    async def _fake_claims_review(**kwargs: object) -> None:
        assert kwargs == {"worker_inventory": worker_inventory}
        return None

    async def _fake_usage(**kwargs: object) -> str:
        seen_inventory.append(kwargs["worker_inventory"])
        return "usage-task"

    async def _fake_llm_usage(**kwargs: object) -> str:
        seen_inventory.append(kwargs["worker_inventory"])
        return "llm-usage-task"

    async def _fake_personalization(_app_settings):
        return None

    monkeypatch.setattr(startup_aux, "_start_claims_alerts_scheduler", _fake_claims_alerts)
    monkeypatch.setattr(startup_aux, "_start_claims_review_metrics_scheduler", _fake_claims_review)
    monkeypatch.setattr(startup_aux, "_start_usage_aggregator", _fake_usage)
    monkeypatch.setattr(startup_aux, "_start_llm_usage_aggregator", _fake_llm_usage)
    monkeypatch.setattr(startup_aux, "_start_personalization_consolidation", _fake_personalization)

    handles = await startup_aux.start_auxiliary_services(
        {},
        worker_inventory=worker_inventory,
    )

    assert seen_inventory == [worker_inventory, worker_inventory]
    assert handles.usage_task == "usage-task"
    assert handles.llm_usage_task == "llm-usage-task"


@pytest.mark.asyncio
async def test_start_auxiliary_services_registers_claims_schedulers_with_worker_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    from tldw_Server_API.app.services.lifecycle_workers import (
        ShutdownPhase,
        WorkerRegistry,
    )

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    created_tasks: list[asyncio.Task[None]] = []

    async def _wait_forever() -> None:
        await asyncio.Event().wait()

    def _make_task(name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(_wait_forever(), name=name)
        created_tasks.append(task)
        return task

    async def _fake_claims_alerts_service() -> asyncio.Task[None]:
        return _make_task("claims_alerts_scheduler")

    async def _fake_claims_review_service() -> asyncio.Task[None]:
        return _make_task("claims_review_metrics_scheduler")

    async def _fake_usage(**kwargs: object) -> None:
        assert kwargs == {"worker_inventory": worker_inventory}

    async def _fake_llm_usage(**kwargs: object) -> None:
        assert kwargs == {"worker_inventory": worker_inventory}

    async def _fake_personalization(_app_settings) -> None:
        return None

    monkeypatch.setattr(
        startup_aux,
        "_start_claims_alerts_scheduler_service",
        _fake_claims_alerts_service,
    )
    monkeypatch.setattr(
        startup_aux,
        "_start_claims_review_metrics_scheduler_service",
        _fake_claims_review_service,
    )
    monkeypatch.setattr(startup_aux, "_start_usage_aggregator", _fake_usage)
    monkeypatch.setattr(startup_aux, "_start_llm_usage_aggregator", _fake_llm_usage)
    monkeypatch.setattr(startup_aux, "_start_personalization_consolidation", _fake_personalization)

    try:
        handles = await startup_aux.start_auxiliary_services(
            {},
            worker_inventory=worker_inventory,
        )

        assert handles.claims_alerts_task is created_tasks[0]
        assert handles.claims_review_metrics_task is created_tasks[1]
        assert [handle.name for handle in worker_inventory.handles] == [
            "claims_alerts_task",
            "claims_review_metrics_task",
        ]
        assert [handle.task for handle in worker_inventory.handles] == created_tasks
        assert [handle.stop_event for handle in worker_inventory.handles] == [None, None]
        assert [handle.category for handle in worker_inventory.handles] == [
            "auxiliary",
            "auxiliary",
        ]
        assert [
            handle.shutdown_phase for handle in worker_inventory.handles
        ] == [
            ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == []
    finally:
        for task in created_tasks:
            task.cancel()
        await asyncio.gather(*created_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_start_auxiliary_services_registers_usage_aggregators_with_worker_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    from tldw_Server_API.app.services import llm_usage_aggregator, usage_aggregator
    from tldw_Server_API.app.services.lifecycle_workers import (
        ShutdownPhase,
        WorkerRegistry,
    )

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)

    class _Settings:
        USAGE_LOG_ENABLED = True
        USAGE_AGGREGATOR_INTERVAL_MINUTES = 60
        LLM_USAGE_AGGREGATOR_ENABLED = True
        LLM_USAGE_AGGREGATOR_INTERVAL_MINUTES = 60

    async def _noop_aggregate(*_args: object, **_kwargs: object) -> None:
        return None

    async def _fake_claims_alerts(**kwargs: object) -> None:
        assert kwargs == {"worker_inventory": worker_inventory}
        return None

    async def _fake_claims_review(**kwargs: object) -> None:
        assert kwargs == {"worker_inventory": worker_inventory}
        return None

    async def _fake_personalization(_app_settings: object) -> None:
        return None

    monkeypatch.setattr(usage_aggregator, "get_settings", lambda: _Settings())
    monkeypatch.setattr(usage_aggregator, "aggregate_usage_daily", _noop_aggregate)
    monkeypatch.setattr(llm_usage_aggregator, "get_settings", lambda: _Settings())
    monkeypatch.setattr(llm_usage_aggregator, "aggregate_llm_usage_daily", _noop_aggregate)
    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda _key: False)
    monkeypatch.setattr(startup_aux, "_start_claims_alerts_scheduler", _fake_claims_alerts)
    monkeypatch.setattr(startup_aux, "_start_claims_review_metrics_scheduler", _fake_claims_review)
    monkeypatch.setattr(startup_aux, "_start_personalization_consolidation", _fake_personalization)

    handles = await startup_aux.start_auxiliary_services(
        {},
        worker_inventory=worker_inventory,
    )

    try:
        await asyncio.sleep(0)

        assert handles.usage_task is not None
        assert handles.llm_usage_task is not None
        assert [handle.name for handle in worker_inventory.handles] == [
            "usage_aggregator",
            "llm_usage_aggregator",
        ]
        assert [handle.task for handle in worker_inventory.handles] == [
            handles.usage_task,
            handles.llm_usage_task,
        ]
        assert [handle.stop_event is not None for handle in worker_inventory.handles] == [
            True,
            True,
        ]
        assert [handle.category for handle in worker_inventory.handles] == [
            "usage",
            "usage",
        ]
        assert [
            handle.shutdown_phase for handle in worker_inventory.handles
        ] == [
            ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == []
    finally:
        tasks = [
            handle.task
            for handle in worker_inventory.handles
            if isinstance(handle.task, asyncio.Task)
        ]
        for handle in worker_inventory.handles:
            if handle.stop_event is not None:
                handle.stop_event.set()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)


@pytest.mark.asyncio
async def test_register_auxiliary_task_preserves_registration_error_when_rollback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    task = object()

    class _FailingInventory:
        def register(self, worker: object) -> None:
            raise AttributeError("registration failed")

    async def _failing_cancel(_task: object) -> None:
        raise LookupError("rollback failed")

    monkeypatch.setattr(startup_aux, "_cancel_unregistered_task", _failing_cancel)

    with pytest.raises(AttributeError, match="registration failed"):
        await startup_aux._register_auxiliary_task(
            worker_inventory=_FailingInventory(),
            task=task,
            worker_name="claims_alerts_task",
        )


@pytest.mark.asyncio
async def test_start_auxiliary_services_rolls_back_claims_task_when_inventory_register_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    from tldw_Server_API.app.services.lifecycle_workers import WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    created_tasks: list[asyncio.Task[None]] = []

    async def _wait_forever() -> None:
        await asyncio.Event().wait()

    def _make_task(name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(_wait_forever(), name=name)
        created_tasks.append(task)
        return task

    async def _fake_claims_alerts_service() -> asyncio.Task[None]:
        return _make_task("claims_alerts_scheduler")

    def _failing_register(worker: object) -> None:
        worker_inventory.handles.append(worker)
        raise LookupError("registration failed")

    monkeypatch.setattr(
        startup_aux,
        "_start_claims_alerts_scheduler_service",
        _fake_claims_alerts_service,
    )
    monkeypatch.setattr(worker_inventory, "register", _failing_register)

    try:
        with pytest.raises(LookupError, match="registration failed"):
            await startup_aux.start_auxiliary_services(
                {},
                worker_inventory=worker_inventory,
            )

        assert len(created_tasks) == 1
        assert created_tasks[0].cancelled()
        assert worker_inventory.handles == []
    finally:
        for task in created_tasks:
            task.cancel()
        await asyncio.gather(*created_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_unregistered_task_swallows_task_exception_during_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    debug_messages: list[str] = []

    async def _raises_on_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            raise LookupError("cleanup failed") from exc

    monkeypatch.setattr(
        startup_aux.logger,
        "debug",
        lambda message, *args: debug_messages.append(message.format(*args) if args else message),
    )

    task = asyncio.create_task(_raises_on_cancel(), name="claims_alerts_scheduler")
    await asyncio.sleep(0)

    await startup_aux._cancel_unregistered_task(task, timeout=0.25)

    assert task.done()
    assert debug_messages == ["Auxiliary scheduler raised during startup rollback: cleanup failed"]


@pytest.mark.asyncio
async def test_start_usage_aggregator_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()

    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: True)

    task = await startup_aux._start_usage_aggregator()

    assert task is None


@pytest.mark.asyncio
async def test_start_llm_usage_aggregator_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()

    async def _fake_start(**kwargs: object) -> str:
        assert kwargs == {"worker_inventory": None}
        return "llm-usage-task"

    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(startup_aux, "_start_llm_usage_aggregator_service", _fake_start)

    task = await startup_aux._start_llm_usage_aggregator()

    assert task == "llm-usage-task"


@pytest.mark.asyncio
async def test_start_personalization_consolidation_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()

    monkeypatch.setattr(startup_aux, "_legacy_get", lambda key, default: False)
    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: False)

    started = []

    class _FakeService:
        async def start(self) -> None:
            started.append("start")

    monkeypatch.setattr(startup_aux, "_get_consolidation_service", lambda: _FakeService())

    await startup_aux._start_personalization_consolidation(
        {"PERSONALIZATION_ENABLED": True},
    )

    assert started == []


@pytest.mark.asyncio
async def test_start_personalization_consolidation_starts_service_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    started = []

    class _FakeService:
        async def start(self) -> None:
            started.append("start")

    monkeypatch.setattr(startup_aux, "_legacy_get", lambda key, default: default)
    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(startup_aux, "_get_consolidation_service", lambda: _FakeService())

    await startup_aux._start_personalization_consolidation(
        {"PERSONALIZATION_ENABLED": True},
    )

    assert started == ["start"]
