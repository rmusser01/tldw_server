import asyncio
import threading
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Collections import reading_digest_jobs_worker
from tldw_Server_API.app.core.File_Artifacts import jobs_worker as file_artifacts_jobs_worker
from tldw_Server_API.app.core.Personalization import companion_reflection_jobs_worker
from tldw_Server_API.app.services import audio_jobs_worker
from tldw_Server_API.app.services import audiobook_jobs_worker
from tldw_Server_API.app.services import admin_backup_jobs_worker
from tldw_Server_API.app.services import admin_byok_validation_jobs_worker
from tldw_Server_API.app.services import admin_maintenance_rotation_jobs_worker
from tldw_Server_API.app.services import connectors_worker
from tldw_Server_API.app.services import core_jobs_worker
from tldw_Server_API.app.services import jobs_crypto_rotate_service
from tldw_Server_API.app.services import jobs_integrity_service
from tldw_Server_API.app.services import jobs_metrics_service
from tldw_Server_API.app.services import lifecycle_workers
from tldw_Server_API.app.services import media_ingest_jobs_worker
from tldw_Server_API.app.services import privilege_snapshot_worker
from tldw_Server_API.app.services import reminder_jobs_worker
from tldw_Server_API.app.services import study_pack_jobs_worker
from tldw_Server_API.app.services import study_suggestions_jobs_worker
from tldw_Server_API.app.core.Evaluations import embeddings_abtest_jobs_worker
from tldw_Server_API.app.core.Evaluations import recipe_runs_jobs_worker


async def _wait_for_stop(stop_event: asyncio.Event) -> None:
    await stop_event.wait()


async def _wait_for_optional_stop_event(stop_event_arg: asyncio.Event | None = None) -> None:
    assert stop_event_arg is not None
    await stop_event_arg.wait()


_FIRST_BATCH_BACKGROUND_WORKER_NAMES = {
    "jobs_metrics_task",
    "jobs_metrics_reconcile_task",
    "jobs_crypto_rotate_task",
    "jobs_integrity_task",
}


def _clear_shutdown_worker_state(app: FastAPI) -> None:
    for attr in (
        "_tldw_shutdown_worker_inventory",
        "_tldw_shutdown_job_poller_inventory",
        "_tldw_shutdown_quiesced_job_poller_names",
        "_tldw_shutdown_stopped_background_worker_names",
    ):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def _enable_first_batch_background_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "1")
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "1")
    monkeypatch.setenv("JOBS_CRYPTO_ROTATE_SERVICE_ENABLED", "1")
    monkeypatch.setenv("JOBS_INTEGRITY_SWEEP_ENABLED", "1")


def _patch_first_batch_background_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {
        name: {
            "stop_event": None,
            "set_calls": 0,
            "completed": 0,
            "started": threading.Event(),
        }
        for name in _FIRST_BATCH_BACKGROUND_WORKER_NAMES
    }

    def _fake_worker(name: str):
        async def _run(stop_event: asyncio.Event) -> None:
            observed[name]["stop_event"] = stop_event
            original_set = stop_event.set

            def _record_set() -> None:
                observed[name]["set_calls"] += 1
                original_set()

            stop_event.set = _record_set
            observed[name]["started"].set()
            await stop_event.wait()
            observed[name]["completed"] += 1

        return _run

    monkeypatch.setattr(
        jobs_metrics_service,
        "run_jobs_metrics_gauges",
        _fake_worker("jobs_metrics_task"),
    )
    monkeypatch.setattr(
        jobs_metrics_service,
        "run_jobs_metrics_reconcile",
        _fake_worker("jobs_metrics_reconcile_task"),
    )
    monkeypatch.setattr(
        jobs_crypto_rotate_service,
        "run_jobs_crypto_rotate",
        _fake_worker("jobs_crypto_rotate_task"),
    )
    monkeypatch.setattr(
        jobs_integrity_service,
        "run_jobs_integrity_sweeper",
        _fake_worker("jobs_integrity_task"),
    )
    return observed


def _wait_until_first_batch_workers_started(observed: dict[str, dict[str, Any]]) -> None:
    for name, record in observed.items():
        assert record["started"].wait(timeout=1), f"{name} did not start"


def _completed_poller_handle(main_module: Any, name: str = "media_ingest_jobs_task") -> Any:
    task = asyncio.get_running_loop().create_future()
    task.set_result(None)
    return main_module._ManagedJobPoller(
        name=name,
        task=task,
        stop_event=asyncio.Event(),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_zero_active_processing_quiesces_job_pollers_without_lease_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    stop_event = asyncio.Event()
    task = asyncio.create_task(_wait_for_stop(stop_event))
    handles = [
        main_module._ManagedJobPoller(
            name="audiobook_jobs_task",
            task=task,
            stop_event=stop_event,
        )
    ]
    stop_calls: list[list[str]] = []
    sleep_calls: list[float] = []

    async def _fake_stop_pollers(_app: FastAPI, poller_handles: list[object]) -> None:
        stop_calls.append([handle.name for handle in poller_handles])
        for handle in poller_handles:
            if handle.stop_event is not None:
                handle.stop_event.set()
            await handle.task

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(main_module, "_stop_registered_job_pollers", _fake_stop_pollers)
    monkeypatch.setattr(main_module.asyncio, "sleep", _fake_sleep)

    await main_module._quiesce_owned_job_pollers_for_shutdown(
        app,
        handles,
        wait_for_leases_sec=30,
        count_active_processing=lambda: 0,
    )

    assert stop_calls == [["audiobook_jobs_task"]]
    assert sleep_calls == []
    segments = getattr(app.state, "_tldw_shutdown_timing_segments")
    assert [entry["segment"] for entry in segments][:2] == [
        "optional_lease_wait",
        "job_poller_quiesce",
    ]
    assert segments[0]["duration_ms"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_active_processing_preserves_bounded_lease_wait_before_quiesce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    counts = iter([2, 1, 0])
    observed_sleeps: list[float] = []
    stop_calls: list[str] = []

    async def _fake_sleep(delay: float) -> None:
        observed_sleeps.append(delay)

    async def _fake_stop_pollers(_app: FastAPI, _handles: list[object]) -> None:
        stop_calls.append("stop")

    monkeypatch.setattr(main_module.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(main_module, "_stop_registered_job_pollers", _fake_stop_pollers)

    await main_module._quiesce_owned_job_pollers_for_shutdown(
        app,
        [_completed_poller_handle(main_module)],
        wait_for_leases_sec=5,
        count_active_processing=lambda: next(counts),
    )

    assert observed_sleeps == [0.5, 0.5]
    assert stop_calls == ["stop"]
    segments = getattr(app.state, "_tldw_shutdown_timing_segments")
    assert [entry["segment"] for entry in segments][:2] == [
        "optional_lease_wait",
        "job_poller_quiesce",
    ]
    assert segments[0]["duration_ms"] >= 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_owned_job_pollers_skips_global_lease_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    observed_sleeps: list[float] = []
    stop_calls: list[str] = []

    async def _fake_sleep(delay: float) -> None:
        observed_sleeps.append(delay)

    async def _fake_stop_pollers(_app: FastAPI, _handles: list[object]) -> None:
        stop_calls.append("stop")

    monkeypatch.setattr(main_module.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(main_module, "_stop_registered_job_pollers", _fake_stop_pollers)

    await main_module._quiesce_owned_job_pollers_for_shutdown(
        app,
        [],
        wait_for_leases_sec=30,
        count_active_processing=lambda: (_ for _ in ()).throw(AssertionError("count should not be called")),
    )

    assert observed_sleeps == []
    assert stop_calls == ["stop"]
    segments = getattr(app.state, "_tldw_shutdown_timing_segments")
    assert segments[0]["segment"] == "optional_lease_wait"
    assert segments[0]["skipped"] is True
    assert segments[0]["initial_active"] == 0

@pytest.mark.asyncio
async def test_active_processing_deadline_uses_remaining_budget_before_quiesce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    observed_sleeps: list[float] = []
    stop_calls: list[str] = []
    monotonic_values = iter([100.0, 100.0, 100.1, 100.25])

    async def _fake_sleep(delay: float) -> None:
        observed_sleeps.append(delay)

    async def _fake_stop_pollers(_app: FastAPI, _handles: list[object]) -> None:
        stop_calls.append("stop")

    monkeypatch.setattr(main_module.time, "monotonic", lambda: next(monotonic_values, 100.25))
    monkeypatch.setattr(main_module.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(main_module, "_stop_registered_job_pollers", _fake_stop_pollers)

    await main_module._quiesce_owned_job_pollers_for_shutdown(
        app,
        [_completed_poller_handle(main_module)],
        wait_for_leases_sec=0.2,
        count_active_processing=lambda: 2,
    )

    assert observed_sleeps == pytest.approx([0.1])
    assert stop_calls == ["stop"]
    segments = getattr(app.state, "_tldw_shutdown_timing_segments")
    assert segments[0]["segment"] == "optional_lease_wait"
    assert segments[0]["skipped"] is False
    assert segments[0]["initial_active"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_job_pollers_timeout_fallback_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    stop_event = asyncio.Event()
    wait_calls: list[tuple[object, float]] = []
    warnings: list[tuple[object, ...]] = []
    task = asyncio.get_running_loop().create_future()

    async def _fake_wait_for(awaitable: object, timeout: float) -> None:
        wait_calls.append((awaitable, timeout))
        raise asyncio.TimeoutError

    monkeypatch.setattr(main_module.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(
        lifecycle_workers.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    await main_module._stop_registered_job_pollers(
        app,
        [
            main_module._ManagedJobPoller(
                name="stuck_poller",
                task=task,
                stop_event=stop_event,
                timeout_sec=0.25,
            )
        ],
    )

    assert stop_event.is_set()
    assert task.cancelled() is True
    assert wait_calls[0][1] == 0.25
    assert wait_calls[0][0] is not task
    assert wait_calls[1] == (task, 1.0)
    assert getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names") == ["stuck_poller"]
    assert warnings


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_job_pollers_skips_quiesced_inventory_for_stubborn_poller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    class _StubbornTask:
        def __init__(self) -> None:
            self.cancel_calls = 0

        def __await__(self):
            async def _never() -> None:
                await asyncio.sleep(3600)

            return _never().__await__()

        def cancel(self) -> None:
            self.cancel_calls += 1

        def done(self) -> bool:
            return False

        def cancelled(self) -> bool:
            return False

    app = FastAPI()
    stop_event = asyncio.Event()
    stubborn_task = _StubbornTask()

    async def _fake_wait_for(awaitable: object, timeout: float) -> None:
        raise asyncio.TimeoutError

    monkeypatch.setattr(main_module.asyncio, "shield", lambda awaitable: awaitable)
    monkeypatch.setattr(main_module.asyncio, "wait_for", _fake_wait_for)
    await main_module._stop_registered_job_pollers(
        app,
        [
            main_module._ManagedJobPoller(
                name="stubborn_poller",
                task=stubborn_task,
                stop_event=stop_event,
                timeout_sec=0.25,
            )
        ],
    )

    assert stop_event.is_set()
    assert stubborn_task.cancel_calls == 1
    assert getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names") == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_job_pollers_continues_when_cancelled_task_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    warnings: list[tuple[object, ...]] = []
    cooperative_stop_event = asyncio.Event()

    async def _raise_after_cancel() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError as exc:
            raise RuntimeError("cancel boom") from exc

    async def _cooperative(stop_event: asyncio.Event) -> None:
        await stop_event.wait()

    async def _fake_wait_for(awaitable: object, timeout: float):
        if timeout == 0.01:
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(main_module.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(
        lifecycle_workers.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    raising_task = asyncio.create_task(_raise_after_cancel(), name="raising-poller")
    cooperative_task = asyncio.create_task(_cooperative(cooperative_stop_event), name="cooperative-poller")
    await asyncio.sleep(0)

    await main_module._stop_registered_job_pollers(
        app,
        [
            main_module._ManagedJobPoller(
                name="raising_poller",
                task=raising_task,
                timeout_sec=0.01,
            ),
            main_module._ManagedJobPoller(
                name="cooperative_poller",
                task=cooperative_task,
                stop_event=cooperative_stop_event,
                timeout_sec=0.25,
            ),
        ],
    )

    assert cooperative_stop_event.is_set() is True
    assert raising_task.done() is True
    assert cooperative_task.done() is True
    assert getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names") == [
        "raising_poller",
        "cooperative_poller",
    ]
    assert any("raised after cancellation" in str(args[0]) for args in warnings)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_job_pollers_only_marks_completed_pollers_as_quiesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    real_wait_for = asyncio.wait_for
    release_event = asyncio.Event()

    async def _ignores_cancel_until_released() -> None:
        while not release_event.is_set():
            try:
                await release_event.wait()
            except asyncio.CancelledError:
                continue

    async def _fake_wait_for(awaitable: object, timeout: float):
        if timeout in {0.01, 1.0}:
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(main_module.asyncio, "wait_for", _fake_wait_for)

    task = asyncio.create_task(_ignores_cancel_until_released(), name="stubborn-poller")
    await asyncio.sleep(0)

    await main_module._stop_registered_job_pollers(
        app,
        [
            main_module._ManagedJobPoller(
                name="stubborn_poller",
                task=task,
                timeout_sec=0.01,
            )
        ],
    )

    assert task.done() is False
    assert getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names") == []

    release_event.set()
    task.cancel()
    await real_wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_job_pollers_waits_for_pollers_concurrently() -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()

    async def _delayed_shutdown(stop_event: asyncio.Event, delay: float) -> None:
        await stop_event.wait()
        await asyncio.sleep(delay)

    stop_a = asyncio.Event()
    stop_b = asyncio.Event()
    task_a = asyncio.create_task(_delayed_shutdown(stop_a, 0.2))
    task_b = asyncio.create_task(_delayed_shutdown(stop_b, 0.2))

    started = asyncio.get_running_loop().time()
    await main_module._stop_registered_job_pollers(
        app,
        [
            main_module._ManagedJobPoller(
                name="poller_a",
                task=task_a,
                stop_event=stop_a,
                timeout_sec=1.0,
            ),
            main_module._ManagedJobPoller(
                name="poller_b",
                task=task_b,
                stop_event=stop_b,
                timeout_sec=1.0,
            ),
        ],
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert stop_a.is_set() is True
    assert stop_b.is_set() is True
    assert elapsed < 0.35
    assert getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names") == [
        "poller_a",
        "poller_b",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_registered_job_pollers_ignores_background_worker_handles() -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    stop_event = asyncio.Event()
    task = asyncio.create_task(_wait_for_stop(stop_event), name="background-worker")

    await main_module._stop_registered_job_pollers(
        app,
        [
            main_module._ManagedJobPoller(
                name="background_worker",
                task=task,
                stop_event=stop_event,
                shutdown_phase=main_module.WorkerShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        ],
    )

    assert stop_event.is_set() is False
    assert task.done() is False
    assert getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names") == []

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publish_shutdown_job_poller_inventory_captures_registered_metadata() -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    stop_event = asyncio.Event()
    task = asyncio.create_task(_wait_for_stop(stop_event), name="connectors-worker")
    handles = [
        main_module._ManagedJobPoller(
            name="connectors_jobs_task",
            task=task,
            stop_event=stop_event,
            timeout_sec=7.5,
        )
    ]

    shutdown_phase = getattr(handles[0], "shutdown_phase", "job_poller_quiesce")
    assert shutdown_phase == "job_poller_quiesce"  # nosec B101

    main_module._publish_shutdown_job_poller_inventory(app, handles)

    assert getattr(app.state, "_tldw_shutdown_job_poller_inventory") == [
        {
            "name": "connectors_jobs_task",
            "task_name": "connectors-worker",
            "has_stop_event": True,
            "timeout_sec": 7.5,
        }
    ]
    assert getattr(app.state, "_tldw_shutdown_worker_inventory") == [  # nosec B101
        {
            "name": "connectors_jobs_task",
            "task_name": "connectors-worker",
            "has_stop_event": True,
            "timeout_sec": 7.5,
            "category": None,
            "shutdown_phase": "job_poller_quiesce",
        }
    ]

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_owned_job_poller_inventory_preserves_background_worker_handles() -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    background_stop_event = asyncio.Event()
    background_task = asyncio.create_task(_wait_for_stop(background_stop_event), name="background-worker")
    handles = [
        main_module._ManagedJobPoller(
            name="background_worker",
            task=background_task,
            stop_event=background_stop_event,
            shutdown_phase=main_module.WorkerShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        )
    ]
    poller_task = asyncio.get_running_loop().create_future()
    poller_task.set_result(None)

    main_module._replace_owned_job_poller_inventory(
        app,
        handles,
        registrations=[
            ("poller_worker", poller_task, asyncio.Event(), 3.0),
        ],
    )

    assert {handle.name for handle in handles} == {"background_worker", "poller_worker"}
    assert {entry["name"] for entry in getattr(app.state, "_tldw_shutdown_worker_inventory")} == {
        "background_worker",
        "poller_worker",
    }
    assert getattr(app.state, "_tldw_shutdown_job_poller_inventory") == [
        {
            "name": "poller_worker",
            "task_name": "Future",
            "has_stop_event": True,
            "timeout_sec": 3.0,
        }
    ]

    background_stop_event.set()
    await asyncio.wait_for(background_task, timeout=1)


@pytest.mark.unit
def test_shutdown_timing_helpers_record_segments_and_total_summary() -> None:
    from tldw_Server_API.app import main as main_module

    app = FastAPI()

    for segment, duration_ms in (
        ("transition_handoff", 1),
        ("optional_lease_wait", 0),
        ("job_poller_quiesce", 2),
        ("evaluations_pool_shutdown", 3),
        ("unified_audit_and_executor_shutdown", 4),
        ("telemetry_shutdown", 5),
    ):
        main_module._record_shutdown_timing_segment(app, segment, duration_ms)

    main_module._record_shutdown_timing_total(app, duration_ms=21)

    segments = getattr(app.state, "_tldw_shutdown_timing_segments")
    assert [entry["segment"] for entry in segments] == [
        "transition_handoff",
        "optional_lease_wait",
        "job_poller_quiesce",
        "evaluations_pool_shutdown",
        "unified_audit_and_executor_shutdown",
        "telemetry_shutdown",
        "total app teardown",
    ]
    assert getattr(app.state, "_tldw_shutdown_timing_total") == {
        "duration_ms": 21,
        "slowest_segment": "telemetry_shutdown",
        "slowest_duration_ms": 5,
    }


@pytest.mark.integration
def test_lifespan_startup_registers_recipe_and_maintenance_pollers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = main_module.app
    if hasattr(app.state, "_tldw_shutdown_job_poller_inventory"):
        delattr(app.state, "_tldw_shutdown_job_poller_inventory")

    start_counts = {"recipe": 0, "maintenance": 0}

    async def _short_lived_task() -> None:
        await asyncio.sleep(0)

    async def _fake_start_recipe_run_jobs_worker(*, stop_event: asyncio.Event | None = None):
        start_counts["recipe"] += 1
        return asyncio.create_task(_short_lived_task(), name="recipe_run_jobs_worker")

    async def _fake_start_admin_maintenance_rotation_jobs_worker(
        *,
        stop_event: asyncio.Event | None = None,
    ):
        start_counts["maintenance"] += 1
        return asyncio.create_task(
            _short_lived_task(),
            name="admin_maintenance_rotation_jobs_worker",
        )

    monkeypatch.setenv("ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setattr(
        admin_maintenance_rotation_jobs_worker,
        "start_admin_maintenance_rotation_jobs_worker",
        _fake_start_admin_maintenance_rotation_jobs_worker,
    )
    monkeypatch.setattr(
        recipe_runs_jobs_worker,
        "start_recipe_run_jobs_worker",
        _fake_start_recipe_run_jobs_worker,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        inventory = list(getattr(app.state, "_tldw_shutdown_job_poller_inventory", []))

    inventory_names = {entry["name"] for entry in inventory}
    assert "admin_maintenance_rotation_jobs_task" in inventory_names
    assert "recipe_run_jobs_task" in inventory_names
    assert start_counts == {"recipe": 1, "maintenance": 1}


@pytest.mark.integration
def test_lifespan_startup_publishes_owned_job_poller_inventory_for_enabled_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = main_module.app
    if hasattr(app.state, "_tldw_shutdown_job_poller_inventory"):
        delattr(app.state, "_tldw_shutdown_job_poller_inventory")

    for key in (
        "CHATBOOKS_CORE_WORKER_ENABLED",
        "FILES_JOBS_WORKER_ENABLED",
        "AUDIO_JOBS_WORKER_ENABLED",
        "AUDIOBOOK_JOBS_WORKER_ENABLED",
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "READING_DIGEST_JOBS_WORKER_ENABLED",
        "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
        "REMINDER_JOBS_WORKER_ENABLED",
        "ADMIN_BACKUP_JOBS_WORKER_ENABLED",
        "ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED",
        "STUDY_PACK_JOBS_WORKER_ENABLED",
        "STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED",
        "PRIVILEGE_SNAPSHOT_WORKER_ENABLED",
        "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED",
        "CONNECTORS_WORKER_ENABLED",
    ):
        monkeypatch.setenv(key, "1")
    for key in (
        "DATA_TABLES_JOBS_WORKER_ENABLED",
        "PROMPT_STUDIO_JOBS_WORKER_ENABLED",
        "PRESENTATION_RENDER_JOBS_WORKER_ENABLED",
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "EVALS_ABTEST_JOBS_WORKER_ENABLED",
    ):
        monkeypatch.setenv(key, "0")

    monkeypatch.setattr(
        core_jobs_worker,
        "run_chatbooks_core_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(
        file_artifacts_jobs_worker,
        "run_file_artifacts_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(audio_jobs_worker, "run_audio_jobs_worker", _wait_for_optional_stop_event)
    monkeypatch.setattr(audiobook_jobs_worker, "run_audiobook_jobs_worker", _wait_for_optional_stop_event)
    monkeypatch.setattr(
        media_ingest_jobs_worker,
        "run_media_ingest_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(
        reading_digest_jobs_worker,
        "run_reading_digest_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(
        companion_reflection_jobs_worker,
        "run_companion_reflection_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(study_pack_jobs_worker, "run_study_pack_jobs_worker", _wait_for_optional_stop_event)
    monkeypatch.setattr(
        study_suggestions_jobs_worker,
        "run_study_suggestions_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(
        privilege_snapshot_worker,
        "run_privilege_snapshot_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(
        embeddings_abtest_jobs_worker,
        "run_embeddings_abtest_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(reminder_jobs_worker, "run_reminder_jobs_worker", _wait_for_optional_stop_event)
    monkeypatch.setattr(
        admin_backup_jobs_worker,
        "run_admin_backup_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(
        admin_byok_validation_jobs_worker,
        "run_admin_byok_validation_jobs_worker",
        _wait_for_optional_stop_event,
    )
    monkeypatch.setattr(connectors_worker, "run_connectors_worker", _wait_for_optional_stop_event)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        inventory = list(getattr(app.state, "_tldw_shutdown_job_poller_inventory", []))

    assert {entry["name"] for entry in inventory} == {
        "core_jobs_task",
        "files_jobs_task",
        "study_pack_jobs_task",
        "study_suggestions_jobs_task",
        "privilege_snapshot_task",
        "audio_jobs_task",
        "audiobook_jobs_task",
        "media_ingest_jobs_task",
        "reading_digest_jobs_task",
        "companion_reflection_jobs_task",
        "reminder_jobs_task",
        "admin_backup_jobs_task",
        "admin_byok_validation_jobs_task",
        "evals_abtest_jobs_task",
        "connectors_jobs_task",
    }
    assert all(
        {"name", "task_name", "has_stop_event", "timeout_sec"} <= set(entry)
        for entry in inventory
    )
    assert all(isinstance(entry["task_name"], str) and entry["task_name"] for entry in inventory)
    assert all(entry["has_stop_event"] is True for entry in inventory)
    assert all(entry["timeout_sec"] == 5.0 for entry in inventory)


@pytest.mark.integration
def test_lifespan_shutdown_stops_jobs_metrics_reconcile_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = main_module.app
    observed: dict[str, object] = {"stop_event": None, "stopped": False}

    async def _fake_reconcile(stop_event: asyncio.Event) -> None:
        observed["stop_event"] = stop_event
        await stop_event.wait()
        observed["stopped"] = True

    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "1")
    monkeypatch.setattr(jobs_metrics_service, "run_jobs_metrics_reconcile", _fake_reconcile)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["stop_event"]
        assert isinstance(stop_event, asyncio.Event)
        assert stop_event.is_set() is False

    stop_event = observed["stop_event"]
    assert isinstance(stop_event, asyncio.Event)
    assert stop_event.is_set() is True
    assert observed["stopped"] is True


@pytest.mark.integration
def test_lifespan_startup_publishes_first_batch_background_worker_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = main_module.app
    _clear_shutdown_worker_state(app)
    _enable_first_batch_background_workers(monkeypatch)
    observed = _patch_first_batch_background_workers(monkeypatch)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        _wait_until_first_batch_workers_started(observed)
        worker_inventory = list(getattr(app.state, "_tldw_shutdown_worker_inventory", []))
        job_poller_inventory = list(getattr(app.state, "_tldw_shutdown_job_poller_inventory", []))

    background_entries = {
        entry["name"]: entry
        for entry in worker_inventory
        if entry["name"] in _FIRST_BATCH_BACKGROUND_WORKER_NAMES
    }
    assert set(background_entries) == _FIRST_BATCH_BACKGROUND_WORKER_NAMES
    assert all(
        entry["shutdown_phase"] == "background_worker_shutdown"
        for entry in background_entries.values()
    )
    assert _FIRST_BATCH_BACKGROUND_WORKER_NAMES.isdisjoint(
        {entry["name"] for entry in job_poller_inventory}
    )


@pytest.mark.integration
def test_background_workers_do_not_enter_job_poller_quiesce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = main_module.app
    _clear_shutdown_worker_state(app)
    _enable_first_batch_background_workers(monkeypatch)
    observed = _patch_first_batch_background_workers(monkeypatch)
    original_stop_pollers = main_module._stop_registered_job_pollers
    quiesce_seen_names: list[str] = []

    async def _spy_stop_pollers(_app: FastAPI, handles: list[object]) -> None:
        quiesce_seen_names.extend(handle.name for handle in handles)
        await original_stop_pollers(_app, handles)

    monkeypatch.setattr(main_module, "_stop_registered_job_pollers", _spy_stop_pollers)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        _wait_until_first_batch_workers_started(observed)

    assert _FIRST_BATCH_BACKGROUND_WORKER_NAMES.isdisjoint(quiesce_seen_names)
    for name, record in observed.items():
        stop_event = record["stop_event"]
        assert isinstance(stop_event, asyncio.Event)
        assert stop_event.is_set() is True
        assert record["set_calls"] == 1, name
        assert record["completed"] == 1, name


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_connectors_worker_uses_caller_owned_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    stop_event = asyncio.Event()

    async def _fake_run(stop_event_arg: asyncio.Event | None = None) -> None:
        observed["stop_event"] = stop_event_arg
        await stop_event_arg.wait()

    monkeypatch.setenv("CONNECTORS_WORKER_ENABLED", "1")
    monkeypatch.setattr(connectors_worker, "run_connectors_worker", _fake_run)

    task = await connectors_worker.start_connectors_worker(stop_event=stop_event)
    assert task is not None
    await asyncio.sleep(0)

    assert observed["stop_event"] is stop_event

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_reminder_jobs_worker_uses_caller_owned_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    stop_event = asyncio.Event()

    async def _fake_run(stop_event_arg: asyncio.Event | None = None) -> None:
        observed["stop_event"] = stop_event_arg
        await stop_event_arg.wait()

    monkeypatch.setenv("REMINDER_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setattr(reminder_jobs_worker, "run_reminder_jobs_worker", _fake_run)

    task = await reminder_jobs_worker.start_reminder_jobs_worker(stop_event=stop_event)
    assert task is not None
    await asyncio.sleep(0)

    assert observed["stop_event"] is stop_event

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_admin_backup_jobs_worker_stops_sdk_when_stop_event_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    observed = {"stopped": False}

    class _FakeWorkerSDK:
        def __init__(self, _jm, _cfg) -> None:
            pass

        def stop(self) -> None:
            observed["stopped"] = True

        async def run(self, *, handler) -> None:
            while not observed["stopped"]:
                await asyncio.sleep(0)

    monkeypatch.setattr(admin_backup_jobs_worker, "WorkerSDK", _FakeWorkerSDK)

    task = asyncio.create_task(admin_backup_jobs_worker.run_admin_backup_jobs_worker(stop_event))
    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert observed["stopped"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_admin_backup_jobs_worker_forwards_caller_owned_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    stop_event = asyncio.Event()

    async def _fake_run(stop_event_arg: asyncio.Event | None = None) -> None:
        observed["stop_event"] = stop_event_arg
        await stop_event_arg.wait()

    monkeypatch.setenv("ADMIN_BACKUP_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setattr(admin_backup_jobs_worker, "run_admin_backup_jobs_worker", _fake_run)

    task = await admin_backup_jobs_worker.start_admin_backup_jobs_worker(stop_event=stop_event)
    assert task is not None
    await asyncio.sleep(0)

    assert observed["stop_event"] is stop_event

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_admin_byok_validation_jobs_worker_stops_sdk_when_stop_event_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    observed = {"stopped": False}

    class _FakeWorkerSDK:
        def __init__(self, _jm, _cfg) -> None:
            pass

        def stop(self) -> None:
            observed["stopped"] = True

        async def run(self, *, handler) -> None:
            while not observed["stopped"]:
                await asyncio.sleep(0)

    monkeypatch.setattr(admin_byok_validation_jobs_worker, "WorkerSDK", _FakeWorkerSDK)

    task = asyncio.create_task(
        admin_byok_validation_jobs_worker.run_admin_byok_validation_jobs_worker(stop_event)
    )
    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert observed["stopped"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_admin_byok_validation_jobs_worker_forwards_caller_owned_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    stop_event = asyncio.Event()

    async def _fake_run(stop_event_arg: asyncio.Event | None = None) -> None:
        observed["stop_event"] = stop_event_arg
        await stop_event_arg.wait()

    monkeypatch.setenv("ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setattr(
        admin_byok_validation_jobs_worker,
        "run_admin_byok_validation_jobs_worker",
        _fake_run,
    )

    task = await admin_byok_validation_jobs_worker.start_admin_byok_validation_jobs_worker(
        stop_event=stop_event
    )
    assert task is not None
    await asyncio.sleep(0)

    assert observed["stop_event"] is stop_event

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)
