from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_startup_shutdown_contract_is_reentrant() -> None:
    from tldw_Server_API.app.main import app

    with TestClient(app) as first_client:
        first_response = first_client.get("/health")
        assert first_response.status_code == 200
        first_state = app.state._tldw_lifecycle_state
        assert first_state.phase == "ready"
        assert first_state.ready is True

    assert app.state._tldw_lifecycle_state is first_state
    assert app.state._tldw_lifecycle_state.phase == "draining"
    assert app.state._tldw_lifecycle_state.ready is False

    with TestClient(app) as second_client:
        second_response = second_client.get("/health")
        assert second_response.status_code == 200
        assert app.state._tldw_lifecycle_state is first_state
        assert app.state._tldw_lifecycle_state.phase == "ready"
        assert app.state._tldw_lifecycle_state.ready is True

    assert app.state._tldw_lifecycle_state is first_state
    assert app.state._tldw_lifecycle_state.phase == "draining"
    assert app.state._tldw_lifecycle_state.ready is False


@pytest.mark.asyncio
async def test_asgi_transport_without_lifespan_bypasses_shutdown_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module

    app = main_module.app

    for attr_name in (
        "_tldw_lifecycle_events",
        "_tldw_lifecycle_state",
        "_tldw_shutdown_legacy_coordinator_summary",
        "_tldw_shutdown_legacy_coordinator_component_names",
        "_tldw_shutdown_legacy_coordinator_phase_groups",
    ):
        if hasattr(app.state, attr_name):
            delattr(app.state, attr_name)

    startup_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    shutdown_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_mark_startup = main_module.mark_lifecycle_startup
    original_mark_shutdown = main_module.mark_lifecycle_shutdown

    def _record_mark_startup(*args, **kwargs):
        startup_calls.append((args, kwargs))
        return original_mark_startup(*args, **kwargs)

    def _record_mark_shutdown(*args, **kwargs):
        shutdown_calls.append((args, kwargs))
        return original_mark_shutdown(*args, **kwargs)

    monkeypatch.setattr(main_module, "mark_lifecycle_startup", _record_mark_startup)
    monkeypatch.setattr(main_module, "mark_lifecycle_shutdown", _record_mark_shutdown)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert startup_calls == []
    assert shutdown_calls == []
    assert not hasattr(app.state, "_tldw_shutdown_legacy_coordinator_summary")

    lifecycle_state = getattr(app.state, "_tldw_lifecycle_state", None)
    assert lifecycle_state is None or lifecycle_state.phase != "draining"
    assert lifecycle_state is None or lifecycle_state.draining is False


def test_build_legacy_shutdown_context_uses_explicit_fields() -> None:
    from tldw_Server_API.app.main import _build_legacy_shutdown_context
    from tldw_Server_API.app.services.shutdown_legacy_adapters import LegacyShutdownContext

    readiness_state = {"ready": True}

    context = _build_legacy_shutdown_context(
        readiness_state=readiness_state,
    )

    assert isinstance(context, LegacyShutdownContext)
    assert context.readiness_state is readiness_state
    assert not hasattr(context, "usage_task")
    assert not hasattr(context, "llm_usage_task")
    assert not hasattr(context, "authnz_scheduler_started")


def test_main_source_does_not_define_legacy_display_startup_info_helper() -> None:
    from tldw_Server_API.app import main as main_module

    main_source = Path(main_module.__file__).read_text()

    assert "async def _display_startup_info_and_warm" not in main_source


def test_apply_shutdown_transition_gate_logs_guard_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from fastapi import FastAPI

    from tldw_Server_API.app import main as main_module

    app = FastAPI()
    debug_messages: list[str] = []
    warning_messages: list[str] = []

    def _raise_lifecycle_state(_app):
        raise RuntimeError("lookup failed")

    def _raise_mark_shutdown(_app, _readiness_state):
        raise RuntimeError("mark failed")

    class _FailingJobManager:
        @classmethod
        def set_acquire_gate(cls, _enabled: bool) -> None:
            raise RuntimeError("gate failed")

    fake_jobs_manager = types.ModuleType("tldw_Server_API.app.core.Jobs.manager")
    fake_jobs_manager.JobManager = _FailingJobManager

    monkeypatch.setattr(main_module, "get_or_create_lifecycle_state", _raise_lifecycle_state)
    monkeypatch.setattr(main_module, "mark_lifecycle_shutdown", _raise_mark_shutdown)
    monkeypatch.setattr(
        main_module.logger, "debug", lambda message, *args, **kwargs: debug_messages.append(str(message))
    )
    monkeypatch.setattr(
        main_module.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )
    monkeypatch.setitem(sys.modules, fake_jobs_manager.__name__, fake_jobs_manager)

    main_module._apply_shutdown_transition_gate(app, {})

    assert any("lifecycle state lookup skipped" in message for message in debug_messages)
    assert any("failed to mark lifecycle shutdown" in message for message in warning_messages)
    assert any("job acquire gate unavailable" in message for message in debug_messages)


@pytest.mark.integration
def test_lifecycle_hooks_called_in_order() -> None:
    from tldw_Server_API.app.main import app

    if hasattr(app.state, "_tldw_lifecycle_events"):
        delattr(app.state, "_tldw_lifecycle_events")
    if hasattr(app.state, "_tldw_lifecycle_state"):
        delattr(app.state, "_tldw_lifecycle_state")

    with TestClient(app):
        assert getattr(app.state, "_tldw_lifecycle_events", [])[-1:] == ["startup"]
        assert app.state._tldw_lifecycle_state.phase == "ready"
        assert app.state._tldw_lifecycle_state.ready is True

    assert getattr(app.state, "_tldw_lifecycle_events", [])[-2:] == ["startup", "shutdown"]
    assert app.state._tldw_lifecycle_state.phase == "draining"
    assert app.state._tldw_lifecycle_state.ready is False


@pytest.mark.integration
def test_lifespan_startup_delegates_startup_bg_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_bg_tasks

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_prepare_startup_bg_tasks(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_bg_tasks,
        "prepare_startup_bg_tasks",
        _fake_prepare_startup_bg_tasks,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_owned_job_pollers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_service_tail, startup_worker_groups

    app = main_module.app
    worker_group_calls: list[dict[str, object]] = []
    service_tail_calls: list[dict[str, object]] = []

    async def _fake_start_worker_groups(**kwargs):
        worker_group_calls.append(kwargs)
        return startup_worker_groups.StartupWorkerGroupHandles()

    async def _fake_initialize_startup_service_tail(**kwargs):
        service_tail_calls.append(kwargs)
        return startup_service_tail.StartupServiceTailHandles()

    monkeypatch.setattr(
        startup_worker_groups,
        "start_worker_groups",
        _fake_start_worker_groups,
    )
    monkeypatch.setattr(
        startup_service_tail,
        "initialize_startup_service_tail",
        _fake_initialize_startup_service_tail,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(worker_group_calls) == 1
    assert len(service_tail_calls) == 1
    owned_job_pollers = worker_group_calls[0]["owned_job_pollers"]
    assert isinstance(owned_job_pollers, list)
    assert service_tail_calls[0]["owned_job_pollers"] is owned_job_pollers
    assert service_tail_calls[0]["worker_inventory"].handles is owned_job_pollers


@pytest.mark.integration
def test_lifespan_startup_delegates_startup_transition_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_transition_gate

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_apply_startup_transition_gate(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_transition_gate,
        "apply_startup_transition_gate",
        _fake_apply_startup_transition_gate,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["readiness_state"] is main_module.READINESS_STATE
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_startup_test_runtime_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_test_runtime_guard

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_validate_startup_test_runtime(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_test_runtime_guard,
        "validate_startup_test_runtime",
        _fake_validate_startup_test_runtime,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_shutdown_stops_embeddings_compactor_and_websub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.core.Embeddings.services import vector_compactor
    from tldw_Server_API.app.core.Watchlists import websub

    app = main_module.app
    observed: dict[str, object] = {
        "compactor_stop_event": None,
        "compactor_stopped": False,
        "websub_cancelled": False,
    }

    async def _fake_compactor(stop_event):
        observed["compactor_stop_event"] = stop_event
        await stop_event.wait()
        observed["compactor_stopped"] = True

    async def _fake_websub_loop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed["websub_cancelled"] = True
            raise

    monkeypatch.setenv("EMBEDDINGS_COMPACTOR_ENABLED", "1")
    monkeypatch.setenv("WEBSUB_CALLBACK_BASE_URL", "http://callback.example")
    monkeypatch.setenv("WEBSUB_RENEWAL_WORKER_ENABLED", "1")
    monkeypatch.setattr(vector_compactor, "run", _fake_compactor)
    monkeypatch.setattr(websub, "websub_renewal_loop", _fake_websub_loop)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["compactor_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["compactor_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["compactor_stopped"] is True
    assert observed["websub_cancelled"] is True


@pytest.mark.integration
def test_lifespan_shutdown_cancels_jobs_notifications_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import jobs_notifications_service

    app = main_module.app
    observed = {"bridge_cancelled": False}

    async def _fake_bridge_loop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed["bridge_cancelled"] = True
            raise

    async def _fake_start_jobs_notifications_service():
        return asyncio.create_task(_fake_bridge_loop(), name="jobs_notifications_bridge")

    monkeypatch.setenv("JOBS_NOTIFICATIONS_BRIDGE_ENABLED", "1")
    monkeypatch.delenv("JOBS_NOTIFICATIONS_BRIDGE_DISABLED", raising=False)
    monkeypatch.setattr(
        jobs_notifications_service,
        "start_jobs_notifications_service",
        _fake_start_jobs_notifications_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert observed["bridge_cancelled"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_recipe_run_and_evals_abtest_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_notifications_abtest_workers, startup_sidecar_owned_jobs_pollers

    app = main_module.app
    observed: dict[str, object] = {
        "recipe_stop_event": None,
        "recipe_stopped": False,
        "abtest_stop_event": None,
        "abtest_stopped": False,
    }

    async def _fake_recipe_worker(stop_event):
        observed["recipe_stop_event"] = stop_event
        await stop_event.wait()
        observed["recipe_stopped"] = True

    async def _fake_abtest_worker(stop_event):
        observed["abtest_stop_event"] = stop_event
        await stop_event.wait()
        observed["abtest_stopped"] = True

    async def _fake_start_recipe_run_jobs_worker_service(*, stop_event):
        return asyncio.create_task(_fake_recipe_worker(stop_event), name="recipe_run_jobs_task")

    monkeypatch.setenv("EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setattr(
        startup_sidecar_owned_jobs_pollers,
        "_start_recipe_run_jobs_worker_service",
        _fake_start_recipe_run_jobs_worker_service,
    )
    monkeypatch.setattr(
        startup_notifications_abtest_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        _fake_abtest_worker,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        recipe_stop_event = observed["recipe_stop_event"]
        abtest_stop_event = observed["abtest_stop_event"]
        assert recipe_stop_event is not None
        assert recipe_stop_event.is_set() is False
        assert abtest_stop_event is not None
        assert abtest_stop_event.is_set() is False

    recipe_stop_event = observed["recipe_stop_event"]
    abtest_stop_event = observed["abtest_stop_event"]
    assert recipe_stop_event is not None
    assert recipe_stop_event.is_set() is True
    assert observed["recipe_stopped"] is True
    assert abtest_stop_event is not None
    assert abtest_stop_event.is_set() is True
    assert observed["abtest_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_cancels_claims_and_maintenance_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_claims_rebuild, startup_maintenance_schedulers
    from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker, ShutdownPhase

    app = main_module.app
    observed = {
        "claims_cancelled": False,
        "jobs_prune_cancelled": False,
        "files_gc_cancelled": False,
        "notifications_prune_cancelled": False,
    }

    async def _wait_forever(flag_key: str) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed[flag_key] = True
            raise

    async def _fake_start_claims_rebuild_worker(_app_settings, **kwargs):
        worker_inventory = kwargs["worker_inventory"]
        assert worker_inventory is not None
        task = asyncio.create_task(_wait_forever("claims_cancelled"), name="claims_task")
        worker_inventory.register(
            ManagedWorker(
                name="claims_rebuild",
                task=task,
                timeout_sec=0.1,
                category="claims",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        )
        return task

    async def _fake_start_maintenance_schedulers(*, worker_inventory=None):
        assert worker_inventory is not None
        files_gc_task = asyncio.create_task(_wait_forever("files_gc_cancelled"), name="files_gc_task")
        notifications_prune_task = asyncio.create_task(
            _wait_forever("notifications_prune_cancelled"),
            name="notifications_prune_task",
        )
        jobs_prune_task = asyncio.create_task(_wait_forever("jobs_prune_cancelled"), name="jobs_prune_task")
        for worker_name, task in (
            ("files_export_gc_task", files_gc_task),
            ("notifications_prune_task", notifications_prune_task),
            ("jobs_prune_task", jobs_prune_task),
        ):
            worker_inventory.register(
                ManagedWorker(
                    name=worker_name,
                    task=task,
                    timeout_sec=0.1,
                    category="maintenance",
                    shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
                )
            )
        return startup_maintenance_schedulers.MaintenanceSchedulerHandles(
            files_export_gc_task=files_gc_task,
            notifications_prune_task=notifications_prune_task,
            jobs_prune_task=jobs_prune_task,
        )

    monkeypatch.setattr(
        startup_claims_rebuild,
        "start_claims_rebuild_worker",
        _fake_start_claims_rebuild_worker,
    )
    monkeypatch.setattr(
        startup_maintenance_schedulers,
        "start_maintenance_schedulers",
        _fake_start_maintenance_schedulers,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert observed["claims_cancelled"] is True
    assert observed["jobs_prune_cancelled"] is True
    assert observed["files_gc_cancelled"] is True
    assert observed["notifications_prune_cancelled"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_reminder_admin_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_sidecar_owned_jobs_pollers

    app = main_module.app
    observed = {
        "reminder_cancelled": False,
        "admin_backup_cancelled": False,
        "admin_maintenance_stop_event": None,
        "admin_maintenance_stopped": False,
    }

    async def _wait_for_cancel(flag_key: str) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed[flag_key] = True
            raise

    async def _wait_for_stop(stop_event, *, flag_key: str) -> None:
        observed["admin_maintenance_stop_event"] = stop_event
        await stop_event.wait()
        observed[flag_key] = True

    async def _fake_start_reminder_jobs_worker_service(*, stop_event):
        del stop_event
        return asyncio.create_task(_wait_for_cancel("reminder_cancelled"), name="reminder_jobs_task")

    async def _fake_start_admin_backup_jobs_worker_service(*, stop_event):
        del stop_event
        return asyncio.create_task(_wait_for_cancel("admin_backup_cancelled"), name="admin_backup_jobs_task")

    async def _fake_start_admin_maintenance_rotation_jobs_worker_service(*, stop_event):
        return asyncio.create_task(
            _wait_for_stop(stop_event, flag_key="admin_maintenance_stopped"),
            name="admin_maintenance_rotation_jobs_task",
        )

    monkeypatch.setattr(
        startup_sidecar_owned_jobs_pollers,
        "_start_reminder_jobs_worker_service",
        _fake_start_reminder_jobs_worker_service,
    )
    monkeypatch.setattr(
        startup_sidecar_owned_jobs_pollers,
        "_start_admin_backup_jobs_worker_service",
        _fake_start_admin_backup_jobs_worker_service,
    )
    monkeypatch.setattr(
        startup_sidecar_owned_jobs_pollers,
        "_start_admin_maintenance_rotation_jobs_worker_service",
        _fake_start_admin_maintenance_rotation_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["admin_maintenance_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["admin_maintenance_stop_event"]
    assert observed["reminder_cancelled"] is True
    assert observed["admin_backup_cancelled"] is True
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["admin_maintenance_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_reading_study_companion_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_content_jobs_pollers, startup_study_privilege_jobs_pollers

    app = main_module.app
    observed = {
        "reading_stop_event": None,
        "reading_stopped": False,
        "study_pack_stop_event": None,
        "study_pack_stopped": False,
        "study_suggestions_stop_event": None,
        "study_suggestions_stopped": False,
        "companion_stop_event": None,
        "companion_stopped": False,
    }

    for key in (
        "READING_DIGEST_JOBS_WORKER_ENABLED",
        "STUDY_PACK_JOBS_WORKER_ENABLED",
        "STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED",
        "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
    ):
        monkeypatch.setenv(key, "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_reading_digest_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="reading_stop_event",
            stopped_key="reading_stopped",
        )

    def _fake_study_pack_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="study_pack_stop_event",
            stopped_key="study_pack_stopped",
        )

    def _fake_study_suggestions_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="study_suggestions_stop_event",
            stopped_key="study_suggestions_stopped",
        )

    def _fake_companion_reflection_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="companion_stop_event",
            stopped_key="companion_stopped",
        )

    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_run_reading_digest_jobs_worker_service",
        _fake_reading_digest_jobs_worker_service,
    )
    monkeypatch.setattr(
        startup_study_privilege_jobs_pollers,
        "_run_study_pack_jobs_worker_service",
        _fake_study_pack_jobs_worker_service,
    )
    monkeypatch.setattr(
        startup_study_privilege_jobs_pollers,
        "_run_study_suggestions_jobs_worker_service",
        _fake_study_suggestions_jobs_worker_service,
    )
    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_run_companion_reflection_jobs_worker_service",
        _fake_companion_reflection_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        for key in (
            "reading_stop_event",
            "study_pack_stop_event",
            "study_suggestions_stop_event",
            "companion_stop_event",
        ):
            stop_event = observed[key]
            assert stop_event is not None
            assert stop_event.is_set() is False

    for key in (
        "reading_stop_event",
        "study_pack_stop_event",
        "study_suggestions_stop_event",
        "companion_stop_event",
    ):
        stop_event = observed[key]
        assert stop_event is not None
        assert stop_event.is_set() is True
    assert observed["reading_stopped"] is True
    assert observed["study_pack_stopped"] is True
    assert observed["study_suggestions_stopped"] is True
    assert observed["companion_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_media_ingest_shutdown_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_content_jobs_pollers

    app = main_module.app
    observed = {
        "media_stop_event": None,
        "media_stopped": False,
        "media_heavy_stop_event": None,
        "media_heavy_stopped": False,
    }

    monkeypatch.setenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_media_ingest_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="media_stop_event",
            stopped_key="media_stopped",
        )

    def _fake_media_ingest_heavy_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="media_heavy_stop_event",
            stopped_key="media_heavy_stopped",
        )

    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_run_media_ingest_jobs_worker_service",
        _fake_media_ingest_jobs_worker_service,
    )
    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_run_media_ingest_heavy_jobs_worker_service",
        _fake_media_ingest_heavy_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        for key in ("media_stop_event", "media_heavy_stop_event"):
            stop_event = observed[key]
            assert stop_event is not None
            assert stop_event.is_set() is False

    for key in ("media_stop_event", "media_heavy_stop_event"):
        stop_event = observed[key]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["media_stopped"] is True
    assert observed["media_heavy_stopped"] is True


@pytest.mark.integration
def test_lifespan_startup_delegates_worker_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_service_tail, startup_worker_bootstrap, startup_worker_groups

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_initialize_startup_worker_bootstrap(**kwargs):
        recorded_calls.append(kwargs)
        return startup_worker_bootstrap.StartupWorkerBootstrapHandles(
            app_settings="settings",
            owned_job_pollers=["poller"],
            startup_worker_group_handles=startup_worker_groups.StartupWorkerGroupHandles(),
            startup_service_tail_handles=startup_service_tail.StartupServiceTailHandles(),
        )

    monkeypatch.setattr(
        startup_worker_bootstrap,
        "initialize_startup_worker_bootstrap",
        _fake_initialize_startup_worker_bootstrap,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["test_mode"] is True
    assert recorded_calls[0]["route_enabled"] is main_module.route_enabled
    assert recorded_calls[0]["run_pg_rls_auto_ensure"] is main_module._run_pg_rls_auto_ensure
    assert recorded_calls[0]["register_owned_job_poller"] is main_module._register_owned_job_poller
    assert recorded_calls[0]["replace_owned_job_poller_inventory"] is (main_module._replace_owned_job_poller_inventory)
    assert "publish_shutdown_job_poller_inventory" not in recorded_calls[0]
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_api_key_log_value"] is main_module._startup_api_key_log_value
    assert recorded_calls[0]["shared_is_truthy"] is main_module._shared_is_truthy
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_pre_core_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_core_initialization, startup_pre_core

    app = main_module.app
    pre_core_calls: list[dict[str, object]] = []
    core_calls: list[dict[str, object]] = []

    async def _fake_prepare_startup_pre_core(**kwargs):
        pre_core_calls.append(kwargs)
        return True

    async def _fake_initialize_startup_core_components(**kwargs):
        core_calls.append(kwargs)
        return startup_core_initialization.StartupCoreInitializationHandles()

    monkeypatch.setattr(
        startup_pre_core,
        "prepare_startup_pre_core",
        _fake_prepare_startup_pre_core,
    )
    monkeypatch.setattr(
        startup_core_initialization,
        "initialize_startup_core_components",
        _fake_initialize_startup_core_components,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(pre_core_calls) == 1
    assert pre_core_calls[0]["app"] is app
    assert pre_core_calls[0]["logger"] is main_module.logger
    assert pre_core_calls[0]["readiness_state"] is main_module.READINESS_STATE
    assert pre_core_calls[0]["shared_is_truthy"] is main_module._shared_is_truthy
    assert pre_core_calls[0]["route_enabled"] is main_module.route_enabled
    assert pre_core_calls[0]["get_mcp_config"] is main_module.get_mcp_config
    assert pre_core_calls[0]["validate_mcp_config"] is main_module.validate_mcp_config
    assert pre_core_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert pre_core_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS
    assert pre_core_calls[0]["test_mode"] is True

    assert len(core_calls) == 1
    assert core_calls[0]["defer_heavy"] is True


@pytest.mark.integration
def test_lifespan_startup_delegates_core_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_core_initialization

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_initialize_startup_core_components(**kwargs):
        recorded_calls.append(kwargs)
        return startup_core_initialization.StartupCoreInitializationHandles(
            db_pool="db-pool",
            session_manager="session-manager",
            heavy_startup_handles="heavy-handles",
        )

    monkeypatch.setattr(
        startup_core_initialization,
        "initialize_startup_core_components",
        _fake_initialize_startup_core_components,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["module_file"] == main_module.__file__
    assert recorded_calls[0]["logger"] is main_module.logger
    assert callable(recorded_calls[0]["route_enabled"])
    assert isinstance(recorded_calls[0]["defer_heavy"], bool)
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_worker_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_worker_groups

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_start_worker_groups(**kwargs):
        recorded_calls.append(kwargs)
        return startup_worker_groups.StartupWorkerGroupHandles()

    monkeypatch.setattr(
        startup_worker_groups,
        "start_worker_groups",
        _fake_start_worker_groups,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["test_mode"] is True
    assert recorded_calls[0]["route_enabled"] is main_module.route_enabled
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert isinstance(recorded_calls[0]["owned_job_pollers"], list)
    assert recorded_calls[0]["register_owned_job_poller"] is main_module._register_owned_job_poller
    assert "app_settings" in recorded_calls[0]


@pytest.mark.integration
def test_lifespan_startup_delegates_service_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_service_tail

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_initialize_startup_service_tail(**kwargs):
        recorded_calls.append(kwargs)
        return startup_service_tail.StartupServiceTailHandles()

    monkeypatch.setattr(
        startup_service_tail,
        "initialize_startup_service_tail",
        _fake_initialize_startup_service_tail,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["logger"] is main_module.logger
    assert isinstance(recorded_calls[0]["owned_job_pollers"], list)
    assert recorded_calls[0]["register_owned_job_poller"] is main_module._register_owned_job_poller
    assert recorded_calls[0]["run_pg_rls_auto_ensure"] is main_module._run_pg_rls_auto_ensure
    assert recorded_calls[0]["replace_owned_job_poller_inventory"] is (main_module._replace_owned_job_poller_inventory)
    assert isinstance(
        recorded_calls[0]["startup_worker_group_handles"],
        object,
    )
    assert recorded_calls[0]["test_mode"] is True
    assert recorded_calls[0]["startup_api_key_log_value"] is main_module._startup_api_key_log_value
    assert recorded_calls[0]["shared_is_truthy"] is main_module._shared_is_truthy
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS
    assert "app_settings" in recorded_calls[0]


@pytest.mark.integration
def test_lifespan_startup_delegates_chacha_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_chacha_warmup

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_warm_chacha_notes_on_startup(**kwargs):
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_chacha_warmup,
        "warm_chacha_notes_on_startup",
        _fake_warm_chacha_notes_on_startup,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_heavy_startup_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_heavy_init, startup_heavy_policy

    app = main_module.app
    policy_calls: list[dict[str, object]] = []
    heavy_init_calls: list[dict[str, object]] = []

    def _fake_resolve_deferred_heavy_startup(**kwargs) -> bool:
        policy_calls.append(kwargs)
        return True

    async def _fake_start_heavy_initializations(app_arg, **kwargs):
        heavy_init_calls.append({"app": app_arg, **kwargs})
        return startup_heavy_init.HeavyStartupHandles()

    monkeypatch.setattr(
        startup_heavy_policy,
        "resolve_deferred_heavy_startup",
        _fake_resolve_deferred_heavy_startup,
    )
    monkeypatch.setattr(
        startup_heavy_init,
        "start_heavy_initializations",
        _fake_start_heavy_initializations,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(policy_calls) == 1
    assert policy_calls[0]["shared_is_truthy"] is main_module._shared_is_truthy
    assert policy_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert len(heavy_init_calls) == 1
    assert heavy_init_calls[0]["app"] is app
    assert heavy_init_calls[0]["route_enabled"] is main_module.route_enabled
    assert heavy_init_calls[0]["defer_heavy"] is True


@pytest.mark.integration
def test_lifespan_startup_delegates_privilege_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_privilege_validation

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_validate_startup_privilege_metadata(**kwargs):
        recorded_calls.append(kwargs)
        return {"media.ingest": []}

    monkeypatch.setattr(
        startup_privilege_validation,
        "validate_startup_privilege_metadata",
        _fake_validate_startup_privilege_metadata,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_catalog_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_catalog_loading

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_load_startup_catalogs(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_catalog_loading,
        "load_startup_catalogs",
        _fake_load_startup_catalogs,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["module_file"] == main_module.__file__
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_sentry_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_sentry

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_initialize_startup_sentry(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_sentry,
        "initialize_startup_sentry",
        _fake_initialize_startup_sentry,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_auth_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_auth_runtime

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_initialize_auth_runtime_services(**kwargs):
        recorded_calls.append(kwargs)
        return SimpleNamespace(db_pool=None, session_manager=None)

    monkeypatch.setattr(
        startup_auth_runtime,
        "initialize_auth_runtime_services",
        _fake_initialize_auth_runtime_services,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_telemetry

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_initialize_startup_telemetry(**kwargs):
        recorded_calls.append(kwargs)
        return None

    monkeypatch.setattr(
        startup_telemetry,
        "initialize_startup_telemetry",
        _fake_initialize_startup_telemetry,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_evaluations_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_evaluations_warmup

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_warm_lazy_evaluations_managers(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_evaluations_warmup,
        "warm_lazy_evaluations_managers",
        _fake_warm_lazy_evaluations_managers,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["route_enabled"] is main_module.route_enabled
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["test_mode"] == bool(getattr(main_module, "_TEST_MODE", False))


@pytest.mark.integration
def test_lifespan_startup_delegates_prompts_close_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_prompts_close_worker

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_start_prompts_close_worker(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_prompts_close_worker,
        "start_prompts_close_worker",
        _fake_start_prompts_close_worker,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_content_backend_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_content_backend_validation

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_validate_startup_content_backend(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_content_backend_validation,
        "validate_startup_content_backend",
        _fake_validate_startup_content_backend,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["logger"] is main_module.logger


@pytest.mark.integration
def test_lifespan_startup_delegates_preflight_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_preflight_reporting

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_run_startup_preflight_checks(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_preflight_reporting,
        "run_startup_preflight_checks",
        _fake_run_startup_preflight_checks,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_mcp_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_mcp_validation

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_validate_startup_mcp_configuration(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_mcp_validation,
        "validate_startup_mcp_configuration",
        _fake_validate_startup_mcp_configuration,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["get_mcp_config"] is main_module.get_mcp_config
    assert recorded_calls[0]["validate_mcp_config"] is main_module.validate_mcp_config
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_acp_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_acp_validation

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_validate_startup_acp_configuration(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_acp_validation,
        "validate_startup_acp_configuration",
        _fake_validate_startup_acp_configuration,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["route_enabled"] is main_module.route_enabled
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_startup_delegates_claims_prompt_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_claims_prompt_validation

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def _fake_validate_startup_claims_prompt_validation(**kwargs) -> None:
        recorded_calls.append(kwargs)

    monkeypatch.setattr(
        startup_claims_prompt_validation,
        "validate_startup_claims_prompt_validation",
        _fake_validate_startup_claims_prompt_validation,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["logger"] is main_module.logger
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_shutdown_delegates_pre_worker_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_pre_worker_cleanup as shutdown_cleanup
    from tldw_Server_API.app.services import startup_cleanup_workers

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_start_cleanup_workers(
        app_settings: object,
        *,
        test_mode: bool,
        worker_inventory: object,
    ) -> startup_cleanup_workers.CleanupWorkerHandles:
        del app_settings, test_mode
        assert worker_inventory is not None
        return startup_cleanup_workers.CleanupWorkerHandles()

    async def _fake_run_shutdown_pre_worker_cleanup(**kwargs):
        recorded_calls.append(kwargs)
        return shutdown_cleanup.PreWorkerCleanupHandles()

    monkeypatch.setattr(
        startup_cleanup_workers,
        "start_cleanup_workers",
        _fake_start_cleanup_workers,
    )
    monkeypatch.setattr(
        shutdown_cleanup,
        "run_shutdown_pre_worker_cleanup",
        _fake_run_shutdown_pre_worker_cleanup,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_shutdown_delegates_transition_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_transition_handoff as transition_handoff
    from tldw_Server_API.app.services import startup_auxiliary_services, startup_cleanup_workers

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    async def _fake_start_auxiliary_services(_app_settings, **kwargs):
        del _app_settings
        assert kwargs["worker_inventory"] is not None
        return startup_auxiliary_services.AuxiliaryStartupHandles()

    async def _fake_start_cleanup_workers(
        app_settings: object,
        *,
        test_mode: bool,
        worker_inventory: object,
    ) -> startup_cleanup_workers.CleanupWorkerHandles:
        del app_settings, test_mode
        assert worker_inventory is not None
        return startup_cleanup_workers.CleanupWorkerHandles()

    async def _fake_shutdown_transition_handoff(**kwargs):
        recorded_calls.append(kwargs)
        kwargs["apply_shutdown_transition_gate"](kwargs["app"], kwargs["readiness_state"])
        return transition_handoff.TransitionHandoffHandles(
            legacy_shutdown_plan=[],
            transition_gate_applied=False,
        )

    monkeypatch.setattr(
        startup_auxiliary_services,
        "start_auxiliary_services",
        _fake_start_auxiliary_services,
    )
    monkeypatch.setattr(
        startup_cleanup_workers,
        "start_cleanup_workers",
        _fake_start_cleanup_workers,
    )
    monkeypatch.setattr(
        transition_handoff,
        "shutdown_transition_handoff",
        _fake_shutdown_transition_handoff,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["readiness_state"] is main_module.READINESS_STATE
    assert "usage_task" not in recorded_calls[0]
    assert "llm_usage_task" not in recorded_calls[0]
    assert "authnz_scheduler_started" not in recorded_calls[0]
    assert recorded_calls[0]["build_legacy_shutdown_context"] is main_module._build_legacy_shutdown_context
    assert recorded_calls[0]["apply_shutdown_transition_gate"] is main_module._apply_shutdown_transition_gate
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_shutdown_delegates_coordinated_legacy_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_coordinated_legacy_components as coordinated_legacy
    from tldw_Server_API.app.services import shutdown_transition_handoff as transition_handoff

    app = main_module.app
    fake_legacy_plan = [object(), object()]
    recorded_calls: list[dict[str, object]] = []

    async def _fake_shutdown_transition_handoff(**kwargs):
        kwargs["apply_shutdown_transition_gate"](kwargs["app"], kwargs["readiness_state"])
        return transition_handoff.TransitionHandoffHandles(
            legacy_shutdown_plan=fake_legacy_plan,
            transition_gate_applied=False,
        )

    async def _fake_run_shutdown_coordinated_legacy_components(**kwargs):
        recorded_calls.append(kwargs)
        return coordinated_legacy.CoordinatedLegacyShutdownHandles(
            coordinated_legacy_component_names=set(),
        )

    monkeypatch.setattr(
        transition_handoff,
        "shutdown_transition_handoff",
        _fake_shutdown_transition_handoff,
    )
    monkeypatch.setattr(
        coordinated_legacy,
        "run_shutdown_coordinated_legacy_components",
        _fake_run_shutdown_coordinated_legacy_components,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert recorded_calls[0]["legacy_shutdown_plan"] is fake_legacy_plan
    assert recorded_calls[0]["run_coordinated_shutdown"] is main_module._run_coordinated_shutdown
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS


@pytest.mark.integration
def test_lifespan_shutdown_delegates_job_poller_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_coordinated_legacy_components as coordinated_legacy
    from tldw_Server_API.app.services import shutdown_core_jobs_worker as core_jobs_shutdown
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as job_poller_handoff
    from tldw_Server_API.app.services import shutdown_transition_handoff as transition_handoff

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []
    core_shutdown_calls: list[dict[str, object]] = []

    async def _fake_shutdown_transition_handoff(**kwargs):
        kwargs["apply_shutdown_transition_gate"](kwargs["app"], kwargs["readiness_state"])
        return transition_handoff.TransitionHandoffHandles(
            legacy_shutdown_plan=[],
            transition_gate_applied=False,
        )

    async def _fake_run_shutdown_job_poller_handoff(**kwargs):
        recorded_calls.append(kwargs)
        return job_poller_handoff.JobPollerShutdownHandoffHandles(
            early_quiesced_job_poller_names={"core_jobs_task"},
            should_run_late_stop=lambda task_name, task: bool(task) and task_name != "core_jobs_task",
        )

    async def _fake_shutdown_coordinated_legacy_components(**kwargs):
        return coordinated_legacy.CoordinatedLegacyShutdownHandles(
            coordinated_legacy_component_names=set(),
        )

    async def _fake_shutdown_core_jobs_worker(**kwargs):
        core_shutdown_calls.append(kwargs)
        return core_jobs_shutdown.CoreJobsShutdownHandles(
            core_jobs_task=kwargs["core_jobs_task"],
            core_jobs_stop_event=kwargs["core_jobs_stop_event"],
        )

    monkeypatch.setattr(
        transition_handoff,
        "shutdown_transition_handoff",
        _fake_shutdown_transition_handoff,
    )
    monkeypatch.setattr(
        job_poller_handoff,
        "run_shutdown_job_poller_handoff",
        _fake_run_shutdown_job_poller_handoff,
    )
    monkeypatch.setattr(
        coordinated_legacy,
        "shutdown_coordinated_legacy_components",
        _fake_shutdown_coordinated_legacy_components,
    )
    monkeypatch.setattr(
        core_jobs_shutdown,
        "shutdown_core_jobs_worker",
        _fake_shutdown_core_jobs_worker,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert isinstance(recorded_calls[0]["owned_job_pollers"], list)
    assert recorded_calls[0]["quiesce_owned_job_pollers_for_shutdown"] is (
        main_module._quiesce_owned_job_pollers_for_shutdown
    )
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS
    assert len(core_shutdown_calls) == 1
    should_run_late_stop = core_shutdown_calls[0]["should_run_late_stop"]
    late_stop_task = object()
    assert should_run_late_stop("core_jobs_task", late_stop_task) is False
    assert should_run_late_stop("files_jobs_task", late_stop_task) is True


@pytest.mark.integration
def test_lifespan_shutdown_delegates_primary_late_stop_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_coordinated_legacy_components as coordinated_legacy
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as job_poller_handoff
    from tldw_Server_API.app.services import shutdown_primary_late_stop_workers as primary_workers
    from tldw_Server_API.app.services import shutdown_transition_handoff as transition_handoff

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def should_run_late_stop(task_name, task):
        return bool(task) and task_name != "core_jobs_task"

    async def _fake_shutdown_transition_handoff(**kwargs):
        kwargs["apply_shutdown_transition_gate"](kwargs["app"], kwargs["readiness_state"])
        return transition_handoff.TransitionHandoffHandles(
            legacy_shutdown_plan=[],
            transition_gate_applied=False,
        )

    async def _fake_shutdown_job_poller_handoff(**kwargs):
        return job_poller_handoff.JobPollerShutdownHandoffHandles(
            early_quiesced_job_poller_names={"core_jobs_task"},
            should_run_late_stop=should_run_late_stop,
        )

    async def _fake_shutdown_coordinated_legacy_components(**kwargs):
        return coordinated_legacy.CoordinatedLegacyShutdownHandles(
            coordinated_legacy_component_names=set(),
        )

    async def _fake_run_shutdown_primary_late_stop_workers(**kwargs):
        recorded_calls.append(kwargs)
        return primary_workers.PrimaryLateStopWorkerHandles(
            core_jobs_task=kwargs["core_jobs_task"],
            core_jobs_stop_event=kwargs["core_jobs_stop_event"],
            files_jobs_task=kwargs["files_jobs_task"],
            files_jobs_stop_event=kwargs["files_jobs_stop_event"],
            data_tables_jobs_task=kwargs["data_tables_jobs_task"],
            data_tables_jobs_stop_event=kwargs["data_tables_jobs_stop_event"],
            prompt_studio_jobs_task=kwargs["prompt_studio_jobs_task"],
            prompt_studio_jobs_stop_event=kwargs["prompt_studio_jobs_stop_event"],
            privilege_snapshot_task=kwargs["privilege_snapshot_task"],
            privilege_snapshot_stop_event=kwargs["privilege_snapshot_stop_event"],
            audio_jobs_task=kwargs["audio_jobs_task"],
            audio_jobs_stop_event=kwargs["audio_jobs_stop_event"],
            presentation_render_jobs_task=kwargs["presentation_render_jobs_task"],
            presentation_render_jobs_stop_event=kwargs["presentation_render_jobs_stop_event"],
        )

    monkeypatch.setattr(
        transition_handoff,
        "shutdown_transition_handoff",
        _fake_shutdown_transition_handoff,
    )
    monkeypatch.setattr(
        job_poller_handoff,
        "shutdown_job_poller_handoff",
        _fake_shutdown_job_poller_handoff,
    )
    monkeypatch.setattr(
        coordinated_legacy,
        "shutdown_coordinated_legacy_components",
        _fake_shutdown_coordinated_legacy_components,
    )
    monkeypatch.setattr(
        primary_workers,
        "run_shutdown_primary_late_stop_workers",
        _fake_run_shutdown_primary_late_stop_workers,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["should_run_late_stop"] is should_run_late_stop
    assert recorded_calls[0]["guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert "core_jobs_task" in recorded_calls[0]
    assert "files_jobs_task" in recorded_calls[0]
    assert "data_tables_jobs_task" in recorded_calls[0]
    assert "prompt_studio_jobs_task" in recorded_calls[0]
    assert "privilege_snapshot_task" in recorded_calls[0]
    assert "audio_jobs_task" in recorded_calls[0]
    assert "presentation_render_jobs_task" in recorded_calls[0]


@pytest.mark.integration
def test_lifespan_shutdown_delegates_grouped_late_stop_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_coordinated_legacy_components as coordinated_legacy
    from tldw_Server_API.app.services import shutdown_grouped_late_stop_workers as grouped_workers
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as job_poller_handoff
    from tldw_Server_API.app.services import shutdown_primary_late_stop_workers as primary_workers
    from tldw_Server_API.app.services import shutdown_transition_handoff as transition_handoff

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def should_run_late_stop(task_name, task):
        return bool(task) and task_name != "core_jobs_task"

    async def _fake_shutdown_transition_handoff(**kwargs):
        kwargs["apply_shutdown_transition_gate"](kwargs["app"], kwargs["readiness_state"])
        return transition_handoff.TransitionHandoffHandles(
            legacy_shutdown_plan=[],
            transition_gate_applied=False,
        )

    async def _fake_shutdown_job_poller_handoff(**kwargs):
        return job_poller_handoff.JobPollerShutdownHandoffHandles(
            early_quiesced_job_poller_names={"core_jobs_task"},
            should_run_late_stop=should_run_late_stop,
        )

    async def _fake_shutdown_coordinated_legacy_components(**kwargs):
        return coordinated_legacy.CoordinatedLegacyShutdownHandles(
            coordinated_legacy_component_names=set(),
        )

    async def _fake_shutdown_primary_late_stop_workers(**kwargs):
        return primary_workers.PrimaryLateStopWorkerHandles(
            core_jobs_task=kwargs["core_jobs_task"],
            core_jobs_stop_event=kwargs["core_jobs_stop_event"],
            files_jobs_task=kwargs["files_jobs_task"],
            files_jobs_stop_event=kwargs["files_jobs_stop_event"],
            data_tables_jobs_task=kwargs["data_tables_jobs_task"],
            data_tables_jobs_stop_event=kwargs["data_tables_jobs_stop_event"],
            prompt_studio_jobs_task=kwargs["prompt_studio_jobs_task"],
            prompt_studio_jobs_stop_event=kwargs["prompt_studio_jobs_stop_event"],
            privilege_snapshot_task=kwargs["privilege_snapshot_task"],
            privilege_snapshot_stop_event=kwargs["privilege_snapshot_stop_event"],
            audio_jobs_task=kwargs["audio_jobs_task"],
            audio_jobs_stop_event=kwargs["audio_jobs_stop_event"],
            presentation_render_jobs_task=kwargs["presentation_render_jobs_task"],
            presentation_render_jobs_stop_event=kwargs["presentation_render_jobs_stop_event"],
        )

    async def _fake_run_shutdown_grouped_late_stop_workers(**kwargs):
        recorded_calls.append(kwargs)
        return grouped_workers.GroupedLateStopWorkerHandles(
            media_ingest_jobs_task=kwargs["media_ingest_jobs_task"],
            media_ingest_jobs_stop_event=kwargs["media_ingest_jobs_stop_event"],
            media_ingest_heavy_jobs_task=kwargs["media_ingest_heavy_jobs_task"],
            media_ingest_heavy_jobs_stop_event=kwargs["media_ingest_heavy_jobs_stop_event"],
            reading_digest_jobs_task=kwargs["reading_digest_jobs_task"],
            reading_digest_jobs_stop_event=kwargs["reading_digest_jobs_stop_event"],
            study_pack_jobs_task=kwargs["study_pack_jobs_task"],
            study_pack_jobs_stop_event=kwargs["study_pack_jobs_stop_event"],
            study_suggestions_jobs_task=kwargs["study_suggestions_jobs_task"],
            study_suggestions_jobs_stop_event=kwargs["study_suggestions_jobs_stop_event"],
            companion_reflection_jobs_task=kwargs["companion_reflection_jobs_task"],
            companion_reflection_jobs_stop_event=kwargs["companion_reflection_jobs_stop_event"],
            reminder_jobs_task=kwargs["reminder_jobs_task"],
            admin_backup_jobs_task=kwargs["admin_backup_jobs_task"],
            admin_maintenance_rotation_jobs_task=kwargs["admin_maintenance_rotation_jobs_task"],
            admin_maintenance_rotation_jobs_stop_event=kwargs["admin_maintenance_rotation_jobs_stop_event"],
            recipe_run_jobs_task=kwargs["recipe_run_jobs_task"],
            recipe_run_jobs_stop_event=kwargs["recipe_run_jobs_stop_event"],
            evals_abtest_jobs_task=kwargs["evals_abtest_jobs_task"],
            evals_abtest_jobs_stop_event=kwargs["evals_abtest_jobs_stop_event"],
        )

    monkeypatch.setattr(
        transition_handoff,
        "shutdown_transition_handoff",
        _fake_shutdown_transition_handoff,
    )
    monkeypatch.setattr(
        job_poller_handoff,
        "shutdown_job_poller_handoff",
        _fake_shutdown_job_poller_handoff,
    )
    monkeypatch.setattr(
        coordinated_legacy,
        "shutdown_coordinated_legacy_components",
        _fake_shutdown_coordinated_legacy_components,
    )
    monkeypatch.setattr(
        primary_workers,
        "shutdown_primary_late_stop_workers",
        _fake_shutdown_primary_late_stop_workers,
    )
    monkeypatch.setattr(
        grouped_workers,
        "run_shutdown_grouped_late_stop_workers",
        _fake_run_shutdown_grouped_late_stop_workers,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["should_run_late_stop"] is should_run_late_stop
    assert recorded_calls[0]["guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert "media_ingest_jobs_task" in recorded_calls[0]
    assert "media_ingest_heavy_jobs_task" in recorded_calls[0]
    assert "reading_digest_jobs_task" in recorded_calls[0]
    assert "study_pack_jobs_task" in recorded_calls[0]
    assert "study_suggestions_jobs_task" in recorded_calls[0]
    assert "companion_reflection_jobs_task" in recorded_calls[0]
    assert "reminder_jobs_task" in recorded_calls[0]
    assert "admin_backup_jobs_task" in recorded_calls[0]
    assert "admin_maintenance_rotation_jobs_task" in recorded_calls[0]
    assert "recipe_run_jobs_task" in recorded_calls[0]
    assert "evals_abtest_jobs_task" in recorded_calls[0]


@pytest.mark.integration
def test_lifespan_shutdown_delegates_post_worker_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_coordinated_legacy_components as coordinated_legacy
    from tldw_Server_API.app.services import shutdown_grouped_late_stop_workers as grouped_workers
    from tldw_Server_API.app.services import shutdown_job_poller_handoff as job_poller_handoff
    from tldw_Server_API.app.services import shutdown_post_worker_services as post_worker_services
    from tldw_Server_API.app.services import shutdown_primary_late_stop_workers as primary_workers
    from tldw_Server_API.app.services import shutdown_transition_handoff as transition_handoff

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []

    def should_run_late_stop(task_name, task):
        return bool(task) and task_name != "core_jobs_task"

    async def _fake_shutdown_transition_handoff(**kwargs):
        kwargs["apply_shutdown_transition_gate"](kwargs["app"], kwargs["readiness_state"])
        return transition_handoff.TransitionHandoffHandles(
            legacy_shutdown_plan=[],
            transition_gate_applied=False,
        )

    async def _fake_shutdown_job_poller_handoff(**kwargs):
        return job_poller_handoff.JobPollerShutdownHandoffHandles(
            early_quiesced_job_poller_names={"core_jobs_task"},
            should_run_late_stop=should_run_late_stop,
        )

    async def _fake_shutdown_coordinated_legacy_components(**kwargs):
        return coordinated_legacy.CoordinatedLegacyShutdownHandles(
            coordinated_legacy_component_names=set(),
        )

    async def _fake_shutdown_primary_late_stop_workers(**kwargs):
        return primary_workers.PrimaryLateStopWorkerHandles(
            core_jobs_task=kwargs["core_jobs_task"],
            core_jobs_stop_event=kwargs["core_jobs_stop_event"],
            files_jobs_task=kwargs["files_jobs_task"],
            files_jobs_stop_event=kwargs["files_jobs_stop_event"],
            data_tables_jobs_task=kwargs["data_tables_jobs_task"],
            data_tables_jobs_stop_event=kwargs["data_tables_jobs_stop_event"],
            prompt_studio_jobs_task=kwargs["prompt_studio_jobs_task"],
            prompt_studio_jobs_stop_event=kwargs["prompt_studio_jobs_stop_event"],
            privilege_snapshot_task=kwargs["privilege_snapshot_task"],
            privilege_snapshot_stop_event=kwargs["privilege_snapshot_stop_event"],
            audio_jobs_task=kwargs["audio_jobs_task"],
            audio_jobs_stop_event=kwargs["audio_jobs_stop_event"],
            presentation_render_jobs_task=kwargs["presentation_render_jobs_task"],
            presentation_render_jobs_stop_event=kwargs["presentation_render_jobs_stop_event"],
        )

    async def _fake_shutdown_grouped_late_stop_workers(**kwargs):
        return grouped_workers.GroupedLateStopWorkerHandles(
            media_ingest_jobs_task=kwargs["media_ingest_jobs_task"],
            media_ingest_jobs_stop_event=kwargs["media_ingest_jobs_stop_event"],
            media_ingest_heavy_jobs_task=kwargs["media_ingest_heavy_jobs_task"],
            media_ingest_heavy_jobs_stop_event=kwargs["media_ingest_heavy_jobs_stop_event"],
            reading_digest_jobs_task=kwargs["reading_digest_jobs_task"],
            reading_digest_jobs_stop_event=kwargs["reading_digest_jobs_stop_event"],
            study_pack_jobs_task=kwargs["study_pack_jobs_task"],
            study_pack_jobs_stop_event=kwargs["study_pack_jobs_stop_event"],
            study_suggestions_jobs_task=kwargs["study_suggestions_jobs_task"],
            study_suggestions_jobs_stop_event=kwargs["study_suggestions_jobs_stop_event"],
            companion_reflection_jobs_task=kwargs["companion_reflection_jobs_task"],
            companion_reflection_jobs_stop_event=kwargs["companion_reflection_jobs_stop_event"],
            reminder_jobs_task=kwargs["reminder_jobs_task"],
            admin_backup_jobs_task=kwargs["admin_backup_jobs_task"],
            admin_maintenance_rotation_jobs_task=kwargs["admin_maintenance_rotation_jobs_task"],
            admin_maintenance_rotation_jobs_stop_event=kwargs["admin_maintenance_rotation_jobs_stop_event"],
            recipe_run_jobs_task=kwargs["recipe_run_jobs_task"],
            recipe_run_jobs_stop_event=kwargs["recipe_run_jobs_stop_event"],
            evals_abtest_jobs_task=kwargs["evals_abtest_jobs_task"],
            evals_abtest_jobs_stop_event=kwargs["evals_abtest_jobs_stop_event"],
        )

    async def _fake_run_shutdown_post_worker_services(**kwargs):
        recorded_calls.append(kwargs)
        return post_worker_services.PostWorkerShutdownHandles(
            jobs_notifications_bridge_task=kwargs["jobs_notifications_bridge_task"],
            jobs_metrics_task=kwargs["jobs_metrics_task"],
            loop_lag_task=kwargs["loop_lag_task"],
            jobs_metrics_reconcile_task=kwargs["jobs_metrics_reconcile_task"],
            jobs_metrics_reconcile_stop=kwargs["jobs_metrics_reconcile_stop"],
            jobs_crypto_rotate_task=kwargs["jobs_crypto_rotate_task"],
            jobs_integrity_task=kwargs["jobs_integrity_task"],
            jobs_webhooks_task=kwargs["jobs_webhooks_task"],
            meetings_webhook_dlq_task=kwargs["meetings_webhook_dlq_task"],
            workflows_dlq_task=kwargs["workflows_dlq_task"],
            workflows_gc_task=kwargs["workflows_gc_task"],
            workflows_maint_task=kwargs["workflows_maint_task"],
        )

    monkeypatch.setattr(
        transition_handoff,
        "shutdown_transition_handoff",
        _fake_shutdown_transition_handoff,
    )
    monkeypatch.setattr(
        job_poller_handoff,
        "shutdown_job_poller_handoff",
        _fake_shutdown_job_poller_handoff,
    )
    monkeypatch.setattr(
        coordinated_legacy,
        "shutdown_coordinated_legacy_components",
        _fake_shutdown_coordinated_legacy_components,
    )
    monkeypatch.setattr(
        primary_workers,
        "shutdown_primary_late_stop_workers",
        _fake_shutdown_primary_late_stop_workers,
    )
    monkeypatch.setattr(
        grouped_workers,
        "shutdown_grouped_late_stop_workers",
        _fake_shutdown_grouped_late_stop_workers,
    )
    monkeypatch.setattr(
        post_worker_services,
        "run_shutdown_post_worker_services",
        _fake_run_shutdown_post_worker_services,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert "coordinated_legacy_component_names" not in recorded_calls[0]
    assert "claims_task" not in recorded_calls[0]
    assert "embeddings_compactor_task" not in recorded_calls[0]
    assert "embeddings_compactor_stop_event" not in recorded_calls[0]
    assert "websub_renewal_task" not in recorded_calls[0]
    assert "usage_task" not in recorded_calls[0]
    assert "llm_usage_task" not in recorded_calls[0]
    assert "jobs_prune_task" not in recorded_calls[0]
    assert "files_export_gc_task" not in recorded_calls[0]
    assert "notifications_prune_task" not in recorded_calls[0]
    assert "jobs_notifications_bridge_task" in recorded_calls[0]
    assert "workflows_sched_task" not in recorded_calls[0]
    assert "reading_digest_sched_task" not in recorded_calls[0]
    assert "admin_backup_sched_task" not in recorded_calls[0]
    assert "companion_reflection_sched_task" not in recorded_calls[0]
    assert "reminders_sched_task" not in recorded_calls[0]
    assert "connectors_sync_sched_task" not in recorded_calls[0]
    assert "jobs_metrics_task" in recorded_calls[0]
    assert "jobs_metrics_reconcile_task" in recorded_calls[0]
    assert "jobs_crypto_rotate_task" in recorded_calls[0]
    assert "workflows_maint_task" in recorded_calls[0]


@pytest.mark.integration
def test_lifespan_shutdown_delegates_final_cleanup_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import shutdown_final_cleanup_tail as cleanup_tail

    app = main_module.app
    recorded_calls: list[dict[str, object]] = []
    original_shutdown_final_cleanup_tail = cleanup_tail.shutdown_final_cleanup_tail

    async def _fake_shutdown_final_cleanup_tail(**kwargs):
        recorded_calls.append(kwargs)
        return await original_shutdown_final_cleanup_tail(**kwargs)

    monkeypatch.setattr(
        cleanup_tail,
        "shutdown_final_cleanup_tail",
        _fake_shutdown_final_cleanup_tail,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["app"] is app
    assert "authnz_scheduler_started" not in recorded_calls[0]
    assert "stopped_background_worker_names" not in recorded_calls[0]
    assert "coordinated_legacy_component_names" not in recorded_calls[0]
    assert recorded_calls[0]["in_pytest_for_db_pool_shutdown"] is True
    assert recorded_calls[0]["in_pytest_for_tts_shutdown"] is True
    assert recorded_calls[0]["timed_shutdown_segment"] is main_module._timed_shutdown_segment
    assert "test_db_instance_ref" in recorded_calls[0]
    assert recorded_calls[0]["import_exceptions"] == main_module._IMPORT_EXCEPTIONS
    assert recorded_calls[0]["startup_guard_exceptions"] == main_module._STARTUP_GUARD_EXCEPTIONS
    assert "db_pool" in recorded_calls[0]
    assert "session_manager" in recorded_calls[0]
    assert "heavy_startup_handles" in recorded_calls[0]


@pytest.mark.integration
def test_lifespan_shutdown_stops_files_jobs_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_primary_jobs_pollers

    app = main_module.app
    observed = {
        "files_stop_event": None,
        "files_stopped": False,
    }

    monkeypatch.setenv("FILES_JOBS_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_files_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="files_stop_event",
            stopped_key="files_stopped",
        )

    monkeypatch.setattr(
        startup_primary_jobs_pollers,
        "_run_file_artifacts_jobs_worker_service",
        _fake_files_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["files_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["files_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["files_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_core_jobs_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_primary_jobs_pollers

    app = main_module.app
    observed = {
        "core_stop_event": None,
        "core_stopped": False,
    }

    monkeypatch.setenv("CHATBOOKS_JOBS_BACKEND", "core")
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_core_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="core_stop_event",
            stopped_key="core_stopped",
        )

    monkeypatch.setattr(
        startup_primary_jobs_pollers,
        "_run_chatbooks_core_jobs_worker_service",
        _fake_core_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["core_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["core_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["core_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_prompt_studio_jobs_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_primary_jobs_pollers

    app = main_module.app
    observed = {
        "prompt_studio_stop_event": None,
        "prompt_studio_stopped": False,
    }

    monkeypatch.setenv("PROMPT_STUDIO_JOBS_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_prompt_studio_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="prompt_studio_stop_event",
            stopped_key="prompt_studio_stopped",
        )

    monkeypatch.setattr(
        startup_primary_jobs_pollers,
        "_run_prompt_studio_jobs_worker_service",
        _fake_prompt_studio_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["prompt_studio_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["prompt_studio_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["prompt_studio_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_data_tables_jobs_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_primary_jobs_pollers

    app = main_module.app
    observed = {
        "data_tables_stop_event": None,
        "data_tables_stopped": False,
    }

    monkeypatch.setenv("DATA_TABLES_JOBS_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_data_tables_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="data_tables_stop_event",
            stopped_key="data_tables_stopped",
        )

    monkeypatch.setattr(
        startup_primary_jobs_pollers,
        "_run_data_tables_jobs_worker_service",
        _fake_data_tables_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["data_tables_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["data_tables_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["data_tables_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_audio_jobs_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_content_jobs_pollers

    app = main_module.app
    observed = {
        "audio_stop_event": None,
        "audio_stopped": False,
    }

    monkeypatch.setenv("AUDIO_JOBS_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_audio_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="audio_stop_event",
            stopped_key="audio_stopped",
        )

    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_run_audio_jobs_worker_service",
        _fake_audio_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["audio_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["audio_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["audio_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_presentation_render_shutdown_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_content_jobs_pollers

    app = main_module.app
    observed = {
        "presentation_stop_event": None,
        "presentation_stopped": False,
    }

    monkeypatch.setenv("PRESENTATION_RENDER_JOBS_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_presentation_render_jobs_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="presentation_stop_event",
            stopped_key="presentation_stopped",
        )

    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_run_presentation_render_jobs_worker_service",
        _fake_presentation_render_jobs_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["presentation_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["presentation_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["presentation_stopped"] is True


@pytest.mark.integration
def test_lifespan_shutdown_stops_privilege_snapshot_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as main_module
    from tldw_Server_API.app.services import startup_study_privilege_jobs_pollers

    app = main_module.app
    observed = {
        "privilege_stop_event": None,
        "privilege_stopped": False,
    }

    monkeypatch.setenv("PRIVILEGE_SNAPSHOT_WORKER_ENABLED", "1")

    async def _wait_for_stop(stop_event, *, event_key: str, stopped_key: str) -> None:
        observed[event_key] = stop_event
        await stop_event.wait()
        observed[stopped_key] = True

    def _fake_privilege_snapshot_worker_service(stop_event):
        return _wait_for_stop(
            stop_event,
            event_key="privilege_stop_event",
            stopped_key="privilege_stopped",
        )

    monkeypatch.setattr(
        startup_study_privilege_jobs_pollers,
        "_run_privilege_snapshot_worker_service",
        _fake_privilege_snapshot_worker_service,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        stop_event = observed["privilege_stop_event"]
        assert stop_event is not None
        assert stop_event.is_set() is False

    stop_event = observed["privilege_stop_event"]
    assert stop_event is not None
    assert stop_event.is_set() is True
    assert observed["privilege_stopped"] is True


@pytest.mark.integration
def test_shutdown_falls_back_to_direct_drain_when_transition_gate_component_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.services import shutdown_legacy_adapters
    from tldw_Server_API.app.services.shutdown_models import (
        ShutdownComponent,
        ShutdownPhase,
        ShutdownPolicy,
    )

    if hasattr(app.state, "_tldw_lifecycle_events"):
        delattr(app.state, "_tldw_lifecycle_events")
    if hasattr(app.state, "_tldw_lifecycle_state"):
        delattr(app.state, "_tldw_lifecycle_state")

    gate_calls: list[bool] = []

    def _record_gate(cls, enabled: bool) -> None:
        gate_calls.append(enabled)

    monkeypatch.setattr(
        JobManager,
        "set_acquire_gate",
        classmethod(_record_gate),
    )

    def _failing_transition_stop() -> None:
        raise RuntimeError("shadow transition component failed")

    monkeypatch.setattr(
        shutdown_legacy_adapters,
        "build_legacy_shutdown_plan",
        lambda *_args, **_kwargs: [
            ShutdownComponent(
                name="lifecycle_gate",
                phase=ShutdownPhase.TRANSITION,
                policy=ShutdownPolicy.DEV_FAST,
                default_timeout_ms=1000,
                stop=_failing_transition_stop,
            )
        ],
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app.state._tldw_lifecycle_state.phase == "ready"
        assert app.state._tldw_lifecycle_state.ready is True
        gate_calls.clear()

    assert gate_calls == [True, False]
    assert app.state._tldw_lifecycle_state.phase == "draining"
    assert app.state._tldw_lifecycle_state.ready is False
    assert app.state._tldw_lifecycle_state.draining is True


@pytest.mark.integration
def test_shutdown_migrated_legacy_slice_uses_prod_drain_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from tldw_Server_API.app.services import shutdown_coordinator as shutdown_coordinator_module
    from tldw_Server_API.app.services import shutdown_legacy_adapters
    from tldw_Server_API.app.services.shutdown_models import (
        ShutdownComponent,
        ShutdownComponentSummary,
        ShutdownPhase,
        ShutdownPhaseSummary,
        ShutdownPolicy,
        ShutdownSummary,
    )

    class _SpyShutdownCoordinator:
        created_profiles: list[str] = []
        instances: list[_SpyShutdownCoordinator] = []

        def __init__(self, profile: str = "dev_fast", **_kwargs) -> None:
            self.profile = profile
            self.registered: list[ShutdownComponent] = []
            type(self).created_profiles.append(profile)
            type(self).instances.append(self)

        def register(self, component: ShutdownComponent) -> ShutdownComponent:
            self.registered.append(component)
            return component

        async def shutdown(self) -> ShutdownSummary:
            phase_names: dict[ShutdownPhase, list[str]] = {}
            component_summaries: dict[str, ShutdownComponentSummary] = {}

            for component in self.registered:
                phase_names.setdefault(component.phase, []).append(component.name)
                component_summaries[component.name] = ShutdownComponentSummary(
                    name=component.name,
                    phase=component.phase,
                    policy=component.policy,
                    result="stopped",
                    started_at=0.0,
                    finished_at=0.0,
                    duration_ms=0,
                    timeout_ms=0,
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
                for phase, component_names in phase_names.items()
            }
            return ShutdownSummary(
                profile=self.profile,
                started_at=0.0,
                finished_at=0.0,
                deadline_at=0.0,
                hard_cutoff_at=0.0,
                wall_time_ms=0,
                soft_overrun_used_ms=0,
                components=component_summaries,
                phases=phase_summaries,
            )

    captured_contexts: list[object] = []

    def _fake_build_legacy_shutdown_plan(_app, _context):
        if _context is not None:
            captured_contexts.append(_context)
        return [
            ShutdownComponent(
                name="lifecycle_gate",
                phase=ShutdownPhase.TRANSITION,
                policy=ShutdownPolicy.DEV_FAST,
                default_timeout_ms=1000,
                stop=lambda: None,
            ),
            ShutdownComponent(
                name="chatbooks_cleanup",
                phase=ShutdownPhase.WORKERS,
                policy=ShutdownPolicy.BEST_EFFORT,
                default_timeout_ms=1000,
                stop=lambda: None,
            ),
            ShutdownComponent(
                name="usage_aggregator",
                phase=ShutdownPhase.RESOURCES,
                policy=ShutdownPolicy.BEST_EFFORT,
                default_timeout_ms=1000,
                stop=lambda: None,
            ),
            ShutdownComponent(
                name="storage_cleanup_service",
                phase=ShutdownPhase.FINALIZERS,
                policy=ShutdownPolicy.PROD_DRAIN,
                default_timeout_ms=5000,
                stop=lambda: None,
            ),
        ]

    fake_shutdown_legacy_adapters = types.ModuleType("tldw_Server_API.app.services.shutdown_legacy_adapters")
    fake_shutdown_legacy_adapters.LegacyShutdownContext = shutdown_legacy_adapters.LegacyShutdownContext
    fake_shutdown_legacy_adapters.build_legacy_shutdown_plan = _fake_build_legacy_shutdown_plan
    fake_shutdown_legacy_adapters.register_legacy_shutdown_components = (
        shutdown_legacy_adapters.register_legacy_shutdown_components
    )
    fake_shutdown_legacy_adapters.get_legacy_shutdown_suppressed_component_names = (
        shutdown_legacy_adapters.get_legacy_shutdown_suppressed_component_names
    )
    monkeypatch.setitem(sys.modules, fake_shutdown_legacy_adapters.__name__, fake_shutdown_legacy_adapters)

    monkeypatch.setattr(shutdown_coordinator_module, "ShutdownCoordinator", _SpyShutdownCoordinator)

    from tldw_Server_API.app.main import app

    if hasattr(app.state, "_tldw_lifecycle_events"):
        delattr(app.state, "_tldw_lifecycle_events")
    if hasattr(app.state, "_tldw_lifecycle_state"):
        delattr(app.state, "_tldw_lifecycle_state")

    expected_migrated_names: list[str] = []

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert _SpyShutdownCoordinator.created_profiles == ["dev_fast", "prod_drain"]
    assert len(captured_contexts) == 1
    assert getattr(captured_contexts[0], "readiness_state", None) is not None
    assert not hasattr(captured_contexts[0], "usage_task")
    assert not hasattr(captured_contexts[0], "llm_usage_task")
    assert not hasattr(captured_contexts[0], "authnz_scheduler_started")
    assert [component.name for component in _SpyShutdownCoordinator.instances[0].registered] == [
        "lifecycle_gate",
    ]
    expected_transport_names = getattr(app.state, "_tldw_shutdown_transport_component_names", [])
    migrated_registered_names = [component.name for component in _SpyShutdownCoordinator.instances[1].registered]
    assert migrated_registered_names == expected_migrated_names + expected_transport_names
    assert "lifecycle_gate" not in migrated_registered_names


@pytest.mark.integration
def test_lifespan_exposes_openapi_after_startup(client_user_only) -> None:
    response = client_user_only.get("/openapi.json")
    assert response.status_code == 200
    payload = response.json()
    assert "paths" in payload
