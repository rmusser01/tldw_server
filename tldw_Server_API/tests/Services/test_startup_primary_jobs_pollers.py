from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_startup_primary_jobs_pollers():
    sys.modules.pop("tldw_Server_API.app.services.startup_primary_jobs_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_primary_jobs_pollers")


def _context(
    *,
    settings: dict[str, object] | None = None,
    route_enabled: Callable[..., bool] | None = None,
) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app="app",
        settings=settings or {},
        test_mode=True,
        route_enabled=route_enabled or (lambda *_args, **_kwargs: True),
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _specs_by_name(startup_pollers: Any) -> dict[str, Any]:
    return {
        spec.name: spec
        for spec in startup_pollers.provide_primary_jobs_worker_specs()
    }


@pytest.mark.parametrize(
    "spec_name",
    [
        "core_jobs_task",
        "files_jobs_task",
        "data_tables_jobs_task",
        "prompt_studio_jobs_task",
        "workspace_file_inventory_jobs_task",
    ],
)
def test_primary_jobs_worker_specs_match_legacy_worker_contract(
    spec_name: str,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()

    spec = _specs_by_name(startup_pollers)[spec_name]

    assert spec.task_name == spec_name
    assert spec.category == "jobs"
    assert spec.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert spec.timeout_sec == 5.0
    assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert spec.factory is not None
    assert callable(spec.factory)


def test_primary_jobs_worker_specs_use_expected_names() -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()

    assert [spec.name for spec in startup_pollers.provide_primary_jobs_worker_specs()] == [
        "core_jobs_task",
        "files_jobs_task",
        "data_tables_jobs_task",
        "prompt_studio_jobs_task",
        "workspace_file_inventory_jobs_task",
    ]


def test_primary_jobs_worker_spec_factories_delegate_to_existing_worker_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    calls: list[tuple[str, object]] = []

    for spec_name, factory_name in [
        ("core_jobs_task", "_run_chatbooks_core_jobs_worker_service"),
        ("files_jobs_task", "_run_file_artifacts_jobs_worker_service"),
        ("data_tables_jobs_task", "_run_data_tables_jobs_worker_service"),
        ("prompt_studio_jobs_task", "_run_prompt_studio_jobs_worker_service"),
        (
            "workspace_file_inventory_jobs_task",
            "_run_workspace_file_inventory_jobs_worker_service",
        ),
    ]:
        monkeypatch.setattr(
            startup_pollers,
            factory_name,
            lambda stop_event, name=spec_name: calls.append((name, stop_event)) or f"{name}-awaitable",
        )

    specs = _specs_by_name(startup_pollers)

    for spec_name, spec in specs.items():
        assert spec.factory is not None
        assert spec.factory(_context(), f"{spec_name}-stop") == f"{spec_name}-awaitable"

    assert calls == [
        ("core_jobs_task", "core_jobs_task-stop"),
        ("files_jobs_task", "files_jobs_task-stop"),
        ("data_tables_jobs_task", "data_tables_jobs_task-stop"),
        ("prompt_studio_jobs_task", "prompt_studio_jobs_task-stop"),
        (
            "workspace_file_inventory_jobs_task",
            "workspace_file_inventory_jobs_task-stop",
        ),
    ]


def test_primary_jobs_worker_spec_predicates_use_route_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _route_enabled(*args: object, **kwargs: object) -> bool:
        calls.append((args, kwargs))
        return False

    context = _context(route_enabled=_route_enabled)
    specs = _specs_by_name(startup_pollers)
    for env_key in [
        "FILES_JOBS_WORKER_ENABLED",
        "DATA_TABLES_JOBS_WORKER_ENABLED",
        "PROMPT_STUDIO_JOBS_WORKER_ENABLED",
        "WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED",
    ]:
        monkeypatch.setenv(env_key, "true")

    assert specs["files_jobs_task"].enabled(context) is False
    assert specs["data_tables_jobs_task"].enabled(context) is False
    assert specs["prompt_studio_jobs_task"].enabled(context) is False
    assert specs["workspace_file_inventory_jobs_task"].enabled(context) is False
    assert calls == [
        (("files",), {}),
        (("data-tables",), {}),
        (("prompt-studio",), {}),
        (("workspaces",), {}),
    ]


@pytest.mark.parametrize(
    ("env", "settings", "expected"),
    [
        ({}, {}, True),
        ({"CHATBOOKS_JOBS_BACKEND": "core"}, {}, True),
        ({"TLDW_JOBS_BACKEND": "core"}, {}, True),
        ({"CHATBOOKS_JOBS_BACKEND": "jobs"}, {}, False),
        ({"CHATBOOKS_CORE_WORKER_ENABLED": "false"}, {}, False),
    ],
)
def test_core_jobs_worker_spec_preserves_backend_flag_predicate(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    settings: dict[str, object],
    expected: bool,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    for key in [
        "CHATBOOKS_JOBS_BACKEND",
        "TLDW_JOBS_BACKEND",
        "CHATBOOKS_CORE_WORKER_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    spec = _specs_by_name(startup_pollers)["core_jobs_task"]

    assert spec.enabled(_context(settings=settings)) is expected


def test_core_jobs_worker_spec_uses_explicit_sidecar_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    for key in [
        "CHATBOOKS_JOBS_BACKEND",
        "TLDW_JOBS_BACKEND",
        "CHATBOOKS_CORE_WORKER_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)

    spec = _specs_by_name(startup_pollers)["core_jobs_task"]

    context = WorkerLifecycleContext(
        app="app",
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
        sidecar_mode=True,
    )
    assert spec.enabled(context) is False


@pytest.mark.asyncio
async def test_start_primary_jobs_pollers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    calls: list[str] = []

    async def _record_core(**kwargs: object) -> tuple[str, str]:
        """Record that the core worker starter ran."""

        del kwargs
        calls.append("core")
        return ("core-stop", "core-task")

    async def _record_files(**kwargs: object) -> tuple[str, str]:
        """Record that the files worker starter ran."""

        del kwargs
        calls.append("files")
        return ("files-stop", "files-task")

    async def _record_data_tables(**kwargs: object) -> tuple[str, str]:
        """Record that the data tables worker starter ran."""

        del kwargs
        calls.append("data-tables")
        return ("data-tables-stop", "data-tables-task")

    async def _record_prompt_studio(**kwargs: object) -> tuple[str, str]:
        """Record that the Prompt Studio worker starter ran."""

        del kwargs
        calls.append("prompt-studio")
        return ("prompt-studio-stop", "prompt-studio-task")

    async def _record_workspace_file_inventory(**kwargs: object) -> tuple[str, str]:
        """Record that the Workspace file inventory worker starter ran."""

        del kwargs
        calls.append("workspace-file-inventory")
        return ("workspace-file-inventory-stop", "workspace-file-inventory-task")

    monkeypatch.setattr(startup_pollers, "_start_core_jobs_worker", _record_core)
    monkeypatch.setattr(startup_pollers, "_start_files_jobs_worker", _record_files)
    monkeypatch.setattr(startup_pollers, "_start_data_tables_jobs_worker", _record_data_tables)
    monkeypatch.setattr(startup_pollers, "_start_prompt_studio_jobs_worker", _record_prompt_studio)
    monkeypatch.setattr(
        startup_pollers,
        "_start_workspace_file_inventory_jobs_worker",
        _record_workspace_file_inventory,
    )

    handles = await startup_pollers.start_primary_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
        sidecar_mode=False,
    )

    assert calls == ["core", "files", "data-tables", "prompt-studio", "workspace-file-inventory"]
    assert handles.core_jobs_stop_event == "core-stop"
    assert handles.core_jobs_task == "core-task"
    assert handles.files_jobs_stop_event == "files-stop"
    assert handles.files_jobs_task == "files-task"
    assert handles.data_tables_jobs_stop_event == "data-tables-stop"
    assert handles.data_tables_jobs_task == "data-tables-task"
    assert handles.prompt_studio_jobs_stop_event == "prompt-studio-stop"
    assert handles.prompt_studio_jobs_task == "prompt-studio-task"
    assert handles.workspace_file_inventory_jobs_stop_event == "workspace-file-inventory-stop"
    assert handles.workspace_file_inventory_jobs_task == "workspace-file-inventory-task"


@pytest.mark.asyncio
async def test_start_primary_jobs_pollers_passes_inventory_to_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    worker_inventory = object()
    captured_kwargs_by_worker: dict[str, dict[str, object]] = {}

    def _record_worker(label: str) -> Callable[..., object]:
        """Build a starter stub that captures kwargs for one worker label."""

        async def _record(**kwargs: object) -> tuple[str, str]:
            """Capture worker startup kwargs and return deterministic handles."""

            captured_kwargs_by_worker[label] = kwargs
            return (f"{label}-stop", f"{label}-task")

        return _record

    monkeypatch.setattr(startup_pollers, "_start_core_jobs_worker", _record_worker("core"))
    monkeypatch.setattr(startup_pollers, "_start_files_jobs_worker", _record_worker("files"))
    monkeypatch.setattr(startup_pollers, "_start_data_tables_jobs_worker", _record_worker("data-tables"))
    monkeypatch.setattr(startup_pollers, "_start_prompt_studio_jobs_worker", _record_worker("prompt-studio"))
    monkeypatch.setattr(
        startup_pollers,
        "_start_workspace_file_inventory_jobs_worker",
        _record_worker("workspace-file-inventory"),
    )

    await startup_pollers.start_primary_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda *args, **kwargs: False,
        sidecar_mode=False,
        worker_inventory=worker_inventory,
    )

    assert {
        worker: kwargs["worker_inventory"]
        for worker, kwargs in captured_kwargs_by_worker.items()
    } == {
        "core": worker_inventory,
        "files": worker_inventory,
        "data-tables": worker_inventory,
        "prompt-studio": worker_inventory,
        "workspace-file-inventory": worker_inventory,
    }


@pytest.mark.asyncio
async def test_start_core_jobs_worker_skips_in_sidecar_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: (_ for _ in ()).throw(AssertionError("no event")))
    monkeypatch.setattr(startup_pollers, "_create_task", lambda coro: (_ for _ in ()).throw(AssertionError("no task")))

    stop_event, task = await startup_pollers._start_core_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=True,
    )

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_core_jobs_worker_registers_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            registrations.append(kwargs)
            return "core-task", "core-stop"

    monkeypatch.setenv("CHATBOOKS_JOBS_BACKEND", "core")
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "true")
    monkeypatch.setattr(
        startup_pollers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    def _register_owned_job_poller(*args, **kwargs):
        raise AssertionError("legacy poller registration should not run")

    stop_event, task = await startup_pollers._start_core_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        sidecar_mode=False,
        worker_inventory=_FakeWorkerInventory(),
    )

    assert stop_event == "core-stop"
    assert task == "core-task"
    assert registrations == [
        {
            "name": "core_jobs_task",
            "task_name": "core_jobs_task",
            "coroutine_factory": startup_pollers._run_chatbooks_core_jobs_worker_service,
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        }
    ]


@pytest.mark.parametrize(
    (
        "starter_name",
        "flag_name",
        "route_name",
        "registered_name",
        "factory_name",
    ),
    [
        (
            "_start_files_jobs_worker",
            "FILES_JOBS_WORKER_ENABLED",
            "files",
            "files_jobs_task",
            "_run_file_artifacts_jobs_worker_service",
        ),
        (
            "_start_data_tables_jobs_worker",
            "DATA_TABLES_JOBS_WORKER_ENABLED",
            "data-tables",
            "data_tables_jobs_task",
            "_run_data_tables_jobs_worker_service",
        ),
        (
            "_start_prompt_studio_jobs_worker",
            "PROMPT_STUDIO_JOBS_WORKER_ENABLED",
            "prompt-studio",
            "prompt_studio_jobs_task",
            "_run_prompt_studio_jobs_worker_service",
        ),
        (
            "_start_workspace_file_inventory_jobs_worker",
            "WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED",
            "workspace_file_inventory_jobs_task",
            "workspace_file_inventory_jobs_task",
            "_run_workspace_file_inventory_jobs_worker_service",
        ),
    ],
)
@pytest.mark.asyncio
async def test_primary_jobs_worker_registers_with_worker_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    starter_name: str,
    flag_name: str,
    route_name: str,
    registered_name: str,
    factory_name: str,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        """Test double that records custom worker registration calls."""

        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            """Capture registration kwargs and return deterministic handles."""

            registrations.append(kwargs)
            return f"{registered_name}-task", f"{registered_name}-stop"

    monkeypatch.setattr(
        startup_pollers,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy poller registration should not run")

    stop_event, task = await getattr(startup_pollers, starter_name)(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route: (flag, route) == (flag_name, route_name),
        worker_inventory=_FakeWorkerInventory(),
    )

    assert stop_event == f"{registered_name}-stop"
    assert task == f"{registered_name}-task"
    assert registrations == [
        {
            "name": registered_name,
            "task_name": registered_name,
            "coroutine_factory": getattr(startup_pollers, factory_name),
            "timeout_sec": 5.0,
            "category": "jobs",
            "shutdown_phase": startup_pollers.ShutdownPhase.JOB_POLLER_QUIESCE,
        }
    ]


@pytest.mark.asyncio
async def test_start_files_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "files-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "files-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_file_artifacts_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "files-coro",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append(
            {
                "app": app,
                "owned_job_pollers": owned_job_pollers,
                "name": name,
                "task": task,
                "stop_event": stop_event,
            }
        )

    owned_job_pollers: list[object] = []
    stop_event, task = await startup_pollers._start_files_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route: (flag, route) == ("FILES_JOBS_WORKER_ENABLED", "files"),
    )

    assert stop_event == "files-stop"
    assert task == "files-task"
    assert captured_stop_events == ["files-stop"]
    assert created_coroutines == ["files-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "files_jobs_task",
            "task": "files-task",
            "stop_event": "files-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_workspace_file_inventory_jobs_worker_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: (_ for _ in ()).throw(AssertionError("no event")))
    monkeypatch.setattr(startup_pollers, "_create_task", lambda coro: (_ for _ in ()).throw(AssertionError("no task")))

    stop_event, task = await startup_pollers._start_workspace_file_inventory_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda flag, route: False,
    )

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_workspace_file_inventory_jobs_worker_registers_legacy_enabled_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()
    registrations: list[dict[str, object]] = []
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "workspace-inventory-stop")

    def _create_task(coro: object) -> str:
        created_coroutines.append(coro)
        return "workspace-inventory-task"

    monkeypatch.setattr(startup_pollers, "_create_task", _create_task)
    monkeypatch.setattr(
        startup_pollers,
        "_run_workspace_file_inventory_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "workspace-inventory-coro",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append(
            {
                "app": app,
                "owned_job_pollers": owned_job_pollers,
                "name": name,
                "task": task,
                "stop_event": stop_event,
            }
        )

    route_checks: list[tuple[str, str]] = []
    owned_job_pollers: list[object] = []
    stop_event, task = await startup_pollers._start_workspace_file_inventory_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route: route_checks.append((flag, route))
        or (flag, route)
        == ("WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED", "workspace_file_inventory_jobs_task"),
    )

    assert route_checks == [
        ("WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED", "workspace_file_inventory_jobs_task")
    ]
    assert stop_event == "workspace-inventory-stop"
    assert task == "workspace-inventory-task"
    assert captured_stop_events == ["workspace-inventory-stop"]
    assert created_coroutines == ["workspace-inventory-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "workspace_file_inventory_jobs_task",
            "task": "workspace-inventory-task",
            "stop_event": "workspace-inventory-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_prompt_studio_jobs_worker_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_primary_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "prompt-studio-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_pollers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_pollers,
        "_run_prompt_studio_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_pollers._start_prompt_studio_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        should_start_worker=lambda flag, route: (flag, route) == (
            "PROMPT_STUDIO_JOBS_WORKER_ENABLED",
            "prompt-studio",
        ),
    )

    assert stop_event is None
    assert task is None
