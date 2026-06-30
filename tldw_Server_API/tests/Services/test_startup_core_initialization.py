from __future__ import annotations

from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_initialize_startup_core_components_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_core_initialization as startup_core

    calls: list[tuple[str, dict[str, object]]] = []
    log_messages: list[str] = []

    async def _record_validations() -> None:
        calls.append(("validations", {}))

    async def _record_auth_runtime_services(**kwargs):
        calls.append(("auth_runtime", kwargs))
        return SimpleNamespace(db_pool="db-pool", session_manager="session-manager")

    async def _record_chacha_warmup(**kwargs):
        calls.append(("chacha", kwargs))

    def _record_privilege_validation(**kwargs):
        calls.append(("privilege", kwargs))

    def _record_catalog_loading(**kwargs):
        calls.append(("catalogs", kwargs))

    async def _record_heavy_initializations(app_arg, **kwargs):
        calls.append(("heavy", {"app": app_arg, **kwargs}))
        return "heavy-startup-handles"

    monkeypatch.setattr(startup_core, "_run_startup_validations", _record_validations)
    monkeypatch.setattr(
        startup_core,
        "_initialize_auth_runtime_services",
        _record_auth_runtime_services,
    )
    monkeypatch.setattr(startup_core, "_warm_chacha_notes_on_startup", _record_chacha_warmup)
    monkeypatch.setattr(
        startup_core,
        "_validate_startup_privilege_metadata",
        _record_privilege_validation,
    )
    monkeypatch.setattr(startup_core, "_load_startup_catalogs", _record_catalog_loading)
    monkeypatch.setattr(startup_core, "_start_heavy_initializations", _record_heavy_initializations)

    logger = SimpleNamespace(info=lambda message: log_messages.append(message))

    handles = await startup_core.initialize_startup_core_components(
        app="app",
        module_file="/tmp/main.py",
        logger=logger,
        route_enabled=lambda route_name: route_name == "chat",
        defer_heavy=True,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert [name for name, _ in calls] == [
        "validations",
        "auth_runtime",
        "chacha",
        "privilege",
        "catalogs",
        "heavy",
    ]
    assert log_messages == [
        "App Startup: Initializing authentication services...",
        "App Startup: Initializing Chat module components...",
    ]
    assert calls[1][1]["app"] == "app"
    assert calls[1][1]["logger"] is logger
    assert calls[1][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[2][1]["logger"] is logger
    assert calls[3][1]["app"] == "app"
    assert calls[4][1]["module_file"] == "/tmp/main.py"
    assert calls[4][1]["import_exceptions"] == (ImportError,)
    assert calls[5][1]["app"] == "app"
    assert calls[5][1]["defer_heavy"] is True
    assert handles.db_pool == "db-pool"
    assert handles.session_manager == "session-manager"
    assert handles.heavy_startup_handles == "heavy-startup-handles"


def test_build_startup_sandbox_orchestrator_skips_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_core_initialization as startup_core

    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", raising=False)
    monkeypatch.setattr(startup_core.platform, "system", lambda: "Darwin")

    assert startup_core._sandbox_startup_warning_configured() is False


def test_build_startup_sandbox_orchestrator_downgrades_failures_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import startup_core_initialization as startup_core

    warnings: list[str] = []

    def _raising_service_factory(*args, **kwargs):
        raise RuntimeError("sandbox init failed")

    monkeypatch.setattr(
        startup_core,
        "_sandbox_startup_warning_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.service.SandboxService",
        _raising_service_factory,
    )

    logger = SimpleNamespace(warning=lambda message, exc: warnings.append(message.format(exc)))

    result = startup_core._build_startup_sandbox_orchestrator(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert result is None
    assert warnings == [
        "Startup sandbox orchestrator unavailable; continuing without startup sandbox warnings: sandbox init failed"
    ]
