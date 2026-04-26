from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_auth_runtime():
    sys.modules.pop("tldw_Server_API.app.services.startup_auth_runtime", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_auth_runtime")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def exception(self, message: str) -> None:
        self.exception_messages.append(str(message))


@pytest.mark.asyncio
async def test_initialize_auth_runtime_services_runs_post_auth_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_runtime = _import_startup_auth_runtime()
    logger = _FakeLogger()
    app = object()
    observed: list[tuple[str, object]] = []
    db_pool = object()
    session_manager = object()

    async def _fake_init_auth_services():
        observed.append(("init_auth_services", None))
        return db_pool

    async def _fake_init_resource_governor(app_arg):
        observed.append(("init_resource_governor", app_arg))

    def _fake_validate_auth_rg_startup_guards(app_arg):
        observed.append(("validate_auth_rg_startup_guards", app_arg))

    async def _fake_get_session_manager():
        observed.append(("get_session_manager", None))
        return session_manager

    dispatcher = SimpleNamespace(
        enabled=True,
        validate_configuration=lambda: observed.append(("validate_configuration", None)),
    )

    monkeypatch.setattr(auth_runtime, "_init_auth_services", _fake_init_auth_services)
    monkeypatch.setattr(auth_runtime, "_init_resource_governor", _fake_init_resource_governor)
    monkeypatch.setattr(
        auth_runtime,
        "_validate_auth_rg_startup_guards",
        _fake_validate_auth_rg_startup_guards,
    )
    monkeypatch.setattr(auth_runtime, "_get_session_manager", _fake_get_session_manager)
    monkeypatch.setattr(
        auth_runtime,
        "_get_security_alert_dispatcher",
        lambda: dispatcher,
    )

    handles = await auth_runtime.initialize_auth_runtime_services(
        app=app,
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert handles.db_pool is db_pool
    assert handles.session_manager is session_manager
    assert observed == [
        ("init_auth_services", None),
        ("init_resource_governor", app),
        ("validate_auth_rg_startup_guards", app),
        ("get_session_manager", None),
        ("validate_configuration", None),
    ]
    assert logger.info_messages == [
        "App Startup: Session manager initialized",
        "App Startup: Security alert configuration validated",
    ]
    assert logger.exception_messages == []


@pytest.mark.asyncio
async def test_initialize_auth_runtime_services_logs_and_preserves_partial_handles_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_runtime = _import_startup_auth_runtime()
    logger = _FakeLogger()
    db_pool = object()

    async def _fake_init_auth_services():
        return db_pool

    async def _raise_resource_governor_error(app_arg):
        del app_arg
        raise RuntimeError("rg boom")

    monkeypatch.setattr(auth_runtime, "_init_auth_services", _fake_init_auth_services)
    monkeypatch.setattr(
        auth_runtime,
        "_init_resource_governor",
        _raise_resource_governor_error,
    )

    handles = await auth_runtime.initialize_auth_runtime_services(
        app=object(),
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert handles.db_pool is db_pool
    assert handles.session_manager is None
    assert logger.info_messages == []
    assert logger.exception_messages == [
        "App Startup: Security alert validation / auth services init failed: rg boom"
    ]


@pytest.mark.asyncio
async def test_initialize_auth_runtime_services_reraises_auth_startup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_runtime = _import_startup_auth_runtime()
    logger = _FakeLogger()

    async def _raise_auth_startup_error():
        raise auth_runtime.AuthStartupError("AUTHNZ_DB_POOL_STARTUP_FAILED")

    monkeypatch.setattr(auth_runtime, "_init_auth_services", _raise_auth_startup_error)

    with pytest.raises(
        auth_runtime.AuthStartupError,
        match="AUTHNZ_DB_POOL_STARTUP_FAILED",
    ):
        await auth_runtime.initialize_auth_runtime_services(
            app=object(),
            logger=logger,
            startup_guard_exceptions=(RuntimeError,),
        )

    assert logger.info_messages == []
    assert logger.exception_messages == []


@pytest.mark.asyncio
async def test_initialize_auth_runtime_services_rejects_missing_db_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_runtime = _import_startup_auth_runtime()
    logger = _FakeLogger()

    async def _fake_init_auth_services():
        return None

    monkeypatch.setattr(auth_runtime, "_init_auth_services", _fake_init_auth_services)

    with pytest.raises(
        auth_runtime.AuthStartupError,
        match="AUTHNZ_DB_POOL_STARTUP_RETURNED_NONE",
    ):
        await auth_runtime.initialize_auth_runtime_services(
            app=object(),
            logger=logger,
            startup_guard_exceptions=(RuntimeError,),
        )

    assert logger.info_messages == []
    assert logger.exception_messages == []


@pytest.mark.asyncio
async def test_initialize_auth_runtime_services_logs_invalid_security_alert_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_runtime = _import_startup_auth_runtime()
    logger = _FakeLogger()
    db_pool = object()
    session_manager = object()

    async def _fake_init_auth_services():
        return db_pool

    async def _fake_init_resource_governor(app_arg):
        del app_arg

    def _fake_validate_auth_rg_startup_guards(app_arg):
        del app_arg

    async def _fake_get_session_manager():
        return session_manager

    dispatcher = SimpleNamespace(
        enabled=True,
        validate_configuration=lambda: (_ for _ in ()).throw(ValueError("invalid alerting")),
    )

    monkeypatch.setattr(auth_runtime, "_init_auth_services", _fake_init_auth_services)
    monkeypatch.setattr(auth_runtime, "_init_resource_governor", _fake_init_resource_governor)
    monkeypatch.setattr(
        auth_runtime,
        "_validate_auth_rg_startup_guards",
        _fake_validate_auth_rg_startup_guards,
    )
    monkeypatch.setattr(auth_runtime, "_get_session_manager", _fake_get_session_manager)
    monkeypatch.setattr(
        auth_runtime,
        "_get_security_alert_dispatcher",
        lambda: dispatcher,
    )

    with pytest.raises(ValueError, match="invalid alerting"):
        await auth_runtime.initialize_auth_runtime_services(
            app=object(),
            logger=logger,
            startup_guard_exceptions=(ValueError,),
        )

    assert logger.info_messages == ["App Startup: Session manager initialized"]
    assert logger.exception_messages == [
        "App Startup: Security alert configuration invalid: invalid alerting",
    ]
