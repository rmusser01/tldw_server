from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_environment_reporting():
    sys.modules.pop("tldw_Server_API.app.services.startup_environment_reporting", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_environment_reporting")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.error_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def warning(self, message: str) -> None:
        self.warning_messages.append(str(message))

    def error(self, message: str) -> None:
        self.error_messages.append(str(message))

    def exception(self, message: str) -> None:
        self.exception_messages.append(str(message))


@pytest.mark.asyncio
async def test_report_startup_environment_logs_single_user_banner_and_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporting = _import_startup_environment_reporting()
    logger = _FakeLogger()
    app = SimpleNamespace(state=SimpleNamespace(limiter=object()))

    monkeypatch.setattr(
        reporting,
        "_get_auth_settings",
        lambda: SimpleNamespace(
            SINGLE_USER_API_KEY="secret-key-123",
            AUTH_MODE="single_user",
            DATABASE_URL="sqlite:///tmp/users.db",
            REDIS_URL="redis://localhost:6379/0",
        ),
    )
    monkeypatch.setattr(reporting, "_is_single_user_mode", lambda: True)
    monkeypatch.setattr(reporting, "_get_csrf_global_settings", lambda: {"CSRF_ENABLED": False})
    monkeypatch.setattr(
        reporting,
        "_get_cors_runtime_diagnostics",
        lambda: {
            "disable_cors": False,
            "disable_cors_source": "env",
            "allow_credentials": True,
            "allow_credentials_source": "env",
            "allowed_origins_count": 2,
            "allowed_origins_source": "env",
            "allowed_origins": ["http://localhost:3000", "http://127.0.0.1:8080"],
            "config_path": "/tmp/config.txt",
            "config_loaded": True,
        },
    )
    monkeypatch.setattr(
        reporting,
        "_get_provider_manager",
        lambda: SimpleNamespace(providers={"openai": object(), "anthropic": object(), "local": object()}),
    )
    monkeypatch.setattr(reporting, "_otel_available", lambda: True)

    async def _fake_get_db_pool():
        return SimpleNamespace(pool=None)

    monkeypatch.setattr(reporting, "_get_db_pool", _fake_get_db_pool)
    monkeypatch.setenv("tldw_production", "false")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    await reporting.report_startup_environment(
        app=app,
        logger=logger,
        startup_api_key_log_value=lambda api_key: "mask...1234",
        shared_is_truthy=lambda value: str(value).lower() in {"true", "1", "yes", "y", "on"},
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert "🚀 TLDW Server Started Successfully" in logger.info_messages
    assert "🔐 Authentication Mode: SINGLE USER" in logger.info_messages
    assert "🔑 API Key: mask...1234 (masked; set SHOW_API_KEY_ON_STARTUP=true to display once)" in logger.info_messages
    assert "📍 API Documentation: http://127.0.0.1:8000/docs" in logger.info_messages
    assert "Preflight Environment Report ─────────────────────────────────────────" in logger.info_messages
    assert "• Mode: single_user | Production: False" in logger.info_messages
    assert "• Database: engine=sqlite" in logger.info_messages
    assert "• Database check: OK" in logger.info_messages
    assert "• Redis: enabled=True" in logger.info_messages
    assert "• CORS: allowed_origins=2 | allow_credentials=True" in logger.info_messages
    assert "• Providers configured: 3" in logger.info_messages
    assert "• OpenTelemetry available: True" in logger.info_messages
    assert logger.warning_messages == []
    assert logger.error_messages == []
    assert logger.exception_messages == []


@pytest.mark.asyncio
async def test_report_startup_environment_logs_multi_user_prod_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporting = _import_startup_environment_reporting()
    logger = _FakeLogger()
    app = SimpleNamespace(state=SimpleNamespace())

    monkeypatch.setattr(
        reporting,
        "_get_auth_settings",
        lambda: SimpleNamespace(
            SINGLE_USER_API_KEY="unused",
            AUTH_MODE="multi_user",
            DATABASE_URL="sqlite:///tmp/users.db",
            REDIS_URL="",
        ),
    )
    monkeypatch.setattr(reporting, "_is_single_user_mode", lambda: False)
    monkeypatch.setattr(reporting, "_get_csrf_global_settings", lambda: {"CSRF_ENABLED": False})
    monkeypatch.setattr(
        reporting,
        "_get_cors_runtime_diagnostics",
        lambda: {
            "disable_cors": True,
            "disable_cors_source": "env",
            "allow_credentials": False,
            "allow_credentials_source": "default",
            "allowed_origins_count": 0,
            "allowed_origins_source": "default",
            "allowed_origins": [],
            "config_path": None,
            "config_loaded": False,
        },
    )
    monkeypatch.setattr(reporting, "_get_provider_manager", lambda: None)
    monkeypatch.setattr(reporting, "_otel_available", lambda: False)

    async def _fake_get_db_pool():
        return SimpleNamespace(pool=None)

    monkeypatch.setattr(reporting, "_get_db_pool", _fake_get_db_pool)
    monkeypatch.setenv("tldw_production", "true")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("TLDW_TEST_MODE", "1")

    await reporting.report_startup_environment(
        app=app,
        logger=logger,
        startup_api_key_log_value=lambda api_key: api_key,
        shared_is_truthy=lambda value: str(value).lower() in {"true", "1", "yes", "y", "on"},
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert "🔐 Authentication Mode: MULTI USER" in logger.info_messages
    assert "JWT Bearer tokens or X-API-KEY (per-user) supported for SQLite setups" in logger.info_messages
    assert "• Mode: multi_user | Production: True" in logger.info_messages
    assert "• Database: engine=sqlite" in logger.info_messages
    assert "• CORS: disabled" in logger.info_messages
    assert "• Global rate limiter: False" in logger.info_messages
    assert "• Providers configured: 0" in logger.info_messages
    assert "• OpenTelemetry available: False" in logger.info_messages
    assert "• Database check: FAIL (SQLite in multi-user prod not supported)" in logger.error_messages
    assert (
        "Test-mode toggles enabled in production: TEST_MODE, TLDW_TEST_MODE - disable these for secure deployments"
        in logger.warning_messages
    )
    assert logger.exception_messages == []


@pytest.mark.asyncio
async def test_report_startup_environment_handles_banner_and_preflight_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporting = _import_startup_environment_reporting()
    logger = _FakeLogger()
    app = SimpleNamespace(state=SimpleNamespace())
    auth_settings_calls = {"count": 0}

    def _fake_get_auth_settings():
        auth_settings_calls["count"] += 1
        if auth_settings_calls["count"] == 1:
            raise ImportError("settings import failed")
        return SimpleNamespace(
            SINGLE_USER_API_KEY="unused",
            AUTH_MODE="single_user",
            DATABASE_URL="sqlite:///tmp/users.db",
            REDIS_URL="",
        )

    monkeypatch.setattr(reporting, "_get_auth_settings", _fake_get_auth_settings)
    monkeypatch.setattr(reporting, "_get_csrf_global_settings", lambda: (_ for _ in ()).throw(RuntimeError("preflight failed")))

    await reporting.report_startup_environment(
        app=app,
        logger=logger,
        startup_api_key_log_value=lambda api_key: api_key,
        shared_is_truthy=lambda value: bool(value),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert logger.exception_messages == ["Failed to display startup info: settings import failed"]
    assert logger.warning_messages == ["Preflight report could not be generated: preflight failed"]


@pytest.mark.asyncio
async def test_report_startup_environment_handles_preflight_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporting = _import_startup_environment_reporting()
    logger = _FakeLogger()
    app = SimpleNamespace(state=SimpleNamespace())
    auth_settings_calls = {"count": 0}

    def _fake_get_auth_settings():
        auth_settings_calls["count"] += 1
        if auth_settings_calls["count"] == 2:
            raise ImportError("preflight import failed")
        return SimpleNamespace(
            SINGLE_USER_API_KEY="unused",
            AUTH_MODE="single_user",
            DATABASE_URL="sqlite:///tmp/users.db",
            REDIS_URL="",
        )

    monkeypatch.setattr(reporting, "_get_auth_settings", _fake_get_auth_settings)
    monkeypatch.setattr(reporting, "_is_single_user_mode", lambda: True)

    await reporting.report_startup_environment(
        app=app,
        logger=logger,
        startup_api_key_log_value=lambda api_key: api_key,
        shared_is_truthy=lambda value: bool(value),
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert logger.warning_messages == [
        "Preflight report could not be generated: preflight import failed"
    ]


@pytest.mark.asyncio
async def test_log_preflight_environment_report_preserves_sqlite_engine_when_pool_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporting = _import_startup_environment_reporting()
    logger = _FakeLogger()
    app = SimpleNamespace(state=SimpleNamespace())

    monkeypatch.setattr(
        reporting,
        "_get_auth_settings",
        lambda: SimpleNamespace(
            AUTH_MODE="multi_user",
            DATABASE_URL="sqlite:///tmp/users.db",
            REDIS_URL="",
        ),
    )
    monkeypatch.setattr(reporting, "_get_csrf_global_settings", lambda: {"CSRF_ENABLED": False})
    monkeypatch.setattr(
        reporting,
        "_get_cors_runtime_diagnostics",
        lambda: {
            "disable_cors": True,
            "disable_cors_source": "env",
            "allow_credentials": False,
            "allow_credentials_source": "default",
            "allowed_origins_count": 0,
            "allowed_origins_source": "default",
            "allowed_origins": [],
            "config_path": None,
            "config_loaded": False,
        },
    )
    monkeypatch.setattr(reporting, "_get_provider_manager", lambda: None)
    monkeypatch.setattr(reporting, "_otel_available", lambda: False)

    async def _failing_get_db_pool():
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(reporting, "_get_db_pool", _failing_get_db_pool)
    monkeypatch.setenv("tldw_production", "true")

    await reporting._log_preflight_environment_report(
        app=app,
        logger=logger,
        shared_is_truthy=lambda value: str(value).lower() in {"true", "1", "yes", "y", "on"},
        startup_guard_exceptions=(RuntimeError,),
    )

    assert "• Database: engine=sqlite" in logger.info_messages
    assert "• Database check: FAIL (SQLite in multi-user prod not supported)" in logger.error_messages
