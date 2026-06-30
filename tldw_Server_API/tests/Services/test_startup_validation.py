from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_validation():
    sys.modules.pop("tldw_Server_API.app.services.startup_validation", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_validation")


def test_startup_validation_exception_guard_is_specific() -> None:
    startup_validation = _import_startup_validation()

    assert startup_validation._STARTUP_GUARD_EXCEPTIONS == (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )


@pytest.mark.asyncio
async def test_run_startup_validations_warns_for_first_time_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_validation = _import_startup_validation()
    warning_messages: list[str] = []
    integrity_calls: list[dict[str, object]] = []

    async def _fake_verify(**kwargs: object) -> None:
        integrity_calls.append(kwargs)

    monkeypatch.setattr(startup_validation, "needs_setup", lambda: True)
    monkeypatch.setattr(
        startup_validation,
        "get_settings",
        lambda: SimpleNamespace(DATABASE_URL="sqlite:///tmp/auth.db", AUTH_MODE="single_user"),
    )
    monkeypatch.setattr(
        startup_validation,
        "verify_authnz_sqlite_startup_integrity",
        _fake_verify,
    )
    monkeypatch.setattr(
        startup_validation.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )
    monkeypatch.delenv("TLDW_ALLOW_CORRUPT_AUTHNZ_STARTUP", raising=False)

    await startup_validation.run_startup_validations()

    assert any("First-time setup is enabled" in message for message in warning_messages)
    assert integrity_calls == [
        {
            "database_url": "sqlite:///tmp/auth.db",
            "auth_mode": "single_user",
            "dispatch_alerts": True,
            "fail_on_error": True,
        }
    ]


@pytest.mark.asyncio
async def test_run_startup_validations_supports_fail_open_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_validation = _import_startup_validation()
    warning_messages: list[str] = []
    integrity_calls: list[dict[str, object]] = []

    async def _fake_verify(**kwargs: object) -> None:
        integrity_calls.append(kwargs)

    monkeypatch.setattr(startup_validation, "needs_setup", lambda: False)
    monkeypatch.setattr(
        startup_validation,
        "get_settings",
        lambda: SimpleNamespace(DATABASE_URL="sqlite:///tmp/auth.db", AUTH_MODE="multi_user"),
    )
    monkeypatch.setattr(
        startup_validation,
        "verify_authnz_sqlite_startup_integrity",
        _fake_verify,
    )
    monkeypatch.setattr(
        startup_validation.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )
    monkeypatch.setenv("TLDW_ALLOW_CORRUPT_AUTHNZ_STARTUP", "true")

    await startup_validation.run_startup_validations()

    assert integrity_calls == [
        {
            "database_url": "sqlite:///tmp/auth.db",
            "auth_mode": "multi_user",
            "dispatch_alerts": True,
            "fail_on_error": False,
        }
    ]
    assert any(
        "Corrupt AuthNZ DB fail-open mode enabled" in message
        for message in warning_messages
    )


@pytest.mark.asyncio
async def test_run_startup_validations_logs_setup_check_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_validation = _import_startup_validation()
    debug_messages: list[str] = []
    integrity_calls: list[dict[str, object]] = []

    async def _fake_verify(**kwargs: object) -> None:
        integrity_calls.append(kwargs)

    def _broken_needs_setup() -> bool:
        raise RuntimeError("setup boom")

    monkeypatch.setattr(startup_validation, "needs_setup", _broken_needs_setup)
    monkeypatch.setattr(
        startup_validation,
        "get_settings",
        lambda: SimpleNamespace(DATABASE_URL="sqlite:///tmp/auth.db", AUTH_MODE="single_user"),
    )
    monkeypatch.setattr(
        startup_validation,
        "verify_authnz_sqlite_startup_integrity",
        _fake_verify,
    )
    monkeypatch.setattr(
        startup_validation.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )

    await startup_validation.run_startup_validations()

    assert any(
        "Setup status check failed during startup" in message for message in debug_messages
    )
    assert integrity_calls == [
        {
            "database_url": "sqlite:///tmp/auth.db",
            "auth_mode": "single_user",
            "dispatch_alerts": True,
            "fail_on_error": True,
        }
    ]


@pytest.mark.asyncio
async def test_run_startup_validations_reraises_integrity_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_validation = _import_startup_validation()
    exception_messages: list[str] = []

    async def _failing_verify(**_kwargs: object) -> None:
        raise RuntimeError("integrity boom")

    monkeypatch.setattr(startup_validation, "needs_setup", lambda: False)
    monkeypatch.setattr(
        startup_validation,
        "get_settings",
        lambda: SimpleNamespace(DATABASE_URL="sqlite:///tmp/auth.db", AUTH_MODE="single_user"),
    )
    monkeypatch.setattr(
        startup_validation,
        "verify_authnz_sqlite_startup_integrity",
        _failing_verify,
    )
    monkeypatch.setattr(
        startup_validation.logger,
        "exception",
        lambda message, *args, **kwargs: exception_messages.append(str(message)),
    )

    with pytest.raises(RuntimeError, match="integrity boom"):
        await startup_validation.run_startup_validations()

    assert any(
        "AuthNZ SQLite integrity preflight failed" in message
        for message in exception_messages
    )
