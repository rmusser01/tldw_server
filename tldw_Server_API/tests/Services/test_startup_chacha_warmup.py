from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_chacha_warmup():
    sys.modules.pop("tldw_Server_API.app.services.startup_chacha_warmup", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_chacha_warmup")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.debug_messages: list[str] = []
        self.warning_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(str(message))

    def debug(self, message: str) -> None:
        self.debug_messages.append(str(message))

    def warning(self, message: str) -> None:
        self.warning_messages.append(str(message))


@pytest.mark.asyncio
async def test_warm_chacha_notes_on_startup_schedules_single_user_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup = _import_startup_chacha_warmup()
    logger = _FakeLogger()
    observed: dict[str, object] = {"reset_calls": 0}

    monkeypatch.setattr(
        warmup,
        "_reset_chacha_shutdown_state",
        lambda: observed.__setitem__("reset_calls", int(observed["reset_calls"]) + 1),
    )
    monkeypatch.setattr(warmup, "_is_single_user_mode", lambda: True)
    monkeypatch.setattr(
        warmup,
        "_get_auth_settings",
        lambda: SimpleNamespace(SINGLE_USER_FIXED_ID="9"),
    )
    monkeypatch.setattr(
        warmup,
        "_schedule_warm_chacha_task",
        lambda user_id, client_id: observed.update({"user_id": user_id, "client_id": client_id}),
    )

    await warmup.warm_chacha_notes_on_startup(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert observed["reset_calls"] == 1
    assert observed["user_id"] == 9
    assert observed["client_id"] == "9"
    assert logger.info_messages == [
        "App Startup: scheduled ChaChaNotes warm-up for single-user id=9"
    ]
    assert logger.debug_messages == []
    assert logger.warning_messages == []


@pytest.mark.asyncio
async def test_warm_chacha_notes_on_startup_skips_multi_user_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup = _import_startup_chacha_warmup()
    logger = _FakeLogger()
    observed: dict[str, object] = {"reset_calls": 0, "scheduled": False}

    monkeypatch.setattr(
        warmup,
        "_reset_chacha_shutdown_state",
        lambda: observed.__setitem__("reset_calls", int(observed["reset_calls"]) + 1),
    )
    monkeypatch.setattr(warmup, "_is_single_user_mode", lambda: False)
    monkeypatch.setattr(
        warmup,
        "_schedule_warm_chacha_task",
        lambda user_id, client_id: observed.__setitem__("scheduled", True),
    )

    await warmup.warm_chacha_notes_on_startup(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert observed["reset_calls"] == 1
    assert observed["scheduled"] is False
    assert logger.info_messages == []
    assert logger.debug_messages == ["ChaChaNotes warm-up skipped (multi-user mode)"]
    assert logger.warning_messages == []


@pytest.mark.asyncio
async def test_warm_chacha_notes_on_startup_logs_best_effort_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup = _import_startup_chacha_warmup()
    logger = _FakeLogger()

    def _raise_reset_error() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(warmup, "_reset_chacha_shutdown_state", _raise_reset_error)

    await warmup.warm_chacha_notes_on_startup(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.warning_messages == ["ChaChaNotes warm-up scheduling failed: boom"]
