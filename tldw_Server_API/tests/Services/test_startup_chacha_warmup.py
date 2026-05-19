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


@pytest.mark.asyncio
async def test_warm_chacha_db_for_user_records_corrupt_db_and_fails_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps

    user_id = 43
    db_path = tmp_path / "43" / "ChaChaNotes.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"not a sqlite database")

    with chacha_deps._chacha_db_lock:
        chacha_deps._chacha_db_instances.clear()
        chacha_deps._chacha_db_init_events.clear()
        chacha_deps._chacha_db_init_errors.clear()
    with chacha_deps._CHACHA_HEALTH_LOCK:
        chacha_deps._CHACHA_HEALTH.update(
            {
                "init_attempts": 0,
                "init_failures": 0,
                "last_init_ms": None,
                "last_error": None,
                "last_warn_dump": None,
                "cached_instances": 0,
                "default_char_ensures": 0,
                "default_char_failures": 0,
                "warm_startups": 0,
                "last_failure": None,
            }
        )

    monkeypatch.setattr(
        chacha_deps.DatabasePaths,
        "get_user_base_directory",
        lambda _user_id: db_path.parent,
    )
    monkeypatch.setattr(
        chacha_deps.DatabasePaths,
        "get_chacha_db_path",
        lambda _user_id: db_path,
    )

    await chacha_deps.warm_chacha_db_for_user(user_id, str(user_id))

    snapshot = chacha_deps.get_chacha_health_snapshot()
    assert snapshot["status"] == "degraded"
    assert snapshot["last_error"] == "sqlite_corruption"
    assert snapshot["last_failure"]["affected_db"] == "user:43/ChaChaNotes.db"
    assert snapshot["last_failure"]["recovery"]["automatic_repair"] is False
    assert str(tmp_path) not in str(snapshot)
    assert "not a sqlite database" not in str(snapshot)
