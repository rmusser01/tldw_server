from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_preflight_reporting():
    sys.modules.pop("tldw_Server_API.app.services.startup_preflight_reporting", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_preflight_reporting")


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str, *args) -> None:
        self.info_messages.append(message.format(*args) if args else str(message))

    def debug(self, message: str, *args) -> None:
        self.debug_messages.append(message.format(*args) if args else str(message))


@pytest.mark.asyncio
async def test_run_startup_preflight_checks_logs_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_preflight = _import_startup_preflight_reporting()
    logger = _FakeLogger()

    async def _fake_run_preflight_checks_in_thread():
        return SimpleNamespace(
            checks=["one", "two", "three"],
            warnings=["warn"],
            failures=[],
        )

    monkeypatch.setattr(
        startup_preflight,
        "_run_preflight_checks_in_thread",
        _fake_run_preflight_checks_in_thread,
    )

    await startup_preflight.run_startup_preflight_checks(
        logger=logger,
        startup_guard_exceptions=(OSError,),
    )

    assert logger.info_messages == ["Preflight: 3 checks, 1 warnings, 0 failures"]
    assert logger.debug_messages == []


@pytest.mark.asyncio
async def test_run_startup_preflight_checks_reraises_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_preflight = _import_startup_preflight_reporting()
    logger = _FakeLogger()

    async def _fake_run_preflight_checks_in_thread():
        raise RuntimeError("preflight boom")

    monkeypatch.setattr(
        startup_preflight,
        "_run_preflight_checks_in_thread",
        _fake_run_preflight_checks_in_thread,
    )

    with pytest.raises(RuntimeError, match="preflight boom"):
        await startup_preflight.run_startup_preflight_checks(
            logger=logger,
            startup_guard_exceptions=(RuntimeError, OSError),
        )

    assert logger.info_messages == []
    assert logger.debug_messages == []


@pytest.mark.asyncio
async def test_run_startup_preflight_checks_logs_debug_for_guard_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_preflight = _import_startup_preflight_reporting()
    logger = _FakeLogger()

    async def _fake_run_preflight_checks_in_thread():
        raise OSError("preflight skipped")

    monkeypatch.setattr(
        startup_preflight,
        "_run_preflight_checks_in_thread",
        _fake_run_preflight_checks_in_thread,
    )

    await startup_preflight.run_startup_preflight_checks(
        logger=logger,
        startup_guard_exceptions=(RuntimeError, OSError),
    )

    assert logger.info_messages == []
    assert logger.debug_messages == ["Preflight checks skipped: preflight skipped"]
