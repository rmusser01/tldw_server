from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_prompts_close_worker():
    sys.modules.pop("tldw_Server_API.app.services.startup_prompts_close_worker", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_prompts_close_worker")


class _FakeLogger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []

    def debug(self, message: str) -> None:
        self.debug_messages.append(str(message))


def test_start_prompts_close_worker_starts_pending_close_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_prompts = _import_startup_prompts_close_worker()
    logger = _FakeLogger()
    calls: list[str] = []

    monkeypatch.setattr(
        startup_prompts,
        "_start_prompts_pending_close_worker",
        lambda: calls.append("started"),
    )

    startup_prompts.start_prompts_close_worker(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert calls == ["started"]
    assert logger.debug_messages == []


def test_start_prompts_close_worker_logs_debug_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_prompts = _import_startup_prompts_close_worker()
    logger = _FakeLogger()

    def _raise_start_error() -> None:
        raise RuntimeError("prompts boom")

    monkeypatch.setattr(
        startup_prompts,
        "_start_prompts_pending_close_worker",
        _raise_start_error,
    )

    startup_prompts.start_prompts_close_worker(
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
    )

    assert logger.debug_messages == [
        "App Startup: Prompts close worker startup skipped/failed: prompts boom"
    ]
