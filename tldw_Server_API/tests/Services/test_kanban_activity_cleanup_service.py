from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from tldw_Server_API.app.services import kanban_activity_cleanup_service as service


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


def test_enumerate_user_ids_base_dir_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_user_db_base_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/kanban-base sk-live-base")),
    )

    assert service._enumerate_user_ids() == []
    assert logger.debugs == ["kanban_activity_cleanup: failed to resolve user db base dir"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/kanban-base" not in rendered
    assert "sk-live-base" not in rendered


@pytest.mark.asyncio
async def test_start_scheduler_cleanup_run_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setenv("KANBAN_ACTIVITY_CLEANUP_ENABLED", "true")
    monkeypatch.setenv("KANBAN_ACTIVITY_CLEANUP_INTERVAL_SEC", "1")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service,
        "_enumerate_user_ids",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/kanban-run sk-live-run")),
    )

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(service.asyncio, "sleep", _fake_sleep)

    task = await service.start_kanban_activity_cleanup_scheduler()
    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert logger.debugs == ["kanban_activity_cleanup: cleanup run failed"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/kanban-run" not in rendered
    assert "sk-live-run" not in rendered
