from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from tldw_Server_API.app.services import kanban_purge_service as service


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)


def _rendered_logs(logger: _LoggerStub) -> str:
    return "\n".join(logger.debugs + logger.infos)


def test_enumerate_user_ids_base_dir_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_user_db_base_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/kanban-purge-base sk-live-base")),
    )

    assert service._enumerate_user_ids() == []
    assert logger.debugs == ["kanban_purge: failed to resolve user db base dir"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/kanban-purge-base" not in _rendered_logs(logger)
    assert "sk-live-base" not in _rendered_logs(logger)


@pytest.mark.asyncio
async def test_start_kanban_purge_scheduler_loop_error_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    sleep_calls = {"count": 0}

    def _failing_purge(_user_id: int, _grace_days: int) -> dict[str, int]:
        raise RuntimeError("secret /tmp/kanban-purge-loop sk-live-loop")

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setenv("KANBAN_PURGE_ENABLED", "true")
    monkeypatch.setenv("KANBAN_PURGE_INTERVAL_SEC", "1")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service, "_enumerate_user_ids", lambda: [17])
    monkeypatch.setattr(service, "_purge_for_user", _failing_purge)
    monkeypatch.setattr(service.asyncio, "sleep", _fake_sleep)

    task = await service.start_kanban_purge_scheduler()

    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert logger.debugs == ["kanban_purge: purge run failed"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/kanban-purge-loop" not in _rendered_logs(logger)
    assert "sk-live-loop" not in _rendered_logs(logger)
