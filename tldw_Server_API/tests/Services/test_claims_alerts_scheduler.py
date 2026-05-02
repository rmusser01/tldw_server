from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.services import claims_alerts_scheduler as service

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


def test_enumerate_sqlite_user_ids_base_dir_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_user_db_base_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/claims-alerts-base sk-live-base")),
    )

    assert service._enumerate_sqlite_user_ids() == []
    assert logger.debugs == ["claims_alerts: failed to resolve user db base dir"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/claims-alerts-base" not in rendered
    assert "sk-live-base" not in rendered


def test_enumerate_sqlite_user_ids_single_user_fallback_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.DatabasePaths, "get_user_db_base_dir", lambda: tmp_path)
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_single_user_id",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/claims-alerts-single sk-live-single")),
    )

    assert service._enumerate_sqlite_user_ids() == []
    assert logger.debugs == ["claims_alerts: failed to derive single_user_id"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/claims-alerts-single" not in rendered
    assert "sk-live-single" not in rendered


@pytest.mark.asyncio
async def test_run_claims_alerts_once_postgres_user_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()

    class _ManagedDb:
        def list_claims_monitoring_user_ids(self) -> list[int]:
            return [7]

    class _ManagedContext:
        def __enter__(self) -> _ManagedDb:
            return _ManagedDb()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _failing_evaluator(**kwargs: Any) -> dict[str, str]:
        raise RuntimeError("secret /tmp/claims-alerts-user sk-live-user")

    async def _fake_send_digest(*, target_user_id: str, db: Any) -> None:
        raise AssertionError("digest should not run when evaluation fails")

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.content_db_settings, "backend_type", service.BackendType.POSTGRESQL)
    monkeypatch.setattr(service, "managed_media_database", lambda **kwargs: _ManagedContext())
    monkeypatch.setattr(service, "send_claims_alert_email_digest_for_scheduler", _fake_send_digest)

    processed = await service.run_claims_alerts_once(
        evaluator=_failing_evaluator,
        window_sec=60,
        baseline_sec=120,
    )

    assert processed == 0
    assert logger.warnings == ["claims_alerts: evaluation failed for user 7"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/claims-alerts-user" not in rendered
    assert "sk-live-user" not in rendered


@pytest.mark.asyncio
async def test_run_claims_alerts_once_postgres_media_db_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()

    class _ManagedContext:
        def __enter__(self):
            raise RuntimeError("secret /tmp/claims-alerts-db sk-live-db")

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.content_db_settings, "backend_type", service.BackendType.POSTGRESQL)
    monkeypatch.setattr(service, "managed_media_database", lambda **kwargs: _ManagedContext())

    processed = await service.run_claims_alerts_once()

    assert processed == 0
    assert logger.warnings == ["claims_alerts: failed to create media db"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/claims-alerts-db" not in rendered
    assert "sk-live-db" not in rendered


@pytest.mark.asyncio
async def test_run_claims_alerts_once_sqlite_user_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()

    class _ManagedContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _failing_evaluator(**kwargs: Any) -> dict[str, str]:
        raise RuntimeError("secret /tmp/claims-alerts-sqlite sk-live-sqlite")

    async def _fake_send_digest(*, target_user_id: str, db: Any) -> None:
        raise AssertionError("digest should not run when evaluation fails")

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.content_db_settings, "backend_type", service.BackendType.SQLITE)
    monkeypatch.setattr(service, "_enumerate_sqlite_user_ids", lambda: [11])
    monkeypatch.setattr(service.DatabasePaths, "get_media_db_path", lambda user_id: Path(f"/tmp/{user_id}.db"))
    monkeypatch.setattr(service, "managed_media_database", lambda **kwargs: _ManagedContext())
    monkeypatch.setattr(service, "send_claims_alert_email_digest_for_scheduler", _fake_send_digest)

    processed = await service.run_claims_alerts_once(
        evaluator=_failing_evaluator,
        window_sec=60,
        baseline_sec=120,
    )

    assert processed == 0
    assert logger.warnings == ["claims_alerts: evaluation failed for user 11"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/claims-alerts-sqlite" not in rendered
    assert "sk-live-sqlite" not in rendered


@pytest.mark.asyncio
async def test_start_claims_alerts_scheduler_loop_error_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    created: dict[str, str | None] = {}
    sleep_calls = {"count": 0}
    original_create_task = asyncio.create_task

    async def _failing_run_once() -> int:
        raise RuntimeError("secret /tmp/claims-alerts-loop sk-live-loop")

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError()

    def _fake_create_task(coro, *, name=None):
        created["name"] = name
        return original_create_task(coro, name=name)

    monkeypatch.setenv("CLAIMS_ALERTS_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_ALERTS_EVAL_INTERVAL_SEC", "1")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service, "run_claims_alerts_once", _failing_run_once)
    monkeypatch.setattr(service.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(service.asyncio, "create_task", _fake_create_task)

    task = await service.start_claims_alerts_scheduler()

    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert created["name"] == "claims_alerts_scheduler"
    assert logger.warnings == ["Claims alerts scheduler loop error"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/claims-alerts-loop" not in rendered
    assert "sk-live-loop" not in rendered


@pytest.mark.asyncio
async def test_start_claims_alerts_scheduler_propagates_cancelled_error_from_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls = {"count": 0}

    async def _cancelled_run_once() -> int:
        raise asyncio.CancelledError()

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] > 1:
            raise AssertionError("scheduler continued after cancellation")

    monkeypatch.setenv("CLAIMS_ALERTS_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_ALERTS_EVAL_INTERVAL_SEC", "1")
    monkeypatch.setattr(service, "run_claims_alerts_once", _cancelled_run_once)
    monkeypatch.setattr(service.asyncio, "sleep", _fake_sleep)

    task = await service.start_claims_alerts_scheduler()

    assert task is not None
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sleep_calls["count"] == 1
