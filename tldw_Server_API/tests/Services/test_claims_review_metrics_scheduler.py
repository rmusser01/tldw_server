from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.services import claims_review_metrics_scheduler as service

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


def _rendered_logs(logger: _LoggerStub) -> str:
    return "\n".join(logger.debugs + logger.infos + logger.warnings)


def test_enumerate_sqlite_user_ids_base_dir_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_user_db_base_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/claims-review-base sk-live-base")),
    )

    assert service._enumerate_sqlite_user_ids() == []
    assert logger.debugs == ["claims_review_metrics: failed to resolve user db base dir"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/claims-review-base" not in _rendered_logs(logger)
    assert "sk-live-base" not in _rendered_logs(logger)


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
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/claims-review-single sk-live-single")),
    )

    assert service._enumerate_sqlite_user_ids() == []
    assert logger.debugs == ["claims_review_metrics: failed to derive single_user_id"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/claims-review-single" not in _rendered_logs(logger)
    assert "sk-live-single" not in _rendered_logs(logger)


@pytest.mark.asyncio
async def test_run_claims_review_metrics_once_postgres_user_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    media_db = SimpleNamespace(list_claims_review_user_ids=lambda: ["7"])

    class _ManagedDbContext:
        def __enter__(self):
            return media_db

        def __exit__(self, exc_type, exc, tb):
            return False

    def _aggregator(**kwargs: Any) -> int:
        raise RuntimeError("secret /tmp/claims-review-postgres-user sk-live-postgres-user")

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.content_db_settings, "backend_type", BackendType.POSTGRESQL)
    monkeypatch.setattr(service, "managed_media_database", lambda **kwargs: _ManagedDbContext())

    processed = await service.run_claims_review_metrics_once(aggregator=_aggregator)

    assert processed == 0
    assert logger.warnings == ["claims_review_metrics: aggregation failed for user 7"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/claims-review-postgres-user" not in _rendered_logs(logger)
    assert "sk-live-postgres-user" not in _rendered_logs(logger)


@pytest.mark.asyncio
async def test_run_claims_review_metrics_once_postgres_db_creation_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()

    def _raise_db_open(**kwargs: Any):
        raise RuntimeError("secret /tmp/claims-review-postgres-db sk-live-postgres-db")

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.content_db_settings, "backend_type", BackendType.POSTGRESQL)
    monkeypatch.setattr(service, "managed_media_database", _raise_db_open)

    processed = await service.run_claims_review_metrics_once(aggregator=lambda **kwargs: 1)

    assert processed == 0
    assert logger.warnings == ["claims_review_metrics: failed to create media db"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/claims-review-postgres-db" not in _rendered_logs(logger)
    assert "sk-live-postgres-db" not in _rendered_logs(logger)


@pytest.mark.asyncio
async def test_run_claims_review_metrics_once_sqlite_user_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()

    class _ManagedDbContext:
        def __enter__(self):
            return SimpleNamespace()

        def __exit__(self, exc_type, exc, tb):
            return False

    def _aggregator(**kwargs: Any) -> int:
        raise RuntimeError("secret /tmp/claims-review-sqlite-user sk-live-sqlite-user")

    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service.content_db_settings, "backend_type", BackendType.SQLITE)
    monkeypatch.setattr(service, "_enumerate_sqlite_user_ids", lambda: [13])
    monkeypatch.setattr(service.DatabasePaths, "get_media_db_path", lambda user_id: Path(f"/tmp/{user_id}.db"))
    monkeypatch.setattr(service, "managed_media_database", lambda **kwargs: _ManagedDbContext())

    processed = await service.run_claims_review_metrics_once(aggregator=_aggregator)

    assert processed == 0
    assert logger.warnings == ["claims_review_metrics: aggregation failed for user 13"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/claims-review-sqlite-user" not in _rendered_logs(logger)
    assert "sk-live-sqlite-user" not in _rendered_logs(logger)


@pytest.mark.asyncio
async def test_start_claims_review_metrics_scheduler_loop_error_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_INTERVAL_SEC", "60")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(
        service,
        "run_claims_review_metrics_once",
        lambda: (_ for _ in ()).throw(RuntimeError("secret /tmp/claims-review-loop sk-live-loop")),
    )

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError
        return None

    monkeypatch.setattr(service.asyncio, "sleep", _fake_sleep)

    task = await service.start_claims_review_metrics_scheduler()
    assert task is not None

    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert logger.warnings == ["Claims review metrics scheduler loop error"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert "/tmp/claims-review-loop" not in _rendered_logs(logger)
    assert "sk-live-loop" not in _rendered_logs(logger)


@pytest.mark.asyncio
async def test_start_claims_review_metrics_scheduler_propagates_cancelled_error_from_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls = {"count": 0}

    async def _cancelled_run_once() -> int:
        raise asyncio.CancelledError()

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] > 1:
            raise AssertionError("scheduler continued after cancellation")

    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_INTERVAL_SEC", "1")
    monkeypatch.setattr(service, "run_claims_review_metrics_once", _cancelled_run_once)
    monkeypatch.setattr(service.asyncio, "sleep", _fake_sleep)

    task = await service.start_claims_review_metrics_scheduler()

    assert task is not None
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sleep_calls["count"] == 1
