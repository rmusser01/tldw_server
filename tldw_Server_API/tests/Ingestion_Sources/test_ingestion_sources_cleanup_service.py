from __future__ import annotations

import asyncio
import importlib
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
import pytest


class _SQLitePool:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    @asynccontextmanager
    async def transaction(self):
        yield self._db


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


def _load_cleanup_service_module():
    try:
        return importlib.import_module(
            "tldw_Server_API.app.services.ingestion_sources_cleanup_service"
        )
    except ModuleNotFoundError:
        return None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_sources_for_retention_cleanup_only_returns_archive_sources(tmp_path):
    from tldw_Server_API.app.core.Ingestion_Sources import service as ingestion_service
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        await ensure_ingestion_sources_schema(db)

        await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "local_directory",
                "sink_type": "notes",
                "policy": "canonical",
                "config": {"path": str(tmp_path / "local")},
            },
        )
        archive_idle = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "config": {},
            },
        )
        archive_active = await create_source(
            db,
            user_id=2,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "notes",
                "policy": "import_only",
                "config": {},
            },
        )
        await db.execute(
            "UPDATE ingestion_source_state SET active_job_id = ? WHERE source_id = ?",
            ("job-7", int(archive_active["id"])),
        )

        rows_fn = getattr(ingestion_service, "list_sources_for_retention_cleanup", None)
        assert callable(rows_fn)

        rows = await rows_fn(db)

    assert [int(row["id"]) for row in rows] == [
        int(archive_idle["id"]),
        int(archive_active["id"]),
    ]
    assert rows[0]["source_type"] == "archive_snapshot"
    assert rows[0]["active_job_id"] in (None, "")
    assert rows[1]["source_type"] == "archive_snapshot"
    assert rows[1]["active_job_id"] == "job-7"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_cleanup_once_prunes_only_idle_archive_sources(tmp_path, monkeypatch):
    cleanup_module = _load_cleanup_service_module()
    assert cleanup_module is not None

    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        await ensure_ingestion_sources_schema(db)

        archive_idle = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "config": {},
            },
        )
        archive_active = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "notes",
                "policy": "canonical",
                "config": {},
            },
        )
        await db.execute(
            "UPDATE ingestion_source_state SET active_job_id = ? WHERE source_id = ?",
            ("job-9", int(archive_active["id"])),
        )

        async def _fake_get_db_pool():
            return _SQLitePool(db)

        pruned_source_ids: list[int] = []

        async def _fake_prune_archive_source_retention(db_conn, *, source_id: int):
            del db_conn
            pruned_source_ids.append(int(source_id))

        monkeypatch.setattr(cleanup_module, "get_db_pool", _fake_get_db_pool, raising=False)
        monkeypatch.setattr(
            cleanup_module,
            "prune_archive_source_retention",
            _fake_prune_archive_source_retention,
            raising=False,
        )

        result = await cleanup_module.run_ingestion_sources_cleanup_once()

    assert result == {
        "scanned": 2,
        "pruned": 1,
        "skipped_active": 1,
        "failed": 0,
    }
    assert pruned_source_ids == [int(archive_idle["id"])]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_cleanup_once_continues_after_prune_failure(tmp_path, monkeypatch):
    cleanup_module = _load_cleanup_service_module()
    assert cleanup_module is not None

    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        await ensure_ingestion_sources_schema(db)

        first_source = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "config": {},
            },
        )
        second_source = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "config": {},
            },
        )

        async def _fake_get_db_pool():
            return _SQLitePool(db)

        calls: list[int] = []

        async def _fake_prune_archive_source_retention(db_conn, *, source_id: int):
            del db_conn
            calls.append(int(source_id))
            if int(source_id) == int(first_source["id"]):
                raise RuntimeError("simulated prune failure")

        monkeypatch.setattr(cleanup_module, "get_db_pool", _fake_get_db_pool, raising=False)
        monkeypatch.setattr(
            cleanup_module,
            "prune_archive_source_retention",
            _fake_prune_archive_source_retention,
            raising=False,
        )

        result = await cleanup_module.run_ingestion_sources_cleanup_once()

    assert calls == [int(first_source["id"]), int(second_source["id"])]
    assert result == {
        "scanned": 2,
        "pruned": 1,
        "skipped_active": 0,
        "failed": 1,
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_cleanup_once_prune_failure_log_is_sanitized(tmp_path, monkeypatch):
    cleanup_module = _load_cleanup_service_module()
    assert cleanup_module is not None

    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
    )

    logger = _LoggerStub()
    monkeypatch.setattr(cleanup_module, "logger", logger, raising=False)

    meta_db_path = tmp_path / "ingestion_sources.sqlite3"
    async with aiosqlite.connect(str(meta_db_path)) as db:
        db.row_factory = aiosqlite.Row
        await ensure_ingestion_sources_schema(db)

        source = await create_source(
            db,
            user_id=1,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "config": {},
            },
        )

        async def _fake_get_db_pool():
            return _SQLitePool(db)

        async def _fake_prune_archive_source_retention(db_conn, *, source_id: int):
            del db_conn, source_id
            raise RuntimeError("secret /tmp/ingestion-prune sk-live-prune")

        monkeypatch.setattr(cleanup_module, "get_db_pool", _fake_get_db_pool, raising=False)
        monkeypatch.setattr(
            cleanup_module,
            "prune_archive_source_retention",
            _fake_prune_archive_source_retention,
            raising=False,
        )

        result = await cleanup_module.run_ingestion_sources_cleanup_once()

    assert result == {
        "scanned": 1,
        "pruned": 0,
        "skipped_active": 0,
        "failed": 1,
    }
    assert logger.warnings == [f"Ingestion sources cleanup failed for source_id={int(source['id'])}"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/ingestion-prune" not in rendered
    assert "sk-live-prune" not in rendered


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_cleanup_loop_failure_log_is_sanitized(monkeypatch):
    cleanup_module = _load_cleanup_service_module()
    assert cleanup_module is not None

    logger = _LoggerStub()
    monkeypatch.setattr(cleanup_module, "logger", logger, raising=False)

    async def _fail_run_once():
        raise RuntimeError("secret /tmp/ingestion-loop sk-live-loop")

    monkeypatch.setattr(cleanup_module, "run_ingestion_sources_cleanup_once", _fail_run_once, raising=False)

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(cleanup_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await cleanup_module.run_ingestion_sources_cleanup_loop()

    assert "Ingestion sources cleanup loop error" in logger.warnings
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/ingestion-loop" not in rendered
    assert "sk-live-loop" not in rendered


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_cleanup_loop_honors_stop_event_during_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_module = _load_cleanup_service_module()
    assert cleanup_module is not None

    stop_event = asyncio.Event()
    runs = {"count": 0}

    async def _run_once() -> dict[str, int]:
        runs["count"] += 1
        stop_event.set()
        return {"scanned": 0, "pruned": 0, "skipped_active": 0, "failed": 0}

    async def _unexpected_sleep(_seconds: float) -> None:
        raise AssertionError("loop should wait on stop_event instead of sleeping")

    monkeypatch.setenv("INGESTION_SOURCES_CLEANUP_INTERVAL_SEC", "3600")
    monkeypatch.setattr(cleanup_module, "run_ingestion_sources_cleanup_once", _run_once, raising=False)
    monkeypatch.setattr(cleanup_module.asyncio, "sleep", _unexpected_sleep)

    await cleanup_module.run_ingestion_sources_cleanup_loop(stop_event)

    assert runs == {"count": 1}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cleanup_scheduler_startup_accepts_single_letter_y(monkeypatch):
    cleanup_module = _load_cleanup_service_module()
    assert cleanup_module is not None

    monkeypatch.setenv("INGESTION_SOURCES_CLEANUP_ENABLED", "y")

    async def _fake_loop(stop_event: asyncio.Event | None = None) -> None:
        del stop_event
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        cleanup_module,
        "run_ingestion_sources_cleanup_loop",
        _fake_loop,
        raising=False,
    )

    task = await cleanup_module.start_ingestion_sources_cleanup_scheduler()

    assert task is not None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
