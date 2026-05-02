from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.services import media_files_cleanup_service as cleanup


pytestmark = pytest.mark.unit

_LEAK = "cleanup failed for /tmp/secret-media-token"


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows
        self.queries: list[str] = []

    def execute(self, query: str):
        self.queries.append(query)
        return _FakeCursor(self._rows)


class _FakeCleanupDb:
    def __init__(self, rows):
        self._connection = _FakeConnection(rows)

    def get_connection(self):
        return self._connection


def _capture_logs():
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    return records, sink_id


def _assert_log_sanitized(rendered: str) -> None:
    assert "cleanup failed" not in rendered
    assert "/tmp/secret-media-token" not in rendered
    assert "RuntimeError" in rendered


def test_enumerate_user_ids_base_dir_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_get_user_db_base_dir():
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(
        cleanup.DatabasePaths,
        "get_user_db_base_dir",
        _fail_get_user_db_base_dir,
    )

    records, sink_id = _capture_logs()
    try:
        assert cleanup._enumerate_user_ids() == []
    finally:
        logger.remove(sink_id)

    _assert_log_sanitized("\n".join(records))


def test_collect_known_storage_paths_query_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_get_media_db_path(_user_id):
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(
        cleanup.DatabasePaths,
        "get_media_db_path",
        _fail_get_media_db_path,
    )

    records, sink_id = _capture_logs()
    try:
        assert cleanup._collect_known_storage_paths(77) == set()
    finally:
        logger.remove(sink_id)

    _assert_log_sanitized("\n".join(records))


def test_get_storage_base_path_config_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_load_comprehensive_config():
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.load_comprehensive_config",
        _fail_load_comprehensive_config,
    )

    records, sink_id = _capture_logs()
    try:
        fallback_path = cleanup._get_storage_base_path()
    finally:
        logger.remove(sink_id)

    assert fallback_path is not None
    rendered = "\n".join(records)
    assert "cleanup failed" not in rendered
    assert "/tmp/secret-media-token" not in rendered
    assert "failed to read storage path from config" in rendered
    assert "RuntimeError" in rendered


def test_collect_known_storage_paths_uses_managed_media_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "media.db"
    db_path.write_text("")
    fake_db = _FakeCleanupDb(
        rows=[
            ("1/media/11/file-one.txt",),
            {"storage_path": "1/media/12/file-two.txt"},
        ]
    )
    captured = {}

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, **kwargs):
        captured["client_id"] = client_id
        captured.update(kwargs)
        yield fake_db

    monkeypatch.setattr(
        cleanup.DatabasePaths,
        "get_media_db_path",
        lambda user_id: str(db_path),
    )
    monkeypatch.setattr(
        cleanup,
        "MediaDatabase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("media_files_cleanup should not construct MediaDatabase directly")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cleanup,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )

    result = cleanup._collect_known_storage_paths(77)

    assert result == {
        "1/media/11/file-one.txt",
        "1/media/12/file-two.txt",
    }
    assert captured == {
        "client_id": "cleanup_service",
        "db_path": str(db_path),
        "initialize": False,
    }
    assert fake_db.get_connection().queries == [
        "SELECT storage_path FROM MediaFiles WHERE storage_path IS NOT NULL"
    ]


@pytest.mark.asyncio
async def test_cleanup_orphaned_files_removal_failure_log_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_base = tmp_path / "media_storage"
    orphan_path = storage_base / "1" / "media" / "99" / "secret-file.txt"
    orphan_path.parent.mkdir(parents=True)
    orphan_path.write_text("payload")

    def _fail_unlink():
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(cleanup, "_get_storage_base_path", lambda: storage_base)
    monkeypatch.setattr(cleanup, "_enumerate_user_ids", lambda: [])
    monkeypatch.setattr(cleanup, "GRACE_PERIOD_DAYS", 0)
    monkeypatch.setattr(Path, "unlink", lambda self: _fail_unlink() if self == orphan_path else None)
    monkeypatch.setattr(
        cleanup,
        "get_metrics_registry",
        lambda: SimpleNamespace(
            increment=lambda *args, **kwargs: None,
            observe=lambda *args, **kwargs: None,
        ),
    )

    records, sink_id = _capture_logs()
    try:
        result = await cleanup.cleanup_orphaned_files()
    finally:
        logger.remove(sink_id)

    assert result == {
        "status": "completed",
        "files_removed": 0,
        "bytes_freed": 0,
        "errors": [str(orphan_path)],
    }
    rendered = "\n".join(records)
    assert "cleanup failed" not in rendered
    assert "/tmp/secret-media-token" not in rendered
    assert str(orphan_path) not in rendered
    assert "failed to remove orphaned media file" in rendered
    assert "RuntimeError" in rendered


@pytest.mark.asyncio
async def test_cleanup_loop_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, sink_id = _capture_logs()
    metrics_calls: list[tuple[str, dict[str, object]]] = []
    sleep_calls = {"count": 0}

    async def _fake_cleanup_orphaned_files():
        raise RuntimeError("cannot clean /tmp/media-loop-secret")

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(cleanup, "cleanup_orphaned_files", _fake_cleanup_orphaned_files)
    monkeypatch.setattr(cleanup.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(
        cleanup,
        "get_metrics_registry",
        lambda: SimpleNamespace(
            increment=lambda metric, **kwargs: metrics_calls.append((metric, kwargs))
        ),
    )

    try:
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup._cleanup_loop()
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "cannot clean" not in rendered
    assert "/tmp/media-loop-secret" not in rendered
    assert "media_files_cleanup: error in cleanup cycle" in rendered
    assert "RuntimeError" in rendered
    assert metrics_calls == [
        (
            "media_files_cleanup_runs_total",
            {"labels": {"status": "error"}},
        )
    ]


@pytest.mark.asyncio
async def test_cleanup_loop_success_result_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, sink_id = _capture_logs()
    sleep_calls = {"count": 0}

    async def _fake_cleanup_orphaned_files():
        return {
            "status": "completed",
            "files_removed": 0,
            "bytes_freed": 0,
            "errors": ["/tmp/media-loop-secret"],
        }

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(cleanup, "cleanup_orphaned_files", _fake_cleanup_orphaned_files)
    monkeypatch.setattr(cleanup.asyncio, "sleep", _fake_sleep)

    try:
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup._cleanup_loop()
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "/tmp/media-loop-secret" not in rendered
    assert "media_files_cleanup: cycle completed" in rendered
    assert "files_removed=0" in rendered
    assert "bytes_freed=0" in rendered
