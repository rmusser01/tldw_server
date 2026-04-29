from __future__ import annotations

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
