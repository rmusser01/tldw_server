"""
Tests for storage cleanup service behavior.

Focus on expired-file cleanup:
- Usage decrement via unregister_generated_file
- Safe path resolution prevents traversal deletes
"""
import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.services import storage_cleanup_service as cleanup


class TestExpiredCleanup:
    """Expired file cleanup tests."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_expired_calls_unregister_and_deletes_file(self, tmp_path, monkeypatch):
        """Expired files should decrement usage and delete the file when path is safe."""
        # Arrange: create a file under the mocked outputs dir
        file_path = tmp_path / "tts_audio" / "file.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("data")

        files_repo = AsyncMock()
        files_repo.get_expired_files = AsyncMock(return_value=[
            {"id": 1, "user_id": 1, "storage_path": "tts_audio/file.txt"},
        ])

        storage_service = AsyncMock()
        storage_service.unregister_generated_file = AsyncMock(return_value=True)

        monkeypatch.setattr(
            cleanup.DatabasePaths,
            "get_user_outputs_dir",
            lambda user_id: Path(tmp_path),
        )

        # Act
        deleted = await cleanup.cleanup_expired_files(storage_service, files_repo, batch_size=10)

        # Assert
        assert deleted == 1
        storage_service.unregister_generated_file.assert_awaited_once_with(1, hard_delete=True)
        assert not file_path.exists()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_expired_does_not_delete_unsafe_path(self, tmp_path, monkeypatch):
        """Unsafe paths should be ignored during filesystem deletion."""
        outside_path = tmp_path.parent / "outside.txt"
        outside_path.write_text("data")

        files_repo = AsyncMock()
        files_repo.get_expired_files = AsyncMock(return_value=[
            {"id": 2, "user_id": 1, "storage_path": "../outside.txt"},
        ])

        storage_service = AsyncMock()
        storage_service.unregister_generated_file = AsyncMock(return_value=True)

        monkeypatch.setattr(
            cleanup.DatabasePaths,
            "get_user_outputs_dir",
            lambda user_id: Path(tmp_path),
        )

        deleted = await cleanup.cleanup_expired_files(storage_service, files_repo, batch_size=10)

        assert deleted == 1
        storage_service.unregister_generated_file.assert_awaited_once_with(2, hard_delete=True)
        assert outside_path.exists()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_expired_skips_unlink_when_unregister_fails(self, tmp_path, monkeypatch):
        """If unregister fails, the on-disk file should remain."""
        file_path = tmp_path / "tts_audio" / "file.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("data")

        files_repo = AsyncMock()
        files_repo.get_expired_files = AsyncMock(return_value=[
            {"id": 3, "user_id": 1, "storage_path": "tts_audio/file.txt"},
        ])

        storage_service = AsyncMock()
        storage_service.unregister_generated_file = AsyncMock(return_value=False)

        monkeypatch.setattr(
            cleanup.DatabasePaths,
            "get_user_outputs_dir",
            lambda user_id: Path(tmp_path),
        )

        deleted = await cleanup.cleanup_expired_files(storage_service, files_repo, batch_size=10)

        assert deleted == 0
        storage_service.unregister_generated_file.assert_awaited_once_with(3, hard_delete=True)
        assert file_path.exists()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_expired_sanitizes_per_file_failure_log(self, tmp_path, monkeypatch):
        """Per-file cleanup failures should not leak raw exception or path details."""
        file_path = tmp_path / "tts_audio" / "file.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("data")

        files_repo = AsyncMock()
        files_repo.get_expired_files = AsyncMock(return_value=[
            {
                "id": 4,
                "user_id": 1,
                "storage_path": "tts_audio/file.txt",
            },
        ])

        storage_service = AsyncMock()
        storage_service.unregister_generated_file = AsyncMock(
            side_effect=RuntimeError("token=secret-123 failed at /private/tmp/raw-file.txt")
        )

        class _LoggerStub:
            def __init__(self):
                self.warnings = []
                self.binds = []

            def bind(self, **kwargs):
                self.binds.append(kwargs)
                return self

            def warning(self, message):
                self.warnings.append(message)

        logger_stub = _LoggerStub()

        monkeypatch.setattr(
            cleanup.DatabasePaths,
            "get_user_outputs_dir",
            lambda user_id: Path(tmp_path),
        )
        monkeypatch.setattr(cleanup, "logger", logger_stub)

        deleted = await cleanup.cleanup_expired_files(storage_service, files_repo, batch_size=10)

        assert deleted == 0
        storage_service.unregister_generated_file.assert_awaited_once_with(4, hard_delete=True)
        assert file_path.exists()
        assert logger_stub.warnings == ["Failed to cleanup expired file"]
        assert logger_stub.binds == [{"error_type": "RuntimeError"}]
        rendered_logs = repr(logger_stub.warnings)
        assert "token=secret-123" not in rendered_logs
        assert "/private/tmp/raw-file.txt" not in rendered_logs
        assert "raw-file.txt" not in rendered_logs
        assert " 4" not in rendered_logs


def test_mark_tts_history_artifact_deleted_uses_managed_media_database(
    monkeypatch: pytest.MonkeyPatch,
):
    events = []

    class _FakeDb:
        def mark_tts_history_artifacts_deleted_for_file_id(self, **kwargs):
            events.append(("mark", kwargs))

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, **kwargs):
        events.append(("open", client_id, kwargs))
        yield _FakeDb()

    monkeypatch.setattr(
        cleanup.DatabasePaths,
        "get_media_db_path",
        lambda user_id: f"/tmp/media-{user_id}.db",
    )
    monkeypatch.setattr(
        cleanup,
        "MediaDatabase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("storage_cleanup should not construct MediaDatabase directly")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cleanup,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )

    cleanup._mark_tts_history_artifact_deleted(user_id=7, file_id=11)

    assert events == [
        ("open", "storage_cleanup", {"db_path": "/tmp/media-7.db", "initialize": False}),
        (
            "mark",
            {
                "user_id": "7",
                "file_id": 11,
            },
        ),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_storage_cleanup_service_names_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_cleanup_loop(
        *,
        stop_event: asyncio.Event,
        interval_seconds: int | None,
        temp_retention_hours: int,
    ) -> None:
        await stop_event.wait()

    monkeypatch.setattr(cleanup, "run_storage_cleanup_loop", _fake_cleanup_loop)
    service = cleanup.StorageCleanupService(interval_seconds=1)

    await service.start()
    try:
        assert service._task is not None
        assert service._task.get_name() == "storage_cleanup_service"
    finally:
        await service.stop()
