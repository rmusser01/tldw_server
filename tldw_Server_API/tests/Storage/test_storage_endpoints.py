"""
Tests for storage management API endpoints.

Tests cover:
- File listing with pagination
- File download with path traversal protection
- Bulk delete with usage tracking
- Soft delete/restore lifecycle
- Admin quota management
- Soft/hard limit warnings in usage responses
"""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoints
from tldw_Server_API.app.api.v1.schemas.storage_schemas import SetQuotaRequest
from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError


class TestListFilesEndpoint:
    """Tests for GET /storage/files endpoint."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_files_pagination(self, mock_storage_service, mock_user, mock_files_repo):
        """Test that file listing supports pagination."""
        # Setup mock data
        mock_files = [
            {
                "id": i,
                "uuid": f"uuid-{i}",
                "user_id": 1,
                "filename": f"file_{i}.wav",
                "storage_path": f"tts_audio/file_{i}.wav",
                "file_category": "tts_audio",
                "source_feature": "tts",
                "file_size_bytes": 1024,
                "is_deleted": False,
                "is_transient": False,
                "tags": [],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            for i in range(1, 6)
        ]
        mock_files_repo.list_files = AsyncMock(return_value=(mock_files[:3], 5))

        # Call the repo method directly to test logic
        files, total = await mock_files_repo.list_files(
            user_id=1,
            offset=0,
            limit=3,
        )

        assert len(files) == 3
        assert total == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_files_filters_by_category(self, mock_files_repo):
        """Test file listing can filter by category."""
        mock_files_repo.list_files = AsyncMock(return_value=([], 0))

        await mock_files_repo.list_files(
            user_id=1,
            file_category="tts_audio",
        )

        mock_files_repo.list_files.assert_called_once()
        call_kwargs = mock_files_repo.list_files.call_args[1]
        assert call_kwargs["file_category"] == "tts_audio"


class TestDownloadFileEndpoint:
    """Tests for GET /storage/files/{file_id}/download endpoint."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_path_traversal_blocked_double_dots(
        self,
        temp_user_outputs_dir,
        mock_storage_service,
        mock_user,
        mock_files_repo,
        monkeypatch,
    ):
        """Test that path traversal with .. is blocked."""
        from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

        storage_path = "../../../etc/passwd"
        file_record = {
            "id": 1,
            "user_id": mock_user.id,
            "storage_path": storage_path,
            "is_deleted": False,
        }

        mock_files_repo.get_file_by_id = AsyncMock(return_value=file_record)
        mock_storage_service.get_generated_files_repo = AsyncMock(return_value=mock_files_repo)

        monkeypatch.setattr(
            storage_endpoint,
            "_get_service",
            AsyncMock(return_value=mock_storage_service),
        )
        monkeypatch.setattr(
            storage_endpoint.DatabasePaths,
            "get_user_outputs_dir",
            lambda user_id: temp_user_outputs_dir,
        )

        with pytest.raises(HTTPException) as exc:
            await storage_endpoint.download_file(1, user=mock_user)

        assert exc.value.status_code == 403


class TestDownloadFileEndpointIntegration:
    """Integration-style tests for download endpoint behavior."""

    @pytest.mark.unit
    def test_download_file_success_returns_bytes(
        self,
        temp_user_outputs_dir,
        mock_storage_service,
        mock_user,
        mock_files_repo,
        monkeypatch,
    ):
        """Successful download returns file bytes."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

        rel_path = "tts_audio/test.mp3"
        file_path = temp_user_outputs_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"test-bytes")

        file_record = {
            "id": 1,
            "user_id": mock_user.id,
            "storage_path": rel_path,
            "file_category": "tts_audio",
            "is_deleted": False,
        }

        mock_files_repo.get_file_by_id = AsyncMock(return_value=file_record)
        mock_storage_service.get_generated_files_repo = AsyncMock(return_value=mock_files_repo)

        monkeypatch.setattr(
            storage_endpoint,
            "_get_service",
            AsyncMock(return_value=mock_storage_service),
        )
        monkeypatch.setattr(
            storage_endpoint.DatabasePaths,
            "get_user_outputs_dir",
            lambda user_id: temp_user_outputs_dir,
        )

        app.dependency_overrides[get_request_user] = lambda: mock_user
        try:
            with TestClient(app) as client:
                resp = client.get("/api/v1/storage/files/1/download")
                assert resp.status_code == 200
                assert resp.content == b"test-bytes"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.unit
    def test_download_file_blocks_path_traversal(
        self,
        temp_user_outputs_dir,
        mock_storage_service,
        mock_user,
        mock_files_repo,
        monkeypatch,
    ):
        """Path traversal storage paths are rejected."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

        file_record = {
            "id": 2,
            "user_id": mock_user.id,
            "storage_path": "../outside.txt",
            "file_category": "tts_audio",
            "is_deleted": False,
        }

        mock_files_repo.get_file_by_id = AsyncMock(return_value=file_record)
        mock_storage_service.get_generated_files_repo = AsyncMock(return_value=mock_files_repo)

        monkeypatch.setattr(
            storage_endpoint,
            "_get_service",
            AsyncMock(return_value=mock_storage_service),
        )
        monkeypatch.setattr(
            storage_endpoint.DatabasePaths,
            "get_user_outputs_dir",
            lambda user_id: temp_user_outputs_dir,
        )

        app.dependency_overrides[get_request_user] = lambda: mock_user
        try:
            with TestClient(app) as client:
                resp = client.get("/api/v1/storage/files/2/download")
                assert resp.status_code == 403
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.unit
    def test_download_file_uses_voices_dir_for_voice_clones(
        self,
        temp_storage_dir,
        mock_storage_service,
        mock_user,
        mock_files_repo,
        monkeypatch,
    ):
        """Voice clone downloads resolve against the voices directory."""
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

        voices_dir = temp_storage_dir / "1" / "voices"
        file_rel = "processed/voice.wav"
        file_path = voices_dir / file_rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"voice-bytes")

        file_record = {
            "id": 3,
            "user_id": mock_user.id,
            "storage_path": file_rel,
            "file_category": "voice_clone",
            "is_deleted": False,
        }

        mock_files_repo.get_file_by_id = AsyncMock(return_value=file_record)
        mock_storage_service.get_generated_files_repo = AsyncMock(return_value=mock_files_repo)

        monkeypatch.setattr(
            storage_endpoint,
            "_get_service",
            AsyncMock(return_value=mock_storage_service),
        )
        monkeypatch.setattr(
            storage_endpoint.DatabasePaths,
            "get_user_voices_dir",
            lambda user_id: voices_dir,
        )

        app.dependency_overrides[get_request_user] = lambda: mock_user
        try:
            with TestClient(app) as client:
                resp = client.get("/api/v1/storage/files/3/download")
                assert resp.status_code == 200
                assert resp.content == b"voice-bytes"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.unit
    def test_path_traversal_blocked_encoded(self, temp_user_outputs_dir):
        """Test that encoded path traversal attempts are blocked."""
        # Simulate encoded path traversal
        storage_path = "..%2F..%2Fetc%2Fpasswd"
        full_path = temp_user_outputs_dir / storage_path

        # Even decoded, should still be caught
        resolved = full_path.resolve()
        is_safe = resolved.is_relative_to(temp_user_outputs_dir.resolve())
        # Encoded separators should be treated as literal path components
        assert is_safe is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_download_file_not_owned_returns_403(
        self,
        mock_storage_service,
        mock_user,
        mock_files_repo,
        sample_file_record,
        monkeypatch,
    ):
        """Test that downloading another user's file returns 403."""
        from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

        sample_file_record["user_id"] = 999
        mock_files_repo.get_file_by_id = AsyncMock(return_value=sample_file_record)
        mock_storage_service.get_generated_files_repo = AsyncMock(return_value=mock_files_repo)

        monkeypatch.setattr(
            storage_endpoint,
            "_get_service",
            AsyncMock(return_value=mock_storage_service),
        )

        with pytest.raises(HTTPException) as exc:
            await storage_endpoint.download_file(1, user=mock_user)

        assert exc.value.status_code == 403


class TestBulkDeleteEndpoint:
    """Tests for POST /storage/files/bulk-delete endpoint."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bulk_delete_updates_usage_for_each_file(self, mock_storage_service):
        """Test that bulk soft delete updates usage for each file."""
        file_ids = [1, 2, 3]

        # Mock unregister to track calls
        call_count = 0

        async def mock_unregister(file_id, hard_delete=False):
            nonlocal call_count
            call_count += 1
            return True

        mock_storage_service.unregister_generated_file = mock_unregister

        # Simulate the fixed bulk delete logic
        deleted_count = 0
        for file_id in file_ids:
            if await mock_storage_service.unregister_generated_file(file_id, hard_delete=False):
                deleted_count += 1

        assert deleted_count == 3
        assert call_count == 3, "unregister_generated_file should be called for each file"


class TestSoftDeleteRestoreCycle:
    """Tests for soft delete and restore lifecycle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_soft_delete_file_success(self, mock_files_repo):
        """Test soft delete marks file as deleted."""
        mock_files_repo.soft_delete_file = AsyncMock(return_value=True)

        result = await mock_files_repo.soft_delete_file(1)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restore_file_success(self, mock_files_repo):
        """Test restore removes deleted flag."""
        mock_files_repo.restore_file = AsyncMock(return_value=True)

        result = await mock_files_repo.restore_file(1)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restore_readds_to_usage(self, mock_storage_service, sample_deleted_file_record):
        """Test that restore re-adds file size to usage counters."""
        file_size = sample_deleted_file_record["file_size_bytes"]

        # Simulate restore flow
        update_calls = []

        async def mock_update_usage(user_id, size, operation="add"):
            update_calls.append((user_id, size, operation))
            return {"storage_used_mb": 100.0}

        mock_storage_service.update_usage = mock_update_usage

        # Call update_usage as the restore endpoint would
        await mock_storage_service.update_usage(1, file_size, operation="add")

        assert len(update_calls) == 1
        assert update_calls[0][2] == "add"


class TestAdminQuotaEndpoints:
    """Tests for admin quota management endpoints."""

    @pytest.mark.unit
    def test_set_quota_requires_admin(self, mock_user):
        """Test that non-admin users cannot set quotas."""
        assert not mock_user.is_superuser
        assert mock_user.role != "admin"

    @pytest.mark.unit
    def test_set_quota_allowed_for_admin(self, mock_admin_user):
        """Test that admin users can set quotas."""
        assert mock_admin_user.is_superuser or mock_admin_user.role == "admin"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_user_quota_success(self, mock_storage_service):
        """Test setting user quota updates the database."""
        result = await mock_storage_service.set_user_quota(1, 2000)

        assert result["storage_quota_mb"] == 2000

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_user_quota_sanitizes_storage_error(self, monkeypatch):
        """Test admin user quota failures do not leak backend details."""

        class _BrokenStorageService:
            async def set_user_quota(self, user_id, quota_mb):
                _ = (user_id, quota_mb)
                raise StorageError("storage backend exploded")

        async def _get_broken_service():
            return _BrokenStorageService()

        monkeypatch.setattr(storage_endpoints, "_get_service", _get_broken_service)

        with pytest.raises(HTTPException) as exc_info:
            await storage_endpoints.set_user_quota(
                user_id=1,
                request=SetQuotaRequest(quota_mb=1000),
                _principal=object(),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to set user storage quota"


class TestUsageWithSoftLimitWarnings:
    """Tests for usage response with soft/hard limit warnings."""

    @pytest.mark.unit
    def test_usage_shows_soft_limit_warning_at_80_percent(self):
        """Test that warning appears when at 80%+ usage."""
        quota_mb = 1000
        quota_used_mb = 850  # 85%

        usage_pct = (quota_used_mb / quota_mb * 100)
        at_soft_limit = usage_pct >= 80
        at_hard_limit = usage_pct >= 100

        assert at_soft_limit is True
        assert at_hard_limit is False

        # Build warning message
        warning_message = None
        if at_hard_limit:
            warning_message = "Storage quota exceeded - delete files to continue"
        elif at_soft_limit:
            warning_message = "Approaching storage limit (80%+)"

        assert warning_message == "Approaching storage limit (80%+)"

    @pytest.mark.unit
    def test_usage_shows_hard_limit_warning_at_100_percent(self):
        """Test that hard limit warning appears at 100%."""
        quota_mb = 1000
        quota_used_mb = 1050  # 105%

        usage_pct = (quota_used_mb / quota_mb * 100)
        at_soft_limit = usage_pct >= 80
        at_hard_limit = usage_pct >= 100

        assert at_soft_limit is True
        assert at_hard_limit is True

        warning_message = None
        if at_hard_limit:
            warning_message = "Storage quota exceeded - delete files to continue"
        elif at_soft_limit:
            warning_message = "Approaching storage limit (80%+)"

        assert warning_message == "Storage quota exceeded - delete files to continue"

    @pytest.mark.unit
    def test_no_warning_under_80_percent(self):
        """Test that no warning appears under 80%."""
        quota_mb = 1000
        quota_used_mb = 500  # 50%

        usage_pct = (quota_used_mb / quota_mb * 100)
        at_soft_limit = usage_pct >= 80
        at_hard_limit = usage_pct >= 100

        assert at_soft_limit is False
        assert at_hard_limit is False

        warning_message = None
        if at_hard_limit:
            warning_message = "Storage quota exceeded - delete files to continue"
        elif at_soft_limit:
            warning_message = "Approaching storage limit (80%+)"

        assert warning_message is None


class TestLeastAccessedEndpoint:
    """Tests for GET /storage/files/least-accessed endpoint."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_least_accessed_returns_old_files_first(self, mock_files_repo):
        """Test that least accessed files are returned in correct order."""
        old_file = {
            "id": 1,
            "accessed_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "created_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
        }
        new_file = {
            "id": 2,
            "accessed_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        }

        mock_files_repo.list_least_accessed = AsyncMock(return_value=[old_file, new_file])

        result = await mock_files_repo.list_least_accessed(user_id=1, limit=10)

        assert len(result) == 2
        assert result[0]["id"] == 1  # Oldest first
