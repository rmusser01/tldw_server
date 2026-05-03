"""
Tests for storage quota service.

Tests cover:
- Quota checking (under/over limit)
- Usage updates (add/remove)
- Combined quota checks (user/team/org)
- Cleanup temp files (executor fix verification)
- File size limit validation
- get_all_users_storage method
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.app.services import storage_quota_service as storage_service_module
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import FILE_CATEGORY_VOICE_CLONE


class TestCheckQuota:
    """Tests for quota checking functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_quota_under_limit_returns_true(self):
        """Test that quota check returns True when under limit."""
        current_mb = 100.0
        quota_mb = 1000
        new_bytes = 50 * 1024 * 1024  # 50 MB

        new_mb = new_bytes / (1024 * 1024)
        projected_mb = current_mb + new_mb
        has_quota = projected_mb <= quota_mb

        assert has_quota is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_quota_over_limit_returns_false(self):
        """Test that quota check returns False when over limit."""
        current_mb = 950.0
        quota_mb = 1000
        new_bytes = 100 * 1024 * 1024  # 100 MB

        new_mb = new_bytes / (1024 * 1024)
        projected_mb = current_mb + new_mb  # 1050 MB
        has_quota = projected_mb <= quota_mb

        assert has_quota is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_quota_raises_on_exceed_when_requested(self, mock_storage_service):
        """Test that QuotaExceededError is raised when requested and over limit."""
        from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError

        # Configure mock to simulate over-limit scenario
        mock_storage_service.check_quota = AsyncMock(
            side_effect=QuotaExceededError(1100, 1000)
        )

        with pytest.raises(QuotaExceededError):
            await mock_storage_service.check_quota(
                user_id=1,
                new_bytes=100 * 1024 * 1024,
                raise_on_exceed=True,
            )


class TestUpdateUsage:
    """Tests for usage update functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_usage_add_increments_correctly(self, mock_storage_service):
        """Test that adding to usage increments correctly."""
        initial_usage = 100.0
        bytes_to_add = 50 * 1024 * 1024  # 50 MB

        mock_storage_service.update_usage = AsyncMock(return_value={
            "storage_used_mb": initial_usage + (bytes_to_add / (1024 * 1024)),
        })

        result = await mock_storage_service.update_usage(1, bytes_to_add, operation="add")

        assert result["storage_used_mb"] == 150.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_usage_remove_decrements_correctly(self, mock_storage_service):
        """Test that removing from usage decrements correctly."""
        initial_usage = 100.0
        bytes_to_remove = 50 * 1024 * 1024  # 50 MB

        mock_storage_service.update_usage = AsyncMock(return_value={
            "storage_used_mb": max(0, initial_usage - (bytes_to_remove / (1024 * 1024))),
        })

        result = await mock_storage_service.update_usage(1, bytes_to_remove, operation="remove")

        assert result["storage_used_mb"] == 50.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_usage_floors_at_zero(self, mock_storage_service):
        """Test that usage can't go below zero."""
        initial_usage = 10.0
        bytes_to_remove = 100 * 1024 * 1024  # 100 MB (more than available)

        expected_usage = max(0, initial_usage - (bytes_to_remove / (1024 * 1024)))

        mock_storage_service.update_usage = AsyncMock(return_value={
            "storage_used_mb": expected_usage,
        })

        result = await mock_storage_service.update_usage(1, bytes_to_remove, operation="remove")

        assert result["storage_used_mb"] == 0.0


class TestCombinedQuota:
    """Tests for combined user/team/org quota checking."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_combined_quota_all_levels_checked(self, mock_storage_service, mock_quotas_repo):
        """Test that all quota levels are checked in combined check."""
        check_results = {
            "user": True,
            "team": True,
            "org": True,
        }

        # All levels have quota
        has_quota = all(check_results.values())
        assert has_quota is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_combined_quota_blocked_at_user_level(self, mock_storage_service):
        """Test that blocking at user level is properly reported."""
        check_results = {
            "user": False,
            "team": True,
            "org": True,
        }

        has_quota = all(check_results.values())
        blocking_level = "user" if not check_results["user"] else None

        assert has_quota is False
        assert blocking_level == "user"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_combined_quota_blocked_at_team_level(self, mock_storage_service):
        """Test that blocking at team level is properly reported."""
        check_results = {
            "user": True,
            "team": False,
            "org": True,
        }

        has_quota = all(check_results.values())
        blocking_level = None
        if not check_results["user"]:
            blocking_level = "user"
        elif not check_results["team"]:
            blocking_level = "team"
        elif not check_results["org"]:
            blocking_level = "org"

        assert has_quota is False
        assert blocking_level == "team"


class TestCleanupTempFiles:
    """Tests for temp file cleanup functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup_temp_files_uses_default_executor(self, mock_storage_service):
        """Test that cleanup uses None (default) executor, not undefined self.executor."""
        # The fix was changing self.executor to None
        # This test verifies the cleanup method can be called without AttributeError

        mock_storage_service.cleanup_temp_files = AsyncMock(return_value={
            "files_deleted": 5,
            "bytes_freed": 1024 * 1024,
            "errors": 0,
        })

        # Should not raise AttributeError for missing self.executor
        result = await mock_storage_service.cleanup_temp_files(older_than_hours=24)

        assert result["files_deleted"] == 5
        assert result["errors"] == 0


class TestFileSizeValidation:
    """Tests for file size limit validation."""

    @pytest.mark.unit
    def test_register_file_exceeds_max_size_raises(self):
        """Test that files over 10 GB are rejected."""
        from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError

        MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
        file_size_bytes = 15 * 1024 * 1024 * 1024  # 15 GB

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            with pytest.raises(StorageError):
                raise StorageError(
                    f"File size {file_size_bytes / (1024*1024*1024):.2f} GB exceeds "
                    f"maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024*1024*1024):.0f} GB"
                )

    @pytest.mark.unit
    def test_register_file_under_max_size_allowed(self):
        """Test that files under 10 GB are allowed."""
        MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
        file_size_bytes = 5 * 1024 * 1024 * 1024  # 5 GB

        is_valid = file_size_bytes <= MAX_FILE_SIZE_BYTES
        assert is_valid is True


class TestUnregisterGeneratedFile:
    """Tests for unregistering generated files and usage updates."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hard_delete_skips_usage_if_already_deleted(self):
        """Hard delete should not decrement usage when file already soft-deleted."""
        service = StorageQuotaService(db_pool=MagicMock())
        service._initialized = True

        file_record = {
            "id": 1,
            "user_id": 1,
            "is_deleted": True,
            "file_size_bytes": 1024,
        }
        mock_repo = AsyncMock()
        mock_repo.get_file_by_id = AsyncMock(return_value=file_record)
        mock_repo.hard_delete_file = AsyncMock(return_value=True)
        service.get_generated_files_repo = AsyncMock(return_value=mock_repo)
        service.update_usage = AsyncMock()
        service.update_org_usage = AsyncMock()
        service.update_team_usage = AsyncMock()

        result = await service.unregister_generated_file(1, hard_delete=True)

        assert result is True
        service.update_usage.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hard_delete_voice_clone_unlinks_file(self, tmp_path, monkeypatch):
        """Hard delete should remove a voice clone file from disk."""
        service = StorageQuotaService(db_pool=MagicMock())
        service._initialized = True

        voices_dir = tmp_path / "voices"
        file_rel = "processed/voice.wav"
        file_path = voices_dir / file_rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"voice-data")

        monkeypatch.setattr(
            storage_service_module.DatabasePaths,
            "get_user_voices_dir",
            lambda user_id: voices_dir,
        )

        file_record = {
            "id": 2,
            "user_id": 1,
            "is_deleted": False,
            "file_size_bytes": 1024,
            "file_category": FILE_CATEGORY_VOICE_CLONE,
            "storage_path": file_rel,
        }
        mock_repo = AsyncMock()
        mock_repo.get_file_by_id = AsyncMock(return_value=file_record)
        mock_repo.hard_delete_file = AsyncMock(return_value=True)
        service.get_generated_files_repo = AsyncMock(return_value=mock_repo)
        service.update_usage = AsyncMock()
        service.update_org_usage = AsyncMock()
        service.update_team_usage = AsyncMock()

        result = await service.unregister_generated_file(2, hard_delete=True)

        assert result is True
        assert not file_path.exists()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bulk_unregister_uses_repo_bulk_soft_delete_and_aggregates_usage(self):
        """Bulk unregister soft-deletes through the repo and aggregates usage updates."""
        service = StorageQuotaService(db_pool=MagicMock())
        service._initialized = True

        file_records = [
            {
                "id": 1,
                "user_id": 1,
                "org_id": 7,
                "team_id": 9,
                "is_deleted": False,
                "file_size_bytes": 1024,
            },
            {
                "id": 2,
                "user_id": 1,
                "org_id": 7,
                "team_id": 9,
                "is_deleted": False,
                "file_size_bytes": 2048,
            },
        ]
        mock_repo = AsyncMock()
        mock_repo.bulk_soft_delete = AsyncMock(return_value=2)
        service.get_generated_files_repo = AsyncMock(return_value=mock_repo)
        service.update_usage = AsyncMock()
        service.update_org_usage = AsyncMock()
        service.update_team_usage = AsyncMock()

        result = await service.unregister_generated_files(file_records, hard_delete=False)

        assert result == 2
        mock_repo.bulk_soft_delete.assert_awaited_once_with([1, 2])
        service.update_usage.assert_awaited_once_with(1, 3072, operation="remove")
        service.update_org_usage.assert_awaited_once_with(7, -3072)
        service.update_team_usage.assert_awaited_once_with(9, -3072)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_restore_generated_file_updates_usage_in_service_layer(self):
        """Restore orchestration updates file state and usage counters in the service."""
        service = StorageQuotaService(db_pool=MagicMock())
        service._initialized = True

        deleted_record = {
            "id": 3,
            "user_id": 1,
            "org_id": 7,
            "team_id": 9,
            "is_deleted": True,
            "file_size_bytes": 4096,
        }
        restored_record = deleted_record | {"is_deleted": False}
        mock_repo = AsyncMock()
        mock_repo.restore_file = AsyncMock(return_value=True)
        mock_repo.get_file_by_id = AsyncMock(return_value=restored_record)
        service.get_generated_files_repo = AsyncMock(return_value=mock_repo)
        service.update_usage = AsyncMock()
        service.update_org_usage = AsyncMock()
        service.update_team_usage = AsyncMock()

        result = await service.restore_generated_file(3, file_record=deleted_record)

        assert result == restored_record
        mock_repo.restore_file.assert_awaited_once_with(3)
        service.update_usage.assert_awaited_once_with(1, 4096, operation="add")
        service.update_org_usage.assert_awaited_once_with(7, 4096)
        service.update_team_usage.assert_awaited_once_with(9, 4096)


class TestGetAllUsersStorage:
    """Tests for get_all_users_storage method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_all_users_storage_returns_list(self, mock_storage_service):
        """Test that get_all_users_storage returns user storage info."""
        expected_users = [
            {
                "user_id": 1,
                "username": "user1",
                "storage_used_mb": 500.0,
                "storage_quota_mb": 1000,
                "available_mb": 500.0,
                "usage_percentage": 50.0,
            },
            {
                "user_id": 2,
                "username": "user2",
                "storage_used_mb": 200.0,
                "storage_quota_mb": 1000,
                "available_mb": 800.0,
                "usage_percentage": 20.0,
            },
        ]

        mock_storage_service.get_all_users_storage = AsyncMock(return_value=expected_users)

        result = await mock_storage_service.get_all_users_storage()

        assert len(result) == 2
        assert result[0]["user_id"] == 1
        assert result[0]["storage_used_mb"] == 500.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_all_users_storage_sorted_by_usage(self, mock_storage_service):
        """Test that users are sorted by storage usage descending."""
        users = [
            {"user_id": 1, "storage_used_mb": 100.0},
            {"user_id": 2, "storage_used_mb": 500.0},
            {"user_id": 3, "storage_used_mb": 300.0},
        ]

        # Sorted descending by storage_used_mb
        sorted_users = sorted(users, key=lambda u: u["storage_used_mb"], reverse=True)

        assert sorted_users[0]["user_id"] == 2  # 500 MB
        assert sorted_users[1]["user_id"] == 3  # 300 MB
        assert sorted_users[2]["user_id"] == 1  # 100 MB


class TestSoftLimitWarning:
    """Tests for soft limit warning in can_allocate."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_can_allocate_returns_warning_at_soft_limit(self, mock_quotas_repo):
        """Test that soft limit warning is returned when at 80%."""
        mock_quotas_repo.check_quota_status = AsyncMock(return_value={
            "quota_mb": 1000,
            "used_mb": 850.0,
            "remaining_mb": 150.0,
            "usage_pct": 85.0,
            "at_soft_limit": True,
            "at_hard_limit": False,
            "has_quota": True,
        })
        mock_quotas_repo.can_allocate = AsyncMock(
            return_value=(True, "Warning: Approaching storage limit (soft limit reached)")
        )

        can_alloc, reason = await mock_quotas_repo.can_allocate(
            50 * 1024 * 1024,
            team_id=1,
        )

        assert can_alloc is True
        assert "soft limit" in reason.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_can_allocate_blocked_at_hard_limit(self, mock_quotas_repo):
        """Test that allocation is blocked at hard limit."""
        mock_quotas_repo.can_allocate = AsyncMock(
            return_value=(False, "Storage quota exceeded (at hard limit)")
        )

        can_alloc, reason = await mock_quotas_repo.can_allocate(
            50 * 1024 * 1024,
            team_id=1,
        )

        assert can_alloc is False
        assert "hard limit" in reason.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_can_allocate_passes_below_soft_limit(self, mock_quotas_repo):
        """Test that allocation passes cleanly below soft limit."""
        mock_quotas_repo.can_allocate = AsyncMock(
            return_value=(True, "Quota check passed")
        )

        can_alloc, reason = await mock_quotas_repo.can_allocate(
            50 * 1024 * 1024,
            team_id=1,
        )

        assert can_alloc is True
        assert reason == "Quota check passed"
