"""
Tests for admin storage quota endpoints and quota enforcement utility.

Covers:
- Admin endpoint response structures
- Quota enforcement check logic
- Edge cases (no quota set, at limits, insufficient space)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Storage.quota_enforcement import check_storage_quota
from tldw_Server_API.app.api.v1.endpoints.admin import admin_storage_quotas as quotas_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pool():
    """Build a MagicMock pool for tests."""
    mock_pool = MagicMock()
    mock_pool.pool = None  # SQLite mode
    return mock_pool


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: Any, **_kwargs: Any) -> None:
        if args:
            self.warnings.append(str(message).format(*args))
        else:
            self.warnings.append(str(message))


def _assert_sanitized_warning_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warnings == [expected_message]
    rendered = " ".join(logger_stub.warnings)
    assert "/private/" not in rendered
    assert "exploded" not in rendered


# ---------------------------------------------------------------------------
# Quota enforcement unit tests
# ---------------------------------------------------------------------------


class TestCheckStorageQuota:
    """Tests for the check_storage_quota utility function."""

    @pytest.mark.asyncio
    async def test_no_quota_set_allows_upload(self):
        """When no quota is configured, uploads are allowed."""
        mock_pool = _make_mock_pool()

        no_quota_status = {
            "quota_mb": None,
            "used_mb": 0.0,
            "remaining_mb": None,
            "usage_pct": 0.0,
            "at_soft_limit": False,
            "at_hard_limit": False,
            "has_quota": False,
        }

        with patch(
            "tldw_Server_API.app.core.Storage.quota_enforcement.AuthnzStorageQuotasRepo"
        ) as MockRepo:
            instance = MockRepo.return_value
            instance.check_quota_status = AsyncMock(return_value=no_quota_status)

            result = await check_storage_quota(
                user_id=1,
                file_size_bytes=10 * 1024 * 1024,  # 10 MB
                db_pool=mock_pool,
                org_id=1,
            )

        assert result["allowed"] is True
        assert result["quota_mb"] is None
        assert result["remaining_mb"] is None

    @pytest.mark.asyncio
    async def test_within_quota_allows_upload(self):
        """Upload is allowed when under quota."""
        mock_pool = _make_mock_pool()

        within_quota_status = {
            "quota_mb": 1000,
            "used_mb": 100.0,
            "remaining_mb": 900.0,
            "usage_pct": 10.0,
            "at_soft_limit": False,
            "at_hard_limit": False,
            "has_quota": True,
        }

        with patch(
            "tldw_Server_API.app.core.Storage.quota_enforcement.AuthnzStorageQuotasRepo"
        ) as MockRepo:
            instance = MockRepo.return_value
            instance.check_quota_status = AsyncMock(return_value=within_quota_status)

            result = await check_storage_quota(
                user_id=1,
                file_size_bytes=50 * 1024 * 1024,  # 50 MB
                db_pool=mock_pool,
                org_id=1,
            )

        assert result["allowed"] is True
        assert result["quota_mb"] == 1000
        assert result["reason"] == "Quota check passed"

    @pytest.mark.asyncio
    async def test_exceeds_quota_blocks_upload(self):
        """Upload is blocked when file would exceed remaining quota."""
        mock_pool = _make_mock_pool()

        tight_quota_status = {
            "quota_mb": 100,
            "used_mb": 95.0,
            "remaining_mb": 5.0,
            "usage_pct": 95.0,
            "at_soft_limit": True,
            "at_hard_limit": False,
            "has_quota": True,
        }

        with patch(
            "tldw_Server_API.app.core.Storage.quota_enforcement.AuthnzStorageQuotasRepo"
        ) as MockRepo:
            instance = MockRepo.return_value
            instance.check_quota_status = AsyncMock(return_value=tight_quota_status)

            result = await check_storage_quota(
                user_id=1,
                file_size_bytes=10 * 1024 * 1024,  # 10 MB > 5 MB remaining
                db_pool=mock_pool,
                org_id=1,
            )

        assert result["allowed"] is False
        assert "Insufficient" in result["reason"]

    @pytest.mark.asyncio
    async def test_hard_limit_blocks_upload(self):
        """Upload is blocked when hard limit is reached."""
        mock_pool = _make_mock_pool()

        hard_limit_status = {
            "quota_mb": 100,
            "used_mb": 100.0,
            "remaining_mb": 0.0,
            "usage_pct": 100.0,
            "at_soft_limit": True,
            "at_hard_limit": True,
            "has_quota": True,
        }

        with patch(
            "tldw_Server_API.app.core.Storage.quota_enforcement.AuthnzStorageQuotasRepo"
        ) as MockRepo:
            instance = MockRepo.return_value
            instance.check_quota_status = AsyncMock(return_value=hard_limit_status)

            result = await check_storage_quota(
                user_id=1,
                file_size_bytes=1024,  # even tiny file
                db_pool=mock_pool,
                org_id=1,
            )

        assert result["allowed"] is False
        assert "hard limit" in result["reason"]

    @pytest.mark.asyncio
    async def test_soft_limit_allows_with_warning(self):
        """Upload is allowed at soft limit but with a warning."""
        mock_pool = _make_mock_pool()

        soft_limit_status = {
            "quota_mb": 100,
            "used_mb": 85.0,
            "remaining_mb": 15.0,
            "usage_pct": 85.0,
            "at_soft_limit": True,
            "at_hard_limit": False,
            "has_quota": True,
        }

        with patch(
            "tldw_Server_API.app.core.Storage.quota_enforcement.AuthnzStorageQuotasRepo"
        ) as MockRepo:
            instance = MockRepo.return_value
            instance.check_quota_status = AsyncMock(return_value=soft_limit_status)

            result = await check_storage_quota(
                user_id=1,
                file_size_bytes=1 * 1024 * 1024,  # 1 MB, fits in 15 MB remaining
                db_pool=mock_pool,
                org_id=1,
            )

        assert result["allowed"] is True
        assert "Warning" in result["reason"]

    @pytest.mark.asyncio
    async def test_quota_check_error_fails_open(self):
        """If the quota check itself raises, we fail-open (allow)."""
        mock_pool = _make_mock_pool()

        with patch(
            "tldw_Server_API.app.core.Storage.quota_enforcement.AuthnzStorageQuotasRepo"
        ) as MockRepo:
            instance = MockRepo.return_value
            instance.check_quota_status = AsyncMock(
                side_effect=RuntimeError("DB unavailable")
            )

            result = await check_storage_quota(
                user_id=1,
                file_size_bytes=1024,
                db_pool=mock_pool,
                org_id=1,
            )

        assert result["allowed"] is True
        assert "fail-open" in result["reason"]

    @pytest.mark.asyncio
    async def test_team_quota_check(self):
        """Quota check works with team_id instead of org_id."""
        mock_pool = _make_mock_pool()

        team_status = {
            "quota_mb": 500,
            "used_mb": 200.0,
            "remaining_mb": 300.0,
            "usage_pct": 40.0,
            "at_soft_limit": False,
            "at_hard_limit": False,
            "has_quota": True,
        }

        with patch(
            "tldw_Server_API.app.core.Storage.quota_enforcement.AuthnzStorageQuotasRepo"
        ) as MockRepo:
            instance = MockRepo.return_value
            instance.check_quota_status = AsyncMock(return_value=team_status)

            result = await check_storage_quota(
                user_id=1,
                file_size_bytes=50 * 1024 * 1024,
                db_pool=mock_pool,
                team_id=10,
            )

        assert result["allowed"] is True
        instance.check_quota_status.assert_called_once_with(team_id=10)


class TestAdminStorageQuotaSummaryPagination:
    @pytest.mark.asyncio
    async def test_get_storage_quota_summary_returns_canonical_pagination(self, monkeypatch):
        repo = MagicMock()
        repo.list_all_quotas = AsyncMock(
            return_value=[
                {"id": 1, "quota_mb": 1000},
                {"id": 2, "quota_mb": 2000},
            ]
        )
        monkeypatch.setattr(quotas_module, "_get_repo", AsyncMock(return_value=repo))

        result = await quotas_module.get_storage_quota_summary(offset=10, limit=2)

        assert result.total_quotas == 2
        assert result.limit == 2
        assert result.offset == 10
        assert result.pagination.model_dump(mode="json") == {
            "mode": "offset",
            "limit": 2,
            "offset": 10,
            "total": None,
            "has_more": True,
            "next_offset": 12,
        }
        assert result.has_more is True
        assert result.next_offset == 12
        repo.list_all_quotas.assert_awaited_once_with(offset=10, limit=2)


class TestAdminStorageQuotaEndpointErrorMapping:
    @pytest.mark.asyncio
    async def test_get_user_storage_quota_sanitizes_generic_failure(self, monkeypatch):
        logger_stub = _LoggerStub()
        repo = MagicMock()
        repo.check_quota_status = AsyncMock(
            side_effect=RuntimeError("quota backend exploded at /private/storage-quota.db")
        )
        monkeypatch.setattr(quotas_module, "logger", logger_stub)
        monkeypatch.setattr(quotas_module, "_get_repo", AsyncMock(return_value=repo))

        with pytest.raises(HTTPException) as exc_info:
            await quotas_module.get_user_storage_quota(1)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to retrieve storage quota"
        _assert_sanitized_warning_log(logger_stub, "Failed to get user storage quota")

    @pytest.mark.asyncio
    async def test_update_user_storage_quota_sanitizes_generic_failure(self, monkeypatch):
        logger_stub = _LoggerStub()
        repo = MagicMock()
        repo.upsert_org_quota = AsyncMock(
            side_effect=RuntimeError("quota backend exploded at /private/storage-quota.db")
        )
        monkeypatch.setattr(quotas_module, "logger", logger_stub)
        monkeypatch.setattr(quotas_module, "_get_repo", AsyncMock(return_value=repo))

        with pytest.raises(HTTPException) as exc_info:
            await quotas_module.update_user_storage_quota(
                1,
                quotas_module.UpdateQuotaRequest(quota_mb=1000),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to update storage quota"
        _assert_sanitized_warning_log(logger_stub, "Failed to update user storage quota")

    @pytest.mark.asyncio
    async def test_get_org_storage_quota_sanitizes_generic_failure(self, monkeypatch):
        logger_stub = _LoggerStub()
        repo = MagicMock()
        repo.check_quota_status = AsyncMock(
            side_effect=RuntimeError("quota backend exploded at /private/storage-quota.db")
        )
        monkeypatch.setattr(quotas_module, "logger", logger_stub)
        monkeypatch.setattr(quotas_module, "_get_repo", AsyncMock(return_value=repo))

        with pytest.raises(HTTPException) as exc_info:
            await quotas_module.get_org_storage_quota(1)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to retrieve org storage quota"
        _assert_sanitized_warning_log(logger_stub, "Failed to get org storage quota")

    @pytest.mark.asyncio
    async def test_update_org_storage_quota_sanitizes_generic_failure(self, monkeypatch):
        logger_stub = _LoggerStub()
        repo = MagicMock()
        repo.upsert_org_quota = AsyncMock(
            side_effect=RuntimeError("quota backend exploded at /private/storage-quota.db")
        )
        monkeypatch.setattr(quotas_module, "logger", logger_stub)
        monkeypatch.setattr(quotas_module, "_get_repo", AsyncMock(return_value=repo))

        with pytest.raises(HTTPException) as exc_info:
            await quotas_module.update_org_storage_quota(
                1,
                quotas_module.UpdateQuotaRequest(quota_mb=1000),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to update org storage quota"
        _assert_sanitized_warning_log(logger_stub, "Failed to update org storage quota")

    @pytest.mark.asyncio
    async def test_get_storage_quota_summary_sanitizes_generic_failure(self, monkeypatch):
        logger_stub = _LoggerStub()
        repo = MagicMock()
        repo.list_all_quotas = AsyncMock(
            side_effect=RuntimeError("quota backend exploded at /private/storage-quota.db")
        )
        monkeypatch.setattr(quotas_module, "logger", logger_stub)
        monkeypatch.setattr(quotas_module, "_get_repo", AsyncMock(return_value=repo))

        with pytest.raises(HTTPException) as exc_info:
            await quotas_module.get_storage_quota_summary()

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to retrieve storage quota summary"
        _assert_sanitized_warning_log(logger_stub, "Failed to get storage quota summary")
