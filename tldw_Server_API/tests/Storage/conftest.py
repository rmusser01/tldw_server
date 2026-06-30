"""
Storage Module Test Configuration and Fixtures

Provides fixtures for testing storage endpoints, quota service,
and generated files repository.
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone

import pytest
import pytest_asyncio

# Set test environment variables
os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("AUTH_MODE", "single_user")
os.environ.setdefault("SINGLE_USER_API_KEY", "test-storage-key-12345")
os.environ.setdefault("STORAGE_CLEANUP_ENABLED", "false")


# =====================================================================
# Mock User Fixtures
# =====================================================================

@pytest.fixture
def mock_user():
    """Create a mock regular user for testing."""
    user = MagicMock()
    user.id = 1
    user.is_superuser = False
    user.role = "user"
    user.username = "testuser"
    return user


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user for testing."""
    user = MagicMock()
    user.id = 1
    user.is_superuser = True
    user.role = "admin"
    user.username = "admin"
    return user


# =====================================================================
# Mock Service Fixtures
# =====================================================================

@pytest.fixture
def mock_files_repo():
    """Create a mock generated files repository."""
    repo = AsyncMock()
    repo.list_files = AsyncMock(return_value=([], 0))
    repo.get_file_by_id = AsyncMock(return_value=None)
    repo.get_files_by_ids = AsyncMock(return_value=[])
    repo.create_file = AsyncMock()
    repo.update_file = AsyncMock()
    repo.soft_delete_file = AsyncMock(return_value=True)
    repo.hard_delete_file = AsyncMock(return_value=True)
    repo.restore_file = AsyncMock(return_value=True)
    repo.bulk_soft_delete = AsyncMock(return_value=0)
    repo.bulk_move_to_folder = AsyncMock(return_value=0)
    repo.list_folders = AsyncMock(return_value=[])
    repo.list_trashed_files = AsyncMock(return_value=([], 0))
    repo.get_user_storage_usage = AsyncMock(return_value={
        "total_bytes": 0,
        "total_mb": 0.0,
        "by_category": {},
        "trash_bytes": 0,
        "trash_mb": 0.0,
    })
    repo.get_expired_files = AsyncMock(return_value=[])
    repo.get_old_trashed_files = AsyncMock(return_value=[])
    repo.update_accessed_at = AsyncMock()
    repo.list_least_accessed = AsyncMock(return_value=[])
    repo.count_least_accessed = AsyncMock(return_value=0)
    return repo


@pytest.fixture
def mock_quotas_repo():
    """Create a mock storage quotas repository."""
    repo = AsyncMock()
    repo.get_org_quota = AsyncMock(return_value=None)
    repo.get_team_quota = AsyncMock(return_value=None)
    repo.check_quota_status = AsyncMock(return_value={
        "quota_mb": 1000,
        "used_mb": 100.0,
        "remaining_mb": 900.0,
        "usage_pct": 10.0,
        "at_soft_limit": False,
        "at_hard_limit": False,
        "has_quota": True,
    })
    repo.can_allocate = AsyncMock(return_value=(True, "Quota check passed"))
    repo.upsert_org_quota = AsyncMock()
    repo.upsert_team_quota = AsyncMock()
    repo.update_org_used_mb = AsyncMock()
    repo.update_team_used_mb = AsyncMock()
    repo.increment_org_used_mb = AsyncMock(return_value=100.0)
    repo.increment_team_used_mb = AsyncMock(return_value=100.0)
    return repo


@pytest.fixture
def mock_storage_service(mock_files_repo, mock_quotas_repo):
    """Create a mock storage quota service."""
    service = AsyncMock()
    service._initialized = True
    service.get_generated_files_repo = AsyncMock(return_value=mock_files_repo)
    service.get_storage_quotas_repo = AsyncMock(return_value=mock_quotas_repo)

    # Quota methods
    service.check_quota = AsyncMock(return_value=(True, {
        "user_id": 1,
        "current_usage_mb": 100.0,
        "quota_mb": 1000,
        "new_size_mb": 10.0,
        "projected_usage_mb": 110.0,
        "available_mb": 900.0,
        "usage_percentage": 10.0,
        "has_quota": True,
    }))
    service.check_combined_quota = AsyncMock(return_value=(True, {}))
    service.update_usage = AsyncMock(return_value={
        "user_id": 1,
        "storage_used_mb": 110.0,
        "storage_quota_mb": 1000,
        "available_mb": 890.0,
        "usage_percentage": 11.0,
    })

    # File management
    service.register_generated_file = AsyncMock(return_value={"id": 1})
    service.unregister_generated_file = AsyncMock(return_value=True)
    service.unregister_generated_files = AsyncMock(return_value=0)
    service.restore_generated_file = AsyncMock(return_value=None)
    service.get_user_generated_files_usage = AsyncMock(return_value={
        "total_bytes": 100 * 1024 * 1024,
        "total_mb": 100.0,
        "by_category": {},
        "trash_bytes": 0,
        "trash_mb": 0.0,
        "quota_mb": 1000,
        "quota_used_mb": 100.0,
    })
    service.get_user_folders = AsyncMock(return_value=[])

    # Quota management
    service.set_user_quota = AsyncMock(return_value={
        "storage_quota_mb": 2000,
        "storage_used_mb": 100.0,
        "available_mb": 1900.0,
        "usage_percentage": 5.0,
    })
    service.get_org_quota = AsyncMock(return_value={
        "quota_mb": 10000,
        "used_mb": 500.0,
        "remaining_mb": 9500.0,
        "usage_pct": 5.0,
        "at_soft_limit": False,
        "at_hard_limit": False,
        "has_quota": True,
    })
    service.get_team_quota = AsyncMock(return_value={
        "quota_mb": 5000,
        "used_mb": 200.0,
        "remaining_mb": 4800.0,
        "usage_pct": 4.0,
        "at_soft_limit": False,
        "at_hard_limit": False,
        "has_quota": True,
    })
    service.set_org_quota = AsyncMock()
    service.set_team_quota = AsyncMock()
    service.update_org_usage = AsyncMock()
    service.update_team_usage = AsyncMock()

    # Cleanup
    service.cleanup_temp_files = AsyncMock(return_value={
        "files_deleted": 0,
        "bytes_freed": 0,
        "errors": 0,
    })

    return service


# =====================================================================
# Sample Data Fixtures
# =====================================================================

@pytest.fixture
def sample_file_record():
    """Create a sample file record for testing."""
    return {
        "id": 1,
        "uuid": "abc123-uuid",
        "user_id": 1,
        "org_id": None,
        "team_id": None,
        "filename": "test_audio.wav",
        "original_filename": "my_audio.wav",
        "storage_path": "tts_audio/test_audio.wav",
        "mime_type": "audio/wav",
        "file_size_bytes": 1024 * 1024,  # 1 MB
        "checksum": "abc123checksum",
        "file_category": "tts_audio",
        "source_feature": "tts",
        "source_ref": None,
        "folder_tag": None,
        "tags": [],
        "is_transient": False,
        "expires_at": None,
        "retention_policy": "user_default",
        "is_deleted": False,
        "deleted_at": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "accessed_at": None,
    }


@pytest.fixture
def sample_deleted_file_record(sample_file_record):
    """Create a sample deleted file record."""
    record = sample_file_record.copy()
    record["is_deleted"] = True
    record["deleted_at"] = datetime.now(timezone.utc)
    return record


# =====================================================================
# Temporary Directory Fixtures
# =====================================================================

@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_user_outputs_dir(temp_storage_dir):
    """Create a temporary user outputs directory."""
    user_dir = temp_storage_dir / "1" / "outputs"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir
