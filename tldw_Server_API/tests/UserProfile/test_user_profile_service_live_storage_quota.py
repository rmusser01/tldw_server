from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.services import storage_quota_service as storage_quota_service_module


class _DummyDbPool:
    """Minimal DB pool stand-in for UserProfileService unit tests."""


@pytest.mark.asyncio
async def test_build_quotas_prefers_live_storage_values(monkeypatch: pytest.MonkeyPatch):
    service = UserProfileService(_DummyDbPool())  # type: ignore[arg-type]

    async def _fake_build_effective_config(
        user_id: int,
        *,
        include_sources: bool,
        mask_secrets: bool,
    ) -> dict[str, Any]:
        del user_id, include_sources, mask_secrets
        return {}

    class _StubStorageService:
        async def calculate_user_storage(
            self,
            user_id: int,
            update_database: bool = True,
        ) -> dict[str, Any]:
            del user_id, update_database
            return {
                "total_mb": 123.45,
                "quota_mb": 2048,
            }

    async def _fake_get_storage_service():
        return _StubStorageService()

    monkeypatch.setattr(
        service,
        "_build_effective_config",
        _fake_build_effective_config,
    )
    monkeypatch.setattr(
        storage_quota_service_module,
        "get_storage_service",
        _fake_get_storage_service,
    )

    quotas = await service._build_quotas(
        {
            "id": 1,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
        }
    )

    if quotas["storage_used_mb"] != 123.45:
        raise AssertionError("Expected live storage_used_mb to override user-context value")
    if quotas["storage_quota_mb"] != 2048:
        raise AssertionError("Expected live storage_quota_mb to override user-context value")


@pytest.mark.asyncio
async def test_build_quotas_falls_back_to_user_values_when_live_storage_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    service = UserProfileService(_DummyDbPool())  # type: ignore[arg-type]

    async def _fake_build_effective_config(
        user_id: int,
        *,
        include_sources: bool,
        mask_secrets: bool,
    ) -> dict[str, Any]:
        del user_id, include_sources, mask_secrets
        return {}

    class _BrokenStorageService:
        async def calculate_user_storage(
            self,
            user_id: int,
            update_database: bool = True,
        ) -> dict[str, Any]:
            del user_id, update_database
            raise StorageError("live storage unavailable")

    async def _fake_get_storage_service():
        return _BrokenStorageService()

    async def _fake_resolved_storage_quota_mb(user_id: int) -> int:
        del user_id
        return 777

    monkeypatch.setattr(
        service,
        "_build_effective_config",
        _fake_build_effective_config,
    )
    monkeypatch.setattr(
        storage_quota_service_module,
        "get_storage_service",
        _fake_get_storage_service,
    )
    monkeypatch.setattr(
        storage_quota_service_module,
        "resolved_storage_quota_mb",
        _fake_resolved_storage_quota_mb,
    )

    quotas = await service._build_quotas(
        {
            "id": 1,
            "storage_quota_mb": 777,
            "storage_used_mb": 12.5,
        }
    )

    # Seeded from the resolver (patched above), not the raw user-context column.
    if quotas["storage_quota_mb"] != 777:
        raise AssertionError("Expected storage_quota_mb to fall back to the resolved value")
    if quotas["storage_used_mb"] != 12.5:
        raise AssertionError("Expected storage_used_mb to fall back to user-context value")
