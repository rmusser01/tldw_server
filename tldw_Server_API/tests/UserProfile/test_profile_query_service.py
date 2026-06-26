from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.UserProfiles.contracts import (
    ProfileContractMode,
    ProfileReadRequest,
)
from tldw_Server_API.app.core.UserProfiles.query_service import ProfileQueryService


class _ProfileServiceStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def build_profile(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"user": {"id": kwargs["user"]["id"]}, "catalog_version": "test"}


@pytest.mark.asyncio
async def test_query_service_delegates_profile_read_request_to_user_profile_service() -> None:
    profile_service = _ProfileServiceStub()
    query_service = ProfileQueryService(profile_service)
    request = ProfileReadRequest(
        actor_user_id=11,
        target_user_id=22,
        sections=frozenset({"identity", "security"}),
        include_sources=True,
        include_raw=True,
        mask_secrets=False,
        contract_mode=ProfileContractMode.LEGACY_V1,
    )
    user = {"id": 22, "username": "target-user"}
    security = {"mfa_enabled": False}

    response = await query_service.build(
        request,
        user=user,
        security=security,
        metrics_scope="admin",
    )

    assert response == {"user": {"id": 22}, "catalog_version": "test"}
    assert profile_service.calls == [
        {
            "user": user,
            "sections": {"identity", "security"},
            "security": security,
            "include_sources": True,
            "include_raw": True,
            "mask_secrets": False,
            "metrics_scope": "admin",
        }
    ]
