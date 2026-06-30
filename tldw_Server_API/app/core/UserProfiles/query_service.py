"""
Read orchestration service for UserProfiles.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.UserProfiles.contracts import ProfileReadRequest
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService


class ProfileQueryService:
    """Delegate profile reads through the typed read contract."""

    def __init__(self, profile_service: UserProfileService) -> None:
        self._profile_service = profile_service

    async def build(
        self,
        request: ProfileReadRequest,
        *,
        user: dict[str, Any],
        security: dict[str, Any] | None = None,
        metrics_scope: str | None = None,
    ) -> dict[str, Any]:
        sections = set(request.sections) if request.sections is not None else None
        return await self._profile_service.build_profile(
            user=user,
            sections=sections,
            security=security,
            include_sources=request.include_sources,
            include_raw=request.include_raw,
            mask_secrets=request.mask_secrets,
            metrics_scope=metrics_scope,
        )
