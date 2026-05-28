"""Profile resolution contracts for MCP Unified hosts and gateways."""

from __future__ import annotations

import logging
from typing import Protocol

from mcp_unified.interfaces.storage import ProfileStore

from .models import MCPProfile
from .resolution import ProfileResolutionResult
from .store import ProfileStoreUnavailableError

logger = logging.getLogger(__name__)


class ProfileResolver(Protocol):
    """Resolve an effective MCP profile for a request principal."""

    async def resolve_profile(
        self,
        profile_id: str | None,
        *,
        user_id: str | None = None,
    ) -> MCPProfile | None: ...


class StoreBackedProfileResolver:
    """Resolve profiles from a profile store with optional standalone default."""

    def __init__(
        self,
        profile_store: ProfileStore,
        *,
        default_profile_id: str | None = None,
    ) -> None:
        self.profile_store = profile_store
        self.default_profile_id = default_profile_id

    async def resolve_profile(
        self,
        profile_id: str | None,
        *,
        user_id: str | None = None,
    ) -> MCPProfile | None:
        """Return the enabled explicit or default profile, failing closed."""
        result = await self.resolve_profile_result(profile_id, user_id=user_id)
        return result.profile if result.status == "resolved" else None

    async def resolve_profile_result(
        self,
        profile_id: str | None,
        *,
        user_id: str | None = None,
    ) -> ProfileResolutionResult:
        """Return a structured profile-resolution result."""
        del user_id
        resolved_id = profile_id or self.default_profile_id
        used_default_profile = profile_id is None and self.default_profile_id is not None
        provenance = {
            "requested_profile_id": profile_id,
            "resolved_profile_id": resolved_id,
            "used_default_profile": used_default_profile,
            "resolver": self.__class__.__name__,
        }
        if resolved_id is None:
            return ProfileResolutionResult(
                status="profile_required",
                reason_code="profile_required",
                provenance=provenance,
            )

        try:
            profile = await self.profile_store.get_profile(resolved_id)
        except ProfileStoreUnavailableError:
            logger.warning(
                "Profile store unavailable while resolving MCP profile %r",
                resolved_id,
                exc_info=True,
            )
            return ProfileResolutionResult(
                status="store_unavailable",
                reason_code="store_unavailable",
                provenance={
                    **provenance,
                    "profile_id": resolved_id,
                },
            )

        provenance = {
            **provenance,
            "profile_id": resolved_id,
        }

        if profile is None:
            return ProfileResolutionResult(
                status="profile_not_found",
                reason_code="profile_not_found",
                provenance=provenance,
            )
        if not profile.enabled:
            return ProfileResolutionResult(
                status="profile_disabled",
                reason_code="profile_disabled",
                provenance=provenance,
            )
        return ProfileResolutionResult(
            status="resolved",
            reason_code="resolved",
            profile=profile,
            provenance=provenance,
        )
