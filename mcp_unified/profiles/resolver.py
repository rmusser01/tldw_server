"""Profile resolution contracts for MCP Unified hosts and gateways."""

from __future__ import annotations

from typing import Protocol

from mcp_unified.interfaces.storage import ProfileStore

from .models import MCPProfile
from .store import ProfileStoreUnavailableError


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
        del user_id
        resolved_id = profile_id or self.default_profile_id
        if resolved_id is None:
            return None

        try:
            profile = await self.profile_store.get_profile(resolved_id)
        except ProfileStoreUnavailableError:
            return None

        if profile is None or not profile.enabled:
            return None
        return profile.model_copy(deep=True)
