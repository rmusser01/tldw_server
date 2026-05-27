"""Package-local profile store primitives for MCP Unified."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import MCPProfile


class ProfileStoreUnavailableError(RuntimeError):
    """Raised when a profile store cannot serve requests."""


class InMemoryProfileStore:
    """In-memory profile store for tests and standalone bootstrap."""

    def __init__(
        self,
        profiles: Iterable[MCPProfile | Mapping[str, Any]] | None = None,
    ) -> None:
        self._profiles: dict[str, MCPProfile] = {}
        for profile in profiles or ():
            validated = self._validate_profile(profile)
            self._profiles[validated.id] = validated

    async def get_profile(self, profile_id: str) -> MCPProfile | None:
        """Return a copy of a stored profile by id, or None when absent."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None
        return profile.model_copy(deep=True)

    async def list_profiles(self) -> list[MCPProfile]:
        """Return copy-isolated profiles sorted by id for deterministic callers."""
        return [
            self._profiles[profile_id].model_copy(deep=True)
            for profile_id in sorted(self._profiles)
        ]

    async def upsert_profile(
        self,
        profile: MCPProfile | Mapping[str, Any],
    ) -> MCPProfile:
        """Store a profile document and return a copy-isolated stored value."""
        validated = self._validate_profile(profile)
        self._profiles[validated.id] = validated
        return validated.model_copy(deep=True)

    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by id and return whether it existed."""
        return self._profiles.pop(profile_id, None) is not None

    @staticmethod
    def _validate_profile(profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
        """Validate and deep-copy a profile-like object for storage."""
        if isinstance(profile, MCPProfile):
            return profile.model_copy(deep=True)
        return MCPProfile.model_validate(dict(profile)).model_copy(deep=True)
