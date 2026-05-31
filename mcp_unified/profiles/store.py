"""Package-local profile store primitives for MCP Unified."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mcp_unified.storage.models import ProfileAssignment

from .models import MCPProfile


class ProfileStoreUnavailableError(RuntimeError):
    """Raised when a profile store cannot serve requests."""


class ProfileAssignmentStoreUnavailableError(RuntimeError):
    """Raised when a profile-assignment store cannot serve requests."""


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
        return [self._profiles[profile_id].model_copy(deep=True) for profile_id in sorted(self._profiles)]

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
        return MCPProfile.model_validate(profile)


class InMemoryProfileAssignmentStore:
    """In-memory profile assignment store for tests and standalone bootstrap."""

    def __init__(
        self,
        assignments: Iterable[ProfileAssignment | Mapping[str, Any]] | None = None,
    ) -> None:
        self._assignments: dict[str, ProfileAssignment] = {}
        for assignment in assignments or ():
            validated = self._validate_assignment(assignment)
            self._assignments[validated.id] = validated

    async def get_assignment(self, assignment_id: str) -> ProfileAssignment | None:
        """Return a copy of a stored assignment by id, or None when absent."""
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            return None
        return assignment.model_copy(deep=True)

    async def list_assignments(
        self,
        *,
        profile_id: str | None = None,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[ProfileAssignment]:
        """Return copy-isolated assignments matching optional filters."""
        assignments = [
            assignment
            for assignment in self._assignments.values()
            if (profile_id is None or assignment.profile_id == profile_id)
            and (principal_id is None or assignment.principal_id == principal_id)
            and (workspace_id is None or assignment.workspace_id == workspace_id)
        ]
        return [assignment.model_copy(deep=True) for assignment in sorted(assignments, key=lambda item: item.id)]

    async def upsert_assignment(
        self,
        assignment: ProfileAssignment | Mapping[str, Any],
    ) -> ProfileAssignment:
        """Store an assignment document and return a copy-isolated value."""
        validated = self._validate_assignment(assignment)
        self._assignments[validated.id] = validated
        return validated.model_copy(deep=True)

    async def delete_assignment(self, assignment_id: str) -> bool:
        """Delete an assignment by id and return whether it existed."""
        return self._assignments.pop(assignment_id, None) is not None

    @staticmethod
    def _validate_assignment(
        assignment: ProfileAssignment | Mapping[str, Any],
    ) -> ProfileAssignment:
        """Validate and deep-copy an assignment-like object for storage."""
        if isinstance(assignment, ProfileAssignment):
            return assignment.model_copy(deep=True)
        return ProfileAssignment.model_validate(assignment)
