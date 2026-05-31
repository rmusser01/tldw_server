"""Profile resolution contracts for MCP Unified hosts and gateways."""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from mcp_unified.interfaces.storage import ProfileAssignmentStore, ProfileStore

from .defaults import load_gateway_default_assignment
from .models import MCPProfile
from .resolution import ProfileResolutionResult
from .store import ProfileAssignmentStoreUnavailableError, ProfileStoreUnavailableError


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
        resolved_id = self.default_profile_id if profile_id is None else profile_id
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
            logger.opt(exception=True).warning(
                "Profile store unavailable while resolving MCP profile {profile_id}",
                profile_id=resolved_id,
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


class AssignmentBackedProfileResolver(StoreBackedProfileResolver):
    """Resolve profiles from explicit ids, default assignments, then fallback id."""

    def __init__(
        self,
        profile_store: ProfileStore,
        *,
        assignment_store: ProfileAssignmentStore,
        fallback_default_profile_id: str | None = None,
    ) -> None:
        super().__init__(
            profile_store,
            default_profile_id=fallback_default_profile_id,
        )
        self.assignment_store = assignment_store

    async def resolve_profile_result(
        self,
        profile_id: str | None,
        *,
        user_id: str | None = None,
    ) -> ProfileResolutionResult:
        """Return a structured assignment-aware profile-resolution result."""
        if profile_id is not None:
            result = await super().resolve_profile_result(profile_id, user_id=user_id)
            result.provenance = {
                **result.provenance,
                "used_default_assignment": False,
                "default_assignment_id": None,
            }
            return result

        provenance = {
            "requested_profile_id": None,
            "resolved_profile_id": None,
            "used_default_assignment": False,
            "used_default_profile": False,
            "default_assignment_id": None,
            "resolver": self.__class__.__name__,
        }
        try:
            assignment = await load_gateway_default_assignment(self.assignment_store)
        except ProfileAssignmentStoreUnavailableError:
            logger.opt(exception=True).warning(
                "Profile assignment store unavailable while resolving MCP default profile",
            )
            return ProfileResolutionResult(
                status="store_unavailable",
                reason_code="assignment_store_unavailable",
                provenance=provenance,
            )

        resolved_id = None
        if assignment is not None:
            resolved_id = assignment.profile_id
            provenance = {
                **provenance,
                "resolved_profile_id": resolved_id,
                "used_default_assignment": True,
                "default_assignment_id": assignment.id,
            }
        elif self.default_profile_id is not None:
            resolved_id = self.default_profile_id
            provenance = {
                **provenance,
                "resolved_profile_id": resolved_id,
                "used_default_profile": True,
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
            logger.opt(exception=True).warning(
                "Profile store unavailable while resolving MCP profile {profile_id}",
                profile_id=resolved_id,
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
