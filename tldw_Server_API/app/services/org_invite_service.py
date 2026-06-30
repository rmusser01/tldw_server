"""
org_invite_service.py

Service for organization invite code management.
Handles invite creation, validation, redemption, and cleanup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.org_invites_repo import AuthnzOrgInvitesRepo
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings


class InviteStatus(str, Enum):
    """Status of an invite for validation."""
    VALID = "valid"
    EXPIRED = "expired"
    EXHAUSTED = "exhausted"
    REVOKED = "revoked"
    NOT_FOUND = "not_found"


@dataclass
class InviteValidationResult:
    """Result of invite validation."""
    status: InviteStatus
    invite: dict[str, Any] | None = None
    message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == InviteStatus.VALID


@dataclass
class RedemptionResult:
    """Result of invite redemption."""
    success: bool
    invite_id: int | None = None
    org_id: int | None = None
    org_name: str | None = None
    team_id: int | None = None
    team_name: str | None = None
    role: str | None = None
    was_already_member: bool = False
    message: str | None = None


class OrgInviteService:
    """
    Service for managing organization invite codes.

    Provides high-level operations for:
    - Creating invites with validation
    - Validating invite codes
    - Redeeming invites (adds user to org/team)
    - Listing and revoking invites
    """

    def __init__(
        self,
        db_pool: DatabasePool | None = None,
        invites_repo: AuthnzOrgInvitesRepo | None = None,
        orgs_repo: AuthnzOrgsTeamsRepo | None = None,
    ):
        self._db_pool = db_pool
        self._invites_repo = invites_repo
        self._orgs_repo = orgs_repo

    async def _get_db_pool(self) -> DatabasePool:
        if self._db_pool is None:
            self._db_pool = await get_db_pool()
        return self._db_pool

    async def _get_invites_repo(self) -> AuthnzOrgInvitesRepo:
        if self._invites_repo is None:
            pool = await self._get_db_pool()
            self._invites_repo = AuthnzOrgInvitesRepo(db_pool=pool)
        return self._invites_repo

    async def _get_orgs_repo(self) -> AuthnzOrgsTeamsRepo:
        if self._orgs_repo is None:
            pool = await self._get_db_pool()
            self._orgs_repo = AuthnzOrgsTeamsRepo(db_pool=pool)
        return self._orgs_repo

    async def create_invite(
        self,
        *,
        org_id: int,
        created_by: int,
        team_id: int | None = None,
        role_to_grant: str = "member",
        max_uses: int = 1,
        expiry_days: int = 7,
        description: str | None = None,
        allowed_email_domain: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new organization invite code.

        Args:
            org_id: Organization ID
            created_by: User ID of the creator (must be org admin/owner)
            team_id: Optional team ID for team-specific invite
            role_to_grant: Role to assign (member, lead, admin). Cannot be 'owner'.
            max_uses: Maximum redemptions (1-1000)
            expiry_days: Days until expiration (1-365)
            description: Internal description
            allowed_email_domain: Restrict invite to an email domain

        Returns:
            Invite details including the generated code

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate role
        valid_roles = ("member", "lead", "admin")
        if role_to_grant not in valid_roles:
            raise ValueError(f"role_to_grant must be one of {valid_roles}, got '{role_to_grant}'")

        # Validate max_uses
        if not 1 <= max_uses <= 1000:
            raise ValueError("max_uses must be between 1 and 1000")

        # Validate expiry_days
        if not 1 <= expiry_days <= 365:
            raise ValueError("expiry_days must be between 1 and 365")

        # If team_id is provided, verify it belongs to the org
        if team_id is not None:
            orgs_repo = await self._get_orgs_repo()
            team = await orgs_repo.get_team(team_id)
            if not team or team.get("org_id") != org_id:
                raise ValueError(f"Team {team_id} does not belong to organization {org_id}")

        normalized_domain = None
        if allowed_email_domain:
            candidate = str(allowed_email_domain).strip().lower()
            if not re.fullmatch(r"@?[a-z0-9.-]+", candidate):
                raise ValueError("allowed_email_domain must be a valid domain")
            normalized_domain = candidate.lstrip("@")

        invites_repo = await self._get_invites_repo()
        invite = await invites_repo.create_invite(
            org_id=org_id,
            created_by=created_by,
            team_id=team_id,
            role_to_grant=role_to_grant,
            max_uses=max_uses,
            expiry_days=expiry_days,
            description=description,
            allowed_email_domain=normalized_domain,
        )

        logger.info(
            f"Created invite {invite['code'][:8]}... for org {org_id}"
            f"{f' team {team_id}' if team_id else ''} by user {created_by}"
        )

        return invite

    async def validate_invite(self, code: str) -> InviteValidationResult:
        """
        Validate an invite code.

        Checks if the invite exists, is active, not expired, and not exhausted.

        Args:
            code: The invite code to validate

        Returns:
            InviteValidationResult with status and details
        """
        invites_repo = await self._get_invites_repo()
        invite = await invites_repo.get_invite_by_code(code)

        if not invite:
            return InviteValidationResult(
                status=InviteStatus.NOT_FOUND,
                message="Invite code not found"
            )

        if not invite.get("is_active"):
            return InviteValidationResult(
                status=InviteStatus.REVOKED,
                invite=invite,
                message="This invite has been revoked"
            )

        # Check expiration
        expires_at = invite.get("expires_at")
        if expires_at:
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            # Ensure both sides of the comparison are timezone-aware UTC
            now_utc = datetime.now(timezone.utc)
            if expires_at < now_utc:
                return InviteValidationResult(
                    status=InviteStatus.EXPIRED,
                    invite=invite,
                    message="This invite has expired"
                )

        # Check if exhausted
        if invite.get("uses_count", 0) >= invite.get("max_uses", 1):
            return InviteValidationResult(
                status=InviteStatus.EXHAUSTED,
                invite=invite,
                message="This invite has reached its usage limit"
            )

        return InviteValidationResult(
            status=InviteStatus.VALID,
            invite=invite,
            message="Invite is valid"
        )

    async def preview_invite(self, code: str) -> dict[str, Any] | None:
        """
        Get a public preview of an invite (no auth required).

        Returns minimal info for displaying to unauthenticated users.

        Args:
            code: The invite code

        Returns:
            Dict with org_name, team_name (if applicable), role, expiration status
            None if invite not found
        """
        validation = await self.validate_invite(code)

        if validation.status == InviteStatus.NOT_FOUND:
            return None

        invite = validation.invite
        return {
            "org_name": invite.get("org_name"),
            "org_slug": invite.get("org_slug"),
            "team_name": invite.get("team_name"),
            "role_to_grant": invite.get("role_to_grant"),
            "is_valid": validation.is_valid,
            "status": validation.status.value,
            "message": validation.message,
            "expires_at": invite.get("expires_at"),
            "allowed_email_domain": invite.get("allowed_email_domain"),
        }

    async def redeem_invite(
        self,
        *,
        code: str,
        user_id: int,
        user_email: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> RedemptionResult:
        """
        Redeem an invite code, adding the user to the org/team.

        This is idempotent - if user is already a member, returns success
        with was_already_member=True.

        Note:
            If the invite is team-specific (has a team_id), the service will attempt
            to add the user to the team after successfully adding them to the org.
            If team membership fails, the operation still returns success because
            org membership already succeeded; this can leave the user as an org
            member but not a team member.

        Args:
            code: The invite code
            user_id: User ID redeeming the invite
            user_email: User email for domain allowlist checks
            ip_address: Client IP for audit logging
            user_agent: Client user agent for audit logging

        Returns:
            RedemptionResult with outcome details
        """
        # Validate the invite first
        validation = await self.validate_invite(code)

        if not validation.is_valid:
            return RedemptionResult(
                success=False,
                message=validation.message
            )

        invite = validation.invite
        org_id = invite["org_id"]
        team_id = invite.get("team_id")
        role = invite["role_to_grant"]
        allowed_domain = invite.get("allowed_email_domain")
        if allowed_domain:
            if not user_email or "@" not in str(user_email):
                settings = get_settings()
                if not settings.ORG_INVITE_ALLOW_MISSING_EMAIL:
                    return RedemptionResult(
                        success=False,
                        message="Invite is restricted to allowed email domain",
                    )
            else:
                email_domain = str(user_email).strip().split("@")[-1].lower()
                if email_domain != str(allowed_domain).strip().lower().lstrip("@"):
                    return RedemptionResult(
                        success=False,
                        message="Invite is restricted to allowed email domain",
                    )

        invites_repo = await self._get_invites_repo()
        orgs_repo = await self._get_orgs_repo()

        # Check if user is already a member of the org
        existing_membership = await orgs_repo.get_org_member(org_id, user_id)

        if existing_membership:
            # User is already a member
            logger.info(f"User {user_id} already member of org {org_id}, invite redemption is idempotent")

            # Still record the redemption attempt if not already recorded
            if not await invites_repo.has_user_redeemed(invite["id"], user_id):
                try:
                    await invites_repo.record_redemption(
                        invite_id=invite["id"],
                        user_id=user_id,
                        ip_address=ip_address,
                        user_agent=user_agent,
                    )
                    await invites_repo.increment_uses_count(invite["id"])
                except ValueError:
                    pass  # Already redeemed, ignore

            return RedemptionResult(
                success=True,
                invite_id=invite.get("id"),
                org_id=org_id,
                org_name=invite.get("org_name"),
                team_id=team_id,
                team_name=invite.get("team_name"),
                role=existing_membership.get("role", role),
                was_already_member=True,
                message="You are already a member of this organization"
            )

        # Add user to org
        try:
            await orgs_repo.add_org_member(org_id=org_id, user_id=user_id, role=role)
        except Exception as e:
            logger.error(f"Failed to add user {user_id} to org {org_id}: {e}")
            return RedemptionResult(
                success=False,
                message="Failed to join organization"
            )

        # If team-specific invite, also add to team
        team_membership_failed = False
        if team_id:
            try:
                await orgs_repo.add_team_member(team_id=team_id, user_id=user_id, role=role)
            except Exception as e:
                logger.warning(f"Failed to add user {user_id} to team {team_id}: {e}")
                # Don't fail the whole operation - org membership succeeded
                team_membership_failed = True

        # Record the redemption
        try:
            await invites_repo.record_redemption(
                invite_id=invite["id"],
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            await invites_repo.increment_uses_count(invite["id"])
        except ValueError:
            # Already redeemed (shouldn't happen but handle gracefully)
            pass

        logger.info(
            f"User {user_id} redeemed invite {code[:8]}... for org {org_id}"
            f"{f' team {team_id}' if team_id else ''} with role {role}"
        )

        message = "Successfully joined the organization"
        if team_id and team_membership_failed:
            message = (
                "Joined the organization, but failed to add you to the team. "
                "An admin may need to add you manually."
            )

        return RedemptionResult(
            success=True,
            invite_id=invite.get("id"),
            org_id=org_id,
            org_name=invite.get("org_name"),
            team_id=team_id,
            team_name=invite.get("team_name"),
            role=role,
            was_already_member=False,
            message=message,
        )

    async def list_org_invites(
        self,
        org_id: int,
        *,
        include_expired: bool = False,
        include_inactive: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List invites for an organization.

        Args:
            org_id: Organization ID
            include_expired: Include expired invites
            include_inactive: Include revoked invites
            limit: Max results per page
            offset: Pagination offset

        Returns:
            Tuple of (list of invites, total count)
        """
        invites_repo = await self._get_invites_repo()
        return await invites_repo.list_org_invites(
            org_id,
            include_expired=include_expired,
            include_inactive=include_inactive,
            limit=limit,
            offset=offset,
        )

    async def revoke_invite(self, invite_id: int, org_id: int) -> bool:
        """
        Revoke an invite by deactivating it.

        Args:
            invite_id: Invite ID to revoke
            org_id: Organization ID (for authorization check)

        Returns:
            True if revoked, False if not found or doesn't belong to org
        """
        invites_repo = await self._get_invites_repo()
        result = await invites_repo.deactivate_invite(invite_id, org_id)

        if result:
            logger.info(f"Revoked invite {invite_id} for org {org_id}")
        else:
            logger.warning(f"Failed to revoke invite {invite_id} for org {org_id} - not found or wrong org")

        return result

    async def get_invite(self, invite_id: int) -> dict[str, Any] | None:
        """Get invite by ID."""
        invites_repo = await self._get_invites_repo()
        return await invites_repo.get_invite_by_id(invite_id)

    async def cleanup_expired_invites(self) -> int:
        """
        Clean up expired invites (background job).

        Returns:
            Number of invites deactivated
        """
        invites_repo = await self._get_invites_repo()
        count = await invites_repo.cleanup_expired_invites(deactivate_only=True)
        if count > 0:
            logger.info(f"Deactivated {count} expired invites")
        return count


# Singleton instance
_invite_service: OrgInviteService | None = None


async def get_invite_service() -> OrgInviteService:
    """Get or create the invite service singleton."""
    global _invite_service
    if _invite_service is None:
        _invite_service = OrgInviteService()
    return _invite_service


async def reset_invite_service() -> None:
    """Reset the invite service singleton (primarily for tests)."""
    global _invite_service
    _invite_service = None
