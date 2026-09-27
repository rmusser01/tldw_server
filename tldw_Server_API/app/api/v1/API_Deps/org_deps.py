"""
org_deps.py

FastAPI dependencies for organization and team permission checks.
Provides role-based access control for self-service org management.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.platform_admin import PLATFORM_ADMIN_PERMISSIONS


@dataclass
class OrgContext:
    """Context for org-scoped operations."""
    org_id: int
    role: str
    is_platform_admin: bool = False


@dataclass
class TeamContext:
    """Context for team-scoped operations."""
    org_id: int
    team_id: int
    org_role: str
    team_role: str | None = None
    is_platform_admin: bool = False


# Role hierarchy for comparison
ROLE_HIERARCHY = {
    "owner": 4,
    "admin": 3,
    "lead": 2,
    "member": 1,
}

ACTIVE_MEMBERSHIP_STATUSES = {"active"}
_ADMIN_CLAIM_PERMISSIONS = PLATFORM_ADMIN_PERMISSIONS  # see core/AuthNZ/platform_admin.py


def _role_at_least(user_role: str, required_role: str) -> bool:
    """Check if user_role meets or exceeds required_role in hierarchy."""
    user_level = ROLE_HIERARCHY.get(str(user_role).lower(), 0)
    required_level = ROLE_HIERARCHY.get(str(required_role).lower(), 0)
    return user_level >= required_level


def _is_membership_active(membership: dict | None) -> bool:
    if not membership:
        return False
    status = membership.get("status")
    if status is None:
        # Treat missing status as inactive; legacy rows should be backfilled to "active".
        return False
    return str(status).strip().lower() in ACTIVE_MEMBERSHIP_STATUSES


def _role_allowed(user_role: str, allowed_roles: list[str]) -> bool:
    if not allowed_roles:
        return True
    return any(_role_at_least(user_role, role) for role in allowed_roles)


def _principal_has_admin_claims(principal: AuthPrincipal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


async def _get_user_org_membership(
    user_id: int, org_id: int, repo: AuthnzOrgsTeamsRepo | None = None
) -> dict | None:
    """Get a user's membership in an organization."""
    if repo is None:
        db_pool = await get_db_pool()
        repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)
    return await repo.get_org_member(org_id, user_id)


async def _get_user_team_membership(
    user_id: int, team_id: int, repo: AuthnzOrgsTeamsRepo | None = None
) -> dict | None:
    """Get a user's membership in a team."""
    if repo is None:
        db_pool = await get_db_pool()
        repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)
    return await repo.get_team_member(team_id, user_id)


def require_org_role(*allowed_roles: str):
    """
    Dependency factory that enforces org membership with role check.

    Platform admins bypass membership checks.

    Args:
        allowed_roles: Roles that are allowed (e.g., "owner", "admin").
                       If empty, any membership is sufficient.

    Returns:
        Dependency function returning OrgContext.

    Usage:
        @router.get("/{org_id}")
        async def get_org(ctx: OrgContext = Depends(require_org_role("owner", "admin"))):
            ...
    """
    allowed_roles_normalized = [str(r).strip().lower() for r in allowed_roles if str(r).strip()]

    async def _checker(
        org_id: int,
        principal: AuthPrincipal = Depends(get_auth_principal),
    ) -> OrgContext:
        # Platform admins bypass role checks
        if _principal_has_admin_claims(principal):
            logger.debug(f"Platform admin {principal.user_id} accessing org {org_id}")
            return OrgContext(
                org_id=org_id,
                role="admin",
                is_platform_admin=True,
            )

        # Check membership
        membership = await _get_user_org_membership(principal.user_id, org_id)

        if not membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not a member of this organization",
            )
        if not _is_membership_active(membership):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your organization membership is not active",
            )

        user_role = membership.get("role", "member")

        # If specific roles are required, check them
        if allowed_roles_normalized and not _role_allowed(user_role, allowed_roles_normalized):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {', '.join(allowed_roles_normalized)}",
            )

        return OrgContext(
            org_id=org_id,
            role=user_role,
            is_platform_admin=False,
        )

    return _checker


def require_org_membership():
    """
    Dependency that requires any org membership (no specific role).

    Use for endpoints where being a member is sufficient.
    """
    return require_org_role()


def require_org_owner():
    """Dependency that requires org owner role."""
    return require_org_role("owner")


def require_org_admin():
    """Dependency that requires org admin or owner role."""
    return require_org_role("owner", "admin")


def require_team_role(*allowed_roles: str):
    """
    Dependency factory for team-level permission checks.

    Checks:
    1. User must be a member of the org that owns the team
    2. User must have one of the allowed roles in the team OR be an org owner/admin

    Args:
        allowed_roles: Team roles that are allowed.
                       Org owners/admins always have access.

    Returns:
        Dependency function returning TeamContext.
    """
    allowed_roles_normalized = [str(r).strip().lower() for r in allowed_roles if str(r).strip()]

    async def _checker(
        org_id: int,
        team_id: int,
        principal: AuthPrincipal = Depends(get_auth_principal),
    ) -> TeamContext:
        # Platform admins bypass all checks
        if _principal_has_admin_claims(principal):
            return TeamContext(
                org_id=org_id,
                team_id=team_id,
                org_role="admin",
                team_role="admin",
                is_platform_admin=True,
            )

        # Check org membership
        db_pool = await get_db_pool()
        repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)
        org_membership = await _get_user_org_membership(principal.user_id, org_id, repo=repo)
        if not org_membership:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not a member of this organization",
            )
        if not _is_membership_active(org_membership):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your organization membership is not active",
            )

        org_role = org_membership.get("role", "member")

        # Org owners and admins have full team access
        if _role_at_least(org_role, "admin"):
            return TeamContext(
                org_id=org_id,
                team_id=team_id,
                org_role=org_role,
                team_role=org_role,
                is_platform_admin=False,
            )

        # Verify team belongs to org
        team = await repo.get_team(team_id)

        if not team or team.get("org_id") != org_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found in this organization",
            )

        # Check team membership
        team_membership = await _get_user_team_membership(principal.user_id, team_id, repo=repo)
        team_role = None
        if team_membership:
            if not _is_membership_active(team_membership):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Your team membership is not active",
                )
            team_role = team_membership.get("role")

        if allowed_roles_normalized:
            if not team_role or not _role_allowed(team_role, allowed_roles_normalized):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient team permissions. Required role: {', '.join(allowed_roles_normalized)}",
                )

        return TeamContext(
            org_id=org_id,
            team_id=team_id,
            org_role=org_role,
            team_role=team_role,
            is_platform_admin=False,
        )

    return _checker


def require_team_lead():
    """Dependency that requires team lead role or higher."""
    return require_team_role("owner", "admin", "lead")


async def get_user_orgs(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[dict]:
    """
    Get all organizations the current user belongs to.

    Returns list of {org_id, role} dicts.
    """
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)
    orgs, _total_count = await repo.list_organizations_for_user(principal.user_id)
    return orgs


async def get_active_org_id(
    principal: AuthPrincipal = Depends(get_auth_principal),
    x_tldw_org_id: int | None = Header(None, alias="X-TLDW-Org-Id"),
    org_id: int | None = Query(None, description="Organization ID (optional)"),
) -> int | None:
    """
    Resolve the active org ID for the current request.

    Priority:
    1. org_id query parameter (when provided)
    2. X-TLDW-Org-Id header
    3. First org in user's membership list
    4. None (user has no orgs)
    """
    # Check explicit org_id first
    if org_id is not None:
        membership = await _get_user_org_membership(principal.user_id, org_id)
        if not membership and not _principal_has_admin_claims(principal):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to the specified organization",
            )
        return org_id

    # Check header next
    if x_tldw_org_id is not None:
        if _principal_has_admin_claims(principal):
            return x_tldw_org_id
        membership = await _get_user_org_membership(principal.user_id, x_tldw_org_id)
        if membership:
            return x_tldw_org_id
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to the specified organization",
        )

    # Fall back to first org
    user_orgs = await get_user_orgs(principal)
    if user_orgs:
        active_org_id = user_orgs[0].get("org_id") or user_orgs[0].get("id")
        if active_org_id is not None:
            return int(active_org_id)

    return None
