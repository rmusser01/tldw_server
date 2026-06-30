"""
orgs.py

Self-service organization management endpoints.
Allows users to manage their own organizations, teams, members, and invites.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
    get_or_create_audit_service_for_user_id,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
    get_registration_service_dep,
)
from tldw_Server_API.app.api.v1.API_Deps.org_deps import (
    OrgContext,
    require_org_admin,
    require_org_membership,
    require_org_owner,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    OrgBudgetItem,
    OrgBudgetSelfUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.org_team_schemas import (
    OrganizationListResponse,
    OrganizationResponse,
    OrgDetailResponse,
    OrgInviteAcceptRequest,
    OrgInviteAcceptResponse,
    OrgInviteCreateRequest,
    OrgInviteListResponse,
    OrgInviteResponse,
    OrgMemberAddRequest,
    OrgMemberListItem,
    OrgMemberResponse,
    OrgMemberRoleUpdateRequest,
    OrgSelfCreateRequest,
    OrgUpdateRequest,
    OwnershipTransferRequest,
    TeamCreateRequest,
    TeamListResponse,
    TeamMemberAddRequest,
    TeamMemberListResponse,
    TeamMemberResponse,
    TeamResponse,
    TeamUpdateRequest,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventType
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DuplicateOrganizationError,
    DuplicateTeamError,
    InvalidRegistrationCodeError,
    RegistrationCodeExhaustedError,
    RegistrationCodeExpiredError,
    RegistrationDisabledError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.Billing.subscription_service import get_subscription_service
from tldw_Server_API.app.services.admin_budgets_service import (
    list_org_budgets as svc_list_org_budgets,
)
from tldw_Server_API.app.services.admin_budgets_service import (
    upsert_org_budget as svc_upsert_org_budget,
)
from tldw_Server_API.app.services.budget_audit_service import emit_budget_audit_event
from tldw_Server_API.app.services.org_invite_service import get_invite_service
from tldw_Server_API.app.services.registration_service import RegistrationService

router = APIRouter(
    prefix="/orgs",
    tags=["organizations"],
    responses={
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
    },
)


# =============================================================================
# Organization CRUD
# =============================================================================


@router.get(
    "",
    response_model=OrganizationListResponse,
    summary="List my organizations",
    description="List all organizations the current user belongs to.",
)
async def list_my_orgs(
    principal: AuthPrincipal = Depends(get_auth_principal),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List organizations the current user is a member of."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    paginated, total = await repo.list_organizations_for_user(
        principal.user_id,
        limit=limit,
        offset=offset,
        with_total=True,
    )

    return OrganizationListResponse(
        items=[OrganizationResponse(**org) for org in paginated],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(paginated) < total,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(paginated),
        ),
    )


@router.post(
    "",
    response_model=OrgDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create organization",
    description="Create a new organization. The creating user becomes the owner.",
)
async def create_org(
    body: OrgSelfCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    """Create a new organization with the current user as owner."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    try:
        org = await repo.create_organization(
            name=body.name,
            slug=body.slug,
            owner_user_id=principal.user_id,
        )

        # Add creator as owner member
        await repo.add_org_member(
            org_id=org["id"],
            user_id=principal.user_id,
            role="owner",
        )

        logger.info(f"User {principal.user_id} created org {org['id']} ({body.name})")

        return OrgDetailResponse(
            id=org["id"],
            name=org["name"],
            slug=org.get("slug"),
            owner_user_id=org.get("owner_user_id"),
            is_active=org.get("is_active", True),
            created_at=org.get("created_at"),
            updated_at=org.get("updated_at"),
            member_count=1,
            team_count=0,
            user_role="owner",
        )
    except DuplicateOrganizationError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e


@router.get(
    "/{org_id}",
    response_model=OrgDetailResponse,
    summary="Get organization details",
)
async def get_org(
    ctx: OrgContext = Depends(require_org_membership()),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    """Get detailed information about an organization."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Get org
    all_orgs, _ = await repo.list_organizations(limit=1000, offset=0)
    org = next((o for o in all_orgs if o["id"] == ctx.org_id), None)

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # Get member count
    members = await repo.list_org_members(org_id=ctx.org_id, limit=1000)
    member_count = len(members)

    # Get team count (we need a method for this)
    from tldw_Server_API.app.services.admin_orgs_service import list_teams_by_org as svc_list_teams

    teams = await svc_list_teams(ctx.org_id)
    team_count = len(teams) if teams else 0

    return OrgDetailResponse(
        id=org["id"],
        name=org["name"],
        slug=org.get("slug"),
        owner_user_id=org.get("owner_user_id"),
        is_active=org.get("is_active", True),
        created_at=org.get("created_at"),
        updated_at=org.get("updated_at"),
        member_count=member_count,
        team_count=team_count,
        user_role=ctx.role,
    )


@router.patch(
    "/{org_id}",
    response_model=OrganizationResponse,
    summary="Update organization",
)
async def update_org(
    body: OrgUpdateRequest,
    ctx: OrgContext = Depends(require_org_admin()),
):
    """Update organization details. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Build update fields
    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.slug is not None:
        updates["slug"] = body.slug

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    try:
        org = await repo.update_organization(org_id=ctx.org_id, **updates)
    except DuplicateOrganizationError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e

    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    logger.info(f"Org {ctx.org_id} updated by user with role {ctx.role}")
    return OrganizationResponse(**org)


@router.delete(
    "/{org_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete organization",
)
async def delete_org(
    ctx: OrgContext = Depends(require_org_owner()),
) -> Response:
    """Delete an organization. Requires owner role."""
    # Check for active paid subscription before allowing deletion
    try:
        subscription_service = await get_subscription_service()
        sub_status = await subscription_service.get_subscription(ctx.org_id)
    except Exception:
        logger.warning("delete_org: failed to load subscription")
        sub_status = None

    if sub_status and sub_status.plan_name != "free" and sub_status.status not in ("canceled",):
        # Block deletion when a non-free subscription is still active/pending
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization has an active paid subscription; cancel billing before deletion.",
        )

    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)
    await repo.delete_organization_with_provider_secrets(org_id=ctx.org_id)

    logger.info(f"Org {ctx.org_id} deleted by owner")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{org_id}/transfer",
    response_model=OrganizationResponse,
    summary="Transfer ownership",
)
async def transfer_ownership(
    body: OwnershipTransferRequest,
    ctx: OrgContext = Depends(require_org_owner()),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    """Transfer organization ownership to another member. Requires owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Prevent transferring ownership to the current owner (no-op that would demote them to admin)
    if principal.user_id is None:
        logger.warning("transfer_ownership called without a concrete user_id for org {}", ctx.org_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ownership transfer requires an authenticated user",
        )
    if body.new_owner_user_id == principal.user_id:
        logger.warning(
            "User {} attempted to transfer ownership of org {} to themselves",
            principal.user_id,
            ctx.org_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New owner must be a different user than the current owner",
        )

    # Verify new owner is a member
    new_owner_membership = await repo.get_org_member(ctx.org_id, body.new_owner_user_id)
    if not new_owner_membership:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New owner must be an existing member of the organization",
        )

    org = await repo.transfer_organization_ownership(
        org_id=ctx.org_id,
        new_owner_user_id=body.new_owner_user_id,
        current_owner_user_id=principal.user_id,
    )

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    logger.info(f"Org {ctx.org_id} ownership transferred from {principal.user_id} to {body.new_owner_user_id}")
    return OrganizationResponse(**org)


# =============================================================================
# Organization Members
# =============================================================================


@router.get(
    "/{org_id}/members",
    response_model=list[OrgMemberListItem],
    summary="List organization members",
)
async def list_org_members(
    ctx: OrgContext = Depends(require_org_membership()),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List members of an organization."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    members = await repo.list_org_members(org_id=ctx.org_id, limit=limit, offset=offset)
    return [OrgMemberListItem(**m) for m in members]


@router.post(
    "/{org_id}/members",
    response_model=OrgMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add organization member",
)
async def add_org_member(
    body: OrgMemberAddRequest,
    ctx: OrgContext = Depends(require_org_admin()),
):
    """Add a member to the organization. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Cannot add owners via this endpoint
    if body.role and body.role.lower() == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use the transfer ownership endpoint to add an owner",
        )

    result = await repo.add_org_member(
        org_id=ctx.org_id,
        user_id=body.user_id,
        role=body.role or "member",
    )

    logger.info(f"Added user {body.user_id} to org {ctx.org_id} as {body.role}")
    return OrgMemberResponse(**result)


@router.patch(
    "/{org_id}/members/{user_id}",
    response_model=OrgMemberResponse,
    summary="Update member role",
)
async def update_member_role(
    user_id: int,
    body: OrgMemberRoleUpdateRequest,
    ctx: OrgContext = Depends(require_org_admin()),
):
    """Update a member's role. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Cannot assign owner role via this endpoint
    if body.role.lower() == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use the transfer ownership endpoint to assign owner role",
        )

    result = await repo.update_org_member_role(
        org_id=ctx.org_id,
        user_id=user_id,
        role=body.role,
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found",
        )

    if result.get("error") == "owner_required":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot demote the last owner. Transfer ownership first.",
        )

    return OrgMemberResponse(
        org_id=ctx.org_id,
        user_id=user_id,
        role=result["role"],
    )


@router.delete(
    "/{org_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove organization member",
)
async def remove_org_member(
    user_id: int,
    ctx: OrgContext = Depends(require_org_admin()),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    """Remove a member from the organization. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Cannot remove yourself if you're the only owner
    if user_id == principal.user_id and ctx.role == "owner":
        members = await repo.list_org_members(org_id=ctx.org_id, role="owner", limit=10)
        owners = [m for m in members if m.get("role") == "owner"]
        if len(owners) <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last owner. Transfer ownership first.",
            )

    result = await repo.remove_org_member(org_id=ctx.org_id, user_id=user_id)

    if result.get("error") == "owner_required":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove the last owner. Transfer ownership first.",
        )

    if not result.get("removed"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found",
        )

    logger.info(f"Removed user {user_id} from org {ctx.org_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Teams
# =============================================================================


@router.get(
    "/{org_id}/teams",
    response_model=TeamListResponse,
    summary="List teams",
)
async def list_teams(
    ctx: OrgContext = Depends(require_org_membership()),
):
    """List teams in the organization."""
    from tldw_Server_API.app.services.admin_orgs_service import list_teams_by_org

    teams = await list_teams_by_org(ctx.org_id)
    return TeamListResponse(
        items=[TeamResponse(**t) for t in (teams or [])],
        total=len(teams) if teams else 0,
    )


@router.post(
    "/{org_id}/teams",
    response_model=TeamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create team",
)
async def create_team(
    body: TeamCreateRequest,
    ctx: OrgContext = Depends(require_org_admin()),
):
    """Create a new team in the organization. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    try:
        team = await repo.create_team(
            org_id=ctx.org_id,
            name=body.name,
            slug=body.slug,
            description=body.description,
        )
        logger.info(f"Created team {team['id']} in org {ctx.org_id}")
        return TeamResponse(**team)
    except DuplicateTeamError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e


@router.get(
    "/{org_id}/teams/{team_id}",
    response_model=TeamResponse,
    summary="Get team details",
)
async def get_team(
    team_id: int,
    ctx: OrgContext = Depends(require_org_membership()),
):
    """Get team details."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    team = await repo.get_team(team_id)
    if not team or team.get("org_id") != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    return TeamResponse(**team)


@router.patch(
    "/{org_id}/teams/{team_id}",
    response_model=TeamResponse,
    summary="Update team",
)
async def update_team(
    team_id: int,
    body: TeamUpdateRequest,
    ctx: OrgContext = Depends(require_org_admin()),
):
    """Update team details. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Verify team belongs to org
    team = await repo.get_team(team_id)
    if not team or team.get("org_id") != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    # Build update
    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.slug is not None:
        updates["slug"] = body.slug
    if body.description is not None:
        updates["description"] = body.description

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    updated = await repo.update_team(team_id=team_id, **updates)

    if not updated:
        raise HTTPException(status_code=404, detail="Team not found")

    return TeamResponse(**updated)


@router.delete(
    "/{org_id}/teams/{team_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete team",
)
async def delete_team(
    team_id: int,
    ctx: OrgContext = Depends(require_org_admin()),
) -> Response:
    """Delete a team. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Verify team belongs to org
    team = await repo.get_team(team_id)
    if not team or team.get("org_id") != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    await repo.delete_team_with_provider_secrets(team_id=team_id)

    logger.info(f"Deleted team {team_id} from org {ctx.org_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Team Members
# =============================================================================


@router.get(
    "/{org_id}/teams/{team_id}/members",
    response_model=TeamMemberListResponse,
    summary="List team members",
)
async def list_team_members(
    team_id: int,
    ctx: OrgContext = Depends(require_org_membership()),
):
    """List members of a team."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Verify team belongs to org
    team = await repo.get_team(team_id)
    if not team or team.get("org_id") != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    members = await repo.list_team_members(team_id)
    return TeamMemberListResponse(
        items=[
            TeamMemberResponse(
                team_id=team_id,
                user_id=m["user_id"],
                role=m.get("role", "member"),
                org_id=ctx.org_id,
            )
            for m in members
        ],
        total=len(members),
    )


@router.post(
    "/{org_id}/teams/{team_id}/members",
    response_model=TeamMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add team member",
)
async def add_team_member(
    team_id: int,
    body: TeamMemberAddRequest,
    ctx: OrgContext = Depends(require_org_admin()),
):
    """Add a member to a team. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Verify team belongs to org
    team = await repo.get_team(team_id)
    if not team or team.get("org_id") != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    # Verify user is an org member
    membership = await repo.get_org_member(ctx.org_id, body.user_id)
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must be an organization member first",
        )

    result = await repo.add_team_member(
        team_id=team_id,
        user_id=body.user_id,
        role=body.role or "member",
    )

    logger.info(f"Added user {body.user_id} to team {team_id}")
    return TeamMemberResponse(**result)


@router.delete(
    "/{org_id}/teams/{team_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove team member",
)
async def remove_team_member(
    team_id: int,
    user_id: int,
    ctx: OrgContext = Depends(require_org_admin()),
) -> Response:
    """Remove a member from a team. Requires admin or owner role."""
    db_pool = await get_db_pool()
    repo = AuthnzOrgsTeamsRepo(db_pool=db_pool)

    # Verify team belongs to org
    team = await repo.get_team(team_id)
    if not team or team.get("org_id") != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    result = await repo.remove_team_member(team_id=team_id, user_id=user_id)
    if not result.get("removed"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team member not found",
        )

    logger.info(f"Removed user {user_id} from team {team_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Organization Invites
# =============================================================================


@router.get(
    "/{org_id}/invites",
    response_model=OrgInviteListResponse,
    summary="List organization invites",
)
async def list_invites(
    ctx: OrgContext = Depends(require_org_admin()),
    include_expired: bool = Query(False),
    include_inactive: bool = Query(False),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List invites for the organization. Requires admin or owner role."""
    invite_service = await get_invite_service()

    invites, total = await invite_service.list_org_invites(
        ctx.org_id,
        include_expired=include_expired,
        include_inactive=include_inactive,
        limit=limit,
        offset=offset,
    )

    return OrgInviteListResponse(
        items=[OrgInviteResponse(**inv) for inv in invites],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(invites),
        ),
    )


@router.post(
    "/{org_id}/invites",
    response_model=OrgInviteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create invite",
)
async def create_invite(
    body: OrgInviteCreateRequest,
    http_request: Request,
    ctx: OrgContext = Depends(require_org_admin()),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    """Create a new invite code. Requires admin or owner role."""
    invite_service = await get_invite_service()

    try:
        invite = await invite_service.create_invite(
            org_id=ctx.org_id,
            created_by=principal.user_id,
            team_id=body.team_id,
            role_to_grant=body.role_to_grant,
            max_uses=body.max_uses,
            expiry_days=body.expiry_days,
            description=body.description,
            allowed_email_domain=body.allowed_email_domain,
        )

        logger.info(f"Created invite {invite['code'][:8]}... for org {ctx.org_id} by user {principal.user_id}")

        async def _safe_audit_log_invite_create() -> None:
            try:
                if principal.user_id is None:
                    return
                svc = await get_or_create_audit_service_for_user_id(int(principal.user_id))
                correlation_id = http_request.headers.get("X-Correlation-ID") or getattr(
                    http_request.state, "correlation_id", None
                )
                request_id = (
                    http_request.headers.get("X-Request-ID") or getattr(http_request.state, "request_id", None) or ""
                )
                audit_ctx = AuditContext(
                    user_id=str(principal.user_id),
                    correlation_id=correlation_id,
                    request_id=request_id,
                    ip_address=(http_request.client.host if http_request.client else None),
                    user_agent=http_request.headers.get("user-agent"),
                    endpoint=str(http_request.url.path),
                    method=http_request.method,
                )
                await svc.log_event(
                    event_type=AuditEventType.DATA_WRITE,
                    context=audit_ctx,
                    resource_type="org_invite",
                    resource_id=str(invite.get("id")),
                    action="org_invite.create",
                    metadata={
                        "org_id": ctx.org_id,
                        "team_id": invite.get("team_id"),
                        "role_to_grant": invite.get("role_to_grant"),
                        "max_uses": invite.get("max_uses"),
                        "expires_at": invite.get("expires_at"),
                        "allowed_email_domain": invite.get("allowed_email_domain"),
                        "code_prefix": str(invite.get("code") or "")[:8],
                    },
                )
            except Exception:
                logger.debug("Org invite audit failed")

        await _safe_audit_log_invite_create()
        return OrgInviteResponse(**invite)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.delete(
    "/{org_id}/invites/{invite_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Revoke invite",
)
async def revoke_invite(
    invite_id: int,
    http_request: Request,
    ctx: OrgContext = Depends(require_org_admin()),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    """Revoke an invite. Requires admin or owner role."""
    invite_service = await get_invite_service()

    success = await invite_service.revoke_invite(invite_id, ctx.org_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invite not found or does not belong to this organization",
        )

    logger.info(f"Revoked invite {invite_id} for org {ctx.org_id}")

    async def _safe_audit_log_invite_revoke() -> None:
        try:
            if principal.user_id is None:
                return
            svc = await get_or_create_audit_service_for_user_id(int(principal.user_id))
            correlation_id = http_request.headers.get("X-Correlation-ID") or getattr(
                http_request.state, "correlation_id", None
            )
            request_id = (
                http_request.headers.get("X-Request-ID") or getattr(http_request.state, "request_id", None) or ""
            )
            audit_ctx = AuditContext(
                user_id=str(principal.user_id),
                correlation_id=correlation_id,
                request_id=request_id,
                ip_address=(http_request.client.host if http_request.client else None),
                user_agent=http_request.headers.get("user-agent"),
                endpoint=str(http_request.url.path),
                method=http_request.method,
            )
            await svc.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                context=audit_ctx,
                resource_type="org_invite",
                resource_id=str(invite_id),
                action="org_invite.revoke",
                metadata={
                    "org_id": ctx.org_id,
                    "invite_id": invite_id,
                },
            )
        except Exception:
            logger.debug("Org invite audit failed")

    await _safe_audit_log_invite_revoke()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/invites/accept",
    response_model=OrgInviteAcceptResponse,
    summary="Accept org-scoped registration code",
)
async def accept_org_invite(
    body: OrgInviteAcceptRequest,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    registration_service: RegistrationService = Depends(get_registration_service_dep),
):
    """Accept an org-scoped registration code for an existing user."""
    try:
        result = await registration_service.accept_org_invite_code(
            code=body.code,
            user_id=principal.user_id,
        )
    except RegistrationDisabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc) or "Registration is currently disabled",
        ) from exc
    except RegistrationCodeExpiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Invite code has expired",
        ) from exc
    except RegistrationCodeExhaustedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Invite code has reached its usage limit",
        ) from exc
    except InvalidRegistrationCodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc) or "Invalid invite code",
        ) from exc

    async def _safe_audit_log_invite_accept() -> None:
        try:
            if principal.user_id is None:
                return
            code_id = result.get("registration_code_id")
            if code_id is None:
                return
            svc = await get_or_create_audit_service_for_user_id(int(principal.user_id))
            correlation_id = http_request.headers.get("X-Correlation-ID") or getattr(
                http_request.state, "correlation_id", None
            )
            request_id = (
                http_request.headers.get("X-Request-ID") or getattr(http_request.state, "request_id", None) or ""
            )
            ctx = AuditContext(
                user_id=str(principal.user_id),
                correlation_id=correlation_id,
                request_id=request_id,
                ip_address=(http_request.client.host if http_request.client else None),
                user_agent=http_request.headers.get("user-agent"),
                endpoint=str(http_request.url.path),
                method=http_request.method,
            )
            await svc.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                context=ctx,
                resource_type="registration_code",
                resource_id=str(code_id),
                action="registration_code.redeemed",
                metadata={
                    "registration_code_id": code_id,
                    "org_id": result.get("org_id"),
                    "org_role": result.get("org_role"),
                    "team_id": result.get("team_id"),
                    "was_already_member": result.get("was_already_member"),
                },
            )
        except Exception:
            logger.debug("Org invite audit failed")

    await _safe_audit_log_invite_accept()
    return OrgInviteAcceptResponse(
        success=True,
        org_id=result.get("org_id"),
        team_id=result.get("team_id"),
        org_role=result.get("org_role"),
        was_already_member=result.get("was_already_member", False),
    )


# =============================================================================
# Org Budget Governance
# =============================================================================


@router.get(
    "/{org_id}/budgets",
    response_model=OrgBudgetItem,
    summary="Get organization budget settings",
)
async def get_org_budgets(
    ctx: OrgContext = Depends(require_org_admin()),
    db=Depends(get_db_transaction),
) -> OrgBudgetItem:
    """Return budget settings and plan context for an org."""
    items, _total = await svc_list_org_budgets(
        db,
        org_ids=[ctx.org_id],
        page=1,
        limit=1,
    )
    if not items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="org_not_found")
    return OrgBudgetItem(**items[0])


@router.post(
    "/{org_id}/budgets",
    response_model=OrgBudgetItem,
    summary="Update organization budget settings",
)
async def update_org_budgets(
    payload: OrgBudgetSelfUpdateRequest,
    request: Request,
    ctx: OrgContext = Depends(require_org_admin()),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> OrgBudgetItem:
    """Update budget settings for an org (org admin/owner only)."""
    budget_updates = None
    if payload.budgets is not None:
        budget_updates = payload.budgets.model_dump(exclude_unset=True, by_alias=True)
    try:
        item, audit_changes = await svc_upsert_org_budget(
            db,
            org_id=ctx.org_id,
            budget_updates=budget_updates,
            clear_budgets=payload.clear_budgets,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "org_not_found":
            raise HTTPException(status_code=404, detail="org_not_found") from exc
        if detail == "plan_not_found":
            raise HTTPException(status_code=500, detail="plan_not_found") from exc
        if detail == "subscription_not_found":
            raise HTTPException(status_code=500, detail="subscription_not_found") from exc
        raise HTTPException(status_code=400, detail="invalid_budget_update") from exc
    except Exception as exc:
        logger.error("Failed to upsert org budget")
        raise HTTPException(status_code=500, detail="Failed to upsert org budget") from exc

    try:
        await emit_budget_audit_event(
            request,
            principal,
            org_id=ctx.org_id,
            budget_updates=budget_updates,
            audit_changes=audit_changes,
            clear_budgets=payload.clear_budgets,
            actor_role=ctx.role,
        )
    except Exception as exc:
        logger.error("Budget audit failed")
        raise HTTPException(status_code=500, detail="audit_failed") from exc

    return OrgBudgetItem(**item)
