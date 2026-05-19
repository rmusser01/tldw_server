from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_rate_limit,
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.org_team_schemas import (
    OrganizationCreateRequest,
    OrganizationListResponse,
    OrganizationResponse,
    OrganizationSTTSettingsResponse,
    OrganizationSTTSettingsUpdate,
    OrganizationWatchlistsSettingsResponse,
    OrganizationWatchlistsSettingsUpdate,
    OrgMemberAddRequest,
    OrgMemberListItem,
    OrgMemberRemoveResponse,
    OrgMemberResponse,
    OrgMemberRoleUpdateRequest,
    OrgMembershipItem,
    TeamCreateRequest,
    TeamMemberAddRequest,
    TeamMemberRoleUpdateRequest,
    TeamMemberRemoveResponse,
    TeamMembershipItem,
    TeamMemberResponse,
    TeamResponse,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_orgs_service

router = APIRouter()


def _get_ensure_sqlite_authnz_ready_if_test_mode():
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._ensure_sqlite_authnz_ready_if_test_mode


@router.post("/orgs", response_model=OrganizationResponse)
async def admin_create_org(
    payload: OrganizationCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OrganizationResponse:
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    return await admin_orgs_service.create_org(payload, principal)


@router.get("/orgs", response_model=OrganizationListResponse)
async def admin_list_orgs(
    principal: AuthPrincipal = Depends(get_auth_principal),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None),
    org_id: int | None = Query(None),
) -> OrganizationListResponse:
    return await admin_orgs_service.list_orgs(
        principal=principal,
        limit=limit,
        offset=offset,
        q=q,
        org_id=org_id,
        wants_wrapper=True,
    )


@router.post("/orgs/{org_id}/teams", response_model=TeamResponse)
async def admin_create_team(
    org_id: int,
    payload: TeamCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TeamResponse:
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    return await admin_orgs_service.create_team(org_id, payload, principal)


@router.get("/orgs/{org_id}/teams", response_model=list[TeamResponse])
async def admin_list_teams(
    org_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> list[TeamResponse]:
    return await admin_orgs_service.list_teams(
        org_id,
        principal=principal,
        limit=limit,
        offset=offset,
        db=db,
    )


@router.get("/teams/{team_id}", response_model=TeamResponse)
async def admin_get_team(
    team_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TeamResponse:
    return await admin_orgs_service.get_team(team_id, principal)


@router.patch(
    "/orgs/{org_id}/stt/settings",
    response_model=OrganizationSTTSettingsResponse,
)
async def admin_update_org_stt_settings(
    org_id: int,
    payload: OrganizationSTTSettingsUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> OrganizationSTTSettingsResponse:
    return await admin_orgs_service.update_org_stt_settings(
        org_id,
        payload,
        principal=principal,
        db=db,
    )


@router.get(
    "/orgs/{org_id}/stt/settings",
    response_model=OrganizationSTTSettingsResponse,
)
async def admin_get_org_stt_settings(
    org_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> OrganizationSTTSettingsResponse:
    return await admin_orgs_service.get_org_stt_settings(
        org_id,
        principal=principal,
        db=db,
    )


@router.patch(
    "/orgs/{org_id}/watchlists/settings",
    response_model=OrganizationWatchlistsSettingsResponse,
)
async def admin_update_org_watchlists_settings(
    org_id: int,
    payload: OrganizationWatchlistsSettingsUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> OrganizationWatchlistsSettingsResponse:
    return await admin_orgs_service.update_org_watchlists_settings(
        org_id,
        payload,
        principal=principal,
        db=db,
    )


@router.get(
    "/orgs/{org_id}/watchlists/settings",
    response_model=OrganizationWatchlistsSettingsResponse,
)
async def admin_get_org_watchlists_settings(
    org_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> OrganizationWatchlistsSettingsResponse:
    return await admin_orgs_service.get_org_watchlists_settings(
        org_id,
        principal=principal,
        db=db,
    )


@router.post("/teams/{team_id}/members", response_model=TeamMemberResponse)
async def admin_add_team_member(
    team_id: int,
    payload: TeamMemberAddRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TeamMemberResponse:
    return await admin_orgs_service.add_team_member(team_id, payload, request, principal)


@router.get(
    "/teams/{team_id}/members",
    response_model=list[TeamMemberResponse],
    dependencies=[Depends(check_rate_limit)],
)
async def admin_list_team_members(
    team_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[TeamMemberResponse]:
    return await admin_orgs_service.list_team_members(team_id, principal=principal)


@router.delete("/teams/{team_id}/members/{user_id}", response_model=TeamMemberRemoveResponse)
async def admin_remove_team_member(
    team_id: int,
    user_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TeamMemberRemoveResponse:
    return await admin_orgs_service.remove_team_member(team_id, user_id, request, principal)


@router.patch("/teams/{team_id}/members/{user_id}", response_model=TeamMemberResponse)
async def admin_update_team_member_role(
    team_id: int,
    user_id: int,
    payload: TeamMemberRoleUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TeamMemberResponse:
    return await admin_orgs_service.update_team_member_role(
        team_id,
        user_id,
        payload,
        request,
        principal,
    )


@router.post("/orgs/{org_id}/members", response_model=OrgMemberResponse)
async def admin_add_org_member(
    org_id: int,
    payload: OrgMemberAddRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OrgMemberResponse:
    return await admin_orgs_service.add_org_member(org_id, payload, request, principal)


@router.get("/orgs/{org_id}/members", response_model=list[OrgMemberListItem])
async def admin_list_org_members(
    org_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    role: str | None = None,
    status: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[OrgMemberListItem]:
    return await admin_orgs_service.list_org_members(
        org_id,
        limit=limit,
        offset=offset,
        role=role,
        status_filter=status,
        principal=principal,
    )


@router.delete("/orgs/{org_id}/members/{user_id}", response_model=OrgMemberRemoveResponse)
async def admin_remove_org_member(
    org_id: int,
    user_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OrgMemberRemoveResponse:
    return await admin_orgs_service.remove_org_member(org_id, user_id, request, principal)


@router.patch("/orgs/{org_id}/members/{user_id}", response_model=OrgMemberResponse)
async def admin_update_org_member_role(
    org_id: int,
    user_id: int,
    payload: OrgMemberRoleUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> OrgMemberResponse:
    return await admin_orgs_service.update_org_member_role(
        org_id,
        user_id,
        payload,
        request,
        principal,
    )


@router.get("/users/{user_id}/org-memberships", response_model=list[OrgMembershipItem])
async def admin_list_user_org_memberships(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[OrgMembershipItem]:
    return await admin_orgs_service.list_user_org_memberships(user_id, principal)


@router.get("/users/{user_id}/team-memberships", response_model=list[TeamMembershipItem])
async def admin_list_user_team_memberships(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[TeamMembershipItem]:
    return await admin_orgs_service.list_user_team_memberships(user_id, principal)
