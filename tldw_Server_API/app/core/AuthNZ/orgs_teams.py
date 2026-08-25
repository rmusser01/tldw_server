from __future__ import annotations

import threading
from typing import Any

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.membership_writer import MembershipWriteContext
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

_PG_CORE_TABLES_ENSURED: set[int] = set()
_PG_CORE_TABLES_ENSURED_LOCK = threading.Lock()


async def _get_orgs_teams_repo() -> AuthnzOrgsTeamsRepo:
    """
    Resolve the shared organizations/teams repository for the current AuthNZ database.

    This helper centralizes DatabasePool acquisition so higher-level helpers do not
    need to duplicate pool wiring or imports.
    """
    pool: DatabasePool = await get_db_pool()
    if getattr(pool, "pool", None) is not None:
        # Ensure Postgres core tables exist (org/team membership, RBAC backstops, etc.).
        # This keeps admin/org flows resilient in test environments that only create
        # a subset of tables.
        key = id(pool)
        should_ensure = False
        with _PG_CORE_TABLES_ENSURED_LOCK:
            if key not in _PG_CORE_TABLES_ENSURED:
                _PG_CORE_TABLES_ENSURED.add(key)
                should_ensure = True
        if should_ensure:
            try:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_authnz_core_tables_pg,
                )

                await ensure_authnz_core_tables_pg(pool)
            except Exception as ensure_tables_error:
                # Best-effort: org APIs will still raise concrete SQL errors if the schema
                # is missing; don't mask them here.
                _ = ensure_tables_error
    return AuthnzOrgsTeamsRepo(pool)


async def create_organization(
    *,
    name: str,
    owner_user_id: int | None = None,
    slug: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a new organization.

    Parameters:
        name: Human-readable organization name.
        owner_user_id: Optional user id to set as the organization owner.
        slug: Optional unique slug identifier.
        metadata: Optional JSON-serializable metadata mapping.

    Returns:
        A dict containing the created organization's fields (e.g. id, name, slug).
    """
    repo = await _get_orgs_teams_repo()
    return await repo.create_organization(
        name=name,
        owner_user_id=owner_user_id,
        slug=slug,
        metadata=metadata,
    )


async def list_organizations(
    limit: int = 100,
    offset: int = 0,
    q: str | None = None,
    *,
    org_ids: list[int] | None = None,
    with_total: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], int]:
    """List organizations with optional server-side filtering and total count.

    When with_total=True, returns a tuple of (rows, total). Otherwise returns rows only.
    """
    repo = await _get_orgs_teams_repo()
    rows, total = await repo.list_organizations(
        limit=limit,
        offset=offset,
        q=q,
        org_ids=org_ids,
        with_total=with_total,
    )
    return (rows, total) if with_total else rows


async def create_team(
    *,
    org_id: int,
    name: str,
    slug: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a team within an organization.

    Parameters:
        org_id: Owning organization id.
        name: Team name (unique per organization).
        slug: Optional slug identifier.
        description: Optional human-readable description.
        metadata: Optional JSON-serializable metadata mapping.

    Returns:
        A dict containing the created team's fields (e.g. id, org_id, name).
    """
    repo = await _get_orgs_teams_repo()
    return await repo.create_team(
        org_id=org_id,
        name=name,
        slug=slug,
        description=description,
        metadata=metadata,
    )


async def add_team_member(
    *,
    team_id: int,
    user_id: int,
    context: MembershipWriteContext,
    role: str = "member",
) -> dict[str, Any]:
    """
    Add a user to a team (idempotent).

    If the user is already a member, returns the existing membership row.
    """
    repo = await _get_orgs_teams_repo()
    return await repo.add_team_member(
        team_id=team_id,
        user_id=user_id,
        context=context,
        role=role,
    )


async def list_team_members(team_id: int) -> list[dict[str, Any]]:
    """
    List members of a team ordered by join time.

    Returns dicts containing at least ``user_id`` and ``role``.
    """
    repo = await _get_orgs_teams_repo()
    return await repo.list_team_members(team_id)


async def get_team(team_id: int) -> dict[str, Any] | None:
    """
    Fetch a team by ID.

    Parameters:
        team_id: The unique identifier of the team to retrieve.

    Returns:
        A dict containing the team's fields (e.g. id, org_id, name, slug,
        description, is_active, created_at, updated_at), or ``None`` if not found.
    """
    repo = await _get_orgs_teams_repo()
    return await repo.get_team(team_id)


async def list_memberships_for_user(user_id: int) -> list[dict[str, Any]]:
    """List team memberships (and org_id) for a given user.

    Returns: list of {team_id, org_id, role}
    """
    repo = await _get_orgs_teams_repo()
    return await repo.list_memberships_for_user(user_id)


async def list_active_team_memberships_for_user(user_id: int) -> list[dict[str, Any]]:
    """List active team memberships (and org_id) for a given user.

    Returns: list of {team_id, org_id, role}
    """
    repo = await _get_orgs_teams_repo()
    return await repo.list_active_team_memberships_for_user(user_id)


# ============================
# Organization membership APIs
# ============================

async def add_org_member(
    *,
    org_id: int,
    user_id: int,
    context: MembershipWriteContext,
    role: str = "member",
) -> dict[str, Any]:
    """
    Add a user to an organization (idempotent).

    Parameters:
        org_id: Organization id the user should join.
        user_id: User id to add to the organization.
        role: Role to assign within the organization (for example, "member" or "owner").

    Returns:
        A dict describing the membership row (for example, containing ``org_id``, ``user_id``, and ``role``).
    """
    repo = await _get_orgs_teams_repo()
    return await repo.add_org_member(
        org_id=org_id,
        user_id=user_id,
        context=context,
        role=role,
    )


async def list_org_members(
    *, org_id: int, limit: int = 100, offset: int = 0, role: str | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    """
    List members of an organization with pagination and optional filters.

    Parameters:
        org_id: Organization id to list members for.
        limit: Maximum number of members to return.
        offset: Number of members to skip before returning results.
        role: Optional role filter (for example, "owner" or "member").
        status: Optional status filter for membership rows.

    Returns:
        A list of dicts describing organization members (for example, with ``user_id``, ``role``, ``status``, and ``added_at``).
    """
    repo = await _get_orgs_teams_repo()
    return await repo.list_org_members(
        org_id=org_id,
        limit=limit,
        offset=offset,
        role=role,
        status=status,
    )


async def remove_org_member(
    *,
    org_id: int,
    user_id: int,
    context: MembershipWriteContext,
) -> dict[str, Any]:
    """
    Remove a user from an organization.

    Parameters:
        org_id: Organization id from which to remove the user.
        user_id: User id to remove.

    Returns:
        A dict with removal status fields such as ``org_id``, ``user_id``, ``removed``,
        and optionally an ``error`` code (for example, ``"owner_required"`` when the
        last remaining owner cannot be removed).
    """
    repo = await _get_orgs_teams_repo()
    return await repo.remove_org_member(
        org_id=org_id,
        user_id=user_id,
        context=context,
    )


async def update_org_member_role(
    *,
    org_id: int,
    user_id: int,
    role: str,
    context: MembershipWriteContext,
) -> dict[str, Any] | None:
    """
    Update an organization member's role.

    Parameters:
        org_id: Organization id of the member.
        user_id: User id whose role should be updated.
        role: New role to assign (for example, "member" or "owner").

    Returns:
        A dict describing the updated membership (typically containing ``org_id``,
        ``user_id``, and ``role``), optionally including an ``error`` field when the
        update is rejected (such as ``"owner_required"``), or ``None`` when the member
        does not exist.
    """
    repo = await _get_orgs_teams_repo()
    return await repo.update_org_member_role(
        org_id=org_id,
        user_id=user_id,
        role=role,
        context=context,
    )


async def update_team_member_role(
    *,
    team_id: int,
    user_id: int,
    role: str,
    context: MembershipWriteContext,
) -> dict[str, Any] | None:
    """
    Update a team member's role.

    Parameters:
        team_id: Team id of the member.
        user_id: User id whose role should be updated.
        role: New role to assign.

    Returns:
        A dict describing the updated membership (team_id, user_id, role), or ``None`` when missing.
    """
    repo = await _get_orgs_teams_repo()
    return await repo.update_team_member_role(
        team_id=team_id,
        user_id=user_id,
        role=role,
        context=context,
    )


async def list_org_memberships_for_user(user_id: int) -> list[dict[str, Any]]:
    """
    List organization memberships for a given user.

    Parameters:
        user_id: User id whose organization memberships should be listed.

    Returns:
        A list of dicts describing memberships, each containing at least ``org_id`` and ``role``.
    """
    repo = await _get_orgs_teams_repo()
    return await repo.list_org_memberships_for_user(user_id)


async def remove_team_member(
    *,
    team_id: int,
    user_id: int,
    context: MembershipWriteContext,
) -> dict[str, Any]:
    """
    Remove a user from a team.

    Parameters:
        team_id: Team id from which to remove the user.
        user_id: User id to remove from the team.

    Returns:
        A dict with removal status fields: ``team_id``, ``user_id``, and ``removed``.
    """
    repo = await _get_orgs_teams_repo()
    return await repo.remove_team_member(
        team_id=team_id,
        user_id=user_id,
        context=context,
    )
