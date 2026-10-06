"""Shared helpers for the limits.storage_quota_mb writer/reader tests (spec 2 §5)."""

import asyncio
import uuid

from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_org_member, add_team_member, create_organization, create_team
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.services.storage_quota_service import resolved_storage_quota_mb

KEY = "limits.storage_quota_mb"


def quota(user_id: int) -> object:
    """The user's enforced storage quota, read past the resolver cache."""
    quota_resolver.invalidate_all()
    return asyncio.run(resolved_storage_quota_mb(user_id))


def me(client, headers: dict) -> int:
    """The authenticated user's id."""
    return int(client.get("/api/v1/users/me/profile", headers=headers).json()["user"]["id"])


def patch_quota(client, headers: dict, user_id: int, value: object):
    """Set (or with None clear) the user's own limits.storage_quota_mb through the admin profile route."""
    return client.patch(
        f"/api/v1/admin/users/{user_id}/profile",
        headers=headers,
        json={"updates": [{"key": KEY, "value": value}]},
    )


def org_and_team_with_member(user_id: int) -> tuple[int, int]:
    """A fresh org and a team inside it, with the user a member of both."""
    suffix = uuid.uuid4().hex[:8]

    async def _go() -> tuple[int, int]:
        """Create and join."""
        org = await create_organization(name=f"Storage Org {suffix}", owner_user_id=None)
        team = await create_team(org_id=int(org["id"]), name=f"Storage Team {suffix}")
        await add_org_member(org_id=int(org["id"]), user_id=user_id)
        await add_team_member(team_id=int(team["id"]), user_id=user_id)
        return int(org["id"]), int(team["id"])

    return asyncio.run(_go())
