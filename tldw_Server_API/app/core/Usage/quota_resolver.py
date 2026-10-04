"""Per-user usage-quota values (spec 2 §2). None means unlimited."""

from __future__ import annotations

import time

from loguru import logger

from tldw_Server_API.app.core.config import usage_quotas_enabled
from tldw_Server_API.app.core.UserProfiles.limits_precedence import effective_limits

_CACHE_TTL_SECONDS = 60.0
_CACHE_MAX_USERS = 4096
_cache: dict[int, tuple[float, dict[str, int | float]]] = {}


async def _load_limits(user_id: int) -> dict[str, int | float]:
    """Read the user's, their active teams' and active orgs' overrides and apply precedence."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.UserProfiles.overrides_repo import (
        OrgProfileOverridesRepo,
        TeamProfileOverridesRepo,
        UserProfileOverridesRepo,
    )

    pool = await get_db_pool()
    user_repo = UserProfileOverridesRepo(pool)
    await user_repo.ensure_tables()
    user_rows = await user_repo.list_overrides_for_user(user_id)

    memberships = AuthnzOrgsTeamsRepo(db_pool=pool)
    org_ids = sorted(
        {
            int(row["org_id"])
            for row in await memberships.list_org_memberships_for_user(user_id)
            if row.get("org_id") is not None and row.get("status") in (None, "active")
        }
    )
    team_ids = sorted(
        {
            int(row["team_id"])
            for row in await memberships.list_active_team_memberships_for_user(user_id)
            if row.get("team_id") is not None
        }
    )
    org_rows: list[dict] = []
    team_rows: list[dict] = []
    if org_ids:
        org_repo = OrgProfileOverridesRepo(pool)
        await org_repo.ensure_tables()
        org_rows = await org_repo.list_overrides_for_orgs(org_ids)
    if team_ids:
        team_repo = TeamProfileOverridesRepo(pool)
        await team_repo.ensure_tables()
        team_rows = await team_repo.list_overrides_for_teams(team_ids)
    return effective_limits(user_rows, team_rows, org_rows)


async def user_quota(user_id: int | None, key: str) -> int | float | None:
    """The user's effective ``limits.<name>`` value, or None for unlimited.

    None when usage quotas are off, the user is unknown, nothing is set at any
    level, or the lookup fails (fail open, logged). Cached per user for 60 s;
    other workers see a change within that window.
    """
    if user_id is None or not usage_quotas_enabled():
        return None
    now = time.monotonic()
    hit = _cache.get(user_id)
    if hit is not None and hit[0] > now:
        return hit[1].get(key)
    try:
        limits = await _load_limits(user_id)
    except Exception:  # noqa: BLE001 - a quota lookup failure must not block requests (spec 2 §2)
        logger.opt(exception=True).warning("Usage quota lookup failed for user {}; treating as unlimited", user_id)
        return None
    if len(_cache) >= _CACHE_MAX_USERS:
        # ponytail: wholesale clear bounds memory; switch to LRU if the churn shows up in profiles.
        _cache.clear()
    _cache[user_id] = (now + _CACHE_TTL_SECONDS, limits)
    return limits.get(key)


def invalidate_user(user_id: int) -> None:
    """Forget one user's cached limits (after a write to their own overrides)."""
    _cache.pop(int(user_id), None)


def invalidate_all() -> None:
    """Forget every cached user (after a team or org override write)."""
    _cache.clear()
