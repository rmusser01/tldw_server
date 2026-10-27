"""Shared helpers for the limits.storage_quota_mb writer/reader tests (spec 2 §5)."""

import asyncio
import uuid
from datetime import datetime
from typing import Any

from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_org_member, add_team_member, create_organization, create_team
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.services.storage_quota_service import resolved_storage_quota_mb

KEY = "limits.storage_quota_mb"
PREFERENCE_KEY = "preferences.chat.default_character_id"


async def new_quota_user(pool: Any, *, used_mb: float = 0.0, surviving_override: bool = False) -> int:
    """Seed a fresh user through the managed AuthNZ abstractions."""
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    user_id = await ensure_test_user(pool, f"quota-regression-{uuid.uuid4().hex[:16]}")
    if used_mb:
        users = UsersDB(pool)
        await users.initialize(ensure_schema=False)
        await users.update_user(user_id, storage_used_mb=used_mb)
    if surviving_override:
        overrides = UserProfileOverridesRepo(pool)
        await overrides.ensure_tables()
        await overrides.upsert_override(user_id=user_id, key=PREFERENCE_KEY, value="quota-survivor", updated_by=None)
    return user_id


def assert_quota_clear_preserves_version(
    client: Any, headers: dict, user_id: int, writer: str, *, surviving_override: bool
) -> None:
    """Set/clear cannot revive a stale profile token, with or without another override."""
    profile_url = f"/api/v1/admin/users/{user_id}/profile"
    before = client.get(profile_url, headers=headers, params={"sections": "preferences"})
    assert before.status_code == 200  # nosec B101 - pytest regression assertion.
    old_version = before.json()["profile_version"]
    writer_url = writer.format(user_id=user_id)
    field = "storage_quota_mb" if writer == "/api/v1/admin/users/{user_id}" else "quota_mb"
    for value in (100, None):
        response = client.put(writer_url, headers=headers, json={field: value, "reason": "quota ABA regression"})
        assert response.status_code == 200  # nosec B101 - pytest regression assertion.
        if value is not None:
            after_set = client.get(profile_url, headers=headers, params={"sections": "preferences"})
            assert after_set.status_code == 200  # nosec B101 - pytest regression assertion.
            set_version = after_set.json()["profile_version"]
    update = {"profile_version": old_version, "updates": [{"key": KEY, "value": 75}]}
    stale = client.patch(profile_url, headers=headers, json=update)
    assert stale.status_code == 409  # nosec B101 - pytest regression assertion.
    assert stale.json()["error_code"] == "profile_version_mismatch"  # nosec B101 - pytest regression assertion.
    current = client.get(profile_url, headers=headers, params={"sections": "preferences"})
    assert current.status_code == 200  # nosec B101 - pytest regression assertion.
    current_version = current.json()["profile_version"]
    assert datetime.fromisoformat(current_version.replace("Z", "+00:00")) > datetime.fromisoformat(  # nosec B101 - pytest regression assertion.
        set_version.replace("Z", "+00:00")
    )
    assert current.json()["preferences"].get(PREFERENCE_KEY) == ("quota-survivor" if surviving_override else None)  # nosec B101 - pytest regression assertion.
    update["profile_version"] = current_version
    retry = client.patch(profile_url, headers=headers, json=update)
    assert retry.status_code == 200  # nosec B101 - pytest regression assertion.
    saved = client.get(f"/api/v1/admin/storage-quotas/users/{user_id}", headers=headers)
    assert saved.status_code == 200  # nosec B101 - pytest regression assertion.
    assert saved.json()["quota_mb"] == 75  # nosec B101 - pytest regression assertion.


def assert_admin_quota_flags(
    client: Any, headers: dict, user_id: int, quota_mb: int | None, *, soft: bool, hard: bool
) -> None:
    """The actual admin PUT and read-only GET report the user's quota thresholds."""
    url = f"/api/v1/admin/storage-quotas/users/{user_id}"
    put = client.put(url, headers=headers, json={"quota_mb": quota_mb})
    assert put.status_code == 200  # nosec B101 - pytest regression assertion.
    profile_url = f"/api/v1/admin/users/{user_id}/profile"
    before = client.get(profile_url, headers=headers, params={"sections": "preferences"})
    assert before.status_code == 200  # nosec B101 - pytest regression assertion.
    get = client.get(url, headers=headers)
    assert get.status_code == 200  # nosec B101 - pytest regression assertion.
    for response in (put, get):
        payload = response.json()
        assert payload["quota_mb"] == quota_mb  # nosec B101 - pytest regression assertion.
        assert payload["has_quota"] is (quota_mb is not None)  # nosec B101 - pytest regression assertion.
        assert (payload["at_soft_limit"], payload["at_hard_limit"]) == (soft, hard)  # nosec B101 - pytest regression assertion.
    after = client.get(profile_url, headers=headers, params={"sections": "preferences"})
    assert after.status_code == 200  # nosec B101 - pytest regression assertion.
    assert after.json()["profile_version"] == before.json()["profile_version"]  # nosec B101 - pytest regression assertion.


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
