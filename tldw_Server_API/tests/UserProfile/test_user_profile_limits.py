"""limits.* keys: platform-admin-only, plain overrides, null deletes, most-generous profile view (spec 2 §3)."""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_org_member,
    add_team_member,
    create_organization,
    create_team,
    remove_org_member,
    remove_team_member,
)
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.core.UserProfiles.overrides_repo import TeamProfileOverridesRepo
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.update_service import _can_edit
from tldw_Server_API.app.core.UserProfiles.user_profile_catalog import load_user_profile_catalog
from tldw_Server_API.app.main import app

pytestmark = pytest.mark.unit

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)

NEW_KEYS = {
    "limits.transcription_minutes_per_month",
    "limits.llm_tokens_per_month",
    "limits.rag_queries_per_day",
    "limits.media_ingest_mb_per_day",
    "limits.workflows_runs_per_day",
    "limits.evaluation_tokens_per_day",
    "limits.chatbooks_exports_per_day",
    "limits.chatbooks_imports_per_day",
    "limits.chatbooks_concurrent_jobs",
}


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas switched on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def _user_id(client: TestClient, auth_headers: dict) -> int:
    """The single user's id.

    Scoped to ``sections=identity`` so this lookup doesn't also build the
    profile's "quotas" section -- that section calls the audio quota helpers,
    which call ``quota_resolver.user_quota`` and would otherwise prime the
    resolver's 60s cache for this user before a test's own setup runs.
    """
    resp = client.get(
        "/api/v1/users/me/profile",
        params={"sections": "identity"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    return int(resp.json()["user"]["id"])


def test_catalog_has_every_limit_key_platform_admin_only() -> None:
    """All limits.* keys exist, default to null, minimum 0, and only platform admins may edit them."""
    entries = {e.key: e for e in load_user_profile_catalog().entries if e.key.startswith("limits.")}
    assert set(entries) >= NEW_KEYS
    for entry in entries.values():
        assert entry.default is None
        assert list(entry.editable_by) == ["platform_admin"]


def test_org_admin_cannot_edit_limits() -> None:
    """An org admin without platform-admin rights is refused every limits.* key."""
    for entry in load_user_profile_catalog().entries:
        if entry.key.startswith("limits."):
            assert _can_edit(entry, {"org_admin", "team_admin"}) is False
            assert _can_edit(entry, {"platform_admin"}) is True


def test_generic_limit_write_and_null_delete(auth_headers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new limits.* key is stored as a user override the resolver sees; null removes it."""
    flips: list[str] = []
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.user_rate_limiter.UserRateLimiter.upgrade_user_tier",
        lambda *a, **k: flips.append("tier"),
    )
    with TestClient(app) as client:
        user_id = _user_id(client, auth_headers)
        try:
            resp = client.patch(
                f"/api/v1/admin/users/{user_id}/profile",
                headers=auth_headers,
                json={"updates": [{"key": "limits.rag_queries_per_day", "value": 5}, {"key": "limits.evaluations_per_day", "value": 9}]},
            )
            assert resp.status_code == 200
            assert set(resp.json()["applied"]) == {"limits.rag_queries_per_day", "limits.evaluations_per_day"}
            assert asyncio.run(quota_resolver.user_quota(user_id, "limits.rag_queries_per_day")) == 5
            assert asyncio.run(quota_resolver.user_quota(user_id, "limits.evaluations_per_day")) == 9
            assert flips == []  # writing an evaluations limit no longer flips the user to the CUSTOM tier

            resp = client.patch(
                f"/api/v1/admin/users/{user_id}/profile",
                headers=auth_headers,
                json={"updates": [{"key": "limits.rag_queries_per_day", "value": None}]},
            )
            assert resp.status_code == 200
            assert "limits.rag_queries_per_day" in resp.json()["applied"]
            assert asyncio.run(quota_resolver.user_quota(user_id, "limits.rag_queries_per_day")) is None
        finally:
            # limits.evaluations_per_day was never nulled above; clear it through the
            # same PATCH path so it doesn't leak a 9/day cap into a later test.
            client.patch(
                f"/api/v1/admin/users/{user_id}/profile",
                headers=auth_headers,
                json={"updates": [{"key": "limits.evaluations_per_day", "value": None}]},
            )


def test_profile_view_shows_most_generous_team_value(auth_headers: dict) -> None:
    """The effective-config view applies the same precedence the resolver enforces."""
    with TestClient(app) as client:
        user_id = _user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]
        org_ids: list[int] = []
        team_ids: dict[str, int] = {}

        async def _setup() -> dict:
            """Two teams with different values; read the effective profile."""
            org = await create_organization(name=f"Limits Org {suffix}", owner_user_id=None)
            await add_org_member(org_id=int(org["id"]), user_id=user_id, context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
            org_ids.append(int(org["id"]))
            low = await create_team(org_id=int(org["id"]), name=f"Low {suffix}")
            high = await create_team(org_id=int(org["id"]), name=f"High {suffix}")
            team_ids["low"] = int(low["id"])
            team_ids["high"] = int(high["id"])
            await add_team_member(team_id=int(low["id"]), user_id=user_id, context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
            await add_team_member(team_id=int(high["id"]), user_id=user_id, context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
            pool = await get_db_pool()
            repo = TeamProfileOverridesRepo(pool)
            await repo.ensure_tables()
            await repo.upsert_override(team_id=int(low["id"]), key="limits.workflows_runs_per_day", value=10, updated_by=None)
            await repo.upsert_override(team_id=int(high["id"]), key="limits.workflows_runs_per_day", value=50, updated_by=None)
            return await UserProfileService(pool)._build_effective_config(user_id, include_sources=True, mask_secrets=False)

        async def _cleanup() -> None:
            """Remove the team overrides and memberships this test created, through the same repos."""
            pool = await get_db_pool()
            repo = TeamProfileOverridesRepo(pool)
            for team_id in team_ids.values():
                await repo.delete_override(team_id=team_id, key="limits.workflows_runs_per_day")
                await remove_team_member(team_id=team_id, user_id=user_id, context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
            for org_id in org_ids:
                await remove_org_member(org_id=org_id, user_id=user_id, context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

        try:
            effective = asyncio.run(_setup())
            assert effective["limits.workflows_runs_per_day"] == {"value": 50, "source": "team"}

            # These team overrides were written directly through the repo (there is
            # no per-user key to invalidate), the same way the platform-admin
            # team/org override endpoint writes them in production
            # (admin_profiles_service.set_group_limit_override calls
            # invalidate_all_quotas() right after its own repo upsert). Without this,
            # the resolver's 60s cache -- already primed empty for this user by the
            # _user_id() lookup above, before these teams/overrides existed -- would
            # make the read below flake between a stale empty result and 50.
            quota_resolver.invalidate_all()
            assert asyncio.run(quota_resolver.user_quota(user_id, "limits.workflows_runs_per_day")) == 50
        finally:
            asyncio.run(_cleanup())
            quota_resolver.invalidate_all()
