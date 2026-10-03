"""Team/org limits.* override routes: platform admin only, validated, null/DELETE removes (spec 2 §3)."""

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_org_member, add_team_member, create_organization, create_team
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.app.services import admin_profiles_service

pytestmark = pytest.mark.unit

KEY = "limits.rag_queries_per_day"


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def _setup_org_and_team(user_id: int) -> tuple[int, int]:
    """An org and a team inside it, with the user a member of both."""
    suffix = uuid.uuid4().hex[:8]

    async def _go() -> tuple[int, int]:
        """Create and join."""
        org = await create_organization(name=f"Quota Org {suffix}", owner_user_id=None)
        team = await create_team(org_id=int(org["id"]), name=f"Quota Team {suffix}")
        await add_org_member(org_id=int(org["id"]), user_id=user_id)
        await add_team_member(team_id=int(team["id"]), user_id=user_id)
        return int(org["id"]), int(team["id"])

    return asyncio.run(_go())


def _resolve(user_id: int) -> object:
    """The user's effective rag-queries limit."""
    return asyncio.run(quota_resolver.user_quota(user_id, KEY))


def test_org_override_applies_to_members_and_delete_removes_it(auth_headers: dict) -> None:
    """Setting an org value gives every member that allowance; DELETE returns them to unlimited."""
    with TestClient(app) as client:
        user_id = int(client.get("/api/v1/users/me/profile", headers=auth_headers).json()["user"]["id"])
        org_id, _team_id = _setup_org_and_team(user_id)
        resp = client.put(f"/api/v1/admin/orgs/{org_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": 30})
        assert resp.status_code == 200
        assert resp.json() == {"scope": "org", "id": org_id, "key": KEY, "value": 30}
        assert _resolve(user_id) == 30
        resp = client.delete(f"/api/v1/admin/orgs/{org_id}/profile/overrides/{KEY}", headers=auth_headers)
        assert resp.status_code == 200 and resp.json()["value"] is None
        assert _resolve(user_id) is None


def test_deleting_team_override_falls_back_to_org(auth_headers: dict) -> None:
    """A team value outranks the org's; removing it drops members back to the org value."""
    with TestClient(app) as client:
        user_id = int(client.get("/api/v1/users/me/profile", headers=auth_headers).json()["user"]["id"])
        org_id, team_id = _setup_org_and_team(user_id)
        assert client.put(f"/api/v1/admin/orgs/{org_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": 100}).status_code == 200
        assert client.put(f"/api/v1/admin/teams/{team_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": 20}).status_code == 200
        assert _resolve(user_id) == 20
        assert client.put(f"/api/v1/admin/teams/{team_id}/profile/overrides/{KEY}", headers=auth_headers, json={"value": None}).status_code == 200
        assert _resolve(user_id) == 100

        # Clean up the org override: the AuthNZ test DB is session-scoped, and the
        # single-user admin stays a member of this org for the rest of the test
        # session, so leaving this set would leak a 100 limit into any later test
        # that checks this same user/key for an unlimited (None) result.
        assert client.delete(f"/api/v1/admin/orgs/{org_id}/profile/overrides/{KEY}", headers=auth_headers).status_code == 200
        assert _resolve(user_id) is None


def test_group_override_rejects_bad_input(auth_headers: dict) -> None:
    """Unknown keys, storage, non-limits keys, invalid values and missing groups are refused."""
    with TestClient(app) as client:
        user_id = int(client.get("/api/v1/users/me/profile", headers=auth_headers).json()["user"]["id"])
        org_id, _ = _setup_org_and_team(user_id)
        base = f"/api/v1/admin/orgs/{org_id}/profile/overrides"
        assert client.put(f"{base}/limits.no_such_key", headers=auth_headers, json={"value": 1}).status_code == 400
        assert client.put(f"{base}/limits.storage_quota_mb", headers=auth_headers, json={"value": 1}).status_code == 400
        assert client.put(f"{base}/preferences.ui.theme", headers=auth_headers, json={"value": 1}).status_code == 400
        assert client.put(f"{base}/{KEY}", headers=auth_headers, json={"value": -1}).status_code == 400
        assert client.put(f"/api/v1/admin/orgs/987654321/profile/overrides/{KEY}", headers=auth_headers, json={"value": 1}).status_code == 404


async def test_group_override_requires_platform_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A principal that is neither single-user nor platform admin gets 403 before any write."""
    monkeypatch.setattr(admin_profiles_service, "is_single_user_principal", lambda _p: False)
    monkeypatch.setattr(admin_profiles_service.admin_scope_service, "is_platform_admin", lambda _p: False)
    principal = AuthPrincipal(kind="user", user_id=5, is_admin=False)
    with pytest.raises(HTTPException) as exc:
        await admin_profiles_service.set_group_limit_override(scope="org", group_id=1, key=KEY, value=3, principal=principal)
    assert exc.value.status_code == 403
