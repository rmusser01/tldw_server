from __future__ import annotations

import asyncio
import uuid

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
)
from tldw_Server_API.app.core.External_Sources.connectors_service import upsert_policy
from tldw_Server_API.app.main import app

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


def _run_async(coro):
    return asyncio.run(coro)


def _get_user_id(client: TestClient, auth_headers) -> int:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return int(resp.json()["user"]["id"])


def test_membership_policy_summary(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup():
            org = await create_organization(name=f"Policy Org {suffix}", owner_user_id=None)
            await add_org_member(org_id=int(org["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
            team = await create_team(org_id=int(org["id"]), name=f"Policy Team {suffix}")
            await add_team_member(team_id=int(team["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await upsert_policy(
                    conn,
                    int(org["id"]),
                    {
                        "enabled_providers": ["drive"],
                        "max_file_size_mb": 123,
                    },
                )
                try:
                    await conn.commit()
                except Exception:
                    _ = None
            return int(org["id"])

        org_id = _run_async(_setup())

        resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "memberships"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        memberships = resp.json().get("memberships", {})
        orgs = memberships.get("orgs", [])
        target = next(item for item in orgs if int(item.get("org_id")) == org_id)
        summary = target.get("policy_summary", {})
        connectors = summary.get("connectors", {})
        assert "drive" in connectors.get("enabled_providers", [])
        assert connectors.get("max_file_size_mb") == 123
        assert summary.get("source") == "db"
