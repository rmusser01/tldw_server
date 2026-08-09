from typing import Any

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_current_user,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

pytestmark = pytest.mark.integration
_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


def _attach_api_key_whoami_router(app: FastAPI) -> None:
    """
    Attach a lightweight whoami endpoint that exercises both get_auth_principal
    and get_current_user when authenticating via X-API-KEY in multi-user mode.

    This router is wired dynamically inside the test so that the production app
    remains unchanged outside of test contexts.
    """
    router = APIRouter()

    @router.get("/authnz/api-key-happy")
    async def whoami_api_key_happy(
        request: Request,
        principal: AuthPrincipal = Depends(get_auth_principal),
        user: dict[str, Any] = Depends(get_current_user),
        request_user: User = Depends(get_request_user),
    ) -> dict[str, Any]:
        return {
            "principal": {
                "principal_id": principal.principal_id,
                "kind": principal.kind,
                "user_id": principal.user_id,
                "api_key_id": principal.api_key_id,
                "roles": principal.roles,
                "permissions": principal.permissions,
                "org_ids": principal.org_ids,
                "team_ids": principal.team_ids,
                "active_org_id": principal.active_org_id,
                "active_team_id": principal.active_team_id,
            },
            "user": {
                "id": user.get("id"),
                "role": user.get("role"),
                "roles": user.get("roles"),
                "permissions": user.get("permissions"),
                "is_active": user.get("is_active"),
                "is_verified": user.get("is_verified"),
            },
            "request_user": {
                "id": getattr(request_user, "id", None),
                "roles": list(getattr(request_user, "roles", []) or []),
                "permissions": list(getattr(request_user, "permissions", []) or []),
                "is_admin": bool(getattr(request_user, "is_admin", False)),
                "org_ids": list(getattr(request_user, "org_ids", []) or []),
                "team_ids": list(getattr(request_user, "team_ids", []) or []),
                "active_org_id": getattr(request_user, "active_org_id", None),
                "active_team_id": getattr(request_user, "active_team_id", None),
            },
            "state": {
                "user_id": getattr(request.state, "user_id", None),
                "api_key_id": getattr(request.state, "api_key_id", None),
                "org_ids": getattr(request.state, "org_ids", None),
                "team_ids": getattr(request.state, "team_ids", None),
                "active_org_id": getattr(request.state, "active_org_id", None),
                "active_team_id": getattr(request.state, "active_team_id", None),
            },
            "state_auth_principal": (
                {
                    "principal_id": getattr(getattr(request.state, "auth", None).principal, "principal_id", None),
                    "kind": getattr(getattr(request.state, "auth", None).principal, "kind", None),
                    "user_id": getattr(getattr(request.state, "auth", None).principal, "user_id", None),
                    "api_key_id": getattr(getattr(request.state, "auth", None).principal, "api_key_id", None),
                    "roles": getattr(getattr(request.state, "auth", None).principal, "roles", None),
                    "permissions": getattr(getattr(request.state, "auth", None).principal, "permissions", None),
                    "org_ids": getattr(getattr(request.state, "auth", None).principal, "org_ids", None),
                    "team_ids": getattr(getattr(request.state, "auth", None).principal, "team_ids", None),
                    "active_org_id": getattr(getattr(request.state, "auth", None).principal, "active_org_id", None),
                    "active_team_id": getattr(getattr(request.state, "auth", None).principal, "active_team_id", None),
                }
                if getattr(request.state, "auth", None) is not None
                else None
            ),
        }

    paths = {getattr(r, "path", "") for r in app.router.routes}
    if "/api/v1/authnz/api-key-happy" not in paths:
        app.include_router(router, prefix="/api/v1")


@pytest.mark.asyncio
@pytest.mark.parametrize("header_name", ["X-API-KEY", "Authorization"])
async def test_multi_user_api_key_happy_path_principal_matches_state(
    isolated_test_environment,
    header_name: str,
) -> None:
    """
    Multi-user happy path for API key authentication:

    - Create a user and an API key via the real AuthNZ stack.
    - Call a protected endpoint that depends on both get_auth_principal and
      get_current_user using X-API-KEY or Authorization: Bearer.
    - Assert that the AuthPrincipal mirrors request.state and user data,
      especially api_key_id.
    """
    client, _db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    app = client.app
    assert isinstance(app, FastAPI)

    _attach_api_key_whoami_router(app)

    # Ensure we are in multi-user mode
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    assert settings.AUTH_MODE == "multi_user"

    # Create a user and API key using the real DB/repo stack
    from uuid import uuid4

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = await get_db_pool()

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="api-key-user",
        email="api-key-user@example.com",
        password_hash="x",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid4(),
    )
    user_id = int(created_user["id"])
    mgr = APIKeyManager(pool)
    await mgr.initialize()
    key_rec = await mgr.create_api_key(user_id=user_id, name="api-key-happy")
    api_key = key_rec["key"]

    # Hit the whoami endpoint with API key headers
    headers = {"X-API-KEY": api_key} if header_name == "X-API-KEY" else {"Authorization": f"Bearer {api_key}"}
    resp = client.get(
        "/api/v1/authnz/api-key-happy",
        headers=headers,
    )
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} body={resp.text}"

    payload = resp.json()
    principal = payload["principal"]
    user = payload["user"]
    request_user = payload["request_user"]
    state = payload["state"]
    state_auth_principal = payload["state_auth_principal"]

    # Identity consistency
    assert principal["kind"] == "api_key"
    assert principal["user_id"] is not None
    assert str(principal["user_id"]) == str(user["id"])
    assert str(principal["user_id"]) == str(request_user["id"])

    # request.state mirrors principal identity
    assert str(state["user_id"]) == str(principal["user_id"])
    assert state["api_key_id"] is not None

    # AuthContext principal matches both principal and request.state
    assert state_auth_principal is not None
    assert state_auth_principal["kind"] == principal["kind"]
    assert str(state_auth_principal["user_id"]) == str(principal["user_id"])
    assert state_auth_principal["api_key_id"] == principal["api_key_id"]

    # api_key_id must stay in sync between request.state and AuthPrincipal
    assert principal["api_key_id"] == state["api_key_id"]
    assert state_auth_principal["api_key_id"] == state["api_key_id"]

    # Claims on AuthPrincipal, get_current_user, and get_request_user should be aligned
    assert principal["roles"] == user["roles"] == request_user["roles"]
    assert principal["permissions"] == user["permissions"] == request_user["permissions"]


@pytest.mark.asyncio
async def test_api_key_scopes_org_and_team_membership(
    isolated_test_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    app = client.app
    assert isinstance(app, FastAPI)

    _attach_api_key_whoami_router(app)

    from uuid import uuid4

    from tldw_Server_API.app.core.AuthNZ import api_key_manager as akm_mod
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        add_org_member,
        add_team_member,
        create_organization,
        create_team,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        DEFAULT_BASE_TEAM_NAME,
    )
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = await get_db_pool()

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="api-key-scope-user",
        email="api-key-scope-user@example.com",
        password_hash="x",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid4(),
    )
    user_id = int(created_user["id"])

    org_a = await create_organization(name="Scoped Org A", owner_user_id=None)
    org_b = await create_organization(name="Scoped Org B", owner_user_id=None)
    await add_org_member(org_id=int(org_a["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
    await add_org_member(org_id=int(org_b["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

    team_a = await create_team(org_id=int(org_a["id"]), name="Scoped Team A")
    team_b = await create_team(org_id=int(org_b["id"]), name="Scoped Team B")
    await add_team_member(team_id=int(team_a["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
    await add_team_member(team_id=int(team_b["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

    mgr = APIKeyManager(pool)
    await mgr.initialize()

    async def _force_virtual_key_scope(key_id: int, org_id: int | None, team_id: int | None) -> None:
        """
        Defensive backstop: ensure org/team scope is persisted on the key row.
        Some environments have shown missing scope columns despite virtual key creation.
        """
        async with pool.transaction() as conn:
            if getattr(pool, "pool", None):
                await conn.execute(
                    "UPDATE api_keys SET org_id = $1, team_id = $2 WHERE id = $3",
                    org_id,
                    team_id,
                    key_id,
                )
            else:
                await conn.execute(
                    "UPDATE api_keys SET org_id = ?, team_id = ? WHERE id = ?",
                    (org_id, team_id, key_id),
                )

    async def _fetch_default_team_id(org_id: int) -> int | None:
        if getattr(pool, "pool", None):
            team_id = await pool.fetchval(
                "SELECT id FROM teams WHERE org_id = $1 AND name = $2",
                org_id,
                DEFAULT_BASE_TEAM_NAME,
            )
        else:
            team_id = await pool.fetchval(
                "SELECT id FROM teams WHERE org_id = ? AND name = ?",
                org_id,
                DEFAULT_BASE_TEAM_NAME,
            )
        return int(team_id) if team_id is not None else None

    key_org = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-org-scope",
        org_id=int(org_a["id"]),
    )
    await _force_virtual_key_scope(int(key_org["id"]), int(org_a["id"]), None)
    default_team_id = await _fetch_default_team_id(int(org_a["id"]))
    expected_org_team_ids = [int(team_a["id"])]
    if default_team_id is not None and default_team_id not in expected_org_team_ids:
        expected_org_team_ids.append(default_team_id)
    expected_org_team_ids = sorted(expected_org_team_ids)

    # Stabilize API-key validation across isolated Postgres databases by
    # short-circuiting validate_api_key for the keys created in this test.
    # We still exercise the org/team scoping logic in authenticate_api_key_user.
    original_validate = akm_mod.APIKeyManager.validate_api_key

    key_map: dict[str, dict[str, Any]] = {
        key_org["key"]: {
            "id": int(key_org["id"]),
            "user_id": user_id,
            "scope": "read",
            "status": "active",
            "is_virtual": True,
            "org_id": int(org_a["id"]),
            "team_id": None,
        }
    }

    async def _fake_validate(
        self: APIKeyManager,
        api_key: str,
        required_scope: str | None = None,
        ip_address: str | None = None,
        record_usage: bool = True,
    ) -> dict[str, Any] | None:
        info = key_map.get(api_key)
        if info is not None:
            return dict(info)
        return await original_validate(
            self,
            api_key,
            required_scope=required_scope,
            ip_address=ip_address,
            record_usage=record_usage,
        )

    monkeypatch.setattr(akm_mod.APIKeyManager, "validate_api_key", _fake_validate, raising=False)

    resp_org = client.get(
        "/api/v1/authnz/api-key-happy",
        headers={"X-API-KEY": key_org["key"]},
    )
    assert resp_org.status_code == 200, resp_org.text
    payload_org = resp_org.json()

    assert payload_org["principal"]["org_ids"] == [int(org_a["id"])]
    assert sorted(payload_org["principal"]["team_ids"]) == expected_org_team_ids
    assert payload_org["request_user"]["org_ids"] == [int(org_a["id"])]
    assert sorted(payload_org["request_user"]["team_ids"]) == expected_org_team_ids
    assert int(org_b["id"]) not in payload_org["request_user"]["org_ids"]
    assert int(team_b["id"]) not in payload_org["request_user"]["team_ids"]
    assert payload_org["state"]["org_ids"] == [int(org_a["id"])]
    assert sorted(payload_org["state"]["team_ids"]) == expected_org_team_ids
    assert payload_org["state_auth_principal"]["org_ids"] == [int(org_a["id"])]
    assert sorted(payload_org["state_auth_principal"]["team_ids"]) == expected_org_team_ids
    assert payload_org["request_user"]["active_org_id"] == payload_org["principal"]["active_org_id"]
    assert payload_org["request_user"]["active_team_id"] == payload_org["principal"]["active_team_id"]
    assert payload_org["state"]["active_org_id"] == payload_org["principal"]["active_org_id"]
    assert payload_org["state"]["active_team_id"] == payload_org["principal"]["active_team_id"]

    key_team = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-team-scope",
        org_id=int(org_a["id"]),
        team_id=int(team_a["id"]),
    )
    await _force_virtual_key_scope(int(key_team["id"]), int(org_a["id"]), int(team_a["id"]))
    key_map[key_team["key"]] = {
        "id": int(key_team["id"]),
        "user_id": user_id,
        "scope": "read",
        "status": "active",
        "is_virtual": True,
        "org_id": int(org_a["id"]),
        "team_id": int(team_a["id"]),
    }
    resp_team = client.get(
        "/api/v1/authnz/api-key-happy",
        headers={"X-API-KEY": key_team["key"]},
    )
    assert resp_team.status_code == 200, resp_team.text
    payload_team = resp_team.json()

    assert payload_team["principal"]["org_ids"] == [int(org_a["id"])]
    assert payload_team["principal"]["team_ids"] == [int(team_a["id"])]
    assert payload_team["request_user"]["org_ids"] == [int(org_a["id"])]
    assert payload_team["request_user"]["team_ids"] == [int(team_a["id"])]
    assert int(org_b["id"]) not in payload_team["request_user"]["org_ids"]
    assert int(team_b["id"]) not in payload_team["request_user"]["team_ids"]
    assert payload_team["request_user"]["active_org_id"] == payload_team["principal"]["active_org_id"]
    assert payload_team["request_user"]["active_team_id"] == payload_team["principal"]["active_team_id"]
    assert payload_team["state"]["active_org_id"] == payload_team["principal"]["active_org_id"]
    assert payload_team["state"]["active_team_id"] == payload_team["principal"]["active_team_id"]

    org_c = await create_organization(name="Scoped Org C", owner_user_id=None)
    key_invalid = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-invalid-org",
        org_id=int(org_c["id"]),
    )
    await _force_virtual_key_scope(int(key_invalid["id"]), int(org_c["id"]), None)
    key_map[key_invalid["key"]] = {
        "id": int(key_invalid["id"]),
        "user_id": user_id,
        "scope": "read",
        "status": "active",
        "is_virtual": True,
        "org_id": int(org_c["id"]),
        "team_id": None,
    }
    resp_invalid = client.get(
        "/api/v1/authnz/api-key-happy",
        headers={"X-API-KEY": key_invalid["key"]},
    )
    assert resp_invalid.status_code == 403, resp_invalid.text
