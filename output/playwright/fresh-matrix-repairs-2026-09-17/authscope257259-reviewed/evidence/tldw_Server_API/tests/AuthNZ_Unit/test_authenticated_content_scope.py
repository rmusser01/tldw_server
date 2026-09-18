"""Content scope follows authenticated request identity even on cached paths."""

import asyncio
from dataclasses import asdict

import pytest
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as users
from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver as resolver
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["principal", "user"])
@pytest.mark.parametrize("prior", [None, 99, 42])
@pytest.mark.parametrize("admin", [False, True])
async def test_cached_identity_restores_content_scope(boundary, prior, admin, monkeypatch):
    """Cached claims replace unrelated scope and retain a matching session role."""
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")
    request = Request({"type": "http", "headers": [], "path": "/", "method": "GET"})
    principal = AuthPrincipal(
        kind="user", user_id=42, roles=["admin"] if admin else ["user"],
        is_admin=admin, org_ids=[3, 4], team_ids=[5, 6], active_org_id=4, active_team_id=6,
    )
    request.state.auth = AuthContext(principal=principal)
    # Deliberately contradictory legacy User state must not replace canonical claims.
    request.state._auth_user = users.User(id=42, username="owner", is_admin=not admin)
    initial = get_scope()

    async def resolve():
        with scoped_context(user_id=prior, org_ids=[3, 4] if prior == 42 else [],
                            team_ids=[5, 6] if prior == 42 else [],
                            active_org_id=4 if prior == 42 else None,
                            active_team_id=6 if prior == 42 else None,
                            is_admin=admin if prior == 42 else False,
                            session_role="matching-role" if prior == 42 else "unrelated-role"):
            if boundary == "principal":
                await resolver.get_auth_principal(request)
            else:
                await users.get_request_user(request, api_key=None, token=None)
            scope = get_scope()
            assert asdict(scope) == {
                "user_id": 42, "org_ids": [3, 4], "team_ids": [5, 6],
                "active_org_id": 4, "active_team_id": 6, "is_admin": admin,
                "session_role": "matching-role" if prior == 42 else None,
            }
            # Child tasks used by streaming must inherit exactly this authenticated scope.
            async def stream_scope():
                return get_scope()
            assert await asyncio.create_task(stream_scope()) == scope

    await asyncio.create_task(resolve())
    assert get_scope() is initial


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["principal", "user"])
@pytest.mark.parametrize("stale", [None, "is_admin", "org_ids", "team_ids", "active_org_id", "active_team_id"])
async def test_cached_identity_clears_absent_or_stale_same_user_authority(boundary, stale, monkeypatch):
    """The same owner ID cannot retain stale elevated claims or selectors."""
    from contextlib import nullcontext

    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")
    request = Request({"type": "http", "headers": [], "path": "/", "method": "GET"})
    request.state.auth = AuthContext(principal=AuthPrincipal(kind="api_key", user_id=42))
    request.state._auth_user = users.User(id=42, username="owner")
    fields = {"user_id": 42, "session_role": "stale-role"}
    if stale is not None:
        fields[stale] = [9] if stale.endswith("ids") else True if stale == "is_admin" else 9
    initial = get_scope()

    async def resolve():
        with scoped_context(**fields) if stale else nullcontext():
            if stale is None:
                assert get_scope() is None
            if boundary == "principal":
                await resolver.get_auth_principal(request)
            else:
                await users.get_request_user(request, api_key=None, token=None)
            assert asdict(get_scope()) == {
                "user_id": 42, "org_ids": [], "team_ids": [],
                "active_org_id": None, "active_team_id": None,
                "is_admin": False, "session_role": None,
            }

    await asyncio.create_task(resolve())
    assert get_scope() is initial
