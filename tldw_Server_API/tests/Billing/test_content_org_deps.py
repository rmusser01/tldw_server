"""Organization selection must keep content and team scope consistent."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.API_Deps import content_org_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context


@pytest.mark.asyncio
async def test_switching_org_drops_prior_org_team_scope(monkeypatch):
    async def resolve(_principal, _org_id, _header_org_id):
        return 20

    monkeypatch.setattr(content_org_deps, "_resolve_org_id", resolve)
    principal = AuthPrincipal(
        kind="user",
        user_id=7,
        org_ids=[10, 20],
        team_ids=[100],
        active_org_id=10,
        active_team_id=100,
    )
    request = SimpleNamespace(state=SimpleNamespace())

    with scoped_context(
        user_id=7,
        org_ids=[10, 20],
        team_ids=[100],
        active_org_id=10,
        active_team_id=100,
    ):
        await content_org_deps._activate_selected_org(request, principal, None, 20)
        assert get_scope().effective_org_id == 20
        assert get_scope().effective_team_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["principal", "user"])
@pytest.mark.parametrize("selection", ["validated", "unvalidated", "changed_principal"])
async def test_cached_auth_preserves_only_validated_org_selection(monkeypatch, boundary, selection):
    """Cached auth retains a validated selection only for its original principal."""
    from starlette.requests import Request

    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as users
    from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver as resolver
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext

    async def resolve(_principal, _org_id, _header_org_id):
        return 20

    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")
    monkeypatch.setattr(content_org_deps, "_resolve_org_id", resolve)
    principal = AuthPrincipal(
        kind="user", user_id=7, org_ids=[10, 20], team_ids=[100],
        active_org_id=10, active_team_id=100,
    )
    request = Request({"type": "http", "headers": [], "path": "/", "method": "GET"})
    request.state.auth = AuthContext(principal=principal)
    request.state._auth_user = users.User(id=7, username="synthetic-owner")
    with scoped_context(
        user_id=7, org_ids=[10, 20], team_ids=[100], active_org_id=10,
        active_team_id=100, session_role="original-role",
    ):
        if selection != "unvalidated":
            await content_org_deps._activate_selected_org(request, principal, None, 20)
        else:
            request.state.active_org_id = 20
            request.state.org_id = 20
        if selection == "changed_principal":
            # Even an in-place canonical claims change invalidates the selection receipt.
            principal.user_id = 8
            principal.org_ids = [10]
        if boundary == "principal":
            await resolver.get_auth_principal(request)
        else:
            await users.get_request_user(request, api_key=None, token=None)
        scope = get_scope()
        if selection == "validated":
            assert (scope.user_id, scope.effective_org_id, scope.team_ids, scope.active_team_id) == (7, 20, [], None)
        else:
            assert (scope.user_id, scope.effective_org_id, scope.team_ids, scope.active_team_id) == (
                principal.user_id, 10, [100], 100,
            )
