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
