import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


def _make_request_with_principal(principal: AuthPrincipal | None) -> Request:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/mcp/tool_catalogs",
            "headers": [],
        }
    )
    if principal is not None:
        request.state.auth = principal
    return request


def _make_principal(
    *,
    is_admin: bool,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="1",
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def _make_token_data(*, roles: list[str] | None = None) -> mcp_mod.TokenData:
    return mcp_mod.TokenData(
        sub="1",
        username="user1",
        roles=roles or [],
        permissions=[],
        token_type="access",
    )


def test_catalog_admin_context_allows_principal_admin_role_claim():
    request = _make_request_with_principal(_make_principal(is_admin=False, roles=["admin"]))
    token = _make_token_data(roles=["user"])
    assert mcp_mod._is_catalog_admin_context(request, token) is True


def test_catalog_admin_context_allows_principal_admin_permission_claim():
    request = _make_request_with_principal(
        _make_principal(is_admin=False, roles=["user"], permissions=[SYSTEM_CONFIGURE])
    )
    token = _make_token_data(roles=["user"])
    assert mcp_mod._is_catalog_admin_context(request, token) is True


def test_catalog_admin_context_principal_non_admin_overrides_token_admin_role():
    request = _make_request_with_principal(_make_principal(is_admin=True, roles=["user"], permissions=[]))
    token = _make_token_data(roles=["admin"])
    assert mcp_mod._is_catalog_admin_context(request, token) is False


def test_catalog_admin_context_falls_back_to_token_roles_when_principal_absent():
    request = _make_request_with_principal(None)
    token = _make_token_data(roles=["admin"])
    assert mcp_mod._is_catalog_admin_context(request, token) is True


def test_catalog_admin_context_returns_false_without_admin_claims_or_roles():
    request = _make_request_with_principal(None)
    token = _make_token_data(roles=["user"])
    assert mcp_mod._is_catalog_admin_context(request, token) is False


def test_catalog_admin_context_prefers_explicit_principal_argument():
    request = _make_request_with_principal(None)
    token = _make_token_data(roles=["admin"])
    principal = _make_principal(is_admin=True, roles=["user"], permissions=[])
    assert mcp_mod._is_catalog_admin_context(request, token, principal=principal) is False


@pytest.mark.asyncio
async def test_list_tool_catalogs_uses_membership_repo_for_team_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_attach_api_key_metadata(*_args, **_kwargs) -> dict:
        return {}

    org_lookup_calls: list[int] = []
    team_lookup_calls: list[int] = []
    service_calls: list[dict] = []

    async def _fake_list_org_memberships_for_user(user_id: int):
        org_lookup_calls.append(user_id)
        return []

    async def _fake_list_active_team_memberships_for_user(user_id: int):
        team_lookup_calls.append(user_id)
        return [{"team_id": 99, "org_id": 1, "role": "member"}]

    async def _fake_list_visible_tool_catalogs(
        _db,
        *,
        scope_norm: str,
        admin_all: bool,
        org_ids: set[int] | None = None,
        team_ids: set[int] | None = None,
    ):
        service_calls.append(
            {
                "scope_norm": scope_norm,
                "admin_all": admin_all,
                "org_ids": set(org_ids or set()),
                "team_ids": set(team_ids or set()),
            }
        )
        return [
            {
                "id": 7,
                "name": "team-cat",
                "description": "team scoped catalog",
                "org_id": 1,
                "team_id": 99,
                "is_active": True,
                "created_at": "2026-02-09T00:00:00Z",
                "updated_at": "2026-02-09T00:00:00Z",
            }
        ]

    monkeypatch.setattr(mcp_mod, "_attach_api_key_metadata", _fake_attach_api_key_metadata)
    monkeypatch.setattr(mcp_mod, "list_org_memberships_for_user", _fake_list_org_memberships_for_user)
    monkeypatch.setattr(
        mcp_mod,
        "list_active_team_memberships_for_user",
        _fake_list_active_team_memberships_for_user,
    )
    monkeypatch.setattr(
        mcp_mod.admin_tool_catalog_service,
        "list_visible_tool_catalogs",
        _fake_list_visible_tool_catalogs,
    )

    request = _make_request_with_principal(None)
    auth = mcp_mod.McpAuthContext(
        user=_make_token_data(roles=["user"]),
        principal=None,
        api_key_info=None,
        raw_api_key=None,
    )
    rows = await mcp_mod.list_tool_catalogs(
        http_request=request,
        scope="team",
        auth=auth,
        _guard=None,
        db=object(),
    )

    assert len(rows) == 1
    assert rows[0].id == 7
    assert org_lookup_calls == [1]
    assert team_lookup_calls == [1]
    assert service_calls == [
        {
            "scope_norm": "team",
            "admin_all": False,
            "org_ids": set(),
            "team_ids": {99},
        }
    ]


@pytest.mark.asyncio
async def test_list_tool_catalogs_membership_lookup_logs_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_attach_api_key_metadata(*_args, **_kwargs) -> dict:
        return {}

    async def _raise_org_lookup(user_id: int):
        raise RuntimeError(f"org membership lookup exploded for {user_id} at /private/mcp.db")

    async def _raise_team_lookup(user_id: int):
        raise RuntimeError(f"team membership lookup exploded for {user_id} at /private/mcp.db")

    async def _fake_list_visible_tool_catalogs(
        _db,
        *,
        scope_norm: str,
        admin_all: bool,
        org_ids: set[int] | None = None,
        team_ids: set[int] | None = None,
    ):
        assert scope_norm == "all"
        assert admin_all is False
        assert org_ids == set()
        assert team_ids == set()
        return []

    logger_stub = _LoggerStub()
    monkeypatch.setattr(mcp_mod, "logger", logger_stub)
    monkeypatch.setattr(mcp_mod, "_attach_api_key_metadata", _fake_attach_api_key_metadata)
    monkeypatch.setattr(mcp_mod, "list_org_memberships_for_user", _raise_org_lookup)
    monkeypatch.setattr(
        mcp_mod,
        "list_active_team_memberships_for_user",
        _raise_team_lookup,
    )
    monkeypatch.setattr(
        mcp_mod.admin_tool_catalog_service,
        "list_visible_tool_catalogs",
        _fake_list_visible_tool_catalogs,
    )

    request = _make_request_with_principal(None)
    auth = mcp_mod.McpAuthContext(
        user=_make_token_data(roles=["user"]),
        principal=None,
        api_key_info=None,
        raw_api_key=None,
    )

    rows = await mcp_mod.list_tool_catalogs(
        http_request=request,
        scope="all",
        auth=auth,
        _guard=None,
        db=object(),
    )

    assert rows == []
    assert logger_stub.debugs == [
        "MCP tool catalogs: org membership lookup failed",
        "MCP tool catalogs: team membership lookup failed",
    ]
    rendered_logs = " ".join(logger_stub.debugs)
    assert "1" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs
