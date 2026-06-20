import pytest

from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoped_permissions_do_not_override_rbac():
    class _RBACDeny:
        async def check_permission(self, *_args, **_kwargs):
            return False

    proto = MCPProtocol()
    proto.rbac_policy = _RBACDeny()

    req = MCPRequest(method="tools/call", params={"name": "media.search"}, id="1")
    ctx = RequestContext(
        request_id="r1",
        user_id="1",
        client_id="unit",
        metadata={"permissions": ["mcp:tool:media.search"]},
    )

    allowed = await proto._check_authorization(req, ctx)
    assert allowed is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoped_permissions_restrict_when_present():
    class _RBACAllow:
        async def check_permission(self, *_args, **_kwargs):
            return True

    proto = MCPProtocol()
    proto.rbac_policy = _RBACAllow()
    req = MCPRequest(method="tools/call", params={"name": "media.search"}, id="2")

    ctx_unscoped = RequestContext(
        request_id="r2",
        user_id="1",
        client_id="unit",
        metadata={"permissions": ["system.logs"]},
    )
    assert await proto._check_authorization(req, ctx_unscoped) is True

    ctx_mismatch = RequestContext(
        request_id="r3",
        user_id="1",
        client_id="unit",
        metadata={"permissions": ["mcp:tool:other"]},
    )
    assert await proto._check_authorization(req, ctx_mismatch) is False

    ctx_match = RequestContext(
        request_id="r4",
        user_id="1",
        client_id="unit",
        metadata={"permissions": ["mcp:tool:media.search"]},
    )
    assert await proto._check_authorization(req, ctx_match) is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tools_list_allowed_with_tool_scope():
    class _RBACAllow:
        async def check_permission(self, *_args, **_kwargs):
            return True

    proto = MCPProtocol()
    proto.rbac_policy = _RBACAllow()
    req = MCPRequest(method="tools/list", params={}, id="list-1")

    ctx = RequestContext(
        request_id="list-ctx",
        user_id="1",
        client_id="unit",
        metadata={"permissions": ["mcp:tool:media.search"]},
    )

    allowed = await proto._check_authorization(req, ctx)
    assert allowed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_forged_single_user_compat_metadata_does_not_bypass_rbac():
    class _RBACDeny:
        async def check_permission(self, *_args, **_kwargs):
            return False

    proto = MCPProtocol()
    proto.rbac_policy = _RBACDeny()
    ctx = RequestContext(
        request_id="forged",
        user_id="1",
        client_id="unit",
        metadata={
            "roles": ["admin"],
            "permissions": ["*"],
            "trusted_auth_claims": True,
            "auth_via": "single_user_api_key",
            "compat_claims_source": "mounted_http",
            "_server_auth_compat_sentinel": "client-supplied",
        },
    )

    assert await proto._has_module_permission(ctx, "media") is False
    assert await proto._has_tool_permission(ctx, "media.search") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_server_created_single_user_compat_metadata_allows_default_tools_list():
    from tldw_Server_API.app.core.MCP_unified.protocol import _trusted_compat_claims_metadata

    class _RBACDeny:
        async def check_permission(self, *_args, **_kwargs):
            return False

    proto = MCPProtocol()
    proto.rbac_policy = _RBACDeny()
    ctx = RequestContext(
        request_id="trusted",
        user_id="1",
        client_id="unit",
        metadata={
            **_trusted_compat_claims_metadata(
                auth_via="single_user_api_key",
                compat_claims_source="mounted_http",
            ),
            "roles": ["admin"],
        },
    )

    assert await proto._has_module_permission(ctx, "media") is True
    assert await proto._has_tool_permission(ctx, "media.search") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_server_created_single_user_compat_metadata_allows_tools_call_authorization():
    from tldw_Server_API.app.core.MCP_unified.protocol import _trusted_compat_claims_metadata

    class _RBACDeny:
        async def check_permission(self, *_args, **_kwargs):
            return False

    proto = MCPProtocol()
    proto.rbac_policy = _RBACDeny()
    req = MCPRequest(method="tools/call", params={"name": "media.search", "arguments": {}}, id="trusted-call")
    ctx = RequestContext(
        request_id="trusted-call",
        user_id="1",
        client_id="unit",
        metadata={
            **_trusted_compat_claims_metadata(
                auth_via="single_user_api_key",
                compat_claims_source="mounted_http",
            ),
            "roles": ["admin"],
        },
    )

    assert await proto._check_authorization(req, ctx) is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_server_created_single_user_compat_metadata_allows_resources_and_prompts():
    from tldw_Server_API.app.core.MCP_unified.protocol import _trusted_compat_claims_metadata

    class _RBACDeny:
        async def check_permission(self, *_args, **_kwargs):
            return False

    proto = MCPProtocol()
    proto.rbac_policy = _RBACDeny()
    ctx = RequestContext(
        request_id="trusted-resource-prompt",
        user_id="1",
        client_id="unit",
        metadata={
            **_trusted_compat_claims_metadata(
                auth_via="single_user_api_key",
                compat_claims_source="mounted_http",
            ),
            "roles": "admin",
        },
    )

    assert await proto._check_authorization(MCPRequest(method="resources/list", id="resources"), ctx) is True
    assert await proto._check_authorization(MCPRequest(method="prompts/list", id="prompts"), ctx) is True
