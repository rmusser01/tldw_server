import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import AuthRefreshRequest
from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import (
    refresh_token as refresh_endpoint,
)
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager

pytestmark = pytest.mark.unit

# The demo-auth surface this endpoint sits behind requires a secret of at least 16 chars.
_DEMO_SECRET = "demo-auth-secret-for-tests"


def _loopback_request() -> Request:
    """A minimal POST Request from loopback.

    `refresh_token` gained a required `request` parameter for
    `_require_demo_auth_enabled(request)`, which rejects anything that is not loopback or
    private. This test predated that guard and called the endpoint with only
    `auth_request=`, so it failed with TypeError before exercising any rotation. See
    TASK-13358.
    """
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/mcp/auth/refresh",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 54321),
        }
    )


@pytest.fixture
def demo_auth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the demo-auth preconditions this endpoint now enforces."""
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "1")
    monkeypatch.setenv("MCP_DEMO_AUTH_SECRET", _DEMO_SECRET)
    monkeypatch.setenv("TEST_MODE", "true")


@pytest.mark.asyncio
async def test_refresh_token_rotation_flow(demo_auth_enabled: None) -> None:
    """A refresh token may be redeemed once; the rotation revokes the one presented."""
    mgr = get_jwt_manager()
    # Create initial refresh token
    refresh, token_id = mgr.create_refresh_token(subject="u1")

    # Rotate
    resp = await refresh_endpoint(
        auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
        request=_loopback_request(),
    )
    assert resp.access_token and isinstance(resp.access_token, str)
    assert resp.refresh_token and isinstance(resp.refresh_token, str)
    assert resp.refresh_token != refresh, "rotation must not hand back the same token"

    # Old should be revoked after rotation
    with pytest.raises(HTTPException) as excinfo:
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            request=_loopback_request(),
        )
    assert excinfo.value.status_code in (400, 401), excinfo.value.status_code


@pytest.mark.asyncio
async def test_refresh_token_requires_demo_auth_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the guard that broke this test is load-bearing, so pin it.

    Without it the endpoint would be an unauthenticated token-minting surface. 501 is the
    documented answer when the direct MCP auth surface is off.
    """
    monkeypatch.delenv("MCP_ENABLE_DEMO_AUTH", raising=False)
    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="u2")

    with pytest.raises(HTTPException) as excinfo:
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            request=_loopback_request(),
        )

    assert excinfo.value.status_code == 501, excinfo.value.status_code


@pytest.mark.asyncio
async def test_refresh_token_rejects_a_public_peer(
    monkeypatch: pytest.MonkeyPatch, demo_auth_enabled: None
) -> None:
    """Control: demo auth is loopback/private only, so a public peer is refused.

    Deliberately 8.8.8.8 and not a documentation address. Python's
    `ipaddress.is_private` is True for the RFC 5737 ranges (192.0.2.0/24,
    198.51.100.0/24, 203.0.113.0/24), so the usual "example" IPs satisfy the guard and
    this test would pass while asserting nothing.
    """
    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="u3")

    public_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/mcp/auth/refresh",
            "headers": [],
            "query_string": b"",
            "client": ("8.8.8.8", 54321),
        }
    )

    with pytest.raises(HTTPException) as excinfo:
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            request=public_request,
        )

    assert excinfo.value.status_code == 403, excinfo.value.status_code


@pytest.mark.asyncio
async def test_refresh_is_refused_without_a_configured_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demo auth enabled but with a secret below the 16-char minimum is not usable."""
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "1")
    monkeypatch.setenv("MCP_DEMO_AUTH_SECRET", "short")
    monkeypatch.setenv("TEST_MODE", "true")
    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="u4")

    with pytest.raises(HTTPException) as exc_info:
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            request=_loopback_request(),
        )
    assert exc_info.value.status_code == 501, exc_info.value.status_code
