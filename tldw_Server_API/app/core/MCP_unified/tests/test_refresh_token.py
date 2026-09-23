"""Refresh-token rotation, behind the demo-auth gate the endpoint now enforces.

This test used to call ``refresh_token(auth_request=...)`` with nothing else. The
endpoint has since gained ``request: Request`` and an opening call to
``_require_demo_auth_enabled``, which refuses the direct MCP auth surface unless
MCP_ENABLE_DEMO_AUTH is set, the process is in debug or test mode, a 16-character
MCP_DEMO_AUTH_SECRET is configured, and the peer is loopback or private. The test
was red with ``TypeError: refresh_token() missing 1 required positional argument``
because it predated that gate -- the product is right and the test had drifted.

Updated rather than deleted, and extended: the gate itself is now covered, which is
the coverage the product gained and the test never had.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import (
    AuthRefreshRequest,
    refresh_token as refresh_endpoint,
)
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager

_DEMO_SECRET = "x" * 32


def _loopback_request() -> SimpleNamespace:
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))


@pytest.fixture
def demo_auth_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "true")
    monkeypatch.setenv("MCP_DEMO_AUTH_SECRET", _DEMO_SECRET)
    monkeypatch.setenv("TEST_MODE", "true")
    return _loopback_request()


@pytest.mark.asyncio
async def test_refresh_token_rotation_flow(demo_auth_enabled) -> None:
    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="u1")

    resp = await refresh_endpoint(
        auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
        request=demo_auth_enabled,  # type: ignore[arg-type]
    )
    assert resp.access_token and isinstance(resp.access_token, str)
    assert resp.refresh_token and isinstance(resp.refresh_token, str)

    # The old token is revoked by rotation, so replaying it must fail.
    with pytest.raises(HTTPException):
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            request=demo_auth_enabled,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_refresh_is_refused_when_demo_auth_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is the point: direct MCP auth is off unless explicitly enabled."""
    monkeypatch.delenv("MCP_ENABLE_DEMO_AUTH", raising=False)
    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="u2")

    with pytest.raises(HTTPException) as exc_info:
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            request=_loopback_request(),  # type: ignore[arg-type]
        )
    assert exc_info.value.status_code == 501


@pytest.mark.asyncio
async def test_refresh_is_refused_from_a_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "true")
    monkeypatch.setenv("MCP_DEMO_AUTH_SECRET", _DEMO_SECRET)
    monkeypatch.setenv("TEST_MODE", "true")
    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="u3")

    with pytest.raises(HTTPException) as exc_info:
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            # Not 203.0.113.x: Python counts the TEST-NET documentation ranges as
        # private, so they pass this gate. 8.8.8.8 is genuinely public.
        request=SimpleNamespace(client=SimpleNamespace(host="8.8.8.8")),  # type: ignore[arg-type]
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_refresh_is_refused_without_a_configured_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "true")
    monkeypatch.setenv("MCP_DEMO_AUTH_SECRET", "short")
    monkeypatch.setenv("TEST_MODE", "true")
    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="u4")

    with pytest.raises(HTTPException) as exc_info:
        await refresh_endpoint(
            auth_request=AuthRefreshRequest(refresh_token=refresh, token_id=token_id),
            request=_loopback_request(),  # type: ignore[arg-type]
        )
    assert exc_info.value.status_code == 501
