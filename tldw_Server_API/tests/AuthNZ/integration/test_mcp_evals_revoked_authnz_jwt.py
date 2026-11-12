from datetime import datetime, timedelta, timezone

import pytest
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist


pytestmark = pytest.mark.integration


def _register_and_login(client, username: str, password: str) -> str:
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert reg.status_code == 200, reg.text

    login = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    assert login.status_code == 200, login.text
    token = login.json().get("access_token")
    assert token, f"Expected access token, got {login.json()}"
    return token


def _coerce_expiry(exp_value) -> datetime:


    if isinstance(exp_value, datetime):
        return exp_value
    if isinstance(exp_value, (int, float)):
        return datetime.fromtimestamp(exp_value, tz=timezone.utc)
    return datetime.now(timezone.utc) + timedelta(hours=1)


async def _revoke_access_token(token: str) -> None:
    jwt_service = get_jwt_service()
    payload = jwt_service.decode_access_token(token)
    jti = payload.get("jti")
    assert jti, "Access token missing jti claim"
    expires_at = _coerce_expiry(payload.get("exp"))
    raw_user_id = payload.get("user_id") or payload.get("sub")
    try:
        user_id = int(raw_user_id) if raw_user_id is not None else None
    except (TypeError, ValueError):
        user_id = None

    blacklist = get_token_blacklist()
    ok = await blacklist.revoke_token(
        jti=jti,
        expires_at=expires_at,
        user_id=user_id,
        token_type="access",
        reason="test-revoke",
    )
    assert ok is True


@pytest.mark.asyncio
async def test_revoked_authnz_jwt_rejected_for_mcp_http(isolated_test_environment):
    client, _db_name = isolated_test_environment

    token = _register_and_login(client, "mcp_revoke_user", "Str0ngP@ssw0rd!")
    await _revoke_access_token(token)

    resp = client.post(
        "/api/v1/mcp/tools/execute",
        headers={"Authorization": f"Bearer {token}"},
        json={"tool_name": "status", "arguments": {}},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_revoked_authnz_jwt_rejected_for_mcp_ws(isolated_test_environment):
    client, _db_name = isolated_test_environment

    token = _register_and_login(client, "mcp_revoke_ws_user", "Str0ngP@ssw0rd!")
    await _revoke_access_token(token)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/api/v1/mcp/ws?client_id=revoked",
            headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            ws.receive_text()

    assert getattr(exc_info.value, "code", None) == 1008


@pytest.mark.asyncio
async def test_revoked_authnz_jwt_rejected_for_evaluations(isolated_test_environment):
    client, _db_name = isolated_test_environment

    token = _register_and_login(client, "eval_revoke_user", "Str0ngP@ssw0rd!")
    await _revoke_access_token(token)

    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials
    from starlette.requests import Request
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import verify_api_key

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    request = Request({"type": "http", "client": ("127.0.0.1", 12345), "headers": []})

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(credentials=creds, request=request)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_authnz_jwt_accepted_then_revoked_rejected_for_workflows_ws(isolated_test_environment, tmp_path):
    """TASK-13950.7 AC#5 on Postgres: the workflows WebSocket used the MCP JWT manager,
    so it refused every AuthNZ token and never consulted the blacklist."""
    from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
    from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

    client, _db_name = isolated_test_environment
    token = _register_and_login(client, "wf_revoke_ws_user", "Str0ngP@ssw0rd!")
    claims = get_jwt_service().decode_access_token(token)
    db = WorkflowsDatabase(str(tmp_path / "workflows.db"))
    db.create_run(
        run_id="wf-revoke",
        tenant_id="default",
        user_id=str(claims.get("user_id") or claims["sub"]),
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "wf-revoke", "version": 1, "steps": []},
    )
    client.app.dependency_overrides[wf_mod._get_db] = lambda: db
    url, headers = "/api/v1/workflows/ws?run_id=wf-revoke", {"Authorization": f"Bearer {token}"}
    try:
        with client.websocket_connect(url, headers=headers) as ws:
            assert ws.receive_json()["type"] == "snapshot"

        await _revoke_access_token(token)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(url, headers=headers):
                pass
        assert exc_info.value.code == 4401
    finally:
        client.app.dependency_overrides.pop(wf_mod._get_db, None)
        db.close()
