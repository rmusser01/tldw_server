"""Every mounted WebSocket route rejects missing credentials and revoked JWTs (TASK-13950.7).

WebSocket routes authenticate inside their handlers, which the route-auth ratchet cannot
see, so it only lists them (kind 4 in Helper_Scripts/ci/route_auth_baseline.txt). This
matrix proves the behaviour on the full app, built with the route families the ratchet
forces on:

- Handshake routes must refuse before accept: a policy close (4401, 1008) or an HTTP 401
  denial.
- First-message-frame routes accept, then must close with a policy code, without
  answering anything but an error, when the first frame is data or carries a revoked
  token. Voice's 4400 ("Expected AUTH message") counts: the frame is refused unread.

The revoked token is a real AuthNZ access token whose jti is on the token blacklist, the
check HTTP auth runs in verify_jwt_and_fetch_user. A gap found later belongs here as a
strict xfail naming the slice that fixes it, so the fix turns it red until the mark goes.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.routing import APIWebSocketRoute
from fastapi.testclient import TestClient
from Helper_Scripts.ci.route_auth_ratchet import ROUTE_POLICY_ENV
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.tests.helpers.app_main_state import reload_app_main, restore_app_main, snapshot_app_main

pytestmark = pytest.mark.integration

HANDSHAKE, FIRST_FRAME = "handshake", "first_frame"

# Served path -> (URL to connect to, where the route reads credentials).
ROUTES: dict[str, tuple[str, str]] = {
    "/api/v1/acp/multiplex": ("/api/v1/acp/multiplex", HANDSHAKE),
    "/api/v1/acp/sessions/{session_id}/ssh": ("/api/v1/acp/sessions/s1/ssh", HANDSHAKE),
    "/api/v1/acp/sessions/{session_id}/stream": ("/api/v1/acp/sessions/s1/stream", HANDSHAKE),
    "/api/v1/audio/chat/stream": ("/api/v1/audio/chat/stream", FIRST_FRAME),
    "/api/v1/audio/realtime": ("/api/v1/audio/realtime", HANDSHAKE),
    "/api/v1/audio/stream/transcribe": ("/api/v1/audio/stream/transcribe", FIRST_FRAME),
    "/api/v1/audio/stream/tts": ("/api/v1/audio/stream/tts", FIRST_FRAME),
    "/api/v1/audio/stream/tts/realtime": ("/api/v1/audio/stream/tts/realtime", FIRST_FRAME),
    "/api/v1/mcp/ws": ("/api/v1/mcp/ws", HANDSHAKE),
    "/api/v1/meetings/sessions/{session_id}/stream": ("/api/v1/meetings/sessions/s1/stream", HANDSHAKE),
    "/api/v1/persona/stream": ("/api/v1/persona/stream", HANDSHAKE),
    "/api/v1/prompt-studio/ws": ("/api/v1/prompt-studio/ws", HANDSHAKE),
    "/api/v1/prompt-studio/ws/{project_id}": ("/api/v1/prompt-studio/ws/1", HANDSHAKE),
    "/api/v1/sandbox/runs/{run_id}/stream": ("/api/v1/sandbox/runs/r1/stream", HANDSHAKE),
    "/api/v1/voice/assistant": ("/api/v1/voice/assistant", FIRST_FRAME),
    "/api/v1/watchlists/runs/{run_id}/stream": ("/api/v1/watchlists/runs/1/stream", HANDSHAKE),
    "/api/v1/workflows/ws": ("/api/v1/workflows/ws?run_id=matrix-run", HANDSHAKE),
    "/v1/realtime": ("/v1/realtime", HANDSHAKE),
}

REFUSED = {"close 1008", "close 4401", "http 401"}
FIRST_FRAME_REFUSED = REFUSED | {"close 4400"}
DATA_FRAME = {"type": "config"}
SINGLE_USER_KEY = "ws-auth-matrix-single-user-key-0123456789"


async def _seed_user_and_tokens() -> tuple[int, str, str]:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist, reset_token_blacklist
    from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

    await reset_db_pool()
    await reset_token_blacklist()
    pool = await get_db_pool()
    assert pool.db_path, "the matrix needs the SQLite AuthNZ store"
    ensure_authnz_tables(Path(pool.db_path))
    user_id = await create_authnz_test_user(pool, username="ws-matrix", email="ws-matrix@example.com")
    jwt_service = get_jwt_service()
    valid = jwt_service.create_access_token(user_id=user_id, username="ws-matrix", role="user")
    revoked = jwt_service.create_access_token(user_id=user_id, username="ws-matrix", role="user")
    assert await get_token_blacklist().revoke_token(
        jti=jwt_service.decode_access_token(revoked)["jti"],
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        user_id=user_id,
        token_type="access",  # nosec B106 - token kind label, not a credential
        reason="ws-auth-matrix",
    )
    # The TestClient runs the app on its own loop, which opens its own pool.
    await reset_token_blacklist()
    await reset_db_pool()
    return user_id, valid, revoked


def _reset_auth_singletons() -> None:
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    reset_jwt_service()


@pytest.fixture(scope="module")
def full_app(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    """The full app on a SQLite AuthNZ store, plus a user and a valid and a revoked token.

    The root conftest restores auth env vars and ``app.main`` after every test, so the
    import-time route switches apply only while the app is built here, and ``env``
    re-applies the auth settings for each test.
    """
    data_dir = tmp_path_factory.mktemp("ws_auth_matrix")
    environ = {
        "AUTH_MODE": "multi_user",
        "DATABASE_URL": f"sqlite:///{data_dir / 'users.db'}",
        "JWT_SECRET_KEY": "ws-auth-matrix-jwt-secret-0123456789abcdef",
        "SINGLE_USER_API_KEY": SINGLE_USER_KEY,
        "USER_DB_BASE_DIR": str(data_dir / "user_databases"),
    }
    route_keys = {*os.getenv("ROUTES_ENABLE", "").split(","), *ROUTE_POLICY_ENV["ROUTES_ENABLE"].split(",")}
    original_main = snapshot_app_main()
    with pytest.MonkeyPatch.context() as patch:
        for key, value in environ.items():
            patch.setenv(key, value)
        patch.setenv("MINIMAL_TEST_APP", "0")
        patch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
        patch.setenv("ROUTES_ENABLE", ",".join(sorted(key for key in route_keys if key)))
        _reset_auth_singletons()
        user_id, valid, revoked = asyncio.run(_seed_user_and_tokens())
        config._route_toggle_policy.cache_clear()
        app = reload_app_main().app
    try:
        yield SimpleNamespace(
            app=app,
            client=TestClient(app),
            environ=environ,
            user_id=user_id,
            valid=valid,
            revoked=revoked,
            workflows_db=WorkflowsDatabase(str(data_dir / "workflows.db")),
        )
    finally:
        restore_app_main(original_main)
        config._route_toggle_policy.cache_clear()
        _reset_auth_singletons()


@pytest.fixture()
def env(full_app: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    for key, value in full_app.environ.items():
        monkeypatch.setenv(key, value)
    _reset_auth_singletons()
    # The AuthNZ conftest clears overrides before every test; keep workflows off the shared DB.
    full_app.app.dependency_overrides[wf_mod._get_db] = lambda: full_app.workflows_db
    return full_app


def _refusal(
    client: TestClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    frame: Mapping[str, object] | None = None,
) -> str:
    """Connect, send ``frame`` if accepted, and describe how the server refused.

    Fails the test if a handshake route accepts, or if a first-frame route answers the
    frame with anything other than an error before closing.
    """
    try:
        with client.websocket_connect(url, headers=dict(headers or {})) as ws:
            if frame is None:
                pytest.fail(f"{url} accepted the connection")
            ws.send_json(frame)
            while True:
                message = ws.receive()
                if message["type"] == "websocket.close":
                    return f"close {message.get('code')}"
                payload = json.loads(message.get("text") or "{}")
                assert payload.get("type") in {"error", "auth_error"}, f"{url} processed a frame: {payload}"
    except WebSocketDenialResponse as denial:
        return f"http {denial.status_code}"
    except WebSocketDisconnect as closed:
        return f"close {closed.code}"


def test_matrix_covers_every_mounted_websocket_route(env: SimpleNamespace) -> None:
    served = {
        route.path for route in iter_served_routes(env.app.routes) if isinstance(route.route, APIWebSocketRoute)
    }
    assert served == set(ROUTES), "Add the new WebSocket route to ROUTES (and its auth to resolve_ws_principal)"


@pytest.mark.parametrize("route", ROUTES)
def test_rejects_missing_credentials(env: SimpleNamespace, route: str) -> None:
    url, style = ROUTES[route]
    if style == HANDSHAKE:
        assert _refusal(env.client, url) in REFUSED
    else:
        assert _refusal(env.client, url, frame=DATA_FRAME) in FIRST_FRAME_REFUSED


@pytest.mark.parametrize("route", ROUTES)
def test_rejects_missing_credentials_in_single_user_mode(
    env: SimpleNamespace, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    monkeypatch.setenv("AUTH_MODE", "single_user")
    reset_settings()
    assert get_settings().AUTH_MODE == "single_user"
    url, style = ROUTES[route]
    if style == HANDSHAKE:
        assert _refusal(env.client, url) in REFUSED
    else:
        assert _refusal(env.client, url, frame=DATA_FRAME) in FIRST_FRAME_REFUSED


@pytest.mark.parametrize("route", ROUTES)
def test_rejects_a_revoked_jwt(env: SimpleNamespace, route: str) -> None:
    url, style = ROUTES[route]
    revoked_header = {"Authorization": f"Bearer {env.revoked}"}
    if style == HANDSHAKE:
        assert _refusal(env.client, url, headers=revoked_header) in REFUSED
    else:
        assert _refusal(env.client, url, frame={"type": "auth", "token": env.revoked}) in REFUSED
        assert _refusal(env.client, url, headers=revoked_header, frame=DATA_FRAME) in FIRST_FRAME_REFUSED


def test_workflows_ws_accepts_an_authnz_access_token_and_rejects_it_revoked(env: SimpleNamespace) -> None:
    """AC#5: the workflows WebSocket verified tokens with the MCP JWT manager, so it
    rejected every AuthNZ access token and never consulted the blacklist."""
    env.workflows_db.create_run(
        run_id="matrix-run",
        tenant_id="default",
        user_id=str(env.user_id),
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "ws-auth-matrix", "version": 1, "steps": []},
    )
    url = ROUTES["/api/v1/workflows/ws"][0]

    with env.client.websocket_connect(url, headers={"Authorization": f"Bearer {env.valid}"}) as ws:
        assert ws.receive_json()["run"]["run_id"] == "matrix-run"
    with env.client.websocket_connect(url, subprotocols=["bearer", env.valid]) as ws:
        assert ws.accepted_subprotocol == "bearer"  # browsers fail a handshake that echoes nothing
        assert ws.receive_json()["type"] == "snapshot"
    assert _refusal(env.client, url, headers={"Authorization": f"Bearer {env.revoked}"}) == "close 4401"
