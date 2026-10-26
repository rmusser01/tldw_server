"""Every mounted WebSocket route rejects missing credentials and revoked JWTs (TASK-13950.7).

WebSocket routes authenticate inside their handlers, which the route-auth ratchet cannot
see, so it only lists them (kind 4 in Helper_Scripts/ci/route_auth_baseline.txt). This
matrix proves the behaviour on the full app, built with the ratchet's own route policy,
and checks its route list against the ratchet's ``WS`` baseline lines:

- Handshake routes must refuse before accept: a policy close (4401, 1008) or an HTTP 401
  denial.
- First-message-frame routes accept, then must close with a policy code, without
  answering anything but an error, when the first frame is data or carries a revoked
  token. Voice's 4400 ("Expected AUTH message") counts: the frame is refused unread.
- The revoked token is tried in the Authorization header, in the ``token`` query
  parameter, and, for first-frame routes, in the auth frame.
- A valid AuthNZ access token must get past authentication on every route (the control
  column), so a refusal above means something.
- ACP and prompt-studio still take the query ``token``/``api_key`` and the ``x-api-key``
  subprotocol that shipped clients send, through a deprecated shim that hands them to the
  resolver's own verification. Their section below proves the shim admits, refuses and
  warns like the resolver, and that the retired channels are refused.

The revoked token is a real AuthNZ access token whose jti is on the token blacklist, the
check HTTP auth runs in verify_jwt_and_fetch_user. Known gaps are strict xfails naming
the slice that fixes them (none today), so the fix turns them red until the mark goes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FrameTimeout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.routing import APIWebSocketRoute
from fastapi.testclient import TestClient
from Helper_Scripts.ci.route_auth_ratchet import ROUTE_POLICY_ENV, read_baseline
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
# A valid token is refused if the route answers like this; anything else (accepted, an
# answered frame, a resource-level 4404) is past authentication.
AUTH_REFUSED = FIRST_FRAME_REFUSED | {"close 4403", "http 403"}
DATA_FRAME = {"type": "config"}
SINGLE_USER_KEY = "ws-auth-matrix-single-user-key-0123456789"
FRAME_TIMEOUT_S = 10.0
# A refused auth frame is answered at once; a route still silent after this has taken the
# credential and is waiting for its config frame.
ADMITTED_AFTER_S = 1.0
MAX_FRAMES = 8
# Routes whose deprecated credential shim (agent_client_protocol._authenticate_ws) reads the
# query and the x-api-key subprotocol; and of those, the ones that accept once authenticated,
# with the API-key scope each needs.
SHIM_ROUTES = {path for path in ROUTES if path.startswith(("/api/v1/acp/", "/api/v1/prompt-studio/"))}
SHIM_ACCEPTING = {"/api/v1/acp/multiplex": "write", "/api/v1/prompt-studio/ws": "read"}
DEPRECATED_CHANNELS = ("query token", "query api_key", "x-api-key subprotocol")
DEPRECATION_MARK = "Deprecated WebSocket credential channel"

def _with_query(url: str, name: str, value: str) -> str:
    return f"{url}{'&' if '?' in url else '?'}{name}={value}"


async def _seed_user_and_tokens() -> tuple[int, str, str, str, str]:
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager, reset_api_key_manager
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
    api_keys = await get_api_key_manager()
    api_key = (await api_keys.create_api_key(user_id=user_id, name="ws-matrix", scope="read"))["key"]
    write_key = (await api_keys.create_api_key(user_id=user_id, name="ws-matrix-write", scope="write"))["key"]
    # The TestClient runs the app on its own loop, which opens its own pool.
    await reset_api_key_manager()
    await reset_token_blacklist()
    await reset_db_pool()
    return user_id, valid, revoked, api_key, write_key


def _reset_auth_singletons() -> None:
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    reset_jwt_service()


@pytest.fixture(scope="module")
def full_app(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    """The full app on a SQLite AuthNZ store, plus a user, a valid and a revoked token, and
    read-scope and write-scope API keys.

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
    original_main = snapshot_app_main()
    with pytest.MonkeyPatch.context() as patch:
        for key, value in {**environ, **ROUTE_POLICY_ENV}.items():  # the ratchet's own route policy
            patch.setenv(key, value)
        patch.setenv("MINIMAL_TEST_APP", "0")
        patch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
        _reset_auth_singletons()
        user_id, valid, revoked, api_key, write_key = asyncio.run(_seed_user_and_tokens())
        config._route_toggle_policy.cache_clear()
        app = reload_app_main().app
    client = TestClient(app)
    workflows_db = WorkflowsDatabase(str(data_dir / "workflows.db"))
    try:
        yield SimpleNamespace(
            app=app,
            client=client,
            environ=environ,
            user_id=user_id,
            valid=valid,
            revoked=revoked,
            api_key=api_key,
            write_key=write_key,
            workflows_db=workflows_db,
        )
    finally:
        client.close()
        workflows_db.close()
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


def _receive(ws: object, timeout: float) -> dict[str, object] | None:
    """The next server message, or None if none arrives within ``timeout`` seconds.

    WebSocketTestSession.receive() blocks forever, and a route that accepts and then
    waits would hang the required gate instead of failing it.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        message: dict[str, object] = pool.submit(ws.receive).result(timeout=timeout)  # type: ignore[attr-defined]
        return message
    except FrameTimeout:
        return None
    finally:
        pool.shutdown(wait=False)


def _outcome(
    client: TestClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    subprotocols: list[str] | None = None,
    frame: Mapping[str, object] | None = None,
    timeout: float = FRAME_TIMEOUT_S,
) -> str:
    """Connect, send ``frame`` if accepted, and describe what the server did.

    Returns ``"close <code>"`` or ``"http <status>"`` for a refusal, ``"accepted"`` when a
    handshake succeeds and no frame is sent, ``"answered <type>"`` when the first
    non-error reply to the frame arrives, and ``"silent"`` when nothing arrives in time.
    """
    try:
        ws = client.websocket_connect(url, headers=dict(headers or {}), subprotocols=subprotocols).__enter__()
    except WebSocketDenialResponse as denial:
        return f"http {denial.status_code}"
    except WebSocketDisconnect as closed:
        return f"close {closed.code}"
    try:
        if frame is None:
            return "accepted"
        ws.send_json(frame)
        for _ in range(MAX_FRAMES):
            message = _receive(ws, timeout)
            if message is None:
                return "silent"
            if message["type"] == "websocket.close":
                return f"close {message.get('code')}"
            payload = json.loads(str(message.get("text") or "{}"))
            if payload.get("type") not in {"error", "auth_error"}:
                return f"answered {payload.get('type')}"
        return "silent"
    finally:
        # Leaving delivers a disconnect; a route that raises on it has already answered.
        with contextlib.suppress(WebSocketDisconnect):
            ws.__exit__(None, None, None)


@contextlib.contextmanager
def _warnings() -> Iterator[list[str]]:
    """Collect the server's WARNING log messages while the block runs."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


def _deprecations(messages: list[str]) -> list[str]:
    return [message for message in messages if DEPRECATION_MARK in message]


def test_outcome_returns_when_a_route_accepts_and_waits() -> None:
    """Self-test: a route that takes a frame and never answers must not hang the gate."""
    app = FastAPI()

    @app.websocket("/waits")
    async def waits(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.receive_json()
        await websocket.receive_json()  # never arrives

    with TestClient(app) as client:
        assert _outcome(client, "/waits", frame=DATA_FRAME, timeout=0.2) == "silent"


def test_matrix_covers_every_mounted_websocket_route(env: SimpleNamespace) -> None:
    served = {
        route.path for route in iter_served_routes(env.app.routes) if isinstance(route.route, APIWebSocketRoute)
    }
    assert served == set(ROUTES), "Add the new WebSocket route to ROUTES (and its auth to resolve_ws_principal)"


def test_matrix_matches_the_route_auth_ratchet_baseline() -> None:
    """The ratchet lists every WebSocket route as a reviewed entry; each must be proven here."""
    baselined = {entry.removeprefix("WS ") for entry in read_baseline() if entry.startswith("WS ")}
    assert baselined == set(ROUTES)


@pytest.mark.parametrize("route", ROUTES)
def test_rejects_missing_credentials(env: SimpleNamespace, route: str) -> None:
    url, style = ROUTES[route]
    if style == HANDSHAKE:
        assert _outcome(env.client, url) in REFUSED
    else:
        assert _outcome(env.client, url, frame=DATA_FRAME) in FIRST_FRAME_REFUSED


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
        assert _outcome(env.client, url) in REFUSED
    else:
        assert _outcome(env.client, url, frame=DATA_FRAME) in FIRST_FRAME_REFUSED


@pytest.mark.parametrize("route", ROUTES)
def test_rejects_a_revoked_jwt(env: SimpleNamespace, route: str) -> None:
    url, style = ROUTES[route]
    revoked_header = {"Authorization": f"Bearer {env.revoked}"}
    if style == HANDSHAKE:
        assert _outcome(env.client, url, headers=revoked_header) in REFUSED
    else:
        assert _outcome(env.client, url, frame={"type": "auth", "token": env.revoked}) in REFUSED
        assert _outcome(env.client, url, headers=revoked_header, frame=DATA_FRAME) in FIRST_FRAME_REFUSED


@pytest.mark.parametrize("route", ROUTES)
def test_rejects_a_revoked_jwt_in_the_query(env: SimpleNamespace, route: str) -> None:
    url, style = ROUTES[route]
    url = _with_query(url, "token", env.revoked)
    with _warnings() as warned:
        if style == HANDSHAKE:
            assert _outcome(env.client, url) in REFUSED
        else:
            # An auth frame without a token makes a route fall back to the query token.
            assert _outcome(env.client, url, frame={"type": "auth"}) in FIRST_FRAME_REFUSED
    if route in SHIM_ROUTES:
        # Refused because the shim verified the token, not because it ignored it.
        assert [message for message in _deprecations(warned) if "query token" in message], warned
        assert not [message for message in warned if env.revoked in message]


@pytest.mark.parametrize("route", ROUTES)
def test_lets_a_valid_authnz_access_token_through(env: SimpleNamespace, route: str) -> None:
    """Control column: without it, a route that refuses everyone passes the matrix."""
    url, style = ROUTES[route]
    if style == HANDSHAKE:
        outcome = _outcome(env.client, url, headers={"Authorization": f"Bearer {env.valid}"})
    else:
        outcome = _outcome(env.client, url, frame={"type": "auth", "token": env.valid}, timeout=ADMITTED_AFTER_S)
    assert outcome not in AUTH_REFUSED, outcome


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
    with env.client.websocket_connect(url, headers={"X-API-KEY": env.api_key}) as ws:  # multi-user key
        assert ws.receive_json()["type"] == "snapshot"
    assert _outcome(env.client, url, headers={"Authorization": f"Bearer {env.revoked}"}) == "close 4401"


def _deprecated_request(env: SimpleNamespace, route: str, channel: str, *, valid: bool) -> tuple[str, dict[str, object]]:
    """The credential, and the websocket_connect arguments that send it on ``channel``."""
    url = ROUTES[route][0]
    if channel == "query token":
        token = env.valid if valid else env.revoked
        return token, {"url": _with_query(url, "token", token)}
    key = (env.write_key if SHIM_ACCEPTING[route] == "write" else env.api_key) if valid else "tldw_ws-matrix-unknown-key"
    if channel == "query api_key":
        return key, {"url": _with_query(url, "api_key", key)}
    return key, {"url": url, "subprotocols": ["x-api-key", key]}


@pytest.mark.parametrize("channel", DEPRECATED_CHANNELS)
@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_acp_deprecated_channel_admits_a_valid_credential_with_one_warning(
    env: SimpleNamespace, route: str, channel: str
) -> None:
    """Shipped clients (useACPSession, ACPWebSocketClient, prompt-studio-stream, the ACP SSH
    panel) still send these; the shim keeps them working until the clients move."""
    credential, request = _deprecated_request(env, route, channel, valid=True)
    with _warnings() as warned:
        with env.client.websocket_connect(**request) as ws:
            # Browsers fail a handshake that echoes nothing; never echo the key itself.
            assert ws.accepted_subprotocol == ("x-api-key" if channel == "x-api-key subprotocol" else None)
    deprecations = _deprecations(warned)
    assert len(deprecations) == 1 and channel in deprecations[0], warned
    assert not [message for message in warned if credential in message]


@pytest.mark.parametrize("channel", DEPRECATED_CHANNELS)
@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_acp_deprecated_channel_refuses_a_revoked_or_unknown_credential(
    env: SimpleNamespace, route: str, channel: str
) -> None:
    credential, request = _deprecated_request(env, route, channel, valid=False)
    with _warnings() as warned:
        assert _outcome(env.client, **request) == "close 4401"  # type: ignore[arg-type]
    assert [message for message in _deprecations(warned) if channel in message], warned
    assert not [message for message in warned if credential in message]


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_acp_retired_channels_are_refused(env: SimpleNamespace, route: str) -> None:
    """``Authorization: X-API-KEY <key>`` and the ``api-key, <key>`` subprotocol had no client."""
    url = ROUTES[route][0]
    key = env.write_key if SHIM_ACCEPTING[route] == "write" else env.api_key
    assert _outcome(env.client, url, headers={"Authorization": f"X-API-KEY {key}"}) == "close 4401"
    assert _outcome(env.client, url, subprotocols=["api-key", key]) == "close 4401"


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_acp_header_credentials_need_no_shim(env: SimpleNamespace, route: str) -> None:
    """The resolver's own channels: no deprecation warning, and the bearer marker is echoed."""
    key = env.write_key if SHIM_ACCEPTING[route] == "write" else env.api_key
    with _warnings() as warned:
        with env.client.websocket_connect(ROUTES[route][0], subprotocols=["bearer", env.valid]) as ws:
            assert ws.accepted_subprotocol == "bearer"
        with env.client.websocket_connect(ROUTES[route][0], subprotocols=["bearer", key]) as ws:  # key as bearer
            assert ws.accepted_subprotocol == "bearer"
        with env.client.websocket_connect(ROUTES[route][0], headers={"X-API-KEY": key}):
            pass
    assert _deprecations(warned) == []


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_acp_deprecated_query_admits_the_single_user_key(
    env: SimpleNamespace, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-user WebUI outside cookie quickstart, and the extension, send ``?api_key=``."""
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("AUTH_MODE", "single_user")
    reset_settings()
    with env.client.websocket_connect(_with_query(ROUTES[route][0], "api_key", SINGLE_USER_KEY)):
        pass
