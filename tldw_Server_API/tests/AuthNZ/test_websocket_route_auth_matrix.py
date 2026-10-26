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
- ACP, prompt-studio and watchlists still take the query ``token``/``api_key`` that shipped
  clients send (ACP and prompt-studio also the ``x-api-key`` subprotocol), through a
  deprecated shim that hands them to the resolver's own verification. Their section below
  proves the shim admits, refuses and warns like the resolver, and that the retired
  channels are refused. Sandbox had no such client, so its query credentials are retired.

The revoked token is a real AuthNZ access token whose jti is on the token blacklist, the
check HTTP auth runs in verify_jwt_and_fetch_user. Known gaps are strict xfails naming
the slice that fixes them (none today), so the fix turns them red until the mark goes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FrameTimeout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
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

# Outcomes: "close N" / "http N" is a refusal before accept; "closed N" is a close after it.
REFUSED = {"close 1008", "close 4401", "http 401"}
# A revoked token in an auth frame: refused before accept or closed right after it.
FRAME_REFUSED = REFUSED | {"closed 1008", "closed 4401"}
# A data frame before auth also counts as refused unread with voice's 4400 ("Expected AUTH message").
FIRST_FRAME_REFUSED = FRAME_REFUSED | {"closed 4400"}
AUTH_CODES = {1008, 4400, 4401, 4403}
# Before accept, only "this resource does not exist" shows a valid caller got past authentication.
RESOURCE_MISSING = {"close 4404", "http 404"}
DATA_FRAME = {"type": "config"}
SINGLE_USER_KEY = "ws-auth-matrix-single-user-key-0123456789"
FRAME_TIMEOUT_S = 10.0
# A refused auth frame is answered at once; a route still silent after this has taken the
# credential and is waiting for its config frame.
ADMITTED_AFTER_S = 1.0
MAX_FRAMES = 8
SANDBOX = "/api/v1/sandbox/runs/{run_id}/stream"
WATCHLISTS = "/api/v1/watchlists/runs/{run_id}/stream"
# Routes whose deprecated credential shim (websocket_principal.resolve_ws_principal_or_deprecated_query)
# reads the query, and for ACP and prompt-studio (agent_client_protocol._authenticate_ws) the
# x-api-key subprotocol too; and of those, the ones that accept once authenticated, with the
# API-key scope each needs. The watchlists URL names run 1, which the fixture seeds.
SHIM_ROUTES = {path for path in ROUTES if path.startswith(("/api/v1/acp/", "/api/v1/prompt-studio/", WATCHLISTS))}
SHIM_ACCEPTING = {"/api/v1/acp/multiplex": "write", "/api/v1/prompt-studio/ws": "read", WATCHLISTS: "read"}
QUERY_CHANNELS = ("query token", "query api_key")
DEPRECATED_CHANNELS = (*QUERY_CHANNELS, "x-api-key subprotocol")
SHIM_CASES = [
    (route, channel)
    for route in SHIM_ACCEPTING
    for channel in (QUERY_CHANNELS if route == WATCHLISTS else DEPRECATED_CHANNELS)
]
DEPRECATION_MARK = "Deprecated WebSocket credential channel"

def _with_query(url: str, name: str, value: str) -> str:
    return f"{url}{'&' if '?' in url else '?'}{name}={value}"


async def _seed_user_and_tokens() -> tuple[int, str, str, str, str, int, str]:
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
    # A second real user, for cross-user ownership checks.
    other_id = await create_authnz_test_user(pool, username="ws-matrix-other", email="ws-matrix-other@example.com")
    other_valid = jwt_service.create_access_token(user_id=other_id, username="ws-matrix-other", role="user")
    # The TestClient runs the app on its own loop, which opens its own pool.
    await reset_api_key_manager()
    await reset_token_blacklist()
    await reset_db_pool()
    return user_id, valid, revoked, api_key, write_key, other_id, other_valid


def _seed_watchlists_runs(user_id: int) -> None:
    """Give the matrix user and the single user a completed run 1, so the watchlists URL in
    ROUTES names a run each owns."""
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    for owner in {user_id, get_settings().SINGLE_USER_FIXED_ID}:
        db = WatchlistsDatabase.for_user(owner)
        job = db.create_job(
            name="ws-auth-matrix",
            description=None,
            scope_json="{}",
            schedule_expr=None,
            schedule_timezone=None,
            active=False,
            max_concurrency=None,
            per_host_delay_ms=None,
            retry_policy_json=None,
            output_prefs_json=None,
        )
        assert db.create_run(job_id=job.id, status="completed").id == 1


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
        user_id, valid, revoked, api_key, write_key, other_id, other_valid = asyncio.run(_seed_user_and_tokens())
        config._route_toggle_policy.cache_clear()
        app = reload_app_main().app
        _seed_watchlists_runs(user_id)
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
            other_id=other_id,
            other_valid=other_valid,
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


def _past_auth(outcome: str) -> bool:
    """True when ``outcome`` shows the credential was taken: anything but an auth refusal.

    After accept, any close other than an auth code counts (a route may close 1000, or 1011
    when its backend is unavailable); before accept, only a missing resource does.
    """
    if outcome in {"accepted", "silent"} or outcome.startswith("answered "):
        return True
    if outcome.startswith("closed "):
        return int(outcome.split()[1]) not in AUTH_CODES
    return outcome in RESOURCE_MISSING


def _is_teardown_noise(exc: Exception) -> bool:
    """The route reacting to the disconnect our teardown sends.

    Starlette refuses a send or receive on a closed socket with a RuntimeError ("Cannot
    call ..."; WebSocketDisconnected from 1.10), e.g. a handler that closes twice.
    """
    return isinstance(exc, WebSocketDisconnect) or (
        isinstance(exc, RuntimeError) and str(exc).startswith('Cannot call "')
    )


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

    Returns ``"close <code>"`` or ``"http <status>"`` when refused before accept,
    ``"accepted"`` when a handshake succeeds and no frame is sent, ``"closed <code>"`` for a
    close after accept, ``"answered <type>"`` for the first non-error reply to the frame,
    ``"silent"`` when nothing arrives in time, and ``"gone"`` when the route's side of the
    socket vanished without a close (its exception then surfaces from the teardown).
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
        # A route that already closed still has its close queued for the reads below.
        with contextlib.suppress(WebSocketDisconnect, anyio.ClosedResourceError, anyio.BrokenResourceError):
            ws.send_json(frame)
        for _ in range(MAX_FRAMES):
            try:
                message = _receive(ws, timeout)
            except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream):
                return "gone"
            if message is None:
                return "silent"
            if message["type"] == "websocket.close":
                return f"closed {message.get('code')}"
            payload = json.loads(str(message.get("text") or "{}"))
            if payload.get("type") not in {"error", "auth_error"}:
                return f"answered {payload.get('type')}"
        return "silent"
    finally:
        # Leaving delivers a disconnect. A route's reaction to it is not part of the outcome
        # (CI saw audio/stream/tts close twice there); any other exception is a real crash.
        try:
            ws.__exit__(None, None, None)
        except Exception as exc:
            if not _is_teardown_noise(exc):
                raise


def _accepted_subprotocol(
    client: TestClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    subprotocols: list[str] | None = None,
) -> str | None:
    """Open a socket the route must accept, and return the subprotocol it echoed.

    A refusal raises, so the test fails. Leaving uses ``_outcome``'s teardown, so a route's
    reaction to our disconnect is not mistaken for a failure. The headers are copied, because
    TestClient writes the handshake headers into the dict it is given.
    """
    ws = client.websocket_connect(url, headers=dict(headers or {}), subprotocols=subprotocols).__enter__()
    try:
        accepted: str | None = ws.accepted_subprotocol
        return accepted
    finally:
        try:
            ws.__exit__(None, None, None)
        except Exception as exc:
            if not _is_teardown_noise(exc):
                raise


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


def _scratch_app() -> FastAPI:
    """Routes shaped like the matrix's routes' reactions to a valid auth frame."""
    app = FastAPI()

    @app.websocket("/waits")
    async def waits(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.receive_json()
        await websocket.receive_json()  # never arrives

    @app.websocket("/backend-missing")
    async def backend_missing(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.receive_json()
        await websocket.close(code=1011)

    @app.websocket("/closes-twice")
    async def closes_twice(websocket: WebSocket) -> None:
        # audio/stream/tts on CI: our teardown disconnect lands in its prompt wait, and its
        # error path closes the socket twice, which Starlette refuses.
        await websocket.accept()
        await websocket.receive_json()
        try:
            await websocket.receive_json()
        except WebSocketDisconnect:
            await websocket.close(code=1008)
            await websocket.close(code=4400)

    @app.websocket("/refuses")
    async def refuses(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.receive_json()
        await websocket.close(code=4401)

    @app.websocket("/refuses-before-accept")
    async def refuses_before_accept(websocket: WebSocket) -> None:
        await websocket.close(code=1000)

    return app


@pytest.mark.parametrize(
    ("path", "expected", "past_auth"),
    [
        ("/waits", "silent", True),
        ("/backend-missing", "closed 1011", True),
        ("/closes-twice", "silent", True),
        ("/refuses", "closed 4401", False),
        ("/refuses-before-accept", "close 1000", False),
    ],
)
def test_outcome_classifies_without_hanging_or_raising(
    path: str, expected: str, past_auth: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-test of the helper: bounded waits, close classification, and teardown races.

    The TestClient's teardown is slowed so a route always handles our disconnect before it
    is cancelled, which is how CI ran into audio/stream/tts closing twice.
    """
    from starlette.testclient import WebSocketTestSession

    close = WebSocketTestSession.close

    def slow_close(self: WebSocketTestSession, code: int = 1000, reason: str | None = None) -> None:
        close(self, code, reason)
        time.sleep(0.2)

    monkeypatch.setattr(WebSocketTestSession, "close", slow_close)
    with TestClient(_scratch_app()) as client:
        outcome = _outcome(client, path, frame={"type": "auth", "token": "t"}, timeout=0.2)
    assert outcome == expected
    assert _past_auth(outcome) is past_auth


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
        assert _outcome(env.client, url, frame={"type": "auth", "token": env.revoked}) in FRAME_REFUSED
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
    assert _past_auth(outcome), outcome


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


@pytest.mark.parametrize(("route", "channel"), SHIM_CASES)
def test_deprecated_channel_admits_a_valid_credential_with_one_warning(
    env: SimpleNamespace, route: str, channel: str
) -> None:
    """Shipped clients (useACPSession, ACPWebSocketClient, prompt-studio-stream, the ACP SSH
    panel, watchlists-stream) still send these; the shim keeps them working until the clients move."""
    credential, request = _deprecated_request(env, route, channel, valid=True)
    with _warnings() as warned:
        echoed = _accepted_subprotocol(env.client, **request)  # type: ignore[arg-type]
    # Browsers fail a handshake that echoes nothing; never echo the key itself.
    assert echoed == ("x-api-key" if channel == "x-api-key subprotocol" else None)
    deprecations = _deprecations(warned)
    assert len(deprecations) == 1 and channel in deprecations[0], warned
    assert not [message for message in warned if credential in message]


@pytest.mark.parametrize(("route", "channel"), SHIM_CASES)
def test_deprecated_channel_refuses_a_revoked_or_unknown_credential(
    env: SimpleNamespace, route: str, channel: str
) -> None:
    credential, request = _deprecated_request(env, route, channel, valid=False)
    with _warnings() as warned:
        assert _outcome(env.client, **request) == "close 4401"  # type: ignore[arg-type]
    assert [message for message in _deprecations(warned) if channel in message], warned
    assert not [message for message in warned if credential in message]


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_retired_channels_are_refused(env: SimpleNamespace, route: str) -> None:
    """``Authorization: X-API-KEY <key>`` and the ``api-key, <key>`` subprotocol had no client."""
    url = ROUTES[route][0]
    key = env.write_key if SHIM_ACCEPTING[route] == "write" else env.api_key
    assert _outcome(env.client, url, headers={"Authorization": f"X-API-KEY {key}"}) == "close 4401"
    assert _outcome(env.client, url, subprotocols=["api-key", key]) == "close 4401"


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_header_credentials_need_no_shim(env: SimpleNamespace, route: str) -> None:
    """The resolver's own channels: no deprecation warning, and the bearer marker is echoed. A header
    credential wins over the query (watchlists used to verify ``?token=`` first), so a revoked
    query token beside a valid header is never read."""
    key = env.write_key if SHIM_ACCEPTING[route] == "write" else env.api_key
    url = ROUTES[route][0]
    with _warnings() as warned:
        assert _accepted_subprotocol(env.client, url, subprotocols=["bearer", env.valid]) == "bearer"
        assert _accepted_subprotocol(env.client, url, subprotocols=["bearer", key]) == "bearer"  # key as bearer
        assert _accepted_subprotocol(env.client, url, headers={"X-API-KEY": key}) is None
        revoked_query = _with_query(url, "token", env.revoked)
        assert _accepted_subprotocol(env.client, revoked_query, headers={"Authorization": f"Bearer {env.valid}"}) is None
    assert _deprecations(warned) == []


def _single_user(monkeypatch: pytest.MonkeyPatch, **environ: str) -> None:
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("AUTH_MODE", "single_user")
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    reset_settings()


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_deprecated_query_admits_the_single_user_key(
    env: SimpleNamespace, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-user WebUI outside cookie quickstart, and the extension, send ``?api_key=``."""
    _single_user(monkeypatch)
    url = ROUTES[route][0]
    with _warnings() as warned:
        assert _accepted_subprotocol(env.client, _with_query(url, "api_key", SINGLE_USER_KEY)) is None
        assert _outcome(env.client, _with_query(url, "api_key", f"{SINGLE_USER_KEY}-wrong")) == "close 4401"
    deprecations = _deprecations(warned)
    assert len(deprecations) == 2 and all("query api_key" in message for message in deprecations), warned
    assert not [message for message in warned if SINGLE_USER_KEY in message]


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_deprecated_query_keeps_the_single_user_ip_allowlist(
    env: SimpleNamespace, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verifier sees the socket's peer (request.client), so the allowlist still binds."""
    _single_user(monkeypatch, SINGLE_USER_ALLOWED_IPS="10.1.2.3")
    url = _with_query(ROUTES[route][0], "api_key", SINGLE_USER_KEY)
    outside, inside = TestClient(env.app, client=("10.9.9.9", 50000)), TestClient(env.app, client=("10.1.2.3", 50000))
    try:
        assert _outcome(outside, url) in {"close 4401", "close 4403"}
        assert _outcome(inside, url) == "accepted"
    finally:
        outside.close()
        inside.close()


BASIC_PROXY_AUTH = {"Authorization": "Basic cHJveHktdXNlcjpwcm94eS1wYXNz"}


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_a_proxy_basic_authorization_is_not_a_credential(
    env: SimpleNamespace, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Basic-auth reverse proxy makes the browser attach its ``Authorization`` to the upgrade.
    It neither blocks the deprecated query nor shadows the bearer subprotocol, and alone it
    authenticates no one."""
    url = ROUTES[route][0]
    assert _outcome(env.client, url, headers=BASIC_PROXY_AUTH) == "close 4401"
    echoed = _accepted_subprotocol(env.client, url, headers=BASIC_PROXY_AUTH, subprotocols=["bearer", env.valid])
    assert echoed == "bearer"
    _single_user(monkeypatch)
    with _warnings() as warned:
        query_key = _with_query(url, "api_key", SINGLE_USER_KEY)
        assert _accepted_subprotocol(env.client, query_key, headers=BASIC_PROXY_AUTH) is None
    assert [message for message in _deprecations(warned) if "query api_key" in message], warned


@pytest.mark.parametrize("route", SHIM_ACCEPTING)
def test_deprecated_query_pair_resolves_like_both_http_headers(env: SimpleNamespace, route: str) -> None:
    """``?token=&api_key=`` goes to the verifier as both headers, as HTTP resolves them: the JWT
    first, and the API key only when the JWT is refused with 401."""
    url = ROUTES[route][0]
    key = env.write_key if SHIM_ACCEPTING[route] == "write" else env.api_key
    assert _outcome(env.client, f"{url}?token={env.valid}&api_key=tldw_ws-matrix-unknown-key") == "accepted"
    assert _outcome(env.client, f"{url}?token={env.revoked}&api_key={key}") == "accepted"
    assert _outcome(env.client, f"{url}?token={env.revoked}&api_key=tldw_ws-matrix-unknown-key") == "close 4401"


def test_sandbox_retired_its_query_credentials(env: SimpleNamespace) -> None:
    """``?auth_token=`` and, with signed URLs off, ``?token=`` had no client, so they are gone
    rather than deprecated (TASK-13950.7). A signed URL's ``token`` is its HMAC, not a credential."""
    url = ROUTES[SANDBOX][0]
    with _warnings() as warned:
        assert _outcome(env.client, _with_query(url, "auth_token", env.valid)) == "close 4401"
        assert _outcome(env.client, _with_query(url, "token", env.valid)) == "close 4401"
    assert _deprecations(warned) == []


@pytest.mark.parametrize("route", [SANDBOX, WATCHLISTS])
def test_browser_bearer_subprotocol_and_single_user_key(
    env: SimpleNamespace, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Browsers cannot set WebSocket headers. Before the core resolver these routes ignored the
    bearer subprotocol, so a browser could only reach them through a query credential; the
    single-user key in ``X-API-KEY`` already worked and still does."""
    url = ROUTES[route][0]
    assert _accepted_subprotocol(env.client, url, subprotocols=["bearer", env.valid]) == "bearer"
    _single_user(monkeypatch)
    assert _accepted_subprotocol(env.client, url, headers={"X-API-KEY": SINGLE_USER_KEY}) is None
    assert _accepted_subprotocol(env.client, url, subprotocols=["bearer", SINGLE_USER_KEY]) == "bearer"


def _track_sandbox_run(owner_id: int) -> str:
    """A run the sandbox store records as ``owner_id``'s, as ``POST /sandbox/runs`` leaves it.

    Only a tracked run is owner-checked: in test mode an untracked one is streamed to anyone.
    """
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_ep
    from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus

    run_id = f"ws-matrix-owned-by-{owner_id}"
    sandbox_ep._service._orch._store.put_run(owner_id, RunStatus(id=run_id, phase=RunPhase.completed))
    return run_id


def _signed_sandbox_url(run_id: str, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Turn signed stream URLs on and return a valid one for ``run_id``, and its signature."""
    secret, exp = "ws-matrix-signing-secret", int(time.time()) + 300
    monkeypatch.setenv("SANDBOX_WS_SIGNED_URLS", "true")
    monkeypatch.setenv("SANDBOX_WS_SIGNING_SECRET", secret)
    signature = hmac.new(secret.encode(), f"{run_id}:{exp}".encode(), hashlib.sha256).hexdigest()
    return f"/api/v1/sandbox/runs/{run_id}/stream?token={signature}&exp={exp}", signature


@pytest.mark.parametrize("signed_url", [False, True], ids=["unsigned", "signed"])
def test_sandbox_stream_refuses_another_users_run(
    env: SimpleNamespace, signed_url: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid credential is not enough: the run must be the caller's. A signed URL names the run,
    not the user, so a leaked one plus anyone's credential still meets the owner check (4404)."""
    run_id = _track_sandbox_run(env.user_id)
    if signed_url:
        url, _signature = _signed_sandbox_url(run_id, monkeypatch)
    else:
        url = f"/api/v1/sandbox/runs/{run_id}/stream"
        monkeypatch.setenv("SANDBOX_WS_SIGNED_URLS", "false")

    assert _outcome(env.client, url, headers={"Authorization": f"Bearer {env.other_valid}"}) == "close 4404"
    assert _outcome(env.client, url, headers={"Authorization": f"Bearer {env.valid}"}) == "accepted"


def test_sandbox_signed_url_with_the_browser_bearer_subprotocol(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented browser path: the signed ``log_stream_url`` plus ``bearer, <token>``. The
    signature is checked in addition to the credential, and is not a credential itself."""
    url, signature = _signed_sandbox_url(_track_sandbox_run(env.user_id), monkeypatch)
    bad_signature = url.replace(signature, signature[:-1] + ("0" if signature[-1] != "0" else "1"))

    assert _accepted_subprotocol(env.client, url, subprotocols=["bearer", env.valid]) == "bearer"
    assert _outcome(env.client, bad_signature, subprotocols=["bearer", env.valid]) == "close 1008"
    assert _outcome(env.client, url) == "close 4401"


@pytest.mark.parametrize("channel", ["X-API-KEY", "bearer subprotocol"])
def test_sandbox_single_user_key_keeps_the_ip_allowlist(
    env: SimpleNamespace, channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old sandbox resolver checked SINGLE_USER_ALLOWED_IPS itself; the HTTP verifier now does,
    from the socket's peer."""
    _single_user(monkeypatch, SINGLE_USER_ALLOWED_IPS="10.1.2.3")
    if channel == "X-API-KEY":
        request: dict[str, object] = {"headers": {"X-API-KEY": SINGLE_USER_KEY}}
    else:
        request = {"subprotocols": ["bearer", SINGLE_USER_KEY]}
    url = ROUTES[SANDBOX][0]
    outside, inside = TestClient(env.app, client=("10.9.9.9", 50000)), TestClient(env.app, client=("10.1.2.3", 50000))
    try:
        assert _outcome(outside, url, **request) == "close 4401"  # type: ignore[arg-type]
        assert _outcome(inside, url, **request) == "accepted"  # type: ignore[arg-type]
    finally:
        outside.close()
        inside.close()
