import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Audio import streaming_service
from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.tests.Audio.ws_test_helpers import scripted_websocket

pytestmark = pytest.mark.unit


_JWT = AuthPrincipal(kind="user", user_id=7, token_type="access")
_KEY = AuthPrincipal(kind="api_key", user_id=7, api_key_id=3, token_type="api_key")
_SINGLE_USER = AuthPrincipal(kind="user", user_id=1, token_type="api_key", subject="single_user")
_COOKIE = AuthPrincipal(kind="user", user_id=1, token_type="single_user_session", subject="single_user")


def _verifier(monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal = _SINGLE_USER) -> list[tuple]:
    """Stand in for HTTP's get_auth_principal; "revoked" is refused."""
    seen: list[tuple] = []

    async def verify(request):
        credential = (request.headers.get("authorization"), request.headers.get("x-api-key"))
        seen.append(credential)
        if "Bearer revoked" in credential:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        return principal

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)
    return seen


def _cookie(monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal | None) -> None:
    async def resolve(websocket, **_kwargs):
        if principal is None:
            return None
        websocket.state.auth_principal = principal
        return object()

    monkeypatch.setattr(websocket_principal, "resolve_single_user_cookie_websocket", resolve)


def _scoped_jwt(monkeypatch: pytest.MonkeyPatch, claims: dict, quota_ok: bool = True) -> list[dict]:
    """A verified JWT whose claims carry virtual-key limits; returns the quota calls."""
    from tldw_Server_API.app.core.AuthNZ import database, jwt_service, quotas

    calls: list[dict] = []

    async def jwt_quota(**kwargs):
        calls.append(kwargs)
        return quota_ok, 1

    async def pool():
        return "pool"

    monkeypatch.setattr(jwt_service, "get_jwt_service", lambda: SimpleNamespace(decode_access_token=lambda _token: claims))
    monkeypatch.setattr(quotas, "increment_and_check_jwt_quota", jwt_quota)
    monkeypatch.setattr(database, "get_db_pool", pool)
    _cookie(monkeypatch, None)
    _verifier(monkeypatch, _JWT)
    return calls


def _authenticate(websocket, **kwargs):
    return streaming_service._audio_ws_authenticate(
        websocket, endpoint_id=kwargs.pop("endpoint_id", "audio.stream.tts"), ws_path="/api/v1/audio/stream/tts", **kwargs
    )


def _closed(sent: list[dict]) -> int | None:
    return next((message["code"] for message in sent if message["type"] == "websocket.close"), None)


def _errors(sent: list[dict]) -> list[str]:
    return [json.loads(message["text"])["message"] for message in sent if message["type"] == "websocket.send"]


@pytest.mark.asyncio
async def test_audio_ws_query_token_auth_rejected_by_default(monkeypatch: pytest.MonkeyPatch):
    _cookie(monkeypatch, None)
    seen = _verifier(monkeypatch)
    monkeypatch.delenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", raising=False)
    monkeypatch.setattr(streaming_service, "_AUTH_FRAME_TIMEOUT_S", 0.05)
    ws, sent = scripted_websocket(query=b"token=single-user-secret")

    assert await _authenticate(ws) == (False, None)
    assert (seen, _closed(sent)) == ([], 4401)  # the query was never read; no auth frame came


@pytest.mark.asyncio
async def test_audio_ws_query_token_auth_can_be_enabled_explicitly(monkeypatch: pytest.MonkeyPatch):
    """AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH keeps the deprecated channel for its opt-in users, through the
    shared query shim, which warns. Only ``?token=``, as before: the shim's ``?api_key=`` is not read."""
    from loguru import logger

    _cookie(monkeypatch, None)
    seen = _verifier(monkeypatch)
    monkeypatch.setenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", "1")
    ws, sent = scripted_websocket(query=b"token=single-user-secret&api_key=another-key")
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(message.record["message"]), level="WARNING")
    try:
        result = await _authenticate(ws)
    finally:
        logger.remove(sink)

    assert result == (True, 1)
    assert seen == [("Bearer single-user-secret", None)]
    assert [m for m in warned if "Deprecated WebSocket credential channel" in m] == [
        "Deprecated WebSocket credential channel query token on /api/v1/audio/stream/transcribe; send Authorization or the bearer subprotocol"
    ]
    assert sent[0]["type"] == "websocket.accept"  # left accepted for the route


@pytest.mark.asyncio
async def test_cookie_audio_auth_buffers_first_non_auth_application_frame(monkeypatch: pytest.MonkeyPatch):
    _cookie(monkeypatch, _COOKIE)
    seen = _verifier(monkeypatch)
    config_frame = json.dumps({"type": "config", "sample_rate": 16000})
    ws, _sent = scripted_websocket(config_frame)

    assert await _authenticate(ws, endpoint_id="audio.stream.transcribe") == (True, 1)
    assert await streaming_service.receive_audio_websocket_text(ws) == config_frame
    assert (seen, ws.scope["_inbox"]) == ([], [])


@pytest.mark.asyncio
async def test_valid_initial_audio_auth_takes_precedence_over_cookie(monkeypatch: pytest.MonkeyPatch):
    _cookie(monkeypatch, _COOKIE)
    seen = _verifier(monkeypatch)
    ws, _sent = scripted_websocket({"type": "auth", "token": "single-user-secret"})

    assert await _authenticate(ws) == (True, 1)
    assert ws.state.auth_principal is _SINGLE_USER
    assert seen == [("Bearer single-user-secret", None)]


@pytest.mark.asyncio
async def test_invalid_initial_audio_auth_suppresses_cookie_fallback(monkeypatch: pytest.MonkeyPatch):
    _cookie(monkeypatch, _COOKIE)
    _verifier(monkeypatch)
    ws, sent = scripted_websocket({"type": "auth", "token": "revoked"})

    assert await _authenticate(ws, endpoint_id="audio.stream.tts.realtime") == (False, None)
    assert _closed(sent) == 4401
    assert _errors(sent) == ["Invalid or expired credentials"]  # never the credential


@pytest.mark.asyncio
async def test_a_data_frame_before_auth_is_refused_4400_unprocessed(monkeypatch: pytest.MonkeyPatch):
    _cookie(monkeypatch, None)
    seen = _verifier(monkeypatch)
    ws, sent = scripted_websocket({"type": "config"}, {"type": "auth", "token": "single-user-secret"})

    assert await _authenticate(ws) == (False, None)
    assert (_closed(sent), seen, len(ws.scope["_inbox"])) == (4400, [], 1)
    assert not hasattr(ws.state, "ws_unread_frame")


@pytest.mark.asyncio
async def test_handshake_credentials_need_no_frame_and_refuse_before_accept(monkeypatch: pytest.MonkeyPatch):
    _scoped_jwt(monkeypatch, {})  # an unscoped JWT
    valid, valid_sent = scripted_websocket({"type": "config"}, headers=[(b"authorization", b"Bearer a.b.c")])
    revoked, revoked_sent = scripted_websocket(headers=[(b"authorization", b"Bearer revoked")])
    subprotocol, subprotocol_sent = scripted_websocket(headers=[(b"sec-websocket-protocol", b"bearer, a.b.c")])

    assert await _authenticate(valid) == (True, 7)
    assert await _authenticate(revoked) == (False, None)
    assert await _authenticate(subprotocol) == (True, 7)

    assert valid.scope["_inbox"] == [{"type": "websocket.receive", "text": '{"type": "config"}'}]  # unread
    assert [m["type"] for m in revoked_sent] == ["websocket.close"] and _closed(revoked_sent) == 4401
    assert subprotocol_sent[0] == {"type": "websocket.accept", "subprotocol": "bearer", "headers": []}


@pytest.mark.asyncio
async def test_a_service_token_is_no_audio_user(monkeypatch: pytest.MonkeyPatch):
    _cookie(monkeypatch, None)
    _verifier(monkeypatch, AuthPrincipal(kind="service", user_id=None, token_type="service"))
    ws, sent = scripted_websocket({"type": "auth", "token": "a.b.c"})

    assert await _authenticate(ws) == (False, None)
    assert _closed(sent) == 4401


@pytest.mark.parametrize(
    ("claims", "quota_ok", "result", "code"),
    [
        ({"allowed_endpoints": ["audio.stream.tts"], "allowed_paths": ["/api/v1/audio/"]}, True, (True, 7), None),
        ({"allowed_endpoints": ["chat.completions"]}, True, (False, None), 4403),
        ({"allowed_paths": ["/api/v1/chat/"]}, True, (False, None), 4403),
        ({"allowed_endpoints": ["chat.completions"], "role": "admin"}, True, (True, 7), None),
        ({"max_calls": 5, "jti": "j1"}, False, (False, None), 4003),
    ],
)
@pytest.mark.asyncio
async def test_a_scoped_jwt_keeps_its_audio_limits(
    monkeypatch: pytest.MonkeyPatch, claims: dict, quota_ok: bool, result: tuple, code: int | None
):
    """Authorization is unchanged by the move to the core resolver: a virtual key's allowed endpoints
    and paths and its call quota still bind (and the core lets a scoped JWT reach these checks)."""
    calls = _scoped_jwt(monkeypatch, claims, quota_ok)
    ws, sent = scripted_websocket({"type": "auth", "token": "a.b.c"})

    assert await _authenticate(ws) == result
    assert _closed(sent) == code
    if "max_calls" in claims:
        assert calls == [{"db_pool": "pool", "jti": "j1", "counter_type": "call", "limit": 5, "bucket": None}]


@pytest.mark.asyncio
async def test_the_core_admits_a_scoped_jwt_only_for_a_route_that_enforces_its_limits(monkeypatch: pytest.MonkeyPatch):
    """HTTP refuses a scoped JWT unless the route enforces its limits; audio says it does."""
    marked: list[bool] = []
    _scoped_jwt(monkeypatch, {})

    async def verify(request):
        marked.append(getattr(request.state, "_token_scope_enforced", False))
        return _JWT

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)
    bearer = [(b"authorization", b"Bearer a.b.c")]

    await websocket_principal.resolve_ws_principal(scripted_websocket(headers=bearer)[0])
    assert await _authenticate(scripted_websocket(headers=bearer)[0]) == (True, 7)
    assert marked == [False, True]


@pytest.mark.parametrize(
    ("info", "result", "code"),
    [
        ({"id": 3, "scope": "read", "llm_allowed_endpoints": '["audio.stream.tts"]'}, (True, 7), None),
        ({"id": 3, "scope": "read", "llm_allowed_endpoints": '["chat.completions"]'}, (False, None), 4403),
        ({"id": 3, "scope": "read", "metadata": '{"allowed_paths": ["/api/v1/chat/"]}'}, (False, None), 4403),
        ({"id": 3, "scope": "admin", "llm_allowed_endpoints": ["chat.completions"]}, (True, 7), None),
        ({"id": 3, "scope": "read", "metadata": {"max_runs": 0}}, (False, None), 4003),
    ],
)
@pytest.mark.asyncio
async def test_a_multi_user_api_key_keeps_its_audio_limits(monkeypatch: pytest.MonkeyPatch, info: dict, result: tuple, code: int | None):
    from tldw_Server_API.app.core.AuthNZ import api_key_manager, database, quotas

    looked_up: list[tuple] = []

    class Keys:
        async def validate_api_key(self, *, api_key, ip_address, record_usage):
            looked_up.append((api_key, record_usage))
            return info

    async def keys():
        return Keys()

    async def key_quota(**kwargs):
        return False, 1

    async def pool():
        return "pool"

    monkeypatch.setattr(api_key_manager, "get_api_key_manager", keys)
    monkeypatch.setattr(quotas, "increment_and_check_api_key_quota", key_quota)
    monkeypatch.setattr(database, "get_db_pool", pool)
    _cookie(monkeypatch, None)
    _verifier(monkeypatch, _KEY)
    ws, sent = scripted_websocket(headers=[(b"x-api-key", b"tldw_key")])

    assert await _authenticate(ws) == result
    assert _closed(sent) == code
    assert looked_up == [("tldw_key", False)]  # the verified key, with no second usage record


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_cancels_producer_when_consumer_send_fails():
    class HangingTTSService:
        def __init__(self):
            self.cancelled = asyncio.Event()

        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            try:
                yield b"first-chunk"
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class FailingWebSocket:
        async def send_bytes(self, _data: bytes):
            raise RuntimeError("client disconnected while reading stream")

        async def close(self, code=1000, reason=None):  # noqa: ARG002
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    service = HangingTTSService()

    await asyncio.wait_for(
        streaming_service._stream_tts_to_websocket(
            websocket=FailingWebSocket(),
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=service,
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        ),
        timeout=0.25,
    )

    assert service.cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_propagates_parent_cancellation_during_cleanup():
    class SlowCancelTTSService:
        def __init__(self):
            self.cleanup_started = asyncio.Event()

        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            try:
                yield b"first-chunk"
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cleanup_started.set()
                await asyncio.sleep(0.2)
                raise

    class FailingWebSocket:
        async def send_bytes(self, _data: bytes):
            raise RuntimeError("client disconnected while reading stream")

        async def close(self, code=1000, reason=None):  # noqa: ARG002
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    service = SlowCancelTTSService()
    stream_task = asyncio.create_task(
        streaming_service._stream_tts_to_websocket(
            websocket=FailingWebSocket(),
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=service,
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        )
    )

    await asyncio.wait_for(service.cleanup_started.wait(), timeout=0.5)
    stream_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stream_task, timeout=0.5)


@pytest.mark.asyncio
async def test_stream_tts_cleanup_reraises_the_cancellation_it_caught():
    """When the host's cancellation lands in the helper's cleanup, the helper re-raises that very
    exception: anyio's cancel scope recognises its own by the message, and a new bare CancelledError
    escaped the host (the TestClient's teardown error; TASK-13950.7). The wait seam returns at once with
    both tasks still running, so the cleanup itself cancels them and the host's cancel lands there."""
    producer_cancelled = asyncio.Event()

    class SlowCancelTTSService:
        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            try:
                await asyncio.Event().wait()
                yield b""
            except asyncio.CancelledError:
                producer_cancelled.set()
                await asyncio.sleep(1)  # still cleaning up when the host cancels
                raise

    async def wait_returns_nothing_done(tasks, return_when=None):  # noqa: ARG001
        await asyncio.sleep(0)
        return set(), set(tasks)

    aio = SimpleNamespace(
        Queue=asyncio.Queue,
        QueueFull=asyncio.QueueFull,
        create_task=asyncio.create_task,
        wait=wait_returns_nothing_done,
        FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
    )
    stream_task = asyncio.create_task(
        streaming_service._stream_tts_to_websocket(
            websocket=cast(Any, SimpleNamespace()),
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=SlowCancelTTSService(),
            provider="test-provider",
            outer_stream=None,
            reg=SimpleNamespace(increment=lambda *_a, **_k: None),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
            asyncio_module=aio,
        )
    )
    await asyncio.wait_for(producer_cancelled.wait(), timeout=1.0)
    stream_task.cancel("Cancelled via cancel scope host")

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(stream_task, timeout=1.0)

    assert cancelled.value.args == ("Cancelled via cancel scope host",)


def _new_cancellations(source: str) -> list[int]:
    """Lines that raise a newly made CancelledError (``raise asyncio.CancelledError`` or a call of it)."""
    import ast

    def is_cancelled_error(expr: ast.expr) -> bool:
        return (isinstance(expr, ast.Attribute) and expr.attr == "CancelledError") or (
            isinstance(expr, ast.Name) and expr.id == "CancelledError"
        )

    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Raise)
        and node.exc is not None
        and is_cancelled_error(node.exc.func if isinstance(node.exc, ast.Call) else node.exc)
    ]


def test_audio_websocket_cleanups_never_raise_a_new_bare_cancellation():
    """Pins the fix where it cannot be seen from outside too: the voice chat turn's cleanup runs in a child
    task whose runner swallows the cancellation. A cleanup re-raises the CancelledError it caught."""
    from pathlib import Path

    from tldw_Server_API.app.api.v1.endpoints.audio import audio_streaming

    seeded = "raise asyncio.CancelledError\nraise CancelledError('x')\nraise caught\nraise\n"
    assert _new_cancellations(seeded) == [1, 2]
    for module in (audio_streaming, streaming_service):
        assert _new_cancellations(Path(str(module.__file__)).read_text(encoding="utf-8")) == [], module.__name__


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_drains_queue_when_producer_finishes():
    class FastTTSService:
        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            yield b"first-chunk"
            yield b"second-chunk"

    class SlowWebSocket:
        def __init__(self):
            self.sent = []

        async def send_bytes(self, data: bytes):
            await asyncio.sleep(0.01)
            self.sent.append(data)

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    websocket = SlowWebSocket()

    await asyncio.wait_for(
        streaming_service._stream_tts_to_websocket(
            websocket=websocket,
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=FastTTSService(),
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        ),
        timeout=0.5,
    )

    assert websocket.sent == [b"first-chunk", b"second-chunk"]


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_observes_consumer_when_both_tasks_done():
    class OneChunkTTSService:
        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            yield b"first-chunk"

    class FailingWebSocket:
        async def send_bytes(self, _data: bytes):
            raise ZeroDivisionError("consumer task failed")

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    async def wait_for_both(tasks, return_when=None):  # noqa: ARG001
        await asyncio.gather(*tasks, return_exceptions=True)
        return set(tasks), set()

    fake_asyncio = SimpleNamespace(
        Queue=asyncio.Queue,
        QueueFull=asyncio.QueueFull,
        create_task=asyncio.create_task,
        wait=wait_for_both,
        FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
    )

    with pytest.raises(ZeroDivisionError, match="consumer task failed"):
        await asyncio.wait_for(
            streaming_service._stream_tts_to_websocket(
                websocket=FailingWebSocket(),
                speech_req=SimpleNamespace(model="test-model"),
                tts_service=OneChunkTTSService(),
                provider="test-provider",
                outer_stream=None,
                reg=DummyRegistry(),
                route="audio.stream.tts",
                component_label="audio_tts_ws",
                asyncio_module=fake_asyncio,
            ),
            timeout=0.5,
        )


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_cancels_consumer_when_completion_signal_fails():
    consumer_cancelled = asyncio.Event()

    class BrokenSentinelQueue:
        def __init__(self, maxsize=0):  # noqa: ARG002
            self.items = []

        def put_nowait(self, item):
            self.items.append(item)

        async def put(self, item):
            if item is None:
                raise RuntimeError("sentinel enqueue failed")
            self.items.append(item)

        async def get(self):
            if self.items:
                return self.items.pop(0)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                consumer_cancelled.set()
                raise

    class EmptyTTSService:
        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            if False:
                yield b"unreachable"

    class PassiveWebSocket:
        async def send_bytes(self, _data: bytes):
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    fake_asyncio = SimpleNamespace(
        Queue=BrokenSentinelQueue,
        QueueFull=asyncio.QueueFull,
        create_task=asyncio.create_task,
        wait=asyncio.wait,
        FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
    )

    await asyncio.wait_for(
        streaming_service._stream_tts_to_websocket(
            websocket=PassiveWebSocket(),
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=EmptyTTSService(),
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
            asyncio_module=fake_asyncio,
        ),
        timeout=0.5,
    )

    assert consumer_cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_keeps_concurrent_credential_snapshots_isolated():
    """Concurrent TTS producers dispatch and account against only their own snapshot."""

    entered = {user_id: asyncio.Event() for user_id in (101, 202)}
    release = {user_id: asyncio.Event() for user_id in (101, 202)}
    calls: list[tuple[int, dict[str, object]]] = []
    marked: list[int] = []

    class RecordingTTSService:
        async def generate_speech(self, _request, **kwargs):  # noqa: ANN001
            user_id = int(kwargs["user_id"])
            calls.append((user_id, dict(kwargs)))
            entered[user_id].set()
            await release[user_id].wait()
            yield f"audio-{user_id}".encode()

    class RecordingWebSocket:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        async def send_bytes(self, data: bytes) -> None:
            self.sent.append(data)

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return None

    async def run_one(user_id: int) -> RecordingWebSocket:
        websocket = RecordingWebSocket()

        async def mark_first_output() -> None:
            marked.append(user_id)

        await streaming_service._stream_tts_to_websocket(
            websocket=websocket,
            speech_req=SimpleNamespace(model="tts-1"),
            tts_service=RecordingTTSService(),
            provider="openai",
            provider_overrides={
                "credentials_resolved": True,
                "openai_api_key": f"user-{user_id}-key",
            },
            user_id=user_id,
            on_first_output=mark_first_output,
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        )
        return websocket

    first = asyncio.create_task(run_one(101))
    second = asyncio.create_task(run_one(202))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release[202].set()
        second_ws = await asyncio.wait_for(second, timeout=1.0)
        assert second_ws.sent == [b"audio-202"]
        assert marked == [202]

        release[101].set()
        first_ws = await asyncio.wait_for(first, timeout=1.0)
        assert first_ws.sent == [b"audio-101"]
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert sorted(
        (user_id, kwargs["provider_overrides"]["openai_api_key"])
        for user_id, kwargs in calls
    ) == [(101, "user-101-key"), (202, "user-202-key")]
    assert all(kwargs["fallback"] is False for _user_id, kwargs in calls)
    assert sorted(marked) == [101, 202]


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_closes_inner_iterator_before_scope_on_disconnect():
    """A disconnected consumer cannot release credential scope before stream close."""

    second_next_started = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    send_attempted = asyncio.Event()
    lifecycle: list[str] = []

    class SpeechIterator:
        def __init__(self) -> None:
            self._first = True

        def __aiter__(self):  # noqa: ANN204
            return self

        async def __anext__(self) -> bytes:
            if self._first:
                self._first = False
                return b"first"
            second_next_started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            close_started.set()
            await close_release.wait()
            lifecycle.append("iterator_close")

    iterator = SpeechIterator()

    class TTSService:
        def generate_speech(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return iterator

    class DisconnectingWebSocket:
        async def send_bytes(self, _data: bytes) -> None:
            send_attempted.set()
            raise RuntimeError("client disconnected")

        async def close(self, **_kwargs) -> None:  # noqa: ANN003
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return None

    async def run_stream() -> None:
        try:
            await streaming_service._stream_tts_to_websocket(
                websocket=DisconnectingWebSocket(),
                speech_req=SimpleNamespace(model="tts-1"),
                tts_service=TTSService(),
                provider="openai",
                provider_overrides={
                    "credentials_resolved": True,
                    "openai_api_key": "disconnect-key",
                },
                user_id=101,
                on_first_output=lambda: None,
                outer_stream=None,
                reg=DummyRegistry(),
                route="audio.stream.tts",
                component_label="audio_tts_ws",
            )
        finally:
            lifecycle.append("scope_close")

    task = asyncio.create_task(run_stream())
    try:
        await asyncio.wait_for(send_attempted.wait(), timeout=1.0)
        await asyncio.wait_for(second_next_started.wait(), timeout=1.0)
        close_waiter = asyncio.create_task(close_started.wait())
        done, _pending = await asyncio.wait(
            {task, close_waiter},
            timeout=1.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert close_waiter in done
        assert task not in done
        assert lifecycle == []
    finally:
        close_release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == ["iterator_close", "scope_close"]
