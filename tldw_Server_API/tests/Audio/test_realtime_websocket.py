import base64
import asyncio
import importlib
import json
import queue
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.api.v1.router_registry import register_router_specs
from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs
from tldw_Server_API.app.core.Audio.Realtime.pipeline import (
    RealtimePipelineAudioDelta,
    RealtimePipelineAudioDone,
    RealtimePipelineTextDelta,
    RealtimePipelineTextDone,
    RealtimePipelineTranscriptDelta,
    RealtimePipelineTranscriptDone,
    RealtimePipelineTurnDone,
)
from tldw_Server_API.app.core import config as config_mod
from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan


pytestmark = pytest.mark.integration


class FakePipeline:
    async def transcribe_pcm16(self, audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:  # noqa: ARG002
        return "hello realtime"

    async def stream_turn(self, transcript: str, *, config):  # noqa: ANN001, ARG002
        yield RealtimePipelineTextDelta("assistant text")
        yield RealtimePipelineTranscriptDelta("assistant transcript")
        yield RealtimePipelineAudioDelta(b"\x01\x02")
        yield RealtimePipelineTextDone()
        yield RealtimePipelineTranscriptDone()
        yield RealtimePipelineAudioDone()
        yield RealtimePipelineTurnDone()


class FakePersistence:
    pass


class BlockingPipeline(FakePipeline):
    def __init__(self) -> None:
        self.stream_started = threading.Event()
        self.cancelled = threading.Event()

    async def stream_turn(self, transcript: str, *, config):  # noqa: ANN001, ARG002
        self.stream_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if False:
            yield RealtimePipelineTurnDone()


@pytest.fixture(autouse=True)
def _realtime_test_env(monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings
    from tldw_Server_API.app.core.Audio import streaming_service

    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    config_mod._route_toggle_policy.cache_clear()
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(
        auth_settings,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "SINGLE_USER_API_KEY": "single-user-secret",
                "SINGLE_USER_ALLOWED_IPS": [],
                "SINGLE_USER_FIXED_ID": 1,
            },
        )(),
    )
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")
    yield
    config_mod._route_toggle_policy.cache_clear()


def _build_app(monkeypatch: pytest.MonkeyPatch, pipeline: FakePipeline | None = None) -> FastAPI:
    audio_realtime = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.audio.audio_realtime")
    realtime_compat = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.realtime_compat")
    resolved_pipeline = pipeline or FakePipeline()
    monkeypatch.setattr(audio_realtime, "DEFAULT_REALTIME_PIPELINE_FACTORY", lambda: resolved_pipeline)
    monkeypatch.setattr(audio_realtime, "DEFAULT_REALTIME_PERSISTENCE_FACTORY", lambda: FakePersistence())
    monkeypatch.setattr(realtime_compat, "DEFAULT_REALTIME_PIPELINE_FACTORY", lambda: resolved_pipeline)
    monkeypatch.setattr(realtime_compat, "DEFAULT_REALTIME_PERSISTENCE_FACTORY", lambda: FakePersistence())

    app = FastAPI()
    register_router_specs(app, iter_minimal_optional_router_specs())
    return app


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer single-user-secret"}


def _recv_json(ws, *, timeout_s: float = 1.0) -> dict:  # noqa: ANN001
    results: queue.Queue[dict | BaseException] = queue.Queue(maxsize=1)

    def _receive() -> None:
        try:
            results.put(ws.receive_json())
        except BaseException as exc:  # noqa: BLE001
            results.put(exc)

    threading.Thread(target=_receive, daemon=True).start()
    try:
        payload = results.get(timeout=timeout_s)
    except queue.Empty as exc:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass
        raise AssertionError("Timed out waiting for realtime WebSocket event") from exc
    if isinstance(payload, BaseException):
        raise payload
    return payload


def _recv_type(ws) -> str:  # noqa: ANN001
    return _recv_json(ws)["type"]


def test_openai_compat_realtime_manual_turn_event_order(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch)
    audio = base64.b64encode(b"\x00\x01").decode("ascii")

    with ws_client_without_lifespan(app) as client:
        with client.websocket_connect("/v1/realtime", headers=_auth_headers()) as ws:
            observed = [_recv_type(ws), _recv_type(ws)]
            ws.send_json({"type": "session.update", "session": {"type": "realtime"}})
            observed.append(_recv_type(ws))
            ws.send_json({"type": "input_audio_buffer.append", "audio": audio})
            observed.append(_recv_type(ws))
            ws.send_json({"type": "input_audio_buffer.commit"})
            observed.extend(_recv_type(ws) for _ in range(4))
            ws.send_json({"type": "response.create"})
            observed.extend(_recv_type(ws) for _ in range(16))

    assert observed == [
        "session.created",
        "rate_limits.updated",
        "session.updated",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "conversation.item.done",
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.content_part.added",
        "response.output_audio_transcript.delta",
        "response.content_part.added",
        "response.output_audio.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_audio_transcript.done",
        "response.content_part.done",
        "response.output_audio.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]


def test_native_realtime_route_uses_openai_event_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch)

    with ws_client_without_lifespan(app) as client:
        with client.websocket_connect("/api/v1/audio/realtime", headers={"X-API-KEY": "single-user-secret"}) as ws:
            first = ws.receive_json()
            second = ws.receive_json()

    assert first["type"] == "session.created"
    assert "session" in first
    assert second == {"type": "rate_limits.updated", "event_id": None, "rate_limits": []}


def test_response_cancel_reaches_active_websocket_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = BlockingPipeline()
    app = _build_app(monkeypatch, pipeline=pipeline)
    audio = base64.b64encode(b"\x00\x01").decode("ascii")

    with ws_client_without_lifespan(app) as client:
        with client.websocket_connect("/v1/realtime", headers=_auth_headers()) as ws:
            ws.receive_json()
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.append", "audio": audio})
            _recv_type(ws)
            ws.send_json({"type": "input_audio_buffer.commit"})
            observed = [_recv_type(ws) for _ in range(4)]
            assert observed[-1] == "conversation.item.done"

            ws.send_json({"type": "response.create"})
            assert _recv_type(ws) == "response.created"
            assert _recv_type(ws) == "response.output_item.added"
            assert pipeline.stream_started.wait(timeout=1.0)

            ws.send_json({"type": "response.cancel"})
            done = _recv_json(ws, timeout_s=2.0)

    assert done["type"] == "response.done"
    assert done["response"]["status"] == "cancelled"
    assert pipeline.cancelled.wait(timeout=1.0)


def test_openai_compat_realtime_unauthenticated_closes_without_auth_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_app(monkeypatch)

    with ws_client_without_lifespan(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/v1/realtime"):
                pass

    assert exc_info.value.code == 4401


def test_native_realtime_capabilities_route(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch)

    with TestClient(app) as client:
        response = client.get("/api/v1/audio/realtime/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert {"path": "/api/v1/audio/realtime", "experimental": True} in payload["routes"]
    assert {"path": "/v1/realtime", "experimental": True} in payload["routes"]


def test_unsupported_conversation_item_create_returns_error_and_socket_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_app(monkeypatch)

    with ws_client_without_lifespan(app) as client:
        with client.websocket_connect("/v1/realtime", headers=_auth_headers()) as ws:
            ws.receive_json()
            ws.receive_json()
            ws.send_json({"type": "conversation.item.create", "event_id": "evt_bad"})
            error = ws.receive_json()
            ws.send_json({"type": "session.update", "session": {"type": "realtime"}})
            followup = ws.receive_json()

    assert error["type"] == "error"
    assert error["error"]["code"] == "unsupported_event"
    assert followup["type"] == "session.updated"


def test_oversized_json_frame_closes_with_1009(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Audio.Realtime.constants import REALTIME_MAX_JSON_FRAME_BYTES

    app = _build_app(monkeypatch)

    with ws_client_without_lifespan(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/v1/realtime", headers=_auth_headers()) as ws:
                ws.receive_json()
                ws.receive_json()
                ws.send_text("x" * (REALTIME_MAX_JSON_FRAME_BYTES + 1))
                ws.receive_json()

    assert exc_info.value.code == 1009


def test_binary_frame_receives_invalid_event_or_documented_close(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_app(monkeypatch)

    with ws_client_without_lifespan(app) as client:
        with client.websocket_connect("/v1/realtime", headers=_auth_headers()) as ws:
            ws.receive_json()
            ws.receive_json()
            ws.send_bytes(b"\x00\x01")
            try:
                payload = ws.receive_json()
            except WebSocketDisconnect as exc:
                assert exc.code in {1003, 1011}
            else:
                assert payload["type"] == "error"
                assert payload["error"]["code"] == "invalid_event"


def test_audio_realtime_route_toggle_removes_native_and_compat_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTES_DISABLE", "audio-realtime")
    config_mod._route_toggle_policy.cache_clear()
    app = _build_app(monkeypatch)

    paths = {getattr(route, "path", "") for route in app.routes}

    assert "/v1/realtime" not in paths
    assert "/api/v1/audio/realtime" not in paths
    assert "/api/v1/audio/realtime/capabilities" not in paths
