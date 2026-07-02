from __future__ import annotations

import base64
import math
import os
import queue
import struct
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs
from tldw_Server_API.app.api.v1.router_registry import register_router_specs
from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan


pytestmark = [pytest.mark.external_api, pytest.mark.local_llm_service]

REQUIRED_PROVIDER_ENV_VARS = (
    "TLDW_REALTIME_LIVE_SMOKE_STT_PROVIDER",
    "TLDW_REALTIME_LIVE_SMOKE_LLM_PROVIDER",
    "TLDW_REALTIME_LIVE_SMOKE_TTS_PROVIDER",
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _require_live_smoke_env() -> str:
    if not _truthy(os.getenv("TLDW_REALTIME_LIVE_SMOKE")):
        pytest.skip("Set TLDW_REALTIME_LIVE_SMOKE=1 to run the realtime live smoke test")

    missing = [name for name in REQUIRED_PROVIDER_ENV_VARS if not os.getenv(name)]
    if missing:
        pytest.skip(f"Set explicit realtime smoke provider env vars: {', '.join(missing)}")

    auth_token = os.getenv("TLDW_REALTIME_LIVE_SMOKE_AUTH_TOKEN") or os.getenv("SINGLE_USER_API_KEY")
    if not auth_token:
        pytest.skip("Set TLDW_REALTIME_LIVE_SMOKE_AUTH_TOKEN or SINGLE_USER_API_KEY for realtime live smoke auth")

    os.environ.setdefault("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    os.environ.setdefault("REALTIME_CHAT_PROVIDER_HINT", os.environ["TLDW_REALTIME_LIVE_SMOKE_LLM_PROVIDER"])
    os.environ.setdefault("REALTIME_TTS_PROVIDER_HINT", os.environ["TLDW_REALTIME_LIVE_SMOKE_TTS_PROVIDER"])
    os.environ.setdefault("REALTIME_PROVIDER_HINT", os.environ["TLDW_REALTIME_LIVE_SMOKE_LLM_PROVIDER"])
    if os.getenv("TLDW_REALTIME_LIVE_SMOKE_LLM_MODEL"):
        os.environ.setdefault("REALTIME_CHAT_MODEL", os.environ["TLDW_REALTIME_LIVE_SMOKE_LLM_MODEL"])
    if os.getenv("TLDW_REALTIME_LIVE_SMOKE_TTS_MODEL"):
        os.environ.setdefault("REALTIME_TTS_MODEL", os.environ["TLDW_REALTIME_LIVE_SMOKE_TTS_MODEL"])
    if os.getenv("TLDW_REALTIME_LIVE_SMOKE_TTS_VOICE"):
        os.environ.setdefault("REALTIME_TTS_VOICE", os.environ["TLDW_REALTIME_LIVE_SMOKE_TTS_VOICE"])

    return auth_token


def _build_app() -> FastAPI:
    app = FastAPI()
    register_router_specs(app, iter_minimal_optional_router_specs())
    return app


def _short_pcm16_test_audio(sample_rate_hz: int = 16000) -> bytes:
    silence_samples = int(sample_rate_hz * 0.1)
    tone_samples = int(sample_rate_hz * 0.35)
    samples: list[int] = [0] * silence_samples
    for index in range(tone_samples):
        value = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate_hz))
        samples.append(value)
    samples.extend([0] * silence_samples)
    return b"".join(struct.pack("<h", sample) for sample in samples)


def _receive_json_with_timeout(ws: Any, *, timeout_s: float = 10.0) -> dict[str, Any]:
    results: queue.Queue[dict[str, Any] | BaseException] = queue.Queue(maxsize=1)

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


def _wait_for_event(ws: Any, event_type: str, *, timeout_s: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    observed: list[str] = []
    while time.monotonic() < deadline:
        payload = _receive_json_with_timeout(ws, timeout_s=min(10.0, max(0.1, deadline - time.monotonic())))
        observed.append(str(payload.get("type")))
        if payload.get("type") == "error":
            raise AssertionError(f"Realtime live smoke received error event: {payload}")
        if payload.get("type") == event_type:
            return payload
    raise AssertionError(f"Timed out waiting for {event_type}; observed events: {observed}")


def test_openai_compat_realtime_live_smoke_reaches_response_done() -> None:
    auth_token = _require_live_smoke_env()
    app = _build_app()
    audio = base64.b64encode(_short_pcm16_test_audio()).decode("ascii")

    with ws_client_without_lifespan(app) as client:
        try:
            with client.websocket_connect("/v1/realtime", headers={"Authorization": f"Bearer {auth_token}"}) as ws:
                assert _receive_json_with_timeout(ws)["type"] == "session.created"
                assert _receive_json_with_timeout(ws)["type"] == "rate_limits.updated"

                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": "Answer with one short sentence.",
                            "turn_detection": None,
                        },
                    }
                )
                _wait_for_event(ws, "session.updated", timeout_s=10.0)

                ws.send_json({"type": "input_audio_buffer.append", "audio": audio})
                _wait_for_event(ws, "input_audio_buffer.speech_started", timeout_s=10.0)
                ws.send_json({"type": "input_audio_buffer.commit"})
                _wait_for_event(ws, "conversation.item.done", timeout_s=45.0)

                ws.send_json({"type": "response.create"})
                done = _wait_for_event(ws, "response.done", timeout_s=180.0)
        except WebSocketDisconnect as exc:
            raise AssertionError(f"Realtime live smoke WebSocket disconnected with code {exc.code}") from exc

    assert done["response"]["status"] == "completed"
