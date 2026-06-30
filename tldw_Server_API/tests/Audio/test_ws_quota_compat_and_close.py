import json
import base64
import time
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan, ws_session_or_skip


def _receive_ws_message(ws, *, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return ws.portal.call(ws._send_rx.receive_nowait)
        except Exception as exc:
            if exc.__class__.__name__ == "WouldBlock":
                time.sleep(0.01)
                continue
            raise
    raise AssertionError("Timed out waiting for websocket frame")


def _receive_json_until_non_ping(ws, *, max_frames: int = 12) -> dict:
    for _ in range(max_frames):
        message = _receive_ws_message(ws)
        ws._raise_on_close(message)
        data = json.loads(message["text"])
        if isinstance(data, dict) and data.get("type") == "ping":
            continue
        return data
    raise AssertionError("Did not receive non-ping JSON frame within limit")


def _receive_disconnect_code(ws, *, max_frames: int = 12) -> int:
    for _ in range(max_frames):
        try:
            message = _receive_ws_message(ws)
            ws._raise_on_close(message)
            text = message.get("text")
        except WebSocketDisconnect as exc:
            return int(getattr(exc, "code", -1))
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("type") == "ping":
            continue
    raise AssertionError("Did not observe websocket disconnect within limit")


def test_audio_ws_quota_error_includes_error_type_and_closes_1008(monkeypatch):
    """Quota errors should include error_type and close with code 1008.

    Compatibility: also accepts legacy top-level 'quota' while data.quota remains the canonical field.
    """
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep

    # Force daily minutes denial
    async def _deny(user_id: int, minutes_requested: float):
        return False, 0.0

    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _deny)

    token = get_settings().SINGLE_USER_API_KEY

    with ws_client_without_lifespan(app) as client:
        try:
            ws = client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}")
        except Exception:
            pytest.skip("audio WebSocket endpoint not available in this build")
        with ws_session_or_skip(ws) as ws:
            # Minimal config and tiny audio to trigger quota path
            ws.send_text(json.dumps({"type": "config", "sample_rate": 16000}))
            audio = (np.zeros(160, dtype=np.float32)).tobytes()
            ws.send_text(json.dumps({"type": "audio", "data": base64.b64encode(audio).decode("ascii")}))

            # Expect standardized error frame with compatibility fields
            msg = _receive_json_until_non_ping(ws)
            assert isinstance(msg, dict)
            assert msg.get("type") == "error"
            # New fields
            assert msg.get("code") == "quota_exceeded"
            assert msg.get("message")
            # Compat field for rollout
            assert msg.get("error_type") == "quota_exceeded"
            # Quota is present in data and (compat) at top-level
            dq = (msg.get("data") or {}).get("quota")
            tq = msg.get("quota")
            assert (dq == "daily_minutes") or (tq == "daily_minutes")

            # Socket should then close with 1008
            assert _receive_disconnect_code(ws) == 1008


def test_audio_ws_quota_error_without_compat_alias(monkeypatch):
    """When AUDIO_WS_COMPAT_ERROR_TYPE=0, omit error_type/top-level quota alias."""
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep

    async def _deny(user_id: int, minutes_requested: float):
        return False, 0.0

    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _deny)
    monkeypatch.setenv("AUDIO_WS_COMPAT_ERROR_TYPE", "0")

    token = get_settings().SINGLE_USER_API_KEY

    with ws_client_without_lifespan(app) as client:
        try:
            ws = client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}")
        except Exception:
            pytest.skip("audio WebSocket endpoint not available in this build")
        with ws_session_or_skip(ws) as ws:
            ws.send_text(json.dumps({"type": "config", "sample_rate": 16000}))
            audio = (np.zeros(160, dtype=np.float32)).tobytes()
            ws.send_text(json.dumps({"type": "audio", "data": base64.b64encode(audio).decode("ascii")}))

            msg = _receive_json_until_non_ping(ws)
            assert isinstance(msg, dict)
            assert msg.get("type") == "error"
            assert msg.get("code") == "quota_exceeded"
            assert msg.get("error_type") is None
            assert msg.get("quota") is None
            assert (msg.get("data") or {}).get("quota") == "daily_minutes"

            assert _receive_disconnect_code(ws) in {1008, 4003}
