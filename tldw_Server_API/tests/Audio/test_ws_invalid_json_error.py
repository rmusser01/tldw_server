import json
import pytest

from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan, ws_session_or_skip


def test_audio_ws_invalid_json_yields_validation_error(monkeypatch):


    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    token = get_settings().SINGLE_USER_API_KEY

    with ws_client_without_lifespan(app) as client:
        try:
            ws = client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}")
        except Exception:
            pytest.skip("audio WebSocket endpoint not available in this build")
        with ws_session_or_skip(ws) as ws:
            # Disable VAD so this route test does not pull real torch-backed Silero
            # imports into an otherwise unrelated invalid-JSON assertion.
            ws.send_text(json.dumps({"type": "config", "sample_rate": 16000, "enable_vad": False}))
            ws.send_text("not-json")
            msg = ws.receive_json()
            assert isinstance(msg, dict)
            assert msg.get("type") == "error"
            assert msg.get("code") == "validation_error"
            # compat shim from WebSocketStream
            assert msg.get("error_type") == "validation_error"
