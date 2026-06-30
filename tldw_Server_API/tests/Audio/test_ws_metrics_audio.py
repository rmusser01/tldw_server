import json

import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan, ws_session_or_skip


@pytest.mark.asyncio
async def test_audio_ws_v2_config_emits_configured_status(monkeypatch: pytest.MonkeyPatch):
    """Negotiated v2 config should emit the configured status frame."""
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    import tldw_Server_API.app.core.config as app_config

    original_get_stt_config = app_config.get_stt_config

    def _enabled_stt_config():
        cfg = dict(original_get_stt_config() or {})
        cfg["ws_control_v2_enabled"] = True
        return cfg

    monkeypatch.setattr(app_config, "get_stt_config", _enabled_stt_config)

    settings = get_settings()
    token = settings.SINGLE_USER_API_KEY

    with ws_client_without_lifespan(app) as client:
        try:
            ws = client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}")
        except Exception:
            pytest.skip("audio WebSocket endpoint not available in this build")

        with ws_session_or_skip(ws) as ws:
            ws.send_text(
                json.dumps(
                    {"type": "config", "sample_rate": 16000, "enable_vad": False, "protocol_version": 2}
                )
            )
            data = ws.receive_json()
            assert data.get("type") == "status"
            assert data.get("state") == "configured"
            assert data.get("protocol_version") == 2


@pytest.mark.asyncio
async def test_audio_ws_emits_bounded_stt_session_metrics_on_commit():
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    metrics_manager._metrics_registry = None
    reg = get_metrics_registry()
    token = get_settings().SINGLE_USER_API_KEY

    try:
        before_started = reg.get_cumulative_counter_total("audio_stt_streaming_sessions_started_total")
        before_ended = reg.get_cumulative_counter_total("audio_stt_streaming_sessions_ended_total")
        before_requests = reg.get_cumulative_counter_total("audio_stt_requests_total")

        with ws_client_without_lifespan(app) as client:
            try:
                ws = client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}")
            except Exception:
                pytest.skip("audio WebSocket endpoint not available in this build")

            with ws_session_or_skip(ws) as ws:
                ws.send_text(json.dumps({"type": "config", "sample_rate": 16000, "enable_vad": False}))
                ws.send_text(json.dumps({"type": "commit"}))
                try:
                    _ = ws.receive_json()
                except Exception:
                    _ = None
                ws.send_text(json.dumps({"type": "stop"}))
                try:
                    _ = ws.receive_json()
                except Exception:
                    _ = None

        assert reg.get_cumulative_counter_total("audio_stt_streaming_sessions_started_total") == before_started + 1
        assert reg.get_cumulative_counter_total("audio_stt_streaming_sessions_ended_total") == before_ended + 1
        assert reg.get_cumulative_counter_total("audio_stt_requests_total") == before_requests + 1
    finally:
        metrics_manager._metrics_registry = None
