import time
import pytest

from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan, ws_session_or_skip


@pytest.mark.asyncio
async def test_audio_ws_pings_increment_metric(monkeypatch):
    """Audio WS should emit ping frames and increment ws_pings_total when enabled.

    We force a short STREAM_HEARTBEAT_INTERVAL_S so the generic WebSocketStream
    ping loop runs during the test. We do not send any client messages; the
    handler awaits a config frame but the ping loop runs concurrently after
    stream.start().
    """
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    # Short ping interval; leave idle disabled to avoid early close
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_S", "0.05")

    settings = get_settings()
    token = settings.SINGLE_USER_API_KEY

    reg = get_metrics_registry()
    labels = {"component": "audio", "endpoint": "audio_unified_ws", "transport": "ws"}
    before = reg.get_metric_stats("ws_pings_total", labels=labels).get("count", 0)

    with ws_client_without_lifespan(app) as client:
        try:
            ws = client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}")
        except Exception:
            pytest.skip("audio WebSocket endpoint not available in this build")

        with ws_session_or_skip(ws):
            # Allow a few ping intervals to elapse
            time.sleep(0.22)

    after = reg.get_metric_stats("ws_pings_total", labels=labels).get("count", 0)

    # Expect at least 3-4 ping attempts over ~220ms with 50ms interval
    assert after >= before + 2
