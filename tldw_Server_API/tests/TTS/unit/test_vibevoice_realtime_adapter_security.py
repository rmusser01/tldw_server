import pytest

from tldw_Server_API.app.core.Security.egress import URLPolicyResult
from tldw_Server_API.app.core.TTS.adapters import vibevoice_realtime_adapter as rt_module
from tldw_Server_API.app.core.TTS.adapters.vibevoice_realtime_adapter import (
    _VibeVoiceRealtimeWebSocketSession,
)
from tldw_Server_API.app.core.TTS.realtime_session import RealtimeSessionConfig
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderInitializationError

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_realtime_websocket_session_rejects_egress_denied_url(monkeypatch):
    calls: list[str] = []

    def fake_evaluate_url_policy(url: str, **_kwargs):
        calls.append(url)
        return URLPolicyResult(False, "blocked by test policy")

    monkeypatch.setattr(rt_module, "evaluate_url_policy", fake_evaluate_url_policy, raising=False)

    session = _VibeVoiceRealtimeWebSocketSession(
        ws_url="ws://blocked.example.test/socket",
        ws_headers=None,
        ws_timeout=1.0,
        config=RealtimeSessionConfig(
            model="vibevoice-realtime-0.5b",
            voice="alloy",
            response_format="pcm",
        ),
    )

    with pytest.raises(TTSProviderInitializationError, match="blocked by test policy"):
        await session.start()

    assert calls == ["http://blocked.example.test/socket"]
