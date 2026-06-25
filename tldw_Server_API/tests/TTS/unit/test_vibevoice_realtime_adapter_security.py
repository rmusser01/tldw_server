"""Security tests for VibeVoice realtime websocket egress handling."""

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
    """Verify realtime websocket startup rejects URLs denied by egress policy."""
    calls: list[str] = []

    def fake_evaluate_url_policy(url: str, **_kwargs):
        """Return a denied policy result while recording the normalized URL."""
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

    assert calls == ["http://blocked.example.test"]


@pytest.mark.asyncio
async def test_realtime_websocket_egress_denial_redacts_sensitive_url(monkeypatch):
    """Verify egress denial details do not expose websocket credentials."""
    def fake_evaluate_url_policy(url: str, **_kwargs):
        """Assert policy evaluation receives the sanitized origin URL."""
        assert url == "https://example.test"
        return URLPolicyResult(False, "blocked by test policy")

    monkeypatch.setattr(rt_module, "evaluate_url_policy", fake_evaluate_url_policy, raising=False)

    session = _VibeVoiceRealtimeWebSocketSession(
        ws_url="wss://user:secret-token@example.test/socket?api_key=leaked#frag",
        ws_headers=None,
        ws_timeout=1.0,
        config=RealtimeSessionConfig(
            model="vibevoice-realtime-0.5b",
            voice="alloy",
            response_format="pcm",
        ),
    )

    with pytest.raises(TTSProviderInitializationError) as exc_info:
        await session.start()

    details = exc_info.value.details
    assert details["url"] == "https://example.test"
    assert "secret-token" not in repr(details)
    assert "api_key" not in repr(details)


def test_validate_websocket_egress_returns_resolved_ips_for_pinning(monkeypatch):
    """Verify websocket egress validation returns resolved IPs for pinning."""
    calls: list[str] = []

    def fake_evaluate_url_policy(url: str, **_kwargs):
        """Return an allowed policy result with a resolved public IP."""
        calls.append(url)
        return URLPolicyResult(True, None, ("93.184.216.34",))

    monkeypatch.setattr(rt_module, "evaluate_url_policy", fake_evaluate_url_policy, raising=False)

    validation = rt_module._validate_websocket_egress("wss://public.example.test/socket")

    assert calls == ["https://public.example.test"]
    assert validation.policy_url == "https://public.example.test"
    assert validation.hostname == "public.example.test"
    assert validation.resolved_ips == ("93.184.216.34",)


@pytest.mark.asyncio
async def test_pinned_websocket_resolver_uses_validated_ips():
    """Verify the pinned resolver only resolves the validated hostname."""
    resolver = rt_module._PinnedWebSocketResolver("public.example.test", ("93.184.216.34",))

    resolved = await resolver.resolve("public.example.test", 443)
    mismatch = await resolver.resolve("other.example.test", 443)

    assert resolved == [
        {
            "hostname": "public.example.test",
            "host": "93.184.216.34",
            "port": 443,
            "family": rt_module.socket.AF_INET,
            "proto": 0,
            "flags": rt_module.socket.AI_NUMERICHOST,
        }
    ]
    assert mismatch == []
