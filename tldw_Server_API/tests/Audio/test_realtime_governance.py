"""Exercise production realtime admission, transcript policy, and credential boundaries."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Audio.Realtime import default_pipeline, handler
from tldw_Server_API.app.core.Audio.Realtime.models import RealtimeSessionConfig
from tldw_Server_API.app.core.Usage import audio_quota

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_quota_denial_prevents_pipeline_construction(monkeypatch):
    monkeypatch.setattr(handler, "authenticate_realtime_websocket", AsyncMock(return_value=(True, 7)))
    monkeypatch.setattr(audio_quota, "can_start_stream", AsyncMock(return_value=(False, "full")))
    socket = SimpleNamespace(state=SimpleNamespace(), close=AsyncMock(), accept=AsyncMock())
    called = []

    def factory():
        called.append(True)
        raise AssertionError("quota-denied connection reached providers")

    await handler.handle_realtime_websocket(socket, "native", factory, lambda: None)
    assert called == []
    socket.close.assert_awaited_once_with(code=4003)


async def test_admitted_stream_releases_slot_on_factory_failure(monkeypatch):
    monkeypatch.setattr(handler, "authenticate_realtime_websocket", AsyncMock(return_value=(True, 7)))
    monkeypatch.setattr(audio_quota, "can_start_stream", AsyncMock(return_value=(True, "OK")))
    release = AsyncMock()
    monkeypatch.setattr(audio_quota, "finish_stream", release)
    socket = SimpleNamespace(state=SimpleNamespace(), close=AsyncMock())

    def factory():
        raise RuntimeError("provider unavailable")

    await handler.handle_realtime_websocket(socket, "native", factory, lambda: None)
    release.assert_awaited_once_with(7)


async def test_daily_minutes_denial_prevents_transcription(monkeypatch):
    transcribe = AsyncMock(return_value="private transcript")
    monkeypatch.setattr(default_pipeline, "default_stt_transcribe_pcm16", transcribe)
    monkeypatch.setattr(audio_quota, "consume_daily_minutes", AsyncMock(return_value=(False, 0)))
    pipeline = default_pipeline.build_default_realtime_pipeline(user_id=7)
    with pytest.raises(default_pipeline.RealtimePipelineError):
        await pipeline.transcribe_pcm16(b"\0\0" * 16000, sample_rate_hz=16000, language=None)
    transcribe.assert_not_awaited()


async def test_transcript_policy_runs_before_text_leaves_pipeline(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy

    policy = stt_policy.STTPolicy(None, True, 0, True, False, ["email"])
    monkeypatch.setattr(stt_policy, "resolve_effective_stt_policy", AsyncMock(return_value=policy))
    monkeypatch.setattr(
        default_pipeline, "default_stt_transcribe_pcm16", AsyncMock(return_value="Email person@example.com")
    )
    monkeypatch.setattr(audio_quota, "consume_daily_minutes", AsyncMock(return_value=(True, None)))
    pipeline = default_pipeline.build_default_realtime_pipeline(user_id=7)
    transcript = await pipeline.transcribe_pcm16(b"\0\0", sample_rate_hz=16000, language=None)
    assert "person@example.com" not in transcript


async def test_admin_tts_provider_denial_prevents_dispatch(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.TTS import tts_service_v2
    from tldw_Server_API.tests.Audio.test_realtime_default_pipeline import FakeRealtimeTTSSession, FakeTTSService

    async def deny(*_args, **_kwargs):
        raise RuntimeError("provider disabled by administrator")

    monkeypatch.setattr(provider_credential_runtime.ProviderCredentialRuntime, "resolve", deny)
    monkeypatch.setattr(provider_credential_runtime.ProviderCredentialRuntime, "close", AsyncMock())
    chat = AsyncMock(return_value="must not dispatch")
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", chat)
    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", lambda: FakeTTSService(FakeRealtimeTTSSession()))
    pipeline = default_pipeline.build_default_realtime_pipeline(user_id=7)
    with pytest.raises(default_pipeline.RealtimePipelineError):
        _ = [event async for event in pipeline.stream_turn("hello", config=RealtimeSessionConfig())]
    chat.assert_not_awaited()


async def test_admitted_disconnect_releases_slot(monkeypatch):
    monkeypatch.setattr(handler, "authenticate_realtime_websocket", AsyncMock(return_value=(True, 7)))
    monkeypatch.setattr(audio_quota, "can_start_stream", AsyncMock(return_value=(True, "OK")))
    release = AsyncMock()
    monkeypatch.setattr(audio_quota, "finish_stream", release)
    socket = SimpleNamespace(
        state=SimpleNamespace(),
        close=AsyncMock(),
        accept=AsyncMock(),
        send_json=AsyncMock(),
        receive=AsyncMock(return_value={"type": "websocket.disconnect"}),
    )
    await handler.handle_realtime_websocket(socket, "native", lambda: object(), lambda: None)
    release.assert_awaited_once_with(7)


async def test_stream_lease_renews_until_cancelled(monkeypatch):
    heartbeat = AsyncMock()
    monkeypatch.setattr(audio_quota, "heartbeat_stream", heartbeat)
    monkeypatch.setattr(handler.asyncio, "sleep", AsyncMock(side_effect=[None, asyncio.CancelledError]))
    with pytest.raises(asyncio.CancelledError):
        await handler._renew_stream_lease(7)
    heartbeat.assert_awaited_once_with(7)


@pytest.mark.parametrize("deny_chat", [False, True])
async def test_production_chat_keeps_scoped_credentials_until_stream_cleanup(monkeypatch, deny_chat):
    from tldw_Server_API.app.core.Audio import tts_service
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.TTS import tts_service_v2
    from tldw_Server_API.tests.Audio.test_realtime_default_pipeline import FakeRealtimeTTSSession, FakeTTSService

    credentials = object()
    scopes = []
    runtime = SimpleNamespace(resolve=AsyncMock(return_value=credentials), mark_used=AsyncMock(), close=AsyncMock())
    if deny_chat:
        runtime.resolve.side_effect = RuntimeError("provider disabled")

    def runtime_factory(**kwargs):
        scopes.append(kwargs)
        return runtime

    @asynccontextmanager
    async def tts_scope(**_kwargs):
        yield 7, {"credentials_resolved": True}, SimpleNamespace(mark_used=AsyncMock()), object()

    monkeypatch.setattr(provider_credential_runtime, "ProviderCredentialRuntime", runtime_factory)
    monkeypatch.setattr(tts_service, "tts_provider_credential_scope", tts_scope)
    chat = AsyncMock(return_value="scoped answer")
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", chat)
    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", lambda: FakeTTSService(FakeRealtimeTTSSession()))
    pipeline = default_pipeline.build_default_realtime_pipeline(user_id=7)
    if deny_chat:
        with pytest.raises(default_pipeline.RealtimePipelineError):
            _ = [event async for event in pipeline.stream_turn("hello", config=RealtimeSessionConfig())]
        chat.assert_not_awaited()
    else:
        _ = [event async for event in pipeline.stream_turn("hello", config=RealtimeSessionConfig())]
        assert chat.await_args.kwargs[provider_credential_runtime.PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is credentials
    assert scopes[0]["user_id"] == 7
    runtime.close.assert_awaited_once()
