import pytest

from tldw_Server_API.app.core.TTS import realtime_session


class _FailingTTSService:
    async def generate_speech(self, *_args, **_kwargs):
        secret_detail = "/Users/example/private/realtime-token-sk-test"
        raise RuntimeError(f"realtime backend failed at {secret_detail}")
        yield b""  # pragma: no cover - keep this as an async generator


@pytest.mark.asyncio
async def test_buffered_realtime_session_failure_log_is_sanitized():
    config = realtime_session.RealtimeSessionConfig(
        model="tts-test",
        voice="alloy",
        response_format="wav",
    )
    session = realtime_session.BufferedRealtimeSession(
        tts_service=_FailingTTSService(),
        config=config,
    )
    secret_detail = "/Users/example/private/realtime-token-sk-test"
    logged_messages: list[str] = []

    sink_id = realtime_session.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        await session.push_text("hello")
        await session.finish()
        chunks = [chunk async for chunk in session.audio_stream()]
    finally:
        realtime_session.logger.remove(sink_id)

    assert chunks == []
    assert isinstance(session.error, RuntimeError)
    assert secret_detail in str(session.error)
    assert any("Buffered realtime TTS session failed" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)
