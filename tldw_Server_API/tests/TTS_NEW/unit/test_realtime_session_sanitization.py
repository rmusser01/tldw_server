import asyncio
from types import SimpleNamespace

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


@pytest.mark.asyncio
async def test_buffered_realtime_session_preserves_overrides_and_closes_iterator():
    """Buffered overlap uses the captured credential snapshot for its full stream."""

    calls: list[dict[str, object]] = []
    iterator_closed = False

    class SpeechIterator:
        def __init__(self) -> None:
            self._chunks = iter((b"audio",))

        def __aiter__(self):  # noqa: ANN204
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self) -> None:
            nonlocal iterator_closed
            iterator_closed = True

    class RecordingTTSService:
        def generate_speech(self, _request, **kwargs):  # noqa: ANN001, ANN202
            calls.append(dict(kwargs))
            return SpeechIterator()

    config = realtime_session.RealtimeSessionConfig(
        model="tts-1",
        voice="alloy",
        response_format="pcm",
    )
    session = realtime_session.BufferedRealtimeSession(
        tts_service=RecordingTTSService(),
        config=config,
        provider_hint="openai",
        provider_overrides={
            "credentials_resolved": True,
            "openai_api_key": "buffered-runtime-key",
        },
        user_id=101,
    )

    await session.push_text("hello")
    await session.finish()
    chunks = [chunk async for chunk in session.audio_stream()]

    assert chunks == [b"audio"]
    assert calls == [
        {
            "provider": "openai",
            "fallback": False,
            "provider_overrides": {
                "credentials_resolved": True,
                "openai_api_key": "buffered-runtime-key",
            },
            "voice_to_voice_route": "audio.stream.tts.realtime",
            "user_id": 101,
        }
    ]
    assert iterator_closed is True


@pytest.mark.asyncio
async def test_buffered_finish_keeps_runtime_until_worker_iterator_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential scope cannot close while the buffered worker still owns its stream."""
    from tldw_Server_API.app.core.Audio import tts_service as credential_service

    next_started = asyncio.Event()
    allow_stop = asyncio.Event()
    iterator_close_started = asyncio.Event()
    iterator_close_release = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):  # noqa: ANN202
            return SimpleNamespace(
                provider=provider,
                api_key="buffered-scope-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            return None

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class SpeechIterator:
        def __aiter__(self):  # noqa: ANN204
            return self

        async def __anext__(self) -> bytes:
            next_started.set()
            await allow_stop.wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            iterator_close_started.set()
            await iterator_close_release.wait()
            lifecycle.append("iterator_close")

    class TTSService:
        def generate_speech(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return SpeechIterator()

    monkeypatch.setattr(
        credential_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        credential_service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "server-key"}},
    )
    monkeypatch.setattr(
        credential_service,
        "_capture_tts_provider_config",
        lambda _provider: {"enabled": True},
    )

    async def run_session() -> None:
        async with credential_service.tts_provider_credential_scope(
            provider="openai",
            model="tts-1",
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=SimpleNamespace(id=101),
        ) as (user_id, overrides, _runtime, _credentials):
            session = realtime_session.BufferedRealtimeSession(
                tts_service=TTSService(),
                config=realtime_session.RealtimeSessionConfig(
                    model="tts-1",
                    voice="alloy",
                    response_format="pcm",
                ),
                provider_hint="openai",
                provider_overrides=overrides,
                user_id=user_id,
            )
            await session.push_text("hello")
            await session.finish()

    task = asyncio.create_task(run_session())
    try:
        await asyncio.wait_for(next_started.wait(), timeout=1.0)
        allow_stop.set()
        await asyncio.wait_for(iterator_close_started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert task.done() is False
        assert runtime_closed.is_set() is False
    finally:
        iterator_close_release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == ["iterator_close", "runtime_close"]
