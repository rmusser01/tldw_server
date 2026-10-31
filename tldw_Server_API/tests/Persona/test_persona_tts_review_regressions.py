"""Review regressions for Persona voice precedence and owned error cleanup."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from tldw_Server_API.app.core.Audio import tts_service as audio_tts
from tldw_Server_API.app.core.Persona import live_tts
from tldw_Server_API.app.core.TTS import tts_config
from tldw_Server_API.app.core.TTS.tts_config import ProviderConfig, TTSConfig
from tldw_Server_API.tests.Persona.test_persona_live_tts_providers import selected_service  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "default_provider,selected,voice,expected",
    [
        ("openai", "openai", None, "nova"),
        ("openai", "openai", "echo", "echo"),
        ("kokoro", "openai", None, ""),
        ("tldw", "kokoro", None, ""),
        (None, "openai", None, ""),
    ],
)
async def test_voice_default_belongs_only_to_its_configured_provider(
    selected_service,  # noqa: F811 - pytest injects the imported fixture.
    monkeypatch,
    default_provider,
    selected,
    voice,
    expected,
):
    config = TTSConfig(
        default_provider=default_provider,
        default_voice="nova",
        providers={"openai": ProviderConfig(enabled=True, model="gpt-4o-mini-tts")},
    )
    monkeypatch.setattr(audio_tts, "get_tts_config", lambda: config)
    monkeypatch.setattr(tts_config, "get_tts_config", lambda: config)
    # Both preparation and synthesis enter this same resolver.
    async with live_tts.resolve_persona_speech(
        text="Use the selected voice", provider=selected, model=None, voice=voice, user_id=42
    ) as (_, _, request, _):
        assert request.voice == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[b"ERROR: provider unavailable"], [b"partial-audio", b"ERROR: provider failed"]])
async def test_error_closes_speech_before_credentials_and_discards_partial_audio(monkeypatch, chunks):
    events = []
    streams = []

    class Service:
        def generate_speech(self, **kwargs):
            async def output():
                try:
                    for chunk in chunks:
                        yield chunk
                finally:
                    events.append("speech_closed")

            stream = output()
            streams.append(stream)  # Prevent GC from hiding missing deterministic close.
            return stream

    @asynccontextmanager
    async def route(**kwargs):
        try:
            yield Service(), "openai", object(), {"credentials_resolved": True}
        finally:
            events.append("credentials_closed")

    monkeypatch.setattr(live_tts, "resolve_persona_speech", route)
    try:
        with pytest.raises(RuntimeError, match="failed to generate audio"):
            await live_tts.generate_persona_speech("A reply", provider="openai", voice="nova")
        assert events == ["speech_closed", "credentials_closed"]
    finally:
        for stream in streams:
            await stream.aclose()


@pytest.mark.asyncio
async def test_cancellation_closes_speech_before_credentials(monkeypatch):
    events = []
    started = asyncio.Event()

    class Service:
        async def generate_speech(self, **kwargs):
            try:
                started.set()
                await asyncio.Event().wait()
                yield b"unreachable"
            finally:
                events.append("speech_closed")

    @asynccontextmanager
    async def route(**kwargs):
        try:
            yield Service(), "openai", object(), {"credentials_resolved": True}
        finally:
            events.append("credentials_closed")

    monkeypatch.setattr(live_tts, "resolve_persona_speech", route)
    task = asyncio.create_task(live_tts.generate_persona_speech("A reply", provider="openai", voice="nova"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["speech_closed", "credentials_closed"]
