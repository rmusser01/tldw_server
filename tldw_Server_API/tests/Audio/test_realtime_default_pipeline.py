from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio.Realtime.default_pipeline import (
    DefaultRealtimePipeline,
    RealtimePipelineError,
)
from tldw_Server_API.app.core.Audio.Realtime.models import RealtimeSessionConfig
from tldw_Server_API.app.core.Audio.Realtime.pipeline import (
    RealtimePipelineAudioDelta,
    RealtimePipelineAudioDone,
    RealtimePipelineTextDelta,
    RealtimePipelineTextDone,
    RealtimePipelineTranscriptDelta,
    RealtimePipelineTranscriptDone,
    RealtimePipelineTurnDone,
)


pytestmark = pytest.mark.asyncio


class FakeRealtimeTTSSession:
    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.pushed: list[str] = []
        self.commits = 0
        self.finished = 0
        self._chunks = chunks or [b"pcm-1", b"pcm-2"]

    async def push_text(self, delta: str) -> None:
        self.pushed.append(delta)

    async def commit(self) -> None:
        self.commits += 1

    async def finish(self) -> None:
        self.finished += 1

    async def audio_stream(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class FakeRealtimeHandle:
    def __init__(self, session: FakeRealtimeTTSSession) -> None:
        self.session = session
        self.provider = "fake"
        self.warning = None


class FakeTTSService:
    def __init__(self, session: FakeRealtimeTTSSession) -> None:
        self.session = session
        self.requests: list[OpenAISpeechRequest] = []
        self.open_kwargs: list[dict[str, Any]] = []

    async def open_realtime_session(
        self,
        *,
        request: OpenAISpeechRequest,
        provider_hint: str | None,
        route: str,
        user_id: int | None,
    ) -> FakeRealtimeHandle:
        self.requests.append(request)
        self.open_kwargs.append(
            {
                "provider_hint": provider_hint,
                "route": route,
                "user_id": user_id,
            }
        )
        return FakeRealtimeHandle(self.session)


async def _fake_stt(_audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
    assert sample_rate_hz == 16000
    assert language == "en"
    return "hello world"


async def _fake_streaming_chat_call(**kwargs: Any) -> AsyncIterator[str]:
    assert kwargs["stream"] is True
    assert kwargs["message"] == "hello world"
    assert kwargs["api_model"] == "gpt-realtime"
    assert kwargs["system_message"] == "speak plainly"

    async def _iter() -> AsyncIterator[str]:
        yield "hello "
        yield "there"

    return _iter()


def _pipeline(
    *,
    stt_transcribe_pcm16: Any = _fake_stt,
    chat_call: Any = _fake_streaming_chat_call,
    tts_service: FakeTTSService | None = None,
) -> DefaultRealtimePipeline:
    service = tts_service or FakeTTSService(FakeRealtimeTTSSession())
    return DefaultRealtimePipeline(
        stt_transcribe_pcm16=stt_transcribe_pcm16,
        chat_call=chat_call,
        tts_service_factory=lambda: service,
        default_model="gpt-realtime",
        default_voice="alloy",
        provider_hint="openai",
        user_id=42,
    )


async def test_transcribe_pcm16_uses_injected_stt_callable() -> None:
    pipeline = _pipeline()

    transcript = await pipeline.transcribe_pcm16(b"\x00\x01", sample_rate_hz=16000, language="en")

    assert transcript == "hello world"


async def test_stream_turn_streams_text_transcript_and_audio_to_tts() -> None:
    tts_session = FakeRealtimeTTSSession()
    tts_service = FakeTTSService(tts_session)
    pipeline = _pipeline(tts_service=tts_service)

    events = [
        event
        async for event in pipeline.stream_turn(
            "hello world",
            config=RealtimeSessionConfig(
                model="gpt-realtime",
                voice="verse",
                instructions="speak plainly",
                output_sample_rate_hz=24000,
                metadata={"tts_model": "tts-1"},
            ),
        )
    ]

    assert [event.delta for event in events if isinstance(event, RealtimePipelineTextDelta)] == [
        "hello ",
        "there",
    ]
    assert [event.delta for event in events if isinstance(event, RealtimePipelineTranscriptDelta)] == [
        "hello ",
        "there",
    ]
    assert [event.audio for event in events if isinstance(event, RealtimePipelineAudioDelta)] == [
        b"pcm-1",
        b"pcm-2",
    ]
    assert any(isinstance(event, RealtimePipelineTextDone) for event in events)
    assert any(isinstance(event, RealtimePipelineTranscriptDone) for event in events)
    assert any(isinstance(event, RealtimePipelineAudioDone) for event in events)
    assert isinstance(events[-1], RealtimePipelineTurnDone)

    assert tts_session.pushed == ["hello ", "there"]
    assert tts_session.commits == 1
    assert tts_session.finished == 1
    assert tts_service.requests == [
        OpenAISpeechRequest(
            model="tts-1",
            input="",
            voice="verse",
            response_format="pcm",
            stream=True,
            target_sample_rate=24000,
        )
    ]
    assert tts_service.open_kwargs == [
        {
            "provider_hint": "openai",
            "route": "audio.stream.tts.realtime",
            "user_id": 42,
        }
    ]


async def test_stream_turn_normalizes_non_streaming_chat_response() -> None:
    async def _non_streaming_chat_call(**_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "one response"}}]}

    pipeline = _pipeline(chat_call=_non_streaming_chat_call)

    events = [
        event
        async for event in pipeline.stream_turn(
            "hello world",
            config=RealtimeSessionConfig(metadata={"tts_model": "tts-1"}),
        )
    ]

    assert [event.delta for event in events if isinstance(event, RealtimePipelineTextDelta)] == ["one response"]
    assert [event.delta for event in events if isinstance(event, RealtimePipelineTranscriptDelta)] == ["one response"]


async def test_stream_turn_buffered_tts_fallback_preserves_pcm_request_options() -> None:
    class BufferedFallbackTTSService:
        def __init__(self) -> None:
            self.requests: list[OpenAISpeechRequest] = []

        async def generate_speech(self, request: OpenAISpeechRequest, **_kwargs: Any) -> AsyncIterator[bytes]:
            self.requests.append(request)
            yield b"fallback-pcm"

    tts_service = BufferedFallbackTTSService()
    pipeline = _pipeline(tts_service=tts_service)

    events = [
        event
        async for event in pipeline.stream_turn(
            "hello world",
            config=RealtimeSessionConfig(
                voice="verse",
                instructions="speak plainly",
                output_sample_rate_hz=24000,
                metadata={"tts_model": "tts-1"},
            ),
        )
    ]

    assert [event.audio for event in events if isinstance(event, RealtimePipelineAudioDelta)] == [b"fallback-pcm"]
    assert tts_service.requests == [
        OpenAISpeechRequest(
            model="tts-1",
            input="hello there",
            voice="verse",
            response_format="pcm",
            stream=True,
            target_sample_rate=24000,
        )
    ]


async def test_transcribe_pcm16_wraps_stt_errors() -> None:
    async def _failing_stt(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("stt failed")

    pipeline = _pipeline(stt_transcribe_pcm16=_failing_stt)

    with pytest.raises(RealtimePipelineError) as exc_info:
        await pipeline.transcribe_pcm16(b"\x00\x01", sample_rate_hz=16000, language=None)

    assert exc_info.value.stage == "stt"


async def test_stream_turn_wraps_llm_errors() -> None:
    async def _failing_chat(**_kwargs: Any) -> AsyncIterator[str]:
        raise RuntimeError("llm failed")

    pipeline = _pipeline(chat_call=_failing_chat)

    with pytest.raises(RealtimePipelineError) as exc_info:
        _ = [event async for event in pipeline.stream_turn("hello", config=RealtimeSessionConfig())]

    assert exc_info.value.stage == "llm"


async def test_stream_turn_wraps_tts_errors() -> None:
    class FailingTTSService(FakeTTSService):
        async def open_realtime_session(self, **_kwargs: Any) -> FakeRealtimeHandle:
            raise RuntimeError("tts failed")

    pipeline = _pipeline(tts_service=FailingTTSService(FakeRealtimeTTSSession()))

    with pytest.raises(RealtimePipelineError) as exc_info:
        _ = [event async for event in pipeline.stream_turn("hello", config=RealtimeSessionConfig())]

    assert exc_info.value.stage == "tts"
