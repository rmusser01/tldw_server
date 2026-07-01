import asyncio
from collections.abc import AsyncIterator

import pytest

from tldw_Server_API.app.core.Audio.Realtime.models import (
    AppendAudioCommand,
    CancelResponseCommand,
    CommitAudioCommand,
    CreateResponseCommand,
    InputAudioCommittedEvent,
    InputAudioSpeechStartedEvent,
    InputAudioSpeechStoppedEvent,
    RealtimeSessionConfig,
    ResponseAudioDeltaEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseTranscriptDeltaEvent,
    ResponseTranscriptDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    UpdateSessionCommand,
)
from tldw_Server_API.app.core.Audio.Realtime.pipeline import (
    RealtimePipelineAudioDelta,
    RealtimePipelineAudioDone,
    RealtimePipelineTextDelta,
    RealtimePipelineTextDone,
    RealtimePipelineTranscriptDelta,
    RealtimePipelineTranscriptDone,
    RealtimePipelineTurnDone,
)
from tldw_Server_API.app.core.Audio.Realtime.session import RealtimeSession


class FakeRealtimePipeline:
    def __init__(
        self,
        *,
        transcript: str = "user transcript",
        events: list[object] | None = None,
    ) -> None:
        self.transcript = transcript
        self.events = events or [
            RealtimePipelineTextDelta("hello "),
            RealtimePipelineTextDelta("world"),
            RealtimePipelineTextDone(),
            RealtimePipelineTranscriptDelta("assistant transcript"),
            RealtimePipelineTranscriptDone(),
            RealtimePipelineAudioDelta(b"\x01\x02"),
            RealtimePipelineAudioDone(),
            RealtimePipelineTurnDone(),
        ]
        self.transcriptions: list[dict[str, object]] = []
        self.stream_requests: list[dict[str, object]] = []

    async def transcribe_pcm16(self, audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
        self.transcriptions.append(
            {
                "audio": audio,
                "sample_rate_hz": sample_rate_hz,
                "language": language,
            }
        )
        return self.transcript

    async def stream_turn(
        self,
        transcript: str,
        *,
        config: RealtimeSessionConfig,
    ) -> AsyncIterator[object]:
        self.stream_requests.append({"transcript": transcript, "config": config})
        for event in self.events:
            await asyncio.sleep(0)
            yield event


class BlockingRealtimePipeline(FakeRealtimePipeline):
    def __init__(self) -> None:
        super().__init__(events=[])
        self.queue: asyncio.Queue[object | None] = asyncio.Queue()

    async def stream_turn(
        self,
        transcript: str,
        *,
        config: RealtimeSessionConfig,
    ) -> AsyncIterator[object]:
        self.stream_requests.append({"transcript": transcript, "config": config})
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event

    async def emit(self, event: object) -> None:
        await self.queue.put(event)

    async def finish(self) -> None:
        await self.queue.put(None)


async def _collect(events: AsyncIterator[object]) -> list[object]:
    return [event async for event in events]


@pytest.mark.asyncio
async def test_session_construction_emits_created_event_with_prefixed_session_id():
    session = RealtimeSession(pipeline=FakeRealtimePipeline())

    events = session.drain_pending_events()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SessionCreatedEvent)
    assert event.session_id == session.session_id
    assert event.session_id.startswith("sess_")
    assert session.drain_pending_events() == []


@pytest.mark.asyncio
async def test_update_session_changes_config_values():
    session = RealtimeSession(pipeline=FakeRealtimePipeline())
    config = RealtimeSessionConfig(
        model="gpt-realtime",
        voice="alloy",
        instructions="Be concise.",
        output_format="pcm16",
        metadata={"tenant": "local"},
    )

    events = await session.apply_update(UpdateSessionCommand(event_id="evt_update", config=config))

    assert events == [
        SessionUpdatedEvent(
            event_id="evt_update",
            session_id=session.session_id,
            model="gpt-realtime",
            voice="alloy",
        )
    ]
    assert session.config.model == "gpt-realtime"
    assert session.config.voice == "alloy"
    assert session.config.instructions == "Be concise."
    assert session.config.output_format == "pcm16"
    assert session.config.metadata == {"tenant": "local"}


@pytest.mark.asyncio
async def test_first_audio_append_emits_speech_started_event_once():
    session = RealtimeSession(pipeline=FakeRealtimePipeline())

    first_events = await session.append_audio(AppendAudioCommand(event_id="evt_audio_1", audio=b"\x00\x01"))
    second_events = await session.append_audio(AppendAudioCommand(event_id="evt_audio_2", audio=b"\x02\x03"))

    assert len(first_events) == 1
    assert isinstance(first_events[0], InputAudioSpeechStartedEvent)
    assert first_events[0].event_id == "evt_audio_1"
    assert first_events[0].item_id.startswith("item_")
    assert second_events == []


@pytest.mark.asyncio
async def test_manual_commit_stops_speech_commits_audio_and_starts_turn():
    pipeline = FakeRealtimePipeline(transcript="hello from user")
    session = RealtimeSession(pipeline=pipeline)
    await session.append_audio(AppendAudioCommand(event_id="evt_audio", audio=b"\x00\x01\x02\x03"))

    events = await session.commit_audio(CommitAudioCommand(event_id="evt_commit"))

    assert isinstance(events[0], InputAudioSpeechStoppedEvent)
    assert isinstance(events[1], InputAudioCommittedEvent)
    assert events[0].item_id == events[1].item_id
    assert events[1].event_id == "evt_commit"
    assert session.turn_index == 1
    assert session.input_audio_buffer == b""
    assert pipeline.transcriptions == [
        {
            "audio": b"\x00\x01\x02\x03",
            "sample_rate_hz": 16000,
            "language": None,
        }
    ]


@pytest.mark.asyncio
async def test_create_response_uses_prefixed_response_id_and_monotonic_generation_id():
    session = RealtimeSession(pipeline=FakeRealtimePipeline())
    await session.append_audio(AppendAudioCommand(event_id="evt_audio", audio=b"\x00\x01"))
    await session.commit_audio(CommitAudioCommand(event_id="evt_commit"))

    first_events = await _collect(session.create_response(CreateResponseCommand(event_id="evt_create_1")))
    first_generation_id = session.generation_id
    second_events = await _collect(session.create_response(CreateResponseCommand(event_id="evt_create_2")))

    first_created = next(event for event in first_events if isinstance(event, ResponseCreatedEvent))
    second_created = next(event for event in second_events if isinstance(event, ResponseCreatedEvent))
    assert first_created.response_id.startswith("resp_")
    assert second_created.response_id.startswith("resp_")
    assert first_generation_id == 1
    assert session.generation_id == 2
    assert second_created.response_id != first_created.response_id


@pytest.mark.asyncio
async def test_fake_pipeline_chunks_become_internal_response_delta_events():
    session = RealtimeSession(pipeline=FakeRealtimePipeline())
    await session.append_audio(AppendAudioCommand(event_id="evt_audio", audio=b"\x00\x01"))
    await session.commit_audio(CommitAudioCommand(event_id="evt_commit"))

    events = await _collect(session.create_response(CreateResponseCommand(event_id="evt_create")))

    text_deltas = [event.delta for event in events if isinstance(event, ResponseTextDeltaEvent)]
    transcript_deltas = [event.delta for event in events if isinstance(event, ResponseTranscriptDeltaEvent)]
    audio_deltas = [event.audio for event in events if isinstance(event, ResponseAudioDeltaEvent)]
    assert text_deltas == ["hello ", "world"]
    assert transcript_deltas == ["assistant transcript"]
    assert audio_deltas == [b"\x01\x02"]
    assert events[-1] == ResponseDoneEvent(
        event_id="evt_create",
        response_id=next(event.response_id for event in events if isinstance(event, ResponseCreatedEvent)),
        status="completed",
    )


@pytest.mark.asyncio
async def test_cancel_response_increments_generation_id_and_suppresses_late_chunks():
    pipeline = BlockingRealtimePipeline()
    session = RealtimeSession(pipeline=pipeline)
    await session.append_audio(AppendAudioCommand(event_id="evt_audio", audio=b"\x00\x01"))
    await session.commit_audio(CommitAudioCommand(event_id="evt_commit"))
    stream = session.create_response(CreateResponseCommand(event_id="evt_create"))

    created = await anext(stream)
    assert isinstance(created, ResponseCreatedEvent)
    assert session.generation_id == 1

    cancel_events = await session.cancel_response(
        CancelResponseCommand(event_id="evt_cancel", response_id=created.response_id)
    )
    await pipeline.emit(RealtimePipelineTextDelta("late"))
    await pipeline.finish()
    remaining_events = await _collect(stream)

    assert session.generation_id == 2
    assert cancel_events == [
        ResponseDoneEvent(
            event_id="evt_cancel",
            response_id=created.response_id,
            status="cancelled",
        )
    ]
    assert not any(isinstance(event, ResponseTextDeltaEvent) for event in remaining_events)


@pytest.mark.asyncio
async def test_cancel_after_response_created_suppresses_stale_scaffolding_output_and_done_events():
    session = RealtimeSession(pipeline=FakeRealtimePipeline())
    await session.append_audio(AppendAudioCommand(event_id="evt_audio", audio=b"\x00\x01"))
    await session.commit_audio(CommitAudioCommand(event_id="evt_commit"))
    stream = session.create_response(CreateResponseCommand(event_id="evt_create"))

    created = await anext(stream)
    cancel_events = await session.cancel_response(
        CancelResponseCommand(event_id="evt_cancel", response_id=created.response_id)
    )
    remaining_events = await _collect(stream)

    assert isinstance(created, ResponseCreatedEvent)
    assert cancel_events == [
        ResponseDoneEvent(
            event_id="evt_cancel",
            response_id=created.response_id,
            status="cancelled",
        )
    ]
    assert not any(
        isinstance(
            event,
            (
                ResponseOutputItemAddedEvent,
                ResponseContentPartAddedEvent,
                ResponseContentPartDoneEvent,
                ResponseOutputItemDoneEvent,
                ResponseTextDeltaEvent,
                ResponseTextDoneEvent,
                ResponseTranscriptDeltaEvent,
                ResponseTranscriptDoneEvent,
                ResponseAudioDeltaEvent,
            ),
        )
        for event in remaining_events
    )
    assert not any(isinstance(event, ResponseDoneEvent) and event.status == "completed" for event in remaining_events)
