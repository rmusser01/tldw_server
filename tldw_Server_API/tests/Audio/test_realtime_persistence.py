from collections.abc import AsyncIterator

import pytest

from tldw_Server_API.app.core.Audio.Realtime.models import (
    AppendAudioCommand,
    CommitAudioCommand,
    CreateResponseCommand,
    RealtimeErrorEvent,
    RealtimeSessionConfig,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
    UpdateSessionCommand,
)
from tldw_Server_API.app.core.Audio.Realtime.persistence import (
    RealtimePersistenceConfig,
    persistence_config_from_metadata,
)
from tldw_Server_API.app.core.Audio.Realtime.pipeline import (
    RealtimePipelineTextDelta,
    RealtimePipelineTextDone,
    RealtimePipelineTurnDone,
)
from tldw_Server_API.app.core.Audio.Realtime.session import RealtimeSession


class FakeRealtimePipeline:
    def __init__(self, *, transcript: str = "persisted user") -> None:
        self.transcript = transcript

    async def transcribe_pcm16(self, audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
        return self.transcript

    async def stream_turn(
        self,
        transcript: str,
        *,
        config: RealtimeSessionConfig,
    ) -> AsyncIterator[object]:
        yield RealtimePipelineTextDelta("persisted ")
        yield RealtimePipelineTextDelta("assistant")
        yield RealtimePipelineTextDone()
        yield RealtimePipelineTurnDone()


class FakeRealtimePersistenceAdapter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.writes: list[dict[str, object]] = []

    async def write_turn(
        self,
        *,
        conversation_id: str,
        session_id: str,
        turn_index: int,
        user_transcript: str,
        assistant_text: str,
    ) -> None:
        if self.fail:
            raise RuntimeError("persistence unavailable")
        self.writes.append(
            {
                "conversation_id": conversation_id,
                "session_id": session_id,
                "turn_index": turn_index,
                "user_transcript": user_transcript,
                "assistant_text": assistant_text,
            }
        )


async def _run_turn(session: RealtimeSession) -> list[object]:
    await session.append_audio(AppendAudioCommand(event_id="evt_audio", audio=b"\x00\x01"))
    await session.commit_audio(CommitAudioCommand(event_id="evt_commit"))
    return [event async for event in session.create_response(CreateResponseCommand(event_id="evt_create"))]


def test_persistence_config_from_metadata_requires_tldw_persist_true():
    assert persistence_config_from_metadata({}) == RealtimePersistenceConfig(enabled=False, conversation_id=None)
    assert persistence_config_from_metadata(
        {"tldw": {"persist": False, "conversation_id": "abc"}}
    ) == RealtimePersistenceConfig(
        enabled=False,
        conversation_id="abc",
    )
    assert persistence_config_from_metadata(
        {"tldw": {"persist": True, "conversation_id": "abc"}}
    ) == RealtimePersistenceConfig(
        enabled=True,
        conversation_id="abc",
    )


def test_persistence_config_rejects_raw_audio_storage_for_stage_two():
    config = persistence_config_from_metadata(
        {
            "tldw": {
                "persist": True,
                "conversation_id": "abc",
                "store_raw_audio": True,
            }
        }
    )

    assert config == RealtimePersistenceConfig(enabled=True, conversation_id="abc", store_raw_audio=False)


@pytest.mark.asyncio
async def test_ephemeral_session_does_not_call_persistence_adapter():
    adapter = FakeRealtimePersistenceAdapter()
    session = RealtimeSession(
        pipeline=FakeRealtimePipeline(),
        persistence_adapter=adapter,
    )

    events = await _run_turn(session)

    assert any(isinstance(event, ResponseDoneEvent) and event.status == "completed" for event in events)
    assert adapter.writes == []


@pytest.mark.asyncio
async def test_persist_metadata_writes_user_transcript_and_assistant_text_after_done():
    adapter = FakeRealtimePersistenceAdapter()
    session = RealtimeSession(
        pipeline=FakeRealtimePipeline(transcript="persisted user"),
        persistence_adapter=adapter,
    )
    await session.apply_update(
        UpdateSessionCommand(
            event_id="evt_update",
            config=RealtimeSessionConfig(metadata={"tldw": {"persist": True, "conversation_id": "abc"}}),
        )
    )

    events = await _run_turn(session)

    done_index = next(index for index, event in enumerate(events) if isinstance(event, ResponseDoneEvent))
    assert done_index < len(events)
    assert adapter.writes == [
        {
            "conversation_id": "abc",
            "session_id": session.session_id,
            "turn_index": 1,
            "user_transcript": "persisted user",
            "assistant_text": "persisted assistant",
        }
    ]


@pytest.mark.asyncio
async def test_persistence_failure_emits_internal_error_after_streamed_output_and_done():
    adapter = FakeRealtimePersistenceAdapter(fail=True)
    session = RealtimeSession(
        pipeline=FakeRealtimePipeline(transcript="persisted user"),
        persistence_adapter=adapter,
    )
    await session.apply_update(
        UpdateSessionCommand(
            event_id="evt_update",
            config=RealtimeSessionConfig(metadata={"tldw": {"persist": True, "conversation_id": "abc"}}),
        )
    )

    events = await _run_turn(session)

    text_delta_index = next(index for index, event in enumerate(events) if isinstance(event, ResponseTextDeltaEvent))
    done_index = next(index for index, event in enumerate(events) if isinstance(event, ResponseDoneEvent))
    error_index = next(index for index, event in enumerate(events) if isinstance(event, RealtimeErrorEvent))
    error = events[error_index]
    assert text_delta_index < done_index < error_index
    assert isinstance(error, RealtimeErrorEvent)
    assert error.code == "internal_error"
