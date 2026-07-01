"""Internal realtime speech session orchestrator."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Any

from tldw_Server_API.app.core.Audio.Realtime.constants import REALTIME_MAX_BUFFERED_AUDIO_BYTES
from tldw_Server_API.app.core.Audio.Realtime.models import (
    AppendAudioCommand,
    CancelResponseCommand,
    ClearAudioCommand,
    ClientCommand,
    CommitAudioCommand,
    ConversationItemAddedEvent,
    ConversationItemDoneEvent,
    CreateResponseCommand,
    InputAudioCommittedEvent,
    InputAudioSpeechStartedEvent,
    InputAudioSpeechStoppedEvent,
    RealtimeErrorEvent,
    RealtimeServerEvent,
    RealtimeSessionConfig,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
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
    UnsupportedCommand,
)
from tldw_Server_API.app.core.Audio.Realtime.persistence import (
    NoopRealtimePersistenceAdapter,
    RealtimePersistenceAdapter,
    RealtimePersistenceConfig,
    persistence_config_from_metadata,
)
from tldw_Server_API.app.core.Audio.Realtime.pipeline import (
    RealtimePipeline,
    RealtimePipelineAudioDelta,
    RealtimePipelineAudioDone,
    RealtimePipelineTextDelta,
    RealtimePipelineTextDone,
    RealtimePipelineTranscriptDelta,
    RealtimePipelineTranscriptDone,
    RealtimePipelineTurnDone,
)


class RealtimeSession:
    """Owns realtime turn state without importing route or provider layers."""

    def __init__(
        self,
        *,
        pipeline: RealtimePipeline,
        persistence_adapter: RealtimePersistenceAdapter | None = None,
        config: RealtimeSessionConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.persistence_adapter = persistence_adapter or NoopRealtimePersistenceAdapter()
        self.session_id = session_id or _new_realtime_id("sess")
        self.turn_index = 0
        self.active_response_id: str | None = None
        self.generation_id = 0
        self.config = config or RealtimeSessionConfig()
        self.input_audio_buffer = b""
        self.buffer_started = False
        self.closed = False
        self.active_task: Any | None = None

        self._pending_audio_item_id: str | None = None
        self._previous_item_id: str | None = None
        self._last_user_transcript = ""
        self._pending_events: list[RealtimeServerEvent] = [self.created_event()]

    def created_event(self, event_id: str | None = None) -> SessionCreatedEvent:
        return SessionCreatedEvent(
            event_id=event_id,
            session_id=self.session_id,
            model=self.config.model,
            voice=self.config.voice,
        )

    def drain_pending_events(self) -> list[RealtimeServerEvent]:
        events = self._pending_events
        self._pending_events = []
        return events

    def set_active_task(self, task: Any | None) -> None:
        self.active_task = task

    async def handle_command(self, command: ClientCommand) -> AsyncIterator[RealtimeServerEvent]:
        if isinstance(command, UpdateSessionCommand):
            for event in await self.apply_update(command):
                yield event
            return
        if isinstance(command, AppendAudioCommand):
            for event in await self.append_audio(command):
                yield event
            return
        if isinstance(command, CommitAudioCommand):
            for event in await self.commit_audio(command):
                yield event
            return
        if isinstance(command, ClearAudioCommand):
            for event in await self.clear_audio(command):
                yield event
            return
        if isinstance(command, CreateResponseCommand):
            async for event in self.create_response(command):
                yield event
            return
        if isinstance(command, CancelResponseCommand):
            for event in await self.cancel_response(command):
                yield event
            return
        if isinstance(command, UnsupportedCommand):
            yield RealtimeErrorEvent(code=command.code, message=command.message, event_id=command.event_id)

    async def apply_update(self, command: UpdateSessionCommand) -> list[RealtimeServerEvent]:
        self.config = _merge_config(self.config, command.config)
        return [
            SessionUpdatedEvent(
                event_id=command.event_id,
                session_id=self.session_id,
                model=self.config.model,
                voice=self.config.voice,
            )
        ]

    async def append_audio(self, command: AppendAudioCommand) -> list[RealtimeServerEvent]:
        if len(self.input_audio_buffer) + len(command.audio) > REALTIME_MAX_BUFFERED_AUDIO_BYTES:
            return [
                RealtimeErrorEvent(
                    code="payload_too_large",
                    message=f"input audio buffer exceeds {REALTIME_MAX_BUFFERED_AUDIO_BYTES} bytes",
                    event_id=command.event_id,
                )
            ]

        events: list[RealtimeServerEvent] = []
        if command.audio and not self.buffer_started:
            self.buffer_started = True
            self._pending_audio_item_id = _new_realtime_id("item")
            events.append(
                InputAudioSpeechStartedEvent(
                    event_id=command.event_id,
                    item_id=self._pending_audio_item_id,
                )
            )

        self.input_audio_buffer += command.audio
        return events

    async def commit_audio(self, command: CommitAudioCommand) -> list[RealtimeServerEvent]:
        if not self.input_audio_buffer:
            return [
                RealtimeErrorEvent(
                    code="invalid_request",
                    message="input audio buffer is empty",
                    event_id=command.event_id,
                )
            ]

        audio = self.input_audio_buffer
        item_id = self._pending_audio_item_id or _new_realtime_id("item")
        self.input_audio_buffer = b""
        self.buffer_started = False
        self._pending_audio_item_id = None

        events: list[RealtimeServerEvent] = [InputAudioSpeechStoppedEvent(event_id=command.event_id, item_id=item_id)]
        try:
            transcript = await self.pipeline.transcribe_pcm16(
                audio,
                sample_rate_hz=self.config.input_sample_rate_hz,
                language=_language_from_metadata(self.config.metadata),
            )
        except Exception:
            events.append(
                RealtimeErrorEvent(
                    code="internal_error",
                    message="Realtime transcription failed",
                    event_id=command.event_id,
                    error_type="server_error",
                )
            )
            return events

        self.turn_index += 1
        self._last_user_transcript = transcript
        previous_item_id = self._previous_item_id
        self._previous_item_id = item_id
        events.extend(
            [
                InputAudioCommittedEvent(
                    event_id=command.event_id,
                    item_id=item_id,
                    previous_item_id=previous_item_id,
                ),
                ConversationItemAddedEvent(
                    event_id=command.event_id,
                    item_id=item_id,
                    role="user",
                    transcript=transcript,
                ),
                ConversationItemDoneEvent(
                    event_id=command.event_id,
                    item_id=item_id,
                    role="user",
                ),
            ]
        )
        return events

    async def clear_audio(self, command: ClearAudioCommand) -> list[RealtimeServerEvent]:
        self.input_audio_buffer = b""
        self.buffer_started = False
        self._pending_audio_item_id = None
        return []

    async def create_response(self, command: CreateResponseCommand) -> AsyncIterator[RealtimeServerEvent]:
        if self.active_response_id is not None:
            yield RealtimeErrorEvent(
                code="invalid_request",
                message="response.create is not allowed while another response is active",
                event_id=command.event_id,
            )
            return

        self.generation_id += 1
        generation_id = self.generation_id
        response_id = _new_realtime_id("resp")
        self.active_response_id = response_id
        item_id = _new_realtime_id("item")
        response_turn_index = self.turn_index
        response_user_transcript = self._last_user_transcript

        assistant_text = ""
        assistant_transcript = ""
        final_status = "completed"
        text_started = False
        transcript_started = False
        audio_started = False

        yield ResponseCreatedEvent(
            event_id=command.event_id,
            response_id=response_id,
            generation_id=generation_id,
        )
        if not self._generation_is_current(generation_id, response_id):
            return
        yield ResponseOutputItemAddedEvent(
            event_id=command.event_id,
            response_id=response_id,
            item_id=item_id,
            output_index=0,
        )
        if not self._generation_is_current(generation_id, response_id):
            return

        try:
            async for pipeline_event in self.pipeline.stream_turn(
                response_user_transcript,
                config=self.config,
            ):
                if not self._generation_is_current(generation_id, response_id):
                    return
                if isinstance(pipeline_event, RealtimePipelineTextDelta):
                    if not text_started:
                        text_started = True
                        yield _content_part_added(command.event_id, response_id, item_id, 0, "text")
                        if not self._generation_is_current(generation_id, response_id):
                            return
                    assistant_text += pipeline_event.delta
                    yield ResponseTextDeltaEvent(
                        event_id=command.event_id,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=pipeline_event.delta,
                    )
                    if not self._generation_is_current(generation_id, response_id):
                        return
                elif isinstance(pipeline_event, RealtimePipelineTranscriptDelta):
                    if not transcript_started:
                        transcript_started = True
                        yield _content_part_added(command.event_id, response_id, item_id, 1, "audio_transcript")
                        if not self._generation_is_current(generation_id, response_id):
                            return
                    assistant_transcript += pipeline_event.delta
                    yield ResponseTranscriptDeltaEvent(
                        event_id=command.event_id,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=1,
                        delta=pipeline_event.delta,
                    )
                    if not self._generation_is_current(generation_id, response_id):
                        return
                elif isinstance(pipeline_event, RealtimePipelineAudioDelta):
                    if not audio_started:
                        audio_started = True
                        yield _content_part_added(command.event_id, response_id, item_id, 2, "audio")
                        if not self._generation_is_current(generation_id, response_id):
                            return
                    yield ResponseAudioDeltaEvent(
                        event_id=command.event_id,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=2,
                        audio=pipeline_event.audio,
                    )
                    if not self._generation_is_current(generation_id, response_id):
                        return
                elif isinstance(pipeline_event, RealtimePipelineTextDone):
                    if not self._generation_is_current(generation_id, response_id):
                        return
                    yield ResponseTextDoneEvent(
                        event_id=command.event_id,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        text=assistant_text,
                    )
                    if not self._generation_is_current(generation_id, response_id):
                        return
                    yield _content_part_done(command.event_id, response_id, item_id, 0, "text")
                    if not self._generation_is_current(generation_id, response_id):
                        return
                elif isinstance(pipeline_event, RealtimePipelineTranscriptDone):
                    if not self._generation_is_current(generation_id, response_id):
                        return
                    yield ResponseTranscriptDoneEvent(
                        event_id=command.event_id,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=1,
                        transcript=assistant_transcript,
                    )
                    if not self._generation_is_current(generation_id, response_id):
                        return
                    yield _content_part_done(command.event_id, response_id, item_id, 1, "audio_transcript")
                    if not self._generation_is_current(generation_id, response_id):
                        return
                elif isinstance(pipeline_event, RealtimePipelineAudioDone):
                    if not self._generation_is_current(generation_id, response_id):
                        return
                    yield ResponseAudioDoneEvent(
                        event_id=command.event_id,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=2,
                    )
                    if not self._generation_is_current(generation_id, response_id):
                        return
                    yield _content_part_done(command.event_id, response_id, item_id, 2, "audio")
                    if not self._generation_is_current(generation_id, response_id):
                        return
                elif isinstance(pipeline_event, RealtimePipelineTurnDone):
                    final_status = pipeline_event.status
        except Exception:
            if not self._generation_is_current(generation_id, response_id):
                return
            yield RealtimeErrorEvent(
                code="internal_error",
                message="Realtime pipeline failed",
                event_id=command.event_id,
                error_type="server_error",
            )
            if not self._generation_is_current(generation_id, response_id):
                return
            yield ResponseDoneEvent(
                event_id=command.event_id,
                response_id=response_id,
                status="failed",
                status_details={"error": {"code": "internal_error", "message": "Realtime pipeline failed"}},
            )
            self._clear_active_response(response_id)
            return

        if not self._generation_is_current(generation_id, response_id):
            return

        yield ResponseOutputItemDoneEvent(
            event_id=command.event_id,
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            status=final_status,
        )
        if not self._generation_is_current(generation_id, response_id):
            return
        response_persistence_config = persistence_config_from_metadata(self.config.metadata)
        yield ResponseDoneEvent(
            event_id=command.event_id,
            response_id=response_id,
            status=final_status,
        )
        self._clear_active_response(response_id)

        if final_status == "completed":
            error = await self._persist_turn(
                command.event_id,
                assistant_text,
                turn_index=response_turn_index,
                user_transcript=response_user_transcript,
                persistence_config=response_persistence_config,
            )
            if error is not None:
                yield error

    async def cancel_response(self, command: CancelResponseCommand) -> list[RealtimeServerEvent]:
        response_id = command.response_id or self.active_response_id
        if command.response_id is not None and command.response_id != self.active_response_id:
            return [
                RealtimeErrorEvent(
                    code="invalid_request",
                    message="response.cancel response_id does not match the active response",
                    event_id=command.event_id,
                )
            ]

        if response_id is None:
            return []

        self.generation_id += 1
        if self.active_task is not None:
            self.active_task.cancel()
            self.active_task = None

        self.active_response_id = None

        return [
            ResponseDoneEvent(
                event_id=command.event_id,
                response_id=response_id,
                status="cancelled",
            )
        ]

    async def _persist_turn(
        self,
        event_id: str | None,
        assistant_text: str,
        *,
        turn_index: int,
        user_transcript: str,
        persistence_config: RealtimePersistenceConfig,
    ) -> RealtimeErrorEvent | None:
        if not persistence_config.enabled or persistence_config.conversation_id is None:
            return None

        try:
            await self.persistence_adapter.write_turn(
                conversation_id=persistence_config.conversation_id,
                session_id=self.session_id,
                turn_index=turn_index,
                user_transcript=user_transcript,
                assistant_text=assistant_text,
            )
        except Exception:
            return RealtimeErrorEvent(
                code="internal_error",
                message="Realtime persistence failed",
                event_id=event_id,
                error_type="server_error",
            )
        return None

    def _clear_active_response(self, response_id: str) -> None:
        if self.active_response_id == response_id:
            self.active_response_id = None
            self.active_task = None

    def _generation_is_current(self, generation_id: int, response_id: str) -> bool:
        if generation_id == self.generation_id:
            return True
        self._clear_active_response(response_id)
        return False


def _new_realtime_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(12)}"


def _merge_config(current: RealtimeSessionConfig, incoming: RealtimeSessionConfig) -> RealtimeSessionConfig:
    return RealtimeSessionConfig(
        model=incoming.model if incoming.model is not None else current.model,
        voice=incoming.voice if incoming.voice is not None else current.voice,
        instructions=incoming.instructions if incoming.instructions is not None else current.instructions,
        input_format=incoming.input_format,
        input_sample_rate_hz=incoming.input_sample_rate_hz,
        output_format=incoming.output_format,
        output_sample_rate_hz=incoming.output_sample_rate_hz,
        turn_detection=incoming.turn_detection,
        metadata=_merge_metadata(current.metadata, incoming.metadata),
    )


def _merge_metadata(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in incoming.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_metadata(existing, value)
        else:
            merged[key] = value
    return merged


def _language_from_metadata(metadata: dict[str, object]) -> str | None:
    language = metadata.get("language")
    return language if isinstance(language, str) else None


def _content_part_added(
    event_id: str | None,
    response_id: str,
    item_id: str,
    content_index: int,
    content_type: str,
) -> ResponseContentPartAddedEvent:
    return ResponseContentPartAddedEvent(
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=0,
        content_index=content_index,
        content_type=content_type,
    )


def _content_part_done(
    event_id: str | None,
    response_id: str,
    item_id: str,
    content_index: int,
    content_type: str,
) -> ResponseContentPartDoneEvent:
    return ResponseContentPartDoneEvent(
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=0,
        content_index=content_index,
        content_type=content_type,
    )
