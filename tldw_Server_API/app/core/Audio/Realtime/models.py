"""Internal realtime speech models.

These models intentionally avoid route, auth, provider, and persistence imports.
OpenAI wire names should be converted at the protocol edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tldw_Server_API.app.core.Audio.Realtime.constants import (
    REALTIME_INPUT_AUDIO_FORMAT,
    REALTIME_INPUT_SAMPLE_RATE_HZ,
    REALTIME_MAX_BUFFERED_AUDIO_BYTES,
    REALTIME_MAX_JSON_FRAME_BYTES,
    REALTIME_MAX_OUTPUT_CHUNK_BYTES,
    REALTIME_OUTPUT_AUDIO_FORMAT,
    REALTIME_OUTPUT_SAMPLE_RATE_HZ,
)

ClientEventType = Literal[
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "response.create",
    "response.cancel",
]


@dataclass(frozen=True)
class RealtimeLimits:
    max_json_frame_bytes: int = REALTIME_MAX_JSON_FRAME_BYTES
    max_buffered_audio_bytes: int = REALTIME_MAX_BUFFERED_AUDIO_BYTES
    max_output_chunk_bytes: int = REALTIME_MAX_OUTPUT_CHUNK_BYTES


@dataclass(frozen=True)
class RealtimeSessionConfig:
    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    input_format: str = REALTIME_INPUT_AUDIO_FORMAT
    input_sample_rate_hz: int = REALTIME_INPUT_SAMPLE_RATE_HZ
    output_format: str = REALTIME_OUTPUT_AUDIO_FORMAT
    output_sample_rate_hz: int = REALTIME_OUTPUT_SAMPLE_RATE_HZ
    turn_detection: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateSessionCommand:
    event_id: str | None
    config: RealtimeSessionConfig


@dataclass(frozen=True)
class AppendAudioCommand:
    event_id: str | None
    audio: bytes


@dataclass(frozen=True)
class CommitAudioCommand:
    event_id: str | None


@dataclass(frozen=True)
class ClearAudioCommand:
    event_id: str | None


@dataclass(frozen=True)
class CreateResponseCommand:
    event_id: str | None
    response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CancelResponseCommand:
    event_id: str | None
    response_id: str | None = None


@dataclass(frozen=True)
class UnsupportedCommand:
    event_id: str | None
    event_type: str | None
    code: str
    message: str


ClientCommand = (
    UpdateSessionCommand
    | AppendAudioCommand
    | CommitAudioCommand
    | ClearAudioCommand
    | CreateResponseCommand
    | CancelResponseCommand
    | UnsupportedCommand
)


@dataclass(frozen=True)
class SessionCreatedEvent:
    event_id: str | None
    session_id: str
    model: str | None = None
    voice: str | None = None


@dataclass(frozen=True)
class SessionUpdatedEvent:
    event_id: str | None
    session_id: str
    model: str | None = None
    voice: str | None = None


@dataclass(frozen=True)
class InputAudioCommittedEvent:
    event_id: str | None
    item_id: str
    previous_item_id: str | None = None


@dataclass(frozen=True)
class ConversationItemAddedEvent:
    event_id: str | None
    item_id: str
    role: str
    transcript: str


@dataclass(frozen=True)
class ResponseCreatedEvent:
    event_id: str | None
    response_id: str


@dataclass(frozen=True)
class ResponseTextDeltaEvent:
    event_id: str | None
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass(frozen=True)
class ResponseAudioDeltaEvent:
    event_id: str | None
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    audio: bytes


@dataclass(frozen=True)
class ResponseTranscriptDeltaEvent:
    event_id: str | None
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass(frozen=True)
class ResponseDoneEvent:
    event_id: str | None
    response_id: str
    status: str
    status_details: dict[str, Any] | None = None
    output: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RateLimitsUpdatedEvent:
    event_id: str | None
    name: str
    limit: int
    remaining: int
    reset_seconds: int


@dataclass(frozen=True)
class RealtimeErrorEvent:
    code: str
    message: str
    event_id: str | None = None
    error_type: str = "invalid_request_error"
    server_event_id: str | None = None


RealtimeServerEvent = (
    SessionCreatedEvent
    | SessionUpdatedEvent
    | InputAudioCommittedEvent
    | ConversationItemAddedEvent
    | ResponseCreatedEvent
    | ResponseTextDeltaEvent
    | ResponseAudioDeltaEvent
    | ResponseTranscriptDeltaEvent
    | ResponseDoneEvent
    | RateLimitsUpdatedEvent
    | RealtimeErrorEvent
)
