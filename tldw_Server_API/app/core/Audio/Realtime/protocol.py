"""OpenAI GA realtime speech protocol adapter."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from tldw_Server_API.app.core.Audio.Realtime.constants import (
    OPENAI_REALTIME_CONVERSATION_ITEM_CREATED,
    OPENAI_REALTIME_ERROR,
    OPENAI_REALTIME_INPUT_AUDIO_APPEND,
    OPENAI_REALTIME_INPUT_AUDIO_CLEAR,
    OPENAI_REALTIME_INPUT_AUDIO_COMMIT,
    OPENAI_REALTIME_INPUT_AUDIO_COMMITTED,
    OPENAI_REALTIME_RATE_LIMITS_UPDATED,
    OPENAI_REALTIME_RESPONSE_CANCEL,
    OPENAI_REALTIME_RESPONSE_CREATE,
    OPENAI_REALTIME_RESPONSE_CREATED,
    OPENAI_REALTIME_RESPONSE_DONE,
    OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_DELTA,
    OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
    OPENAI_REALTIME_RESPONSE_OUTPUT_TEXT_DELTA,
    OPENAI_REALTIME_SESSION_CREATED,
    OPENAI_REALTIME_SESSION_UPDATE,
    OPENAI_REALTIME_SESSION_UPDATED,
    OPENAI_REALTIME_UNSUPPORTED_CONVERSATION_ITEM_CREATE,
    REALTIME_INPUT_AUDIO_FORMAT,
    REALTIME_INPUT_CHANNELS,
    REALTIME_INPUT_SAMPLE_RATE_HZ,
    REALTIME_OUTPUT_AUDIO_FORMAT,
    REALTIME_OUTPUT_CHANNELS,
    REALTIME_OUTPUT_SAMPLE_RATE_HZ,
)
from tldw_Server_API.app.core.Audio.Realtime.models import (
    AppendAudioCommand,
    CancelResponseCommand,
    ClearAudioCommand,
    ClientCommand,
    CommitAudioCommand,
    ConversationItemAddedEvent,
    CreateResponseCommand,
    InputAudioCommittedEvent,
    RateLimitsUpdatedEvent,
    RealtimeErrorEvent,
    RealtimeLimits,
    RealtimeServerEvent,
    RealtimeSessionConfig,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTranscriptDeltaEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    UpdateSessionCommand,
)


def parse_client_event(payload: dict[str, Any], limits: RealtimeLimits) -> ClientCommand | RealtimeErrorEvent:
    """Parse a supported OpenAI GA JSON event into an internal command."""

    event_type = payload.get("type")
    event_id = _optional_str(payload.get("event_id"))

    if event_type == OPENAI_REALTIME_SESSION_UPDATE:
        return _parse_session_update(payload, event_id)
    if event_type == OPENAI_REALTIME_INPUT_AUDIO_APPEND:
        return _parse_append_audio(payload, event_id, limits)
    if event_type == OPENAI_REALTIME_INPUT_AUDIO_COMMIT:
        return CommitAudioCommand(event_id=event_id)
    if event_type == OPENAI_REALTIME_INPUT_AUDIO_CLEAR:
        return ClearAudioCommand(event_id=event_id)
    if event_type == OPENAI_REALTIME_RESPONSE_CREATE:
        return _parse_response_create(payload, event_id)
    if event_type == OPENAI_REALTIME_RESPONSE_CANCEL:
        response_id = _optional_str(payload.get("response_id"))
        return CancelResponseCommand(event_id=event_id, response_id=response_id)
    if event_type == OPENAI_REALTIME_UNSUPPORTED_CONVERSATION_ITEM_CREATE:
        return RealtimeErrorEvent(
            code="unsupported_event",
            message="conversation.item.create is not supported by the Stage 1 realtime endpoint",
            event_id=event_id,
        )

    return RealtimeErrorEvent(
        code="unsupported_event",
        message=f"{event_type!r} is not supported by the Stage 1 realtime endpoint",
        event_id=event_id,
    )


def to_openai_server_event(event: RealtimeServerEvent) -> dict[str, Any]:
    """Serialize an internal realtime server event into an OpenAI-shaped dict."""

    if isinstance(event, SessionCreatedEvent):
        return {
            "type": OPENAI_REALTIME_SESSION_CREATED,
            "event_id": event.event_id,
            "session": _session_payload(event.session_id, event.model, event.voice),
        }
    if isinstance(event, SessionUpdatedEvent):
        return {
            "type": OPENAI_REALTIME_SESSION_UPDATED,
            "event_id": event.event_id,
            "session": _session_payload(event.session_id, event.model, event.voice),
        }
    if isinstance(event, InputAudioCommittedEvent):
        return {
            "type": OPENAI_REALTIME_INPUT_AUDIO_COMMITTED,
            "event_id": event.event_id,
            "item_id": event.item_id,
            "previous_item_id": event.previous_item_id,
        }
    if isinstance(event, ConversationItemAddedEvent):
        return {
            "type": OPENAI_REALTIME_CONVERSATION_ITEM_CREATED,
            "event_id": event.event_id,
            "item": {
                "id": event.item_id,
                "type": "message",
                "role": event.role,
                "content": [
                    {
                        "type": "input_audio",
                        "transcript": event.transcript,
                    }
                ],
            },
        }
    if isinstance(event, ResponseCreatedEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_CREATED,
            "event_id": event.event_id,
            "response": _response_payload(event.response_id, "in_progress"),
        }
    if isinstance(event, ResponseTextDeltaEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_TEXT_DELTA,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "delta": event.delta,
        }
    if isinstance(event, ResponseAudioDeltaEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_DELTA,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "delta": base64.b64encode(event.audio).decode("ascii"),
        }
    if isinstance(event, ResponseTranscriptDeltaEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "delta": event.delta,
        }
    if isinstance(event, ResponseDoneEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_DONE,
            "event_id": event.event_id,
            "response": _response_payload(
                event.response_id,
                event.status,
                status_details=event.status_details,
                output=event.output,
            ),
        }
    if isinstance(event, RateLimitsUpdatedEvent):
        return {
            "type": OPENAI_REALTIME_RATE_LIMITS_UPDATED,
            "event_id": event.event_id,
            "rate_limits": [
                {
                    "name": event.name,
                    "limit": event.limit,
                    "remaining": event.remaining,
                    "reset_seconds": event.reset_seconds,
                }
            ],
        }
    if isinstance(event, RealtimeErrorEvent):
        return {
            "type": OPENAI_REALTIME_ERROR,
            "event_id": event.server_event_id,
            "error": {
                "type": event.error_type,
                "code": event.code,
                "message": event.message,
                "event_id": event.event_id,
            },
        }

    raise TypeError(f"Unsupported realtime server event: {type(event)!r}")


def _parse_session_update(payload: dict[str, Any], event_id: str | None) -> UpdateSessionCommand | RealtimeErrorEvent:
    if "output_audio_format" in payload:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="output_audio_format is a beta-era field; use session.audio.output.format",
            event_id=event_id,
        )

    session = payload.get("session")
    if not isinstance(session, dict):
        session = {}

    unsupported = _validate_session_options(session, event_id)
    if unsupported is not None:
        return unsupported

    audio = session.get("audio")
    if not isinstance(audio, dict):
        audio = {}

    input_audio = audio.get("input")
    if not isinstance(input_audio, dict):
        input_audio = {}

    output_audio = audio.get("output")
    if not isinstance(output_audio, dict):
        output_audio = {}

    input_format = input_audio.get("format", REALTIME_INPUT_AUDIO_FORMAT)
    input_sample_rate_hz = input_audio.get("sample_rate_hz", REALTIME_INPUT_SAMPLE_RATE_HZ)
    input_channels = input_audio.get("channels", REALTIME_INPUT_CHANNELS)
    output_format = output_audio.get("format", REALTIME_OUTPUT_AUDIO_FORMAT)
    output_sample_rate_hz = output_audio.get("sample_rate_hz", REALTIME_OUTPUT_SAMPLE_RATE_HZ)
    output_channels = output_audio.get("channels", REALTIME_OUTPUT_CHANNELS)

    if (
        input_format != REALTIME_INPUT_AUDIO_FORMAT
        or input_sample_rate_hz != REALTIME_INPUT_SAMPLE_RATE_HZ
        or input_channels != REALTIME_INPUT_CHANNELS
    ):
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="only pcm16 16000 Hz mono input audio is supported",
            event_id=event_id,
        )
    if (
        output_format != REALTIME_OUTPUT_AUDIO_FORMAT
        or output_sample_rate_hz != REALTIME_OUTPUT_SAMPLE_RATE_HZ
        or output_channels != REALTIME_OUTPUT_CHANNELS
    ):
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="only pcm16 24000 Hz mono output audio is supported",
            event_id=event_id,
        )

    metadata = session.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    config = RealtimeSessionConfig(
        model=_optional_str(session.get("model")),
        voice=_optional_str(session.get("voice")),
        instructions=_optional_str(session.get("instructions")),
        input_format=REALTIME_INPUT_AUDIO_FORMAT,
        input_sample_rate_hz=REALTIME_INPUT_SAMPLE_RATE_HZ,
        output_format=REALTIME_OUTPUT_AUDIO_FORMAT,
        output_sample_rate_hz=REALTIME_OUTPUT_SAMPLE_RATE_HZ,
        turn_detection="manual",
        metadata=dict(metadata),
    )
    return UpdateSessionCommand(event_id=event_id, config=config)


def _validate_session_options(session: dict[str, Any], event_id: str | None) -> RealtimeErrorEvent | None:
    session_type = session.get("type")
    if session_type is not None and session_type != "realtime":
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="session.type must be 'realtime' when provided",
            event_id=event_id,
        )

    if "output_audio_format" in session:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="output_audio_format is a beta-era field; use session.audio.output.format",
            event_id=event_id,
        )

    turn_detection = session.get("turn_detection")
    if turn_detection not in (None, "manual"):
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="only manual turn detection is supported",
            event_id=event_id,
        )

    if _contains_tools(session):
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="tool calls are not supported by the Stage 1 realtime endpoint",
            event_id=event_id,
        )

    return None


def _parse_append_audio(
    payload: dict[str, Any],
    event_id: str | None,
    limits: RealtimeLimits,
) -> AppendAudioCommand | RealtimeErrorEvent:
    audio = payload.get("audio")
    if not isinstance(audio, str):
        return RealtimeErrorEvent(
            code="invalid_audio",
            message="input_audio_buffer.append audio must be base64-encoded pcm16",
            event_id=event_id,
        )

    try:
        decoded = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError):
        return RealtimeErrorEvent(
            code="invalid_audio",
            message="input_audio_buffer.append audio must be base64-encoded pcm16",
            event_id=event_id,
        )

    if len(decoded) > limits.max_buffered_audio_bytes:
        return RealtimeErrorEvent(
            code="payload_too_large",
            message=f"decoded audio exceeds {limits.max_buffered_audio_bytes} byte realtime input buffer limit",
            event_id=event_id,
        )

    return AppendAudioCommand(event_id=event_id, audio=decoded)


def _parse_response_create(payload: dict[str, Any], event_id: str | None) -> CreateResponseCommand | RealtimeErrorEvent:
    response = payload.get("response")
    if response is None:
        response = {}
    if not isinstance(response, dict):
        return RealtimeErrorEvent(
            code="invalid_request",
            message="response.create response must be an object when provided",
            event_id=event_id,
        )

    if _contains_tools(response):
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="tool calls are not supported by the Stage 1 realtime endpoint",
            event_id=event_id,
        )

    return CreateResponseCommand(event_id=event_id, response=dict(response))


def _contains_tools(value: dict[str, Any]) -> bool:
    tools = value.get("tools")
    tool_choice = value.get("tool_choice")
    return bool(tools) or tool_choice not in (None, "none")


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _session_payload(session_id: str, model: str | None, voice: str | None) -> dict[str, Any]:
    return {
        "id": session_id,
        "type": "realtime",
        "model": model,
        "voice": voice,
        "audio": {
            "input": {
                "format": REALTIME_INPUT_AUDIO_FORMAT,
                "sample_rate_hz": REALTIME_INPUT_SAMPLE_RATE_HZ,
                "channels": REALTIME_INPUT_CHANNELS,
            },
            "output": {
                "format": REALTIME_OUTPUT_AUDIO_FORMAT,
                "sample_rate_hz": REALTIME_OUTPUT_SAMPLE_RATE_HZ,
                "channels": REALTIME_OUTPUT_CHANNELS,
            },
        },
        "turn_detection": None,
    }


def _response_payload(
    response_id: str,
    status: str,
    *,
    status_details: dict[str, Any] | None = None,
    output: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "realtime.response",
        "status": status,
        "status_details": status_details,
        "output": output if output is not None else [],
    }
