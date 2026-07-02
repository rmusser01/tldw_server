"""OpenAI GA realtime speech protocol adapter."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from tldw_Server_API.app.core.Audio.Realtime.constants import (
    OPENAI_REALTIME_CONVERSATION_ITEM_CREATED,
    OPENAI_REALTIME_CONVERSATION_ITEM_DONE,
    OPENAI_REALTIME_ERROR,
    OPENAI_REALTIME_INPUT_AUDIO_APPEND,
    OPENAI_REALTIME_INPUT_AUDIO_CLEAR,
    OPENAI_REALTIME_INPUT_AUDIO_COMMIT,
    OPENAI_REALTIME_INPUT_AUDIO_COMMITTED,
    OPENAI_REALTIME_INPUT_AUDIO_SPEECH_STARTED,
    OPENAI_REALTIME_INPUT_AUDIO_SPEECH_STOPPED,
    OPENAI_REALTIME_RATE_LIMITS_UPDATED,
    OPENAI_REALTIME_RESPONSE_CANCEL,
    OPENAI_REALTIME_RESPONSE_CONTENT_PART_ADDED,
    OPENAI_REALTIME_RESPONSE_CONTENT_PART_DONE,
    OPENAI_REALTIME_RESPONSE_CREATE,
    OPENAI_REALTIME_RESPONSE_CREATED,
    OPENAI_REALTIME_RESPONSE_DONE,
    OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_DELTA,
    OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_DONE,
    OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
    OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
    OPENAI_REALTIME_RESPONSE_OUTPUT_ITEM_ADDED,
    OPENAI_REALTIME_RESPONSE_OUTPUT_ITEM_DONE,
    OPENAI_REALTIME_RESPONSE_OUTPUT_TEXT_DELTA,
    OPENAI_REALTIME_RESPONSE_OUTPUT_TEXT_DONE,
    OPENAI_REALTIME_SESSION_CREATED,
    OPENAI_REALTIME_SESSION_UPDATE,
    OPENAI_REALTIME_SESSION_UPDATED,
    OPENAI_REALTIME_UNSUPPORTED_CONVERSATION_ITEM_CREATE,
    REALTIME_INPUT_AUDIO_FORMAT,
    REALTIME_INPUT_CHANNELS,
    REALTIME_INPUT_SAMPLE_RATE_HZ,
    REALTIME_INPUT_SAMPLE_WIDTH_BYTES,
    REALTIME_MAX_OUTPUT_CHUNK_BYTES,
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
    ConversationItemDoneEvent,
    CreateResponseCommand,
    InputAudioCommittedEvent,
    InputAudioSpeechStartedEvent,
    InputAudioSpeechStoppedEvent,
    RateLimitsUpdatedEvent,
    RealtimeErrorEvent,
    RealtimeLimits,
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
    if isinstance(event, InputAudioSpeechStartedEvent):
        return {
            "type": OPENAI_REALTIME_INPUT_AUDIO_SPEECH_STARTED,
            "event_id": event.event_id,
            "item_id": event.item_id,
            "audio_start_ms": event.audio_start_ms,
        }
    if isinstance(event, InputAudioSpeechStoppedEvent):
        return {
            "type": OPENAI_REALTIME_INPUT_AUDIO_SPEECH_STOPPED,
            "event_id": event.event_id,
            "item_id": event.item_id,
            "audio_end_ms": event.audio_end_ms,
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
    if isinstance(event, ConversationItemDoneEvent):
        return {
            "type": OPENAI_REALTIME_CONVERSATION_ITEM_DONE,
            "event_id": event.event_id,
            "item": {
                "id": event.item_id,
                "type": "message",
                "role": event.role,
                "status": event.status,
            },
        }
    if isinstance(event, ResponseCreatedEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_CREATED,
            "event_id": event.event_id,
            "response": _response_payload(event.response_id, "in_progress"),
        }
    if isinstance(event, ResponseOutputItemAddedEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_ITEM_ADDED,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "output_index": event.output_index,
            "item": _response_output_item_payload(event.item_id, event.role),
        }
    if isinstance(event, ResponseContentPartAddedEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_CONTENT_PART_ADDED,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "part": {"type": event.content_type},
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
    if isinstance(event, ResponseTextDoneEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_TEXT_DONE,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "text": event.text,
        }
    if isinstance(event, ResponseAudioDeltaEvent):
        if len(event.audio) > REALTIME_MAX_OUTPUT_CHUNK_BYTES:
            return to_openai_server_event(
                RealtimeErrorEvent(
                    code="payload_too_large",
                    message=(
                        "response.output_audio.delta exceeds "
                        f"{REALTIME_MAX_OUTPUT_CHUNK_BYTES} byte realtime output chunk limit"
                    ),
                    event_id=event.event_id,
                )
            )
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_DELTA,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "delta": base64.b64encode(event.audio).decode("ascii"),
        }
    if isinstance(event, ResponseAudioDoneEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_DONE,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
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
    if isinstance(event, ResponseTranscriptDoneEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "transcript": event.transcript,
        }
    if isinstance(event, ResponseContentPartDoneEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_CONTENT_PART_DONE,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "item_id": event.item_id,
            "output_index": event.output_index,
            "content_index": event.content_index,
            "part": {"type": event.content_type},
        }
    if isinstance(event, ResponseOutputItemDoneEvent):
        return {
            "type": OPENAI_REALTIME_RESPONSE_OUTPUT_ITEM_DONE,
            "event_id": event.event_id,
            "response_id": event.response_id,
            "output_index": event.output_index,
            "item": _response_output_item_payload(event.item_id, "assistant", status=event.status),
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
    if "input_audio_format" in payload:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="input_audio_format is a beta-era field; use session.audio.input.format",
            event_id=event_id,
        )
    if "output_audio_format" in payload:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="output_audio_format is a beta-era field; use session.audio.output.format",
            event_id=event_id,
        )

    session, invalid = _optional_object(
        payload,
        "session",
        "session.update session must be an object when provided",
        event_id,
    )
    if invalid is not None:
        return invalid

    unsupported = _validate_session_options(session, event_id)
    if unsupported is not None:
        return unsupported

    audio, invalid = _optional_object(
        session,
        "audio",
        "session.audio must be an object when provided",
        event_id,
    )
    if invalid is not None:
        return invalid

    input_audio, invalid = _optional_object(
        audio,
        "input",
        "session.audio.input must be an object when provided",
        event_id,
    )
    if invalid is not None:
        return invalid

    output_audio, invalid = _optional_object(
        audio,
        "output",
        "session.audio.output must be an object when provided",
        event_id,
    )
    if invalid is not None:
        return invalid

    nested_turn_detection = _validate_turn_detection(input_audio.get("turn_detection"), event_id)
    if nested_turn_detection is not None:
        return nested_turn_detection

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

    metadata, invalid = _optional_object(
        session,
        "metadata",
        "session.metadata must be an object when provided",
        event_id,
    )
    if invalid is not None:
        return invalid

    invalid_scalar = _validate_optional_str_fields(
        session,
        {
            "model": "session.model must be a string when provided",
            "voice": "session.voice must be a string when provided",
            "instructions": "session.instructions must be a string when provided",
        },
        event_id,
    )
    if invalid_scalar is not None:
        return invalid_scalar

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

    if "input_audio_format" in session:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="input_audio_format is a beta-era field; use session.audio.input.format",
            event_id=event_id,
        )

    if "output_audio_format" in session:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="output_audio_format is a beta-era field; use session.audio.output.format",
            event_id=event_id,
        )

    if "modalities" in session:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="session.modalities overrides are not supported in Stage 1",
            event_id=event_id,
        )

    turn_detection = _validate_turn_detection(session.get("turn_detection"), event_id)
    if turn_detection is not None:
        return turn_detection

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

    if len(decoded) % REALTIME_INPUT_SAMPLE_WIDTH_BYTES != 0:
        return RealtimeErrorEvent(
            code="invalid_audio",
            message="input_audio_buffer.append audio must contain whole pcm16 samples",
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

    unsupported = _validate_response_create_options(response, event_id)
    if unsupported is not None:
        return unsupported

    return CreateResponseCommand(event_id=event_id, response=dict(response))


def _validate_response_create_options(
    response: dict[str, Any],
    event_id: str | None,
) -> RealtimeErrorEvent | None:
    if "input_audio_format" in response:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="input_audio_format is a beta-era field; use session.audio.input.format",
            event_id=event_id,
        )
    if "output_audio_format" in response:
        return RealtimeErrorEvent(
            code="unsupported_session_option",
            message="output_audio_format is a beta-era field; use response.audio.output.format",
            event_id=event_id,
        )

    unsupported_overrides = ("modalities", "model", "voice", "instructions", "audio")
    for field in unsupported_overrides:
        if field in response:
            return RealtimeErrorEvent(
                code="unsupported_session_option",
                message=f"response.create {field} overrides are not supported in Stage 1; use session.update",
                event_id=event_id,
            )

    return None


def _validate_turn_detection(value: Any, event_id: str | None) -> RealtimeErrorEvent | None:
    if value in (None, "manual"):
        return None
    return RealtimeErrorEvent(
        code="unsupported_session_option",
        message="only manual turn detection is supported",
        event_id=event_id,
    )


def _optional_object(
    source: dict[str, Any],
    key: str,
    message: str,
    event_id: str | None,
) -> tuple[dict[str, Any], RealtimeErrorEvent | None]:
    if key not in source:
        return {}, None

    value = source[key]
    if not isinstance(value, dict):
        return {}, RealtimeErrorEvent(code="invalid_event", message=message, event_id=event_id)
    return value, None


def _contains_tools(value: dict[str, Any]) -> bool:
    tools = value.get("tools")
    tool_choice = value.get("tool_choice")
    return bool(tools) or tool_choice not in (None, "none")


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _validate_optional_str_fields(
    source: dict[str, Any],
    messages: dict[str, str],
    event_id: str | None,
) -> RealtimeErrorEvent | None:
    for key, message in messages.items():
        if key in source and not isinstance(source[key], str):
            return RealtimeErrorEvent(code="invalid_event", message=message, event_id=event_id)
    return None


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


def _response_output_item_payload(item_id: str, role: str, *, status: str | None = None) -> dict[str, Any]:
    item = {
        "id": item_id,
        "type": "message",
        "role": role,
    }
    if status is not None:
        item["status"] = status
    item["content"] = []
    return item
