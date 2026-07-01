import base64

import pytest

from tldw_Server_API.app.core.Audio.Realtime.constants import (
    REALTIME_MAX_BUFFERED_AUDIO_BYTES,
    REALTIME_MAX_OUTPUT_CHUNK_BYTES,
)
from tldw_Server_API.app.core.Audio.Realtime.models import (
    AppendAudioCommand,
    CancelResponseCommand,
    ClearAudioCommand,
    CommitAudioCommand,
    ConversationItemAddedEvent,
    CreateResponseCommand,
    InputAudioCommittedEvent,
    RateLimitsUpdatedEvent,
    RealtimeErrorEvent,
    RealtimeLimits,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTranscriptDeltaEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    UpdateSessionCommand,
)
from tldw_Server_API.app.core.Audio.Realtime.protocol import (
    parse_client_event,
    to_openai_server_event,
)


def _parse(payload: dict) -> object:
    return parse_client_event(payload, RealtimeLimits())


def test_session_update_realtime_parses_to_internal_command():
    command = _parse(
        {
            "type": "session.update",
            "event_id": "evt_client_1",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "voice": "alloy",
                "instructions": "Be concise.",
                "audio": {
                    "input": {"format": "pcm16", "sample_rate_hz": 16000},
                    "output": {"format": "pcm16", "sample_rate_hz": 24000},
                },
                "turn_detection": None,
                "metadata": {"tenant": "local"},
            },
        }
    )

    assert isinstance(command, UpdateSessionCommand)
    assert command.event_id == "evt_client_1"
    assert command.config.model == "gpt-realtime"
    assert command.config.voice == "alloy"
    assert command.config.instructions == "Be concise."
    assert command.config.input_format == "pcm16"
    assert command.config.input_sample_rate_hz == 16000
    assert command.config.output_format == "pcm16"
    assert command.config.output_sample_rate_hz == 24000
    assert command.config.turn_detection == "manual"
    assert command.config.metadata == {"tenant": "local"}


def test_session_update_accepts_supported_output_audio_shape():
    command = _parse(
        {
            "type": "session.update",
            "session": {
                "audio": {
                    "output": {
                        "format": "pcm16",
                        "sample_rate_hz": 24000,
                    }
                }
            },
        }
    )

    assert isinstance(command, UpdateSessionCommand)
    assert command.config.output_format == "pcm16"
    assert command.config.output_sample_rate_hz == 24000


@pytest.mark.parametrize(
    ("session", "expected_message"),
    [
        (
            {"audio": {"input": {"channels": 2}}},
            "only pcm16 16000 Hz mono input audio is supported",
        ),
        (
            {"audio": {"output": {"channels": 2}}},
            "only pcm16 24000 Hz mono output audio is supported",
        ),
    ],
)
def test_session_update_rejects_non_mono_audio_contract(session, expected_message):
    result = _parse(
        {
            "type": "session.update",
            "event_id": "evt_audio_contract",
            "session": session,
        }
    )

    assert result == RealtimeErrorEvent(
        code="unsupported_session_option",
        message=expected_message,
        event_id="evt_audio_contract",
    )


def test_session_update_rejects_nested_server_vad_turn_detection():
    result = _parse(
        {
            "type": "session.update",
            "event_id": "evt_nested_vad",
            "session": {
                "audio": {
                    "input": {
                        "turn_detection": {"type": "server_vad"},
                    }
                }
            },
        }
    )

    assert result == RealtimeErrorEvent(
        code="unsupported_session_option",
        message="only manual turn detection is supported",
        event_id="evt_nested_vad",
    )


def test_session_update_accepts_missing_type_but_rejects_transcription_type():
    omitted_type = _parse({"type": "session.update", "session": {}})
    transcription = _parse(
        {
            "type": "session.update",
            "event_id": "evt_bad_session",
            "session": {"type": "transcription"},
        }
    )

    assert isinstance(omitted_type, UpdateSessionCommand)
    assert transcription == RealtimeErrorEvent(
        code="unsupported_session_option",
        message="session.type must be 'realtime' when provided",
        event_id="evt_bad_session",
    )


def test_session_update_rejects_beta_top_level_output_audio_format():
    result = _parse(
        {
            "type": "session.update",
            "event_id": "evt_beta",
            "output_audio_format": "pcm16",
            "session": {},
        }
    )

    assert result == RealtimeErrorEvent(
        code="unsupported_session_option",
        message="output_audio_format is a beta-era field; use session.audio.output.format",
        event_id="evt_beta",
    )


def test_append_audio_decodes_base64_pcm_bytes():
    pcm = b"\x01\x00\x02\x00"
    result = _parse(
        {
            "type": "input_audio_buffer.append",
            "event_id": "evt_audio",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }
    )

    assert result == AppendAudioCommand(event_id="evt_audio", audio=pcm)


def test_append_audio_rejects_malformed_base64():
    result = _parse(
        {
            "type": "input_audio_buffer.append",
            "event_id": "evt_bad_audio",
            "audio": "not base64!?",
        }
    )

    assert result == RealtimeErrorEvent(
        code="invalid_audio",
        message="input_audio_buffer.append audio must be base64-encoded pcm16",
        event_id="evt_bad_audio",
    )


def test_append_audio_rejects_odd_length_pcm16_payload():
    result = _parse(
        {
            "type": "input_audio_buffer.append",
            "event_id": "evt_odd_audio",
            "audio": base64.b64encode(b"\x00").decode("ascii"),
        }
    )

    assert result == RealtimeErrorEvent(
        code="invalid_audio",
        message="input_audio_buffer.append audio must contain whole pcm16 samples",
        event_id="evt_odd_audio",
    )


def test_append_audio_rejects_decoded_audio_larger_than_buffer_limit():
    too_large = b"\x00" * (REALTIME_MAX_BUFFERED_AUDIO_BYTES + 1)

    result = _parse(
        {
            "type": "input_audio_buffer.append",
            "event_id": "evt_large_audio",
            "audio": base64.b64encode(too_large).decode("ascii"),
        }
    )

    assert result == RealtimeErrorEvent(
        code="payload_too_large",
        message="decoded audio exceeds 960000 byte realtime input buffer limit",
        event_id="evt_large_audio",
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"type": "input_audio_buffer.commit", "event_id": "evt_commit"},
            CommitAudioCommand(event_id="evt_commit"),
        ),
        (
            {"type": "input_audio_buffer.clear", "event_id": "evt_clear"},
            ClearAudioCommand(event_id="evt_clear"),
        ),
        (
            {
                "type": "response.create",
                "event_id": "evt_create",
                "response": {"modalities": ["audio", "text"]},
            },
            CreateResponseCommand(
                event_id="evt_create",
                response={"modalities": ["audio", "text"]},
            ),
        ),
        (
            {"type": "response.cancel", "event_id": "evt_cancel"},
            CancelResponseCommand(event_id="evt_cancel"),
        ),
    ],
)
def test_supported_non_session_commands_parse_to_internal_commands(payload, expected):
    assert _parse(payload) == expected


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (
            {"modalities": ["image"]},
            "response.create modalities must be a subset of audio and text",
        ),
        (
            {"output_audio_format": "pcm16"},
            "output_audio_format is a beta-era field; use response.audio.output.format",
        ),
        (
            {"audio": {"output": {"format": "mulaw"}}},
            "only pcm16 24000 Hz mono response output audio is supported",
        ),
        (
            {"audio": {"output": {"sample_rate_hz": 16000}}},
            "only pcm16 24000 Hz mono response output audio is supported",
        ),
        (
            {"audio": {"output": {"channels": 2}}},
            "only pcm16 24000 Hz mono response output audio is supported",
        ),
    ],
)
def test_response_create_rejects_unsupported_stage_one_options(response, expected_message):
    result = _parse(
        {
            "type": "response.create",
            "event_id": "evt_bad_response",
            "response": response,
        }
    )

    assert result == RealtimeErrorEvent(
        code="unsupported_session_option",
        message=expected_message,
        event_id="evt_bad_response",
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"type": "conversation.item.create", "event_id": "evt_item"},
            RealtimeErrorEvent(
                code="unsupported_event",
                message="conversation.item.create is not supported by the Stage 1 realtime endpoint",
                event_id="evt_item",
            ),
        ),
        (
            {
                "type": "session.update",
                "event_id": "evt_vad",
                "session": {"turn_detection": {"type": "server_vad"}},
            },
            RealtimeErrorEvent(
                code="unsupported_session_option",
                message="only manual turn detection is supported",
                event_id="evt_vad",
            ),
        ),
        (
            {
                "type": "response.create",
                "event_id": "evt_tools",
                "response": {"tools": [{"type": "function", "name": "lookup"}]},
            },
            RealtimeErrorEvent(
                code="unsupported_session_option",
                message="tool calls are not supported by the Stage 1 realtime endpoint",
                event_id="evt_tools",
            ),
        ),
    ],
)
def test_explicitly_unsupported_stage_one_shapes_return_openai_errors(payload, expected):
    assert _parse(payload) == expected


def test_realtime_error_serializes_to_exact_openai_shape():
    event = RealtimeErrorEvent(
        code="invalid_audio",
        message="input_audio_buffer.append audio must be base64-encoded pcm16",
        event_id="evt_bad_audio",
    )

    assert to_openai_server_event(event) == {
        "type": "error",
        "event_id": None,
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_audio",
            "message": "input_audio_buffer.append audio must be base64-encoded pcm16",
            "event_id": "evt_bad_audio",
        },
    }


def test_response_audio_delta_oversized_chunk_serializes_to_error_event():
    event = ResponseAudioDeltaEvent(
        event_id="evt_large_audio_delta",
        response_id="resp_1",
        item_id="item_2",
        output_index=0,
        content_index=0,
        audio=b"\x00" * (REALTIME_MAX_OUTPUT_CHUNK_BYTES + 1),
    )

    assert to_openai_server_event(event) == {
        "type": "error",
        "event_id": None,
        "error": {
            "type": "invalid_request_error",
            "code": "payload_too_large",
            "message": "response.output_audio.delta exceeds 65536 byte realtime output chunk limit",
            "event_id": "evt_large_audio_delta",
        },
    }


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (
            {"type": "session.update", "event_id": "evt_bad_type", "session": []},
            "session.update session must be an object when provided",
        ),
        (
            {
                "type": "session.update",
                "event_id": "evt_bad_type",
                "session": {"audio": []},
            },
            "session.audio must be an object when provided",
        ),
        (
            {
                "type": "session.update",
                "event_id": "evt_bad_type",
                "session": {"audio": {"input": []}},
            },
            "session.audio.input must be an object when provided",
        ),
        (
            {
                "type": "session.update",
                "event_id": "evt_bad_type",
                "session": {"audio": {"output": []}},
            },
            "session.audio.output must be an object when provided",
        ),
        (
            {
                "type": "session.update",
                "event_id": "evt_bad_type",
                "session": {"metadata": []},
            },
            "session.metadata must be an object when provided",
        ),
    ],
)
def test_session_update_rejects_present_fields_with_wrong_json_type(payload, expected_message):
    assert _parse(payload) == RealtimeErrorEvent(
        code="invalid_event",
        message=expected_message,
        event_id="evt_bad_type",
    )


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            SessionCreatedEvent(
                event_id="evt_server_session_created",
                session_id="sess_1",
                model="gpt-realtime",
                voice="alloy",
            ),
            {
                "type": "session.created",
                "event_id": "evt_server_session_created",
                "session": {
                    "id": "sess_1",
                    "type": "realtime",
                    "model": "gpt-realtime",
                    "voice": "alloy",
                    "audio": {
                        "input": {
                            "format": "pcm16",
                            "sample_rate_hz": 16000,
                            "channels": 1,
                        },
                        "output": {
                            "format": "pcm16",
                            "sample_rate_hz": 24000,
                            "channels": 1,
                        },
                    },
                    "turn_detection": None,
                },
            },
        ),
        (
            SessionUpdatedEvent(
                event_id="evt_server_session_updated",
                session_id="sess_1",
                model="gpt-realtime",
                voice="verse",
            ),
            {
                "type": "session.updated",
                "event_id": "evt_server_session_updated",
                "session": {
                    "id": "sess_1",
                    "type": "realtime",
                    "model": "gpt-realtime",
                    "voice": "verse",
                    "audio": {
                        "input": {
                            "format": "pcm16",
                            "sample_rate_hz": 16000,
                            "channels": 1,
                        },
                        "output": {
                            "format": "pcm16",
                            "sample_rate_hz": 24000,
                            "channels": 1,
                        },
                    },
                    "turn_detection": None,
                },
            },
        ),
        (
            InputAudioCommittedEvent(
                event_id="evt_server_commit",
                item_id="item_1",
                previous_item_id=None,
            ),
            {
                "type": "input_audio_buffer.committed",
                "event_id": "evt_server_commit",
                "item_id": "item_1",
                "previous_item_id": None,
            },
        ),
        (
            ConversationItemAddedEvent(
                event_id="evt_server_item",
                item_id="item_1",
                role="user",
                transcript="hello",
            ),
            {
                "type": "conversation.item.created",
                "event_id": "evt_server_item",
                "item": {
                    "id": "item_1",
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "transcript": "hello",
                        }
                    ],
                },
            },
        ),
        (
            ResponseCreatedEvent(
                event_id="evt_server_response_created",
                response_id="resp_1",
            ),
            {
                "type": "response.created",
                "event_id": "evt_server_response_created",
                "response": {
                    "id": "resp_1",
                    "object": "realtime.response",
                    "status": "in_progress",
                    "status_details": None,
                    "output": [],
                },
            },
        ),
        (
            ResponseTextDeltaEvent(
                event_id="evt_server_text",
                response_id="resp_1",
                item_id="item_2",
                output_index=0,
                content_index=0,
                delta="hello",
            ),
            {
                "type": "response.output_text.delta",
                "event_id": "evt_server_text",
                "response_id": "resp_1",
                "item_id": "item_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            },
        ),
        (
            ResponseAudioDeltaEvent(
                event_id="evt_server_audio",
                response_id="resp_1",
                item_id="item_2",
                output_index=0,
                content_index=0,
                audio=b"\x03\x00\x04\x00",
            ),
            {
                "type": "response.output_audio.delta",
                "event_id": "evt_server_audio",
                "response_id": "resp_1",
                "item_id": "item_2",
                "output_index": 0,
                "content_index": 0,
                "delta": base64.b64encode(b"\x03\x00\x04\x00").decode("ascii"),
            },
        ),
        (
            ResponseTranscriptDeltaEvent(
                event_id="evt_server_transcript",
                response_id="resp_1",
                item_id="item_2",
                output_index=0,
                content_index=0,
                delta="hello",
            ),
            {
                "type": "response.output_audio_transcript.delta",
                "event_id": "evt_server_transcript",
                "response_id": "resp_1",
                "item_id": "item_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            },
        ),
        (
            ResponseDoneEvent(
                event_id="evt_server_done",
                response_id="resp_1",
                status="completed",
            ),
            {
                "type": "response.done",
                "event_id": "evt_server_done",
                "response": {
                    "id": "resp_1",
                    "object": "realtime.response",
                    "status": "completed",
                    "status_details": None,
                    "output": [],
                },
            },
        ),
        (
            RateLimitsUpdatedEvent(
                event_id="evt_server_limits",
                name="audio_seconds",
                limit=30,
                remaining=12,
                reset_seconds=60,
            ),
            {
                "type": "rate_limits.updated",
                "event_id": "evt_server_limits",
                "rate_limits": [
                    {
                        "name": "audio_seconds",
                        "limit": 30,
                        "remaining": 12,
                        "reset_seconds": 60,
                    }
                ],
            },
        ),
    ],
)
def test_server_events_serialize_to_exact_openai_dictionaries(event, expected):
    assert to_openai_server_event(event) == expected
