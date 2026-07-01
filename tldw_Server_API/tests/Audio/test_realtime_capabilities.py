from dataclasses import asdict

from tldw_Server_API.app.core.Audio.Realtime.capabilities import (
    RealtimeCapabilities,
    build_realtime_capabilities,
)
from tldw_Server_API.app.core.Audio.Realtime.constants import (
    REALTIME_AUTH_FAILURE_CLOSE_CODE,
    REALTIME_ENDPOINT_DENIED_CLOSE_CODE,
    OPENAI_REALTIME_INPUT_AUDIO_APPEND,
    OPENAI_REALTIME_INPUT_AUDIO_CLEAR,
    OPENAI_REALTIME_INPUT_AUDIO_COMMIT,
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
    REALTIME_INPUT_CHANNELS,
    REALTIME_INPUT_SAMPLE_RATE_HZ,
    REALTIME_INPUT_SAMPLE_WIDTH_BYTES,
    REALTIME_MAX_BUFFERED_AUDIO_BYTES,
    REALTIME_MAX_BUFFERED_AUDIO_SECONDS,
    REALTIME_MAX_JSON_FRAME_BYTES,
    REALTIME_MAX_OUTPUT_CHUNK_BYTES,
    REALTIME_INTERNAL_ERROR_CLOSE_CODE,
    REALTIME_NORMAL_CLOSE_CODE,
    REALTIME_OUTPUT_CHANNELS,
    REALTIME_OUTPUT_SAMPLE_RATE_HZ,
    REALTIME_PAYLOAD_TOO_LARGE_CLOSE_CODE,
    REALTIME_POLICY_VIOLATION_CLOSE_CODE,
    REALTIME_QUOTA_DENIED_CLOSE_CODE,
    REALTIME_TLDW_TTS_REQUEST_FORMAT,
)


def _capabilities() -> dict:
    capabilities = build_realtime_capabilities()

    assert isinstance(capabilities, RealtimeCapabilities)
    return asdict(capabilities)


def test_realtime_capabilities_include_supported_routes_modalities_and_events():
    payload = _capabilities()

    assert payload["routes"] == [
        {"path": "/api/v1/audio/realtime", "experimental": True},
        {"path": "/v1/realtime", "experimental": True},
    ]
    assert payload["modalities"] == {
        "input": ["audio"],
        "output": ["audio", "text"],
    }
    assert payload["events"]["client"] == [
        OPENAI_REALTIME_SESSION_UPDATE,
        OPENAI_REALTIME_INPUT_AUDIO_APPEND,
        OPENAI_REALTIME_INPUT_AUDIO_COMMIT,
        OPENAI_REALTIME_INPUT_AUDIO_CLEAR,
        OPENAI_REALTIME_RESPONSE_CREATE,
        OPENAI_REALTIME_RESPONSE_CANCEL,
    ]
    assert payload["events"]["server"] == [
        OPENAI_REALTIME_SESSION_CREATED,
        OPENAI_REALTIME_SESSION_UPDATED,
        "input_audio_buffer.committed",
        "conversation.item.created",
        OPENAI_REALTIME_RESPONSE_CREATED,
        OPENAI_REALTIME_RESPONSE_OUTPUT_TEXT_DELTA,
        OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_DELTA,
        OPENAI_REALTIME_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
        OPENAI_REALTIME_RESPONSE_DONE,
        OPENAI_REALTIME_RATE_LIMITS_UPDATED,
        "error",
    ]
    assert payload["events"]["unsupported"] == [
        "conversation.item.create",
        "tool_calls",
        "server_vad",
        "output_audio_format",
        "binary_websocket_frames",
    ]


def test_realtime_capabilities_include_exact_audio_contract_and_limits():
    payload = _capabilities()

    assert payload["audio"] == {
        "input": {
            "format": "pcm16",
            "channels": REALTIME_INPUT_CHANNELS,
            "sample_rate_hz": REALTIME_INPUT_SAMPLE_RATE_HZ,
            "sample_width_bytes": REALTIME_INPUT_SAMPLE_WIDTH_BYTES,
            "wire_encoding": "base64_json_audio",
            "wire_field": "audio",
        },
        "output": {
            "openai_wire_format": "pcm16",
            "tldw_tts_request_format": REALTIME_TLDW_TTS_REQUEST_FORMAT,
            "channels": REALTIME_OUTPUT_CHANNELS,
            "sample_rate_hz": REALTIME_OUTPUT_SAMPLE_RATE_HZ,
            "wire_encoding": "base64_json_response_output_audio_delta",
            "wire_field": "response.output_audio.delta",
        },
    }
    assert payload["limits"] == {
        "max_json_frame_bytes": REALTIME_MAX_JSON_FRAME_BYTES,
        "max_buffered_audio_seconds": REALTIME_MAX_BUFFERED_AUDIO_SECONDS,
        "input_sample_rate_hz": REALTIME_INPUT_SAMPLE_RATE_HZ,
        "input_sample_width_bytes": REALTIME_INPUT_SAMPLE_WIDTH_BYTES,
        "max_buffered_audio_bytes": REALTIME_MAX_BUFFERED_AUDIO_BYTES,
        "max_output_chunk_bytes": REALTIME_MAX_OUTPUT_CHUNK_BYTES,
    }
    assert payload["close_codes"] == {
        "auth_failure": REALTIME_AUTH_FAILURE_CLOSE_CODE,
        "endpoint_denied": REALTIME_ENDPOINT_DENIED_CLOSE_CODE,
        "quota_denied": REALTIME_QUOTA_DENIED_CLOSE_CODE,
        "quota_denied_when_AUDIO_WS_QUOTA_CLOSE_1008": REALTIME_POLICY_VIOLATION_CLOSE_CODE,
        "oversized_frame": REALTIME_PAYLOAD_TOO_LARGE_CLOSE_CODE,
        "fatal_internal_error": REALTIME_INTERNAL_ERROR_CLOSE_CODE,
        "normal_completion": REALTIME_NORMAL_CLOSE_CODE,
    }


def test_realtime_capabilities_mark_stage_one_constraints_explicitly():
    payload = _capabilities()

    assert payload["experimental"] is True
    assert payload["turn_detection"] == {
        "supported": ["manual"],
        "unsupported": ["server_vad"],
    }
    assert payload["rate_limits"] == {
        "event": OPENAI_REALTIME_RATE_LIMITS_UPDATED,
        "semantics": "tldw quota compatibility; not OpenAI quota parity",
    }
