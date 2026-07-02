"""Capability data for the Stage 1 realtime speech protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.Audio.Realtime.constants import (
    REALTIME_AUTH_FAILURE_CLOSE_CODE,
    REALTIME_ENDPOINT_DENIED_CLOSE_CODE,
    REALTIME_INPUT_AUDIO_FORMAT,
    REALTIME_INPUT_CHANNELS,
    REALTIME_INPUT_SAMPLE_RATE_HZ,
    REALTIME_INPUT_SAMPLE_WIDTH_BYTES,
    REALTIME_INTERNAL_ERROR_CLOSE_CODE,
    REALTIME_MAX_BUFFERED_AUDIO_BYTES,
    REALTIME_MAX_BUFFERED_AUDIO_SECONDS,
    REALTIME_MAX_JSON_FRAME_BYTES,
    REALTIME_MAX_OUTPUT_CHUNK_BYTES,
    REALTIME_NORMAL_CLOSE_CODE,
    REALTIME_OUTPUT_AUDIO_FORMAT,
    REALTIME_OUTPUT_CHANNELS,
    REALTIME_OUTPUT_SAMPLE_RATE_HZ,
    REALTIME_PAYLOAD_TOO_LARGE_CLOSE_CODE,
    REALTIME_POLICY_VIOLATION_CLOSE_CODE,
    REALTIME_QUOTA_DENIED_CLOSE_CODE,
    REALTIME_SUPPORTED_CLIENT_EVENTS,
    REALTIME_SUPPORTED_SERVER_EVENTS,
    REALTIME_TLDW_TTS_REQUEST_FORMAT,
    OPENAI_REALTIME_RATE_LIMITS_UPDATED,
)


@dataclass(frozen=True)
class RealtimeCapabilities:
    experimental: bool
    routes: list[dict[str, Any]]
    modalities: dict[str, list[str]]
    events: dict[str, list[str]]
    audio: dict[str, dict[str, Any]]
    limits: dict[str, int]
    close_codes: dict[str, int]
    turn_detection: dict[str, list[str]]
    rate_limits: dict[str, str]
    persistence: dict[str, Any]
    optional_events: dict[str, dict[str, Any]]
    deferred_features: list[str]
    notes: list[str] = field(default_factory=list)


def build_realtime_capabilities() -> RealtimeCapabilities:
    """Build Stage 1 realtime capability metadata without provider imports."""

    return RealtimeCapabilities(
        experimental=True,
        routes=[
            {"path": "/api/v1/audio/realtime", "experimental": True},
            {"path": "/v1/realtime", "experimental": True},
        ],
        modalities={
            "input": ["audio"],
            "output": ["audio", "text"],
        },
        events={
            "client": list(REALTIME_SUPPORTED_CLIENT_EVENTS),
            "server": list(REALTIME_SUPPORTED_SERVER_EVENTS),
            "unsupported": [
                "conversation.item.create",
                "tool_calls",
                "server_vad",
                "input_audio_format",
                "output_audio_format",
                "binary_websocket_frames",
            ],
        },
        audio={
            "input": {
                "format": REALTIME_INPUT_AUDIO_FORMAT,
                "channels": REALTIME_INPUT_CHANNELS,
                "sample_rate_hz": REALTIME_INPUT_SAMPLE_RATE_HZ,
                "sample_width_bytes": REALTIME_INPUT_SAMPLE_WIDTH_BYTES,
                "wire_encoding": "base64_json_audio",
                "wire_field": "audio",
            },
            "output": {
                "openai_wire_format": REALTIME_OUTPUT_AUDIO_FORMAT,
                "tldw_tts_request_format": REALTIME_TLDW_TTS_REQUEST_FORMAT,
                "channels": REALTIME_OUTPUT_CHANNELS,
                "sample_rate_hz": REALTIME_OUTPUT_SAMPLE_RATE_HZ,
                "wire_encoding": "base64_json_response_output_audio_delta",
                "wire_field": "response.output_audio.delta",
            },
        },
        limits={
            "max_json_frame_bytes": REALTIME_MAX_JSON_FRAME_BYTES,
            "max_buffered_audio_seconds": REALTIME_MAX_BUFFERED_AUDIO_SECONDS,
            "input_sample_rate_hz": REALTIME_INPUT_SAMPLE_RATE_HZ,
            "input_sample_width_bytes": REALTIME_INPUT_SAMPLE_WIDTH_BYTES,
            "max_buffered_audio_bytes": REALTIME_MAX_BUFFERED_AUDIO_BYTES,
            "max_output_chunk_bytes": REALTIME_MAX_OUTPUT_CHUNK_BYTES,
        },
        close_codes={
            "auth_failure": REALTIME_AUTH_FAILURE_CLOSE_CODE,
            "endpoint_denied": REALTIME_ENDPOINT_DENIED_CLOSE_CODE,
            "quota_denied": REALTIME_QUOTA_DENIED_CLOSE_CODE,
            "quota_denied_when_AUDIO_WS_QUOTA_CLOSE_1008": REALTIME_POLICY_VIOLATION_CLOSE_CODE,
            "oversized_frame": REALTIME_PAYLOAD_TOO_LARGE_CLOSE_CODE,
            "fatal_internal_error": REALTIME_INTERNAL_ERROR_CLOSE_CODE,
            "normal_completion": REALTIME_NORMAL_CLOSE_CODE,
        },
        turn_detection={
            "supported": ["manual"],
            "unsupported": ["server_vad"],
        },
        rate_limits={
            "event": OPENAI_REALTIME_RATE_LIMITS_UPDATED,
            "semantics": "tldw quota compatibility; not OpenAI quota parity",
        },
        persistence={
            "supported": True,
            "default": "ephemeral",
            "enable_with": {
                "metadata.tldw.persist": True,
                "metadata.tldw.conversation_id": "integer",
            },
            "raw_audio_persistence": False,
        },
        optional_events={
            "conversation.item.create": {
                "supported": False,
                "status": "deferred",
                "reason": "Stage 1 only supports committed input audio items generated by input_audio_buffer.commit",
            }
        },
        deferred_features=[
            "conversation.item.create",
            "tool_calls",
            "server_vad",
            "response_scoped_overrides",
            "output_audio_format_selection",
            "binary_websocket_frames",
        ],
        notes=[
            "Stage 1 accepts JSON OpenAI GA event frames only.",
            "Binary WebSocket frames and provider adapters are outside Stage 1.",
        ],
    )
