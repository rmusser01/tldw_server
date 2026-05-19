"""Bounded STT metrics helpers and metric family definitions."""

from __future__ import annotations

import re
from typing import Any


_STT_LATENCY_BUCKETS_SECONDS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
_STT_QUEUE_BUCKETS_SECONDS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5]
_STT_TOKEN_BUCKETS_SECONDS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]

_ALLOWED_ENDPOINTS = {
    "audio.transcriptions",
    "audio.stream.transcribe",
    "audio.chat.stream",
    "ingestion",
}
_ENDPOINT_ALIASES = {
    "audio.transcription": "audio.transcriptions",
    "audio_transcriptions": "audio.transcriptions",
    "audio_stream_transcribe": "audio.stream.transcribe",
    "audio.stream.ws": "audio.stream.transcribe",
    "audio_stream_ws": "audio.stream.transcribe",
    "audio_chat_stream": "audio.chat.stream",
    "audio.chat.ws": "audio.chat.stream",
    "audio_chat_ws": "audio.chat.stream",
    "ingestion": "ingestion",
}

_ALLOWED_STATUSES = {
    "ok",
    "quota_exceeded",
    "bad_request",
    "provider_error",
    "model_unavailable",
    "internal_error",
}
_STATUS_ALIASES = {
    "success": "ok",
    "created": "ok",
    "done": "ok",
    "quota": "quota_exceeded",
    "rate_limited": "quota_exceeded",
    "validation_error": "bad_request",
    "invalid_request": "bad_request",
    "provider_failure": "provider_error",
    "error": "internal_error",
    "failed": "internal_error",
}

_ALLOWED_REASONS = {
    "auth",
    "quota",
    "provider_error",
    "model_unavailable",
    "invalid_control",
    "validation_error",
    "timeout",
    "internal",
}
_REASON_ALIASES = {
    "bad_request": "validation_error",
    "validation": "validation_error",
    "invalid_json": "validation_error",
    "invalid_base64": "validation_error",
    "quota_exceeded": "quota",
    "rate_limited": "quota",
    "llm_error": "provider_error",
    "tts_error": "provider_error",
    "stt_error": "provider_error",
    "provider_failure": "provider_error",
    "server_shutdown": "internal",
    "error": "internal",
    "failed": "internal",
}

_ALLOWED_SESSION_CLOSE_REASONS = {
    "client_stop",
    "client_disconnect",
    "server_shutdown",
    "error",
}
_SESSION_CLOSE_REASON_ALIASES = {
    "stop": "client_stop",
    "client_cancel": "client_stop",
    "disconnect": "client_disconnect",
    "websocket_disconnect": "client_disconnect",
    "shutdown": "server_shutdown",
    "internal": "error",
    "timeout": "error",
}

_ALLOWED_WRITE_RESULTS = {
    "created",
    "deduped",
    "superseded",
    "failed",
}
_WRITE_RESULT_ALIASES = {
    "updated": "deduped",
    "reused": "deduped",
    "exists": "deduped",
    "conflict_retry": "deduped",
    "wrapped_unique_conflict": "failed",
    "error": "failed",
}

_ALLOWED_REDACTION_OUTCOMES = {
    "applied",
    "not_requested",
    "skipped",
    "failed",
}
_REDACTION_OUTCOME_ALIASES = {
    "disabled": "not_requested",
    "not_enabled": "not_requested",
    "none": "not_requested",
    "allow_unredacted_partials": "skipped",
    "partial_bypass": "skipped",
    "error": "failed",
}

_ALLOWED_PROVIDERS = {
    "whisper",
    "nemo",
    "qwen2audio",
    "external",
    "other",
}
_PROVIDER_HINTS = (
    ("qwen2audio", ("qwen2audio", "qwen2-audio")),
    ("nemo", ("nemo", "parakeet", "canary")),
    ("whisper", ("whisper",)),
    ("external", ("external",)),
)

_MODEL_HINTS = (
    ("qwen2audio", ("qwen2audio", "qwen2-audio")),
    ("parakeet", ("parakeet",)),
    ("canary", ("canary",)),
    ("whisper", ("whisper",)),
)

_VARIANT_HINTS = (
    ("onnx", ("onnx",)),
    ("turbo", ("turbo",)),
    ("large", ("large",)),
    ("standard", ("standard", "default")),
)

_ALLOWED_READ_PATHS = {
    "latest_run",
    "legacy_fallback",
}
_READ_PATH_ALIASES = {
    "latest": "latest_run",
    "fallback": "legacy_fallback",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_STT_METRIC_SPECS = (
    {
        "name": "audio_stt_requests_total",
        "type": "counter",
        "description": "Total STT request attempts by endpoint/provider/model/status",
        "labels": ["endpoint", "provider", "model", "status"],
    },
    {
        "name": "audio_stt_streaming_sessions_started_total",
        "type": "counter",
        "description": "Total STT streaming sessions started",
        "labels": ["provider"],
    },
    {
        "name": "audio_stt_streaming_sessions_ended_total",
        "type": "counter",
        "description": "Total STT streaming sessions ended",
        "labels": ["provider", "session_close_reason"],
    },
    {
        "name": "audio_stt_errors_total",
        "type": "counter",
        "description": "Total STT errors by endpoint/provider/reason",
        "labels": ["endpoint", "provider", "reason"],
    },
    {
        "name": "audio_stt_run_writes_total",
        "type": "counter",
        "description": "Total transcript run write attempts by provider/result",
        "labels": ["provider", "write_result"],
    },
    {
        "name": "audio_stt_redaction_total",
        "type": "counter",
        "description": "Total transcript redaction outcomes by endpoint",
        "labels": ["endpoint", "redaction_outcome"],
    },
    {
        "name": "audio_stt_latency_seconds",
        "type": "histogram",
        "description": "STT latency by endpoint/provider/model",
        "unit": "s",
        "labels": ["endpoint", "provider", "model"],
        "buckets": _STT_LATENCY_BUCKETS_SECONDS,
    },
    {
        "name": "audio_stt_queue_wait_seconds",
        "type": "histogram",
        "description": "STT queue wait duration by endpoint",
        "unit": "s",
        "labels": ["endpoint"],
        "buckets": _STT_QUEUE_BUCKETS_SECONDS,
    },
    {
        "name": "audio_stt_streaming_token_latency_seconds",
        "type": "histogram",
        "description": "Streaming STT token latency by provider/model",
        "unit": "s",
        "labels": ["provider", "model"],
        "buckets": _STT_TOKEN_BUCKETS_SECONDS,
    },
    {
        "name": "audio_stt_transcript_read_path_total",
        "type": "counter",
        "description": "Total transcript read-path selections",
        "labels": ["path"],
    },
)


def _slugify(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return _SLUG_RE.sub("_", raw).strip("_")


def _normalize_enum(value: Any, *, allowed: set[str], aliases: dict[str, str], fallback: str) -> str:
    slug = _slugify(value)
    if not slug:
        return fallback
    slug = aliases.get(slug, slug)
    return slug if slug in allowed else fallback


def normalize_stt_endpoint(endpoint: Any) -> str:
    return _normalize_enum(
        endpoint,
        allowed=_ALLOWED_ENDPOINTS,
        aliases=_ENDPOINT_ALIASES,
        fallback="other",
    )


def normalize_stt_status(status: Any) -> str:
    return _normalize_enum(
        status,
        allowed=_ALLOWED_STATUSES,
        aliases=_STATUS_ALIASES,
        fallback="internal_error",
    )


def normalize_stt_reason(reason: Any) -> str:
    return _normalize_enum(
        reason,
        allowed=_ALLOWED_REASONS,
        aliases=_REASON_ALIASES,
        fallback="internal",
    )


def normalize_stt_session_close_reason(reason: Any) -> str:
    return _normalize_enum(
        reason,
        allowed=_ALLOWED_SESSION_CLOSE_REASONS,
        aliases=_SESSION_CLOSE_REASON_ALIASES,
        fallback="error",
    )


def normalize_stt_write_result(result: Any) -> str:
    return _normalize_enum(
        result,
        allowed=_ALLOWED_WRITE_RESULTS,
        aliases=_WRITE_RESULT_ALIASES,
        fallback="failed",
    )


def normalize_stt_redaction_outcome(outcome: Any) -> str:
    return _normalize_enum(
        outcome,
        allowed=_ALLOWED_REDACTION_OUTCOMES,
        aliases=_REDACTION_OUTCOME_ALIASES,
        fallback="failed",
    )


def normalize_stt_read_path(path: Any) -> str:
    return _normalize_enum(
        path,
        allowed=_ALLOWED_READ_PATHS,
        aliases=_READ_PATH_ALIASES,
        fallback="legacy_fallback",
    )


def normalize_stt_provider(provider: Any) -> str:
    slug = _slugify(provider)
    if not slug:
        return "other"
    if slug in _ALLOWED_PROVIDERS:
        return slug
    for normalized, hints in _PROVIDER_HINTS:
        if any(hint in slug for hint in hints):
            return normalized
    return "other"


def normalize_stt_model(model: Any) -> str:
    slug = _slugify(model)
    if not slug:
        return "other"
    for normalized, hints in _MODEL_HINTS:
        if any(hint in slug for hint in hints):
            return normalized
    return "other"


def normalize_stt_variant(variant: Any) -> str:
    slug = _slugify(variant)
    if not slug:
        return "other"
    for normalized, hints in _VARIANT_HINTS:
        if any(hint in slug for hint in hints):
            return normalized
    return "other"


def iter_stt_metric_definitions(metric_definition_cls: Any, metric_type_enum: Any) -> list[Any]:
    type_map = {
        "counter": metric_type_enum.COUNTER,
        "histogram": metric_type_enum.HISTOGRAM,
    }
    definitions = []
    for spec in _STT_METRIC_SPECS:
        definitions.append(
            metric_definition_cls(
                name=spec["name"],
                type=type_map[spec["type"]],
                description=spec["description"],
                unit=spec.get("unit", ""),
                labels=list(spec.get("labels", [])),
                buckets=list(spec.get("buckets", [])) or None,
            )
        )
    return definitions


def emit_stt_request_total(*, endpoint: Any, provider: Any, model: Any, status: Any, value: float = 1.0) -> None:
    from .metrics_manager import increment_counter

    increment_counter(
        "audio_stt_requests_total",
        value=value,
        labels={
            "endpoint": normalize_stt_endpoint(endpoint),
            "provider": normalize_stt_provider(provider),
            "model": normalize_stt_model(model),
            "status": normalize_stt_status(status),
        },
    )


def emit_stt_session_start_total(*, provider: Any, value: float = 1.0) -> None:
    from .metrics_manager import increment_counter

    increment_counter(
        "audio_stt_streaming_sessions_started_total",
        value=value,
        labels={"provider": normalize_stt_provider(provider)},
    )


def emit_stt_session_end_total(*, provider: Any, session_close_reason: Any, value: float = 1.0) -> None:
    from .metrics_manager import increment_counter

    increment_counter(
        "audio_stt_streaming_sessions_ended_total",
        value=value,
        labels={
            "provider": normalize_stt_provider(provider),
            "session_close_reason": normalize_stt_session_close_reason(session_close_reason),
        },
    )


def emit_stt_error_total(*, endpoint: Any, provider: Any, reason: Any, value: float = 1.0) -> None:
    from .metrics_manager import increment_counter

    increment_counter(
        "audio_stt_errors_total",
        value=value,
        labels={
            "endpoint": normalize_stt_endpoint(endpoint),
            "provider": normalize_stt_provider(provider),
            "reason": normalize_stt_reason(reason),
        },
    )


def emit_stt_run_write_total(*, provider: Any, write_result: Any, value: float = 1.0) -> None:
    from .metrics_manager import increment_counter

    increment_counter(
        "audio_stt_run_writes_total",
        value=value,
        labels={
            "provider": normalize_stt_provider(provider),
            "write_result": normalize_stt_write_result(write_result),
        },
    )


def emit_stt_redaction_total(*, endpoint: Any, redaction_outcome: Any, value: float = 1.0) -> None:
    from .metrics_manager import increment_counter

    increment_counter(
        "audio_stt_redaction_total",
        value=value,
        labels={
            "endpoint": normalize_stt_endpoint(endpoint),
            "redaction_outcome": normalize_stt_redaction_outcome(redaction_outcome),
        },
    )


def emit_stt_transcript_read_path_total(*, path: Any, value: float = 1.0) -> None:
    from .metrics_manager import increment_counter

    increment_counter(
        "audio_stt_transcript_read_path_total",
        value=value,
        labels={"path": normalize_stt_read_path(path)},
    )


def observe_stt_latency_seconds(*, endpoint: Any, provider: Any, model: Any, value: float) -> None:
    from .metrics_manager import observe_histogram

    observe_histogram(
        "audio_stt_latency_seconds",
        value=value,
        labels={
            "endpoint": normalize_stt_endpoint(endpoint),
            "provider": normalize_stt_provider(provider),
            "model": normalize_stt_model(model),
        },
    )


def observe_stt_queue_wait_seconds(*, endpoint: Any, value: float) -> None:
    from .metrics_manager import observe_histogram

    observe_histogram(
        "audio_stt_queue_wait_seconds",
        value=value,
        labels={"endpoint": normalize_stt_endpoint(endpoint)},
    )


def observe_stt_streaming_token_latency_seconds(*, provider: Any, model: Any, value: float) -> None:
    from .metrics_manager import observe_histogram

    observe_histogram(
        "audio_stt_streaming_token_latency_seconds",
        value=value,
        labels={
            "provider": normalize_stt_provider(provider),
            "model": normalize_stt_model(model),
        },
    )


def observe_stt_final_latency_seconds(*, endpoint: Any, model: Any, variant: Any, value: float) -> None:
    from .metrics_manager import observe_histogram

    observe_histogram(
        "stt_final_latency_seconds",
        value=value,
        labels={
            "endpoint": normalize_stt_endpoint(endpoint),
            "model": normalize_stt_model(model),
            "variant": normalize_stt_variant(variant),
        },
    )
