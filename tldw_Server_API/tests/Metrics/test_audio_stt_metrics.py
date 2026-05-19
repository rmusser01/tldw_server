import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager


pytestmark = pytest.mark.unit


def test_audio_stt_metric_families_are_registered():
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        assert "audio_stt_requests_total" in registry.metrics
        assert "audio_stt_streaming_sessions_started_total" in registry.metrics
        assert "audio_stt_streaming_sessions_ended_total" in registry.metrics
        assert "audio_stt_errors_total" in registry.metrics
        assert "audio_stt_run_writes_total" in registry.metrics
        assert "audio_stt_redaction_total" in registry.metrics
        assert "audio_stt_latency_seconds" in registry.metrics
        assert "audio_stt_queue_wait_seconds" in registry.metrics
        assert "audio_stt_streaming_token_latency_seconds" in registry.metrics
        assert "audio_stt_transcript_read_path_total" in registry.metrics
    finally:
        metrics_manager._metrics_registry = None


def test_stt_metrics_normalize_labels_and_bucket_unknown_models():
    from tldw_Server_API.app.core.Metrics.stt_metrics import (
        emit_stt_error_total,
        emit_stt_request_total,
        emit_stt_run_write_total,
        emit_stt_session_end_total,
        emit_stt_session_start_total,
        emit_stt_redaction_total,
        emit_stt_transcript_read_path_total,
        observe_stt_latency_seconds,
    )

    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        emit_stt_request_total(
            endpoint="totally.custom.endpoint",
            provider="mystery-provider",
            model="parakeet-ctc-0.6b",
            status="definitely_ok",
        )
        emit_stt_session_start_total(provider="mystery-provider")
        emit_stt_session_end_total(provider="mystery-provider", session_close_reason="network_reset")
        emit_stt_error_total(
            endpoint="weird.endpoint",
            provider="mystery-provider",
            reason="stacktrace-goes-here",
        )
        emit_stt_run_write_total(provider="mystery-provider", write_result="wrapped_unique_conflict")
        emit_stt_redaction_total(endpoint="audio.chat.stream", redaction_outcome="maybe")
        emit_stt_transcript_read_path_total(path="fallback")
        observe_stt_latency_seconds(
            endpoint="audio.stream.transcribe",
            provider="mystery-provider",
            model="totally-unknown-model",
            value=0.25,
        )

        assert registry.get_cumulative_counter(
            "audio_stt_requests_total",
            {
                "endpoint": "other",
                "provider": "other",
                "model": "parakeet",
                "status": "internal_error",
            },
        ) == 1
        assert registry.get_cumulative_counter(
            "audio_stt_streaming_sessions_started_total",
            {"provider": "other"},
        ) == 1
        assert registry.get_cumulative_counter(
            "audio_stt_streaming_sessions_ended_total",
            {"provider": "other", "session_close_reason": "error"},
        ) == 1
        assert registry.get_cumulative_counter(
            "audio_stt_errors_total",
            {"endpoint": "other", "provider": "other", "reason": "internal"},
        ) == 1
        assert registry.get_cumulative_counter(
            "audio_stt_run_writes_total",
            {"provider": "other", "write_result": "failed"},
        ) == 1
        assert registry.get_cumulative_counter(
            "audio_stt_redaction_total",
            {"endpoint": "audio.chat.stream", "redaction_outcome": "failed"},
        ) == 1
        assert registry.get_cumulative_counter(
            "audio_stt_transcript_read_path_total",
            {"path": "legacy_fallback"},
        ) == 1

        metrics_text = registry.export_prometheus_format()
        assert 'audio_stt_latency_seconds_count{endpoint="audio.stream.transcribe",model="other",provider="other"} 1' in metrics_text
        assert "totally-unknown-model" not in metrics_text
        assert "mystery-provider" not in metrics_text
    finally:
        metrics_manager._metrics_registry = None
