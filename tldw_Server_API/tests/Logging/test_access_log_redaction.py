"""Tests for access log path redaction."""

from __future__ import annotations

from tldw_Server_API.app.core.Logging.access_log_middleware import redact_access_log_path


def test_redacts_audio_studio_media_ticket_token() -> None:
    path = "/api/v1/audio-studio/media-tickets/raw-secret-token-123"

    assert redact_access_log_path(path) == "/api/v1/audio-studio/media-tickets/[REDACTED]"


def test_redacts_ticket_token_without_changing_other_paths() -> None:
    assert redact_access_log_path("/api/v1/audio-studio/projects/p1") == "/api/v1/audio-studio/projects/p1"
    assert redact_access_log_path("/api/v1/audio-studio/media-tickets/token?download=1") == (
        "/api/v1/audio-studio/media-tickets/[REDACTED]"
    )
