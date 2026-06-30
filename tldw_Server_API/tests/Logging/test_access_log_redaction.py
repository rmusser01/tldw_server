"""Tests for access log path redaction."""

from __future__ import annotations

import logging

from tldw_Server_API.app.core.Logging.access_log_middleware import redact_access_log_message, redact_access_log_path


def test_redacts_audio_studio_media_ticket_token() -> None:
    path = "/api/v1/audio-studio/media-tickets/raw-secret-token-123"

    assert redact_access_log_path(path) == "/api/v1/audio-studio/media-tickets/[REDACTED]"


def test_redacts_ticket_token_without_changing_other_paths() -> None:
    assert redact_access_log_path("/api/v1/audio-studio/projects/p1") == "/api/v1/audio-studio/projects/p1"
    assert redact_access_log_path("/api/v1/audio-studio/media-tickets/token?download=1") == (
        "/api/v1/audio-studio/media-tickets/[REDACTED]?download=1"
    )


def test_redacts_ticket_token_in_uvicorn_access_message() -> None:
    message = (
        '127.0.0.1:54000 - "GET '
        '/api/v1/audio-studio/media-tickets/raw-secret-token-123?download=1 '
        'HTTP/1.1" 206'
    )

    redacted = redact_access_log_message(message)

    assert "raw-secret-token-123" not in redacted
    assert "/api/v1/audio-studio/media-tickets/[REDACTED]?download=1" in redacted
    assert redacted.endswith('HTTP/1.1" 206')


def test_intercept_handler_redacts_uvicorn_access_ticket_token(monkeypatch) -> None:
    from tldw_Server_API.app import main as app_main

    captured: list[tuple[str | int, str]] = []

    class _Level:
        name = "INFO"

    class _StubLogger:
        def level(self, _level_name: str) -> _Level:
            return _Level()

        def opt(self, **_kwargs):
            return self

        def log(self, level: str | int, message: str) -> None:
            captured.append((level, message))

    monkeypatch.setattr(app_main, "logger", _StubLogger())
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='127.0.0.1:54000 - "GET /api/v1/audio-studio/media-tickets/raw-secret-token-123 HTTP/1.1" 206',
        args=(),
        exc_info=None,
    )

    app_main.InterceptHandler().emit(record)

    assert captured
    assert "raw-secret-token-123" not in captured[0][1]
    assert "/api/v1/audio-studio/media-tickets/[REDACTED]" in captured[0][1]
