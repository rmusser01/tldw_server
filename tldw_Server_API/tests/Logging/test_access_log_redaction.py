"""Tests for access log path redaction."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.core.Logging.access_log_middleware import (
    AccessLogMiddleware,
    redact_access_log_message,
    redact_access_log_path,
)


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


def test_access_log_never_captures_standalone_source_body() -> None:
    sentinel = "PRIVATE_HTML_ACCESS_LOG_52e7a5"
    captured: list[str] = []
    app = FastAPI()

    @app.post("/api/v1/slides/generations")
    async def _generation(request: Request) -> dict[str, bool]:
        await request.body()
        return {"ok": True}

    app.add_middleware(AccessLogMiddleware)
    sink_id = logger.add(lambda message: captured.append(str(message)))
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/slides/generations",
                content=sentinel,
            )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 200
    assert sentinel not in "".join(captured)
