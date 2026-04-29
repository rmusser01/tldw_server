from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from loguru import logger
from starlette import status

import pytest

from tldw_Server_API.app.core.Audio import tts_service
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSError,
    TTSAuthenticationError,
    TTSInvalidVoiceReferenceError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSNetworkError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSProviderUnavailableError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSResourceError,
    TTSTimeoutError,
    TTSValidationError,
)


pytestmark = pytest.mark.unit

_LEAK = "backend exploded /tmp/secret-token"


def _assert_safe_log(rendered: str) -> None:
    assert "backend exploded" not in rendered
    assert "/tmp/secret-token" not in rendered
    assert "exc_info" not in rendered


@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_detail"),
    [
        (
            TTSInvalidVoiceReferenceError(_LEAK, provider="kokoro"),
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "TTS voice reference invalid",
        ),
        (
            TTSValidationError(_LEAK, provider="kokoro"),
            status.HTTP_400_BAD_REQUEST,
            "TTS validation failed",
        ),
        (
            TTSModelNotFoundError(_LEAK, provider="kokoro"),
            status.HTTP_404_NOT_FOUND,
            "Requested TTS model not found",
        ),
        (
            TTSProviderNotConfiguredError(_LEAK, provider="kokoro"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "TTS service unavailable",
        ),
        (
            TTSProviderInitializationError(_LEAK, provider="kokoro"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "TTS service unavailable",
        ),
        (
            TTSModelLoadError(_LEAK, provider="kokoro"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "TTS model unavailable",
        ),
        (
            TTSResourceError(_LEAK, provider="kokoro"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "TTS service unavailable",
        ),
        (
            TTSProviderUnavailableError(_LEAK, provider="openai"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "TTS provider unavailable",
        ),
        (
            TTSAuthenticationError(_LEAK, provider="openai"),
            status.HTTP_502_BAD_GATEWAY,
            "TTS provider authentication failed",
        ),
        (
            TTSNetworkError(_LEAK, provider="openai"),
            status.HTTP_502_BAD_GATEWAY,
            "TTS provider request failed",
        ),
        (
            TTSTimeoutError(_LEAK, provider="openai"),
            status.HTTP_504_GATEWAY_TIMEOUT,
            "TTS provider timed out",
        ),
        (
            TTSRateLimitError(_LEAK, provider="openai"),
            status.HTTP_429_TOO_MANY_REQUESTS,
            "TTS provider rate limit exceeded. Please try again later.",
        ),
        (
            TTSQuotaExceededError(_LEAK, provider="openai"),
            status.HTTP_402_PAYMENT_REQUIRED,
            "TTS quota exceeded. Please review your plan or quota.",
        ),
        (
            TTSError(_LEAK, provider="openai"),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "TTS generation failed",
        ),
        (
            RuntimeError(_LEAK),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An unexpected error occurred during audio generation",
        ),
    ],
)
def test_tts_error_mapping_logs_do_not_leak_raw_exception(
    exc: Exception,
    expected_status: int,
    expected_detail: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DEBUG_ERROR_DETAILS", raising=False)

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    try:
        with pytest.raises(HTTPException) as raised:
            tts_service._raise_for_tts_error(exc, request_id="req-123")
    finally:
        logger.remove(sink_id)

    assert raised.value.status_code == expected_status
    assert raised.value.detail == {
        "message": expected_detail,
        "request_id": "req-123",
    }
    rendered = "\n".join(records)
    _assert_safe_log(rendered)


def test_speech_request_validation_log_does_not_leak_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DEBUG_ERROR_DETAILS", raising=False)

    class _Config:
        strict_validation = False

    class _FailingValidator:
        def __init__(self, config: dict[str, Any]):
            self.config = config

        def sanitize_text(self, text: str, *, provider: str | None = None) -> str:
            raise TTSValidationError(_LEAK, provider=provider)

    class _Request:
        model = "tts-1"
        input = "hello"
        voice = "alloy"
        model_fields_set = {"model", "input", "voice"}

    monkeypatch.setattr(tts_service, "get_tts_config", lambda: _Config())
    monkeypatch.setattr(tts_service, "TTSInputValidator", _FailingValidator)

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    try:
        with pytest.raises(HTTPException) as raised:
            tts_service._sanitize_speech_request(_Request(), request_id="req-456")
    finally:
        logger.remove(sink_id)

    assert raised.value.status_code == status.HTTP_400_BAD_REQUEST
    assert raised.value.detail == {
        "message": "TTS validation failed",
        "request_id": "req-456",
    }
    rendered = "\n".join(records)
    _assert_safe_log(rendered)
