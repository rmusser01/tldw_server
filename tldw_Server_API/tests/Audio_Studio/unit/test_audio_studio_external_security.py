"""Unit tests for Audio Studio external provider security helpers."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Audio_Studio.security import (
    redact_audio_studio_secret,
    validate_external_audio_endpoint,
)


pytestmark = pytest.mark.unit


def test_validate_external_endpoint_requires_exact_allowlist_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST",
        "https://ace.example.test,https://music.example.test:8443",
    )

    assert validate_external_audio_endpoint("https://ace.example.test/v1/generate") == (
        "https",
        "ace.example.test",
        443,
    )
    assert validate_external_audio_endpoint("https://music.example.test:8443/api") == (
        "https",
        "music.example.test",
        8443,
    )

    with pytest.raises(ValueError, match="external_audio_endpoint_not_allowlisted"):
        validate_external_audio_endpoint("https://ace.example.test:444/v1/generate")
    with pytest.raises(ValueError, match="external_audio_endpoint_not_allowlisted"):
        validate_external_audio_endpoint("https://evil.example.test/v1/generate")


def test_validate_external_endpoint_rejects_http_unless_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "http://127.0.0.1:7865")
    monkeypatch.delenv("AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS", raising=False)

    with pytest.raises(ValueError, match="external_audio_endpoint_requires_https"):
        validate_external_audio_endpoint("http://127.0.0.1:7865/generate")

    monkeypatch.setenv("AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS", "1")
    assert validate_external_audio_endpoint("http://127.0.0.1:7865/generate") == (
        "http",
        "127.0.0.1",
        7865,
    )


def test_validate_redirect_target_uses_same_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "https://ace.example.test")

    assert validate_external_audio_endpoint(
        "https://ace.example.test/redirected",
        redirect_from="https://ace.example.test/generate",
    ) == ("https", "ace.example.test", 443)

    with pytest.raises(ValueError, match="external_audio_redirect_not_allowlisted"):
        validate_external_audio_endpoint(
            "https://evil.example.test/capture",
            redirect_from="https://ace.example.test/generate",
        )


def test_redact_audio_studio_secret_masks_known_secret_values() -> None:
    message = redact_audio_studio_secret(
        "using key sk-live-abc123 and token secret-token",
        secrets=["sk-live-abc123", "secret-token"],
    )

    assert "sk-live-abc123" not in message
    assert "secret-token" not in message
    assert message == "using key [REDACTED] and token [REDACTED]"
