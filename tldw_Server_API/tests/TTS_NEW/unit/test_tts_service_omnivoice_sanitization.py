from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audio import tts_service


@pytest.mark.unit
def test_sanitize_speech_request_omnivoice_omitted_voice_ignores_configured_default(monkeypatch) -> None:
    request = SimpleNamespace(
        model="omnivoice",
        input="Hello OmniVoice",
        voice="af_heart",
        model_fields_set={"model", "input"},
    )

    monkeypatch.setattr(
        tts_service,
        "get_tts_config",
        lambda: SimpleNamespace(
            strict_validation=True,
            default_provider="omnivoice",
            default_voice="configured-non-auto",
        ),
        raising=True,
    )

    class _FakeValidator:
        def __init__(self, _config):
            pass

        def sanitize_text(self, text, provider=None):  # noqa: ARG002
            return text

    monkeypatch.setattr(tts_service, "TTSInputValidator", _FakeValidator, raising=True)

    provider_hint = tts_service._sanitize_speech_request(request, request_id="req-1")

    assert provider_hint == "omnivoice"
    assert request.voice == "auto"


@pytest.mark.unit
def test_sanitize_speech_request_omnivoice_preserves_explicit_voice(monkeypatch) -> None:
    request = SimpleNamespace(
        model="omnivoice",
        input="Hello OmniVoice",
        voice="speaker-a",
        model_fields_set={"model", "input", "voice"},
    )

    monkeypatch.setattr(
        tts_service,
        "get_tts_config",
        lambda: SimpleNamespace(
            strict_validation=True,
            default_provider="omnivoice",
            default_voice="configured-non-auto",
        ),
        raising=True,
    )

    class _FakeValidator:
        def __init__(self, _config):
            pass

        def sanitize_text(self, text, provider=None):  # noqa: ARG002
            return text

    monkeypatch.setattr(tts_service, "TTSInputValidator", _FakeValidator, raising=True)

    provider_hint = tts_service._sanitize_speech_request(request, request_id="req-2")

    assert provider_hint == "omnivoice"
    assert request.voice == "speaker-a"


@pytest.mark.unit
def test_sanitize_speech_request_omnivoice_preserves_explicit_voice_with_pydantic_v1_fields(monkeypatch) -> None:
    request = SimpleNamespace(
        model="omnivoice",
        input="Hello OmniVoice",
        voice="custom:voice-1",
        __fields_set__={"model", "input", "voice"},
    )

    monkeypatch.setattr(
        tts_service,
        "get_tts_config",
        lambda: SimpleNamespace(
            strict_validation=True,
            default_provider="omnivoice",
            default_voice="configured-non-auto",
        ),
        raising=True,
    )

    class _FakeValidator:
        def __init__(self, _config):
            pass

        def sanitize_text(self, text, provider=None):  # noqa: ARG002
            return text

    monkeypatch.setattr(tts_service, "TTSInputValidator", _FakeValidator, raising=True)

    provider_hint = tts_service._sanitize_speech_request(request, request_id="req-3")

    assert provider_hint == "omnivoice"
    assert request.voice == "custom:voice-1"
