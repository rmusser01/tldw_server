import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_validation import TTSInputValidator


pytestmark = pytest.mark.unit


def _make_reference_wav(duration_seconds: float, *, sample_rate: int = 24000) -> bytes:
    import wave
    from io import BytesIO

    frame_count = max(1, int(duration_seconds * sample_rate))
    payload = b"\x00\x00" * frame_count
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)
    return buffer.getvalue()


def test_omnivoice_clone_requires_reference_text():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.WAV,
        voice_reference=b"RIFF" + b"\x00" * 64,
        extra_params={},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "reference_text" in str(error)


def test_omnivoice_clone_accepts_reference_text():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.WAV,
        voice_reference=_make_reference_wav(3.5),
        extra_params={"reference_text": "reference transcript"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is True
    assert error is None


def test_omnivoice_clone_requires_reference_audio():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.WAV,
        extra_params={"reference_text": "reference transcript"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "voice_reference" in str(error)


def test_omnivoice_clone_rejects_too_short_reference_audio():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.WAV,
        voice_reference=_make_reference_wav(1.0),
        extra_params={"reference_text": "reference transcript"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "too short" in str(error).lower()


def test_omnivoice_clone_rejects_too_long_reference_audio():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.WAV,
        voice_reference=_make_reference_wav(31.0),
        extra_params={"reference_text": "reference transcript"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "too long" in str(error).lower()


def test_omnivoice_custom_voice_requires_resolved_reference_audio():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="custom:voice-1",
        format=AudioFormat.WAV,
        extra_params={"reference_text": "reference transcript"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "custom" in str(error).lower()
    assert "voice_reference" in str(error)


def test_omnivoice_custom_voice_rejects_too_short_reference_audio():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="custom:voice-1",
        format=AudioFormat.WAV,
        voice_reference=_make_reference_wav(1.0),
        extra_params={"reference_text": "reference transcript"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "too short" in str(error).lower()


def test_omnivoice_custom_voice_accepts_resolved_reference_audio():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="custom:voice-1",
        format=AudioFormat.WAV,
        voice_reference=_make_reference_wav(3.5),
        extra_params={"reference_text": "reference transcript"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is True
    assert error is None


def test_omnivoice_allows_non_english_language_passthrough():
    validator = TTSInputValidator()
    request = TTSRequest(text="hola", voice="auto", language="es", format=AudioFormat.WAV)

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is True
    assert error is None


def test_omnivoice_rejects_unknown_generation_param():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="auto",
        format=AudioFormat.WAV,
        extra_params={"omnivoice_unknown_knob": 1},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "generation" in str(error).lower() or "unknown" in str(error).lower()


@pytest.mark.parametrize(
    ("extra_params", "voice", "voice_reference", "expected_error"),
    [
        ({"mode": "auto", "instruct": "calm"}, "auto", None, "auto"),
        ({"mode": "auto", "reference_text": "reference transcript"}, "clone", _make_reference_wav(3.5), "auto"),
        ({"mode": "design"}, "auto", None, "instruct"),
        (
            {"mode": "design", "instruct": "calm", "reference_text": "reference transcript"},
            "clone",
            _make_reference_wav(3.5),
            "design",
        ),
        ({"mode": "clone", "instruct": "calm", "reference_text": "reference transcript"}, "clone", _make_reference_wav(3.5), "clone"),
        ({"mode": "clone"}, "auto", None, "reference"),
    ],
)
def test_omnivoice_rejects_mode_conflicts(extra_params, voice, voice_reference, expected_error):
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice=voice,
        format=AudioFormat.WAV,
        voice_reference=voice_reference,
        extra_params=extra_params,
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert expected_error in str(error).lower()


def test_omnivoice_language_alias_conflict_rejected():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hola",
        voice="auto",
        language="es",
        format=AudioFormat.WAV,
        extra_params={"language_id": "fr"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "language" in str(error).lower()


def test_omnivoice_matching_language_aliases_pass():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hola",
        voice="auto",
        language="es",
        format=AudioFormat.WAV,
        extra_params={"language_id": "es", "language": "es"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is True
    assert error is None


def test_omnivoice_rejects_invalid_bool_generation_param():
    validator = TTSInputValidator()
    request = TTSRequest(
        text="hello",
        voice="auto",
        format=AudioFormat.WAV,
        extra_params={"denoise": "maybe"},
    )

    is_valid, error = validator.validate_request(request, provider="omnivoice")

    assert is_valid is False
    assert "denoise" in str(error)
