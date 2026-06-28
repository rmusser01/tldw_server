import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError
from tldw_Server_API.app.core.TTS.tts_validation import validate_tts_request


@pytest.mark.unit
def test_fish_s2_accepts_hosted_api_opus_format():
    request = TTSRequest(
        text="hello",
        provider="fish_s2",
        format=AudioFormat.OPUS,
        stream=True,
        extra_params={"reference_id": "voice-123"},
    )

    validate_tts_request(request, provider="fish_s2")


@pytest.mark.unit
def test_fish_s2_rejects_unsupported_format():
    request = TTSRequest(
        text="hello",
        provider="fish_s2",
        format=AudioFormat.FLAC,
        stream=False,
    )

    with pytest.raises(TTSValidationError):
        validate_tts_request(request, provider="fish_s2")
