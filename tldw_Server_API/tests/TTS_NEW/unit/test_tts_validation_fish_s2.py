import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError
from tldw_Server_API.app.core.TTS.tts_validation import validate_tts_request


@pytest.mark.unit
def test_fish_s2_streaming_requires_wav():
    request = TTSRequest(
        text="hello",
        provider="fish_s2",
        format=AudioFormat.MP3,
        stream=True,
        extra_params={"reference_id": "voice-123"},
    )

    with pytest.raises(TTSValidationError) as exc:
        validate_tts_request(request, provider="fish_s2")

    assert "wav" in str(exc.value).lower()
