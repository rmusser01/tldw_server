import pytest

import tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming as audio_streaming


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_status_shape():
    """
    Verify the audio stream status endpoint returns the expected JSON structure and values.

    Asserts that /api/v1/audio/stream/status returns a JSON object containing:
    - a "status" key with value "available" or "unavailable";
    - an "available_models" list of strings whose entries start with a known model prefix;
    - a "websocket_endpoint" equal to "/api/v1/audio/stream/transcribe";
    - a "supported_features" dictionary containing feature flags: "partial_results", "multiple_languages", "concurrent_streams", "segment_metadata", "live_insights", "meeting_notes", "speaker_diarization", and "audio_persistence".
    """
    data = await audio_streaming.streaming_status()

    # Basic top-level keys
    assert isinstance(data, dict)
    assert "status" in data
    assert data["status"] in {"available", "unavailable"}
    assert "available_models" in data
    assert isinstance(data["available_models"], list)
    assert all(isinstance(m, str) for m in data["available_models"])

    # Endpoint path advertised
    assert data.get("websocket_endpoint") == "/api/v1/audio/stream/transcribe"

    # Feature flags present
    sf = data.get("supported_features")
    assert isinstance(sf, dict)
    for key in (
        "partial_results",
        "multiple_languages",
        "concurrent_streams",
        "segment_metadata",
        "live_insights",
        "meeting_notes",
        "speaker_diarization",
        "audio_persistence",
    ):
        assert key in sf

    # If models are advertised, they should start with a known base model
    for m in data["available_models"]:
        assert m.startswith("parakeet"), f"Unknown model: {m}"
