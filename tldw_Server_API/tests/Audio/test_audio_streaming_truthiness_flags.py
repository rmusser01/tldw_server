import pytest

from tldw_Server_API.app.api.v1.endpoints.audio import audio_streaming
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Streaming_Unified as unified


pytestmark = pytest.mark.unit


def test_audio_ws_compat_error_type_enabled_accepts_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_WS_COMPAT_ERROR_TYPE", "y")

    assert audio_streaming._audio_ws_compat_error_type_enabled() is True


def test_audio_ws_compat_error_type_enabled_false_when_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_WS_COMPAT_ERROR_TYPE", "0")

    assert audio_streaming._audio_ws_compat_error_type_enabled() is False


def test_build_transcript_diagnostics_normalizes_invalid_status_values() -> None:
    payload = unified._build_transcript_diagnostics(
        auto_commit=1,
        vad_status="bogus",
        diarization_status="wrong",
        diarization_details={"code": "persist_disabled", "summary": "x" * 300, "raw": "ignored"},
    )

    assert payload["auto_commit"] is True
    assert payload["vad_status"] == "disabled"
    assert payload["diarization_status"] == "unavailable"
    assert payload["diarization_details"] == {
        "code": "persist_disabled",
        "summary": "x" * 160,
    }


def test_build_transcript_diagnostics_drops_invalid_details_code() -> None:
    payload = unified._build_transcript_diagnostics(
        auto_commit=False,
        vad_status="enabled",
        diarization_status="enabled",
        diarization_details={"code": "not_allowed", "summary": "ignored"},
    )

    assert payload == {
        "auto_commit": False,
        "vad_status": "enabled",
        "diarization_status": "enabled",
    }
