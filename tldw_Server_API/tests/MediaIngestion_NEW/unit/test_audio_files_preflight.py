import wave
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files


@pytest.mark.unit
def test_process_audio_files_uses_check_transcription_model_status(monkeypatch, tmp_path):
    """process_audio_files should consult check_transcription_model_status and surface warnings."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"\x00" * 2048)

    # Stub speech_to_text so no real STT runs
    def fake_speech_to_text(audio_file_path=None, whisper_model=None, selected_source_lang=None, vad_filter=None, diarize=None, **kwargs):
        return [{"start_seconds": 0, "end_seconds": 0, "Text": "hello"}]

    monkeypatch.setattr(audio_files, "speech_to_text", fake_speech_to_text)

    # Pretend the canonical Whisper model is not yet available locally
    def fake_check_status(model_name: str):
        return {
            "available": False,
            "message": f"Model {model_name} is not available locally",
            "model": model_name,
        }

    monkeypatch.setattr(audio_files, "check_transcription_model_status", fake_check_status)

    # Ensure the model name is parsed as a Whisper model with canonical id "large-v3"
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(atlib, "parse_transcription_model", lambda _: ("whisper", "large-v3", None), raising=True)

    result = audio_files.process_audio_files(
        inputs=[str(audio_path)],
        transcription_model="whisper-large-v3",
        transcription_language="en",
        perform_chunking=False,
        perform_analysis=False,
    )

    assert result["processed_count"] == 1
    item = result["results"][0]
    warnings = item.get("warnings") or []
    assert any("Model large-v3 is not available locally" in w for w in warnings)


@pytest.mark.unit
def test_process_audio_files_sanitizes_item_processing_failure(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(8000)
        wave_file.writeframes(b"\x00\x00" * 8)

    monkeypatch.setattr(
        audio_files,
        "check_transcription_model_status",
        lambda _model_name: {
            "available": True,
            "message": "ok",
            "model": "base",
            "usable": True,
        },
    )

    def fail_speech_to_text(**_kwargs):
        raise RuntimeError("transcriber exploded at /private/cache/audio.wav")

    monkeypatch.setattr(audio_files, "speech_to_text", fail_speech_to_text)

    result = audio_files.process_audio_files(
        inputs=[str(audio_path)],
        transcription_model="base",
        transcription_language="en",
        perform_chunking=False,
        perform_analysis=False,
    )

    item = result["results"][0]
    assert result["errors"] == ["Audio processing failed"]
    assert item["status"] == "Error"
    assert item["error"] == "Audio processing failed"
    assert "transcriber exploded" not in result["errors"][0]
    assert "/private/cache/audio.wav" not in result["errors"][0]


@pytest.mark.unit
def test_process_audio_files_sanitizes_temp_dir_setup_failure(monkeypatch):
    monkeypatch.setattr(
        audio_files.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("tempdir creation exploded at /private/tmp/audio_proc")
        ),
    )

    result = audio_files.process_audio_files(
        inputs=["episode.wav"],
        transcription_model="base",
        transcription_language="en",
        perform_chunking=False,
        perform_analysis=False,
    )

    assert result["processed_count"] == 0
    assert result["errors_count"] == 1
    assert result["errors"] == ["Audio setup failed"]
    assert result["results"][0]["error"] == "Audio setup failed"
    assert "tempdir creation exploded" not in str(result)
    assert "/private/tmp/audio_proc" not in str(result)


@pytest.mark.unit
def test_process_audio_files_sanitizes_fatal_batch_failure(tmp_path):
    class _FailingInputs:
        def __iter__(self):
            raise RuntimeError("input iterator exploded at /private/audio/list.txt")

        def __len__(self):
            return 1

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return "episode.wav"

    result = audio_files.process_audio_files(
        inputs=_FailingInputs(),
        transcription_model="base",
        transcription_language="en",
        perform_chunking=False,
        perform_analysis=False,
        temp_dir=str(tmp_path),
    )

    assert result["processed_count"] == 0
    assert result["errors_count"] == 1
    assert set(result["errors"]) == {"Audio batch processing failed"}
    assert result["results"][0]["error"] == "Audio batch processing failed"
    assert "input iterator exploded" not in str(result)
    assert "/private/audio/list.txt" not in str(result)


@pytest.mark.unit
def test_process_podcast_sanitizes_processing_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        audio_files,
        "download_audio_file",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("podcast downloader exploded at /private/cache/podcast.mp3")
        ),
    )

    result = audio_files.process_podcast(
        url="https://example.com/podcast.mp3",
        perform_chunking=False,
        api_name=None,
        temp_dir=str(tmp_path),
    )

    assert result["status"] == "Error"
    assert result["error"] == "Podcast processing failed"
    assert "podcast downloader exploded" not in str(result)
    assert "/private/cache/podcast.mp3" not in str(result)


@pytest.mark.unit
def test_check_transcription_model_status_marks_uncached_whisper_usable(monkeypatch):
    """Uncached Whisper models should remain request-usable via first-use download."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(atlib, "parse_transcription_model", lambda _: ("whisper", "large-v3", None), raising=True)
    monkeypatch.setattr(atlib, "validate_whisper_model_identifier", lambda value: value, raising=True)
    monkeypatch.setattr(atlib, "check_model_exists", lambda _model_name: False, raising=True)

    status = audio_files.check_transcription_model_status("whisper-large-v3")

    assert status["provider"] == "whisper"
    assert status["model"] == "large-v3"
    assert status["available"] is False
    assert status["usable"] is True
    assert status["on_demand"] is True


@pytest.mark.unit
def test_check_transcription_model_status_hides_validation_exception_details(monkeypatch):
    """Validation failures should not echo filesystem paths or traceback text."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(atlib, "parse_transcription_model", lambda _: ("whisper", "bad-model", None), raising=True)

    def _raise_validation(_value: str) -> str:
        raise ValueError("Traceback: /Users/private/models/model.bin")

    monkeypatch.setattr(atlib, "validate_whisper_model_identifier", _raise_validation, raising=True)

    status = audio_files.check_transcription_model_status("whisper-bad-model")

    assert status["provider"] == "whisper"
    assert status["available"] is False
    assert status["usable"] is False
    assert status["message"] == "Invalid transcription model identifier."
    assert "/Users/private" not in str(status)


@pytest.mark.unit
def test_default_title_from_audio_path_strips_hex_suffix():
    title = audio_files._default_title_from_audio_path("/tmp/My_Clip_ab12cd34.wav")  # nosec B108
    assert title == "My_Clip"


@pytest.mark.unit
def test_default_title_from_audio_path_keeps_non_hex_suffix():
    title = audio_files._default_title_from_audio_path("/tmp/My_Clip_ab12cd3g.wav")  # nosec B108
    assert title == "My_Clip_ab12cd3g"


@pytest.mark.unit
def test_process_audio_files_url_uses_sanitized_default_title(monkeypatch, tmp_path):
    downloaded_wav = tmp_path / "session_ab12cd34.wav"
    with wave.open(str(downloaded_wav), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(8000)
        wave_file.writeframes(b"\x00\x00" * 8)

    monkeypatch.setattr(
        audio_files,
        "download_audio_file",
        lambda *args, **kwargs: str(downloaded_wav),
    )
    monkeypatch.setattr(
        audio_files,
        "check_transcription_model_status",
        lambda _model_name: {
            "available": True,
            "message": "ok",
            "model": "base",
        },
    )

    def fake_speech_to_text(**_kwargs):
        return [{"start_seconds": 0.0, "end_seconds": 1.0, "Text": "hello"}]

    monkeypatch.setattr(audio_files, "speech_to_text", fake_speech_to_text)

    result = audio_files.process_audio_files(
        inputs=["https://example.com/audio.mp3"],
        transcription_model="base",
        transcription_language="en",
        perform_chunking=False,
        perform_analysis=False,
        temp_dir=str(tmp_path),
    )

    assert result["processed_count"] == 1
    item = result["results"][0]
    assert item["status"] == "Success"
    assert item["metadata"]["title"] == "session"


@pytest.mark.unit
def test_process_audio_files_url_post_download_validation_rejected(monkeypatch, tmp_path):
    downloaded_payload = tmp_path / "session_payload.exe"
    downloaded_payload.write_bytes(b"MZ")

    monkeypatch.setattr(
        audio_files,
        "download_audio_file",
        lambda *args, **kwargs: str(downloaded_payload),
    )
    monkeypatch.setattr(
        audio_files,
        "check_transcription_model_status",
        lambda _model_name: {
            "available": True,
            "message": "ok",
            "model": "base",
        },
    )

    def _unexpected_stt(**_kwargs):
        raise AssertionError("speech_to_text should not run when URL validation fails")

    monkeypatch.setattr(audio_files, "speech_to_text", _unexpected_stt)

    result = audio_files.process_audio_files(
        inputs=["https://example.com/audio.mp3"],
        transcription_model="base",
        transcription_language="en",
        perform_chunking=False,
        perform_analysis=False,
        temp_dir=str(tmp_path),
    )

    assert result["processed_count"] == 0
    assert result["errors_count"] == 1
    assert result["results"][0]["status"] == "Error"
    assert "downloaded file failed validation" in str(
        result["results"][0].get("error", "")
    ).lower()


@pytest.mark.unit
def test_process_audio_files_url_rejects_when_downloaded_file_exceeds_quota(monkeypatch, tmp_path):
    downloaded_wav = tmp_path / "session_ab12cd34.wav"
    with wave.open(str(downloaded_wav), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(8000)
        wave_file.writeframes(b"\x00\x00" * 16)

    monkeypatch.setattr(
        audio_files,
        "download_audio_file",
        lambda *args, **kwargs: str(downloaded_wav),
    )
    monkeypatch.setattr(
        audio_files,
        "check_transcription_model_status",
        lambda _model_name: {
            "available": True,
            "message": "ok",
            "model": "base",
        },
    )

    class _RejectingQuotaService:
        async def check_quota(self, user_id: int, new_bytes: int, raise_on_exceed: bool = False):
            assert user_id == 42
            assert new_bytes == downloaded_wav.stat().st_size
            return False, {
                "current_usage_mb": 10,
                "new_size_mb": 1,
                "quota_mb": 10,
                "available_mb": 0,
            }

    fake_quota_module = SimpleNamespace(
        get_storage_quota_service=lambda: _RejectingQuotaService()
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "tldw_Server_API.app.services.storage_quota_service",
        fake_quota_module,
    )

    def _unexpected_stt(**_kwargs):
        raise AssertionError("speech_to_text should not run when quota is exceeded")

    monkeypatch.setattr(audio_files, "speech_to_text", _unexpected_stt)

    result = audio_files.process_audio_files(
        inputs=["https://example.com/audio.wav"],
        transcription_model="base",
        transcription_language="en",
        perform_chunking=False,
        perform_analysis=False,
        temp_dir=str(tmp_path),
        user_id=42,
    )

    assert result["processed_count"] == 0
    assert result["errors_count"] == 1
    assert result["results"][0]["status"] == "Error"
    assert "storage quota exceeded" in str(result["results"][0].get("error", "")).lower()
