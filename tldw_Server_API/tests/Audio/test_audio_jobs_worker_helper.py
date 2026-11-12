import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_audio_transcribe_stage_uses_normalized_artifact(monkeypatch, tmp_path):
    """
    The audio_jobs_worker transcribe helper should populate segments/text from
    the normalized STT artifact and keep the rest of the artifact (not its text and segments again) as normalized_stt.
    """
    import tldw_Server_API.app.services.audio_jobs_worker as worker
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"\x00\x00")

    fake_segments = [
        {"Text": "hello", "start_seconds": 0.0, "end_seconds": 0.5},
        {"Text": "worker", "start_seconds": 0.5, "end_seconds": 1.0},
    ]
    fake_artifact = {
        "text": "hello worker",
        "language": "en",
        "segments": fake_segments,
        "diarization": {"enabled": False, "speakers": None},
        "usage": {"duration_ms": 1000, "tokens": None},
        "metadata": {"provider": "faster-whisper", "model": "large-v3"},
    }

    def _fake_run_stt_job_via_registry(path, model, language, hotwords=None, base_dir=None, **kwargs):
        assert str(path) == str(wav_path)
        assert hotwords is None
        assert base_dir is None
        return fake_artifact

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _fake_run_stt_job_via_registry, raising=True)

    payload = {
        "wav_path": str(wav_path),
        "model": "whisper-1",
        "perform_chunking": True,
        "perform_analysis": False,
    }

    updated_payload, next_type = await worker._handle_audio_transcribe_stage(dict(payload))

    assert next_type == "audio_chunk"
    assert updated_payload["segments"] == fake_segments
    assert updated_payload["text"] == "hello worker"
    assert updated_payload.get("normalized_stt") == {
        k: v for k, v in fake_artifact.items() if k not in ("text", "segments")
    }
