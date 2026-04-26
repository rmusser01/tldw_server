from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import audio_batch


@pytest.mark.asyncio
async def test_run_audio_batch_counts_warnings_as_processed(monkeypatch, tmp_path):
    def fake_process_audio_files(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "results": [
                {
                    "status": "Warning",
                    "input_ref": "warn-input",
                    "processing_source": "warn-input",
                    "media_type": "audio",
                    "metadata": {},
                    "content": None,
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "error": None,
                    "warnings": ["warn"],
                },
                {
                    "status": "Success",
                    "input_ref": "ok-input",
                    "processing_source": "ok-input",
                    "media_type": "audio",
                    "metadata": {},
                    "content": "ok",
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "error": None,
                    "warnings": None,
                },
                {
                    "status": "Error",
                    "input_ref": "bad-input",
                    "processing_source": "bad-input",
                    "media_type": "audio",
                    "metadata": {},
                    "content": None,
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "error": "boom",
                    "warnings": None,
                },
            ],
            "processed_count": 1,
            "errors_count": 1,
            "errors": ["boom"],
        }

    monkeypatch.setattr(audio_batch, "process_audio_files", fake_process_audio_files)

    form_data = SimpleNamespace(
        transcription_model=None,
        transcription_language=None,
        perform_chunking=False,
        chunk_method=None,
        chunk_size=None,
        chunk_overlap=None,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        diarize=False,
        vad_use=False,
        timestamp_option=None,
        perform_analysis=False,
        api_name=None,
        custom_prompt=None,
        system_prompt=None,
        summarize_recursively=False,
        use_cookies=False,
        cookies=None,
        title=None,
        author=None,
    )

    batch = await audio_batch.run_audio_batch(
        all_inputs=["warn-input", "ok-input", "bad-input"],
        form_data=form_data,
        temp_dir=str(tmp_path),
        temp_path_to_original_name={},
        saved_files=[],
        file_errors_raw=[],
    )

    assert batch["processed_count"] == 2
    assert batch["errors_count"] == 1
    statuses = [item.get("status") for item in batch["results"]]
    assert "Warning" in statuses
    assert "Success" in statuses
    assert "Error" in statuses


@pytest.mark.asyncio
async def test_run_audio_batch_dedupes_errors_in_stable_order(monkeypatch, tmp_path):
    def fake_process_audio_files(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "results": [
                {
                    "status": "Error",
                    "input_ref": "bad-input",
                    "processing_source": "bad-input",
                    "media_type": "audio",
                    "metadata": {},
                    "content": None,
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "error": "gamma",
                    "warnings": None,
                }
            ],
            "processed_count": 0,
            "errors_count": 1,
            "errors": ["beta", "alpha", "gamma", "beta"],
        }

    monkeypatch.setattr(audio_batch, "process_audio_files", fake_process_audio_files)

    form_data = SimpleNamespace(
        transcription_model=None,
        transcription_language=None,
        perform_chunking=False,
        chunk_method=None,
        chunk_size=None,
        chunk_overlap=None,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        diarize=False,
        vad_use=False,
        timestamp_option=None,
        perform_analysis=False,
        api_name=None,
        custom_prompt=None,
        system_prompt=None,
        summarize_recursively=False,
        use_cookies=False,
        cookies=None,
        title=None,
        author=None,
    )

    batch = await audio_batch.run_audio_batch(
        all_inputs=["bad-input"],
        form_data=form_data,
        temp_dir=str(tmp_path),
        temp_path_to_original_name={},
        saved_files=[],
        file_errors_raw=[
            {"input_ref": "upload-a", "error": "alpha"},
            {"input_ref": "upload-b", "error": "beta"},
        ],
    )

    assert batch["errors"] == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_run_audio_batch_sanitizes_processing_execution_failure(
    monkeypatch,
    tmp_path,
):
    def fake_process_audio_files(**_kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("audio backend exploded at /private/cache/audio.wav")

    monkeypatch.setattr(audio_batch, "process_audio_files", fake_process_audio_files)

    form_data = SimpleNamespace(
        transcription_model=None,
        transcription_language=None,
        perform_chunking=False,
        chunk_method=None,
        chunk_size=None,
        chunk_overlap=None,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        diarize=False,
        vad_use=False,
        timestamp_option=None,
        perform_analysis=False,
        api_name=None,
        custom_prompt=None,
        system_prompt=None,
        summarize_recursively=False,
        use_cookies=False,
        cookies=None,
        title=None,
        author=None,
    )

    batch = await audio_batch.run_audio_batch(
        all_inputs=["audio.wav"],
        form_data=form_data,
        temp_dir=str(tmp_path),
        temp_path_to_original_name={},
        saved_files=[],
        file_errors_raw=[],
    )

    result = batch["results"][0]
    assert batch["errors"] == ["Audio processing failed"]
    assert result["status"] == "Error"
    assert result["error"] == "Audio processing failed"
    assert "audio backend exploded" not in batch["errors"][0]
    assert "/private/cache/audio.wav" not in batch["errors"][0]
