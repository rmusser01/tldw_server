from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import video_batch


@pytest.mark.asyncio
async def test_run_video_batch_sanitizes_processing_execution_failure(
    monkeypatch,
    tmp_path,
):
    def fake_process_videos(**_kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("video backend exploded at /private/cache/video.mp4")

    monkeypatch.setattr(video_batch, "process_videos", fake_process_videos)

    form_data = SimpleNamespace(
        start_time=None,
        end_time=None,
        diarize=False,
        vad_use=False,
        transcription_model=None,
        transcription_language=None,
        perform_analysis=False,
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=False,
        chunk_method=None,
        chunk_size=None,
        chunk_overlap=None,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language=None,
        summarize_recursively=False,
        api_name=None,
        use_cookies=False,
        cookies=None,
        timestamp_option=None,
        perform_confabulation_check_of_analysis=False,
    )

    batch = await video_batch.run_video_batch(
        all_inputs_to_process=["video.mp4"],
        form_data=form_data,
        current_user=SimpleNamespace(id=1),
        temp_dir=str(tmp_path),
        temp_path_to_original_name={},
        file_handling_errors_structured=[],
    )

    result = batch["results"][0]
    assert batch["errors"] == ["Video processing failed"]
    assert result["status"] == "Error"
    assert result["error"] == "Video processing failed"
    assert "RuntimeError" not in batch["errors"][0]
    assert "video backend exploded" not in batch["errors"][0]
    assert "/private/cache/video.mp4" not in batch["errors"][0]
