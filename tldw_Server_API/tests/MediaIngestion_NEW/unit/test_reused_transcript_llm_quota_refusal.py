"""A media item that reuses a transcript stops summarizing at the first LLM quota refusal (TASK-13638)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_reused_transcript_stops_at_the_first_refusal_and_keeps_the_transcript() -> None:
    analyze = MagicMock(
        side_effect=LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )
    )
    transcript = " ".join(f"Sentence number {i} is here." for i in range(400))  # 2000 words: 20 chunks of 100
    result = {"content": transcript}
    form = SimpleNamespace(
        api_name="openai",
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=True,
        summarize_recursively=False,
        perform_analysis=True,
    )

    with patch("tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze", analyze):
        await persistence._populate_reused_transcript_summary(
            process_result=result,
            form_data=form,
            chunk_options={"method": "words", "max_size": 100, "overlap": 0},
            loop=asyncio.get_running_loop(),
        )

    assert analyze.call_count == 1
    assert result["content"] == transcript
    assert not result.get("summary")
    assert result["video_lite_summary_status"] == "failed"
    assert result["video_lite_summary_error"] == "llm_quota_exceeded"
    assert persistence._merge_video_lite_summary_safe_metadata({}, process_result=result)["video_lite"] == {
        "summary_status": "failed",
        "summary_error": "llm_quota_exceeded",
    }
    assert any("limit_exceeded" in warning for warning in result["warnings"])
