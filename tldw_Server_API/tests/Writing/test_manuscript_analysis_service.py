"""Tests for manuscript analysis service (mocked LLM)."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch

MODULE = "tldw_Server_API.app.core.Chat.chat_service"


def _mock_llm_response(content: str):
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
async def test_analyze_pacing_parses_json():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    result_json = json.dumps({
        "pacing": 0.7,
        "tension": 0.5,
        "atmosphere": 0.6,
        "engagement": 0.8,
        "assessment": "Good",
        "beats": ["intro"],
    })
    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(result_json),
    ):
        result = await analyze_pacing("Some text here")
        assert result["pacing"] == 0.7
        assert result["assessment"] == "Good"


@pytest.mark.asyncio
async def test_analyze_pacing_strips_markdown_fences():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response('```json\n{"pacing": 0.5}\n```'),
    ):
        result = await analyze_pacing("Text")
        assert result["pacing"] == 0.5


@pytest.mark.asyncio
async def test_analyze_plot_holes():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_plot_holes

    content = json.dumps({
        "plot_holes": [
            {
                "title": "Gap",
                "description": "Missing",
                "severity": "high",
                "location_hint": "ch2",
            }
        ],
        "inconsistencies": [],
    })
    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(content),
    ):
        result = await analyze_plot_holes("Text", characters="Alice", world_info="Mordor")
        assert len(result["plot_holes"]) == 1
        assert result["plot_holes"][0]["severity"] == "high"


@pytest.mark.asyncio
async def test_analyze_consistency():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_consistency

    content = json.dumps({
        "character_issues": [],
        "world_issues": [],
        "timeline_issues": [],
        "overall_score": 0.95,
    })
    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(content),
    ):
        result = await analyze_consistency("Text")
        assert result["overall_score"] == 0.95


@pytest.mark.asyncio
async def test_handles_llm_error():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        side_effect=RuntimeError("API down"),
    ):
        result = await analyze_pacing("Text")
        assert result == {
            "error": "analysis_failed",
            "message": "Analysis service unavailable",
        }


@pytest.mark.asyncio
async def test_handles_invalid_json():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response("not valid json {"),
    ):
        result = await analyze_pacing("Text")
        assert "error" in result


@pytest.mark.asyncio
async def test_analyze_pacing_passes_provider_and_model():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    result_json = json.dumps({"pacing": 0.5, "tension": 0.3, "atmosphere": 0.4,
                               "engagement": 0.6, "assessment": "Ok", "beats": []})
    mock_fn = AsyncMock(return_value=_mock_llm_response(result_json))
    with patch(f"{MODULE}.perform_chat_api_call_async", mock_fn):
        await analyze_pacing("Text", provider="openai", model="gpt-4o")
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["api_endpoint"] == "openai"
        assert call_kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_extract_content_string_response():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    result_json = json.dumps({"pacing": 0.3})
    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=result_json,  # raw string, not wrapped
    ):
        result = await analyze_pacing("Text")
        assert result["pacing"] == 0.3


@pytest.mark.asyncio
async def test_handles_list_message_content_fragments():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    with patch(
        f"{MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value={
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"pacing": 0.4, '},
                            {"type": "text", "text": '"tension": 0.2, '},
                            {
                                "type": "text",
                                "text": '"assessment": "Strong", "beats": []}',
                            },
                        ]
                    }
                }
            ]
        },
    ):
        result = await analyze_pacing("Text")
        assert result["pacing"] == 0.4
        assert result["tension"] == 0.2
        assert result["assessment"] == "Strong"


def test_extract_content_dict_message_content_block():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import _extract_content

    response = {
        "choices": [
            {"message": {"content": {"type": "text", "text": '{"pacing": 0.9}'}}}
        ]
    }

    assert _extract_content(response) == '{"pacing": 0.9}'


@pytest.mark.asyncio
async def test_strip_markdown_fences_no_fences():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import _strip_markdown_fences

    assert _strip_markdown_fences('{"a": 1}') == '{"a": 1}'


@pytest.mark.asyncio
async def test_strip_markdown_fences_json_prefix():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import _strip_markdown_fences

    assert json.loads(_strip_markdown_fences('```json\n{"a": 1}\n```')) == {"a": 1}
