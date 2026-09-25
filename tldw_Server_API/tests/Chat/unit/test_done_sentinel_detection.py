"""One provider-DONE detector, case-insensitive everywhere (TASK-13332, ADR-025)."""

import pytest

from tldw_Server_API.app.core.Chat.streaming_utils import (
    StreamingResponseHandler,
    _extract_text_from_upstream_sse,
)
from tldw_Server_API.app.core.LLM_Calls.sse import is_done_line


@pytest.mark.parametrize(
    "line",
    ["data: [DONE]", "data: [DONE]\n\n", "  DATA: [done] ", "data:[DONE]", "data:   [Done]", "﻿data: [DONE]"],
)
def test_is_done_line_accepts_every_done_spelling(line):
    assert is_done_line(line) is True


@pytest.mark.parametrize("line", ["", "[DONE]", "data: [DONE]x", "data: DONE", "event: [DONE]", 'data: "[DONE]"'])
def test_is_done_line_rejects_non_done(line):
    assert is_done_line(line) is False


def test_extract_text_treats_lowercase_done_as_done():
    assert _extract_text_from_upstream_sse("data: [done]\n\n") == (None, None, True)


async def test_lowercase_provider_done_is_suppressed_not_forwarded():
    handler = StreamingResponseHandler("conv_done_case", "gpt-4")

    async def provider_stream():
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [done]\n\n"

    messages = [m async for m in handler.safe_stream_generator(provider_stream())]

    done_frames = [m for m in messages if is_done_line(m)]
    assert done_frames == ["data: [DONE]\n\n"]
    assert handler.error_occurred is False


# TASK-13370: the remaining hand-rolled DONE checks route through is_done_line.
_DONE_SPELLINGS = ["data: [done]", "data:[DONE]", "data:  [Done]"]


@pytest.mark.parametrize("line", _DONE_SPELLINGS)
def test_anthropic_messages_parser_treats_any_done_spelling_as_done(line):
    from tldw_Server_API.app.core.LLM_Calls.anthropic_messages import _parse_openai_sse_line

    assert _parse_openai_sse_line(line) == {"_done": True}


@pytest.mark.parametrize("line", _DONE_SPELLINGS)
def test_chat_endpoint_inspector_marks_any_done_spelling_complete(line):
    from tldw_Server_API.app.api.v1.endpoints.chat import _inspect_provider_stream_chunk

    assert _inspect_provider_stream_chunk(f"{line}\n\n") == (None, False, True)


@pytest.mark.parametrize("line", _DONE_SPELLINGS)
def test_chat_documents_classifier_treats_any_done_spelling_as_done(line):
    from tldw_Server_API.app.api.v1.endpoints.chat_documents import _classify_document_stream_chunk

    assert _classify_document_stream_chunk(line) == ("[DONE]", False, "done")


@pytest.mark.parametrize("line", _DONE_SPELLINGS)
def test_audio_chat_stream_done_is_not_content(line):
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming import (
        _audio_provider_chunk_has_nonempty_content,
    )

    assert _audio_provider_chunk_has_nonempty_content(line) is False


@pytest.mark.parametrize("line", _DONE_SPELLINGS)
def test_character_stream_done_is_not_semantic_output(line):
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
        _character_stream_line_has_semantic_output,
    )

    assert _character_stream_line_has_semantic_output(line) is False


@pytest.mark.parametrize("line", _DONE_SPELLINGS)
def test_rag_stream_text_ignores_any_done_spelling(line):
    from tldw_Server_API.app.core.RAG.rag_service.generation import _extract_stream_text

    assert _extract_stream_text(line) is None


@pytest.mark.parametrize("line", _DONE_SPELLINGS)
def test_realtime_pipeline_ignores_any_done_spelling(line):
    from tldw_Server_API.app.core.Audio.Realtime.default_pipeline import _extract_text_from_string_chunk

    assert _extract_text_from_string_chunk(line) == ""
