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
