"""The SSE helpers are byte-for-byte the inline frames they replaced (TASK-13332)."""

import json

import pytest

from tldw_Server_API.app.core.LLM_Calls.sse import sse_data, sse_done, sse_event

PAYLOADS = [
    {},
    {"choices": [{"delta": {"content": "hi"}}]},
    {"error": {"code": "x", "type": "x", "message": "café ☃ \"q\"\nline"}},
    {"n": 1.5, "b": True, "none": None, "list": [1, "two"]},
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_sse_data_matches_inline_frame(payload):
    assert sse_data(payload) == f"data: {json.dumps(payload)}\n\n"


@pytest.mark.parametrize("payload", PAYLOADS)
def test_sse_event_matches_inline_frame(payload):
    assert sse_event("stream_end", payload) == f"event: stream_end\ndata: {json.dumps(payload)}\n\n"


def test_sse_done_matches_inline_sentinel():
    assert sse_done() == "data: [DONE]\n\n"


def test_chat_stream_control_prefixes_derive_from_sse_fields():
    """streaming_utils splits the same SSE field list; it must not drift from sse.py."""
    from tldw_Server_API.app.core.Chat import streaming_utils
    from tldw_Server_API.app.core.LLM_Calls.sse import SSE_CONTROL_FIELD_PREFIXES

    always = set(streaming_utils._ALWAYS_SSE_CONTROL_PREFIXES)
    framed_only = set(streaming_utils._FRAMED_ONLY_SSE_CONTROL_PREFIXES)
    assert always | framed_only == {":"} | set(SSE_CONTROL_FIELD_PREFIXES)
    assert framed_only == {"id:", "retry:"}
