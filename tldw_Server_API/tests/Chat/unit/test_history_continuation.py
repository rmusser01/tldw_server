"""Unit tests for the native Continue helpers (CM-01)."""

import pytest

from tldw_Server_API.app.core.Chat.history_continuation import (
    UNSUPPORTED_CONTINUATION,
    continued_reply_prefix,
    merge_continued_reply,
)
from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (" and more.", "Saved start and more."),
        (None, "Saved start"),
        ([{"type": "text", "text": " tail"}], [{"type": "text", "text": "Saved start tail"}]),
        (
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
            [
                {"type": "text", "text": "Saved start"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        ),
    ],
)
def test_merge_continued_reply_prefixes_the_saved_text(content, expected):
    payload = {"role": "assistant", "content": content, "name": "Assistant"}
    merged = merge_continued_reply(payload, "Saved start")
    assert merged["content"] == expected
    assert merged["name"] == "Assistant"
    assert payload["content"] == content


def test_continued_reply_prefix_returns_plain_saved_text():
    assert continued_reply_prefix({"message": "Saved start", "images": (), "tool_calls": None}) == "Saved start"


@pytest.mark.parametrize(
    "item",
    [
        {"message": "", "images": ()},
        {"message": "   ", "images": ()},
        {"message": "Text", "images": ("data:image/png;base64,AA==",)},
        {"message": "Text", "images": (), "tool_calls": [{"id": "t"}]},
        {"message": "Text", "images": (), "extra_metadata": {"function_call": {"name": "f"}}},
        {"message": "Text", "images": (), "extra_metadata": {"content_placeholder_reason": "tool_calls"}},
    ],
)
def test_continued_reply_prefix_refuses_replies_without_continuable_text(item):
    with pytest.raises(HistorySelectionError) as excinfo:
        continued_reply_prefix(item)
    assert excinfo.value.code == UNSUPPORTED_CONTINUATION
