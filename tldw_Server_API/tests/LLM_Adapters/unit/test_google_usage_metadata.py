"""Gemini usageMetadata is costed in full: thinking tokens, tool-use prompt tokens, the service tier, and the
stream's usage chunk (TASK-13952)."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage

pytestmark = pytest.mark.unit

# The repo fixture (tests/Chat/unit/test_chat_service_content.py): totalTokenCount 115 = 80 prompt + 30 candidates
# + 5 thoughts; 50 of the prompt tokens were cached.
USAGE_METADATA = {
    "promptTokenCount": 80,
    "candidatesTokenCount": 30,
    "totalTokenCount": 115,
    "cachedContentTokenCount": 50,
    "thoughtsTokenCount": 5,
}
EXPECTED_USAGE = {
    "prompt_tokens": 80,
    "completion_tokens": 35,
    "total_tokens": 115,
    "thoughtsTokenCount": 5,
    "cachedContentTokenCount": 50,
}


def _reply(usage_metadata: dict[str, Any]) -> dict[str, Any]:
    """A generateContent response with one text candidate and ``usage_metadata``."""
    return {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": usage_metadata,
    }


def test_thinking_tokens_are_counted_as_output() -> None:
    """Google bills thoughtsTokenCount as output, apart from candidatesTokenCount (AC #1); the block keeps the
    thinking share under its own name for the row's reasoning_tokens, and total_tokens is Gemini's total."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply(USAGE_METADATA))["usage"]

    assert usage == EXPECTED_USAGE


def test_the_normalized_block_is_not_double_counted() -> None:
    """The usage normalizer reads completion_tokens (thinking included) and reports the thinking share apart."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply(USAGE_METADATA))["usage"]

    normalized = normalize_llm_usage(provider="google", usage=usage)

    assert (normalized.input_tokens, normalized.output_tokens, normalized.total_tokens) == (80, 35, 115)
    assert (normalized.reasoning_tokens, normalized.cache_read_input_tokens) == (5, 50)


def test_tool_use_prompt_tokens_are_counted_as_input() -> None:
    """toolUsePromptTokenCount is outside promptTokenCount (Gemini's total = prompt + tool-use prompt + thoughts +
    candidates), so it joins prompt_tokens and the block's sides add up to its total."""
    usage = GoogleAdapter._normalize_to_openai_shape(
        _reply({**USAGE_METADATA, "toolUsePromptTokenCount": 7, "totalTokenCount": 122})
    )["usage"]

    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (87, 35, 122)
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def test_a_reply_without_thinking_or_cache_is_unchanged() -> None:
    """A plain usageMetadata gives the plain block, a missing count staying absent."""
    usage = GoogleAdapter._normalize_to_openai_shape(
        _reply({"promptTokenCount": 10, "candidatesTokenCount": 5})
    )["usage"]

    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": None}


@pytest.mark.parametrize(
    "usage_metadata",
    ["garbage", None, {"candidatesTokenCount": "30", "thoughtsTokenCount": True, "serviceTier": 3}],
    ids=["string", "missing", "wrong-types"],
)
def test_a_malformed_usage_metadata_never_raises(usage_metadata: Any) -> None:
    """A count that is not an int is left out of the sums; the reply is still normalized (not returned raw)."""
    shaped = GoogleAdapter._normalize_to_openai_shape({**_reply({}), "usageMetadata": usage_metadata})

    assert shaped["object"] == "chat.completion"
    assert "service_tier" not in shaped["usage"] and "thoughtsTokenCount" not in shaped["usage"]


@pytest.mark.parametrize(
    ("reported", "expected"),
    [("flex", "flex"), ("priority", "priority"), ("SERVICE_TIER_PRIORITY", "priority"), ("unspecified", "unspecified")],
)
def test_the_service_tier_is_passed_on(reported: str, expected: str) -> None:
    """UsageMetadata.serviceTier (unspecified, standard, flex, priority) is kept as service_tier, lowercased and
    without an enum prefix, so the catalog's flex and priority rates apply (AC #3)."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply({**USAGE_METADATA, "serviceTier": reported}))["usage"]

    assert usage["service_tier"] == expected
