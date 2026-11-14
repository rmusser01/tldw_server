"""Allow-list for generation metadata on settled assistant replies (D7 P2)."""

import pytest

from tldw_Server_API.app.core.Chat.generation_metadata import (
    GENERATION_METADATA_KEYS,
    GENERATION_STATUSES,
    GenerationMetadataError,
    build_generation_metadata,
    sanitize_generation_metadata,
    validate_generation_metadata,
)

pytestmark = pytest.mark.unit


def test_allow_list_names_the_settlement_keys_and_status_vocabulary():
    assert set(GENERATION_METADATA_KEYS) == {"model_id", "provider", "finish_reason", "usage", "generation_status"}
    assert set(GENERATION_STATUSES) == {"complete", "stopped", "interrupted", "length", "error"}


def test_sanitize_strips_unknown_keys_and_unknown_usage_fields():
    cleaned = sanitize_generation_metadata(
        {
            "model_id": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "generation_status": "complete",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "cost_usd": 9},
            "api_key": "sk-secret",
            "sender_role": "assistant",
        }
    )
    assert cleaned == {
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "generation_status": "complete",
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


@pytest.mark.parametrize("status", ["done", 1, None, "Complete"])
def test_sanitize_drops_status_outside_the_vocabulary(status):
    assert sanitize_generation_metadata({"generation_status": status, "provider": "openai"}) == {
        "provider": "openai"
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_id", ""),
        ("model_id", "x" * 257),
        ("model_id", "bad\nmodel"),
        ("provider", "open ai"),
        ("provider", None),
        ("finish_reason", "stop now"),
        ("usage", {"prompt_tokens": -1}),
        ("usage", {"prompt_tokens": True}),
        ("usage", {"prompt_tokens": 1.5}),
        ("usage", []),
        ("usage", {}),
    ],
)
def test_sanitize_drops_invalid_values(key, value):
    assert sanitize_generation_metadata({key: value, "generation_status": "interrupted"}) == {
        "generation_status": "interrupted"
    }


def test_sanitize_rejects_non_mapping_input():
    assert sanitize_generation_metadata(None) == {}
    assert sanitize_generation_metadata(["generation_status", "complete"]) == {}


def test_validate_rejects_unknown_keys_and_invalid_values():
    with pytest.raises(GenerationMetadataError):
        validate_generation_metadata({"generation_status": "complete", "colour": "blue"})
    with pytest.raises(GenerationMetadataError):
        validate_generation_metadata({"generation_status": "unknown"})
    with pytest.raises(GenerationMetadataError):
        validate_generation_metadata({"usage": {"prompt_tokens": 1, "reasoning_tokens": 2}})


def test_validate_returns_normalized_copy():
    raw = {"model_id": "  llama-3.1-8b  ", "generation_status": "length", "usage": {"total_tokens": 12}}
    assert validate_generation_metadata(raw) == {
        "model_id": "llama-3.1-8b",
        "generation_status": "length",
        "usage": {"total_tokens": 12},
    }


def test_build_omits_unknown_or_missing_values():
    assert build_generation_metadata(
        generation_status="interrupted",
        model_id="gpt-4o-mini",
        provider="openai",
        finish_reason=None,
        usage=None,
    ) == {"generation_status": "interrupted", "model_id": "gpt-4o-mini", "provider": "openai"}


def test_build_rejects_status_outside_the_vocabulary():
    with pytest.raises(GenerationMetadataError):
        build_generation_metadata(generation_status="cancelled")
