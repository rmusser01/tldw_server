from __future__ import annotations

import json

from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage


def test_normalize_openai_usage_extracts_cache_read_and_reasoning_tokens() -> None:
    normalized = normalize_llm_usage(
        provider="openai",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 6},
        },
        choices=[{"index": 0}, {"index": 1}],
    )

    assert normalized.input_tokens == 100
    assert normalized.output_tokens == 20
    assert normalized.total_tokens == 120
    assert normalized.cached_input_tokens == 40
    assert normalized.cache_read_input_tokens == 40
    assert normalized.cache_write_input_tokens == 0
    assert normalized.billable_input_tokens == 60
    assert normalized.reasoning_tokens == 6
    assert normalized.choice_count == 2
    assert normalized.estimate_source == "provider_usage"


def test_normalize_anthropic_usage_separates_cache_write_read_and_normal_input() -> None:
    normalized = normalize_llm_usage(
        provider="anthropic",
        usage={
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 70,
        },
    )

    assert normalized.input_tokens == 100
    assert normalized.output_tokens == 25
    assert normalized.total_tokens == 125
    assert normalized.cached_input_tokens == 70
    assert normalized.cache_read_input_tokens == 70
    assert normalized.cache_write_input_tokens == 10
    assert normalized.billable_input_tokens == 20


def test_normalize_gemini_usage_extracts_cached_content_and_thought_tokens() -> None:
    normalized = normalize_llm_usage(
        provider="google",
        usage={
            "promptTokenCount": 80,
            "candidatesTokenCount": 30,
            "totalTokenCount": 115,
            "cachedContentTokenCount": 50,
            "thoughtsTokenCount": 5,
        },
    )

    assert normalized.input_tokens == 80
    assert normalized.output_tokens == 30
    assert normalized.total_tokens == 115
    assert normalized.cached_input_tokens == 50
    assert normalized.cache_read_input_tokens == 50
    assert normalized.reasoning_tokens == 5
    assert normalized.billable_input_tokens == 30


def test_normalize_openrouter_and_local_openai_compatible_usage() -> None:
    openrouter = normalize_llm_usage(
        provider="openrouter",
        usage={
            "prompt_tokens": 90,
            "completion_tokens": 10,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 20},
            "provider": "anthropic",
        },
    )
    openrouter_anthropic = normalize_llm_usage(
        provider="openrouter",
        usage={
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 70,
            "provider": "anthropic",
        },
    )
    assert openrouter.input_tokens == 90
    assert openrouter.output_tokens == 10
    assert openrouter.cache_read_input_tokens == 20
    assert openrouter.billable_input_tokens == 70
    assert openrouter.raw_usage_metadata["provider"] == "anthropic"
    assert openrouter_anthropic.cached_input_tokens == 70
    assert openrouter_anthropic.cache_write_input_tokens == 10
    assert openrouter_anthropic.billable_input_tokens == 20
    for provider in ("llama.cpp", "vllm"):
        local = normalize_llm_usage(
            provider=provider,
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        )
        assert local.input_tokens == 12
        assert local.output_tokens == 3
        assert local.total_tokens == 15
        assert local.cached_input_tokens == 0
        assert local.billable_input_tokens == 12


def test_normalize_bounds_and_redacts_raw_usage_metadata() -> None:
    normalized = normalize_llm_usage(
        provider="custom",
        usage={
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "api_key": "sk-secret",
            "headers": {"authorization": "Bearer token"},
            "messages": [{"role": "user", "content": "private prompt"}],
            "content": "private output",
            "debug_prompt": "private debug prompt",
            "huge": "x" * 5000,
        },
    )

    serialized = json.dumps(normalized.raw_usage_metadata, sort_keys=True)
    assert len(serialized) <= 4096
    assert "sk-secret" not in serialized
    assert "Bearer token" not in serialized
    assert "private prompt" not in serialized
    assert "private output" not in serialized
    assert "private debug prompt" not in serialized
    assert normalized.raw_usage_metadata["api_key"] == "[redacted]"
    assert normalized.raw_usage_metadata["messages"] == "[redacted]"
    assert normalized.raw_usage_metadata["content"] == "[redacted]"
    assert normalized.raw_usage_metadata["debug_prompt"] == "[redacted]"


def test_raw_usage_metadata_preserves_nested_numeric_prompt_like_usage_counters() -> None:
    normalized = normalize_llm_usage(
        provider="custom",
        usage={
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "input": {
                "text_tokens": 12,
                "audio_tokens": "4",
                "unit": "tokens",
                "content": "private source prompt",
            },
            "text": {
                "tokens": 16,
                "characters": 128,
                "sample": "private text sample",
            },
        },
    )

    metadata = normalized.raw_usage_metadata
    serialized = json.dumps(metadata, sort_keys=True)
    assert metadata["input"]["text_tokens"] == 12
    assert metadata["input"]["audio_tokens"] == "4"
    assert metadata["input"]["unit"] == "tokens"
    assert metadata["input"]["content"] == "[redacted]"
    assert metadata["text"]["tokens"] == 16
    assert metadata["text"]["characters"] == 128
    assert metadata["text"]["sample"] == "[redacted]"
    assert "private source prompt" not in serialized
    assert "private text sample" not in serialized


def test_normalize_missing_and_stream_estimated_usage_sources() -> None:
    stream_estimate = normalize_llm_usage(
        provider="openai",
        usage=None,
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
        estimate_source="stream_estimate",
    )
    disconnect_estimate = normalize_llm_usage(
        provider="openai",
        usage=None,
        prompt_tokens=5,
        completion_tokens=0,
        total_tokens=5,
        estimate_source="disconnect_estimate",
    )
    missing_usage = normalize_llm_usage(provider="openai", usage=None)

    assert stream_estimate.input_tokens == 5
    assert stream_estimate.output_tokens == 2
    assert stream_estimate.total_tokens == 7
    assert stream_estimate.estimate_source == "stream_estimate"
    assert disconnect_estimate.estimate_source == "disconnect_estimate"
    assert missing_usage.input_tokens == 0
    assert missing_usage.output_tokens == 0
    assert missing_usage.total_tokens == 0
    assert missing_usage.estimate_source == "missing_usage"
