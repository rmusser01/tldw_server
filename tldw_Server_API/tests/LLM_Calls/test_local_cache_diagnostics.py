import pytest

from tldw_Server_API.app.core.LLM_Calls.local_cache_diagnostics import (
    build_local_cache_diagnostic,
    parse_inference_prefix_cache_intent,
)


pytestmark = pytest.mark.unit


def test_vllm_prefix_cache_diagnostic_is_cost_neutral_and_warns_on_unstable_shape() -> None:
    diagnostic = build_local_cache_diagnostic(
        provider="vllm",
        request={
            "inference_prefix_cache_intent": {
                "enabled": True,
                "scope": ["world_books", "character"],
                "static_segment_fingerprint": "worldbook:v3:stable",
            }
        },
        payload={
            "model": "local/qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.8,
            "n": 2,
            "stream": False,
        },
    )

    metadata = diagnostic.to_metadata()

    assert metadata["provider"] == "vllm"
    assert metadata["prefix_cache_intent_requested"] is True
    assert metadata["billing_cache_authoritative"] is False
    assert metadata["billing_cache_savings_reported"] is False
    assert metadata["request_shape_stable"] is False
    assert "request_shape_unstable" in metadata["warnings"]
    assert "cached_input_tokens" not in metadata
    assert "cache_creation_input_tokens" not in metadata


def test_llamacpp_prompt_cache_flags_are_reported_without_raw_paths() -> None:
    diagnostic = build_local_cache_diagnostic(
        provider="llama.cpp",
        request={
            "inference_prefix_cache_intent": {"enabled": True, "scope": ["world_books"]},
            "extra_body": {
                "cache_prompt": True,
                "cache_reuse": 128,
                "prompt_cache": "/tmp/request-cache.bin",
            },
        },
        payload={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
        app_config={
            "llama_api": {
                "prompt_cache": "/tmp/runtime-cache.bin",
                "prompt_cache_all": True,
                "prompt_cache_ro": True,
                "cache_prompt": True,
                "cache_reuse": 256,
            }
        },
    )

    metadata = diagnostic.to_metadata()

    assert metadata["provider"] == "llama.cpp"
    assert metadata["runtime_cache_mode"] == "read_only"
    assert set(metadata["runtime_flags"]) == {
        "prompt_cache",
        "prompt_cache_all",
        "prompt_cache_ro",
        "cache_prompt",
        "cache_reuse",
    }
    assert metadata["runtime_flags"]["cache_reuse"]["tokens"] == 256
    assert metadata["request_extension_keys"] == ["cache_prompt", "cache_reuse", "prompt_cache"]
    assert "/tmp/runtime-cache.bin" not in repr(metadata)
    assert "/tmp/request-cache.bin" not in repr(metadata)
    assert metadata["billing_cache_authoritative"] is False
    assert metadata["billing_cache_savings_reported"] is False


def test_inference_prefix_cache_intent_parser_is_separate_from_billing_cache_intent() -> None:
    intent = parse_inference_prefix_cache_intent(
        {
            "enabled": True,
            "scope": ["world_books", "world_books", "character"],
            "static_segment_fingerprint": "x" * 200,
            "provider_hint": {"ignored_path": "/tmp/not-metadata-safe.bin"},
        }
    )

    assert intent.enabled is True
    assert intent.scope == ("world_books", "character")
    assert len(intent.static_segment_fingerprint or "") <= 96
    assert not hasattr(intent, "ttl_seconds")
    assert "billing" not in intent.__class__.__name__.lower()
