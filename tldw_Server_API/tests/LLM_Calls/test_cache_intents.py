from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.chat_service import build_call_params_from_request


pytestmark = pytest.mark.unit


def test_cache_intent_defaults_to_no_request_changes() -> None:
    from tldw_Server_API.app.core.LLM_Calls.cache_intents import (
        apply_billing_prompt_cache_intent,
        parse_billing_prompt_cache_intent,
    )

    intent = parse_billing_prompt_cache_intent(None)
    payload, diagnostic = apply_billing_prompt_cache_intent(
        "openai",
        {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
        {"billing_prompt_cache_intent": None},
    )

    assert intent.enabled is False
    assert payload == {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
    assert diagnostic.cache_intent_requested is False
    assert diagnostic.cache_intent_applied is False


def test_unsupported_provider_ignores_enabled_intent_with_diagnostic() -> None:
    from tldw_Server_API.app.core.LLM_Calls.cache_intents import apply_billing_prompt_cache_intent

    payload, diagnostic = apply_billing_prompt_cache_intent(
        "local-llm",
        {"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        {
            "billing_prompt_cache_intent": {
                "enabled": True,
                "scope": ["system"],
                "ttl_seconds": 3600,
                "static_segment_fingerprint": "prompt-v1:abc123",
                "provider_hint": {"prompt_cache_key": "stable-session"},
            }
        },
    )

    assert payload == {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
    assert diagnostic.cache_intent_requested is True
    assert diagnostic.cache_intent_applied is False
    assert diagnostic.provider == "local-llm"
    assert diagnostic.reason == "provider_not_supported"
    assert diagnostic.provider_usage_authoritative is False
    assert diagnostic.to_metadata()["provider_usage_authoritative"] is False


def test_cache_intent_metadata_is_bounded_and_omits_hint_values() -> None:
    from tldw_Server_API.app.core.LLM_Calls.cache_intents import apply_billing_prompt_cache_intent

    _, diagnostic = apply_billing_prompt_cache_intent(
        "openai",
        {"model": "gpt-5.4", "messages": []},
        {
            "billing_prompt_cache_intent": {
                "enabled": True,
                "scope": ["system", "world_book"],
                "ttl_seconds": 86400,
                "static_segment_fingerprint": "prompt-v1:" + ("a" * 128),
                "provider_hint": {
                    "prompt_cache_key": "sensitive-user-specific-value",
                    "prompt_cache_retention": "24h",
                },
            }
        },
    )

    metadata = diagnostic.to_metadata()
    assert metadata["cache_intent_requested"] is True
    assert metadata["cache_intent_applied"] is True
    assert metadata["provider_hint_keys"] == ["prompt_cache_key", "prompt_cache_retention"]
    assert "sensitive-user-specific-value" not in str(metadata)
    assert len(metadata["static_segment_fingerprint"]) <= 96


def test_chat_request_passes_explicit_cache_intent_to_provider_call_params() -> None:
    request = ChatCompletionRequest(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        billing_prompt_cache_intent={
            "enabled": True,
            "scope": ["system"],
            "ttl_seconds": 86400,
            "static_segment_fingerprint": "prompt-v1:stable",
        },
    )

    params = build_call_params_from_request(
        request_data=request,
        target_api_provider="openai",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        final_system_message="Stable system prompt",
        app_config={"openai_api": {}},
    )

    assert params["billing_prompt_cache_intent"]["enabled"] is True
    assert params["billing_prompt_cache_intent"]["static_segment_fingerprint"] == "prompt-v1:stable"
