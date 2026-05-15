import pytest

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.Chat.chat_service import build_call_params_from_request
from tldw_Server_API.app.core.LLM_Calls.llamacpp_request_extensions import (
    resolve_llamacpp_request_extensions,
    resolve_llamacpp_runtime_caps,
)


pytestmark = pytest.mark.unit


def test_resolver_merges_saved_grammar_into_extra_body() -> None:
    payload = resolve_llamacpp_request_extensions(
        request_fields={
            "grammar_mode": "library",
            "grammar_id": "grammar_1",
            "grammar_override": None,
            "thinking_budget_tokens": None,
            "extra_body": {"mirostat": 2},
        },
        provider="llama.cpp",
        grammar_record={"id": "grammar_1", "grammar_text": 'root ::= "ok"'},
        runtime_caps={"strict_openai_compat": False, "thinking_budget": {"supported": False}},
    )

    assert payload["extra_body"]["grammar"] == 'root ::= "ok"'
    assert payload["extra_body"]["mirostat"] == 2


def test_resolver_overrides_conflicting_raw_extra_body_grammar() -> None:
    payload = resolve_llamacpp_request_extensions(
        request_fields={
            "grammar_mode": "inline",
            "grammar_inline": 'root ::= "inline"',
            "extra_body": {"grammar": 'root ::= "raw"'},
        },
        provider="llama.cpp",
        grammar_record=None,
        runtime_caps={"strict_openai_compat": False, "thinking_budget": {"supported": False}},
    )

    assert payload["extra_body"]["grammar"] == 'root ::= "inline"'


def test_build_call_params_merges_llamacpp_inline_grammar_into_extra_body() -> None:
    request = ChatCompletionRequest(
        model="llama.cpp/local-model",
        messages=[{"role": "user", "content": "reply in JSON"}],
        grammar_mode="inline",
        grammar_inline='root ::= "inline"',
        extra_body={"mirostat": 2},
    )

    params = build_call_params_from_request(
        request_data=request,
        target_api_provider="llama.cpp",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "reply in JSON"}],
        final_system_message=None,
        app_config={"llama_api": {"strict_openai_compat": False}},
    )

    assert params["extra_body"]["grammar"] == 'root ::= "inline"'
    assert params["extra_body"]["mirostat"] == 2
    assert "grammar_mode" not in params
    assert "grammar_inline" not in params


def test_build_call_params_preserves_inference_prefix_cache_intent_for_local_diagnostics() -> None:
    request = ChatCompletionRequest(
        model="llama.cpp/local-model",
        messages=[{"role": "user", "content": "hello"}],
        inference_prefix_cache_intent={
            "enabled": True,
            "scope": ["world_books"],
            "static_segment_fingerprint": "worldbook:v1",
        },
    )

    params = build_call_params_from_request(
        request_data=request,
        target_api_provider="llama.cpp",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "hello"}],
        final_system_message=None,
        app_config={"llama_api": {"strict_openai_compat": False}},
    )

    assert params["inference_prefix_cache_intent"]["enabled"] is True
    assert "billing_prompt_cache_intent" not in params


def test_build_call_params_rejects_llamacpp_fields_for_non_llamacpp_provider() -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "reply in JSON"}],
        grammar_mode="inline",
        grammar_inline='root ::= "inline"',
    )

    with pytest.raises(ChatBadRequestError):
        build_call_params_from_request(
            request_data=request,
            target_api_provider="openai",
            provider_api_key="test-key",
            templated_llm_payload=[{"role": "user", "content": "reply in JSON"}],
            final_system_message=None,
            app_config=None,
        )


def test_llamacpp_request_extension_helpers_have_docstrings() -> None:
    assert resolve_llamacpp_runtime_caps.__doc__
    assert resolve_llamacpp_request_extensions.__doc__
