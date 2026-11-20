from configparser import ConfigParser
from pathlib import Path

import pytest

from tldw_Server_API.app.core.LLM_Calls.model_metadata import get_model_metadata
from tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter import DeepSeekAdapter
from tldw_Server_API.app.core.Usage import pricing_catalog


def test_bundled_deepseek_default_has_current_model():
    config = ConfigParser(interpolation=None)
    config.read(Path(__file__).resolve().parents[2] / "Config_Files/config.txt")
    assert config.get("API", "deepseek_model") == "deepseek-flash"


def test_retired_deepseek_models_remain_priced_but_hidden(monkeypatch):
    catalog = pricing_catalog.PricingCatalog()
    monkeypatch.setattr(pricing_catalog, "get_pricing_catalog", lambda: catalog)
    assert "deepseek-flash" in pricing_catalog.list_provider_models("deepseek")
    assert not {"deepseek-chat", "deepseek-reasoner"} & set(pricing_catalog.list_provider_models("deepseek"))
    assert catalog.get_rates("deepseek", "deepseek-chat")[0] > 0


def test_deepseek_metadata_advertises_provider_ceiling():
    metadata = get_model_metadata("deepseek", "deepseek-flash")
    assert metadata["context_window"] == 1_048_576
    assert metadata["max_output_tokens"] == 393_216


@pytest.mark.parametrize(
    ("model", "base_url", "configured_limit", "requested_limit", "expected"),
    [
        ("deepseek-flash", None, None, None, 200_000),
        ("deepseek-flash", None, 4096, None, 4096),
        ("deepseek-flash", None, 4096, 512, 512),
        ("deepseek-flash", "https://example.com/v1", None, None, None),
        ("alternate-model", None, None, None, None),
    ],
)
def test_deepseek_payload_uses_model_default_and_preserves_overrides(
    monkeypatch, model, base_url, configured_limit, requested_limit, expected
):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    adapter = DeepSeekAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    request = adapter._bind_request_credentials(
        {
            "model": model,
            "base_url": base_url,
            "max_tokens": requested_limit,
            "app_config": {"deepseek_api": {"max_tokens": configured_limit}},
        }
    )
    payload = adapter._build_payload(adapter._apply_config_defaults(request))
    assert payload.get("max_tokens") == expected


def test_stock_budget_leaves_room_for_prompt(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.estimate_tokens_from_json", lambda _prompt: 1_048_400
    )
    adapter = DeepSeekAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    assert adapter._bind_request_credentials({"model": "deepseek-flash"})["max_tokens"] == 176


def test_openai_luna_uses_native_output_field_and_preserves_explicit_budget(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    base = {"base_url": "https://api.openai.com/v1", "model": "gpt-6-luna"}
    default = adapter._build_openai_payload(adapter._bind_request_credentials(base))
    explicit = adapter._build_openai_payload(adapter._bind_request_credentials({**base, "max_completion_tokens": 512}))
    assert default["max_completion_tokens"] == 128_000
    assert default["reasoning_effort"] == "none"
    assert explicit["max_completion_tokens"] == 512
    assert "max_tokens" not in default


@pytest.mark.parametrize(("stream", "expected"), [(False, 21_333), (True, 128_000)])
def test_anthropic_budget_depends_on_transport(monkeypatch, stream, expected):
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    request = adapter._bind_request_credentials(
        {"base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-6", "stream": stream}
    )
    assert adapter._build_payload(request)["max_tokens"] == expected


def test_groq_new_default_uses_max_completion_tokens(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter import GroqAdapter

    adapter = GroqAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    request = adapter._bind_request_credentials(
        {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b"}
    )
    payload = adapter._build_payload(request)
    assert payload["max_completion_tokens"] == 65_536
    assert "max_tokens" not in payload


def test_commercial_gemini_metadata_includes_the_output_limit():
    assert get_model_metadata("google", "gemini-3.5-flash-lite")["max_output_tokens"] == 65_536


def test_provider_native_completion_limit_wins_over_generic_limit():
    from tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter import GroqAdapter
    from tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter import MoonshotAdapter
    from tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter import QwenAdapter

    request = {"max_tokens": 4096, "max_completion_tokens": 512}
    for adapter in (GroqAdapter(), QwenAdapter()):
        payload = adapter._build_payload(request)
        assert payload["max_completion_tokens"] == 512
        assert "max_tokens" not in payload
    args = MoonshotAdapter()._to_handler_args(request, streaming=False)
    assert args["max_completion_tokens"] == 512


def test_native_extra_body_budget_wins_over_configured_budget(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.payload_utils import merge_extra_body
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    request = adapter._bind_request_credentials(
        {
            "model": "gpt-6-luna",
            "base_url": "https://api.openai.com/v1",
            "app_config": {"openai_api": {"max_tokens": "4096"}},
            "extra_body": {"max_completion_tokens": 512},
        }
    )
    payload = merge_extra_body(adapter._build_openai_payload(adapter._apply_config_defaults(request)), request)
    assert payload["max_completion_tokens"] == 512
    assert "max_tokens" not in payload


def test_current_gemini_default_carries_tool_thought_signature_and_explicit_budget(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_request_schemas import ChatCompletionAssistantMessageParam
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    adapter = GoogleAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    response = adapter._normalize_to_openai_shape(
        {
            "candidates": [
                {"content": {"parts": [
                    {"functionCall": {"name": "lookup", "args": {"key": "x"}}, "thoughtSignature": "opaque"}
                ]}}
            ]
        }
    )
    assistant = ChatCompletionAssistantMessageParam.model_validate(response["choices"][0]["message"])
    request = adapter._bind_request_credentials({
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-3.5-flash-lite",
        "max_output_tokens": 512,
        "messages": [
            assistant.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": "call_0", "content": '{"value": 1}'},
        ],
    })
    payload = adapter._build_payload(request)
    assert payload["generationConfig"]["maxOutputTokens"] == 512
    assert payload["contents"][0]["parts"][0]["thoughtSignature"] == "opaque"
    assert payload["contents"][1]["parts"] == [{"functionResponse": {"name": "lookup", "response": {"value": 1}}}]


def test_current_anthropic_default_preserves_text_continuation(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    messages = [{"role": "user", "content": "Write a sentence."}, {"role": "assistant", "content": "This is"}]
    request = adapter._bind_request_credentials(
        {"model": "claude-sonnet-4-6", "base_url": "https://api.anthropic.com", "messages": messages}
    )
    provider_messages = adapter._build_payload(request)["messages"]
    assert provider_messages[-2]["role"] == "assistant"
    assert provider_messages[-1]["role"] == "user"
    assert len(messages) == 2



def test_streamed_gemini_tool_call_preserves_thought_signature():
    from tldw_Server_API.app.core.Chat.streaming_utils import StreamingResponseHandler

    handler = StreamingResponseHandler.__new__(StreamingResponseHandler)
    handler.tool_call_accumulator = {}
    handler.tool_call_order = []
    extra_content = {"google": {"thought_signature": "opaque-signature"}}
    handler._accumulate_tool_calls([{
        "id": "call_0",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"key":'},
        "extra_content": extra_content,
    }])
    handler._accumulate_tool_calls([{"function": {"arguments": '"x"}'}}])
    assert handler.get_accumulated_tool_calls() == [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"key":"x"}'},
        "extra_content": extra_content,
    }]


def test_openai_luna_reasoning_effort_removes_incompatible_sampling(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    request = adapter._bind_request_credentials({
        "model": "gpt-6-luna",
        "base_url": "https://api.openai.com/v1",
        "max_completion_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.8,
        "extra_body": {"reasoning_effort": "high", "logprobs": True, "top_logprobs": 2},
    })
    payload = adapter._finalize_openai_payload(adapter._build_openai_payload(request), request)
    assert payload["reasoning_effort"] == "high"
    assert payload["max_completion_tokens"] == 512
    assert not {"temperature", "top_p", "logprobs", "top_logprobs"}.intersection(payload)


def test_openrouter_preserves_explicit_completion_limit(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter import OpenRouterAdapter

    adapter = OpenRouterAdapter()
    monkeypatch.setattr(adapter, "_validate_model_inventory", lambda *_args: None)
    request = adapter._bind_request_credentials({
        "model": "google/gemini-3.1-flash-lite",
        "base_url": "https://openrouter.ai/api/v1",
        "app_config": {"openrouter_api": {"max_tokens": "4096"}},
        "extra_body": {"max_completion_tokens": 512},
    })
    assert adapter._build_payload(request)["max_tokens"] == 512
