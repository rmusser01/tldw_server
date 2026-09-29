from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.LLM_Calls import capability_registry as cr


def test_normalize_payload_alias_does_not_override_canonical():


    payload = {"top_k": 9, "topk": 3}
    normalized = cr.normalize_payload("openrouter", payload)
    assert normalized["top_k"] == 9
    assert "topk" not in normalized


def test_normalize_payload_alias_fills_missing_canonical():


    payload = {"topk": 7}
    normalized = cr.normalize_payload("openrouter", payload)
    assert normalized["top_k"] == 7
    assert "topk" not in normalized


def test_normalize_payload_prefers_maxp_over_topp():
    payload = {"topp": 0.2, "maxp": 0.9}
    normalized = cr.normalize_payload("openai", payload)
    assert normalized["top_p"] == 0.9


def test_normalize_payload_alias_maps_custom_prompt():
    payload = {"custom_prompt": "use this"}
    normalized = cr.normalize_payload("openai", payload)
    assert normalized["custom_prompt_arg"] == "use this"


def test_validate_payload_rejects_unknown_fields():


    payload = {"messages": [], "model": "test", "unknown_field": 1}
    with pytest.raises(ChatBadRequestError) as exc:
        cr.validate_payload("openai", payload)
    assert "unknown_field" in str(exc.value)


def test_validate_payload_records_rejection_metric(monkeypatch):
    calls = []

    def fake_record(provider_key: str) -> None:
        calls.append(provider_key)

    monkeypatch.setattr(cr, "_record_validation_rejection", fake_record)

    payload = {"messages": [], "model": "test", "unknown_field": 1}
    with pytest.raises(ChatBadRequestError):
        cr.validate_payload("openai", payload)

    assert calls == ["openai"]


def test_validate_payload_allows_provider_extensions():


    payload = {"messages": [], "model": "test", "top_k": 5, "min_p": 0.1}
    normalized = cr.validate_payload("openrouter", payload)
    assert normalized["top_k"] == 5
    assert normalized["min_p"] == 0.1


def test_validate_payload_maps_llama_n_predict():
    payload = {"messages": [], "model": "test", "n_predict": 64}
    normalized = cr.validate_payload("llama.cpp", payload)
    assert normalized["max_tokens"] == 64


def test_validate_payload_maps_kobold_aliases():
    payload = {
        "messages": [],
        "model": "test",
        "max_length": 120,
        "stop_sequence": ["END"],
        "num_responses": 2,
    }
    normalized = cr.validate_payload("kobold", payload)
    assert normalized["max_tokens"] == 120
    assert normalized["stop"] == ["END"]
    assert normalized["n"] == 2


def test_validate_payload_maps_mistral_random_seed():
    payload = {"messages": [], "model": "test", "random_seed": 7}
    normalized = cr.validate_payload("mistral", payload)
    assert normalized["seed"] == 7


def test_validate_payload_allows_base_url_override():
    payload = {"messages": [], "model": "test", "base_url": "https://example.com/v1"}
    normalized = cr.validate_payload("openai", payload)
    assert normalized["base_url"] == "https://example.com/v1"


def test_validate_payload_rejects_extension_for_other_provider():


    payload = {"messages": [], "model": "test", "top_k": 5}
    with pytest.raises(ChatBadRequestError):
        cr.validate_payload("openai", payload)


def test_validate_payload_ignores_none_values():


    payload = {"messages": [], "model": "test", "unknown_field": None}
    normalized = cr.validate_payload("openai", payload)
    assert normalized["messages"] == []


def test_validate_payload_rejects_blocked_fields():


    payload = {"messages": [], "model": "test", "tool_choice": "auto"}
    with pytest.raises(ChatBadRequestError) as exc:
        cr.validate_payload("cohere", payload)
    assert "tool_choice" in str(exc.value)


def test_validate_payload_rejects_invalid_tools_shape():


    payload = {"messages": [], "model": "test", "tools": {"type": "function"}}
    with pytest.raises(ChatBadRequestError) as exc:
        cr.validate_payload("openai", payload)
    assert "tools" in str(exc.value).lower()


def test_validate_payload_rejects_tool_missing_name():


    payload = {
        "messages": [],
        "model": "test",
        "tools": [{"type": "function", "function": {"parameters": {}}}],
    }
    with pytest.raises(ChatBadRequestError) as exc:
        cr.validate_payload("openai", payload)
    assert "function.name" in str(exc.value)


def test_validate_payload_allows_valid_tool_definition():


    payload = {
        "messages": [],
        "model": "test",
        "tools": [{"type": "function", "function": {"name": "do", "parameters": {}}}],
    }
    normalized = cr.validate_payload("openai", payload)
    assert normalized["tools"][0]["function"]["name"] == "do"


def test_validate_payload_rejects_invalid_logit_bias():


    payload = {"messages": [], "model": "test", "logit_bias": {"not-a-token": 1}}
    with pytest.raises(ChatBadRequestError) as exc:
        cr.validate_payload("openai", payload)
    assert "logit_bias" in str(exc.value).lower()


def test_validate_payload_rejects_invalid_response_format():


    payload = {"messages": [], "model": "test", "response_format": {"type": "json_schema"}}
    with pytest.raises(ChatBadRequestError) as exc:
        cr.validate_payload("openai", payload)
    assert "response_format" in str(exc.value).lower()


def test_validate_payload_allows_tool_choice_none_without_tools():


    payload = {"messages": [], "model": "test", "tool_choice": "none"}
    normalized = cr.validate_payload("openai", payload)
    assert normalized["tool_choice"] == "none"


def test_validate_payload_normalizes_tool_choice_auto_when_tools_empty():
    payload = {"messages": [], "model": "test", "tool_choice": "auto", "tools": []}
    normalized = cr.validate_payload("openai", payload)
    assert normalized.get("tool_choice") == "none"
    assert normalized.get("tools") is None


def test_validate_payload_allows_unknown_response_format_type():


    payload = {"messages": [], "model": "test", "response_format": {"type": "custom"}}
    normalized = cr.validate_payload("openai", payload)
    assert normalized["response_format"]["type"] == "custom"


def test_validate_payload_allows_json_schema_response_format():


    payload = {
        "messages": [],
        "model": "test",
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    }
    normalized = cr.validate_payload("openai", payload)
    assert normalized["response_format"]["type"] == "json_schema"


def test_validate_payload_allows_unknown_tool_type():


    payload = {"messages": [], "model": "test", "tools": [{"type": "custom_tool", "payload": {"ok": True}}]}
    normalized = cr.validate_payload("openai", payload)
    assert normalized["tools"][0]["type"] == "custom_tool"


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "llama.cpp", "custom-openai-api-12"])
def test_validate_payload_omits_unsupported_neutral_repetition_penalty(provider: str) -> None:
    """Neutral character snapshot defaults do not require adapter support."""
    payload = {"messages": [], "repetition_penalty": 1.0}
    assert cr.validate_payload(provider, payload) == {"messages": []}
    assert payload["repetition_penalty"] == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("penalty", [0.0, 1.1, True, "1.0"])
def test_validate_payload_rejects_unsupported_non_neutral_repetition_penalty(penalty: object) -> None:
    """Meaningful values and invalid types must not evade capability validation."""
    with pytest.raises(ChatBadRequestError, match="repetition_penalty"):
        cr.validate_payload("llama.cpp", {"messages": [], "repetition_penalty": penalty})


@pytest.mark.unit
@pytest.mark.parametrize("penalty", [1.0, 1.1])
def test_validate_payload_preserves_supported_repetition_penalty(
    monkeypatch: pytest.MonkeyPatch,
    penalty: float,
) -> None:
    """An adapter that declares support keeps the requested value unchanged."""
    monkeypatch.setitem(cr.PROVIDER_EXTENSIONS, "local-llm", {"repetition_penalty"})
    payload = {"messages": [], "repetition_penalty": penalty}
    assert cr.validate_payload("local-llm", payload) == payload
