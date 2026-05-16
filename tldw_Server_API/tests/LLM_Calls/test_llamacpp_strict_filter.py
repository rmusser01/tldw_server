import pytest
from unittest.mock import patch

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.Chat.chat_service import build_call_params_from_request
from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):

        return None

    def json(self):

        return {}

    def close(self):

        return None


@pytest.mark.unit
@pytest.mark.strict_mode
def test_llamacpp_strict_filter_drops_top_k_from_payload_non_streaming():
    fake_settings = {
        "llama_api": {
            "api_ip": "http://localhost:8001/v1/chat/completions",
            "streaming": False,
            "strict_openai_compat": True,
        }
    }

    captured_payload = {}

    class FakeClient:
        def __init__(self):
            self.closed = False

        def post(self, url, headers=None, json=None, timeout=None):  # noqa: ANN001
            captured_payload.clear()
            if json:
                captured_payload.update(json)
            return DummyResponse({})

        def stream(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("Streaming should not be invoked in this test")

        def close(self):

            self.closed = True

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.load_settings",
        return_value=fake_settings,
    ):

        response = chat_api_call(
            api_endpoint="llama.cpp",
            api_key=None,
            messages_payload=[{"role": "user", "content": "hello"}],
            topk=5,
            streaming=False,
            extra_body={
                "cache_prompt": True,
                "cache_reuse": 128,
                "prompt_cache": "/tmp/request-cache.bin",
            },
            inference_prefix_cache_intent={
                "enabled": True,
                "scope": ["world_books"],
                "static_segment_fingerprint": "worldbook:v1",
            },
            http_client_factory=lambda timeout: FakeClient(),
        )

    assert "top_k" not in captured_payload
    assert "cache_prompt" not in captured_payload
    assert "cache_reuse" not in captured_payload
    assert "prompt_cache" not in captured_payload
    assert "inference_prefix_cache_intent" not in captured_payload
    assert "messages" in captured_payload
    assert "stream" in captured_payload
    diagnostics = response["tldw_local_cache_diagnostics"]
    assert diagnostics["provider"] == "llama.cpp"
    assert diagnostics["request_extension_keys"] == ["cache_prompt", "cache_reuse", "prompt_cache"]
    assert diagnostics["billing_cache_authoritative"] is False
    assert "/tmp/request-cache.bin" not in repr(diagnostics)


@pytest.mark.unit
def test_llamacpp_tools_rejected_by_contract():
    with pytest.raises(ChatBadRequestError):
        validate_payload(
            "llama.cpp",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
            },
        )


@pytest.mark.unit
@pytest.mark.strict_mode
def test_build_call_params_rejects_llamacpp_advanced_fields_in_strict_mode():
    request = ChatCompletionRequest(
        model="llama.cpp/local-model",
        messages=[{"role": "user", "content": "hello"}],
        grammar_mode="inline",
        grammar_inline='root ::= "x"',
    )

    with pytest.raises(ChatBadRequestError):
        build_call_params_from_request(
            request_data=request,
            target_api_provider="llama.cpp",
            provider_api_key="test-key",
            templated_llm_payload=[{"role": "user", "content": "hello"}],
            final_system_message=None,
            app_config={"llama_api": {"strict_openai_compat": True}},
        )
