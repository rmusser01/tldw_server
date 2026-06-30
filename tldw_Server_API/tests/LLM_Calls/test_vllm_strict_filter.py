import pytest
from unittest.mock import MagicMock, patch

from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call


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
def test_vllm_strict_filter_drops_top_k_from_payload_non_streaming():
    # vLLM strict mode configuration
    fake_settings = {
        "vllm_api": {
            "api_ip": "http://localhost:9001/v1/chat/completions",
            "streaming": False,
            "strict_openai_compat": True,
        }
    }

    captured_payload = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured_payload.clear()
        if json:
            captured_payload.update(json)
        return DummyResponse({})

    with patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters.load_settings",
        return_value=fake_settings,
    ), patch(
        "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters._hc_create_client"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.side_effect = fake_post
        mock_client.close.return_value = None
        mock_client_cls.return_value = mock_client

        response = chat_api_call(
            api_endpoint="vllm",
            api_key=None,
            messages_payload=[{"role": "user", "content": "hello"}],
            topk=7,
            temp=0.8,
            n=2,
            streaming=False,
            extra_body={"prompt_cache_key": "must-not-forward"},
            inference_prefix_cache_intent={
                "enabled": True,
                "scope": ["world_books"],
                "static_segment_fingerprint": "worldbook:v1",
            },
        )

    assert "top_k" not in captured_payload
    assert "prompt_cache_key" not in captured_payload
    assert "inference_prefix_cache_intent" not in captured_payload
    assert "messages" in captured_payload
    assert "stream" in captured_payload
    diagnostics = response["tldw_local_cache_diagnostics"]
    assert diagnostics["provider"] == "vllM".lower()
    assert diagnostics["prefix_cache_intent_requested"] is True
    assert diagnostics["billing_cache_authoritative"] is False
    assert "cached_input_tokens" not in diagnostics
