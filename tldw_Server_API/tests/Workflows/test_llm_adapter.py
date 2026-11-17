import pytest

import tldw_Server_API.app.core.Workflows.adapters as wf_adapters

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("healthy_no_override_tts_credential_snapshot")]


@pytest.mark.asyncio
async def test_llm_adapter_builds_messages_and_metadata(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    calls = {}

    async def fake_call(**kwargs):
        calls["kwargs"] = kwargs
        return {
            "choices": [{"message": {"content": "Hello Nina"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_call)

    config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt": "Hi {{ inputs.name }}",
        "include_response": True,
    }
    context = {"inputs": {"name": "Nina"}, "user_id": "1"}

    out = await wf_adapters.run_llm_adapter(config, context)

    assert out["text"] == "Hello Nina"
    assert out["metadata"]["token_usage"]["prompt_tokens"] == 3
    assert out["response"]["choices"][0]["message"]["content"] == "Hello Nina"
    assert calls["kwargs"]["api_endpoint"] == "openai"
    assert calls["kwargs"]["messages_payload"][0]["content"] == "Hi Nina"
    assert calls["kwargs"]["user"] == "1"
