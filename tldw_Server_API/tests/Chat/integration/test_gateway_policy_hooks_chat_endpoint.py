"""/chat/completions reaches the gateway's policy hooks: a pre-call refusal reaches the client, and a post-call
replacement keeps the provider's usage (TASK-13950.13)."""

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway

pytestmark = pytest.mark.integration

_PROVIDER = "anthropic"  # outside chat.py's TEST_MODE mock set (openai, groq, mistral)
_BODY = {"model": "claude-3-5-haiku-20241022", "api_provider": _PROVIDER, "messages": [{"role": "user", "content": "hi"}]}
_USAGE = {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}


class _Adapter:
    def chat(self, request, *, timeout=None):
        return {"id": "x", "object": "chat.completion", "model": "m", "usage": dict(_USAGE),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "provider reply"}, "finish_reason": "stop"}]}

    def stream(self, request, *, timeout=None):
        return iter([
            'data: {"choices":[{"index":0,"delta":{"content":"provider reply"}}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":12,"total_tokens":42}}\n\n',
            "data: [DONE]\n\n",
        ])

    def capabilities(self):
        return {}


class _Registry:
    def get_adapter(self, name):
        return _Adapter()

    def get_audited_call_policy_transport(self, name):
        return None


@pytest.fixture
def fake_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: _Registry())


@pytest.fixture
def recorded_usage(monkeypatch):
    rows = []

    async def log_llm_usage(**kwargs):
        rows.append((kwargs["prompt_tokens"], kwargs["completion_tokens"]))

    monkeypatch.setattr(chat_service, "log_llm_usage", log_llm_usage)
    return rows


def _redact(call, result):
    """The post-call contract: replace the reply, keep the provider's usage, never raise."""
    if not call.stream:
        return {**result, "choices": [{"index": 0, "message": {"role": "assistant", "content": "[redacted]"}, "finish_reason": "stop"}]}
    return (chunk.replace("provider reply", "[redacted]") for chunk in result)


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_a_registered_hook_fires_for_chat_completions(
    authenticated_client, mock_chacha_db, setup_dependencies, fake_anthropic, stream
):
    seen = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: seen.append((call.surface, call.provider, call.stream)))
    try:
        response = authenticated_client.post("/api/v1/chat/completions", json={**_BODY, "stream": stream})
    finally:
        remove()

    assert response.status_code == 200, response.text
    assert "provider reply" in response.text
    assert seen == [("chat.completions", _PROVIDER, stream)]


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_a_policy_refusal_reaches_the_chat_client(
    authenticated_client, mock_chacha_db, setup_dependencies, fake_anthropic, stream
):
    def refuse(call):
        raise LLMPolicyRefusal("Input violates moderation policy")

    remove = gateway.add_policy_hooks(pre_call=refuse)
    try:
        response = authenticated_client.post("/api/v1/chat/completions", json={**_BODY, "stream": stream})
    finally:
        remove()

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "Input violates moderation policy"}


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_a_post_call_replacement_keeps_the_provider_usage(
    authenticated_client, mock_chacha_db, setup_dependencies, fake_anthropic, recorded_usage, stream
):
    remove = gateway.add_policy_hooks(post_call=_redact)
    try:
        response = authenticated_client.post("/api/v1/chat/completions", json={**_BODY, "stream": stream})
    finally:
        remove()

    assert response.status_code == 200, response.text
    assert "[redacted]" in response.text and "provider reply" not in response.text
    assert recorded_usage == [(30, 12)]
