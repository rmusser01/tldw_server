import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatConfigurationError,
)


@pytest.mark.asyncio
async def test_first_chat_verifier_records_ready_openai_like_response(monkeypatch):
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    calls = []

    async def _fake_call_chat_completion(**kwargs):
        calls.append(kwargs)
        return {
            "id": "chatcmpl-first-run",
            "choices": [
                {
                    "message": {
                        "content": "Hello from the configured model.",
                    }
                }
            ],
        }

    monkeypatch.setattr(first_chat_verifier, "_call_chat_completion", _fake_call_chat_completion)

    result = await first_chat_verifier.verify_first_chat(
        provider="openai",
        model="gpt-4.1-mini",
        prompt="Say hello once.",
    )

    assert result.status == "ready"
    assert result.provider == "openai"
    assert result.model == "gpt-4.1-mini"
    assert result.response_id == "chatcmpl-first-run"
    assert result.response_text == "Hello from the configured model."
    assert result.failure_category is None
    assert calls == [
        {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "prompt": "Say hello once.",
        }
    ]


@pytest.mark.asyncio
async def test_first_chat_verifier_extracts_object_response_text(monkeypatch):
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    class Message:
        content = "Object style response."

    class Choice:
        message = Message()

    class Response:
        id = "object-response-id"
        choices = [Choice()]

    async def _fake_call_chat_completion(**_kwargs):
        return Response()

    monkeypatch.setattr(first_chat_verifier, "_call_chat_completion", _fake_call_chat_completion)

    result = await first_chat_verifier.verify_first_chat(
        provider="anthropic",
        model="claude-test",
    )

    assert result.status == "ready"
    assert result.response_id == "object-response-id"
    assert result.response_text == "Object style response."


@pytest.mark.asyncio
async def test_first_chat_verifier_maps_auth_failures_without_raw_detail(monkeypatch):
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    raw_detail = "invalid sk-secret-token from /Users/local/private/config.txt"

    async def _fake_call_chat_completion(**_kwargs):
        raise ChatAuthenticationError(raw_detail, provider="openai")

    monkeypatch.setattr(first_chat_verifier, "_call_chat_completion", _fake_call_chat_completion)

    result = await first_chat_verifier.verify_first_chat(
        provider="openai",
        model="gpt-4.1-mini",
    )

    assert result.status == "failed"
    assert result.failure_category == "auth_failed"
    assert result.response_id is None
    assert result.response_text is None
    assert "sk-secret-token" not in str(result)
    assert "/Users/local/private" not in str(result)


@pytest.mark.asyncio
async def test_first_chat_verifier_maps_configuration_failures_without_raw_detail(monkeypatch):
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    raw_detail = "missing config at C:\\secret\\config.txt with token=abc123"

    async def _fake_call_chat_completion(**_kwargs):
        raise ChatConfigurationError(raw_detail, provider="openai")

    monkeypatch.setattr(first_chat_verifier, "_call_chat_completion", _fake_call_chat_completion)

    result = await first_chat_verifier.verify_first_chat(
        provider="openai",
        model="gpt-4.1-mini",
    )

    assert result.status == "failed"
    assert result.failure_category == "configuration_error"
    assert "C:\\secret" not in str(result)
    assert "token=abc123" not in str(result)
