from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError


class _FailingRegistry:
    def get_adapter(self, name: str):
        return None

    def register_adapter(self, name: str, adapter: object) -> None:
        return None


@pytest.mark.asyncio
async def test_openrouter_async_adapter_unavailable(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.gateway._get_llm_registry",
        lambda: _FailingRegistry(),
        raising=True,
    )

    with pytest.raises(ChatConfigurationError):
        await perform_chat_api_call_async(
            api_provider="openrouter",
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            api_key="k",
        )


def test_anthropic_adapter_unavailable(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.gateway._get_llm_registry",
        lambda: _FailingRegistry(),
        raising=True,
    )

    with pytest.raises(ChatConfigurationError):
        perform_chat_api_call(
            api_provider="anthropic",
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            api_key="k",
        )
