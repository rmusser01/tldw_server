"""
Async orchestrator tests for adapter-backed shims of Stage 3 providers:
Qwen, DeepSeek, HuggingFace, and Custom OpenAI-compatible.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


@pytest.mark.asyncio
async def test_qwen_async_non_streaming(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call_async
    from tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter import QwenAdapter

    def _fake_chat(self, *args, **kwargs):
        return {"object": "chat.completion", "choices": [{"index": 0, "message": {"content": "ok"}}]}

    monkeypatch.setattr(QwenAdapter, "chat", _fake_chat, raising=True)

    resp = await chat_api_call_async(
        api_endpoint="qwen",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-qwen-test",
        model="qwen-2",
        streaming=False,
    )
    assert resp.get("object") == "chat.completion"


@pytest.mark.asyncio
async def test_deepseek_async_streaming(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call_async
    from tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter import DeepSeekAdapter

    async def _fake_astream(self, *args, **kwargs) -> AsyncIterator[str]:
        yield "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(DeepSeekAdapter, "astream", _fake_astream, raising=True)

    stream = await chat_api_call_async(
        api_endpoint="deepseek",
        messages_payload=[{"role": "user", "content": "hi"}],
        model="deepseek-chat",
        streaming=True,
    )
    chunks = []
    async for line in stream:  # type: ignore[union-attr]
        chunks.append(line)
    assert any("data:" in c for c in chunks)
    assert sum(1 for c in chunks if "[DONE]" in c) == 1


@pytest.mark.asyncio
async def test_huggingface_async_non_streaming(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call_async
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    def _fake_chat(self, *args, **kwargs):
        return {"object": "chat.completion", "choices": [{"index": 0, "message": {"content": "ok"}}]}

    monkeypatch.setattr(HuggingFaceAdapter, "chat", _fake_chat, raising=True)

    resp = await chat_api_call_async(
        api_endpoint="huggingface",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-hf-test",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        streaming=False,
    )
    assert resp.get("object") == "chat.completion"


@pytest.mark.asyncio
async def test_custom_openai_async_streaming(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call_async
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter

    async def _fake_astream(self, *args, **kwargs) -> AsyncIterator[str]:
        yield "data: {\"choices\":[{\"delta\":{\"content\":\"y\"}}]}\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(CustomOpenAIAdapter, "astream", _fake_astream, raising=True)

    stream = await chat_api_call_async(
        api_endpoint="custom-openai-api",
        messages_payload=[{"role": "user", "content": "hi"}],
        model="my-openai-compatible",
        streaming=True,
    )
    lines = []
    async for ch in stream:  # type: ignore[union-attr]
        lines.append(ch)
    assert any("data:" in l for l in lines)
    assert sum(1 for l in lines if "[DONE]" in l) == 1
