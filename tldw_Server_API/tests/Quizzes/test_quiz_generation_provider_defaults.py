"""Exercise ordinary quiz calls at the adapter boundary with configured defaults."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.services import quiz_generator

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def configured_adapter(monkeypatch: pytest.MonkeyPatch):
    snapshot = {
        "llama_api": {"model": "configured-local-model", "api_ip": "http://127.0.0.1:8010/v1/chat/completions"},
        "ollama_api": {"model": "configured-ollama-model"},
    }
    calls: list[dict[str, Any]] = []
    response = {"choices": [{"message": {"content": '{"questions": []}'}}]}

    class Adapter:
        def __init__(self, provider: str):
            self.provider = provider

        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            calls.append({"selected_provider": self.provider, **request})
            return response

    monkeypatch.setattr(quiz_generator, "load_and_log_configs", lambda: snapshot)
    monkeypatch.setattr(quiz_generator, "resolve_provider_api_key", lambda *_args, **_kwargs: (None, {}))
    monkeypatch.setattr(quiz_generator, "_get_adapter", lambda provider: Adapter(provider))
    return snapshot, calls, response


@pytest.mark.parametrize("provider", ["llamacpp", "llama.cpp", "llama_cpp", "llama-cpp"])
async def test_quiz_call_uses_configured_model_for_local_provider_alias(provider, configured_adapter):
    snapshot, calls, response = configured_adapter
    result = await quiz_generator._call_quiz_generation_llm(prompt="Biology facts", api_provider=provider)
    # These assertions verify the actual adapter request, with no provider network call.
    assert result is response  # nosec B101
    assert [call["model"] for call in calls] == ["configured-local-model"]  # nosec B101
    assert calls[0]["app_config"] is snapshot  # nosec B101
    assert calls[0]["selected_provider"] == "llama.cpp"  # nosec B101 -- Canonical no-key local routing.


async def test_quiz_call_keeps_explicit_model_precedence(configured_adapter):
    _, calls, _ = configured_adapter
    await quiz_generator._call_quiz_generation_llm(
        prompt="Biology facts", api_provider="llama.cpp", model="explicit-model"
    )
    assert [call["model"] for call in calls] == ["explicit-model"]  # nosec B101 -- Test request oracle.


async def test_quiz_call_keeps_ordinary_provider_default(configured_adapter):
    _, calls, _ = configured_adapter
    await quiz_generator._call_quiz_generation_llm(prompt="Biology facts", api_provider="ollama")
    assert [call["model"] for call in calls] == ["configured-ollama-model"]  # nosec B101 -- Test request oracle.


async def test_quiz_call_refuses_missing_model_before_adapter_call(configured_adapter):
    snapshot, calls, _ = configured_adapter
    snapshot["llama_api"].clear()
    with pytest.raises(ChatConfigurationError):
        await quiz_generator._call_quiz_generation_llm(prompt="Biology facts", api_provider="llama.cpp")
    assert calls == []  # nosec B101 -- Missing configuration must not invoke a provider.


async def test_quiz_call_canonicalizes_runtime_default(monkeypatch, configured_adapter):
    _, calls, _ = configured_adapter
    monkeypatch.setattr(quiz_generator, "get_default_provider", lambda: "llama_cpp")
    await quiz_generator._call_quiz_generation_llm(prompt="Biology facts")
    assert [(call["selected_provider"], call["model"]) for call in calls] == [
        ("llama.cpp", "configured-local-model")
    ]  # nosec B101 -- Pytest oracle for runtime default alias/model routing.

@pytest.mark.parametrize("policy", ["disabled", "explicit_model", "default_model", "unhealthy"])
async def test_quiz_denied_provider_policy_never_dispatches(configured_adapter, policy):
    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError

    _, calls, _ = configured_adapter
    overrides.set_llm_provider_overrides_cache_for_tests({
        "llama.cpp": overrides.LLMProviderOverride(
            provider="llama.cpp", is_enabled=policy != "disabled", allowed_models=["approved-model"]
        )
    }, healthy=policy != "unhealthy")
    with pytest.raises(ByokResolutionError):
        await quiz_generator._call_quiz_generation_llm(
            prompt="Private source", api_provider="llamacpp",
            model="denied-model" if policy == "explicit_model" else None,
        )
    assert calls == []  # nosec B101 - denied configuration cannot reach the adapter.


async def test_quiz_allowed_configured_model_dispatches(configured_adapter):
    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides

    _, calls, _ = configured_adapter
    overrides.set_llm_provider_overrides_cache_for_tests({
        "llama.cpp": overrides.LLMProviderOverride(provider="llama.cpp", allowed_models=["configured-local-model"])
    })
    await quiz_generator._call_quiz_generation_llm(prompt="Source facts", api_provider="llamacpp")
    assert [call["model"] for call in calls] == ["configured-local-model"]  # nosec B101 - policy uses the dispatched default.

async def test_quiz_cancellation_retains_provider_ownership(configured_adapter, monkeypatch):
    import asyncio
    import threading

    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id
    from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root

    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    owners = []
    class HeldAdapter:
        def chat(self, request):
            owners.append(current_llm_user_id())
            entered.set()
            try:
                if not release.wait(2):
                    raise RuntimeError("test provider was not released")
                return {"choices": [{"message": {"content": "retained result"}}]}
            finally:
                exited.set()
    monkeypatch.setattr(quiz_generator, "_get_adapter", lambda _provider: HeldAdapter())
    with llm_call_root(37):
        work = asyncio.create_task(quiz_generator._call_quiz_generation_llm(prompt="Private facts", api_provider="llamacpp"))
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0)
        work.cancel()
        await asyncio.sleep(0.03)
        assert not work.done() and not exited.is_set()  # nosec B101 - async scope still owns the running provider.
    finally:
        release.set()
        try:
            await work
        except asyncio.CancelledError:
            pass
    assert exited.is_set() and work.cancelled() and owners == [37]  # nosec B101 - real provider drain and restored cancellation.
