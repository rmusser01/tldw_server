"""Concurrency regressions for atomic provider-override routing snapshots."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import (
    character_chat_sessions as character_chat_endpoint,
)
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
)
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
    CharacterChatCompletionV2Request,
)
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides_module
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.core.LLM_Calls.routing.decision_store import (
    InMemoryRoutingDecisionStore,
)
from tldw_Server_API.app.core.LLM_Calls.routing.runtime import (
    build_provider_order_for_routing as real_build_provider_order,
)

pytestmark = [pytest.mark.integration, pytest.mark.concurrent]


def _override(model_order: list[str]) -> LLMProviderOverride:
    return LLMProviderOverride(
        provider="openai",
        is_enabled=True,
        allowed_models=["model-a", "model-b"],
        config={
            "routing": {
                "model_rankings": {
                    "highest_quality": model_order,
                }
            }
        },
    )


def _provider_listing() -> dict[str, Any]:
    return {
        "default_provider": "openai",
        "providers": [
            {
                "name": "openai",
                "is_configured": True,
                "models": ["model-b", "model-a"],
                "models_info": [{"name": "model-b"}, {"name": "model-a"}],
            }
        ],
    }


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
        }
    )


def _gated_provider_order_builder(
    *,
    ready: threading.Event,
    release: threading.Event,
    captured: dict[str, Any],
) -> Callable[..., dict[str, list[str]]]:
    def build(provider_listing, *, objective, priority_resolver):
        captured["listing"] = provider_listing
        ready.set()
        if not release.wait(10):
            raise TimeoutError("provider-order rotation gate was not released")
        captured["priority"] = priority_resolver("openai", objective)
        return real_build_provider_order(
            provider_listing,
            objective=objective,
            priority_resolver=priority_resolver,
        )

    return build


def _tracked_override_reads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    reads: list[list[str]] = []
    real_snapshot = overrides_module._get_healthy_override_snapshot

    def tracked(provider: str = "provider-overrides"):
        snapshot = real_snapshot(provider)
        override = snapshot.get("openai")
        reads.append(
            list(
                override.config["routing"]["model_rankings"]["highest_quality"]
            )
            if override is not None
            else []
        )
        return snapshot

    monkeypatch.setattr(overrides_module, "_get_healthy_override_snapshot", tracked)
    return reads


def _capture_route_inputs(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_module: Any,
    captured: dict[str, Any],
) -> None:
    real_route_model = endpoint_module.route_model

    def route_model(**kwargs):
        captured["candidates"] = [candidate.model for candidate in kwargs["candidates"]]
        captured["provider_order"] = kwargs["provider_order"]
        return real_route_model(**kwargs)

    monkeypatch.setattr(endpoint_module, "route_model", route_model)


def test_chat_auto_routing_keeps_listing_candidates_and_priority_on_one_override_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = overrides_module.get_llm_provider_overrides_snapshot()
    ready = threading.Event()
    release = threading.Event()
    captured: dict[str, Any] = {}
    reads = _tracked_override_reads(monkeypatch)

    async def no_llm_router(**_kwargs):
        return None, {"skipped": "test"}

    monkeypatch.setattr(chat_endpoint, "get_configured_providers", _provider_listing)
    monkeypatch.setattr(
        chat_endpoint,
        "_select_auto_chat_llm_router_choice",
        no_llm_router,
    )
    monkeypatch.setattr(
        chat_endpoint,
        "build_provider_order_for_routing",
        _gated_provider_order_builder(
            ready=ready,
            release=release,
            captured=captured,
        ),
    )
    _capture_route_inputs(monkeypatch, chat_endpoint, captured)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": _override(["model-a", "model-b"])}
    )

    request_data = ChatCompletionRequest(
        api_provider="openai",
        model="auto",
        routing={"strategy": "rules_router"},
        messages=[{"role": "user", "content": "route atomically"}],
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                chat_endpoint._resolve_auto_chat_routing_decision(
                    request_data,
                    request=_request("/api/v1/chat/completions"),
                    sticky_store=InMemoryRoutingDecisionStore(),
                    current_user=None,
                    request_id="chat-routing-snapshot",
                    credential_runtime=object(),
                ),
            )
            assert ready.wait(10)
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                {"openai": _override(["model-b", "model-a"])}
            )
            release.set()
            decision, _debug = future.result(timeout=10)
    finally:
        release.set()
        overrides_module.set_llm_provider_overrides_cache_for_tests(original)

    assert reads == [["model-a", "model-b"]]
    assert [
        item["name"] for item in captured["listing"]["providers"][0]["models_info"]
    ] == ["model-a", "model-b"]
    assert captured["candidates"] == ["model-a", "model-b"]
    assert captured["provider_order"] == {"openai": ["model-a", "model-b"]}
    assert captured["priority"] == ["model-a", "model-b"]
    assert decision is not None
    assert decision.model == "model-a"


def test_character_chat_auto_routing_keeps_listing_candidates_and_priority_on_one_override_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = overrides_module.get_llm_provider_overrides_snapshot()
    ready = threading.Event()
    release = threading.Event()
    captured: dict[str, Any] = {}
    reads = _tracked_override_reads(monkeypatch)

    async def no_llm_router(**_kwargs):
        return None, {"skipped": "test"}

    monkeypatch.setattr(
        character_chat_endpoint,
        "get_configured_providers",
        _provider_listing,
    )
    monkeypatch.setattr(
        character_chat_endpoint,
        "_select_auto_character_llm_router_choice",
        no_llm_router,
    )
    monkeypatch.setattr(
        character_chat_endpoint,
        "build_provider_order_for_routing",
        _gated_provider_order_builder(
            ready=ready,
            release=release,
            captured=captured,
        ),
    )
    _capture_route_inputs(monkeypatch, character_chat_endpoint, captured)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": _override(["model-a", "model-b"])}
    )

    body = CharacterChatCompletionV2Request(
        provider="openai",
        model="auto",
        routing={"strategy": "rules_router"},
        append_user_message="route atomically",
        include_character_context=False,
        save_to_db=False,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                character_chat_endpoint._resolve_auto_character_chat_routing_decision(
                    chat_id="character-routing-snapshot",
                    body=body,
                    raw_provider="openai",
                    formatted_messages=[
                        {"role": "user", "content": "route atomically"}
                    ],
                    sticky_store=InMemoryRoutingDecisionStore(),
                    current_user=None,
                    credential_runtime=object(),
                ),
            )
            assert ready.wait(10)
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                {"openai": _override(["model-b", "model-a"])}
            )
            release.set()
            decision, _debug = future.result(timeout=10)
    finally:
        release.set()
        overrides_module.set_llm_provider_overrides_cache_for_tests(original)

    assert reads == [["model-a", "model-b"]]
    assert [
        item["name"] for item in captured["listing"]["providers"][0]["models_info"]
    ] == ["model-a", "model-b"]
    assert captured["candidates"] == ["model-a", "model-b"]
    assert captured["provider_order"] == {"openai": ["model-a", "model-b"]}
    assert captured["priority"] == ["model-a", "model-b"]
    assert decision is not None
    assert decision.model == "model-a"
