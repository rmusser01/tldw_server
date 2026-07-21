"""RAG endpoint regressions for provider-override store outages."""

from __future__ import annotations

import asyncio
import copy
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import rag_unified as rag_endpoint
from tldw_Server_API.app.core.AuthNZ import byok_runtime, provider_credential_runtime
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides_module
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _simulate_incoming_unhealthy_override_cache():
    """Reproduce an unhealthy cache inherited from an earlier RAG test shard."""
    overrides_module.set_llm_provider_overrides_cache_for_tests({}, healthy=False)
    try:
        yield
    finally:
        overrides_module.set_llm_provider_overrides_cache_for_tests({})


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/rag/simple",
            "headers": [],
            "query_string": b"",
        }
    )


def test_rag_provider_override_fixture_starts_healthy_and_empty() -> None:
    """A failed lifespan refresh cannot poison the next RAG test."""
    assert overrides_module.get_llm_provider_overrides_snapshot() == {}


@pytest.mark.asyncio
async def test_rag_store_outage_is_503_before_static_fallback_or_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_calls: list[str] = []
    dispatch_calls: list[str] = []

    def static_fallback(provider: str, _snapshot: dict[str, Any]):
        static_calls.append(provider)
        return byok_runtime.ServerFallbackCredentials(
            api_key="must-not-be-used",
            credential_fields={},
            app_config={"openai_api": {"source": "static"}},
        )

    async def provider_boundary(*_args: Any, **kwargs: Any) -> list[Any]:
        runtime = kwargs["credential_runtime"]
        await runtime.resolve("openai")
        dispatch_calls.append("provider")
        return []

    async def no_usage_log(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        provider_credential_runtime,
        "resolve_static_server_fallback_from_snapshot",
        static_fallback,
    )
    monkeypatch.setattr(
        rag_endpoint,
        "_trusted_credential_runtime_scope",
        lambda *_args: (42, [], [], False),
    )
    monkeypatch.setattr(rag_endpoint, "_resolve_kanban_db_path", lambda _user: "kanban.db")
    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)
    monkeypatch.setattr(rag_endpoint, "simple_search", provider_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")},
        healthy=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await rag_endpoint.simple_search_endpoint(
            request=_request(),
            query="credential boundary",
            current_user=SimpleNamespace(id=42, id_int=42, username="rag-user"),
            media_db=SimpleNamespace(db_path="media.db"),
            chacha_db=SimpleNamespace(db_path="notes.db"),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "error_code": "credential_store_unavailable",
        "message": "Provider credential storage is temporarily unavailable.",
    }
    assert static_calls == []
    assert dispatch_calls == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_late_override_after_structured_absence_cannot_mix_rag_adapter_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG passes one static snapshot after an authoritative override absence."""
    structured_absent = threading.Event()
    static_lookup_started = threading.Event()
    release_static_lookup = threading.Event()
    static_providers: list[str] = []
    adapter_calls: list[dict[str, Any]] = []

    def capture_snapshot(provider: str):
        snapshot = overrides_module.capture_provider_override_call_snapshot(provider)
        structured_absent.set()
        return snapshot

    def gated_static_lookup(provider: str, _snapshot: dict[str, Any]):
        assert structured_absent.is_set()
        static_providers.append(provider)
        static_lookup_started.set()
        if not release_static_lookup.wait(10):
            raise TimeoutError("RAG static-key race gate was not released")
        return byok_runtime.ServerFallbackCredentials(
            api_key="configured-static-key",
            credential_fields={},
            app_config={},
        )

    async def provider_adapter_boundary(*_args: Any, **kwargs: Any) -> list[Any]:
        credentials = await kwargs["credential_runtime"].resolve("openai")
        adapter_calls.append(
            {
                "api_key": credentials.api_key,
                "app_config": credentials.app_config,
            }
        )
        return []

    async def no_usage_log(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        rag_endpoint,
        "capture_provider_override_call_snapshot",
        capture_snapshot,
    )
    monkeypatch.setattr(
        provider_credential_runtime,
        "resolve_static_server_fallback_from_snapshot",
        gated_static_lookup,
    )
    monkeypatch.setattr(
        rag_endpoint,
        "_trusted_credential_runtime_scope",
        lambda *_args: (42, [], [], False),
    )
    monkeypatch.setattr(rag_endpoint, "_resolve_kanban_db_path", lambda _user: "kanban.db")
    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)
    monkeypatch.setattr(rag_endpoint, "simple_search", provider_adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    def invoke_endpoint() -> dict[str, Any]:
        return asyncio.run(
            rag_endpoint.simple_search_endpoint(
                request=_request(),
                query="atomic credential snapshot",
                current_user=SimpleNamespace(id=42, id_int=42, username="rag-user"),
                media_db=SimpleNamespace(db_path="media.db"),
                chacha_db=SimpleNamespace(db_path="notes.db"),
            )
        )

    endpoint_task = asyncio.create_task(asyncio.to_thread(invoke_endpoint))
    structured_seen = await asyncio.to_thread(structured_absent.wait, 10)
    static_seen = await asyncio.to_thread(static_lookup_started.wait, 10)
    if structured_seen and static_seen:
        overrides_module.set_llm_provider_overrides_cache_for_tests(
            {
                "openai": LLMProviderOverride(
                    provider="openai",
                    api_key="late-override-key",
                    credential_fields={
                        "base_url": "https://late-override.example/v1",
                    },
                )
            }
        )
    release_static_lookup.set()
    result = await asyncio.wait_for(endpoint_task, timeout=10)

    assert structured_seen
    assert static_seen
    assert result["count"] == 0
    assert static_providers == ["openai"]
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == "configured-static-key"
    assert "late-override.example" not in repr(adapter_calls[0]["app_config"])


@pytest.mark.parametrize(
    ("initial_config", "expected_key", "expected_app_config"),
    (
        (
            {
                "openai_api": {
                    "api_key": "static-key-a",
                    "api_base_url": "https://static-a.example/v1",
                }
            },
            "static-key-a",
            {
                "openai_api": {
                    "api_base_url": "https://static-a.example/v1",
                }
            },
        ),
        ({}, None, None),
    ),
    ids=("a-to-b", "absent-to-b"),
)
@pytest.mark.asyncio
async def test_rag_static_fallback_freezes_one_config_generation_at_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    initial_config: dict[str, Any],
    expected_key: str | None,
    expected_app_config: dict[str, Any] | None,
) -> None:
    """RAG cannot combine an earlier static decision with a later config load."""
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    rotated_config = {
        "openai_api": {
            "api_key": "static-key-b",
            "api_base_url": "https://static-b.example/v1",
        }
    }
    legacy_lookup_finished = False
    dynamic_key_reads = 0
    adapter_calls: list[dict[str, Any]] = []

    def legacy_dynamic_keys() -> dict[str, str]:
        nonlocal dynamic_key_reads, legacy_lookup_finished
        dynamic_key_reads += 1
        legacy_lookup_finished = True
        monkeypatch.setattr(byok_runtime, "loaded_config_data", rotated_config)
        initial_section = initial_config.get("openai_api")
        initial_key = (
            initial_section.get("api_key")
            if isinstance(initial_section, dict)
            else None
        )
        return {"openai": initial_key} if isinstance(initial_key, str) else {}

    def load_static_snapshot() -> dict[str, Any]:
        selected = rotated_config if legacy_lookup_finished else initial_config
        monkeypatch.setattr(byok_runtime, "loaded_config_data", rotated_config)
        return copy.deepcopy(selected)

    async def provider_adapter_boundary(*_args: Any, **kwargs: Any) -> list[Any]:
        credentials = await kwargs["credential_runtime"].resolve("openai")
        adapter_calls.append(
            {
                "api_key": credentials.api_key,
                "app_config": credentials.app_config,
            }
        )
        return []

    async def no_usage_log(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(byok_helpers, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(
        provider_credential_runtime,
        "load_server_config_snapshot",
        load_static_snapshot,
    )
    monkeypatch.setattr(byok_runtime, "loaded_config_data", initial_config)
    monkeypatch.setattr(chat_request_schemas, "get_api_keys", legacy_dynamic_keys)
    monkeypatch.setattr(chat_request_schemas, "API_KEYS", {})
    monkeypatch.setattr(chat_endpoint, "API_KEYS", {})
    monkeypatch.setattr(
        rag_endpoint,
        "_trusted_credential_runtime_scope",
        lambda *_args: (42, [], [], False),
    )
    monkeypatch.setattr(rag_endpoint, "_resolve_kanban_db_path", lambda _user: "kanban.db")
    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)
    monkeypatch.setattr(rag_endpoint, "simple_search", provider_adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    result = await rag_endpoint.simple_search_endpoint(
        request=_request(),
        query="static generation boundary",
        current_user=SimpleNamespace(id=42, id_int=42, username="rag-user"),
        media_db=SimpleNamespace(db_path="media.db"),
        chacha_db=SimpleNamespace(db_path="notes.db"),
    )

    assert result["count"] == 0
    assert dynamic_key_reads == 0
    assert adapter_calls == [
        {
            "api_key": expected_key,
            "app_config": expected_app_config,
        }
    ]


@pytest.mark.parametrize(
    ("late_policy", "expected_code", "pass_model"),
    (
        ({"is_enabled": False, "allowed_models": ["gpt-4o-mini"]}, "provider_disabled", False),
        ({"is_enabled": True, "allowed_models": ["gpt-4.1"]}, "model_not_allowed", True),
    ),
)
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_late_rag_override_policy_change_blocks_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    late_policy: dict[str, Any],
    expected_code: str,
    pass_model: bool,
) -> None:
    """RAG resolves policy and credentials atomically at its adapter boundary."""
    adapter_ready = asyncio.Event()
    release_adapter = asyncio.Event()
    outbound_calls: list[str] = []

    async def provider_adapter_boundary(*_args: Any, **kwargs: Any) -> list[Any]:
        adapter_ready.set()
        await release_adapter.wait()
        resolve_kwargs = {"model": "gpt-4o-mini"} if pass_model else {}
        await kwargs["credential_runtime"].resolve("openai", **resolve_kwargs)
        outbound_calls.append("openai")
        return []

    async def no_usage_log(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        rag_endpoint,
        "_trusted_credential_runtime_scope",
        lambda *_args: (42, [], [], False),
    )
    monkeypatch.setattr(rag_endpoint, "_resolve_kanban_db_path", lambda _user: "kanban.db")
    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)
    monkeypatch.setattr(rag_endpoint, "simple_search", provider_adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=True,
                allowed_models=["gpt-4o-mini"],
                api_key="enabled-key",
            )
        }
    )

    endpoint_task = asyncio.create_task(
        rag_endpoint.simple_search_endpoint(
            request=_request(),
            query="late provider policy",
            current_user=SimpleNamespace(id=42, id_int=42, username="rag-user"),
            media_db=SimpleNamespace(db_path="media.db"),
            chacha_db=SimpleNamespace(db_path="notes.db"),
        )
    )
    await asyncio.wait_for(adapter_ready.wait(), timeout=10)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="late-policy-key",
                **late_policy,
            )
        }
    )
    release_adapter.set()

    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(endpoint_task, timeout=10)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == expected_code
    assert outbound_calls == []
