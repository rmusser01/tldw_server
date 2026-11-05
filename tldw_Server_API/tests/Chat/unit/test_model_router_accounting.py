from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls.routing.accounting import (
    RoutingUsageContext,
    build_routing_telemetry_payload,
    get_router_operation_name,
    log_model_router_usage,
)
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision


@pytest.mark.unit
def test_get_router_operation_name_is_surface_specific():
    assert get_router_operation_name("chat") == "chat_router"


@pytest.mark.unit
def test_build_routing_telemetry_payload_separates_router_and_execution_models():
    payload = build_routing_telemetry_payload(
        decision=RoutingDecision(provider="openai", model="gpt-4.1", decision_source="llm_router"),
        execution_provider="openai",
        execution_model="gpt-4.1-mini",
        fallback_used=True,
    )

    assert payload["router_selected_provider"] == "openai"
    assert payload["router_selected_model"] == "gpt-4.1"
    assert payload["execution_provider"] == "openai"
    assert payload["execution_model"] == "gpt-4.1-mini"
    assert payload["fallback_used"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_log_model_router_usage_records_router_operation():
    calls: list[dict[str, object]] = []

    async def fake_usage_logger(**kwargs: Any) -> None:
        calls.append(kwargs)

    await log_model_router_usage(
        context=RoutingUsageContext(
            surface="chat",
            endpoint="POST:/api/v1/chat/completions",
            user_id=1,
            key_id=2,
            request_id="req-router",
            conversation_id="conv-1",
        ),
        provider="openai",
        model="gpt-4.1-mini",
        prompt_tokens=10,
        completion_tokens=2,
        total_cost_usd=0.001,
        usage_logger=fake_usage_logger,
    )

    assert len(calls) == 1
    assert calls[0]["operation"] == "chat_router"
    assert calls[0]["provider"] == "openai"
    assert calls[0]["model"] == "gpt-4.1-mini"
    assert calls[0]["conversation_id"] == "conv-1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_log_model_router_usage_matches_log_llm_usage_signature():
    calls: list[dict[str, object]] = []

    async def strict_usage_logger(
        *,
        user_id,
        key_id,
        endpoint,
        operation,
        provider,
        model,
        status,
        latency_ms,
        prompt_tokens,
        completion_tokens,
        total_tokens=None,
        currency="USD",
        request_id=None,
        estimated=None,
        request=None,
        remote_ip=None,
        user_agent=None,
        token_name=None,
        conversation_id=None,
        credential_source=None,
    ):
        calls.append(
            {
                "user_id": user_id,
                "key_id": key_id,
                "endpoint": endpoint,
                "operation": operation,
                "provider": provider,
                "model": model,
                "status": status,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "currency": currency,
                "request_id": request_id,
                "estimated": estimated,
                "request": request,
                "remote_ip": remote_ip,
                "user_agent": user_agent,
                "token_name": token_name,
                "conversation_id": conversation_id,
                "credential_source": credential_source,
            }
        )

    await log_model_router_usage(
        context=RoutingUsageContext(
            surface="chat",
            endpoint="POST:/api/v1/chat/completions",
        ),
        provider="openai",
        model="gpt-4.1-mini",
        total_cost_usd=0.001,
        usage_logger=strict_usage_logger,
    )

    assert len(calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["user", "server_default", None])
async def test_log_model_router_usage_records_whose_key_the_router_call_used(source):
    """The router's row carries the router call's own source, so a call on the user's key does not count (TASK-13956)."""
    calls: list[dict[str, object]] = []

    async def fake_usage_logger(**kwargs: Any) -> None:
        calls.append(kwargs)

    await log_model_router_usage(
        context=RoutingUsageContext(surface="chat", endpoint="POST:/api/v1/chat/completions", user_id=1),
        provider="openai",
        model="gpt-4.1-mini",
        prompt_tokens=10,
        completion_tokens=2,
        usage_logger=fake_usage_logger,
        credential_source=source,
    )

    assert [(call["operation"], call["credential_source"]) for call in calls] == [("chat_router", source)]
