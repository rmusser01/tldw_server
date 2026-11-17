"""Usage logging and telemetry helpers for model router calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage

from .models import RoutingDecision


UsageLogger = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class RoutingUsageContext:
    surface: str
    endpoint: str
    user_id: int | None = None
    key_id: int | None = None
    request_id: str | None = None
    remote_ip: str | None = None
    user_agent: str | None = None
    token_name: str | None = None
    conversation_id: str | None = None


def get_router_operation_name(surface: str) -> str:
    normalized_surface = str(surface or "").strip().lower() or "llm"
    return f"{normalized_surface}_router"


def build_routing_telemetry_payload(
    *,
    decision: RoutingDecision,
    execution_provider: str | None = None,
    execution_model: str | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "decision_source": decision.decision_source,
        "fallback_used": fallback_used,
        "router_selected_provider": decision.provider,
        "router_selected_model": decision.model,
        "execution_provider": execution_provider,
        "execution_model": execution_model,
    }


async def log_model_router_usage(
    *,
    context: RoutingUsageContext,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    total_cost_usd: float = 0.0,
    latency_ms: int = 0,
    status: int = 200,
    estimated: bool = False,
    usage_logger: UsageLogger = log_llm_usage,
    credential_source: str | None = None,
) -> None:
    """Record an LLM router call as a distinct llm_usage_log operation.

    ``credential_source`` is whose key the router call used, as its caller resolved it;
    the LLM quotas leave out a ``user`` row (ADR-083).
    """

    prompt_tokens = int(prompt_tokens)
    completion_tokens = int(completion_tokens)
    total_tokens = int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens

    await usage_logger(
        user_id=context.user_id,
        key_id=context.key_id,
        endpoint=context.endpoint,
        operation=get_router_operation_name(context.surface),
        provider=provider,
        model=model,
        status=int(status),
        latency_ms=int(latency_ms),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated=bool(estimated),
        request_id=context.request_id,
        remote_ip=context.remote_ip,
        user_agent=context.user_agent,
        token_name=context.token_name,
        conversation_id=context.conversation_id,
        credential_source=credential_source,
    )
