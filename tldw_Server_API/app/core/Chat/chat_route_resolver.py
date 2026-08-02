"""Canonical provider/model route resolution shared by chat-like operations."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_runtime import resolve_byok_credentials
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    apply_llm_provider_overrides_to_listing,
    get_override_model_priority,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call_async,
    resolve_provider_and_model,
    resolve_provider_api_key,
)
from tldw_Server_API.app.core.LLM_Calls.routing import (
    InMemoryRoutingDecisionStore,
    RouterRequest,
    RoutingDecision,
    RoutingUsageContext,
    build_provider_order_for_routing,
    flatten_provider_listing_for_routing,
    log_model_router_usage,
    resolve_routing_policy,
    route_model,
    select_llm_router_choice,
)
from tldw_Server_API.app.core.LLM_Calls.routing.candidate_pool import (
    build_candidate_pool,
)

_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)


@dataclass(frozen=True)
class ResolvedChatRoute:
    """The provider/model pair selected for a chat-compatible request."""

    provider: str
    model: str
    was_auto: bool
    routing_decision: RoutingDecision | None
    debug: dict[str, Any]
    metrics_provider: str
    metrics_model: str


class ChatRouteResolutionError(RuntimeError):
    """A routing failure that callers can map to their public error contract."""

    def __init__(self, code: str, message: str, *, debug: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.debug = debug or {}


def _request_uses_vision_input(messages: list[Any]) -> bool:
    """Return whether messages contain an image-content part."""
    for message in messages or []:
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                if str(part.get("type") or "").strip().lower() == "image_url":
                    return True
                continue
            if str(getattr(part, "type", "") or "").strip().lower() == "image_url":
                return True
    return False


def extract_routing_requested_capabilities(request_data: Any) -> dict[str, Any]:
    """Derive hard capability filters from a chat-compatible request."""
    response_format = getattr(request_data, "response_format", None)
    if isinstance(response_format, dict):
        response_type = str(response_format.get("type") or "").strip().lower()
    else:
        response_type = str(getattr(response_format, "type", "") or "").strip().lower()

    return {
        "tools": bool(getattr(request_data, "tools", None)),
        "vision": _request_uses_vision_input(getattr(request_data, "messages", [])),
        "json_mode": response_type in {"json_object", "json_schema"},
        "reasoning": bool(getattr(request_data, "thinking_budget_tokens", None)),
    }


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"].strip())
            continue
        if getattr(part, "type", None) == "text" and isinstance(getattr(part, "text", None), str):
            text_parts.append(part.text.strip())
    return "\n".join(part for part in text_parts if part).strip()


def _extract_latest_user_turn_text(messages: list[Any]) -> str:
    """Return the most recent textual user turn from request messages."""
    for message in reversed(messages or []):
        role = getattr(message, "role", None) if not isinstance(message, dict) else message.get("role")
        if role != "user":
            continue
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        text = _extract_text_from_message_content(content)
        if text:
            return text
    return ""


async def _select_auto_llm_router_choice(
    *,
    router_request: RouterRequest,
    policy: Any,
    candidates: list[dict[str, Any]],
    provider_listing: dict[str, Any],
    request: Any,
    current_user: Any | None,
    request_id: str | None,
    surface: str,
    endpoint: str,
    resolve_provider_api_key_fn: Callable[..., tuple[str | None, dict[str, Any]]],
    resolve_byok_credentials_fn: Callable[..., Any],
    perform_chat_api_call_async_fn: Callable[..., Any],
    log_model_router_usage_fn: Callable[..., Any],
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    def fallback_resolver(name: str) -> str | None:
        key_value, _ = resolve_provider_api_key_fn(name, prefer_module_keys_in_tests=True)
        return key_value

    user_id_int = getattr(current_user, "id_int", None)
    if user_id_int is None:
        try:
            user_id_int = int(getattr(current_user, "id", None))
        except _NONCRITICAL_EXCEPTIONS:
            user_id_int = None

    try:
        request_state = getattr(request, "state", None)
        user_id = getattr(request_state, "user_id", None)
        api_key_id = getattr(request_state, "api_key_id", None)
    except _NONCRITICAL_EXCEPTIONS:
        user_id = None
        api_key_id = None

    async def execute_router_call(router_model: Any, router_messages: list[dict[str, str]]) -> Any:
        byok_resolution = await resolve_byok_credentials_fn(
            router_model.provider,
            user_id=user_id_int,
            request=request,
            fallback_resolver=fallback_resolver,
        )
        try:
            return await perform_chat_api_call_async_fn(
                api_endpoint=router_model.provider,
                messages_payload=router_messages,
                api_key=byok_resolution.api_key,
                model=router_model.model,
                max_tokens=64,
                streaming=False,
                user_identifier=str(getattr(current_user, "id", "auto-router")),
                app_config=byok_resolution.app_config,
            )
        finally:
            await byok_resolution.touch_last_used()

    async def log_router_usage(router_model: Any, usage: dict[str, int], latency_ms: int) -> None:
        try:
            await log_model_router_usage_fn(
                context=RoutingUsageContext(
                    surface=surface,
                    endpoint=endpoint,
                    user_id=user_id,
                    key_id=api_key_id,
                    request_id=request_id,
                    conversation_id=router_request.scope,
                ),
                provider=router_model.provider,
                model=router_model.model,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                latency_ms=latency_ms,
                estimated=usage["total_tokens"] == 0,
            )
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Auto chat router usage logging skipped: {}", exc)

    try:
        return await select_llm_router_choice(
            router_request=router_request,
            policy=policy,
            candidates=candidates,
            provider_listing=provider_listing,
            execute_router_call=execute_router_call,
            log_router_usage=log_router_usage,
        )
    except _NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Auto chat LLM router call failed: {}", exc)
        return None, {"error": type(exc).__name__}


async def resolve_chat_route(
    request_data: Any,
    *,
    request: Any,
    sticky_store: InMemoryRoutingDecisionStore,
    current_user: Any | None,
    request_id: str | None,
    configured_providers_getter: Callable[[], dict[str, Any]],
    surface: str = "chat",
    endpoint: str = "POST:/api/v1/chat/completions",
    scope: str | None = None,
    latest_user_turn: str | None = None,
    requested_capabilities: dict[str, Any] | None = None,
    default_provider: str | None = None,
    metrics_default_provider: str | None = None,
    route_model_fn: Callable[..., RoutingDecision | None] = route_model,
    resolve_provider_and_model_fn: Callable[..., tuple[str, str, str, str, dict[str, Any]]] = resolve_provider_and_model,
    resolve_provider_api_key_fn: Callable[..., tuple[str | None, dict[str, Any]]] = resolve_provider_api_key,
    resolve_byok_credentials_fn: Callable[..., Any] = resolve_byok_credentials,
    perform_chat_api_call_async_fn: Callable[..., Any] = perform_chat_api_call_async,
    log_model_router_usage_fn: Callable[..., Any] = log_model_router_usage,
    priority_resolver: Callable[..., list[str] | None] = get_override_model_priority,
    apply_provider_overrides_fn: Callable[[dict[str, Any]], dict[str, Any]] = apply_llm_provider_overrides_to_listing,
) -> ResolvedChatRoute:
    """Resolve concrete and automatic chat routes through the canonical policy path."""
    was_auto = str(getattr(request_data, "model", "") or "").strip().lower() == "auto"
    route_scope = scope if scope is not None else getattr(request_data, "conversation_id", None)
    capabilities = requested_capabilities or extract_routing_requested_capabilities(request_data)
    route_debug: dict[str, Any] = {"requested_capabilities": capabilities}
    decision: RoutingDecision | None = None

    if was_auto:
        provider_listing = apply_provider_overrides_fn(configured_providers_getter())
        resolved_default_provider = str(
            provider_listing.get("default_provider") or default_provider or ""
        ).strip().lower()
        policy = resolve_routing_policy(
            request_model="auto",
            explicit_provider=getattr(request_data, "api_provider", None),
            routing_override=getattr(request_data, "routing", None),
            server_default_provider=resolved_default_provider,
        )
        candidates = build_candidate_pool(
            boundary_mode=policy.boundary_mode,
            pinned_provider=policy.pinned_provider,
            server_default_provider=policy.server_default_provider,
            requested_capabilities=capabilities,
            catalog=flatten_provider_listing_for_routing(provider_listing),
        )
        router_request = RouterRequest(
            model="auto",
            surface=surface,
            latest_user_turn=(
                latest_user_turn
                if latest_user_turn is not None
                else _extract_latest_user_turn_text(getattr(request_data, "messages", []))
            ),
            scope=route_scope,
            requested_capabilities=capabilities,
            routing_context={
                "stream": bool(getattr(request_data, "stream", False)),
                "response_format": bool(getattr(request_data, "response_format", None)),
            },
        )
        llm_router_choice, llm_router_debug = await _select_auto_llm_router_choice(
            router_request=router_request,
            policy=policy,
            candidates=candidates,
            provider_listing=provider_listing,
            request=request,
            current_user=current_user,
            request_id=request_id,
            surface=surface,
            endpoint=endpoint,
            resolve_provider_api_key_fn=resolve_provider_api_key_fn,
            resolve_byok_credentials_fn=resolve_byok_credentials_fn,
            perform_chat_api_call_async_fn=perform_chat_api_call_async_fn,
            log_model_router_usage_fn=log_model_router_usage_fn,
        )
        decision = route_model_fn(
            request=router_request,
            policy=policy,
            candidates=candidates,
            sticky_store=sticky_store,
            llm_router_choice=llm_router_choice,
            provider_order=build_provider_order_for_routing(
                provider_listing,
                objective=policy.objective,
                priority_resolver=priority_resolver,
            ),
        )
        route_debug.update(
            {
                "policy": {
                    "boundary_mode": policy.boundary_mode,
                    "pinned_provider": policy.pinned_provider,
                    "server_default_provider": policy.server_default_provider,
                    "objective": policy.objective,
                    "mode": policy.mode,
                    "strategy": policy.strategy,
                    "failure_mode": policy.failure_mode,
                },
                "candidate_count": len(candidates),
                "llm_router": llm_router_debug,
            }
        )
        if decision is None:
            if candidates:
                raise ChatRouteResolutionError(
                    "auto_routing_failed",
                    "Auto-routing failed and the current routing policy did not allow deterministic fallback.",
                    debug=route_debug,
                )
            raise ChatRouteResolutionError(
                "auto_routing_no_candidates",
                "No eligible models matched the current auto-routing constraints.",
                debug=route_debug,
            )

    resolved_default_provider = str(default_provider or "").strip()
    metrics_provider, metrics_model, provider, model, provider_debug = resolve_provider_and_model_fn(
        request_data=request_data,
        metrics_default_provider=metrics_default_provider or resolved_default_provider or "local-llm",
        normalize_default_provider=resolved_default_provider or "local-llm",
        routing_decision=decision,
    )
    route_debug["provider_resolution"] = provider_debug
    return ResolvedChatRoute(
        provider=provider,
        model=model,
        was_auto=was_auto,
        routing_decision=decision,
        debug=route_debug,
        metrics_provider=metrics_provider,
        metrics_model=metrics_model,
    )
