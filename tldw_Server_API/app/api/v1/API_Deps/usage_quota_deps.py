"""FastAPI dependencies for per-user usage quotas (spec 2 §4)."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import asdict
from typing import Any

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage.llm_call_context import (
    USER_KEY_SOURCE,
    billed_credential_source,
    mark_llm_quota_prechecked,
)
from tldw_Server_API.app.core.Usage.quota_checks import (
    as_quota_user_id,
    llm_quota_refusal,
    rag_queries_decision,
    seconds_until_utc_midnight,
)


async def enforce_rag_query_quota(user_id: Any, units: int) -> None:
    """402 limit_exceeded when the user's daily RAG-query allowance would be exceeded."""
    decision = await rag_queries_decision(user_id, units)
    if decision.allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "error": "limit_exceeded",
            "category": "rag_queries_day",
            "current": int(decision.used),
            "limit": decision.limit,
            "message": "Daily RAG query limit reached",
        },
        headers={"Retry-After": str(seconds_until_utc_midnight())},
    )


def require_rag_query_quota(units: int = 1):
    """Dependency factory enforcing ``limits.rag_queries_per_day`` for the request user."""

    async def _check(current_user: User = Depends(get_request_user)) -> None:
        await enforce_rag_query_quota(getattr(current_user, "id", None), units)

    return _check


async def users_own_key_serves(user_id: Any, provider: str) -> bool:
    """True when a call to ``provider`` for this user would use the user's own stored key, billed to that key.

    The same first test credential resolution makes: BYOK on, the provider
    allowlisted, and an active key. A provider whose endpoint the operator runs
    (a custom OpenAI-compatible slot, a local server, a billed provider whose
    base URL is not its vendor's) answers False, as does a failed lookup.
    """
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
    from tldw_Server_API.app.core.Provider_Credentials.byok_helpers import (
        is_byok_enabled,
        is_provider_allowlisted,
        load_server_config_snapshot,
    )
    from tldw_Server_API.app.core.Provider_Credentials.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )

    uid = as_quota_user_id(user_id)
    try:
        name = canonical_provider_name(provider)
        if uid is None or billed_credential_source(name, USER_KEY_SOURCE) != USER_KEY_SOURCE:
            return False
        if not is_byok_enabled() or not is_provider_allowlisted(name):
            return False
        repo = AuthnzUserProviderSecretsRepo(await get_db_pool())
        if await repo.fetch_secret_for_active_user(uid, name) is None:
            return False
        # Last, as the dearest test: the endpoint the server's config gives this provider.
        snapshot = load_server_config_snapshot()
        return billed_credential_source(name, USER_KEY_SOURCE, snapshot) == USER_KEY_SOURCE
    except Exception as error:  # noqa: BLE001 - a failed lookup refuses as before
        logger.bind(exception_type=type(error).__name__).warning("Own-key lookup for the LLM quota failed")
        return False


async def enforce_llm_quota(
    user_id: Any,
    estimated_tokens: int = 1,
    *,
    provider: str | Collection[str | None] | None = None,
    own_key: bool | None = None,
) -> None:
    """402 limit_exceeded when the user is over an LLM quota; otherwise mark the request admitted (spec §4a).

    An over-limit user whose ``provider`` runs on their own key is let through
    unmarked: each call is then checked where its key is known, and a call on
    the user's own key is not checked (TASK-13955). ``provider`` names the
    provider of every LLM call the request makes: one name, or several (an
    evaluation batch's items, TASK-13956). A request that names none, or names
    one the user has no own key for, is refused. ``own_key`` is the caller's
    ``users_own_key_serves`` answer for a single provider, when it already has
    one; None looks it up, and so does a collection, whatever ``own_key`` says.
    """
    refusal = await llm_quota_refusal(user_id, estimated_tokens)
    if refusal is None:
        mark_llm_quota_prechecked()
        return
    names = [provider] if isinstance(provider, str) else list(provider or ())
    if not isinstance(provider, str):
        own_key = None  # a caller's answer is for one provider; several are each looked up
    if names and all(names) and (own_key if own_key is not None else await _own_key_serves_each(user_id, names)):
        return
    raise LLMQuotaExceededError(**asdict(refusal))


async def _own_key_serves_each(user_id: Any, providers: list[str]) -> bool:
    """``users_own_key_serves`` for each provider once, stopping at the first that is not the user's own key."""
    for name in dict.fromkeys(providers):
        if not await users_own_key_serves(user_id, name):
            return False
    return True


async def require_llm_quota(current_user: User = Depends(get_request_user)) -> None:
    """``enforce_llm_quota`` as a dependency, for endpoints that always do LLM work."""
    await enforce_llm_quota(getattr(current_user, "id", None))


async def enforce_llm_quota_if_the_router_was_refused(user_id: Any, routing_debug: Any) -> None:
    """The 402 when auto-routing found candidates but no route because its router call was refused at the call.

    The quota refused the router call, so the request gets the 402, not ``503 auto_routing_failed``
    (TASK-13956). A router call that failed for any other reason keeps the 503, for every user.
    """
    router = routing_debug.get("llm_router") if isinstance(routing_debug, dict) else None
    if isinstance(router, dict) and router.get("error") == LLMQuotaExceededError.__name__:
        await enforce_llm_quota(user_id)
