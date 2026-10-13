"""FastAPI dependencies for per-user usage quotas (spec 2 §4)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import Depends, HTTPException, status

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage.llm_call_context import mark_llm_quota_prechecked
from tldw_Server_API.app.core.Usage.quota_checks import (
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


async def enforce_llm_quota(user_id: Any, estimated_tokens: int = 1) -> None:
    """402 limit_exceeded when the user is over an LLM quota; otherwise mark the request admitted (spec §4a)."""
    refusal = await llm_quota_refusal(user_id, estimated_tokens)
    if refusal is not None:
        raise LLMQuotaExceededError(**asdict(refusal))
    mark_llm_quota_prechecked()


async def require_llm_quota(current_user: User = Depends(get_request_user)) -> None:
    """``enforce_llm_quota`` as a dependency, for endpoints that always do LLM work."""
    await enforce_llm_quota(getattr(current_user, "id", None))
