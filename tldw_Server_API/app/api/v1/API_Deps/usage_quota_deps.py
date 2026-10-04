"""FastAPI dependencies for per-user usage quotas (spec 2 §4)."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, status

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Usage.quota_checks import rag_queries_decision, seconds_until_utc_midnight


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
