from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.key_resolution import resolve_api_key_by_hash
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.virtual_keys import (
    get_key_limits,
    is_key_over_budget,
    summarize_usage_for_key_day,
    summarize_usage_for_key_month,
)

router = APIRouter()

_DEBUG_ALLOWED_ROLES = {"super_admin", "owner"}


async def require_debug_roles(
    principal: AuthPrincipal = Depends(get_auth_principal),  # noqa: B008
) -> AuthPrincipal:
    """
    Authorize access to AuthNZ debug endpoints.

    Single-user compatibility principals are allowed so the default self-hosted
    admin can continue using the debug tools. In multi-user mode, the caller
    must hold either the ``super_admin`` or ``owner`` role. Raises ``HTTP 403``
    when the principal does not satisfy either policy.
    """
    if getattr(principal, "subject", None) == "single_user":
        return principal

    roles = {str(role).strip().lower() for role in (principal.roles or []) if str(role).strip()}
    if roles & _DEBUG_ALLOWED_ROLES:
        return principal
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied. Required role(s): super_admin, owner",
    )


async def _resolve_api_key_id(request: Request, x_api_key: str | None) -> dict[str, Any]:
    # Prefer principal-first resolution from AuthContext, then legacy fallbacks.
    """
    Resolve an API key to its `api_key_id` and associated `user_id` for the incoming request.

    Prefers values from an `AuthContext` on `request.state` (principal-first), short-circuiting
    to `principal.api_key_id` and `principal.user_id` when available. If no suitable principal
    is present, falls back to legacy resolution: previously set `request.state.api_key_id` /
    `request.state.user_id`, then an explicit `x_api_key` parameter or a Bearer token in the
    Authorization header resolved via `resolve_api_key_by_hash`.

    Parameters:
        request (Request): The incoming FastAPI request; may contain an `AuthContext` on
            `request.state.auth` and/or pre-resolved `state.api_key_id` and `state.user_id`.
        x_api_key (Optional[str]): An explicit API key (typically from the X-API-KEY header)
            to resolve; if omitted, the Authorization header is inspected.

    Returns:
        dict: A mapping with keys:
            - "api_key_id": int or None - the resolved API key ID as an integer when found, otherwise None.
            - "user_id": Any or None - the associated user identifier when available, otherwise None.
    """
    # Prefer principal-first resolution when AuthContext is available.
    try:
        auth_ctx = getattr(request.state, "auth", None)
        if isinstance(auth_ctx, AuthContext):
            principal = getattr(auth_ctx, "principal", None)
            if isinstance(principal, AuthPrincipal):
                key_id = getattr(principal, "api_key_id", None)
                user_id = getattr(principal, "user_id", None)
                if key_id is not None:
                    return {"api_key_id": int(key_id), "user_id": user_id}
    except (AttributeError, TypeError, ValueError):
        # Fall back to legacy request.state attributes and header-based resolution.
        logger.debug("_resolve_api_key_id: principal-first resolution failed, falling back")

    key_id = getattr(request.state, "api_key_id", None)
    user_id = getattr(request.state, "user_id", None)
    if key_id:
        return {"api_key_id": int(key_id), "user_id": user_id}

    api_key = x_api_key
    if not api_key:
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if isinstance(auth, str) and auth.lower().startswith("bearer "):
            api_key = auth.split(" ", 1)[1].strip()

    if not api_key:
        return {"api_key_id": None, "user_id": None}

    result = await resolve_api_key_by_hash(api_key)
    if not result:
        return {"api_key_id": None, "user_id": None}

    return {"api_key_id": int(result["id"]), "user_id": result["user_id"]}


@router.get("/authnz/debug/api-key-id", tags=["authnz-debug"])
async def debug_api_key_id(
    request: Request,
    X_API_KEY: str | None = Header(None, alias="X-API-KEY"),
    _: AuthPrincipal = Depends(require_debug_roles),
):
    """
    Resolve the provided API key and return its associated api_key_id and user_id.

    This endpoint is for debugging and does not enforce authentication.

    Returns:
        result (dict): A dictionary with `"status": "ok"` plus `api_key_id` (int or None) and `user_id` (user identifier or None).
    """
    resolved = await _resolve_api_key_id(request, X_API_KEY)
    return {"status": "ok", **resolved}


@router.get("/authnz/debug/budget-summary", tags=["authnz-debug"])
async def debug_budget_summary(
    request: Request,
    X_API_KEY: str | None = Header(None, alias="X-API-KEY"),
    _: AuthPrincipal = Depends(require_debug_roles),
):
    """
    Provide limits, daily and monthly usage summaries, and an over-budget evaluation for the resolved API key.

    If no API key is resolved from the request or headers, returns a response with `"api_key_id": None` and a `"message"` explaining no key was resolved.

    Returns:
        dict: A response object containing:
            - status (str): Always `"ok"`.
            - api_key_id (int | None): The resolved API key ID, or `None` if no key was resolved.
            - message (str, optional): Present when no API key was resolved.
            - limits (Any): Key limits as returned by `get_key_limits`.
            - day (Any): Daily usage summary as returned by `summarize_usage_for_key_day`.
            - month (Any): Monthly usage summary as returned by `summarize_usage_for_key_month`.
            - over_budget (bool): `true` if the key is over its budget, `false` otherwise.
            - reasons (Any): Explanation or list of reasons from `is_key_over_budget`.
    """
    resolved = await _resolve_api_key_id(request, X_API_KEY)
    key_id = resolved.get("api_key_id")
    if not key_id:
        return {"status": "ok", "api_key_id": None, "message": "no api key resolved"}

    limits = await get_key_limits(int(key_id))
    day = await summarize_usage_for_key_day(int(key_id))
    month = await summarize_usage_for_key_month(int(key_id))
    decision = await is_key_over_budget(int(key_id))
    return {
        "status": "ok",
        "api_key_id": key_id,
        "limits": limits,
        "day": day,
        "month": month,
        "over_budget": decision.get("over"),
        "reasons": decision.get("reasons"),
    }
