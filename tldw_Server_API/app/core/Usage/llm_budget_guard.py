from __future__ import annotations

import os

from fastapi import HTTPException, Request
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.auth_governor import get_auth_governor
from tldw_Server_API.app.core.AuthNZ.key_resolution import resolve_api_key_by_hash
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.testing import env_flag_enabled


async def enforce_llm_budget(request: Request) -> None:
    """
    Enforces LLM budgets for virtual API keys on incoming requests.

    If budget enforcement is enabled and the request is associated with an API key that is marked virtual, verifies the key's token and USD limits and raises HTTPException(402) with a detail payload when the key is over its budget. This dependency is a no-op for requests without an API key or when virtual keys or budget enforcement are disabled.
    """
    settings = get_settings()
    _dbg = (
        env_flag_enabled("BUDGET_MW_DEBUG")
        or os.getenv("PYTEST_CURRENT_TEST") is not None
    )
    if not getattr(settings, "VIRTUAL_KEYS_ENABLED", True):
        return
    if not getattr(settings, "LLM_BUDGET_ENFORCE", True):
        return

    # Already bound by earlier auth dependency?
    key_id: int | None = getattr(request.state, "api_key_id", None)

    # Otherwise, resolve from headers deterministically
    if not key_id:
        api_key = request.headers.get("X-API-KEY") or request.headers.get("x-api-key")
        if not api_key:
            auth = request.headers.get("authorization") or request.headers.get("Authorization")
            if isinstance(auth, str) and auth.lower().startswith("bearer "):
                api_key = auth.split(" ", 1)[1].strip()

        if api_key:
            info = await resolve_api_key_by_hash(api_key, settings=settings)
            if info:
                key_id = info.get("id")
                try:
                    request.state.api_key_id = key_id
                    request.state.user_id = info.get("user_id")
                except Exception as exc:
                    # Do not proceed silently with missing auth state; log with context and stop.
                    user_id_value = info.get("user_id") if isinstance(info, dict) else None
                    path = getattr(getattr(request, "url", None), "path", None) or request.scope.get("path")
                    logger.exception(
                        "LLM guard: failed to set request.state attributes (path={path}, api_key_id={key_id}, user_id={user_id})",
                        path=path,
                        key_id=key_id,
                        user_id=user_id_value,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "internal_state_error",
                            "message": "Failed to attach authorization context to request state",
                            "details": {
                                "path": path,
                                "attributes": ["api_key_id", "user_id"],
                            },
                        },
                    ) from exc
            elif _dbg:
                logger.debug("LLM guard: no key_id found via hash lookup")

    if not key_id:
        # JWT or anonymous path; budgets apply only to API keys
        if _dbg:
            logger.debug("LLM guard: skipping budget enforcement (no api_key_id)")
        return

    # Construct an AuthPrincipal for governance decisions. Prefer an existing
    # AuthContext when available; otherwise derive a minimal principal from
    # request.state.
    principal: AuthPrincipal | None = None
    existing_ctx = getattr(request.state, "auth", None)
    if isinstance(existing_ctx, AuthContext):
        principal = existing_ctx.principal
    if principal is None:
        # Best-effort principal derived from request.state attributes
        user_id_value = getattr(request.state, "user_id", None)
        try:
            user_id_int = int(user_id_value) if user_id_value is not None else None
        except (TypeError, ValueError):
            user_id_int = None

        def _coerce_int_list(raw):
            try:
                if isinstance(raw, (list, tuple)):
                    return [int(x) for x in raw if x is not None]
            except (TypeError, ValueError):
                pass
            return []

        org_ids = _coerce_int_list(getattr(request.state, "org_ids", None))
        team_ids = _coerce_int_list(getattr(request.state, "team_ids", None))
        principal = AuthPrincipal(
            kind="api_key",
            user_id=user_id_int,
            api_key_id=key_id,
            subject=None,
            token_type="api_key",
            jti=None,
            roles=[],
            permissions=[],
            is_admin=False,
            org_ids=org_ids,
            team_ids=team_ids,
        )

    auth_gov = await get_auth_governor()
    result = await auth_gov.check_llm_budget_for_api_key(principal, key_id)

    limits = result.get("limits") or {}
    if not limits or not limits.get("is_virtual"):
        if _dbg:
            logger.debug(f"LLM guard: key {key_id} not virtual or limits missing; skipping")
        return

    if _dbg:
        debug_limits = result.get("limits", {}) or {}
        subset = {
            "llm_budget_day_tokens": debug_limits.get("llm_budget_day_tokens"),
            "llm_budget_day_usd": debug_limits.get("llm_budget_day_usd"),
            "llm_budget_month_tokens": debug_limits.get("llm_budget_month_tokens"),
            "llm_budget_month_usd": debug_limits.get("llm_budget_month_usd"),
        }
        logger.debug(
            f"LLM guard: over_budget={result.get('over')} reasons={result.get('reasons')} "
            f"day={result.get('day')} month={result.get('month')} limits={subset}"
        )
    if result.get("over"):
        raise HTTPException(status_code=402, detail={
            "error": "budget_exceeded",
            "message": "Virtual key budget exceeded",
            "details": result,
        })
