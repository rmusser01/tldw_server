from __future__ import annotations

import json
import os
from typing import Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from tldw_Server_API.app.core.AuthNZ.auth_governor import get_auth_governor
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip
from tldw_Server_API.app.core.AuthNZ.key_resolution import resolve_api_key_by_hash
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, get_settings_generation
from tldw_Server_API.app.core.AuthNZ.virtual_keys import get_key_limits
from tldw_Server_API.app.core.LLM_Calls.anthropic_model_ids import allowlist_has_model
from tldw_Server_API.app.core.testing import env_flag_enabled

_LLM_BUDGET_PARSE_EXCEPTIONS = (TypeError, ValueError, UnicodeError, json.JSONDecodeError)
_LLM_BUDGET_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeError,
    json.JSONDecodeError,
)


class LLMBudgetMiddleware(BaseHTTPMiddleware):
    """
    Enforce Virtual Key endpoint allowlists and LLM budgets for API-key authenticated requests.
    Applies only to configured LLM endpoints.
    """

    def __init__(self, app):
        """
        Initialize the LLMBudgetMiddleware and prepare per-request settings handling.

        Parameters:
            app: The ASGI application to wrap.

        Notes:
            Uses a lightweight settings cache with a generation-based invalidation
            (see AuthNZ.settings.get_settings_generation). This avoids repeated
            calls per request in production while allowing tests to call
            reset_settings() to invalidate the cache deterministically.
        """
        super().__init__(app)
        # Lightweight settings cache with explicit invalidation via generation
        self._settings_cache = None
        self._settings_gen = -1

    def _get_settings_cached(self):
        """Return cached settings, refreshing when generation changes.

        Uses get_settings_generation() so tests that call reset_settings() will
        still invalidate the cache. Falls back to direct get_settings() when
        generation is unavailable.
        """
        try:
            gen = get_settings_generation()
        except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS:
            gen = -1
        if self._settings_cache is None or (gen >= 0 and gen != self._settings_gen):
            self._settings_cache = get_settings()
            self._settings_gen = gen
        return self._settings_cache

    def _set_key_state(self, request: Request, key_id, user_id):
        """Helper to safely set API key state on request with robust error handling.

        Ensures both `api_key_id` and `user_id` are attached to `request.state`.
        On failure, logs with context and returns a JSONResponse(500) to avoid
        BaseHTTPMiddleware task group bubbling. Callers should return the
        response when not None.
        """
        try:
            request.state.api_key_id = key_id
            request.state.user_id = user_id
        except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS as _state_exc:
            try:
                path_ctx = getattr(getattr(request, "url", None), "path", None) or request.scope.get("path")
            except (AttributeError, KeyError, TypeError):
                path_ctx = None
            logger.exception(
                "LLM budget: failed to set request.state attributes (path={path}, api_key_id={key_id}, user_id={user_id})",
                path=path_ctx,
                key_id=key_id,
                user_id=user_id,
            )
            return JSONResponse({
                "error": "internal_state_error",
                "message": "Failed to attach authorization context to request state",
                "details": {
                    "path": path_ctx,
                    "attributes": ["api_key_id", "user_id"],
                },
            }, status_code=500)

    def _should_check(self, path: str) -> bool:
        """
        Determine whether LLM budget and virtual-key enforcement should be applied to the given request path.

        Parameters:
            path (str): The request URL path to evaluate (e.g., "/api/v1/chat/completions").

        Returns:
            `True` if enforcement should be applied to the provided request path, `False` otherwise.
        """
        try:
            settings = self._get_settings_cached()
            if not getattr(settings, 'VIRTUAL_KEYS_ENABLED', True):
                return False
            if not getattr(settings, 'LLM_BUDGET_ENFORCE', True):
                return False
            endpoints = getattr(settings, 'LLM_BUDGET_ENDPOINTS', None)
            if not isinstance(endpoints, (list, tuple, set)) or len(endpoints) == 0:
                # Fallback to sane defaults if overrides are missing/malformed
                endpoints = [
                    "/api/v1/chat/completions",
                    "/api/v1/embeddings",
                ]
            return any(isinstance(p, str) and path.startswith(p) for p in endpoints)
        except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS:
            return False

    @staticmethod
    def _endpoint_code(path: str) -> str:
        # simple map for v1
        if path.startswith('/api/v1/chat/completions'):
            return 'chat.completions'
        if path.startswith('/api/v1/embeddings'):
            return 'embeddings'
        return path

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Enforces virtual API-key LLM allowlists and budget limits for incoming requests to configured LLM endpoints.

        If the request targets a monitored LLM endpoint and the resolved API key is a virtual key, this middleware will validate endpoint, provider, and model allowlists and check the key's budget; it forwards the request unchanged when enforcement is not applicable or passes all checks.

        Returns:
            Response: An HTTP response. May be a 403 JSONResponse when the key is forbidden, a 402 JSONResponse when the virtual key is over budget, or the downstream handler's Response when the request is allowed or enforcement is skipped.
        """
        path = request.url.path
        if not self._should_check(path):
            return await call_next(request)

        # Optional debug toggle for diagnosis. Enabled automatically in pytest contexts.
        _mw_debug = (
            env_flag_enabled("BUDGET_MW_DEBUG")
            or os.getenv("PYTEST_CURRENT_TEST") is not None
        )
        if _mw_debug:
            try:
                settings = self._get_settings_cached()
                logger.debug(
                    f"LLM budget dispatch path={path} enforce={getattr(settings,'LLM_BUDGET_ENFORCE', True)} vkeys={getattr(settings,'VIRTUAL_KEYS_ENABLED', True)}"
                )
            except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS:
                logger.debug(f"LLM budget dispatch path={path} (settings unavailable)")

        # Resolve key_id deterministically from header first (DB hash lookup),
        # then fall back to manager validation if needed. This avoids cases
        # where a stale singleton or init-order peculiarity misses the key.
        key_id = getattr(request.state, 'api_key_id', None)
        if not key_id:
            # Read API key from either X-API-KEY or Authorization: Bearer
            api_key = request.headers.get('X-API-KEY') or request.headers.get('x-api-key')
            if not api_key:
                auth = request.headers.get('authorization') or request.headers.get('Authorization')
                if isinstance(auth, str) and auth.lower().startswith('bearer '):
                    api_key = auth.split(' ', 1)[1].strip()
            if _mw_debug:
                redacted = (api_key[:8] + "…") if api_key else None
                logger.debug(f"LLM budget: resolving api_key via headers -> {bool(api_key)} ({redacted})")

            # 1) Direct DB lookup by HMAC(hash)
            if api_key:
                try:
                    info = await resolve_api_key_by_hash(api_key, settings=self._get_settings_cached())
                    if info:
                        key_id = info.get('id')
                        _resp = self._set_key_state(request, key_id, info.get('user_id'))
                        if _resp is not None:
                            return _resp
                        if _mw_debug:
                            logger.debug(f"LLM budget: resolved key_id via hash lookup: {key_id}")
                    elif _mw_debug:
                        logger.debug("LLM budget: hash lookup found no matching key")
                except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS as _e_hash:
                    if _mw_debug:
                        logger.debug(f"LLM budget: hash-lookup failed: {_e_hash}")
                    else:
                        logger.warning("LLM budget: hash-lookup failed (enable BUDGET_MW_DEBUG for details)")

            # 2) Fallback to manager validation if still unresolved
            if not key_id and api_key:
                try:
                    from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
                    mgr = await get_api_key_manager()
                    info = await mgr.validate_api_key(
                        api_key=api_key,
                        ip_address=resolve_client_ip(request, self._get_settings_cached()),
                    )
                    if info:
                        key_id = info.get('id')
                        _resp = self._set_key_state(request, key_id, info.get('user_id'))
                        if _resp is not None:
                            return _resp
                        if _mw_debug:
                            logger.debug(f"LLM budget: resolved key_id via manager.validate: {key_id}")
                    elif _mw_debug:
                        logger.debug("LLM budget: manager.validate returned no info for api key")
                except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS as _e_mgr:
                    if _mw_debug:
                        logger.debug(f"LLM budget: manager.validate failed: {_e_mgr}")
                    else:
                        logger.warning("LLM budget: manager.validate failed (enable BUDGET_MW_DEBUG for details)")

            # If still no key_id, treat as JWT/no-key and skip enforcement
            if not key_id:
                if _mw_debug:
                    logger.debug("LLM budget: no api_key_id resolved; skipping budget enforcement")
                return await call_next(request)

        try:
            limits = await get_key_limits(int(key_id))
        except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS:
            logger.exception("LLM budget: failed to read key limits; rejecting request")
            return JSONResponse(
                {
                    "error": "budget_limits_unavailable",
                    "message": "Failed to load virtual key limits; request rejected",
                    "details": {"reason": "internal_error"},
                },
                status_code=503,
            )

        if limits is None or "is_virtual" not in limits:
            if _mw_debug:
                logger.debug(f"LLM budget: key limits missing for key_id={key_id}; rejecting")
            return JSONResponse(
                {
                    "error": "budget_limits_unavailable",
                    "message": "Failed to load virtual key limits; request rejected",
                },
                status_code=503,
            )

        if not limits.get('is_virtual'):
            if _mw_debug:
                logger.debug(f"LLM budget: key {key_id} not virtual; skipping")
            return await call_next(request)

        # Endpoint allowlist enforcement
        try:
            allowed_raw = limits.get('llm_allowed_endpoints')
            if allowed_raw:
                allowed = json.loads(allowed_raw) if isinstance(allowed_raw, str) else allowed_raw
                code = self._endpoint_code(path)
                if _mw_debug:
                    logger.debug(f"LLM budget: endpoint allowlist={allowed} code={code}")
                if isinstance(allowed, list) and code not in allowed:
                    return JSONResponse({
                        "error": "forbidden",
                        "message": f"Endpoint '{code}' not allowed for this key"
                    }, status_code=403)
        except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"LLM budget: allowlist check skipped/failed: {e}")

        # Optional provider/model allowlist enforcement
        try:
            allowed_models_raw = limits.get('llm_allowed_models')
            allowed_providers_raw = limits.get('llm_allowed_providers')
            allowed_models = None
            allowed_providers = None
            if allowed_models_raw:
                allowed_models = json.loads(allowed_models_raw) if isinstance(allowed_models_raw, str) else allowed_models_raw
            if allowed_providers_raw:
                allowed_providers = json.loads(allowed_providers_raw) if isinstance(allowed_providers_raw, str) else allowed_providers_raw

            # Provider by explicit header, if present
            if allowed_providers:
                provider = request.headers.get('X-LLM-Provider')
                if provider and provider not in allowed_providers:
                    return JSONResponse({
                        "error": "forbidden",
                        "message": f"Provider '{provider}' not allowed for this key"
                    }, status_code=403)

            # Model from JSON body when parseable and POST
            if allowed_models and request.method in {"POST", "PUT", "PATCH"}:
                ctype = request.headers.get('content-type', '')
                if 'application/json' in ctype:
                    try:
                        body_bytes = await request.body()
                        if body_bytes:
                            data = json.loads(body_bytes.decode('utf-8'))
                            model = data.get('model')
                            # A dotted Anthropic name (claude-sonnet-4.5) and its dashed ID are the same model.
                            if model and not allowlist_has_model(allowed_models, model):
                                return JSONResponse({
                                    "error": "forbidden",
                                    "message": f"Model '{model}' not allowed for this key"
                                }, status_code=403)
                    except _LLM_BUDGET_PARSE_EXCEPTIONS as _e:
                        # If body cannot be parsed, skip enforcement rather than break requests
                        logger.debug("LLM budget: model allowlist parse skipped/failed: {}", _e)
        except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("LLM budget: provider/model allowlist skipped/failed: {}", e)

        # Budget enforcement
        try:
            principal: AuthPrincipal | None = None
            existing_ctx = getattr(request.state, "auth", None)
            if isinstance(existing_ctx, AuthContext):
                principal = existing_ctx.principal
            if principal is None:
                user_id_value = getattr(request.state, "user_id", None)
                try:
                    user_id_int = int(user_id_value) if user_id_value is not None else None
                except (TypeError, ValueError):
                    user_id_int = None
                org_ids: list[int] = []
                team_ids: list[int] = []
                try:
                    raw_org_ids = getattr(request.state, "org_ids", None)
                    if isinstance(raw_org_ids, (list, tuple)):
                        org_ids = [int(o) for o in raw_org_ids if o is not None]
                except (TypeError, ValueError):
                    org_ids = []
                try:
                    raw_team_ids = getattr(request.state, "team_ids", None)
                    if isinstance(raw_team_ids, (list, tuple)):
                        team_ids = [int(t) for t in raw_team_ids if t is not None]
                except (TypeError, ValueError):
                    team_ids = []
                principal = AuthPrincipal(
                    kind="api_key",
                    user_id=user_id_int,
                    api_key_id=int(key_id),
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
            result = await auth_gov.check_llm_budget_for_api_key(principal, int(key_id))
            limits = result.get("limits") or {}
            if _mw_debug:
                subset = {
                    'llm_budget_day_tokens': limits.get('llm_budget_day_tokens'),
                    'llm_budget_day_usd': limits.get('llm_budget_day_usd'),
                    'llm_budget_month_tokens': limits.get('llm_budget_month_tokens'),
                    'llm_budget_month_usd': limits.get('llm_budget_month_usd'),
                }
                logger.debug(
                    f"LLM budget: over_budget={result.get('over')} reasons={result.get('reasons')} "
                    f"day={result.get('day')} month={result.get('month')} limits={subset}"
                )
            if result.get('over'):
                return JSONResponse({
                    "error": "budget_exceeded",
                    "message": "Virtual key budget exceeded",
                    "details": result,
                }, status_code=402)
        except _LLM_BUDGET_NONCRITICAL_EXCEPTIONS:
            # Fail-closed: if we cannot evaluate the budget, do NOT allow the request.
            # Log with full exception details for operational visibility.
            logger.exception("LLM budget: budget check failed; rejecting request")
            return JSONResponse(
                {
                    "error": "budget_check_failed",
                    "message": "Failed to evaluate budget enforcement; request rejected",
                    "details": {"reason": "internal_error"},
                },
                status_code=503,
            )

        return await call_next(request)
