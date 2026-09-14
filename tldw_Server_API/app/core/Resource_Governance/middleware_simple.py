from __future__ import annotations

"""
Minimal ASGI middleware that derives a policy_id from route tags or path and
calls the Resource Governor before and after handlers.

This is a thin adapter for Stage 1/2 validation and can be replaced by a
full-featured middleware later.
"""

import contextlib
import os
import re
import uuid

from loguru import logger
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .deps import derive_client_ip, derive_entity_key
from .governor import RGRequest
from .tenant import TenantScopeConfig, parse_tenant_config

_RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    re.error,
)


class RGSimpleMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # Compile simple path matchers from stub mapping
        self._compiled_map: list[tuple[re.Pattern[str], str]] = []

    async def _ensure_loader_matches_env(self, request: Request) -> None:
        """Ensure app.state.rg_policy_loader reflects current RG_POLICY_PATH.

        Tests may change RG_POLICY_PATH between runs while reusing the same
        FastAPI app instance. This helper refreshes the loader if the source
        path differs from the current env so that route_map lookups work.
        """
        try:
            env_path = os.getenv("RG_POLICY_PATH")
            if not env_path:
                return
            loader = getattr(request.app.state, "rg_policy_loader", None)
            snap = None
            try:
                snap = loader.get_snapshot() if loader else None
            except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                snap = None
            current_path = str(getattr(snap, "source_path", "")) if snap else None
            if (loader is None) or (snap is None) or (current_path and str(current_path) != str(env_path)):
                from .policy_loader import PolicyLoader, PolicyReloadConfig
                # Respect reload flags from env for consistency
                reload_enabled = (os.getenv("RG_POLICY_RELOAD_ENABLED", "true").lower() in {"1", "true", "yes"})
                interval = int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10")
                new_loader = PolicyLoader(env_path, PolicyReloadConfig(enabled=reload_enabled, interval_sec=interval))
                await new_loader.load_once()
                request.app.state.rg_policy_loader = new_loader
                request.app.state.rg_policy_store = "file"
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            # Best-effort only; never block the request
            pass

    def _init_route_map(self, request: Request) -> None:
        try:
            loader = getattr(request.app.state, "rg_policy_loader", None)
            if not loader:
                return
            snap = loader.get_snapshot()
            route_map = getattr(snap, "route_map", {}) or {}
            by_path = dict(route_map.get("by_path") or {})
            compiled = []
            for pat, pol in by_path.items():
                # Convert glob patterns (supports '*' anywhere, anchored unless trailing '*')
                pat = str(pat)
                if "*" in pat:
                    regex = re.escape(pat).replace("\\*", ".*")
                    if not pat.endswith("*"):
                        regex += "$"
                else:
                    regex = re.escape(pat) + "$"
                compiled.append((re.compile(regex), str(pol)))
            self._compiled_map = compiled
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"RGSimpleMiddleware: route_map init skipped: {e}")

    def _derive_policy_id(self, request: Request) -> str | None:
        # Prefer path-based routing (works even before route resolution)
        try:
            # Use compiled route_map if available
            if not self._compiled_map:
                self._init_route_map(request)
            path = request.url.path or "/"
            for pat, pol in self._compiled_map:
                try:
                    if pat.match(path):
                        return str(pol)
                except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                    continue
            # Fallback to simple string matching from snapshot if compiled map unavailable
            loader = getattr(request.app.state, "rg_policy_loader", None)
            snap = loader.get_snapshot() if loader else None
            route_map = getattr(snap, "route_map", {}) or {}
            by_path = dict(route_map.get("by_path") or {})
            # Simple wildcard matching: '*' anywhere, anchored unless trailing '*'
            for pat, pol in by_path.items():
                pat = str(pat)
                if "*" in pat:
                    regex = re.escape(pat).replace("\\*", ".*")
                    if not pat.endswith("*"):
                        regex += "$"
                    if re.match(regex, path):
                        return str(pol)
                elif path == pat:
                    return str(pol)
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            pass

        # Fallback to tag-based routing (may not be available early in ASGI pipeline)
        try:
            by_tag = {}
            loader = getattr(request.app.state, "rg_policy_loader", None)
            snap = loader.get_snapshot() if loader else None
            route_map = getattr(snap, "route_map", {}) or {}
            by_tag = dict(route_map.get("by_tag") or {})
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            by_tag = {}
        try:
            route = request.scope.get("route")
            tags = list(getattr(route, "tags", []) or [])
            for t in tags:
                if t in by_tag:
                    return str(by_tag[t])
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            pass
        # Heuristic fallback by path segments for common endpoints
        try:
            p = request.url.path or "/"
            if p.startswith("/api/v1/chat/") or p == "/api/v1/chat/completions":
                return "chat.default"
            if p.startswith("/api/v1/audio/"):
                return "audio.default"
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            pass
        return None

    def _derive_tenant_config(self, request: Request) -> TenantScopeConfig | None:
        try:
            loader = getattr(request.app.state, "rg_policy_loader", None)
            snap = loader.get_snapshot() if loader else None
            tenant_data = getattr(snap, "tenant", None) or {}
            if isinstance(tenant_data, dict) and tenant_data:
                return parse_tenant_config(tenant_data)
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            return None
        return None

    def _derive_entity(self, request: Request) -> str:
        """Derive the RG entity key for this request.

        Enforcement details:
        - Prefer auth-derived scopes (user/api_key) as implemented in deps.derive_entity_key.
        - Fall back to IP only when safe: derive_client_ip honors RG_TRUSTED_PROXIES (CIDRs)
          and RG_CLIENT_IP_HEADER, otherwise uses request.client.host.

        The resolved client IP is also attached to request.state.rg_client_ip for
        downstream diagnostics.
        """
        try:
            # Always compute and attach normalized client IP for diagnostics
            request.state.rg_client_ip = derive_client_ip(request)
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            # best-effort only
            pass
        return derive_entity_key(request, tenant_config=self._derive_tenant_config(request))

    def _effective_fail_mode(self, request: Request, policy_id: str | None) -> str:
        try:
            if policy_id:
                loader = getattr(request.app.state, "rg_policy_loader", None)
                pol = loader.get_policy(policy_id) if loader else {}
                req_cfg = dict((pol.get("requests") or {}) if isinstance(pol, dict) else {})
                mode = str(req_cfg.get("fail_mode") or (pol or {}).get("fail_mode") or "").strip().lower()
                if mode in {"fail_closed", "fail_open", "fallback_memory"}:
                    return mode
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if (os.getenv("RG_BACKEND", "memory").strip().lower() or "memory") == "redis":
                mode = str(os.getenv("RG_REDIS_FAIL_MODE") or "fallback_memory").strip().lower()
                if mode in {"fail_closed", "fail_open", "fallback_memory"}:
                    return mode
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            pass
        return "fail_open"

    async def _enforcement_unavailable(self, scope: Scope, receive: Receive, send: Send, *, policy_id: str | None, reason: str) -> None:
        resp = JSONResponse(
            {
                "error": "resource_governance_unavailable",
                "policy_id": policy_id,
                "reason": reason,
            },
            status_code=503,
        )
        await resp(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        # Make sure loader (and its route_map) tracks current env path
        await self._ensure_loader_matches_env(request)
        # Compile route map for fast path matches (best-effort)
        with contextlib.suppress(_RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS):
            self._init_route_map(request)
        policy_id = self._derive_policy_id(request)
        if not policy_id:
            await self.app(scope, receive, send)
            return

        # Cookie sessions must spend their validated owner's user quota when
        # the policy cannot admit an anonymous request. The
        # canonical resolver caches AuthContext on shared ASGI request state,
        # so endpoint auth reuses this validation. Explicit headers (even empty
        # ones) keep precedence over cookies, matching the resolver contract.
        if (
            request.cookies
            and request.headers.get("Authorization") is None
            and request.headers.get("X-API-KEY") is None
        ):
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings

            settings = get_settings()
            loader = getattr(request.app.state, "rg_policy_loader", None)
            policy = (loader.get_policy(policy_id) or {}) if loader else {}
            policy_scopes = set(policy.get("scopes") or ["global", "entity"])
            requires_owner = bool(policy_scopes & {"user", "api_key"}) and not (
                policy_scopes & {"global", "ip", "entity"}
            )
            if (
                requires_owner
                and settings.AUTH_MODE == "single_user"
                and request.cookies.get(settings.SINGLE_USER_SESSION_COOKIE_NAME)
            ):
                from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import (
                    get_auth_principal,
                )

                try:
                    await get_auth_principal(request)
                except HTTPException as exc:
                    # This middleware runs outside ExceptionMiddleware.
                    response = JSONResponse(
                        {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
                    )
                    await response(scope, receive, send)
                    return
        # If governor not initialized, lazily create one using loader + backend env
        gov = getattr(request.app.state, "rg_governor", None)
        if gov is None:
            try:
                loader = getattr(request.app.state, "rg_policy_loader", None)
                if loader is not None:
                    backend = (os.getenv("RG_BACKEND", "memory").strip().lower() or "memory")
                    if backend == "redis":
                        from .governor_redis import RedisResourceGovernor as _RG
                        request.app.state.rg_governor = _RG(policy_loader=loader)
                    else:
                        from .governor import MemoryResourceGovernor as _RG
                        request.app.state.rg_governor = _RG(policy_loader=loader)
                    gov = request.app.state.rg_governor
            except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                gov = None
        if gov is None:
            if self._effective_fail_mode(request, policy_id) == "fail_closed":
                await self._enforcement_unavailable(scope, receive, send, policy_id=policy_id, reason="governor_missing")
                return
            await self.app(scope, receive, send)
            return

        # Attach policy_id to request.state so downstream dependencies can
        # detect RG-governed routes and avoid double-enforcement.
        with contextlib.suppress(_RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS):
            request.state.rg_policy_id = policy_id

        # Build RG request. Always include 'requests'. Specialized categories
        # (tokens/streams/jobs/minutes/etc.) are enforced at endpoint level.
        entity = self._derive_entity(request)
        op_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        cats: dict[str, dict[str, int]] = {"requests": {"units": 1}}
        # Note: tokens/streams/jobs require correct per-request units and are enforced
        # at the endpoint level (reserve/commit) rather than in this minimal middleware.
        rg_req = RGRequest(entity=entity, categories=cats, tags={"policy_id": policy_id, "endpoint": request.url.path})

        try:
            decision, handle_id = await gov.reserve(rg_req, op_id=op_id)
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"RGSimpleMiddleware reserve error: {e}")
            if self._effective_fail_mode(request, policy_id) == "fail_closed":
                await self._enforcement_unavailable(scope, receive, send, policy_id=policy_id, reason="reserve_error")
                return
            await self.app(scope, receive, send)
            return

        if not decision.allowed:
            retry_after = int(decision.retry_after or 1)
            # Map basic rate-limit headers for compatibility
            # Extract per-category details if available
            categories = {}
            try:
                categories = dict((decision.details or {}).get("categories") or {})
            except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                categories = {}
            # Choose a primary category for header mapping: prefer requests, else tokens, else streams/jobs
            primary = None
            if "requests" in categories and not (categories.get("requests") or {}).get("allowed", True):
                primary = "requests"
            elif "tokens" in categories and not (categories.get("tokens") or {}).get("allowed", True):
                primary = "tokens"
            elif "streams" in categories and not (categories.get("streams") or {}).get("allowed", True):
                primary = "streams"
            else:
                # fallback to requests for compatibility
                primary = "requests"

            # Use the primary category to derive base headers
            prim_cat = categories.get(primary) or {}
            limit = int(prim_cat.get("limit") or 0)
            if not limit:
                # Fallback to policy rpm for deny headers when decision omitted limit
                try:
                    loader = getattr(request.app.state, "rg_policy_loader", None)
                    if loader is not None and policy_id:
                        pol = loader.get_policy(policy_id) or {}
                        if primary == "requests":
                            limit = int((pol.get("requests") or {}).get("rpm") or 0)
                        elif primary == "tokens":
                            limit = int((pol.get("tokens") or {}).get("per_min") or 0)
                        elif primary in ("streams", "jobs"):
                            limit = int((pol.get(primary) or {}).get("max_concurrent") or 0)
                except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                    limit = 0

            resp = JSONResponse({
                "error": "rate_limited",
                "policy_id": policy_id,
                "retry_after": retry_after,
            }, status_code=429)
            resp.headers["Retry-After"] = str(retry_after)
            # Generic X-RateLimit-* headers apply only to requests/tokens to
            # avoid misleading headers on concurrency-only denials.
            if primary in {"requests", "tokens"} and limit:
                resp.headers["X-RateLimit-Limit"] = str(limit)
                resp.headers["X-RateLimit-Remaining"] = "0"
                resp.headers["X-RateLimit-Reset"] = str(retry_after)
                # Tokens per-minute headers if tokens is the denying category
                if primary == "tokens":
                    try:
                        loader = getattr(request.app.state, "rg_policy_loader", None)
                        if loader is not None:
                            pol = loader.get_policy(policy_id) or {}
                            per_min = int((pol.get("tokens") or {}).get("per_min") or 0)
                            if per_min > 0:
                                resp.headers["X-RateLimit-PerMinute-Limit"] = str(per_min)
                                resp.headers["X-RateLimit-PerMinute-Remaining"] = "0"
                                resp.headers["X-RateLimit-Tokens-Remaining"] = "0"
                    except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                        pass
            await resp(scope, receive, send)
            return

        # Allowed; run handler with header injection wrapper and then commit in finally
        # Prepare success-path rate-limit headers (using precise peek when available)
        try:
            _cats = dict((decision.details or {}).get("categories") or {})
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
            _cats = {}
        _req_cat = _cats.get("requests") or {}
        _limit = int(_req_cat.get("limit") or 0)
        # Determine categories to peek for precise Remaining/Reset
        _categories_to_peek = list(_cats.keys()) or ["requests"]

        async def _send_wrapped(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                try:
                    # Try to get accurate remaining/reset via governor.peek
                    peek = getattr(gov, "peek_with_policy", None)
                    peek_result = None
                    if callable(peek):
                        try:
                            peek_result = await peek(entity, _categories_to_peek, policy_id)
                        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                            peek_result = None
                    # requests headers (compat)
                    # Fallback to policy rpm if decision did not include limit
                    eff_limit = _limit
                    if not eff_limit:
                        try:
                            loader = getattr(request.app.state, "rg_policy_loader", None)
                            if loader is not None and policy_id:
                                pol = loader.get_policy(policy_id) or {}
                                eff_limit = int((pol.get("requests") or {}).get("rpm") or 0)
                        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                            eff_limit = 0
                    if eff_limit:
                        headers.append((b"x-ratelimit-limit", str(eff_limit).encode()))
                    req_remaining = None
                    req_reset = None
                    if isinstance(peek_result, dict):
                        rinfo = peek_result.get("requests") or {}
                        if rinfo.get("remaining") is not None:
                            req_remaining = int(rinfo.get("remaining"))
                        if rinfo.get("reset") is not None:
                            req_reset = int(rinfo.get("reset"))
                    if req_remaining is None and eff_limit:
                        req_remaining = max(0, eff_limit - 1)
                    if req_reset is None:
                        req_reset = 0
                    if eff_limit:
                        headers.append((b"x-ratelimit-remaining", str(req_remaining).encode()))
                        headers.append((b"x-ratelimit-reset", str(req_reset).encode()))

                    # If additional categories are present (e.g., tokens), set namespaced headers
                    if isinstance(peek_result, dict):
                        # Compute overall reset as max across categories for compatibility
                        try:
                            resets = [int((peek_result.get(c) or {}).get("reset") or 0) for c in _categories_to_peek]
                            overall_reset = max(resets) if resets else req_reset
                            if overall_reset is not None and overall_reset > req_reset and _limit:
                                # override generic reset with stricter value
                                headers = [(k, v) for (k, v) in headers if k != b"x-ratelimit-reset"]
                                headers.append((b"x-ratelimit-reset", str(overall_reset).encode()))
                        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                            pass
                        # Tokens headers are only emitted when the request actually
                        # reserved tokens via middleware (not the default behavior).
                        if "tokens" in _categories_to_peek:
                            tinfo = peek_result.get("tokens") or {}
                            tokens_remaining_val = None
                            try:
                                if tinfo.get("remaining") is not None:
                                    tokens_remaining_val = int(tinfo.get("remaining") or 0)
                            except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                                tokens_remaining_val = None
                            # Expose per-minute headers when policy defines per_min.
                            try:
                                loader = getattr(request.app.state, "rg_policy_loader", None)
                                if loader is not None:
                                    pol = loader.get_policy(policy_id) or {}
                                    per_min = int((pol.get("tokens") or {}).get("per_min") or 0)
                                    if per_min > 0:
                                        headers.append((b"x-ratelimit-perminute-limit", str(per_min).encode()))
                                        if tinfo.get("remaining") is not None:
                                            headers.append(
                                                (
                                                    b"x-ratelimit-perminute-remaining",
                                                    str(int(tinfo.get("remaining") or 0)).encode(),
                                                )
                                            )
                                        else:
                                            headers.append((b"x-ratelimit-perminute-remaining", str(max(0, per_min - 1)).encode()))
                                        if tokens_remaining_val is None:
                                            tokens_remaining_val = max(0, per_min - 1)
                            except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                                pass
                            if tokens_remaining_val is None:
                                tokens_remaining_val = 0
                            headers.append((b"x-ratelimit-tokens-remaining", str(int(tokens_remaining_val)).encode()))
                except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS:
                    pass
                message = {**message, "headers": headers}
            await send(message)

        response = None
        try:
            response = await self.app(scope, receive, _send_wrapped)
        finally:
            try:
                if handle_id:
                    await gov.commit(handle_id, actuals=None)
            except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"RGSimpleMiddleware commit error: {e}")

        return response
