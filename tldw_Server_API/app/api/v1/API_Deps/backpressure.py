from __future__ import annotations

import os
import time

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, Response
from loguru import logger
from redis.exceptions import RedisError

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_profile_mode
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.Infrastructure.redis_factory import (
    create_async_redis_client,
    ensure_async_client_closed,
)

_BACKPRESSURE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RedisError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def _get_redis_client() -> aioredis.Redis:
    return await create_async_redis_client(context="ingest_backpressure")


def _cfg_int(name: str, default_val: int) -> int:
    try:
        v = settings.get(name, None)
        if isinstance(v, (int, float)):
            return int(v)
    except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        env = os.getenv(name)
        if env is not None and str(env).strip() != "":
            return int(env)
    except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
        pass
    return int(default_val)


def _cfg_float(name: str, default_val: float) -> float:
    try:
        v = settings.get(name, None)
        if isinstance(v, (int, float)):
            return float(v)
    except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        env = os.getenv(name)
        if env is not None and str(env).strip() != "":
            return float(env)
    except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
        pass
    return float(default_val)


def _bp_limits() -> tuple[int, float]:
    """Return the current backpressure limits using latest config/env overrides."""
    max_depth = _cfg_int("EMB_BACKPRESSURE_MAX_DEPTH", 25000)
    max_age = _cfg_float("EMB_BACKPRESSURE_MAX_AGE_SECONDS", 300.0)
    return max_depth, max_age


async def _orchestrator_depth_and_age(client: aioredis.Redis) -> tuple[int, float]:
    queues = ["embeddings:chunking", "embeddings:embedding", "embeddings:storage", "embeddings:content"]
    depths = []
    ages = []
    now = time.time()
    for q in queues:
        try:
            d = await client.xlen(q)
        except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
            d = 0
        depths.append(int(d or 0))
        try:
            items = await client.xrange(q, "-", "+", count=1)
            if items:
                first_id = items[0][0]
                ts_ms = float(first_id.split("-", 1)[0])
                ages.append(max(0.0, now - (ts_ms / 1000.0)))
            else:
                ages.append(0.0)
        except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
            ages.append(0.0)
    return (max(depths) if depths else 0, max(ages) if ages else 0.0)


def _principal_has_admin_claims(principal: AuthPrincipal) -> bool:
    try:
        roles = {
            str(role).strip().lower()
            for role in (principal.roles or [])
            if str(role).strip()
        }
        permissions = {
            str(perm).strip().lower()
            for perm in (principal.permissions or [])
            if str(perm).strip()
        }
        if "admin" in roles:
            return True
        if "*" in permissions:
            return True
        if "system.configure" in permissions:
            return True
    except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
        return False
    return False


def _should_enforce_ingest_tenant_rps(request: Request, current_user: User) -> bool:
    """
    Decide whether to enforce per-tenant RPS quotas for ingestion.

    Behaviour:
    - In single-user-style profiles (local/desktop), do not enforce quotas
      for admin principals to keep local/dev runs free of tenant-style 429s.
    - Otherwise, enforce quotas when a positive RPS limit is configured.
    """
    _ = current_user
    try:
        ctx = getattr(request.state, "auth", None)
        principal: AuthPrincipal | None = ctx.principal if isinstance(ctx, AuthContext) else None
    except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Failed to extract principal from request.state.auth: {}", type(exc).__name__)
        principal = None

    is_admin = _principal_has_admin_claims(principal) if principal is not None else False

    try:
        single_profile = is_single_user_profile_mode()
    except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Failed to determine single-user profile mode: {}", type(exc).__name__)
        single_profile = False

    # Local single-user-style profiles: never enforce tenant quotas for admin principals.
    if is_admin and single_profile:
        return False

    # If we lack principal context entirely, treat local single-user-style profiles
    # as non-tenant for ingestion quotas.
    return not (principal is None and single_profile)


async def guard_backpressure_and_quota(
    request: Request,
    response: Response,
    current_user: User = Depends(get_request_user),
):
    # Backpressure by orchestrator depth/age
    client: aioredis.Redis | None = None
    try:
        try:
            client = await _get_redis_client()
        except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
            client = None
        if client is not None:
            max_depth, max_age = _bp_limits()
            depth, age = await _orchestrator_depth_and_age(client)
            if depth >= max_depth or age >= max_age:
                retry_after = 5
                if age >= max_age:
                    retry_after = min(60, int(max(5, age / 2)))
                raise HTTPException(status_code=429, detail="Backpressure: queue overload", headers={"Retry-After": str(retry_after)})
    finally:
        try:
            if client is not None:
                await ensure_async_client_closed(client)
        except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
            pass

    # Tenant quota (allow override key for ingestion; fallback to embeddings quota)
    rps = _cfg_int("INGEST_TENANT_RPS", 0) or _cfg_int("EMBEDDINGS_TENANT_RPS", 0)
    if rps > 0 and _should_enforce_ingest_tenant_rps(request, current_user):
        client2: aioredis.Redis | None = None
        try:
            client2 = await _get_redis_client()
            ts = int(time.time())
            key = f"ingest:tenant:rps:{getattr(current_user, 'id', 'anon')}:{ts}"
            current = await client2.incr(key)
            await client2.expire(key, 2)
            remaining = max(0, rps - int(current or 0))
            if current > rps:
                raise HTTPException(status_code=429, detail="Tenant quota exceeded", headers={"Retry-After": "1", "X-RateLimit-Limit": str(rps), "X-RateLimit-Remaining": str(0)})
            else:
                try:
                    response.headers["X-RateLimit-Limit"] = str(rps)
                    response.headers["X-RateLimit-Remaining"] = str(remaining)
                except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
                    pass
        finally:
            try:
                if client2 is not None:
                    await ensure_async_client_closed(client2)
            except _BACKPRESSURE_NONCRITICAL_EXCEPTIONS:
                pass

    # No return value; dependency completes
    return None
