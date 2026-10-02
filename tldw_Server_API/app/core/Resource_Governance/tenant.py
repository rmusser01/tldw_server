from __future__ import annotations

import contextlib
import hmac
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from loguru import logger
from tldw_Server_API.app.core.testing import is_truthy


@dataclass(frozen=True)
class TenantScopeConfig:
    enabled: bool = False
    header: str = "X-TLDW-Tenant"
    # Parsed for config compatibility only: the tenant is the principal's own org (TASK-13402).
    jwt_claim: str = "tenant_id"


def get_tenant_id(
    headers: Mapping[str, str],
    own_tenant: str | None = None,
    config: TenantScopeConfig | None = None,
    member_of: Collection[str] = (),
) -> str | None:
    """
    Pick the caller's tenant from a validated principal (TASK-13402).

    ``own_tenant`` (the principal's own org, in deps.tenant_claims_from_state's order) and
    ``member_of`` (the principal's org ids) must come from a validated principal. The
    header only selects one of ``member_of``; a header naming any other tenant is ignored,
    so an unvalidated header never names a bucket. Otherwise the tenant is ``own_tenant``.
    """
    cfg = config or TenantScopeConfig()
    if not cfg.enabled:
        return None

    val = str(headers.get(cfg.header) or headers.get(cfg.header.lower()) or "").strip()
    if val and val in member_of:
        return val
    return (str(own_tenant).strip() or None) if own_tenant is not None else None


_LOG_HASH_SECRET_WARNED = False
_HASH_SECRET_FALLBACK_ENV_KEYS = (
    "API_KEY_PEPPER",
    "JWT_SECRET_KEY",
    "SINGLE_USER_API_KEY",
    "API_KEY",
)


def _resolve_hash_secret(secret: str | None, env_secret: str | None) -> str | None:
    if secret:
        return secret
    if env_secret:
        return env_secret
    for key in _HASH_SECRET_FALLBACK_ENV_KEYS:
        candidate = os.getenv(key)
        if candidate:
            return candidate
    return None


def hash_entity(value: str, secret: str | None = None) -> str:
    """
    Produce a stable, non-reversible identifier for logging/metrics.

    HMAC-SHA256 with a server-side secret (env: TLDW_LOG_HASH_SECRET).
    When no secret is supplied, reads from env; if still missing, uses a
    process-unique fallback (less ideal for multi-process correlation).
    """
    global _LOG_HASH_SECRET_WARNED
    env_secret = os.getenv("TLDW_LOG_HASH_SECRET")
    enforce = is_truthy(os.getenv("TLDW_ENFORCE_LOG_HASH_SECRET"))
    if enforce and not env_secret and not secret:
        # In enforced mode, require the dedicated log-hash secret explicitly.
        raise RuntimeError("TLDW_LOG_HASH_SECRET is required but not set (TLDW_ENFORCE_LOG_HASH_SECRET=1)")
    resolved_secret = _resolve_hash_secret(secret=secret, env_secret=env_secret)
    if not resolved_secret:
        if not _LOG_HASH_SECRET_WARNED:
            with contextlib.suppress(Exception):
                logger.warning("hash_entity using process-local fallback; set TLDW_LOG_HASH_SECRET for stable hashing across processes")
            _LOG_HASH_SECRET_WARNED = True
    key = (resolved_secret or os.getpid().__repr__()).encode()
    return hmac.new(key, value.encode(), sha256).hexdigest()


def parse_tenant_config(data: Mapping[str, Any]) -> TenantScopeConfig:
    """
    Build a TenantScopeConfig from a dictionary, e.g., policy snapshot's `tenant` section.
    Unknown keys are ignored.
    """
    enabled = bool(data.get("enabled", False))
    header = str(data.get("header", TenantScopeConfig.header))
    claim = str(data.get("jwt_claim", TenantScopeConfig.jwt_claim))
    return TenantScopeConfig(enabled=enabled, header=header, jwt_claim=claim)
