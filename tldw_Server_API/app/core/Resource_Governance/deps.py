from __future__ import annotations

"""
Helpers and FastAPI dependencies for deriving Resource Governor entity keys.

Preference order:
  1) Auth scopes (request.state.user_id → user:{id})
  2) API key scope (request.state.api_key_id → api_key:{id})
  3) IP scope (trusted header via RG_CLIENT_IP_HEADER else request.client.host)

Unvalidated credentials (a raw X-API-KEY or Authorization header that auth has
not yet populated request.state for) are never hashed into their own bucket
here: that would let a rotating fake key mint unlimited fresh buckets. Ingress
identity for those cases is resolved by RGSimpleMiddleware._principal_entity,
which validates the credential (or falls back to the IP) before this function
ever runs.
"""

import os

from fastapi import Request
from loguru import logger

from tldw_Server_API.app.core.Security.trusted_proxy import resolve_trusted_client_ip

from .tenant import TenantScopeConfig, get_tenant_id, parse_tenant_config

_RG_DEPS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def derive_client_ip(request: Request) -> str:
    """Derive client IP with trusted proxy handling.

    - Trust header specified by RG_CLIENT_IP_HEADER (e.g., X-Forwarded-For) only when
      the immediate peer (request.client.host) is within RG_TRUSTED_PROXIES (CIDR/IP list).
    - Otherwise, fall back to request.client.host.
    """
    try:
        peer = request.client.host if request.client and request.client.host else None
    except _RG_DEPS_NONCRITICAL_EXCEPTIONS:
        peer = None
    trusted = tuple(
        part.strip()
        for part in (os.getenv("RG_TRUSTED_PROXIES") or "").split(",")
        if part.strip()
    )
    header_name = (os.getenv("RG_CLIENT_IP_HEADER") or "").strip()
    xff_values: tuple[str, ...] = ()
    single_value = None
    if header_name:
        if header_name.lower() == "x-forwarded-for":
            xff_values = tuple(request.headers.getlist(header_name))
        else:
            header_values = tuple(request.headers.getlist(header_name))
            single_value = header_values[0] if len(header_values) == 1 else None
    resolved = resolve_trusted_client_ip(
        peer,
        trusted,
        forwarded_for_values=xff_values,
        single_forwarded_value=single_value,
    )
    return resolved or "unknown"


def tenant_claims_from_state(request: Request) -> dict[str, object]:
    """Extract tenant-related claims from trusted request state/auth context.

    ``tenant_id`` is the caller's own tenant: the tenant claim, else the active org,
    else the org (for an API key, its scoped org or first org). RG ingress calls this
    too, so ingress and endpoint reservations agree on a caller's tenant (TASK-13402).
    """
    claims: dict[str, object] = {}
    for attr in ("tenant_id", "active_org_id", "org_id"):
        try:
            value = getattr(request.state, attr, None)
        except _RG_DEPS_NONCRITICAL_EXCEPTIONS:
            value = None
        if value is not None:
            claims[attr] = value
    try:
        auth = getattr(request.state, "auth", None)
        principal = getattr(auth, "principal", None)
        for attr in ("tenant_id", "active_org_id", "org_id"):
            value = getattr(principal, attr, None)
            if value is not None:
                claims.setdefault(attr, value)
    except _RG_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("RG tenant claims: auth principal lookup failed; continuing with request.state claims: {}", exc)
    for attr in ("tenant_id", "active_org_id", "org_id"):
        if attr in claims:
            claims["tenant_id"] = claims[attr]
            break
    return claims


def _member_tenant_ids(request: Request, claims: dict[str, object]) -> set[str]:
    """Tenants the validated principal on request state belongs to: its org ids and claims."""
    ids = {str(value) for value in claims.values()}
    auth = getattr(request.state, "auth", None)
    for source in (request.state, getattr(auth, "principal", None)):
        org_ids = getattr(source, "org_ids", None)
        if isinstance(org_ids, (list, tuple, set)):
            ids.update(str(org_id) for org_id in org_ids)
    return ids


def _tenant_config_from_request(request: Request) -> TenantScopeConfig | None:
    """Read tenant-scope config from the app's current RG policy snapshot."""
    try:
        app_state = getattr(getattr(request, "app", None), "state", None)
        loader = getattr(app_state, "rg_policy_loader", None)
        snapshot = loader.get_snapshot() if loader is not None else None
        tenant = getattr(snapshot, "tenant", None) or {}
        if isinstance(tenant, dict):
            return parse_tenant_config(tenant)
    except _RG_DEPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("RG tenant config lookup failed; falling back to non-tenant entity derivation: {}", exc)
    return None


def derive_entity_key(request: Request, tenant_config: TenantScopeConfig | None = None) -> str:
    """Derive the Resource Governor entity key for a request."""
    if tenant_config is None:
        tenant_config = _tenant_config_from_request(request)
    if tenant_config and tenant_config.enabled:
        try:
            claims = tenant_claims_from_state(request)
            tenant_id = get_tenant_id(
                request.headers, claims=claims, config=tenant_config, member_of=_member_tenant_ids(request, claims)
            )
            if tenant_id:
                return f"tenant:{tenant_id}"
        except _RG_DEPS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("RG tenant entity derivation failed; falling back to user/api_key/ip scope: {}", exc)

    # Prefer authenticated user scope
    try:
        uid = getattr(request.state, "user_id", None)
        if isinstance(uid, int) or (isinstance(uid, str) and uid):
            return f"user:{uid}"
    except _RG_DEPS_NONCRITICAL_EXCEPTIONS:
        pass

    # Prefer API key id scope when available
    try:
        kid = getattr(request.state, "api_key_id", None)
        if kid is not None:
            return f"api_key:{kid}"
    except _RG_DEPS_NONCRITICAL_EXCEPTIONS:
        pass

    # IP fallback
    ip = derive_client_ip(request)
    return f"ip:{ip}"


async def get_entity_key(request: Request) -> str:
    """FastAPI dependency that returns an entity key for Resource Governor."""
    return derive_entity_key(request)
