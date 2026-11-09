"""Jobs administration authorization without ambient database privilege changes."""

import os
from typing import Any

from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequireRole
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.testing import is_truthy as _is_truthy

_RBAC_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _make_admin_user_from_principal(principal: AuthPrincipal) -> dict[str, Any]:
    """Preserve the real user ID for authorization; labels are informational only."""
    user_id = principal.user_id
    if user_id is not None:
        username = principal.subject or principal.principal_id or f"user:{user_id}"
    else:
        username = principal.subject or principal.principal_id or principal.kind or "service"
    return {"id": user_id, "username": str(username)}


def _enforce_domain_scope(user: dict, domain: str | None) -> None:
    """Retain the existing feature-flagged domain allowlist semantics."""
    try:
        if not _is_truthy(os.getenv("JOBS_DOMAIN_SCOPED_RBAC")):
            return
        try:
            uid_val = getattr(user, "id", None)
        except _RBAC_EXCEPTIONS:
            uid_val = None
        if uid_val is None and isinstance(user, dict):
            uid_val = user.get("id")
        uid = str(uid_val or "")
        if _is_truthy(os.getenv("JOBS_REQUIRE_DOMAIN_FILTER")) and not (domain and domain.strip()):
            raise HTTPException(status_code=403, detail="Domain filter is required for this operation")
        allow = os.getenv(f"JOBS_DOMAIN_ALLOWLIST_{uid}", "").strip()
        if allow:
            allowed = {d.strip() for d in allow.split(",") if d.strip()}
            if domain and domain.strip():
                if domain not in allowed:
                    raise HTTPException(status_code=403, detail=f"Not allowed for domain {domain}")
            else:
                raise HTTPException(status_code=403, detail="Domain filter required for allowlisted admin")
    except HTTPException:
        raise
    except _RBAC_EXCEPTIONS as exc:
        if _is_truthy(os.getenv("JOBS_RBAC_FORCE")):
            raise HTTPException(status_code=403, detail="RBAC enforcement error") from exc


def _enforce_domain_scope_from_principal(principal: AuthPrincipal, domain: str | None) -> None:
    """Apply the existing opt-in principal-driven domain check."""
    if not _is_truthy(os.getenv("JOBS_DOMAIN_RBAC_PRINCIPAL")):
        return
    _enforce_domain_scope(_make_admin_user_from_principal(principal), domain)


def _enforce_domain_scope_unified(principal: AuthPrincipal, domain: str | None) -> dict[str, Any]:
    """Authorize domain access and return the existing admin-user mapping."""
    admin_user = _make_admin_user_from_principal(principal)
    if _is_truthy(os.getenv("JOBS_DOMAIN_RBAC_PRINCIPAL")):
        _enforce_domain_scope_from_principal(principal, domain)
    else:
        _enforce_domain_scope(admin_user, domain)
    return admin_user


async def authorize_jobs_admin_domain_read(principal: AuthPrincipal, *, domain: str) -> None:
    """Preflight the same admin/domain guards; never set Jobs RLS privileges."""
    await RequireRole("admin")(principal)
    _enforce_domain_scope_unified(principal, domain)
