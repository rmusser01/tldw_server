"""Authorization helpers for managed vLLM request routing."""

from __future__ import annotations

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, is_single_user_principal


def can_select_managed_vllm_instance(principal: AuthPrincipal | None) -> bool:
    """Return whether the caller may explicitly select a managed vLLM instance."""
    if principal is None:
        return False
    if is_single_user_principal(principal):
        return True
    if bool(getattr(principal, "is_admin", False)):
        return True
    roles = {str(role).strip().lower() for role in (principal.roles or []) if str(role).strip()}
    return "admin" in roles
