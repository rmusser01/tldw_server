"""
principal_model.py

Core models for representing the authenticated principal and request-level
auth context within the AuthNZ system.

These models are intentionally small and dependency-light so they can be
referenced from multiple AuthNZ components without creating import cycles.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

PrincipalKind = Literal["user", "api_key", "service", "anonymous"]


def compute_principal_id(kind: str, subject_key: str) -> str:
    """
    Compute a stable, pseudonymous identifier for a principal.

    The returned value is suitable for logging and metrics and is designed
    to avoid exposing raw user or API key identifiers directly.
    """
    base = f"{kind}:{subject_key}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    # Shorten for readability while remaining collision-resistant for our scale.
    return f"{kind}:{digest[:16]}"


class AuthPrincipal(BaseModel):
    """
    Represents the authenticated caller and their claims.

    This model is the canonical, in-memory representation of "who is calling"
    and what they are allowed to do. It is intended to be created once per
    request and reused throughout the stack.
    """

    kind: PrincipalKind = Field(
        ...,
        description="High-level principal kind (user, api_key, service, anonymous).",
    )
    # Identity fields (only some will be populated depending on kind)
    user_id: int | None = Field(
        default=None,
        description="Numeric user identifier when the principal is a user or API-key-backed user.",
    )
    api_key_id: int | None = Field(
        default=None,
        description="Numeric API key identifier when the principal is authenticated via API key.",
    )
    username: str | None = Field(
        default=None,
        description="Username for user principals when available.",
    )
    email: str | None = Field(
        default=None,
        description="Email for user principals when available.",
    )
    subject: str | None = Field(
        default=None,
        description=(
            "Optional subject descriptor (e.g. 'user:123', 'service:workflow-engine'). "
            "When present, this is used as the primary key input for principal_id computation."
        ),
    )
    token_type: str | None = Field(
        default=None,
        description="Logical token type when authenticated via JWT or similar (access, refresh, service, virtual, etc.).",
    )
    jti: str | None = Field(
        default=None,
        description="JWT ID or similar opaque token identifier, when available.",
    )
    impersonation: bool = Field(
        default=False,
        description="True when this principal uses an admin impersonation token.",
    )
    impersonated_by: int | None = Field(
        default=None,
        description="Admin user id that issued the impersonation token, when applicable.",
    )

    # Claims
    roles: list[str] = Field(
        default_factory=list,
        description="Role names associated with this principal.",
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="Effective permission strings associated with this principal.",
    )
    is_admin: bool = Field(
        default=False,
        description="Convenience flag for admin-like principals.",
    )
    org_ids: list[int] = Field(
        default_factory=list,
        description="Organization identifiers this principal is a member of.",
    )
    team_ids: list[int] = Field(
        default_factory=list,
        description="Team identifiers this principal is a member of.",
    )
    active_org_id: int | None = Field(
        default=None,
        description="Active organization identifier for the current token scope.",
    )
    active_team_id: int | None = Field(
        default=None,
        description="Active team identifier for the current token scope.",
    )

    model_config = ConfigDict(frozen=False)

    @model_validator(mode="before")
    @classmethod
    def _normalize_kind(cls, data: Any) -> Any:
        """
        Normalize legacy kinds to the documented set.

        Older code paths may emit kind="single_user"; normalize this to
        the canonical "user" kind so downstream comparisons remain stable.
        """
        if isinstance(data, dict):
            legacy_kind = data.get("kind")
            if legacy_kind == "single_user":
                data = dict(data)
                data["kind"] = "user"
        return data

    @computed_field  # type: ignore[misc]
    @property
    def principal_id(self) -> str:
        """
        Stable, pseudonymous identifier for this principal.

        This identifier is derived from the principal kind and a subject key
        (subject/user_id/api_key_id) and is suitable for logging, metrics,
        and guardrail lookups without leaking raw identifiers.
        """
        if self.subject:
            subject_key = self.subject
        elif self.user_id is not None:
            subject_key = f"user:{self.user_id}"
        elif self.api_key_id is not None:
            subject_key = f"api_key:{self.api_key_id}"
        else:
            subject_key = "anonymous"
        return compute_principal_id(self.kind, subject_key)


class AuthContext(BaseModel):
    """
    Request-scoped authentication context.

    Wraps an AuthPrincipal together with transient request metadata that is
    often useful for guardrails, auditing, and observability.
    """

    principal: AuthPrincipal = Field(
        ...,
        description="Canonical principal for the current request.",
    )
    ip: str | None = Field(
        default=None,
        description="Best-effort client IP address for the request.",
    )
    user_agent: str | None = Field(
        default=None,
        description="User-Agent string from the request, when available.",
    )
    request_id: str | None = Field(
        default=None,
        description="Correlated request identifier used for tracing.",
    )

    model_config = ConfigDict(frozen=False)


def is_single_user_principal(principal: AuthPrincipal | None) -> bool:
    """
    Determine whether a principal represents the bootstrapped single-user profile.

    Detection prefers an explicit subject marker but also tolerates legacy contexts
    that relied on AUTH_MODE, token_type, and fixed user id.
    """
    if not isinstance(principal, AuthPrincipal):
        return False

    if getattr(principal, "subject", None) == "single_user":
        return True

    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    except Exception:
        return False

    try:
        settings = get_settings()
        if getattr(settings, "AUTH_MODE", None) != "single_user":
            return False
        fixed_id = getattr(settings, "SINGLE_USER_FIXED_ID", None)
        if fixed_id is None:
            return False
        return principal.user_id == int(fixed_id)
    except Exception:
        return False
