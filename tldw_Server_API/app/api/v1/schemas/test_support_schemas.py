from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AdminE2EResetResponse(BaseModel):
    """Response for resetting admin e2e fixture state."""

    ok: bool = True


class AdminE2ERunDueBackupSchedulesResponse(BaseModel):
    """Response for deterministic backup schedule trigger helpers."""

    ok: bool = True
    triggered_runs: int = 0


class AdminE2EPrepareAdminWebhooksResponse(BaseModel):
    """Sanitized result for preparing canonical webhooks in an e2e database."""

    ok: bool = True
    phase: Literal["complete"]
    active_primary_key_id: str
    state_revision: int = Field(..., ge=1)


class AdminE2ESeedRequest(BaseModel):
    """Request payload for deterministic admin e2e fixture seeding."""

    scenario: Literal["jwt_admin", "dsr_jwt_admin", "single_user_admin"]

    model_config = ConfigDict(from_attributes=True)


class AdminE2EBootstrapJwtSessionRequest(BaseModel):
    """Request payload for generating seeded JWT browser cookies."""

    principal_key: str = Field(..., min_length=1, max_length=200)

    model_config = ConfigDict(from_attributes=True)


class AdminE2ECookiePayload(BaseModel):
    """Serializable cookie payload for Playwright context bootstrapping."""

    name: str
    value: str
    path: str = "/"
    http_only: bool = False
    same_site: Literal["Lax", "Strict", "None"] = "Lax"

    model_config = ConfigDict(from_attributes=True)


class AdminE2ESeededUser(BaseModel):
    """Stable seeded user reference returned to browser tests."""

    id: int
    username: str
    email: str
    key: str

    model_config = ConfigDict(from_attributes=True)


class AdminE2EAlertFixture(BaseModel):
    """Stable alert identity fixture for monitoring overlay tests."""

    alert_id: str
    alert_identity: str | None = None
    message: str | None = None

    model_config = ConfigDict(from_attributes=True)


class AdminE2EOrganizationFixture(BaseModel):
    """Stable organization fixture metadata."""

    id: int
    name: str
    slug: str

    model_config = ConfigDict(from_attributes=True)


class AdminE2ESeedResponse(BaseModel):
    """Seed response returned to admin-ui real-backend browser helpers."""

    scenario: str
    users: dict[str, AdminE2ESeededUser]
    fixtures: dict[str, list[AdminE2EAlertFixture] | list[AdminE2EOrganizationFixture]]

    model_config = ConfigDict(from_attributes=True)


class AdminE2EBootstrapJwtSessionResponse(BaseModel):
    """JWT bootstrap payload for Playwright cookie injection."""

    principal_key: str
    cookies: list[AdminE2ECookiePayload]

    model_config = ConfigDict(from_attributes=True)
