"""Validated response contracts for the versioned health API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReadinessDatabaseStatus(BaseModel):
    """Database fields retained for compatibility with existing typed clients."""

    model_config = ConfigDict(strict=True)

    ok: bool
    backend: str | None


class ReadinessResponse(BaseModel):
    """Sanitized operator readiness plus the legacy client compatibility fields.

    Collector-owned diagnostic maps may grow new metrics. Optional sections are
    omitted when the collector does not provide them, including during shutdown.
    """

    model_config = ConfigDict(strict=True, extra="allow")

    status: Literal["ready", "not_ready"]
    ready: bool
    engine: dict[str, Any]
    db: ReadinessDatabaseStatus
    time: str = Field(description="UTC ISO 8601 time at which the response was assembled.")
    reason: str | None = None
    database: dict[str, Any] | None = None
    workflows_db: dict[str, Any] | None = None
    providers_initialized: bool | None = None
    provider_health: dict[str, Any] | None = None
    otel_available: bool | None = None
    rg_policy: dict[str, Any] | None = None
