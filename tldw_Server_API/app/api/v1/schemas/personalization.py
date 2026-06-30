"""
Pydantic schemas for Personalization API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta, PagePaginationMeta


class OptInRequest(BaseModel):
    enabled: bool = Field(..., description="Enable/disable personalization for the user")


class PersonalizationProfile(BaseModel):
    enabled: bool = True
    alpha: float = 0.2
    beta: float = 0.6
    gamma: float = 0.2
    recency_half_life_days: int = 14
    topic_count: int = 0
    memory_count: int = 0
    session_count: int = 0
    proactive_enabled: bool = True
    proactive_frequency: str = "normal"
    response_style: str = "balanced"
    preferred_format: str = "auto"
    companion_reflections_enabled: bool = True
    companion_daily_reflections_enabled: bool = True
    companion_weekly_reflections_enabled: bool = True
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PreferencesUpdate(BaseModel):
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    recency_half_life_days: int | None = None
    proactive_enabled: bool | None = None
    proactive_frequency: str | None = None
    proactive_types: list[str] | None = None
    quiet_hours: dict[str, str] | None = None
    response_style: str | None = None
    preferred_format: str | None = None
    companion_reflections_enabled: bool | None = None
    companion_daily_reflections_enabled: bool | None = None
    companion_weekly_reflections_enabled: bool | None = None

    @field_validator("alpha", "beta", "gamma", mode="before")
    @classmethod
    def _clamp_weight(cls, v: float | None) -> float | None:
        if v is None:
            return v
        return max(0.0, min(1.0, float(v)))

    @field_validator("recency_half_life_days", mode="before")
    @classmethod
    def _clamp_half_life(cls, v: int | None) -> int | None:
        if v is None:
            return v
        return max(1, min(365, int(v)))


class MemoryItem(BaseModel):
    id: str
    type: Literal["semantic", "episodic"] = "semantic"
    content: str
    pinned: bool = False
    hidden: bool = False
    tags: list[str] | None = None
    timestamp: datetime | None = None


class MemoryCreate(BaseModel):
    """Schema for POST /memories - no id field (server-generated)."""
    content: str
    type: Literal["semantic", "episodic"] = "semantic"
    pinned: bool = False
    tags: list[str] | None = None


class MemoryUpdate(BaseModel):
    """Schema for PATCH /memories/{id}."""
    content: str | None = None
    pinned: bool | None = None
    hidden: bool | None = None
    tags: list[str] | None = None


class MemoryValidateRequest(BaseModel):
    """Schema for POST /memories/validate."""
    memory_ids: list[str]


class MemoryImportRequest(BaseModel):
    """Schema for POST /memories/import."""
    memories: list[dict[str, Any]]


class MemoryListResponse(BaseModel):
    items: list[MemoryItem]
    total: int
    page: int = 1
    size: int = 50
    pagination: PagePaginationMeta


class PurgeResponse(BaseModel):
    """Structured response from purge endpoint."""
    status: str
    deleted_counts: dict[str, int]
    enabled: bool
    purged_at: datetime


class ExplanationSignal(BaseModel):
    name: str
    value: float
    detail: str | None = None


class ExplanationEntry(BaseModel):
    timestamp: datetime
    context: Literal["rag", "chat"]
    signals: list[ExplanationSignal]


class ExplanationListResponse(BaseModel):
    items: list[ExplanationEntry]
    total: int
    pagination: OffsetPaginationMeta


class DetailResponse(BaseModel):
    detail: str
    model_config = ConfigDict(from_attributes=True)
