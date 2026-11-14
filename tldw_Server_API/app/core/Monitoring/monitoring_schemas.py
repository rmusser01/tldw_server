"""Core-owned monitoring watchlist models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WatchlistRule(BaseModel):
    rule_id: str | None = Field(None, description="Optional stable rule identifier")
    pattern: str = Field(..., description="Literal or /regex/ pattern")
    category: str | None = Field(None, description="Logical category, e.g., 'adult', 'violence', 'self_harm'")
    severity: str | None = Field("info", description="info | warning | critical")
    note: str | None = Field(None, description="Free-text note for admins")
    tags: list[str] | None = Field(default=None, description="Optional tags")


class Watchlist(BaseModel):
    id: str | None = Field(None, description="Watchlist ID (UUID). If absent on create, a new one is generated")
    name: str
    description: str | None = None
    enabled: bool = True
    scope_type: str = Field("user", description="global | user | team | org")
    scope_id: str | None = Field(None, description="The corresponding subject id (null for global)")
    managed_by: str | None = Field(None, description="config | api")
    rules: list[WatchlistRule] = Field(default_factory=list)
