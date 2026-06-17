"""Typed schemas for the scheduled-tasks control plane."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ScheduledTaskPrimitive = Literal["reminder_task", "watchlist_job", "automation_definition"]
ScheduledTaskEditMode = Literal["native", "external"]


class ScheduledTask(BaseModel):
    """Normalized scheduled-task row for the unified management page."""

    id: str
    primitive: ScheduledTaskPrimitive
    title: str
    description: str | None = None
    status: str
    enabled: bool
    schedule_summary: str | None = None
    timezone: str | None = None
    next_run_at: str | None = None
    last_run_at: str | None = None
    edit_mode: ScheduledTaskEditMode
    manage_url: str | None = None
    source_ref: dict[str, Any] = Field(default_factory=dict)


class ScheduledTaskListResponse(BaseModel):
    """List response for the scheduled-tasks control plane."""

    items: list[ScheduledTask] = Field(default_factory=list)
    total: int = 0
    partial: bool = False
    errors: list[str] = Field(default_factory=list)


class ScheduledTaskDeleteResponse(BaseModel):
    """Delete outcome for native reminder mutations."""

    deleted: bool
