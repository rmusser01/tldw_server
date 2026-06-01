"""Shared first-run setup domain models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FirstRunStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    FIRST_CHAT_COMPLETE = "first_chat_complete"
    COMPLETED = "completed"


class FirstRunStepStatus(str, Enum):
    NOT_STARTED = "not_started"
    CURRENT = "current"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class FirstRunChatResult(BaseModel):
    completed: bool = False
    provider: str | None = None
    model: str | None = None
    response_id: str | None = None
    completed_at: datetime | None = None


class FirstRunStateResponse(BaseModel):
    status: FirstRunStatus
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[str] = Field(default_factory=list)
    step_data: dict[str, dict[str, Any]] = Field(default_factory=dict)
    first_chat: FirstRunChatResult = Field(default_factory=FirstRunChatResult)
    acknowledged_steps: list[str] = Field(default_factory=list)
    skip_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
