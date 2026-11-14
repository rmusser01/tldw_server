"""Core-owned chat loop event models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChatLoopEventType = Literal[
    "run_started",
    "llm_chunk",
    "llm_complete",
    "tool_proposed",
    "approval_required",
    "approval_resolved",
    "tool_started",
    "tool_finished",
    "tool_failed",
    "assistant_message_committed",
    "run_complete",
    "run_error",
    "run_cancelled",
]


class ChatLoopEvent(BaseModel):
    """Canonical event envelope emitted by chat loop runs."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    seq: int = Field(..., ge=1)
    ts: datetime = Field(default_factory=datetime.utcnow)
    event: ChatLoopEventType
    data: dict[str, Any] = Field(default_factory=dict)
