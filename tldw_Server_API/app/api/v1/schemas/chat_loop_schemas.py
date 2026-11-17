"""Schemas for the chat loop protocol and approval actions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Chat.chat_loop_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    ChatLoopEvent,
    ChatLoopEventType,
)

ChatLoopApprovalDecision = Literal["approve", "reject"]


class ChatLoopStartRequest(BaseModel):
    """Request payload for starting a chat loop run."""

    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("messages must not be empty")
        return value


class ChatLoopStartResponse(BaseModel):
    """Response payload returned when a run is accepted."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)


class ChatLoopApprovalDecisionRequest(BaseModel):
    """Request payload for approving or rejecting one tool approval prompt."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(..., min_length=1)
    decision: ChatLoopApprovalDecision


class ChatLoopActionResponse(BaseModel):
    """Simple acknowledgement payload for mutating loop actions."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True


class ChatLoopEventsResponse(BaseModel):
    """Replay payload containing ordered events after a sequence cursor."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    events: list[ChatLoopEvent] = Field(default_factory=list)
