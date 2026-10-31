"""
chat_commands_schemas.py

Pydantic schemas for the Chat Commands discovery endpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatCommandInfo(BaseModel):
    """Represents a single slash command available to the user."""
    name: str = Field(..., description="Command name without the leading slash")
    description: str = Field(..., description="Brief description of what the command does")
    required_permission: str | None = Field(
        None,
        description="Permission required to invoke this command",
    )
    usage: str | None = Field(
        None,
        description="Usage string shown to users, e.g. `/weather [location]`",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Ordered argument names expected by this command",
    )
    requires_api_key: bool | None = Field(
        None,
        description="Whether invoking this command requires API key-backed access",
    )
    rate_limit: str | None = Field(
        None,
        description="Human-readable rate-limit summary for this command",
    )
    rbac_required: bool | None = Field(
        None,
        description="Whether command invocation is guarded by permissions",
    )


class ChatCommandsListResponse(BaseModel):
    """Container for a list of available chat commands."""
    commands: list[ChatCommandInfo] = Field(..., description="List of available commands")
