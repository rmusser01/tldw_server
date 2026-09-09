"""Accepted-turn input and metadata-only status responses."""

from typing import Literal

from pydantic import Field, field_validator

from tldw_Server_API.app.api.v1.schemas.buddies import BuddyModel, ResourceId


class BuddyTurnCreate(BuddyModel):
    conversation_id: ResourceId
    text: str = Field(min_length=1, max_length=12000)
    client_request_id: ResourceId
    expected_attachment_version: int = Field(ge=1)
    model: str | None = Field(None, min_length=1, max_length=512)
    provider: str | None = Field(None, min_length=1, max_length=128)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a message")
        if value.lstrip().startswith("/"):
            raise ValueError("Use Chat to run slash commands")
        return value


class BuddyTurn(BuddyModel):
    id: str
    client_slot: str
    client_request_id: str
    conversation_id: str
    conversation_title: str
    workspace_id: str | None
    attachment_version: int
    status: Literal["queued", "running", "completed", "failed", "stopped"]
    result_message_id: str | None
    error_code: str | None
    created_at: str
    updated_at: str


class BuddyTurnList(BuddyModel):
    turns: list[BuddyTurn]
    limit: int
    offset: int
