"""Accepted-turn input and metadata-only status responses."""

from typing import Literal

from pydantic import Field, field_validator

from tldw_Server_API.app.api.v1.schemas.buddies import BuddyModel, ResourceId


class BuddyTurnCreate(BuddyModel):
    """Accept one plain-text turn for an exact authorized conversation.

    Text is limited to 12,000 characters and excludes blank input and slash
    commands. The request key is bounded and idempotent; attachment version must
    be positive. Optional provider/model overrides have bounded nonempty values."""

    conversation_id: ResourceId
    text: str = Field(min_length=1, max_length=12000)
    client_request_id: ResourceId
    expected_attachment_version: int = Field(ge=1)
    model: str | None = Field(None, min_length=1, max_length=512)
    provider: str | None = Field(None, min_length=1, max_length=128)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Validate plain chat text while preserving the submitted content.

        Args:
            value: Length-validated message text.

        Returns:
            The original text, including intentional whitespace.

        Raises:
            ValueError: The message is blank or requests a slash command.
        """
        if not value.strip():
            raise ValueError("Enter a message")
        if value.lstrip().startswith("/"):
            raise ValueError("Use Chat to run slash commands")
        return value


class BuddyTurn(BuddyModel):
    """Return metadata-only lifecycle status for one principal-owned accepted turn.

    No message body or credentials are returned. A completed result references
    its exact persisted message; stopped/failed turns may carry a safe error code."""

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
    """Return a bounded turn ledger page with the applied limit and offset."""

    turns: list[BuddyTurn]
    limit: int
    offset: int
