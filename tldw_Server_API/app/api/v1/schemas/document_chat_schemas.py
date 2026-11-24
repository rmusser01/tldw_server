"""Request and response contracts for Document Workspace chat."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentChatLinkRequest(BaseModel):
    """Attach the original Chat record to a verified document."""

    model_config = ConfigDict(extra="forbid")
    media_uuid: UUID
    document_chat_record: str = Field(pattern=r"^[a-f0-9]{64}$")


class DocumentChatSummary(BaseModel):
    """List entry for a linked live conversation."""

    id: str
    title: str | None
    last_modified: datetime


class DocumentChatPage(BaseModel):
    """A page of the current account's saved conversations for a document."""

    items: list[DocumentChatSummary]
    has_more: bool
