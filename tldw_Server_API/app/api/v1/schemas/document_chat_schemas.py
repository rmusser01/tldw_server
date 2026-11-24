"""Request and response contracts for Document Workspace chat."""

from datetime import datetime
from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter


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


class DocumentChatCitation(BaseModel):
    """A saved passage reference contains no external URL or access grant."""

    model_config = ConfigDict(extra="forbid")
    media_uuid: UUID
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
    excerpt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
    page: int | None = Field(default=None, ge=1)
    epub_cfi: str | None = Field(default=None, max_length=1024, pattern=r"^epubcfi\(.+\)$")


DocumentChatCitations = Annotated[list[DocumentChatCitation], Field(min_length=1, max_length=64)]
_CITATIONS = TypeAdapter(DocumentChatCitations)


def validate_document_chat_citations(citations: DocumentChatCitations, media_uuid: UUID) -> list[dict[str, Any]]:
    """Scope each citation to the verified document and cap encoded storage."""
    if any(citation.media_uuid != media_uuid for citation in citations):
        raise ValueError("Document chat sources must refer to the current document")
    if len(quote(_CITATIONS.dump_json(citations, exclude_none=True), safe="~()*!.'-")) > 262_144:
        raise ValueError("Document chat sources exceed the encoded size limit")
    return [citation.model_dump(mode="json", exclude_none=True) for citation in citations]
