"""Pydantic schemas for cross-session character memory endpoints."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CharacterMemoryCreate(BaseModel):
    """Body for POST /characters/{character_id}/memories."""
    content: str = Field(..., min_length=1, max_length=2000)
    memory_type: Literal["fact", "relationship", "event", "preference", "manual"] = "manual"
    salience: float = Field(0.7, ge=0.0, le=1.0)


class CharacterMemoryUpdate(BaseModel):
    """Body for PATCH /characters/{character_id}/memories/{memory_id}."""
    content: str | None = Field(None, min_length=1, max_length=2000)
    memory_type: Literal["fact", "relationship", "event", "preference", "manual"] | None = None
    salience: float | None = Field(None, ge=0.0, le=1.0)


class CharacterMemoryArchiveRequest(BaseModel):
    """Body for POST .../memories/{memory_id}/archive."""
    archived: bool = True


class CharacterMemoryExtractRequest(BaseModel):
    """Body for POST .../memories/extract — manual extraction trigger."""
    chat_id: str
    message_limit: int = Field(50, ge=1, le=200)
    provider: str | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CharacterMemoryResponse(BaseModel):
    """Single memory entry."""
    id: str
    character_id: str
    memory_type: str
    content: str
    salience: float
    source_conversation_id: str | None = None
    archived: bool
    created_at: str
    last_modified: str


class CharacterMemoryListResponse(BaseModel):
    """Paginated list of memory entries."""
    memories: list[CharacterMemoryResponse]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta


class CharacterMemoryExtractResponse(BaseModel):
    """Result of extraction (manual or automatic)."""
    extracted: int
    skipped_duplicates: int
    memories: list[CharacterMemoryResponse]
