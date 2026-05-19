"""
Pydantic schemas for llama.cpp grammar-library API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta, default_offset_pagination_aliases

ChatGrammarValidationStatus = Literal["unchecked", "valid", "invalid"]


class ChatGrammarBase(BaseModel):
    """Shared fields for stored chat grammars."""

    name: str = Field(..., min_length=1, max_length=200, description="Display name for the saved grammar")
    description: str | None = Field(None, max_length=2000, description="Optional user-authored description")
    grammar_text: str = Field(
        ...,
        min_length=1,
        max_length=200_000,
        description="Inline GBNF grammar text stored for later reuse",
    )


class ChatGrammarCreate(ChatGrammarBase):
    """Request schema for creating a saved grammar."""


class ChatGrammarUpdate(BaseModel):
    """Request schema for updating a saved grammar."""

    version: int | None = Field(None, ge=1, description="Optional optimistic-lock version")
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    grammar_text: str | None = Field(None, min_length=1, max_length=200_000)
    validation_status: ChatGrammarValidationStatus | None = Field(None)
    validation_error: str | None = Field(None, max_length=4000)
    last_validated_at: datetime | None = Field(None)
    is_archived: bool | None = Field(None)

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "ChatGrammarUpdate":
        if (
            self.name is None
            and self.description is None
            and self.grammar_text is None
            and self.validation_status is None
            and self.validation_error is None
            and self.last_validated_at is None
            and self.is_archived is None
        ):
            raise ValueError("At least one updatable field must be provided")
        return self


class ChatGrammarResponse(ChatGrammarBase):
    """Response schema for a saved grammar."""

    id: str = Field(..., description="Stable grammar identifier")
    validation_status: ChatGrammarValidationStatus = Field(..., description="Current validation state")
    validation_error: str | None = Field(None, description="Most recent validation error, if any")
    last_validated_at: datetime | None = Field(None, description="When the grammar was last validated")
    is_archived: bool = Field(False, description="Whether the grammar is archived from normal listings")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: int = Field(..., description="Optimistic-lock version")

    model_config = ConfigDict(from_attributes=True)


class ChatGrammarListResponse(BaseModel):
    """Response schema for listing saved grammars."""

    items: list[ChatGrammarResponse] = Field(default_factory=list, description="Saved grammar records")
    total: int = Field(..., description="Total number of matching saved grammars")
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> "ChatGrammarListResponse":
        return default_offset_pagination_aliases(self)
