"""Validated input for advisory model classification."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModerationClassificationRequest(BaseModel):
    """An explicit sample sent to a configured model and saved for staff review."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    provider: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    model: str = Field(min_length=1, max_length=200)
    criterion: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("provider", "model", "criterion", "text")
    @classmethod
    def _nonblank_classification_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank")
        return value
