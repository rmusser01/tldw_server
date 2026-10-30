"""Core-owned study-pack request models, re-exported by the API schemas package (TASK-13950.15)."""

from collections.abc import Mapping
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

StudyPackSourceType = Literal["note", "media", "message"]


class StudyPackSourceSelection(BaseModel):
    """API-facing source selection for study-pack generation requests."""

    model_config = ConfigDict(populate_by_name=True)

    source_type: StudyPackSourceType
    source_id: str = Field(..., min_length=1)
    label: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("label", "source_title"),
    )
    excerpt_text: Optional[str] = None
    locator: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", mode="before")
    @classmethod
    def validate_source_id(cls, value: Any) -> str:
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError("source_id must not be blank")
        return str(value)

    @field_validator("label", "excerpt_text", mode="before")
    @classmethod
    def validate_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("locator", mode="before")
    @classmethod
    def validate_locator(cls, value: Any) -> dict[str, Any]:
        if value in (None, "", [], ()):
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("locator must be a mapping")
        return {
            str(key): item
            for key, item in value.items()
            if item not in (None, "", [], {})
        }


class StudyPackCreateJobRequest(BaseModel):
    """Request body for enqueuing a new study-pack generation job."""

    title: str = Field(..., min_length=1)
    workspace_id: Optional[str] = None
    deck_mode: Literal["new"] = "new"
    source_items: list[StudyPackSourceSelection] = Field(..., min_length=1)
