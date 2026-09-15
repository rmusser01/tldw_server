import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class KnowledgeSaveRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation to backlink")
    message_id: Optional[str] = Field(None, description="Optional message ID to backlink")
    scope_type: Literal["global", "workspace"] = Field(
        "global",
        description="Conversation scope type",
    )
    workspace_id: Optional[str] = Field(
        None,
        description="Workspace ID when scope_type='workspace'",
    )
    snippet: str = Field(..., min_length=1, description="Snippet content to save")
    tags: Optional[list[str]] = Field(None, description="Optional tags to attach as keywords")
    make_flashcard: bool = Field(False, description="If true, also create a flashcard from the snippet")
    flashcard_front: Optional[str] = Field(None, description="Reviewed question; required when creating a flashcard")
    flashcard_back: Optional[str] = Field(None, description="Reviewed answer; required when creating a flashcard")
    export_to: Literal["none", "notion", "wiki"] = Field(
        "none", description="Optional export target; disabled unless chat connectors v2 is enabled"
    )

    @field_validator("snippet", "flashcard_front", "flashcard_back")
    @classmethod
    def _visible_content(cls, value: Optional[str]) -> Optional[str]:
        """Persist answer text, excluding closed or unfinished model reasoning blocks."""
        if value is None:
            return None
        return re.sub(
            r"<(think|reason|reasoning|thought)>.*?(?:</\1>|$)",
            "",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        cleaned = []
        seen = set()
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s)
        return cleaned or None

    @model_validator(mode="after")
    def _validate_scope(self) -> "KnowledgeSaveRequest":
        if not self.snippet:
            raise ValueError("Snippet must contain visible answer text")
        if self.make_flashcard and (not self.flashcard_front or not self.flashcard_back):
            raise ValueError("A flashcard requires both a question and an answer")
        if self.scope_type == "workspace" and not self.workspace_id:
            raise ValueError("workspace_id is required when scope_type='workspace'")
        if self.scope_type == "global":
            self.workspace_id = None
        return self


class KnowledgeSaveResponse(BaseModel):
    note_id: str
    flashcard_id: Optional[str] = None
    conversation_id: str
    message_id: Optional[str] = None
    export_status: Literal["not_requested", "skipped_disabled", "queued", "completed"] = "not_requested"
    export_job_id: Optional[str] = None
