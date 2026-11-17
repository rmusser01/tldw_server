from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from tldw_Server_API.app.core.Chat.knowledge_save import visible_knowledge_text
from tldw_Server_API.app.core.Utils.keyword_identity import dedupe_keywords


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
    flashcard_front: Optional[str] = Field(None, description="Reviewed question; legacy requests may use the linked parent question")
    flashcard_back: Optional[str] = Field(None, description="Reviewed answer; legacy requests may use a linked public answer excerpt")
    export_to: Literal["none", "notion", "wiki"] = Field(
        "none", description="Optional export target; disabled unless chat connectors v2 is enabled"
    )

    @field_validator("snippet", "flashcard_front", "flashcard_back")
    @classmethod
    def _visible_content(cls, value: Optional[str]) -> Optional[str]:
        """Persist answer text, excluding closed or unfinished model reasoning blocks."""
        if value is None:
            return None
        return visible_knowledge_text(value)

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return (dedupe_keywords(value) or None) if value is not None else None

    @model_validator(mode="after")
    def _validate_scope(self) -> "KnowledgeSaveRequest":
        if not self.snippet:
            raise ValueError("Snippet must contain visible answer text")
        if (
            self.make_flashcard
            and {"flashcard_front", "flashcard_back"}.intersection(self.model_fields_set)
            and (not self.flashcard_front or not self.flashcard_back)
        ):
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
