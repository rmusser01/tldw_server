# app/api/v1/schemas/notes_schemas.py
#
# Imports
from typing import Optional, List, Any, Dict
from datetime import datetime
# 3rd-party Libraries
from pydantic import BaseModel, Field
#
# Local Imports
#
#######################################################################################################################
#
# Schemas:

# --- Note Schemas ---
class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Title of the note")
    content: str = Field(..., description="Content of the note")


class NoteCreate(NoteBase):
    id: Optional[str] = Field(None,
                              description="Optional client-provided UUID for the note. If None, will be auto-generated.")


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="New title for the note")
    content: Optional[str] = Field(None, description="New content for the note")
    # Ensure at least one field is provided for update, or handle in endpoint if empty update is no-op
    # Pydantic v2: model_validator


class NoteResponse(NoteBase):
    id: str = Field(..., description="UUID of the note")
    created_at: datetime = Field(..., description="Timestamp of note creation")
    last_modified: datetime = Field(..., description="Timestamp of last modification")
    version: int = Field(..., description="Version number for optimistic locking")
    client_id: str = Field(..., description="Client ID that last modified the note")
    deleted: bool = Field(..., description="Whether the note is soft-deleted")

    class Config:
        from_attributes = True  # Pydantic V2 (formerly orm_mode)


# --- Keyword Schemas ---
class KeywordBase(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=100, description="The keyword text")


class KeywordCreate(KeywordBase):
    pass


class KeywordResponse(KeywordBase):
    id: int = Field(..., description="Integer ID of the keyword")
    created_at: datetime = Field(..., description="Timestamp of keyword creation")
    last_modified: datetime = Field(..., description="Timestamp of last modification")
    version: int = Field(..., description="Version number for optimistic locking")
    client_id: str = Field(..., description="Client ID that last modified the keyword")
    deleted: bool = Field(..., description="Whether the keyword is soft-deleted")

    class Config:
        from_attributes = True


# --- Linking Schemas ---
class NoteKeywordLinkResponse(BaseModel):
    success: bool
    message: Optional[str] = None


class KeywordsForNoteResponse(BaseModel):
    note_id: str
    keywords: List[KeywordResponse]


class NotesForKeywordResponse(BaseModel):
    keyword_id: int
    notes: List[NoteResponse]


# --- General API Response Schemas ---
class DetailResponse(BaseModel):
    detail: str

#
# End of notes_schemas.py
#######################################################################################################################
