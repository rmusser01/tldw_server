# tldw_Server_API/app/api/v1/schemas/prompts_schemas.py
#
# Imports
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, validator
from datetime import datetime
#
# Third-party Imports
#
# Local Imports
#
########################################################################################################################
#
# --- Keyword Schemas ---
class KeywordBase(BaseModel):
    keyword_text: str = Field(..., min_length=1, max_length=100, description="The text of the keyword.")


class KeywordCreate(KeywordBase):
    pass


class KeywordResponse(KeywordBase):
    id: int
    uuid: str

    # last_modified: datetime # If you want to expose these
    # version: int

    model_config = ConfigDict(from_attributes=True)  # For compatibility if directly mapping from DB model in future


# --- Prompt Schemas ---
class PromptBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Unique name of the prompt.")
    author: Optional[str] = Field(None, max_length=100, description="Author of the prompt.")
    details: Optional[str] = Field(None, max_length=4000, description="Detailed description or notes about the prompt.")
    system_prompt: Optional[str] = Field(None, max_length=20000, description="The system part of the prompt.")
    user_prompt: Optional[str] = Field(None, max_length=20000, description="The user part of the prompt.")


class PromptCreate(PromptBase):
    keywords: Optional[List[str]] = Field(None, description="List of keyword strings to associate with the prompt.")


class PromptUpdate(BaseModel):  # For partial updates if we add a PATCH endpoint
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    author: Optional[str] = Field(None, max_length=100)
    details: Optional[str] = Field(None, max_length=4000)
    system_prompt: Optional[str] = Field(None, max_length=20000)
    user_prompt: Optional[str] = Field(None, max_length=20000)
    keywords: Optional[List[str]] = None  # To update keywords


class PromptResponse(PromptBase):
    id: int
    uuid: str
    last_modified: datetime
    version: int
    keywords: List[str] = Field(default_factory=list, description="Keywords associated with the prompt.")
    deleted: bool = Field(..., description="Indicates if the prompt is soft-deleted.")

    model_config = ConfigDict(from_attributes=True)


class PromptBriefResponse(BaseModel):
    id: int
    uuid: str
    name: str
    author: Optional[str]
    last_modified: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedPromptsResponse(BaseModel):
    items: List[PromptBriefResponse]
    total_pages: int
    current_page: int
    total_items: int


class PromptSearchResultItem(PromptResponse):  # Or a more specific search result schema
    relevance_score: Optional[float] = None  # If FTS provides it


class PromptSearchResponse(BaseModel):
    items: List[PromptSearchResultItem]
    total_matches: int
    page: int
    per_page: int


class ExportResponse(BaseModel):
    message: str
    file_path: Optional[str] = None  # Could be a download link or internal path for admin
    file_content_b64: Optional[str] = None  # For direct download via API


# --- Sync Log (Admin/Debug) ---
class SyncLogEntryResponse(BaseModel):
    change_id: int
    entity: str
    entity_uuid: str
    operation: str
    timestamp: datetime
    client_id: str
    version: int
    payload: Optional[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)

#
# End of prompts_schemas.py
#######################################################################################################################
