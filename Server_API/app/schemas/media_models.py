# Server_API/app/api/schemas/media_models.py
# Description: This code provides schema models for usage with the /media endpoint.
#
# Imports
from datetime import datetime
from typing import Dict, List, Any, Optional
#
# 3rd-party imports
from fastapi import HTTPException
from pydantic import BaseModel
#
# Local Imports
from Server_API.app.core.DB_Management.DB_Manager import fetch_item_details_single
from Server_API.app.api.v1.endpoints.media import router
#
#######################################################################################################################
#
# Functions:

class MediaBase(BaseModel):
    title: str
    url: str
    content: Optional[str] = None
    is_trash: Optional[bool] = False
    trash_date: Optional[datetime] = None

class MediaCreate(MediaBase):
    """Schema for creating new media items"""
    pass

class MediaUpdate(MediaBase):
    """Schema for updating media items (could be partial)"""
    # If partial updates are needed, you can make all fields optional or handle patch-like logic
    pass

class MediaRead(MediaBase):
    id: int

    class Config:
        orm_mode = True



# MediaBase includes fields shared by creation, reading, and updating.
#
# MediaCreate extends MediaBase but doesn’t add anything new yet. (You might add required fields or validations that apply only to creation.)
#
# MediaUpdate extends MediaBase for partial updates. Often we set fields as optional in the update schema, but that’s up to your preference.
#
# MediaRead is what we return when reading from the database. Notice orm_mode = True so we can directly return SQLAlchemy objects.



class MediaRead(BaseModel):
    media_id: int
    prompt: str
    summary: str
    content: str

@router.get("/{media_id}", response_model=MediaRead)
def get_media_item(media_id: int):
    try:
        prompt, summary, content = fetch_item_details_single(media_id)
        return MediaRead(
            media_id=media_id,
            prompt=prompt,
            summary=summary,
            content=content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


######################## Video Ingestion Model ###################################
#
# This is a schema for video ingestion and analysis. It includes various options for processing the video, such as chunking, summarization, and more.

class VideoIngestRequest(BaseModel):
    # You can rename / remove / add fields as you prefer:
    mode: str = "persist"  # "ephemeral" or "persist"

    urls: Optional[List[str]] = None  # e.g., YouTube, Vimeo, local-file references

    whisper_model: str = "distil-large-v3"
    diarize: bool = False
    vad: bool = True
    custom_prompt_checkbox: bool = False
    custom_prompt: Optional[str] = None
    system_prompt: Optional[str] = None

    perform_chunking: bool = False
    chunk_method: Optional[str] = None
    max_chunk_size: int = 400
    chunk_overlap: int = 100
    use_adaptive_chunking: bool = False
    use_multi_level_chunking: bool = False
    chunk_language: Optional[str] = None
    summarize_recursively: bool = False

    api_name: Optional[str] = None
    api_key: Optional[str] = None
    keywords: Optional[str] = "default,no_keyword_set"

    use_cookies: bool = False
    cookies: Optional[str] = None

    timestamp_option: bool = True
    keep_original_video: bool = False
    confab_checkbox: bool = False
    overwrite_existing: bool = False

    start_time: Optional[str] = None
    end_time: Optional[str] = None

#
# End of Video ingestion and analysis model schema
####################################################################################


#
# End of media_models.py
#######################################################################################################################
