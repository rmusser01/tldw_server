# Server_API/app/api/schemas/media_models.py
# Description: This code provides schema models for usage with the /media endpoint.
#
# Imports
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
#
# 3rd-party imports
from fastapi import HTTPException
from pydantic import BaseModel, Field
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.DB_Manager import fetch_item_details_single
#
#######################################################################################################################
#
# Functions:

######################## /api/v1/media/ Endpoint Models ########################
#
#
class MediaItemResponse(BaseModel):
    media_id: int
    source: dict
    processing: dict
    content: dict
    keywords: List[str]
    timestamps: List[str]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PaginationInfo(BaseModel):
    page: int
    per_page: int
    total: int
    total_pages: int

class MediaItem(BaseModel):
    id: int
    url: str
    title: str
    type: str
    content_preview: Optional[str]
    author: str
    date: Optional[datetime]
    keywords: List[str]

class MediaSearchResponse(BaseModel):
    results: List[MediaItem]
    pagination: PaginationInfo

class MediaUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    prompt: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None

# Make prompt and summary REQUIRED so missing them yields 422
class VersionCreateRequest(BaseModel):
    content: str
    prompt: str
    summary: str

class VersionResponse(BaseModel):
    id: int
    version_number: int
    created_at: str
    content_length: int

class VersionRollbackRequest(BaseModel):
    version_number: int

######################## Multi-File Ingestion Model ###################################
#
# This is a schema for all media except web scraping ingestion and analysis.

class AddMediaRequest(BaseModel):
    url: str = Field(..., description="The URL of the media to add")
    media_type: str = Field(..., description="Type of media (e.g., 'audio', 'video', 'pdf')")
    title: Optional[str] = Field(None, description="Optional title of the media")
    author: Optional[str] = Field(None, description="Optional author of the media")
    keywords: str = Field("", description="Comma-separated keywords for the media")
    custom_prompt: Optional[str] = Field(None, description="Optional custom prompt")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    whisper_model: str = Field("deepml/distil-large-v3", description="Model for audio/video transcription")
    transcription_language: str = Field("en", description="Language for audio/video transcription")
    diarize: bool = Field(False, description="Enable speaker diarization")
    timestamp_option: bool = Field(True, description="Include timestamps in the transcription")
    keep_original_file: bool = Field(False, description="Whether to retain original file after processing")
    overwrite_existing: bool = Field(False, description="Overwrite any existing media with the same URL")
    perform_analysis: bool = Field(True, description="Perform analysis on the media (default=True)")
    perform_rolling_summarization: bool = Field(False, description="Perform rolling summarization on the media")
    api_name: Optional[str] = Field(None, description="Optional API name for integration")
    api_key: Optional[str] = Field(None, description="Optional API key for integration")
    pdf_parsing_engine: Optional[str] = Field(None, description="Optional PDF parsing engine")
    perform_chunking: bool = Field(True, description="Enable chunk-based processing of the media")
    chunk_method: Optional[str] = Field(None, description="Method used to chunk the media content")
    use_adaptive_chunking: bool = Field(False, description="Whether to enable adaptive chunking")
    use_multi_level_chunking: bool = Field(False, description="Whether to enable multi-level chunking")
    chunk_language: Optional[str] = Field(None, description="Optional language override for chunking")
    chunk_size: int = Field(500, description="Size of each chunk for processing")
    chunk_overlap: int = Field(200, description="Overlap size between chunks")
    use_cookies: bool = Field(False, description="Whether to attach cookies to the request (scraping scenarios)")
    cookies: Optional[str] = Field(None, description="Cookie string if `use_cookies` is set to True")
    perform_confabulation_check_of_analysis: bool = Field(
        False, description="Enable a confabulation check on analysis"
    )
    custom_chapter_pattern: Optional[str] = Field(
        None, description="Optional regex pattern for custom chapter splitting"
    )

######################## Video Ingestion Model ###################################
#
# This is a schema for video ingestion and analysis.

class VideoIngestRequest(BaseModel):
    # You can rename / remove / add fields as you prefer:
    mode: str = "persist"  # "ephemeral" or "persist"

    urls: Optional[List[str]] = None  # e.g., YouTube, Vimeo, local-file references

    whisper_model: str = "distil-large-v3"
    diarize: bool = False
    vad: bool = True
    use_custom_prompt: bool = False
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


######################## Audio Ingestion Model ###################################
#
# This is a schema for audio ingestion and analysis.

class AudioIngestRequest(BaseModel):
    mode: str = "persist"  # "ephemeral" or "persist"

    # Normal audio vs. podcast
    is_podcast: bool = False

    urls: Optional[List[str]] = None
    whisper_model: str = "distil-large-v3"
    diarize: bool = False
    keep_timestamps: bool = True

    api_name: Optional[str] = None
    api_key: Optional[str] = None
    custom_prompt: Optional[str] = None
    chunk_method: Optional[str] = None
    max_chunk_size: int = 300
    chunk_overlap: int = 0
    use_adaptive_chunking: bool = False
    use_multi_level_chunking: bool = False
    chunk_language: str = "english"

    keywords: str = ""
    keep_original_audio: bool = False
    use_cookies: bool = False
    cookies: Optional[str] = None
    custom_title: Optional[str] = None

#
# End of Audio ingestion and analysis model schema
####################################################################################


######################## Web-Scraping Ingestion Model ###################################
#
# This is a schema for Web-Scraping ingestion and analysis.

class ScrapeMethod(str, Enum):
    INDIVIDUAL = "individual"          # “Individual URLs”
    SITEMAP = "sitemap"               # “Sitemap”
    URL_LEVEL = "url_level"           # “URL Level”
    RECURSIVE = "recursive_scraping"  # “Recursive Scraping”

class IngestWebContentRequest(BaseModel):
    # Core fields
    urls: List[str]                      # Usually 1+ URLs.
    titles: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

    # Advanced scraping selection
    scrape_method: ScrapeMethod = ScrapeMethod.INDIVIDUAL
    url_level: Optional[int] = 2
    max_pages: Optional[int] = 10
    max_depth: Optional[int] = 3

    # Summarization / analysis fields
    custom_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    perform_translation: bool = False
    translation_language: str = "en"
    timestamp_option: bool = True
    overwrite_existing: bool = False
    perform_analysis: bool = True
    perform_rolling_summarization: bool = False
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    perform_chunking: bool = True
    chunk_method: Optional[str] = None
    use_adaptive_chunking: bool = False
    use_multi_level_chunking: bool = False
    chunk_language: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 200
    use_cookies: bool = False
    cookies: Optional[str] = None
    perform_confabulation_check_of_analysis: bool = False
    custom_chapter_pattern: Optional[str] = None

#
# End of Web-Scraping ingestion and analysis model schema
####################################################################################




#
# End of media_models.py
#######################################################################################################################
