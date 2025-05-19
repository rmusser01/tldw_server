# Server_API/app/api/schemas/media_models.py
# Description: This code provides schema models for usage with the /media endpoint.
#
# Imports
import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Literal
#
# 3rd-party imports
from fastapi import HTTPException
from pydantic import BaseModel, Field, validator, computed_field, field_validator
from pydantic_core.core_schema import ValidationInfo


#
# Local Imports
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
    analysis: Optional[str] = None
    prompt: Optional[str] = None
    keywords: Optional[List[str]] = None

# Make prompt and analysis_content REQUIRED so missing them yields 422
class VersionCreateRequest(BaseModel):
    content: str
    prompt: str
    analysis_content: str

class VersionResponse(BaseModel):
    id: int
    version_number: int
    created_at: str
    content_length: int

class VersionRollbackRequest(BaseModel):
    version_number: int

class SearchRequest(BaseModel):
    query: Optional[str] = None
    fields: List[str] = ["title", "content"]
    exact_phrase: Optional[str] = None
    media_types: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None
    must_have: Optional[List[str]] = None
    must_not_have: Optional[List[str]] = None
    sort_by: Optional[str] = "relevance"
    boost_fields: Optional[Dict[str, float]] = None

# Define allowed media types using Literal for validation
MediaType = Literal['video', 'audio', 'document', 'pdf', 'ebook']

# Define allowed chunking methods (adjust as needed based on your library)
ChunkMethod = Literal['semantic', 'tokens', 'paragraphs', 'sentences','words', 'chapter', 'json' ]

# Define allowed PDF parsing engines
PdfEngine = Literal['pymupdf4llm', 'pymupdf', 'docling'] # Add others if supported

class ChunkingOptions(BaseModel):
    """Pydantic model for chunking specific options"""
    perform_chunking: bool = Field(True, description="Enable chunk-based processing of the media content")
    chunk_method: Optional[ChunkMethod] = Field(None, description="Method used to chunk content (e.g., 'sentences', 'recursive', 'chapter')")
    use_adaptive_chunking: bool = Field(False, description="Whether to enable adaptive chunking")
    use_multi_level_chunking: bool = Field(False, description="Whether to enable multi-level chunking")
    chunk_language: Optional[str] = Field(None, description="Optional language override for chunking (ISO 639-1 code, e.g., 'en')")
    chunk_size: int = Field(500, gt=0, description="Target size of each chunk (positive integer)")
    chunk_overlap: int = Field(200, ge=0, description="Overlap size between chunks (non-negative integer)")
    custom_chapter_pattern: Optional[str] = Field(None, description="Optional regex pattern for custom chapter splitting (ebook/docs)")

    @field_validator('chunk_method', mode='before')
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Optional[Any]: # Accept Any for input
        if v == "":
            return None
        return v

    @field_validator('chunk_overlap')
    @classmethod
    def overlap_less_than_size(cls, v: int, info: ValidationInfo) -> int:
        # Check if 'chunk_size' is available in the already validated data
        if 'chunk_size' in info.data and info.data['chunk_size'] is not None:
            chunk_size = info.data['chunk_size']
            if v >= chunk_size:
                raise ValueError('chunk_overlap must be less than chunk_size')
        # If chunk_size hasn't been validated yet or is missing, this check might implicitly pass
        # or you might want to handle that case explicitly if chunk_size is always required before overlap.
        return v

    @field_validator('custom_chapter_pattern')
    @classmethod
    def validate_regex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e: # Catch specific error
                raise ValueError(f"Invalid regex pattern provided for custom_chapter_pattern: {v}. Error: {e}")
        return v

class AudioVideoOptions(BaseModel):
    """Pydantic model for Audio/Video specific options"""
    transcription_model: str = Field("deepdml/faster-distil-whisper-large-v3.5", description="Model ID for audio/video transcription (e.g., from Hugging Face)")
    transcription_language: str = Field("en", description="Language for audio/video transcription (ISO 639-1 code)")
    diarize: bool = Field(False, description="Enable speaker diarization (audio/video)")
    timestamp_option: bool = Field(True, description="Include timestamps in the transcription (audio/video)")
    vad_use: bool = Field(False, description="Enable Voice Activity Detection filter during transcription (audio/video)")
    perform_confabulation_check_of_analysis: bool = Field(False, description="Enable a confabulation check on analysis (if applicable)")

class PdfOptions(BaseModel):
    """Pydantic model for PDF specific options"""
    pdf_parsing_engine: Optional[PdfEngine] = Field("pymupdf4llm", description="PDF parsing engine to use")

class AddMediaForm(ChunkingOptions, AudioVideoOptions, PdfOptions):
    """
    Pydantic model representing the form data for the /add endpoint.
    Excludes 'files' (handled via File(...)) and 'token' (handled via Header(...)).
    """
    # --- Required Fields ---
    media_type: MediaType = Field(..., description="Type of media")

    # --- Input Sources ---
    # Note: 'files' is handled separately by FastAPI's File() parameter
    urls: Optional[List[str]] = Field(None, description="List of URLs of the media items to add")

    # --- Common Optional Fields ---
    title: Optional[str] = Field(None, description="Optional title (applied if only one item processed)")
    author: Optional[str] = Field(None, description="Optional author (applied similarly to title)")
    keywords_str: str = Field("", alias="keywords", description="Comma-separated keywords (applied to all processed items)")
    custom_prompt: Optional[str] = Field(None, description="Optional custom prompt (applied to all)")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt (applied to all)")
    overwrite_existing: bool = Field(False, description="Overwrite any existing media with the same identifier (URL/filename)")
    keep_original_file: bool = Field(False, description="Whether to retain original uploaded files after processing")
    perform_analysis: bool = Field(True, description="Perform analysis (e.g., summarization) if applicable (default=True)")

    # --- Video/Audio Specific Timing --- ADD THESE ---
    start_time: Optional[str] = Field(None, description="Optional start time for processing (e.g., HH:MM:SS or seconds)")
    end_time: Optional[str] = Field(None, description="Optional end time for processing (e.g., HH:MM:SS or seconds)")
    # -----------------------------------------------

    # --- Integration Options ---
    api_name: Optional[str] = Field(None, description="Optional API name for integration (e.g., OpenAI)")
    api_key: Optional[str] = Field(None, description="Optional API key for integration") # Consider secure handling/storage
    use_cookies: bool = Field(False, description="Whether to attach cookies to URL download requests")
    cookies: Optional[str] = Field(None, description="Cookie string if `use_cookies` is set to True")

    # --- Deprecated/Less Common ---
    perform_rolling_summarization: bool = Field(False, description="Perform rolling summarization (legacy?)")
    summarize_recursively: bool = Field(False, description="Perform recursive summarization on chunks (if chunking enabled)")

    # --- Computed Fields / Validators ---
    @computed_field
    @property
    def keywords(self) -> List[str]:
        """Parses the comma-separated keywords string into a list."""
        if not self.keywords_str:
            return []
        return [k.strip() for k in self.keywords_str.split(",") if k.strip()]

    # Use alias for 'keywords' field to accept 'keywords' in the form data
    # but internally work with 'keywords_str' before parsing.
    model_config = {
        "populate_by_name": True # Allows using 'alias' for fields
    }

    @field_validator('start_time', 'end_time')
    @classmethod
    def check_time_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":  # MODIFIED: Treat empty string like None
            return None  # Return None, which is valid for Optional[str]

        # Example basic check: Allow seconds or HH:MM:SS format
        if re.fullmatch(r'\d+', v) or re.fullmatch(r'\d{1,2}:\d{2}:\d{2}', v):
            return v
        raise ValueError("Time format must be seconds or HH:MM:SS")

    # Validator to ensure 'cookies' is provided if 'use_cookies' is True
    @field_validator('cookies')
    @classmethod
    def check_cookies_provided(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # Access other validated fields via info.data
        # Check if 'use_cookies' exists in data AND is True
        if info.data.get('use_cookies') and not v:
            raise ValueError("Cookie string must be provided when 'use_cookies' is set to True.")
        return v


class MediaItemProcessResponse(BaseModel):
    """
    Pydantic model for media item details after processing. Details returned from a processing request
    """
    status: Literal['Success', 'Error', 'Warning']
    input_ref: str # The original URL or filename provided by the user
    processing_source: str # The actual path or URL used by the processor, e.g., temp file path
    media_type: (Literal['video', 'audio', 'document', 'pdf', 'ebook']) # 'video', 'pdf', 'audio', etc.
    metadata: Dict[str, Any] # Extracted info like title, author, duration, etc.
    content: str # The main extracted text or full transcript
    segments: Optional[List[Dict[str, Any]]] # For timestamped transcripts, if applicable
    chunks: Optional[List[Dict[str, Any]]] # If chunking happened within the processor
    analysis: Optional[str] # The generated analysis, if analysis was performed
    analysis_details: Optional[Dict[str, Any]] # e.g., whisper model used, summarization prompt
    error: Optional[str] # Detailed error message if status != 'Success'
    warnings: Optional[List[str]] # For non-critical issues
    model_config = {
        "extra": "forbid",  # Disallow extra fields not defined in the model
    }

######################## Video Ingestion Model ###################################
#
# This is a schema for video ingestion and analysis.

class ProcessVideosForm(AddMediaForm):
    """
    Same field-surface as AddMediaForm, but:
      • media_type forced to `"video"` (so client does not need to send it)
      • keep_original_file defaults to False (tmp dir always wiped)
      • perform_analysis stays default=True (caller may disable)
    """
    media_type: Literal["video"] = "video"
    keep_original_file: bool = Field(False)

class VideoIngestRequest(BaseModel):
    # You can rename / remove / add fields as you prefer:
    mode: str = "persist"  # "ephemeral" or "persist"

    urls: Optional[List[str]] = None  # e.g., YouTube, Vimeo, local-file references

    transcription_model: str = "distil-large-v3"
    diarize: bool = False
    vad: bool = True
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
    confab_checkbox: bool = False

    start_time: Optional[str] = None
    end_time: Optional[str] = None

#
# End of Video ingestion and analysis model schema
####################################################################################


######################## Audio Ingestion Model ###################################
#
# This is a schema for audio ingestion and analysis.

class ProcessAudiosForm(AddMediaForm):
    """Identical surface to AddMediaForm but restricted to 'audio'."""
    media_type: Literal["audio"] = "audio"
    keep_original_file: bool = Field(False)

class AudioIngestRequest(BaseModel):
    mode: str = "persist"  # "ephemeral" or "persist"

    # Normal audio vs. podcast
    is_podcast: bool = False

    urls: Optional[List[str]] = None
    transcription_model: str = "distil-large-v3"
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
