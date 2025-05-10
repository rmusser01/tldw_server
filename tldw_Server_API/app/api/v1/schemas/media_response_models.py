# Server_API/app/api/schemas/media_response_models.py
# Description: This code provides schema models for responses from the /media endpoint.
#
# Imports
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
#
# 3rd-party imports
from pydantic import BaseModel, Field, HttpUrl
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

# --- Common Reusable Models ---

class PaginationInfo(BaseModel):
    """Model for pagination details used in list responses."""
    page: int = Field(..., description="The current page number.", json_schema_extra={"example": 1})
    results_per_page: int = Field(..., description="Number of items requested per page.", json_schema_extra={"example": 10})
    total_pages: int = Field(..., description="Total number of pages available.", json_schema_extra={"example": 5})
    total_items: int = Field(..., description="Total number of items matching the query.", json_schema_extra={"example": 48})

class PaginationInfoSearch(BaseModel):
    """Model for pagination details used specifically in search responses."""
    page: int = Field(..., description="The current page number.", json_schema_extra={"example": 1})
    per_page: int = Field(..., description="Number of items requested per page.", json_schema_extra={"example": 10})
    total: int = Field(..., description="Total number of matching items found.", json_schema_extra={"example": 42})
    total_pages: int = Field(..., description="Total number of pages available based on total items and per_page.", json_schema_extra={"example": 5})

class SuccessMessage(BaseModel):
    """A generic success message response."""
    message: str = Field(..., description="Success confirmation message.", json_schema_extra={"example": "Operation successful."})
    detail: Optional[str] = Field(None, description="Optional additional details.", json_schema_extra={"example": "Item processed."})


######################## /api/v1/media/ Endpoint Response Models ########################
#
#

# --- /api/v1/media/ (List All Media) ---

class MediaListItem(BaseModel):
    """Represents a single media item in a list response."""
    id: int = Field(..., description="Unique identifier for the media item.", json_schema_extra={"example": 123})
    title: str = Field(..., description="Title of the media item.", json_schema_extra={"example": "My Awesome Video"})
    url: str = Field(..., description="Relative API URL to fetch the full details of this item.", json_schema_extra={"example": "/api/v1/media/123"})
    type: str = Field(..., description="Type of the media (e.g., 'video', 'audio', 'pdf').", json_schema_extra={"example": "video"}) # <-- ADDED

class MediaListResponse(BaseModel):
    """Response model for listing all media items (GET /)."""
    items: List[MediaListItem] = Field(..., description="A list of media items on the current page.")
    pagination: PaginationInfo = Field(..., description="Pagination details for the media list.")


# --- /api/v1/media/{media_id} (Get Media Item Details) ---

class MediaSourceDetail(BaseModel):
    """Details about the original source of the media."""
    url: Optional[str] = Field(None, description="Original URL of the media source (if applicable).", json_schema_extra={"example": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    title: str = Field(..., description="Title of the media item.", json_schema_extra={"example": "Understanding AI"})
    duration: Optional[Union[float, str]] = Field(None, description="Duration of the media (e.g., seconds or HH:MM:SS string).", json_schema_extra={"example": 3600.5})
    type: str = Field(..., description="Type of the media (e.g., 'video', 'audio', 'pdf').", json_schema_extra={"example": "video"})

class MediaProcessingDetail(BaseModel):
    """Details about the processing applied to the media."""
    prompt: Optional[str] = Field(None, description="The last prompt used for analysis or processing.", json_schema_extra={"example": "Summarize the key points."})
    analysis: Optional[str] = Field(None, description="The last analysis or summary generated.", json_schema_extra={"example": "The video discusses the fundamentals of AI..."})
    model: Optional[str] = Field(None, description="The transcription model used (if applicable).", json_schema_extra={"example": "whisper-large-v3"})
    timestamp_option: Optional[bool] = Field(None, description="Whether timestamps were requested/generated during transcription.", json_schema_extra={"example": True})

class MediaContentDetail(BaseModel):
    """Details about the extracted content of the media."""
    metadata: Dict[str, Any] = Field(..., description="Extracted metadata from the media file or source.", json_schema_extra={"example": """"channel": "Tech Explained", "upload_date": "2023-10-26" """})
    text: str = Field(..., description="The full extracted text or transcript.", json_schema_extra={"example": "Hello and welcome to the channel..."})
    word_count: int = Field(..., description="Approximate word count of the extracted text.", json_schema_extra={"example": 1500})

class VersionDetailResponse(BaseModel):
    """Represents the details of a single media version."""
    media_id: int = Field(..., description="ID of the parent media item.", json_schema_extra={"example": 123})
    version_number: int = Field(..., description="Sequential version number.", json_schema_extra={"example": 2})
    created_at: datetime = Field(..., description="Timestamp when this version was created.")
    #content_hash: Optional[str] = Field(None, description="SHA-256 hash of the content for this version.", json_schema_extra={"example": "a1b2c3d4..."})
    prompt: Optional[str] = Field(None, description="Prompt associated with this version.", json_schema_extra={"example": "Summarize the previous version."})
    analysis_content: Optional[str] = Field(None, description="Analysis content associated with this version.", json_schema_extra={"example": "This version focuses on..."})
    # Conditionally include content based on 'include_content' query param
    content: Optional[str] = Field(None, description="The full content of this version (if requested).", json_schema_extra={"example": "This is the text content for version 2..."})

    class Config:
        orm_mode = True # If fetching directly from an ORM model like SQLAlchemy

class MediaDetailResponse(BaseModel):
    """Response model for retrieving a single media item's details (GET /{media_id})."""
    media_id: int = Field(..., description="Unique identifier for the media item.", json_schema_extra={"example": 123})
    source: MediaSourceDetail = Field(..., description="Details about the original source.")
    processing: MediaProcessingDetail = Field(..., description="Details about the processing applied.")
    content: MediaContentDetail = Field(..., description="Details about the extracted content.")
    keywords: List[str] = Field(..., description="Keywords associated with the media item.", json_schema_extra={"example": ["ai", "machine learning", "tech"]})
    timestamps: List[str] = Field(..., description="List of timestamps extracted from the content (if applicable).", json_schema_extra={"example": ["00:00:05", "00:01:12"]})
    versions: List[VersionDetailResponse] = Field([], description="List of document versions, if applicable.") # <-- ADDED, default empty list


    class Config:
        # If your source data might have extra fields not in the model, use this:
        # extra = "ignore"
        # Example for schema generation
        schema_extra = {
            "example": {
                "media_id": 123,
                "source": {
                    "url": "https://example.com/podcast.mp3",
                    "title": "Tech Podcast Ep. 5",
                    "duration": "01:15:30",
                    "type": "audio"
                },
                "processing": {
                    "prompt": "Summarize the main topics discussed.",
                    "analysis": "The podcast covers new AI advancements and their implications.",
                    "model": "deepdml/faster-distil-whisper-large-v3.5",
                    "timestamp_option": True
                },
                "content": {
                    "metadata": {"episode": 5, "guests": ["Alice", "Bob"]},
                    "text": "[00:00:00] Intro music...\n[00:00:15] Welcome back to the show...",
                    "word_count": 8500
                },
                "keywords": ["podcast", "technology", "ai", "summary"],
                "timestamps": ["00:00:00", "00:00:15"]
            }
        }


# --- /api/v1/media/{media_id}/versions ---

# class VersionDetailResponse(BaseModel):
#     """Represents the details of a single media version."""
#     media_id: int = Field(..., description="ID of the parent media item.", json_schema_extra={"example": 123})
#     version_number: int = Field(..., description="Sequential version number.", json_schema_extra={"example": 2})
#     created_at: datetime = Field(..., description="Timestamp when this version was created.")
#     #content_hash: Optional[str] = Field(None, description="SHA-256 hash of the content for this version.", json_schema_extra={"example": "a1b2c3d4..."})
#     prompt: Optional[str] = Field(None, description="Prompt associated with this version.", json_schema_extra={"example": "Summarize the previous version."})
#     analysis_content: Optional[str] = Field(None, description="Analysis content associated with this version.", json_schema_extra={"example": "This version focuses on..."})
#     # Conditionally include content based on 'include_content' query param
#     content: Optional[str] = Field(None, description="The full content of this version (if requested).", json_schema_extra={"example": "This is the text content for version 2..."})
#
#     class Config:
#         orm_mode = True # If fetching directly from an ORM model like SQLAlchemy

class VersionListResponse(BaseModel):
    """Response model for listing versions of a media item."""
    versions: List[VersionDetailResponse] = Field(..., description="A list of available versions.")

class VersionCreateResponse(BaseModel):
    """Response model after successfully creating a new version."""
    message: str = Field(..., description="Confirmation message.", json_schema_extra={"example": "New version created successfully."})
    media_id: int = Field(..., description="ID of the media item.", json_schema_extra={"example": 123})
    version_number: int = Field(..., description="The number of the newly created version.", json_schema_extra={"example": 3})

class VersionRollbackResponse(BaseModel):
    """Response model after successfully rolling back to a version."""
    success: str = Field(..., description="Confirmation message.", json_schema_extra={"example": "Rolled back to version 2."})
    new_version_number: int = Field(..., description="The version number created as a result of the rollback.", json_schema_extra={"example": 4})


# --- /api/v1/media/{media_id} (Update) ---

class MediaUpdateResponse(BaseModel):
    """Response model after successfully updating a media item (PUT /{media_id})."""
    message: str = Field(..., description="Confirmation message.", json_schema_extra={"example": "Media item 123 updated successfully."})
    media_id: int = Field(..., description="ID of the updated media item.", json_schema_extra={"example": 123})
    new_version: Optional[int] = Field(None, description="The version number created if content was updated.", json_schema_extra={"example": 5})


# --- /api/v1/media/search ---

class MediaSearchResultItem(BaseModel):
    """Represents a single item in the media search results."""
    id: int = Field(..., description="Unique identifier for the media item.", json_schema_extra={"example": 123})
    url: str = Field(..., description="Relative API URL to fetch the full details of this item.", json_schema_extra={"example": "/api/v1/media/123"})
    title: str = Field(..., description="Title of the media item.", json_schema_extra={"example": "Searchable Document"})
    type: str = Field(..., description="Type of the media.", json_schema_extra={"example": "document"})
    content_preview: Optional[str] = Field(None, description="A short preview of the media's content.", json_schema_extra={"example": "This document contains important information about..."})
    author: Optional[str] = Field(None, description="Author of the media item.", json_schema_extra={"example": "Jane Doe"})
    date: Optional[datetime] = Field(None, description="Creation or publication date of the media.")
    keywords: List[str] = Field(..., description="Keywords associated with the media item.", json_schema_extra={"example": ["search", "document", "important"]})

class MediaSearchResponse(BaseModel):
    """Response model for the media search endpoint (GET /search)."""
    results: List[MediaSearchResultItem] = Field(..., description="List of media items matching the search criteria.")
    pagination: PaginationInfoSearch = Field(..., description="Pagination details for the search results.")

######################## Media Processing Response Model ###################################
#
# This is a schema for media processing response models.

# --- /api/v1/media/add (Ingest with Persistence) ---
# --- AND /api/v1/media/process-* (Ingest without Persistence) ---

class MediaItemProcessResult(BaseModel):
    """
    Standard response structure for a single item processed via /add or /process-* endpoints.
    """
    status: str = Field(..., description="Processing status ('Success', 'Skipped', 'Warning', 'Error').", json_schema_extra={"example": "Success"})
    input_ref: str = Field(..., description="The original URL or filename provided by the user.", json_schema_extra={"example": "https://example.com/my_video.mp4"})
    processing_source: Optional[str] = Field(None, description="The actual source used by the processor (e.g., temp file path or URL).",json_schema_extra={"example": "/tmp/media_processing_xyz/my_video.mp4"})
    media_type: str = Field(..., description="Detected or specified media type.", json_schema_extra={"example": "video"})
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata.", json_schema_extra={"example": {"title": "My Video", "duration": 120.5}})
    content: Optional[str] = Field(None, description="Extracted text content or transcript.")
    transcript: Optional[str] = Field(None, description="Alias or specific field for transcript.") # Added for clarity
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Time-coded transcript segments, if applicable.")
    chunks: Optional[List[Any]] = Field(None, description="List of content chunks, if chunking was performed.") # Type depends on chunking output
    analysis: Optional[str] = Field(None, description="Generated analysis or summary.")
    summary: Optional[str] = Field(None, description="Alias or specific field for summary.") # Added for clarity
    analysis_details: Optional[Dict[str, Any]] = Field(None, description="Details about the analysis process (e.g., model used).", json_schema_extra={"example": {"transcription_model": "whisper-large-v3"}})
    error: Optional[str] = Field(None, description="Error message if status is 'Error'.")
    warnings: Optional[List[str]] = Field(None, description="List of warnings if status is 'Warning' or non-critical issues occurred.")
    # --- Persistence Specific (Null for /process-* endpoints) ---
    db_id: Optional[int] = Field(None, description="Database ID if persisted, null otherwise.", json_schema_extra={"example": 456})
    db_message: Optional[str] = Field(None, description="Message related to database interaction.", json_schema_extra={"example": "Media added to database."})
    # --- General Message ---
    message: Optional[str] = Field(None, description="General status message (e.g., for skipped items).", json_schema_extra={"example": "Media exists (ID: 456), overwrite=False"})

    class Config:
        # Allow extra fields if processing functions might return them
        extra = "allow"


class BatchMediaAddResponse(BaseModel):
    """Response model for the /add endpoint (POST /api/v1/media/add)."""
    results: List[MediaItemProcessResult] = Field(..., description="List containing the processing result for each input item.")

class BatchMediaProcessResponse(BaseModel):
    """
    Response model for the /process-* endpoints (e.g., POST /api/v1/media/process-videos).
    These endpoints process media but *do not* persist to the database.
    """
    processed_count: int = Field(..., description="Number of items successfully processed (or with warnings).", json_schema_extra={"example": 2})
    errors_count: int = Field(..., description="Number of items that failed processing.", json_schema_extra={"example": 1})
    errors: List[str] = Field(..., description="List of unique error messages encountered.", json_schema_extra={"example": ["Download failed for URL X", "Transcription timed out for file Y"]})
    results: List[MediaItemProcessResult] = Field(..., description="List containing the processing result for each input item.")
    # Specific outputs like confabulation might be added optionally
    confabulation_results: Optional[Any] = Field(None, description="Results from confabulation check, if performed.")

#
# End of Media Processing Response Model Schemas
####################################################################################


#####################################################################################
#
# /api/v1/media/process-xml Response Model

# --- /api/v1/media/process-xml ---

class ProcessXmlResponse(BaseModel):
    """Response model for the /process-xml endpoint."""
    status: str = Field(..., description="Outcome status ('ephemeral-ok' or 'persist-ok').", json_schema_extra={"example": "persist-ok"})
    media_id: Union[str, int] = Field(..., description="Database ID (if persisted) or ephemeral storage ID.", json_schema_extra={"example": "eph-xyz123"})
    title: Optional[str] = Field(None, description="Title extracted or provided for the XML content.", json_schema_extra={"example": "My XML Document"})

#
# End of /api/v1/media/process-xml Response Model
#####################################################################################

######################## Web-Scraping Ingestion Model ###################################
#
# This is a schema for Web-Scraping ingestion and analysis response.

# --- /api/v1/media/ingest-web-content ---
# --- /api/v1/media/process-web-scraping ---

class WebScrapedItemResult(BaseModel):
    """Represents a single scraped article's data in the response."""
    # Fields depend heavily on your scraping library's output, adjust as needed
    url: HttpUrl = Field(..., description="The URL that was scraped.")
    title: Optional[str] = Field(None, description="Extracted or provided title.")
    author: Optional[str] = Field(None, description="Extracted or provided author.")
    content: Optional[str] = Field(None, description="Main text content extracted from the page.")
    # Optional fields based on processing steps
    keywords: Optional[Union[str, List[str]]] = Field(None, description="Keywords associated with the content.")
    analysis: Optional[str] = Field(None, description="Generated summary or analysis (if requested).")
    chunks: Optional[List[Any]] = Field(None, description="Content chunks (if requested).")
    ingested_at: Optional[datetime] = Field(None, description="Timestamp when the content was processed.")
    # Add other fields returned by your scraping/processing functions
    metadata: Optional[Dict[str, Any]] = Field(None, description="Other extracted metadata.")
    extraction_successful: Optional[bool] = Field(None, description="Flag indicating if scraping succeeded for this URL.")

class WebProcessResponse(BaseModel):
    """Response model for web scraping endpoints."""
    status: str = Field(..., description="Overall status of the batch operation ('success', 'warning', 'error').", json_schema_extra={"example": "success"})
    message: str = Field(..., description="Summary message about the operation.", json_schema_extra={"example": "Web content processed"})
    count: int = Field(..., description="Number of items successfully processed/scraped.", json_schema_extra={"example": 5})
    # One of these will be present depending on persistence
    results: Optional[List[WebScrapedItemResult]] = Field(None, description="List of processed web content data (if not persisted or mode=ephemeral).")
    media_ids: Optional[List[Union[int, str]]] = Field(None, description="List of database IDs or ephemeral IDs for the persisted/stored items.")


#
# End of Web-Scraping ingestion and analysis response model schema
####################################################################################


####################################################################################
#
# Debug Response Models

# --- /api/v1/media/debug/schema ---

class DebugSchemaResponse(BaseModel):
    """Response model for the debug schema endpoint."""
    tables: List[str] = Field(..., description="List of tables found in the database.")
    media_columns: List[str] = Field(..., description="List of column names in the 'Media' table.")
    media_mods_columns: List[str] = Field(..., description="List of column names in the 'MediaModifications' table.")
    media_count: int = Field(..., description="Total number of rows in the 'Media' table.")

#
# End of Debug Response Model Schemas
#####################################################################################

#
# End of media_response_models.py
#######################################################################################################################
