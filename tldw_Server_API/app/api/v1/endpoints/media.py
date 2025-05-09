# Server_API/app/api/v1/endpoints/media.py
# Description: This code provides a FastAPI endpoint for media ingestion, processing, and
#   storage under the `/media` endpoint
#   Filetypes supported:
#       video: `.mp4`, `.mkv`, `.avi`, `.mov`, `.flv`, `.webm`,
#       audio: `.mp3`, `.aac`, `.flac`, `.wav`, `.ogg`,
#       document: `.PDF`, `.docx`, `.txt`, `.rtf`,
#       XML,
#       archive: `.zip`,
#       eBook: `.epub`,
# FIXME
#
# Imports
import re
import sqlite3

import aiofiles
import asyncio
import functools
import hashlib
import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Literal, Union, Set
#
# 3rd-party imports
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    status,
    UploadFile
)
import httpx
from pydantic import BaseModel, ValidationError
import redis
from pydantic.v1 import Field
# API Rate Limiter/Caching via Redis
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse, Response

#
# Local Imports
#
# --- Core Libraries (New) ---
# Configuration (Import settings if needed directly, else handled by dependencies)
# from tldw_Server_API.app.core.config import settings
# Authentication & User Identification (Primary Dependency)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
# Database Instance Dependency (Gets DB based on User)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_db_for_user
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    get_paginated_files,
)
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import (
    Database, DatabaseError, InputError, ConflictError, SchemaError, # Core class & exceptions
    # Standalone Functions AVAILABLE in Media_DB_v2.py (Import ONLY what's provided):
    get_document_version,
    check_media_exists,
    fetch_keywords_for_media,
    search_media_db,
    get_all_content_from_database, # Might be useful for listing
    # Standalone functions needed for versioning endpoints:
    # Note: These are now INSTANCE methods in Media_DB_v2:
    # create_document_version -> db_instance.create_document_version(...)
    # rollback_to_version -> db_instance.rollback_to_version(...)
    # soft_delete_document_version -> db_instance.soft_delete_document_version(...)
    # get_all_document_versions -> Needs replacement logic (likely query DocumentVersions table)
    # fetch_item_details -> Needs replacement logic
    # get_full_media_details2 -> Needs replacement logic
    # get_paginated_files -> Needs replacement logic (likely query Media table with pagination)
    # check_should_process_by_url -> Needs replacement logic (likely using check_media_exists)
    # add_media_with_keywords -> db_instance.add_media_with_keywords(...)
)
from tldw_Server_API.app.api.v1.API_Deps.validations_deps import file_validator_instance
from tldw_Server_API.app.api.v1.schemas.media_response_models import PaginationInfo, MediaListResponse, MediaListItem, \
    MediaDetailResponse, VersionDetailResponse
# Media Processing
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import process_audio_files
from tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib import process_epub
from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files import process_document_content
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import process_videos
#
# Document Processing
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Utils.Utils import logging, \
    sanitize_filename, smart_download
from tldw_Server_API.app.core.Utils.Utils import logging as logger
#
# Web Scraping
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article, scrape_from_sitemap, \
    scrape_by_url_level, recursive_scrape
from tldw_Server_API.app.api.v1.schemas.media_request_models import MediaUpdateRequest, VersionCreateRequest, \
    VersionRollbackRequest, \
    IngestWebContentRequest, ScrapeMethod, MediaType, AddMediaForm, ChunkMethod, PdfEngine, ProcessVideosForm, \
    ProcessAudiosForm
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.services.web_scraping_service import process_web_scraping_task
#
#
#######################################################################################################################
#
# Functions:

# All functions below are endpoints callable via HTTP requests and the corresponding code executed as a result of it.
#

# The router is a FastAPI object that allows us to define multiple endpoints under a single prefix.
# Create a new router instance
router = APIRouter()

# Rate Limiter + Cache Setup
limiter = Limiter(key_func=get_remote_address)
# FIXME - Should be optional
# Configure Redis cache
cache = redis.Redis(host='localhost', port=6379, db=0)
CACHE_TTL = 300  # 5 minutes


# ---------------------------
# Caching Implementation
#
def get_cache_key(request: Request) -> str:
    """Generate unique cache key from request parameters"""
    params = dict(request.query_params)
    params.pop('token', None)  # Exclude security token
    return f"cache:{request.url.path}:{hash(frozenset(params.items()))}"

def cache_response(key: str, response: Dict) -> None:
    """Store response in cache with ETag"""
    content = json.dumps(response)
    etag = hashlib.md5(content.encode()).hexdigest()
    cache.setex(key, CACHE_TTL, f"{etag}|{content}")

async def get_cached_response(key: str) -> Optional[tuple]: # Changed to async def
    """Retrieve cached response with ETag (Async Version)"""
    # Await the asynchronous cache retrieval operation
    cached_value = await cache.get(key) # Added await

    if cached_value:
        # Now cached_value should be the actual data (likely bytes)
        try:
            # Decode assuming UTF-8, handle potential errors
            decoded_string = cached_value.decode('utf-8')
            # Split carefully, ensure it splits correctly
            parts = decoded_string.split('|', 1)
            if len(parts) == 2:
                etag, content_str = parts
                # Parse JSON, handle potential errors
                content = json.loads(content_str)
                return (etag, content)
            else:
                # Log or handle cases where the format is unexpected
                # logging.warning(f"Cached value for key '{key}' has unexpected format: {decoded_string}")
                print(f"Warning: Cached value for key '{key}' has unexpected format: {decoded_string}")
                return None
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, ValueError) as e:
            # Log or handle errors during decoding/parsing
            # logging.error(f"Error processing cached value for key '{key}': {e}")
            print(f"Error processing cached value for key '{key}': {e}")
            return None # Or raise an exception if appropriate

    return None # Cache miss
# --- How to call this function ---
# You would now need to call it from within another async function:
#
# async def some_other_async_function():
#     result = await get_cached_response("some_cache_key")
#     if result:
#         etag, data = result
#         print(f"Got from cache: ETag={etag}, Data={data}")
#     else:
#         print("Cache miss or error processing cache.")
#
# # To run it:
# # import asyncio
# # asyncio.run(some_other_async_function())

# ---------------------------
# Cache Invalidation
#
def invalidate_cache(media_id: int):
    """Invalidate all cache entries related to specific media"""
    # Ensure key pattern matches what get_cache_key generates
    # Assuming key format is like "cache:/api/v1/media/{media_id}:{hash}" or similar
    # This pattern might be too broad or too narrow depending on actual keys.
    # Adjust pattern if needed. Example: "cache:/api/v1/media/{media_id}*" ?
    # Using a more precise pattern if possible is better for performance.
    # Let's assume keys contain the media_id clearly.
    pattern = f"cache:*:{media_id}*" # Example pattern, adjust if needed
    try:
        # Note: KEYS can be slow on large Redis DBs. Consider alternatives like
        # storing related keys in a Set if performance becomes an issue.
        keys_to_delete = [key.decode('utf-8') for key in cache.keys(pattern)]
        if keys_to_delete:
            deleted_count = cache.delete(*keys_to_delete)
            logger.info(f"Invalidated {deleted_count} cache entries matching media ID {media_id} (pattern: '{pattern}')")
        else:
            logger.debug(f"No cache keys found to invalidate for media ID {media_id} (pattern: '{pattern}')")
    except redis.RedisError as e:
        logger.error(f"Redis error invalidating cache for media ID {media_id}: {e}")
    except Exception as e:
         logger.error(f"Unexpected error invalidating cache for media ID {media_id}: {e}")


##################################################################
#
# Bare Media Endpoint
#
# Endpoints:\
#     GET /api/v1/media - `"/"`
#     GET /api/v1/media/{media_id} - `"/{media_id}"`

# =============================================================================
# Dependency Function for Add Media Form Processing
# =============================================================================
def get_add_media_form(
    # Replicate ALL Form(...) fields from the endpoint signature
    media_type: MediaType = Form(..., description="Type of media (e.g., 'audio', 'video', 'pdf')"),
    urls: Optional[List[str]] = Form(None, description="List of URLs of the media items to add"),
    title: Optional[str] = Form(None, description="Optional title (applied if only one item processed)"),
    author: Optional[str] = Form(None, description="Optional author (applied similarly to title)"),
    keywords: str = Form("", description="Comma-separated keywords (applied to all processed items)"), # Receive as string
    custom_prompt: Optional[str] = Form(None, description="Optional custom prompt (applied to all)"),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt (applied to all)"),
    overwrite_existing: bool = Form(False, description="Overwrite existing media"),
    keep_original_file: bool = Form(False, description="Retain original uploaded files"),
    perform_analysis: bool = Form(True, description="Perform analysis (default=True)"),
    api_name: Optional[str] = Form(None, description="Optional API name"),
    api_key: Optional[str] = Form(None, description="Optional API key"), # Consider secure handling
    use_cookies: bool = Form(False, description="Use cookies for URL download requests"),
    cookies: Optional[str] = Form(None, description="Cookie string if `use_cookies` is True"),
    transcription_model: str = Form("deepdml/faster-distil-whisper-large-v3.5", description="Transcription model"),
    transcription_language: str = Form("en", description="Transcription language"),
    diarize: bool = Form(False, description="Enable speaker diarization"),
    timestamp_option: bool = Form(True, description="Include timestamps in transcription"),
    vad_use: bool = Form(False, description="Enable VAD filter"),
    perform_confabulation_check_of_analysis: bool = Form(False, description="Enable confabulation check"),
    start_time: Optional[str] = Form(None, description="Optional start time (HH:MM:SS or seconds)"),
    end_time: Optional[str] = Form(None, description="Optional end time (HH:MM:SS or seconds)"),
    pdf_parsing_engine: Optional[PdfEngine] = Form("pymupdf4llm", description="PDF parsing engine"),
    perform_chunking: bool = Form(True, description="Enable chunking"),
    chunk_method: Optional[ChunkMethod] = Form(None, description="Chunking method"),
    use_adaptive_chunking: bool = Form(False, description="Enable adaptive chunking"),
    use_multi_level_chunking: bool = Form(False, description="Enable multi-level chunking"),
    chunk_language: Optional[str] = Form(None, description="Chunking language override"),
    chunk_size: int = Form(500, description="Target chunk size"),
    chunk_overlap: int = Form(200, description="Chunk overlap size"),
    custom_chapter_pattern: Optional[str] = Form(None, description="Regex pattern for custom chapter splitting"),
    perform_rolling_summarization: bool = Form(False, description="Perform rolling summarization"),
    summarize_recursively: bool = Form(False, description="Perform recursive summarization"),
    # Don't need token here, it's a Header dep
    # Don't need files here, it's a File dep
    # Don't need db here, it's a separate Depends
) -> AddMediaForm:
    """
    Dependency function to parse form data for the /add endpoint
    and validate it against the AddMediaForm model.
    """
    try:
        # Create the Pydantic model instance using the parsed form data.
        # Pass the received Form(...) parameters to the model constructor
        form_instance = AddMediaForm(
            media_type=media_type,
            urls=urls,
            title=title,
            author=author,
            keywords=keywords, # Pydantic model handles alias mapping to keywords_str
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            overwrite_existing=overwrite_existing,
            keep_original_file=keep_original_file,
            perform_analysis=perform_analysis,
            start_time=start_time,
            end_time=end_time,
            api_name=api_name,
            api_key=api_key,
            use_cookies=use_cookies,
            cookies=cookies,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            diarize=diarize,
            timestamp_option=timestamp_option,
            vad_use=vad_use,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
            pdf_parsing_engine=pdf_parsing_engine,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            custom_chapter_pattern=custom_chapter_pattern,
            perform_rolling_summarization=perform_rolling_summarization,
            summarize_recursively=summarize_recursively,
        )
        return form_instance
    except ValidationError as e:
        # Reuse the detailed error handling from get_process_videos_form
        serializable_errors = []
        for error in e.errors():
             serializable_error = error.copy()
             if 'ctx' in serializable_error and isinstance(serializable_error.get('ctx'), dict):
                 new_ctx = {}
                 for k, v in serializable_error['ctx'].items():
                     if isinstance(v, Exception): new_ctx[k] = str(v)
                     else: new_ctx[k] = v
                 serializable_error['ctx'] = new_ctx
             serializable_errors.append(serializable_error)
        logger.warning(f"Pydantic validation failed for /add endpoint: {json.dumps(serializable_errors)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=serializable_errors,
        ) from e
    except Exception as e: # Catch other potential errors during instantiation
        logger.error(f"Unexpected error creating AddMediaForm: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during form processing: {type(e).__name__}"
        )

#Obtain details of a single media item using its ID
@router.get(
    "/{media_id}", # Endpoint for retrieving a specific item
    status_code=status.HTTP_200_OK,
    summary="Get Media Item Details",
    tags=["Media Management"],
    # response_model=MediaDetailResponse # Define a Pydantic model for this response if desired
)
async def get_media_item(
    media_id: int,
    db: Database = Depends(get_db_for_user)
):
    """
    **Retrieve Media Item by ID**

    Fetches the details for a specific *active* (non-deleted, non-trash) media item,
    including its associated keywords, its latest prompt/analysis, and document versions.
    """
    logger.debug(f"Attempting to fetch details for media_id: {media_id}")
    try:
        media_record_raw = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)

        if not media_record_raw:
            logger.warning(f"Media not found or not active for ID: {media_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found or is inactive/trashed")

        # Ensure media_record is a dictionary
        media_record = dict(media_record_raw)
        logger.debug(f"Found active media record for ID: {media_id}")

        keywords_list = fetch_keywords_for_media(media_id=media_id, db_instance=db)
        logger.debug(f"Fetched keywords for media ID {media_id}: {keywords_list}")

        latest_version_info = None
        prompt = None
        analysis = None
        try:
             latest_version_info = get_document_version(
                 db_instance=db,
                 media_id=media_id,
                 version_number=None, # Get latest
                 include_content=False
             )
             if latest_version_info:
                  prompt = latest_version_info.get('prompt')
                  analysis = latest_version_info.get('analysis_content')
                  logger.debug(f"Fetched latest version info (prompt/analysis) for media ID {media_id}")
             else:
                   logger.warning(f"No active document version found for media ID {media_id} to get latest prompt/analysis.")
        except Exception as dv_e:
            logger.error(f"Error fetching latest document version for media {media_id}: {dv_e}", exc_info=True)


        # --- FIX for Structured Content (Video/Audio) ---
        content_from_db = media_record.get('content', '')
        final_content_text = content_from_db
        final_metadata = {} # Default to empty dict

        if media_record.get('type') in ['video', 'audio']:
            try:
                # This assumes JSON part, then "\n\n", then text transcript.
                parts = content_from_db.split("\n\n", 1)
                if len(parts) == 2:
                    possible_json_str = parts[0]
                    remaining_text = parts[1]
                    try:
                        parsed_json_metadata = json.loads(possible_json_str)
                        if isinstance(parsed_json_metadata, dict):
                            final_metadata = parsed_json_metadata
                            final_content_text = remaining_text
                        else:
                            logger.warning(f"Parsed JSON metadata for media {media_id} is not a dict. Treating as text.")
                    except json.JSONDecodeError:
                        logger.warning(f"First part of content for media {media_id} is not valid JSON. Treating all as text.")
                # If no "\n\n", the whole content_from_db is treated as text
            except Exception as e_parse_meta:
                logger.error(f"Could not parse structured metadata from content for media {media_id}: {e_parse_meta}", exc_info=True)
        # --- END FIX for Structured Content ---

        word_count = len(final_content_text.split()) if final_content_text else 0

        # --- FIX for Document Versions List ---
        doc_versions_list = []
        # Only fetch versions if it's a 'document' type, or adjust logic as needed
        if media_record.get('type') == 'document':
            try:
                # Fetch active versions, without full content to keep the list light
                raw_versions = db.get_all_document_versions(media_id=media_id, include_content=False, include_deleted=False)
                for rv_row in raw_versions:
                    rv = dict(rv_row) # Convert row to dict
                    # Map to VersionDetailResponse fields
                    # Ensure datetime objects are handled correctly if not already strings
                    created_at_dt = rv.get("created_at")
                    if isinstance(created_at_dt, str):
                        try:
                            created_at_dt = datetime.fromisoformat(created_at_dt.replace('Z', '+00:00'))
                        except ValueError:
                            logger.warning(f"Could not parse created_at string '{created_at_dt}' for version {rv.get('version_number')}")
                            # Handle as per Pydantic model requirements, maybe skip or use a default
                            pass # Pydantic will handle it if it's not a valid datetime

                    doc_versions_list.append(
                        VersionDetailResponse(
                            media_id=rv.get("media_id"),
                            version_number=rv.get("version_number"),
                            created_at=created_at_dt, # Pass datetime object
                            prompt=rv.get("prompt"),
                            analysis_content=rv.get("analysis_content"),
                            content=None # Don't include content in the list
                        )
                    )
                logger.debug(f"Fetched {len(doc_versions_list)} document versions for media ID {media_id}")
            except Exception as e_vers:
                logger.error(f"Error fetching document versions for media {media_id}: {e_vers}", exc_info=True)

        response_data = {
            "media_id": media_id,
            "source": { # Corresponds to MediaSourceDetail model
                "url": media_record.get('url'),
                "title": media_record.get('title'),
                "duration": media_record.get('duration'), # Assuming 'duration' exists in Media table
                "type": media_record.get('type')
            },
            "processing": { # Corresponds to MediaProcessingDetail model
                "prompt": prompt,
                "analysis": analysis,
                "model": media_record.get('transcription_model'),
                "timestamp_option": media_record.get('timestamp_option') # Assuming 'timestamp_option' exists
            },
            "content": { # Corresponds to MediaContentDetail model
                "metadata": final_metadata,
                "text": final_content_text,
                "word_count": word_count
            },
            "keywords": keywords_list if keywords_list else [],
            "timestamps": media_record.get('timestamps', []), # Assuming 'timestamps' list exists in Media table or is parsed
            "versions": doc_versions_list, # Add the fetched versions
            # Add other top-level fields from MediaDetailResponse if they come directly from media_record
            # e.g., "uuid": media_record.get('uuid'),
            # "author": media_record.get('author'),
            # "ingestion_date": media_record.get('ingestion_date'),
            # "last_modified": media_record.get('last_modified'),
            # "version": media_record.get('version'), # Sync version
        }

        # To ensure the response matches MediaDetailResponse, instantiate it:
        try:
            return MediaDetailResponse(**response_data)
        except Exception as pydantic_err: # Catch Pydantic validation errors
            logger.error(f"Pydantic validation error for MediaDetailResponse for media {media_id}: {pydantic_err}", exc_info=True)
            # Log the data that failed validation
            logger.debug(f"Data causing Pydantic error: {response_data}")
            raise HTTPException(status_code=500, detail=f"Internal server error creating response for media item.")


    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error fetching details for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error retrieving media details: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching details for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred retrieving media details: {type(e).__name__}")
# async def get_media_item( # Changed to `async def` for consistency
#     media_id: int,
#     # --- Use the new DB dependency ---
#     db: Database = Depends(get_db_for_user) # Inject the Database instance
# ):
#     """
#     **Retrieve Media Item by ID**
#
#     Fetches the details for a specific *active* (non-deleted, non-trash) media item,
#     including its associated keywords and the prompt/analysis from its latest version.
#     """
#     logger.debug(f"Attempting to fetch details for media_id: {media_id}")
#     try:
#         # --- 1. Fetch the main Media record (active only) ---
#         # Use the instance method get_media_by_id, asking for non-deleted/non-trash
#         media_record = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
#
#         if not media_record:
#             logger.warning(f"Media not found or not active for ID: {media_id}")
#             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found or is inactive/trashed")
#
#         logger.debug(f"Found active media record for ID: {media_id}")
#         # Convert row object to dict if not already (depends on DB class setup)
#         # Assuming db.get_media_by_id returns a dict-like object or row factory handles it
#         if not isinstance(media_record, dict):
#              # Explicitly convert if needed, though db.get_media_by_id SHOULD return a dict
#              try:
#                   media_record = dict(media_record)
#              except TypeError:
#                    logger.error(f"Could not convert media_record to dict for ID {media_id}")
#                    raise HTTPException(status_code=500, detail="Internal server error processing media data format.")
#
#
#         # --- 2. Fetch Associated Keywords (active only) ---
#         # Use the standalone function from the new DB library
#         keywords_list = fetch_keywords_for_media(media_id=media_id, db_instance=db)
#         logger.debug(f"Fetched keywords for media ID {media_id}: {keywords_list}")
#
#         # --- 3. Fetch Latest Prompt & Analysis from DocumentVersions ---
#         latest_version_info = None
#         prompt = None
#         analysis = None
#         try:
#              # Use the standalone function, get latest by passing version_number=None
#              # Don't include full content to save bandwidth/processing
#              latest_version_info = get_document_version(
#                  db_instance=db,
#                  media_id=media_id,
#                  version_number=None, # Explicitly get latest
#                  include_content=False # We only need prompt/analysis from here
#              )
#              if latest_version_info:
#                   prompt = latest_version_info.get('prompt')
#                   analysis = latest_version_info.get('analysis_content')
#                   logger.debug(f"Fetched latest version info (prompt/analysis) for media ID {media_id}")
#              else:
#                    logger.warning(f"No active document version found for media ID {media_id}")
#
#         except DatabaseError as dv_e:
#             # Log error fetching version, but don't fail the whole request
#             logger.error(f"Database error fetching latest document version for media {media_id}: {dv_e}")
#         except Exception as dv_e:
#             logger.error(f"Unexpected error fetching latest document version for media {media_id}: {dv_e}", exc_info=True)
#
#         # --- 4. Prepare Response ---
#         # Extract data primarily from the main media_record
#         # NOTE: Metadata like 'duration', 'webpage_url', and detailed 'timestamps'
#         # are NOT standard fields in the new `Media` table schema provided.
#         # They would need to be added to the schema or stored differently (e.g., in content or a dedicated metadata field/table)
#         # if they are required. The response below reflects data *available* in the new schema.
#
#         content_text = media_record.get('content', '') # Main content/transcript
#         word_count = len(content_text.split()) if content_text else 0
#
#         # Reconstruct the response structure using available data
#         response_data = {
#             "media_id": media_id,
#             "uuid": media_record.get('uuid'), # Add UUID
#             "source": {
#                 "url": media_record.get('url'), # Original URL if available
#                 "title": media_record.get('title'),
#                 "duration": None, # <<< Not directly available in schema
#                 "type": media_record.get('type')
#             },
#             "processing": {
#                 "prompt": prompt, # From latest version
#                 "analysis": analysis, # From latest version
#                 "model": media_record.get('transcription_model'), # From Media table
#                 "timestamp_option": None # <<< Not directly available, unclear how determined
#             },
#             "content": {
#                 # 'metadata' dictionary is unclear how it would be populated now
#                 # If you stored JSON in a 'metadata' TEXT column, parse it here.
#                 "metadata": {}, # <<< Placeholder, needs clarification
#                 "text": content_text, # Main content from Media table
#                 "word_count": word_count
#             },
#             "keywords": keywords_list if keywords_list else [], # Use fetched list
#             "timestamps": [], # <<< Not directly available in schema/content format
#             "author": media_record.get('author'),
#             "ingestion_date": media_record.get('ingestion_date'),
#             "last_modified": media_record.get('last_modified'),
#             "version": media_record.get('version'), # Sync version
#         }
#
#         return response_data
#
#     except HTTPException: # Re-raise HTTP exceptions directly
#         raise
#     except DatabaseError as e: # Catch specific DB errors
#         logger.error(f"Database error fetching details for media {media_id}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Database error retrieving media details: {e}")
#     except Exception as e: # Catch other potential errors
#         logger.error(f"Unexpected error fetching details for media {media_id}: {e}", exc_info=True)
#         # Print traceback if needed during debugging:
#         # import traceback
#         # traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"An unexpected error occurred retrieving media details: {type(e).__name__}")

##############################################################################
############################## MEDIA Versioning ##############################
#
# Endpoints:
#   POST /api/v1/media/{media_id}/versions
#   GET /api/v1/media/{media_id}/versions
#   GET /api/v1/media/{media_id}/versions/{version_number}
#   DELETE /api/v1/media/{media_id}/versions/{version_number}
#   POST /api/v1/media/{media_id}/versions/rollback
#   PUT /api/v1/media/{media_id}

@router.post(
    "/{media_id}/versions",
    tags=["Media Versioning"],
    summary="Create Media Version",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, Any], # Update if you create a specific Pydantic model
)
async def create_version(
    media_id: int,
    request_body: VersionCreateRequest, # Renamed for clarity (vs request object)
    # --- Use the new DB dependency ---
    db: Database = Depends(get_db_for_user)
):
    """
    **Create a New Document Version**

    Creates a new version record for an existing *active* media item based on the
    provided content, prompt, and analysis.
    """
    logger.debug(f"Attempting to create version for media_id: {media_id}")
    try:
        # No explicit media check needed here if db.create_document_version handles it
        # (It checks for active parent Media ID internally)

        # Use the Database instance method within a transaction context
        # The method handles its own sync logging
        with db.transaction():
            result_dict = db.create_document_version(
                media_id=media_id,
                content=request_body.content,
                prompt=request_body.prompt,
                analysis_content=request_body.analysis_content,
            )

        # New method returns a dict with id, uuid, media_id, version_number
        logger.info(f"Successfully created version {result_dict.get('version_number')} (UUID: {result_dict.get('uuid')}) for media_id: {media_id}")

        # Return the useful info from the result
        return {
            "message": "Document version created successfully.",
            "media_id": result_dict.get("media_id"),
            "version_number": result_dict.get("version_number"),
            "version_uuid": result_dict.get("uuid")
        }

    except InputError as e: # Catch specific error if media_id not found/inactive
        logger.warning(f"Cannot create version for media {media_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (DatabaseError, ConflictError) as e: # Catch DB errors from new library
        logger.error(f"Database error creating version for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except HTTPException: # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating version for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during version creation")


@router.get(
    "/{media_id}/versions",
    tags=["Media Versioning"],
    summary="List Media Versions",
    response_model=List[Dict[str, Any]], # Update if you create a specific Pydantic model
)
async def list_versions(
    media_id: int,
    include_content: bool = Query(False, description="Include full content in response"),
    limit: int = Query(10, ge=1, le=100, description="Results per page"),
    page: int = Query(1, ge=1, description="Page number"), # Use page instead of offset
    # --- Use the new DB dependency ---
    db: Database = Depends(get_db_for_user)
):
    """
    **List Active Versions for an Active Media Item**

    Retrieves a paginated list of *active* versions (`deleted=0`) for a specific
    *active* media item (`deleted=0`, `is_trash=0`).
    Optionally includes the full content for each version. Ordered by version number descending.
    """
    logger.debug(f"Listing versions for media_id: {media_id} (Page: {page}, Limit: {limit}, Content: {include_content})")
    offset = (page - 1) * limit

    try:
        # Check if the parent media item is active first
        media_exists = check_media_exists(db_instance=db, media_id=media_id) # Uses standalone check
        if not media_exists:
             logger.warning(f"Cannot list versions: Media ID {media_id} not found or deleted.")
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media item not found or deleted")

        # --- Query active versions directly ---
        select_cols_list = ["dv.id", "dv.uuid", "dv.media_id", "dv.version_number", "dv.created_at",
                           "dv.prompt", "dv.analysis_content", "dv.last_modified", "dv.version"]
        if include_content: select_cols_list.append("dv.content")
        select_cols = ", ".join(select_cols_list)

        # Query Explanation:
        # - Select columns from DocumentVersions (dv)
        # - WHERE clause ensures:
        #   - Matching media_id
        #   - DocumentVersion is not deleted (dv.deleted = 0)
        # - ORDER BY version_number descending (latest first)
        # - LIMIT and OFFSET for pagination
        query = f"""
            SELECT {select_cols}
            FROM DocumentVersions dv
            WHERE dv.media_id = ? AND dv.deleted = 0
            ORDER BY dv.version_number DESC
            LIMIT ? OFFSET ?
        """
        params = (media_id, limit, offset)
        cursor = db.execute_query(query, params)
        versions = [dict(row) for row in cursor.fetchall()]

        # Optionally, add pagination info (total count) - requires another query
        count_cursor = db.execute_query("SELECT COUNT(*) FROM DocumentVersions WHERE media_id = ?", (media_id,))
        total_versions = count_cursor.fetchone()[0]

        return versions  #, total_versions

    except DatabaseError as e: # Catch DB errors from new library
        logger.error(f"Database error listing versions for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except HTTPException: # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error listing versions for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error listing versions")


@router.get(
    "/{media_id}/versions/{version_number}",
    tags=["Media Versioning"],
    summary="Get Specific Media Version",
    response_model=Dict[str, Any], # Update if you create a specific Pydantic model
)
async def get_version(
    media_id: int,
    version_number: int,
    include_content: bool = Query(True, description="Include full content in response"),
    # --- Use the new DB dependency ---
    db: Database = Depends(get_db_for_user)
):
    """
    **Get Specific Active Version Details**

    Retrieves the details of a single, specific *active* version (`deleted=0`)
    for an *active* media item (`deleted=0`, `is_trash=0`).
    """
    logger.debug(f"Getting version {version_number} for media_id: {media_id} (Content: {include_content})")
    try:
        # Use the standalone function from the new DB library.
        # It handles checking for active media and active version.
        version_dict = get_document_version(
            db_instance=db,
            media_id=media_id,
            version_number=version_number,
            include_content=include_content
        )

        if version_dict is None:
            # Function returns None if media inactive, version inactive, or version number doesn't exist
            logger.warning(f"Active version {version_number} not found for active media {media_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found or media/version is inactive")

        return version_dict

    except ValueError as e: # Catch invalid version_number from standalone function
        logger.warning(f"Invalid input for get_document_version: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DatabaseError as e: # Catch DB errors from new library
        logger.error(f"Database error getting version {version_number} for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except HTTPException: # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting version {version_number} for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error getting version")


@router.delete(
    "/{media_id}/versions/{version_number}",
    tags=["Media Versioning"],
    summary="Soft Delete Media Version", # Changed summary: Soft Delete
    status_code=status.HTTP_204_NO_CONTENT, # Keep 204 on success
)
async def delete_version(
    media_id: int,
    version_number: int,
    # --- Use the new DB dependency ---
    db: Database = Depends(get_db_for_user)
):
    """
    **Soft Delete a Specific Version**

    Marks a specific version of an active media item as deleted (`deleted=1`).
    Cannot delete the only remaining active version for a media item.
    This action is logged for synchronization but does not permanently remove data.
    """
    logger.debug(f"Attempting to soft delete version {version_number} for media_id: {media_id}")
    try:
        # 1. Find the UUID for the given media_id and version_number
        # Ensure both the target version and the parent media are active
        query_uuid = """
            SELECT dv.uuid
            FROM DocumentVersions dv
            JOIN Media m ON dv.media_id = m.id
            WHERE dv.media_id = ?
              AND dv.version_number = ?
              AND dv.deleted = 0
              AND m.deleted = 0
              AND m.is_trash = 0
        """
        cursor = db.execute_query(query_uuid, (media_id, version_number))
        result_uuid = cursor.fetchone()

        if not result_uuid:
            logger.warning(f"Active version {version_number} for active media {media_id} not found.")
            # Raise 404 whether media or version wasn't found/active
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active media or specific active version not found.")

        version_uuid = result_uuid['uuid']
        logger.debug(f"Found UUID {version_uuid} for version {version_number} of media {media_id}")

        # 2. Call the instance method using the UUID
        # The method handles sync logging and checks for 'last active version'.
        with db.transaction(): # Use transaction for consistency
             success = db.soft_delete_document_version(version_uuid=version_uuid)

        if success:
            # Return 204 No Content, FastAPI handles the response body
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        else:
            # soft_delete_document_version returns False if it was the last active version
            logger.warning(f"Failed to delete version {version_number} (UUID: {version_uuid}) for media {media_id} - likely the last active version.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete the only active version of the document.")

    except ConflictError as e: # Catch conflict during DB update
         logger.error(f"Conflict deleting version {version_number} (UUID: {version_uuid}) for media {media_id}: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Conflict during deletion: {e}")
    except InputError as e: # Catch invalid input errors from DB method
        logger.error(f"Input error deleting version {version_number} for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Input error: {e}")
    except DatabaseError as e: # Catch general DB errors
        logger.error(f"Database error deleting version {version_number} for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {e}")
    except HTTPException: # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting version {version_number} for media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting version")


@router.post(
    "/{media_id}/versions/rollback",
    tags=["Media Versioning"],
    summary="Rollback to Media Version",
    response_model=Dict[str, Any], # Update if you create a specific Pydantic model
)
async def rollback_version(
    media_id: int,
    request_body: VersionRollbackRequest, # Renamed for clarity
    # --- Use the new DB dependency ---
    db: Database = Depends(get_db_for_user)
):
    """
    **Rollback to a Previous Version**

    Restores the main content of an *active* media item to the state of a specified *active*
    previous version. Creates a *new* version reflecting the rolled-back content and
    updates the main Media record.
    """
    target_version_number = request_body.version_number
    logger.debug(f"Attempting to rollback media_id {media_id} to version {target_version_number}")
    try:
        # Use the Database instance method within a transaction context
        # The method handles checking media/version existence, 'cannot rollback to latest',
        # creating the new version, updating Media, and logging sync events.
        with db.transaction():
            rollback_result = db.rollback_to_version(
                media_id=media_id,
                target_version_number=target_version_number
            )

        # Check the result dictionary from the DB method
        if "error" in rollback_result:
            error_msg = rollback_result["error"]
            logger.warning(f"Rollback failed for media {media_id} to version {target_version_number}: {error_msg}")
            # Map specific errors to HTTP status codes
            if "not found" in error_msg.lower():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
            elif "Cannot rollback to the current latest version" in error_msg:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
            else: # Other rollback errors reported by DB function
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

        # Success case - DB method returns success details
        logger.info(f"Rollback successful for media {media_id} to version {target_version_number}. New doc version: {rollback_result.get('new_document_version_number')}")
        return {
            "message": rollback_result.get("success", "Rollback successful."),
            "media_id": media_id,
            "rolled_back_from_version": target_version_number,
            "new_document_version_number": rollback_result.get("new_document_version_number"),
            "new_document_version_uuid": rollback_result.get("new_document_version_uuid"),
            "new_media_version": rollback_result.get("new_media_version") # Sync version of Media record
        }

    except ValueError as e: # Catch invalid target_version_number
        logger.warning(f"Invalid input for rollback media {media_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConflictError as e: # Catch conflict during Media update
         logger.error(f"Conflict rolling back media {media_id} to version {target_version_number}: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Conflict during rollback: {e}")
    except (InputError, DatabaseError) as e: # Catch DB errors
        logger.error(f"Database error rolling back media {media_id} to version {target_version_number}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error during rollback: {e}")
    except HTTPException: # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error rolling back media {media_id} to version {target_version_number}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during rollback")


@router.put(
    "/{media_id}",
    tags=["Media Management"],
    summary="Update Media Item",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any], # Update if specific model created
)
async def update_media_item(
    media_id: int,
    payload: MediaUpdateRequest,
    # --- Use the new DB dependency ---
    db: Database = Depends(get_db_for_user)
):
    """
    **Update Media Item Details**

    Modifies attributes of an *active* main media item record (e.g., title, author).

    If **content** is updated (`payload.content` is not None):
      - A *new document version* is created using the provided `payload.content`.
      - The `payload.prompt` and `payload.analysis` (if provided) are stored in this *new* version.
      - The main `Media` record's `content`, `content_hash`, `last_modified`, and `version` (sync) are updated.
      - FTS index for the media item is updated.

    If only non-content fields (e.g., title, author) are updated:
      - Only the main `Media` record is updated (fields, `last_modified`, `version`).
      - No new document version is created.
      - FTS index is updated if `title` changed.
    """
    logger.debug(f"Received request to update media_id={media_id} with payload: {payload.model_dump(exclude_unset=True)}")

    # Prepare data for the update, excluding None values from payload
    # Use `exclude_unset=True` to only include fields explicitly set in the request
    update_fields = payload.model_dump(exclude_unset=True)

    # Check if any fields were actually provided for update
    if not update_fields:
        logger.info(f"Update request for media {media_id} received with no fields to update.")
        # Return 200 OK but indicate no changes were made
        # Fetch current data to return a representation? Or just a message?
        # Fetching current data seems appropriate for a PUT response.
        current_data = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
        if not current_data:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media item not found or inactive.")
        return {"message": "No update fields provided.", "media_item": current_data}


    new_doc_version_info: Optional[Dict] = None
    updated_media_info: Optional[Dict] = None
    message: str = ""

    try:
        # --- Use a single transaction for all potential DB operations ---
        with db.transaction() as conn: # Get connection for potential direct use if needed

            # --- 1. Get Current State (needed for hash comparison & version increment) ---
            cursor = conn.cursor()
            cursor.execute("SELECT id, uuid, content_hash, version FROM Media WHERE id = ? AND deleted = 0 AND is_trash = 0", (media_id,))
            current_media = cursor.fetchone()
            if not current_media:
                logger.warning(f"Update failed: Media not found or inactive/trashed for ID {media_id}")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media item not found or is inactive/trashed")

            current_hash = current_media['content_hash']
            current_sync_version = current_media['version']
            media_uuid = current_media['uuid']
            new_sync_version = current_sync_version + 1

            # --- 2. Check if Content is being Updated ---
            content_updated = 'content' in update_fields and update_fields['content'] is not None
            new_content = update_fields.get('content') if content_updated else None
            new_content_hash = hashlib.sha256(new_content.encode()).hexdigest() if content_updated else current_hash
            content_actually_changed = content_updated and (new_content_hash != current_hash)

            # --- 3. Prepare SQL SET clause and parameters ---
            set_parts = []
            params = []
            # Always update last_modified, version, and client_id on any change
            current_time = db._get_current_utc_timestamp_str() # Use internal helper
            client_id = db.client_id
            set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
            params.extend([current_time, new_sync_version, client_id])

            # Add specific fields from payload
            if 'title' in update_fields:
                set_parts.append("title = ?")
                params.append(update_fields['title'])
            if 'author' in update_fields:
                set_parts.append("author = ?")
                params.append(update_fields['author'])
            if 'type' in update_fields: # Allow updating type?
                set_parts.append("type = ?")
                params.append(update_fields['type'])
            # Add other updatable Media fields here...

            # Handle content change specifically
            if content_actually_changed:
                logger.info(f"Content changed for media {media_id}. Updating content and hash.")
                set_parts.extend(["content = ?", "content_hash = ?"])
                params.extend([new_content, new_content_hash])
                # Reset chunking status if content changes
                set_parts.append("chunking_status = ?")
                params.append('pending')
            elif content_updated and not content_actually_changed:
                 logger.info(f"Content provided for media {media_id} but hash is identical. Content field not updated.")
                 # Do not add content/hash to SET clause
                 # We still create a new version if content was in payload (as per original logic)

            # --- 4. Execute Media Table Update ---
            sql_set_clause = ", ".join(set_parts)
            update_query = f"UPDATE Media SET {sql_set_clause} WHERE id = ? AND version = ?"
            update_params = tuple(params + [media_id, current_sync_version])

            logger.debug(f"Executing Media UPDATE: {update_query} | Params: {update_params}")
            update_cursor = conn.cursor()
            update_cursor.execute(update_query, update_params)

            if update_cursor.rowcount == 0:
                # Check if it was a conflict or if the item disappeared
                cursor.execute("SELECT version FROM Media WHERE id = ?", (media_id,))
                check_conflict = cursor.fetchone()
                if check_conflict and check_conflict['version'] != current_sync_version:
                     raise ConflictError("Media", media_id)
                else: # Item disappeared between read and write? Unlikely in transaction but possible.
                      raise DatabaseError(f"Failed to update media {media_id}, possibly deleted concurrently.")

            logger.info(f"Successfully updated Media record for ID: {media_id}. New sync version: {new_sync_version}")
            message = f"Media item {media_id} updated successfully."

            # --- 5. Update FTS if title or content changed ---
            fts_title = update_fields.get('title', None) # Use new title if provided
            if fts_title is None: # If title not in payload, fetch current title for FTS
                cursor.execute("SELECT title FROM Media WHERE id = ?", (media_id,))
                fts_title = cursor.fetchone()['title']

            fts_content = new_content if content_actually_changed else None # Only pass content if it changed
            if fts_content is None and not content_updated: # If content didn't change and wasn't in payload, fetch current for FTS
                 cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
                 fts_content = cursor.fetchone()['content']

            if 'title' in update_fields or content_actually_changed:
                 logger.debug(f"Updating FTS for media {media_id} due to title/content change.")
                 db._update_fts_media(conn, media_id, fts_title, fts_content) # Use internal helper

            # --- 6. Create New Document Version if Content was in Payload ---
            if content_updated: # Create version even if content hash was identical (matches original logic)
                logger.info(f"Content was present in update payload for media {media_id}. Creating new document version.")
                # Use the payload content, prompt, and analysis
                # db.create_document_version handles its own sync logging internally
                new_doc_version_info = db.create_document_version(
                    media_id=media_id,
                    content=new_content, # Content from payload
                    prompt=payload.prompt, # Prompt from payload (can be None)
                    analysis_content=payload.analysis # Analysis from payload (can be None)
                )
                message += f" New version {new_doc_version_info.get('version_number')} created."
                logger.info(f"Created new version {new_doc_version_info.get('version_number')} (UUID: {new_doc_version_info.get('uuid')}) for media {media_id} during update.")

            # --- 7. Log Media Update Sync Event ---
            # Fetch the final state of the updated media record for the payload
            cursor.execute("SELECT * FROM Media WHERE id = ?", (media_id,))
            updated_media_info = dict(cursor.fetchone())
            if new_doc_version_info:
                 # Add context about the new version to the sync payload (optional)
                 updated_media_info['created_doc_ver_uuid'] = new_doc_version_info.get('uuid')
                 updated_media_info['created_doc_ver_num'] = new_doc_version_info.get('version_number')

            db._log_sync_event(conn, 'Media', media_uuid, 'update', new_sync_version, updated_media_info)

            # Commit happens automatically via context manager 'with db.transaction()'

        # --- 8. Prepare and Return Response ---
        response = {
            "message": message,
            "media_id": media_id,
            "media_uuid": media_uuid,
            "new_media_sync_version": new_sync_version,
            "new_document_version": { # Include details if version was created
                 "version_number": new_doc_version_info.get('version_number') if new_doc_version_info else None,
                 "uuid": new_doc_version_info.get('uuid') if new_doc_version_info else None,
            } if new_doc_version_info else None,
            # "updated_media": updated_media_info # Optionally return full updated object
        }
        return response

    except HTTPException: # Re-raise FastAPI/manual HTTP exceptions
        raise
    except ConflictError as e:
        logger.error(f"Conflict updating media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Conflict detected during update: {e}")
    except (DatabaseError, InputError) as e: # Catch DB errors from new library
        logger.error(f"Database/Input error updating media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error during update: {e}")
    except Exception as e:
        logger.error(f"Unexpected error updating media {media_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")


##############################################################################
############################## MEDIA Search ##################################
#
# Search Media Endpoints

# Endpoints:
#     GET /api/v1/media/search - `"/search"`

# Retrieve a listing of all media, returning a list of media items. Limited by paging and rate limiting.
@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="List All Media Items",
    tags=["Media Management"],
    response_model=MediaListResponse
)
@limiter.limit("50/minute")
async def list_all_media(
    request: Request, # Keep request for limiter
    page: int = Query(1, ge=1, description="Page number"),
    results_per_page: int = Query(10, ge=1, le=100, description="Results per page"),
    db: Database = Depends(get_db_for_user)
):
    """
    Retrieve a paginated listing of all active (non-deleted, non-trash) media items.
    Returns "items" and a "pagination" dictionary matching the MediaListResponse schema.
    """
    try:
        # Use the new Database method
        items_data, total_pages, current_page, total_items = db.get_paginated_media_list(
            page=page,
            results_per_page=results_per_page
        )

        formatted_items = [
            MediaListItem(
                id=item["id"],
                title=item["title"],
                type=item["type"],
                # UUID is now available from items_data if needed, but MediaListItem doesn't use it.
                # The URL can still be constructed here.
                url=f"/api/v1/media/{item['id']}"
            )
            for item in items_data # items_data is now a list of dicts
        ]

        pagination_info = PaginationInfo(
             page=current_page, # Use current_page returned from DB method
             results_per_page=results_per_page,
             total_pages=total_pages,
             total_items=total_items
         )

        try:
            response_obj = MediaListResponse(
                items=formatted_items,
                pagination=pagination_info
            )
            return response_obj
        except ValidationError as ve:
            logger.error(f"Pydantic validation error creating MediaListResponse: {ve.errors()}", exc_info=True) # Log Pydantic errors
            logger.debug(f"Data causing validation error: items_count={len(formatted_items)}, pagination={pagination_info.model_dump_json(indent=2) if pagination_info else 'None'}")
            raise HTTPException(status_code=500, detail="Internal server error: Response creation failed.")

    except ValueError as ve: # Catch ValueError from db.get_paginated_media_list
        logger.warning(f"Invalid pagination parameters for list_all_media: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except DatabaseError as e:
        logger.error(f"Database error fetching paginated media in list_all_media endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error retrieving media list.")
    except HTTPException: # Re-raise existing HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_all_media endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")


# FIXME - Add an 'advanced search' option for searching by date range, media type, etc. - update DB schema to add new fields
# ---------------------------
# Enhanced Search Endpoint with ETags
#

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

def parse_advanced_query(search_request: SearchRequest) -> Dict:
    """Convert advanced search request to DB query format"""
    query_params = {
        'search_query': search_request.query,
        'exact_phrase': search_request.exact_phrase,
        'filters': {
            'media_types': search_request.media_types,
            'date_range': search_request.date_range,
            'must_have': search_request.must_have,
            'must_not_have': search_request.must_not_have
        },
        'sort': search_request.sort_by,
        'boost': search_request.boost_fields or {'title': 2.0, 'content': 1.0}
    }
    return query_params

#
# End of Bare Media Endpoint Functions/Routes
#######################################################################


#######################################################################
#
# Pure Media Ingestion endpoint - for adding media to the DB with no analysis/modifications
#
# Endpoints:
#


# Per-User Media Ingestion and Analysis
# FIXME - Ensure that each function processes multiple files/URLs at once
class TempDirManager:
    def __init__(self, prefix: str = "media_processing_", *, cleanup: bool = True):
        self.temp_dir_path = None
        self.prefix = prefix
        self._cleanup = cleanup
        self._created = False

    def __enter__(self):
        self.temp_dir_path = Path(tempfile.mkdtemp(prefix=self.prefix))
        self._created = True
        logging.info(f"Created temporary directory: {self.temp_dir_path}")
        return self.temp_dir_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._created and self.temp_dir_path and self._cleanup:
            # remove the fragile exists-check and always try to clean up
            try:
                shutil.rmtree(self.temp_dir_path, ignore_errors=True)
                logging.info(f"Cleaned up temporary directory: {self.temp_dir_path}")
            except Exception as e:
                logging.error(f"Failed to cleanup temporary directory {self.temp_dir_path}: {e}",
                exc_info=True)
        self.temp_dir_path = None
        self._created = False

    def get_path(self):
         if not self._created:
              raise RuntimeError("Temporary directory not created or already cleaned up.")
         return self.temp_dir_path


def _validate_inputs(media_type: MediaType, urls: Optional[List[str]], files: Optional[List[UploadFile]]):
    """Validates initial media type and presence of input sources."""
    # media_type validation is handled by Pydantic's Literal type
    # Ensure at least one URL or file is provided
    if not urls and not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid media sources supplied. At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided."
        )


async def _save_uploaded_files(
    files: List[UploadFile],
    temp_dir: Path,
    validator: FileValidator,
    expected_media_type_key: Optional[str] = None,
    allowed_extensions: Optional[List[str]] = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """
    Saves uploaded files to a temporary directory, validating them.
    Requires a FileValidator instance.
    """
    """
    Saves uploaded files to a temporary directory, optionally filtering by extension.

    Args:
        :param files: List of UploadFile objects from FastAPI.
        :param temp_dir: The Path object representing the temporary directory to save files in.
        :param expected_media_type_key: An optional key to check against the file's media type.
        :param allowed_extensions: An optional list of allowed file extensions (e.g., ['.epub', '.pdf']).
                           Comparison is case-insensitive. If None, all files are attempted.

    Returns:
        A tuple containing:
        - processed_files: List of dicts for successfully saved files [{'path': Path, 'original_filename': str, 'input_ref': str}].
        - file_handling_errors: List of dicts for files that failed validation or saving [{'original_filename': str, 'input_ref': str, 'status': str, 'error': str}].
    """
    processed_files: List[Dict[str, Any]] = []
    file_handling_errors: List[Dict[str, Any]] = []
    # Keep track of filenames used within this batch in the temp dir to avoid collisions
    used_secure_names: Set[str] = set()

    # Normalize allowed extensions for case-insensitive comparison (if provided)
    normalized_allowed_extensions = {ext.lower().strip() for ext in allowed_extensions} if allowed_extensions else None
    logger.debug(f"Allowed extensions for upload: {normalized_allowed_extensions}")

    for file in files:
        # Use original filename if available, otherwise generate a ref
        # input_ref is primarily for logging/error correlation if filename is missing
        original_filename = file.filename
        input_ref = original_filename or f"upload_{uuid.uuid4()}"
        local_file_path: Optional[Path] = None # Track path for potential cleanup on error

        try:
            if not original_filename:
                logger.warning("Received file upload with no filename. Skipping.")
                file_handling_errors.append({
                    "original_filename": "N/A", # Indicate filename was missing
                    "input_ref": input_ref,
                    "status": "Error", # Use "Error" for consistency with other failures
                    "error": "File uploaded without a filename."
                })
                continue # Skip to the next file in the loop

            # --- Extension Validation ---
            file_extension = Path(original_filename).suffix.lower()
            if normalized_allowed_extensions and file_extension not in normalized_allowed_extensions:
                logger.warning(f"Skipping file '{original_filename}' due to disallowed extension '{file_extension}'. Allowed: {allowed_extensions}")
                file_handling_errors.append({
                    "original_filename": original_filename,
                    "input_ref": input_ref,
                    "status": "Error",
                    "error": f"Invalid file type ('{file_extension}'). Allowed extensions: {', '.join(allowed_extensions or [])}"
                })
                continue # Skip to the next file

            # --- Sanitize and Create Unique Filename ---
            original_stem = Path(original_filename).stem
            secure_base = sanitize_filename(original_stem) # Sanitize the base name

            # Construct filename and ensure uniqueness within the temp dir for this batch
            secure_filename = f"{secure_base}{file_extension}"
            counter = 0
            temp_path_to_check = temp_dir / secure_filename
            # Check against names already used *in this batch* and existing files (less likely but possible)
            while secure_filename in used_secure_names or temp_path_to_check.exists():
                counter += 1
                secure_filename = f"{secure_base}_{counter}{file_extension}"
                temp_path_to_check = temp_dir / secure_filename
                if counter > 100: # Safety break for edge cases
                    raise OSError(f"Could not generate unique filename for {original_filename} after {counter} attempts.")

            used_secure_names.add(secure_filename)
            local_file_path = temp_dir / secure_filename

            # --- Save File ---
            logger.info(f"Attempting to save uploaded file '{original_filename}' securely as: {local_file_path}")
            content = await file.read() # Read file content asynchronously

            # Check for empty file content after reading
            if not content:
                 logger.warning(f"Uploaded file '{original_filename}' is empty. Skipping save.")
                 file_handling_errors.append({
                     "original_filename": original_filename,
                     "input_ref": input_ref,
                     "status": "Error",
                     "error": "Uploaded file content is empty."
                 })
                 # Clean up zero-byte file if created by mistake (though 'wb' should handle it)
                 if local_file_path.exists(): local_file_path.unlink(missing_ok=True)
                 continue # Skip to the next file

            # Write content to the secure path
            with open(local_file_path, "wb") as buffer:
                buffer.write(content)

            file_size = local_file_path.stat().st_size
            logger.info(f"Successfully saved '{original_filename}' ({file_size} bytes) to {local_file_path}")

            # Add the necessary info for the endpoint to process the file
            processed_files.append({
                "path": local_file_path, # Return Path object
                "original_filename": original_filename, # Keep original name for reference
                "input_ref": input_ref # Consistent reference
            })

        except Exception as e:
            logger.error(f"Failed to save or validate uploaded file '{original_filename or input_ref}': {e}", exc_info=True)
            file_handling_errors.append({
                "original_filename": original_filename or "N/A",
                "input_ref": input_ref,
                "status": "Error",
                "error": f"Failed during upload processing: {type(e).__name__} - {e}"
            })
            # Attempt cleanup if file was partially created before the error
            if local_file_path and local_file_path.exists():
                try:
                    local_file_path.unlink(missing_ok=True) # missing_ok=True handles race conditions
                    logger.debug(f"Cleaned up partially saved/failed file: {local_file_path}")
                except OSError as unlink_err:
                    logger.warning(f"Failed to clean up partially saved/failed file {local_file_path}: {unlink_err}")
        finally:
            # Ensure the UploadFile is closed, releasing resources
            # FastAPI typically handles this, but explicit close is safer in manual processing loops
            await file.close()


    return processed_files, file_handling_errors


def _prepare_chunking_options_dict(form_data: AddMediaForm) -> Optional[Dict[str, Any]]:
    """Prepares the dictionary of chunking options based on form data."""
    if not form_data.perform_chunking:
        logging.info("Chunking disabled.")
        return None

    # Determine default chunk method based on media type if not specified
    default_chunk_method = 'sentences'
    if form_data.media_type == 'ebook':
        default_chunk_method = 'chapter'
        logging.info("Setting chunk method to 'chapter' for ebook type.")
    elif form_data.media_type in ['video', 'audio']:
        default_chunk_method = 'sentences' # Example default

    final_chunk_method = form_data.chunk_method or default_chunk_method

    # Override to 'chapter' if media_type is 'ebook', regardless of user input
    if form_data.media_type == 'ebook':
        final_chunk_method = 'chapter'

    chunk_options = {
        'method': final_chunk_method,
        'max_size': form_data.chunk_size,
        'overlap': form_data.chunk_overlap,
        'adaptive': form_data.use_adaptive_chunking,
        'multi_level': form_data.use_multi_level_chunking,
        # Use specific chunk language, fallback to transcription lang, else None
        'language': form_data.chunk_language or (form_data.transcription_language if form_data.media_type in ['audio', 'video'] else None),
        'custom_chapter_pattern': form_data.custom_chapter_pattern,
    }
    logging.info(f"Chunking enabled with options: {chunk_options}")
    return chunk_options

def _prepare_common_options(form_data: AddMediaForm, chunk_options: Optional[Dict]) -> Dict[str, Any]:
    """Prepares the dictionary of common processing options."""
    return {
        "keywords": form_data.keywords, # Use the parsed list from the model
        "custom_prompt": form_data.custom_prompt,
        "system_prompt": form_data.system_prompt,
        "overwrite_existing": form_data.overwrite_existing,
        "perform_analysis": form_data.perform_analysis,
        "chunk_options": chunk_options, # Pass the prepared dict
        "api_name": form_data.api_name,
        "api_key": form_data.api_key,
        "store_in_db": True, # Assume we always want to store for this endpoint
        "summarize_recursively": form_data.summarize_recursively,
        "author": form_data.author # Pass common author
    }

async def _process_batch_media(
    media_type: MediaType,
    urls: List[str],
    uploaded_file_paths: List[str],
    source_to_ref_map: Dict[str, Union[str, Tuple[str, str]]],
    form_data: AddMediaForm,
    chunk_options: Optional[Dict],
    loop: asyncio.AbstractEventLoop,
    db_path: str,
    client_id: str,
    temp_dir: Path # Pass temp_dir Path object
) -> List[Dict[str, Any]]:
    """
    Handles PRE-CHECK, external processing, and DB persistence for video/audio.
    """
    combined_results = []
    all_processing_sources = urls + uploaded_file_paths
    items_to_process = [] # Sources that pass pre-check or overwrite=True

    logger.debug(f"Starting pre-check for {len(all_processing_sources)} {media_type} items...")

    # --- 1. Pre-check ---
    for source_path_or_url in all_processing_sources:
        input_ref_info = source_to_ref_map.get(source_path_or_url)
        input_ref = input_ref_info[0] if isinstance(input_ref_info, tuple) else input_ref_info
        if not input_ref:
            logger.error(f"CRITICAL: Could not find original input reference for {source_path_or_url}.")
            input_ref = source_path_or_url

        identifier_for_check = input_ref # Use original URL/filename for DB check
        should_process = True
        existing_id = None
        reason = "Ready for processing."
        pre_check_warning = None

        # --- Perform DB pre-check only if overwrite is False AND for relevant types ---
        if not form_data.overwrite_existing and media_type in ['video', 'audio']:
            try:
                # --- Create a temporary DB instance JUST for the check ---
                # NOTE: This adds overhead. Consider if pre-check is strictly needed here.
                # If the check is vital, this ensures it uses the correct DB file.
                # FIXME
                # Alternatively, move the check inside the executor task later.
                # For now, let's instantiate temporarily for the check:
                temp_db_for_check = Database(db_path=db_path, client_id=client_id)
                model_for_check = form_data.transcription_model
                pre_check_query = """
                                  SELECT id \
                                  FROM Media
                                  WHERE url = ?
                                    AND transcription_model = ?
                                    AND is_trash = 0 \
                                  """
                cursor = temp_db_for_check.execute_query(pre_check_query, (identifier_for_check, model_for_check))
                existing_record = cursor.fetchone()
                temp_db_for_check.close_connection()  # Close the temporary connection
                # --- End temporary DB instance ---

                if existing_record:
                    existing_id = existing_record['id']
                    should_process = False
                    reason = f"Media exists (ID: {existing_id}) with the same URL/identifier and transcription model ('{model_for_check}'). Overwrite is False."
                else:
                    should_process = True # No matching item found
                    reason = "Media not found with this URL/identifier and transcription model."

            except (DatabaseError, sqlite3.Error) as check_err: # Catch specific DB errors
                logger.error(f"DB pre-check (custom query) failed for {identifier_for_check}: {check_err}", exc_info=True)
                should_process, existing_id, reason = True, None, f"DB pre-check failed: {check_err}"
                pre_check_warning = f"Database pre-check failed: {check_err}"
            except Exception as check_err: # Catch unexpected errors during check
                logger.error(f"Unexpected error during DB pre-check (custom query) for {identifier_for_check}: {check_err}", exc_info=True)
                should_process, existing_id, reason = True, None, f"Unexpected pre-check error: {check_err}"
                pre_check_warning = f"Unexpected database pre-check error: {check_err}"
        else:
             # Overwrite is True, so no need to check existence beforehand
             should_process = True
             reason = "Overwrite requested or not applicable, proceeding regardless of existence."

        # --- Skip Logic ---
        if not should_process: # This now correctly handles the overwrite=False case
            logger.info(f"Skipping processing for {input_ref}: {reason}")
            skipped_result = {
                "status": "Skipped", "input_ref": input_ref, "processing_source": source_path_or_url,
                "media_type": media_type, "message": reason, "db_id": existing_id,
                "metadata": {}, "content": None, "transcript": None, "segments": None, "chunks": None,
                "analysis": None, "summary": None, "analysis_details": None, "error": None, "warnings": None,
                "db_message": "Skipped processing, no DB action."
            }
            combined_results.append(skipped_result)
        else:
            items_to_process.append(source_path_or_url)
            log_msg = f"Proceeding with processing for {input_ref}: {reason}"
            if pre_check_warning:
                log_msg += f" (Pre-check Warning: {pre_check_warning})"
                # Store warning with ref
                source_to_ref_map[source_path_or_url] = (input_ref, pre_check_warning)
            logger.info(log_msg)


    # --- 2. Perform Batch Processing (External Library Call) ---
    if not items_to_process:
        logging.info("No items require processing after pre-checks.")
        return combined_results # Return only skipped items if any

    processing_output: Optional[Dict] = None # Result from process_videos / process_audio_files
    try:
        if media_type == 'video':
            # Import here or ensure it's available globally
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import process_videos
            video_args = {
                 "inputs": items_to_process,
                 "temp_dir": str(temp_dir), # <<< Pass the temp_dir path
                 "start_time": form_data.start_time, "end_time": form_data.end_time,
                 "diarize": form_data.diarize, "vad_use": form_data.vad_use,
                 "transcription_model": form_data.transcription_model,
                 "transcription_language": form_data.transcription_language,
                 "custom_prompt": form_data.custom_prompt, "system_prompt": form_data.system_prompt,
                 "perform_analysis": form_data.perform_analysis,
                 "perform_chunking": form_data.perform_chunking,
                 "chunk_method": chunk_options.get('method') if chunk_options else None,
                 "max_chunk_size": chunk_options.get('max_size') if chunk_options else 500,
                 "chunk_overlap": chunk_options.get('overlap') if chunk_options else 200,
                 "use_adaptive_chunking": chunk_options.get('adaptive', False) if chunk_options else False,
                 "use_multi_level_chunking": chunk_options.get('multi_level', False) if chunk_options else False,
                 "chunk_language": chunk_options.get('language') if chunk_options else None,
                 "summarize_recursively": form_data.summarize_recursively,
                 "api_name": form_data.api_name if form_data.perform_analysis else None,
                 "api_key": form_data.api_key,
                 "use_cookies": form_data.use_cookies, "cookies": form_data.cookies,
                 "timestamp_option": form_data.timestamp_option,
                 "perform_confabulation_check": form_data.perform_confabulation_check_of_analysis,
            }
            logging.debug(f"Calling external process_videos with args including temp_dir: {list(video_args.keys())}")
            target_func = functools.partial(process_videos, **video_args)
            processing_output = await loop.run_in_executor(None, target_func)

        elif media_type == 'audio':
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import process_audio_files
            audio_args = {
                 "inputs": items_to_process,
                 "temp_dir": str(temp_dir), # <<< Pass the temp_dir path
                 "transcription_model": form_data.transcription_model,
                 "transcription_language": form_data.transcription_language,
                 "perform_chunking": form_data.perform_chunking,
                 "chunk_method": chunk_options.get('method') if chunk_options else None,
                 "max_chunk_size": chunk_options.get('max_size') if chunk_options else 500,
                 "chunk_overlap": chunk_options.get('overlap') if chunk_options else 200,
                 "use_adaptive_chunking": chunk_options.get('adaptive', False) if chunk_options else False,
                 "use_multi_level_chunking": chunk_options.get('multi_level', False) if chunk_options else False,
                 "chunk_language": chunk_options.get('language') if chunk_options else None,
                 "diarize": form_data.diarize, "vad_use": form_data.vad_use, "timestamp_option": form_data.timestamp_option,
                 "perform_analysis": form_data.perform_analysis,
                 "api_name": form_data.api_name if form_data.perform_analysis else None,
                 "api_key": form_data.api_key,
                 "custom_prompt_input": form_data.custom_prompt, "system_prompt_input": form_data.system_prompt,
                 "summarize_recursively": form_data.summarize_recursively,
                 "use_cookies": form_data.use_cookies, "cookies": form_data.cookies,
                 "keep_original": form_data.keep_original_file,
                 "custom_title": form_data.title, "author": form_data.author,
                 # temp_dir: Managed by the caller endpoint
                 # NOTE: No DB argument passed to process_audio_files
            }
            logging.debug(f"Calling external process_audio_files with args including temp_dir: {list(audio_args.keys())}")
            target_func = functools.partial(process_audio_files, **audio_args)
            processing_output = await loop.run_in_executor(None, target_func)

        else:
             raise ValueError(f"Invalid media type '{media_type}' for batch processing.")

    except Exception as call_e:
        logging.error(f"Error calling external batch processor for {media_type}: {call_e}", exc_info=True)
        # Create error results for all items intended for processing
        failed_items_results = [
            {
                "status": "Error", "input_ref": source_to_ref_map.get(item, (item, None))[0], # Get ref from tuple/str
                "processing_source": item,
                "media_type": media_type, "error": f"Failed to call processor: {type(call_e).__name__}",
                "metadata": None, "content": None, "transcript": None, "segments": None, "chunks": None,
                "analysis": None, "summary": None, "analysis_details": None, "warnings": None, "db_id": None, "db_message": None
            } for item in items_to_process
        ]
        combined_results.extend(failed_items_results)
        return combined_results # Return early

    # --- 3. Process Results and Perform DB Interaction ---
    final_batch_results = []
    processing_results_list = [] # Individual results from the batch output

    # Extract the list of individual results from the batch processor's output
    if processing_output and isinstance(processing_output.get("results"), list):
        processing_results_list = processing_output["results"]
        if processing_output.get("errors_count", 0) > 0:
             logging.warning(f"Batch {media_type} processor reported errors: {processing_output.get('errors')}")
    else:
        logging.error(f"Batch {media_type} processor returned unexpected output format: {processing_output}")
        # Create error entries based on items_to_process
        processing_results_list = []
        for item in items_to_process:
            input_ref = source_to_ref_map.get(item, (item, None))[0] # Get ref
            processing_results_list.append({"input_ref": input_ref, "processing_source": item, "status": "Error", "error": f"Batch {media_type} processor returned invalid data or failed execution."})


    for process_result in processing_results_list:
        # Standardize: Ensure result is a dict and has necessary keys
        if not isinstance(process_result, dict):
            logging.error(f"Processor returned non-dict item: {process_result}")
            # Create a placeholder error result
            malformed_result = {
                 "status": "Error", "input_ref": "Unknown Input", "processing_source": "Unknown",
                 "media_type": media_type, "error": "Processor returned invalid result format.",
                 "metadata": None, "content": None, "transcript": None, "segments": None, "chunks": None,
                 "analysis": None, "summary": None, "analysis_details": None, "warnings": None, "db_id": None, "db_message": None
             }
            final_batch_results.append(malformed_result)
            continue

        # Determine input_ref (original URL/filename) and processing source
        input_ref = process_result.get("input_ref")
        processing_source = process_result.get("processing_source")
        if processing_source:
            # Use the processing_source (temp path or URL) to look up the original ref
            ref_info = source_to_ref_map.get(str(processing_source))  # Ensure key is string

            if isinstance(ref_info, tuple):
                original_input_ref = ref_info[0]  # Get the ref part from (ref, warning) tuple
            elif isinstance(ref_info, str):
                original_input_ref = ref_info  # It's just the ref string
            else:  # Lookup failed or ref_info is None/unexpected
                logger.warning(
                    f"Could not find original input reference in source_to_ref_map for processing_source: {processing_source}. Falling back.")
                # Fallback: Try using input_ref from result if present, else use processing_source itself
                original_input_ref = process_result.get("input_ref") or processing_source or "Unknown Input"
        else:
            # If processing_source is missing, try input_ref from result, fallback to Unknown
            original_input_ref = process_result.get("input_ref") or "Unknown Input (Missing Source)"
            logger.warning(
                f"Processing result missing 'processing_source'. Using fallback input_ref: {original_input_ref}")
            # Try to set processing_source if possible for consistency, though it's unknown
            # Make sure original_input_ref is a string before assigning
            process_result["processing_source"] = str(original_input_ref) if original_input_ref else "Unknown"

        # Store it in the dictionary that will be added to the final results list
        # Make sure original_input_ref is a string before assigning
        process_result["input_ref"] = str(original_input_ref) if original_input_ref else "Unknown"

        pre_check_info = source_to_ref_map.get(processing_source) if processing_source else None
        pre_check_warning_msg = None
        if isinstance(pre_check_info, tuple): pre_check_warning_msg = pre_check_info[1]
        if pre_check_warning_msg:
             process_result.setdefault("warnings", []).append(pre_check_warning_msg)


        # --- DB Interaction Logic ---
        db_id = None
        db_message = "DB interaction skipped (Processing failed or DB not provided)."

        # Perform DB add/update ONLY if processing succeeded
        # No need to check for db object here, we check db_path/client_id
        if db_path and client_id and process_result.get("status") in ["Success", "Warning"]:
            # Extract data needed for the database from the process_result dict
            # Use transcript as content for audio/video
            content_for_db = process_result.get('transcript', process_result.get('content'))
            analysis_for_db = process_result.get('summary', process_result.get('analysis'))
            metadata_for_db = process_result.get('metadata', {})
            analysis_details_for_db = process_result.get('analysis_details', {})
            # Use the model reported by the processor if available, else fallback to form data
            transcription_model_used = metadata_for_db.get('model', form_data.transcription_model) # Use metadata['model'] if present
            extracted_keywords = metadata_for_db.get('keywords', [])
            # Ensure keywords from form_data (which is a list) are combined correctly
            combined_keywords = set(form_data.keywords or []) # Use the list directly
            if isinstance(extracted_keywords, list):
                 combined_keywords.update(k.strip().lower() for k in extracted_keywords if k and k.strip())
            final_keywords_list = sorted(list(combined_keywords))
            title_for_db = metadata_for_db.get('title', form_data.title or (Path(str(original_input_ref)).stem if original_input_ref else 'Untitled')) # Use original_input_ref here
            author_for_db = metadata_for_db.get('author', form_data.author)

            if content_for_db:
                try:
                    logger.info(f"Attempting DB persistence for item: {input_ref}")
                    # --- FIX 2: Use lambda for run_in_executor ---
                    db_add_kwargs = dict(
                        url=str(original_input_ref),
                        title=title_for_db,
                        media_type=media_type,
                        content=content_for_db,
                        keywords=final_keywords_list,
                        prompt=form_data.custom_prompt,
                        analysis_content=analysis_for_db,
                        transcription_model=transcription_model_used,
                        author=author_for_db,
                        overwrite=form_data.overwrite_existing,
                        chunk_options=chunk_options,
                        segments=process_result.get('segments'),
                    )

                    # --- Function to run in executor ---
                    def _db_worker():
                        worker_db = None  # Initialize
                        try:
                            # --- Instantiate DB inside the worker ---
                            worker_db = Database(db_path=db_path, client_id=client_id)
                            # --- Call the INSTANCE method ---
                            return worker_db.add_media_with_keywords(**db_add_kwargs)
                        finally:
                            # --- Ensure connection is closed ---
                            if worker_db:
                                worker_db.close_connection()

                    # --------------------------------------

                    media_id_result, media_uuid_result, db_message_result = await loop.run_in_executor(
                        None, _db_worker  # Pass the worker function
                    )

                    db_id = media_id_result
                    media_uuid = media_uuid_result
                    db_message = db_message_result # Use message from DB method

                    process_result["db_id"] = db_id
                    process_result["db_message"] = db_message
                    process_result["media_uuid"] = media_uuid # Add UUID to result if useful

                    logger.info(f"DB persistence result for {original_input_ref}: ID={db_id}, UUID={media_uuid}, Msg='{db_message}'") # Log original ref

                except (DatabaseError, InputError, ConflictError) as db_err:  # Catch specific DB errors
                    logging.error(f"Database operation failed for {original_input_ref}: {db_err}", exc_info=True) # Log original ref
                    process_result['status'] = 'Warning'  # Downgrade to Warning if DB fails after successful processing
                    process_result['error'] = (process_result.get('error') or "") + f" | DB Error: {db_err}"
                    process_result.setdefault("warnings", []).append(f"Database operation failed: {db_err}")
                    process_result["db_message"] = f"DB Error: {db_err}"
                    process_result["db_id"] = None  # Ensure db_id is None on error
                    process_result["media_uuid"] = None

                except Exception as e:
                    logging.error(f"Unexpected error during DB persistence for {original_input_ref}: {e}", exc_info=True) # Log original ref
                    process_result['status'] = 'Warning'  # Downgrade to Warning
                    process_result['error'] = (process_result.get(
                        'error') or "") + f" | Persistence Error: {type(e).__name__}"
                    process_result.setdefault("warnings", []).append(f"Unexpected persistence error: {e}")
                    process_result["db_message"] = f"Persistence Error: {type(e).__name__}"
                    process_result["db_id"] = None  # Ensure db_id is None on error
                    process_result["media_uuid"] = None

            else:
                logging.warning(f"Skipping DB persistence for {original_input_ref} due to missing content.") # Log original ref
                process_result["db_message"] = "DB persistence skipped (no content)."
                process_result["db_id"] = None  # Ensure db_id is None
                process_result["media_uuid"] = None

        # Add the (potentially updated) result to the final list
        final_batch_results.append(process_result)

    # Combine skipped results with processed results
    combined_results.extend(final_batch_results)

    # --- 4. Final Standardization ---
    final_standardized_results = []
    processed_input_refs = set() # Track to avoid duplicates

    for res in combined_results:
        input_ref = res.get("input_ref", "Unknown")
        if input_ref in processed_input_refs and input_ref != "Unknown":
            continue
        processed_input_refs.add(input_ref)

        # Ensure standard fields exist
        standardized = {
            "status": res.get("status", "Error"),
            "input_ref": input_ref,
            "processing_source": res.get("processing_source", "Unknown"),
            "media_type": res.get("media_type", media_type),
            "metadata": res.get("metadata", {}),
            "content": res.get("content", res.get("transcript")),
            "transcript": res.get("transcript"),
            "segments": res.get("segments"),
            "chunks": res.get("chunks"),
            "analysis": res.get("analysis", res.get("summary")),
            "summary": res.get("summary"),
            "analysis_details": res.get("analysis_details"),
            "error": res.get("error"),
            "warnings": res.get("warnings"),
            "db_id": res.get("db_id"),
            "db_message": res.get("db_message"),
            "message": res.get("message"),
            "media_uuid": res.get("media_uuid"),
        }
        # Ensure warnings list is None if empty
        if isinstance(standardized.get("warnings"), list) and not standardized["warnings"]:
            standardized["warnings"] = None

        final_standardized_results.append(standardized)

    return final_standardized_results


async def _process_document_like_item(
    item_input_ref: str,
    processing_source: str, # URL or upload path string
    media_type: MediaType,
    is_url: bool,
    form_data: AddMediaForm,
    chunk_options: Optional[Dict],
    temp_dir: Path, # Use Path object
    loop: asyncio.AbstractEventLoop,
    db_path: str,
    client_id: str
) -> Dict[str, Any]:
    """
    Handles PRE-CHECK, download/prep, processing, and DB persistence for document-like items.
    """
    # Initialize result structure (including DB fields)
    final_result = {
        "status": "Pending", "input_ref": item_input_ref, "processing_source": processing_source,
        "media_type": media_type, "metadata": {}, "content": None, "segments": None,
        "chunks": None, "analysis": None, "summary": None, "analysis_details": None, "error": None,
        "warnings": [], # <<< Initialize warnings as list
        "db_id": None, "db_message": None, "message": None
    }

    # --- 1. Pre-check ---
    # identifier_for_check = item_input_ref
    # existing_id = None
    # pre_check_warning = None
    # # Perform DB pre-check only if overwrite is False
    # if not form_data.overwrite_existing:
    #     try:
    #         # Use run_in_executor as check_media_exists is likely sync
    #         check_func = functools.partial(check_media_exists, db_instance=db, url=identifier_for_check)
    #         existing_id = await loop.run_in_executor(None, check_func)
    #         if existing_id is not None:
    #              logger.info(f"Skipping processing for {item_input_ref}: Media exists (ID: {existing_id}) and overwrite=False.")
    #              final_result.update({
    #                  "status": "Skipped", "message": f"Media exists (ID: {existing_id}), overwrite=False",
    #                  "db_id": existing_id, "db_message": "Skipped - Exists in DB."
    #              })
    #              # Clean up warnings if empty before returning
    #              if not final_result.get("warnings"): final_result["warnings"] = None
    #              return final_result
    #     except (DatabaseError, sqlite3.Error) as check_err:
    #         logger.error(f"Database pre-check failed for {item_input_ref}: {check_err}", exc_info=True)
    #         # Don't fail, just add a warning and proceed
    #         final_result.setdefault("warnings", []).append(f"Database pre-check failed: {check_err}")
    #     except Exception as check_err:
    #         logger.error(f"Unexpected error during DB pre-check for {item_input_ref}: {check_err}", exc_info=True)
    #         final_result.setdefault("warnings", []).append(f"Unexpected database pre-check error: {check_err}")

    # --- 2. Download/Prepare File ---
    file_bytes: Optional[bytes] = None
    processing_filepath: Optional[Path] = None # Use Path object
    processing_filename: Optional[str] = None
    try:
        if is_url:
            logger.info(f"Downloading URL: {processing_source}")
            download_func = functools.partial(smart_download, processing_source, temp_dir) # Pass url, temp_dir positionally
            downloaded_path = await loop.run_in_executor(None, download_func)
            if downloaded_path and isinstance(downloaded_path, Path) and downloaded_path.exists():
                 processing_filepath = downloaded_path
                 processing_filename = downloaded_path.name
                 if media_type == 'pdf':
                     # Use aiofiles directly here since we have the path
                     async with aiofiles.open(processing_filepath, "rb") as f:
                         file_bytes = await f.read()
                 # Update source to the actual path used for processing
                 final_result["processing_source"] = str(processing_filepath) # Keep track of the temp file path
            else:
                 raise IOError(f"Download failed or did not return a valid path for {processing_source}")

        else: # It's an uploaded file path string
            path_obj = Path(processing_source)
            if not path_obj.is_file(): # More specific check
                 raise FileNotFoundError(f"Uploaded file path not found or is not a file: {processing_source}")
            processing_filepath = path_obj
            processing_filename = path_obj.name
            if media_type == 'pdf':
                 async with aiofiles.open(processing_filepath, "rb") as f: # Use processing_filepath here
                      file_bytes = await f.read()
            # processing_source is already the path string
            final_result["processing_source"] = processing_source

    except (httpx.HTTPStatusError, httpx.RequestError, IOError, OSError, FileNotFoundError) as prep_err:
         logging.error(f"File preparation/download error for {item_input_ref}: {prep_err}", exc_info=True)
         final_result.update({"status": "Error", "error": f"File preparation/download failed: {prep_err}"})
         # Clean up warnings list if empty
         if not final_result.get("warnings"): final_result["warnings"] = None
         return final_result


    # --- 3. Select and Call Refactored Processing Function ---
    process_result_dict: Optional[Dict[str, Any]] = None
    try:
        processing_func: Optional[Callable] = None
        common_args = {
            "title_override": form_data.title,
            "author_override": form_data.author,
            "keywords": form_data.keywords, # Pass the list
            "perform_chunking": form_data.perform_chunking,
            "chunk_options": chunk_options,
            "perform_analysis": form_data.perform_analysis,
            # --- FIX: Pass these arguments ---
            "api_name": form_data.api_name,
            "api_key": form_data.api_key,
            "custom_prompt": form_data.custom_prompt,
            "system_prompt": form_data.system_prompt,
            # ----------------------------------
            "summarize_recursively": form_data.summarize_recursively,
        }
        specific_args = {}
        run_in_executor = True # Default for sync library functions

        if media_type == 'pdf':
             # --- FIX: Check file_bytes which were read earlier ---
             if file_bytes is None: raise ValueError("PDF processing requires file bytes, but they were not read.")
             processing_func = process_pdf_task # Use the async task wrapper
             run_in_executor = False # Task is already async
             specific_args = {
                 "file_bytes": file_bytes,
                 "filename": processing_filename or item_input_ref,
                 "parser": str(form_data.pdf_parsing_engine) or "pymupdf4llm",
                 # Pass individual chunk params expected by process_pdf_task
                 "chunk_method": chunk_options.get('method') if chunk_options else None,
                 "max_chunk_size": chunk_options.get('max_size') if chunk_options else None,
                 "chunk_overlap": chunk_options.get('overlap') if chunk_options else None,
                 # Keep common args like api_name, api_key etc for analysis within process_pdf_task
             }
             # Remove chunk_options dict if passing individually to avoid confusion
             common_args.pop("chunk_options", None)


        elif media_type == "document":
             if not processing_filepath: raise ValueError("Document processing requires a file path.")
             processing_func = process_document_content
             specific_args = {"doc_path": processing_filepath} # Pass Path object
             # --- FIX: Ensure process_document_content receives all its required args ---
             # Common args already contain api_name, api_key, prompts etc.

        elif media_type == "ebook":
             if not processing_filepath: raise ValueError("Ebook processing requires a file path.")
             # Need a wrapper if process_epub is sync
             def _sync_process_ebook_wrapper(**kwargs):
                 return process_epub(**kwargs)
             processing_func = _sync_process_ebook_wrapper
             specific_args = {
                 "file_path": str(processing_filepath),
                 "extraction_method": 'filtered', # Get from form_data if available
                 # Pass other ebook specific args if needed
             }
             # Add custom chapter pattern if provided in form_data
             if form_data.custom_chapter_pattern:
                 specific_args["custom_chapter_pattern"] = form_data.custom_chapter_pattern
             # Ensure all necessary args from common_args are passed if needed by process_epub

        else:
             raise NotImplementedError(f"Processor not implemented for media type: '{media_type}'")

        # Combine common and specific args, overriding common with specific if keys clash
        all_args = {**common_args, **specific_args}
        # Remove None values ONLY if the target function cannot handle them
        # Usually better to let the function handle defaults
        # final_args = {k: v for k, v in all_args.items() if v is not None}
        final_args = all_args # Pass all prepared args

        # --- Execute Processing ---
        if processing_func:
            func_name = getattr(processing_func, "__name__", str(processing_func))
            logging.info(f"Calling refactored '{func_name}' for '{item_input_ref}' {'in executor' if run_in_executor else 'directly'}")
            if run_in_executor:
                # Use run_in_executor for synchronous functions
                target_func = functools.partial(processing_func, **final_args)
                process_result_dict = await loop.run_in_executor(None, target_func)
            else: # For async functions like process_pdf_task
                process_result_dict = await processing_func(**final_args)

            if not isinstance(process_result_dict, dict):
                raise TypeError(f"Processor '{func_name}' returned non-dict: {type(process_result_dict)}")

            # Merge the result from the processing function into our final_result
            final_result.update(process_result_dict)
            final_result["status"] = process_result_dict.get("status", "Error" if process_result_dict.get("error") else "Success")
            proc_warnings = process_result_dict.get("warnings")
            if isinstance(proc_warnings, list):
                 # Ensure warnings list exists before extending
                 if not isinstance(final_result.get("warnings"), list): final_result["warnings"] = []
                 final_result["warnings"].extend(proc_warnings)
            elif proc_warnings: # If it's a single string warning
                 if not isinstance(final_result.get("warnings"), list): final_result["warnings"] = []
                 final_result["warnings"].append(str(proc_warnings))


        else: # Should not happen
            final_result.update({"status": "Error", "error": "No processing function selected."})

    except Exception as proc_err:
        logging.error(f"Error during processing call for {item_input_ref}: {proc_err}", exc_info=True)
        final_result.update({"status": "Error", "error": f"Processing error: {type(proc_err).__name__}: {proc_err}"})

    # Ensure essential fields are always present after processing attempt
    final_result.setdefault("status", "Error")
    final_result["input_ref"] = item_input_ref # Already set
    final_result["media_type"] = media_type # Already set

    # --- 4. Post-Processing DB Logic ---
    # Only attempt if processing status is Success or Warning
    if final_result.get("status") in ["Success", "Warning"]:
        content_for_db = final_result.get('content', '')
        analysis_for_db = final_result.get('summary') or final_result.get('analysis')
        metadata_for_db = final_result.get('metadata', {})
        # Use parsed keywords list from form_data, combined with any extracted
        extracted_keywords = final_result.get('keywords', [])
        combined_keywords = set(form_data.keywords or []) # Use list from form
        if isinstance(extracted_keywords, list):
            combined_keywords.update(k.strip().lower() for k in extracted_keywords if k and k.strip())
        final_keywords_list = sorted(list(combined_keywords))

        model_used = metadata_for_db.get('parser_used', 'Imported') # Check metadata first
        if not model_used and media_type == 'pdf': model_used = final_result.get('analysis_details', {}).get('parser', 'Imported')
        title_for_db = metadata_for_db.get('title', form_data.title or (Path(item_input_ref).stem if item_input_ref else 'Untitled'))
        author_for_db = metadata_for_db.get('author', form_data.author or 'Unknown')


        if content_for_db:
            try:
                logger.info(f"Attempting DB persistence for item: {item_input_ref} using user DB")
                db_add_kwargs = dict(
                    url=item_input_ref, title=title_for_db, media_type=media_type,
                    content=content_for_db, keywords=final_keywords_list,
                    prompt=form_data.custom_prompt, analysis_content=analysis_for_db,
                    transcription_model=model_used, author=author_for_db,
                    overwrite=form_data.overwrite_existing, chunk_options=chunk_options,
                    segments=None
                )

                # --- Function to run in executor ---
                def _db_worker():
                    worker_db = None
                    try:
                        # --- Instantiate DB inside the worker ---
                        worker_db = Database(db_path=db_path, client_id=client_id)
                        # --- Call the INSTANCE method ---
                        return worker_db.add_media_with_keywords(**db_add_kwargs)
                    finally:
                        if worker_db:
                            worker_db.close_connection()
                # --------------------------------------

                media_id_result, media_uuid_result, db_message_result = await loop.run_in_executor(
                    None, _db_worker # Pass the worker function
                )

                final_result["db_id"] = media_id_result
                final_result["db_message"] = db_message_result
                final_result["media_uuid"] = media_uuid_result # Add UUID
                logger.info(f"DB persistence result for {item_input_ref}: ID={media_id_result}, UUID={media_uuid_result}, Msg='{db_message_result}'")

            except (DatabaseError, InputError, ConflictError) as db_err:
                 logger.error(f"Database operation failed for {item_input_ref}: {db_err}", exc_info=True)
                 final_result['status'] = 'Warning' # Keep Warning status
                 final_result['error'] = (final_result.get('error') or "") + f" | DB Error: {db_err}"
                 # Ensure warnings list exists before appending
                 if not isinstance(final_result.get("warnings"), list):
                     final_result["warnings"] = []
                 final_result["warnings"].append(f"Database operation failed: {db_err}")
                 final_result["db_message"] = f"DB Error: {db_err}"
                 final_result["db_id"] = None # Ensure None on error
                 final_result["media_uuid"] = None
            except Exception as e:
                 logger.error(f"Unexpected error during DB persistence for {item_input_ref}: {e}", exc_info=True)
                 final_result['status'] = 'Warning' # Keep Warning status
                 final_result['error'] = (final_result.get('error') or "") + f" | Persistence Error: {type(e).__name__}"
                 # Ensure warnings list exists before appending
                 if not isinstance(final_result.get("warnings"), list):
                     final_result["warnings"] = []
                 final_result["warnings"].append(f"Unexpected persistence error: {e}")
                 final_result["db_message"] = f"Persistence Error: {type(e).__name__}"
                 final_result["db_id"] = None # Ensure None on error
                 final_result["media_uuid"] = None
        else:
             logger.warning(f"Skipping DB persistence for {item_input_ref} due to missing content.")
             final_result["db_message"] = "DB persistence skipped (no content)."
             final_result["db_id"] = None # Ensure None
             final_result["media_uuid"] = None
    else:
        # If processing failed, set DB message accordingly
        final_result["db_message"] = "DB operation skipped (processing failed)."
        final_result["db_id"] = None # Ensure None
        final_result["media_uuid"] = None


    # Clean up warnings if empty list
    if not final_result.get("warnings"):
         final_result["warnings"] = None

    # Standardize output keys (map content to content/transcript)
    final_result["content"] = final_result.get("content")
    final_result["transcript"] = final_result.get("content") # For consistency with A/V
    final_result["analysis"] = final_result.get("analysis") # For consistency

    return final_result


def _determine_final_status(results: List[Dict[str, Any]]) -> int:
    """Determines the overall HTTP status code based on individual results."""
    if not results:
        # This case should ideally be handled earlier if no inputs were valid
        return status.HTTP_400_BAD_REQUEST

    # Consider only results from actual processing attempts (exclude file saving errors if desired)
    # processing_results = [r for r in results if "Failed to save uploaded file" not in r.get("error", "")]
    processing_results = results # Or consider all results

    if not processing_results:
        return status.HTTP_200_OK # Or 207 if file saving errors occurred but no processing started

    if all(r.get("status", "").lower() == "success" for r in processing_results):
        return status.HTTP_200_OK
    else:
        # If any result is not "Success", return 207 Multi-Status
        return status.HTTP_207_MULTI_STATUS


# --- Main Endpoint ---
@router.post("/add",
             # status_code=status.HTTP_200_OK, # Determined dynamically
             dependencies=[Depends(get_db_for_user)],
             summary="Add media (URLs/files) with processing and persistence",
             tags=["Media Ingestion & Persistence"], # Changed tag
             )
async def add_media(
    background_tasks: BackgroundTasks,
    # # --- Required Fields ---
    # #media_type: MediaType = Form(..., description="Type of media (e.g., 'audio', 'video', 'pdf')"),
    # # --- Input Sources (Validation needed in code) ---
    # urls: Optional[List[str]] = Form(None, description="List of URLs of the media items to add"),
    # # --- Common Optional Fields ---
    # title: Optional[str] = Form(None, description="Optional title (applied if only one item processed)"),
    # author: Optional[str] = Form(None, description="Optional author (applied similarly to title)"),
    # keywords: str = Form("", description="Comma-separated keywords (applied to all processed items)"), # Receive as string
    # custom_prompt: Optional[str] = Form(None, description="Optional custom prompt (applied to all)"),
    # system_prompt: Optional[str] = Form(None, description="Optional system prompt (applied to all)"),
    # overwrite_existing: bool = Form(False, description="Overwrite existing media"),
    # keep_original_file: bool = Form(False, description="Retain original uploaded files"),
    # perform_analysis: bool = Form(True, description="Perform analysis (default=True)"),
    # # --- Integration Options ---
    # api_name: Optional[str] = Form(None, description="Optional API name"),
    # api_key: Optional[str] = Form(None, description="Optional API key"), # Consider secure handling
    # use_cookies: bool = Form(False, description="Use cookies for URL download requests"),
    # cookies: Optional[str] = Form(None, description="Cookie string if `use_cookies` is True"),
    # # --- Audio/Video Specific ---
    # transcription_model: str = Form("deepdml/faster-distil-whisper-large-v3.5", description="Transcription model"),
    # transcription_language: str = Form("en", description="Transcription language"),
    # diarize: bool = Form(False, description="Enable speaker diarization"),
    # timestamp_option: bool = Form(True, description="Include timestamps in transcription"),
    # vad_use: bool = Form(False, description="Enable VAD filter"),
    # perform_confabulation_check_of_analysis: bool = Form(False, description="Enable confabulation check"),
    # start_time: Optional[str] = Form(None, description="Optional start time (HH:MM:SS or seconds)"),
    # end_time: Optional[str] = Form(None, description="Optional end time (HH:MM:SS or seconds)"),
    # # --- PDF Specific ---
    # pdf_parsing_engine: Optional[PdfEngine] = Form("pymupdf4llm", description="PDF parsing engine"),
    # # --- Chunking Specific ---
    # perform_chunking: bool = Form(True, description="Enable chunking"),
    # chunk_method: Optional[ChunkMethod] = Form(None, description="Chunking method"),
    # use_adaptive_chunking: bool = Form(False, description="Enable adaptive chunking"),
    # use_multi_level_chunking: bool = Form(False, description="Enable multi-level chunking"),
    # chunk_language: Optional[str] = Form(None, description="Chunking language override"),
    # chunk_size: int = Form(500, description="Target chunk size"),
    # chunk_overlap: int = Form(200, description="Chunk overlap size"),
    # custom_chapter_pattern: Optional[str] = Form(None, description="Regex pattern for custom chapter splitting"),
    # # --- Deprecated/Less Common ---
    # perform_rolling_summarization: bool = Form(False, description="Perform rolling summarization"),
    # summarize_recursively: bool = Form(False, description="Perform recursive summarization"),
    # # --- Use Dependency Injection for Form Data ---
    form_data: AddMediaForm = Depends(get_add_media_form),
    # --- Keep File and Header Dependencies Separate ---
    token: str = Header(..., description="Authentication token"), # Placeholder for auth
    files: Optional[List[UploadFile]] = File(None, description="List of files to upload"),
    # --- DB Dependency ---
    db: Database = Depends(get_db_for_user) # Use the correct dependency
):
    """
    **Add Media Endpoint**

    Add multiple media items (from URLs and/or uploaded files) to the database with processing.

    Ingests media from URLs or uploads, processes it (transcription, analysis, etc.),
    and **persists** the results and metadata to the database.

    Use this endpoint for adding new content to the system permanently.
    """
    # --- 1. Validation (Now handled by get_add_media_form dependency) ---
    # Basic check for presence of inputs still useful here
    _validate_inputs(form_data.media_type, form_data.urls, files)
    logger.info(f"Received request to add {form_data.media_type} media.")
    # TODO: Implement actual authentication logic using the 'token' if needed

    # --- 2. Database Dependency (Handled by `db` parameter) ---
    # Ensure client_id is available (should be set by db dependency logic)
    if not hasattr(db, 'client_id') or not db.client_id:
        logger.error("CRITICAL: Database instance dependency missing client_id.")
        # Attempt to set it from settings as a fallback, but log error
        db.client_id = settings.get("SERVER_CLIENT_ID", "SERVER_API_V1_FALLBACK")
        logger.warning(f"Manually set missing client_id on DB instance to: {db.client_id}")
        # Consider raising 500 if client_id is absolutely essential and shouldn't be missing
        # raise HTTPException(status_code=500, detail="Internal server error: DB configuration issue.")

    # --- 2. Database Dependency ---
    # The line : `db = Depends(get_db)` in the func args takes care of this

    results = []
    # --- Use TempDirManager ---
    temp_dir_manager = TempDirManager(cleanup=not form_data.keep_original_file)
    temp_dir_path: Optional[Path] = None
    loop = asyncio.get_running_loop()

    try:
        # --- 3. Setup Temporary Directory ---
        with temp_dir_manager as temp_dir: # Context manager handles creation/cleanup
            temp_dir_path = temp_dir # Store the path for potential use
            logger.info(f"Using temporary directory: {temp_dir_path}")

            # --- 4. Save Uploaded Files ---
            saved_files_info, file_save_errors = await _save_uploaded_files(files or [], temp_dir_path, validator=file_validator_instance,)
            # Adapt file saving errors to the standard result format
            for err_info in file_save_errors:
                 results.append({
                      "status": "Error",
                      "input_ref": err_info.get("input_ref", "Unknown Upload"),
                      "processing_source": None, # No processing source if save failed
                      "media_type": form_data.media_type, # Assume intended type
                      "metadata": {}, "content": None, "transcript": None, "segments": None,
                      "chunks": None, "analysis": None, "summary": None,
                      "analysis_details": None, "error": err_info.get("error", "File save failed."),
                      "warnings": None, "db_id": None, "db_message": "File saving failed.",
                      "message": "File saving failed."
                  })


            # --- 5. Prepare Inputs and Options ---
            uploaded_file_paths = [str(pf["path"]) for pf in saved_files_info]
            url_list = form_data.urls or []
            all_valid_input_sources = url_list + uploaded_file_paths # Only those that saved/downloaded

            # Check if any valid sources remain after potential save errors
            if not all_valid_input_sources:
                 if file_save_errors:
                      logger.warning("No valid inputs remaining after file handling errors.")
                      # Return 207 with only the file save errors
                      return JSONResponse(status_code=status.HTTP_207_MULTI_STATUS, content={"results": results})
                 else:
                      # This case should be caught by _validate_inputs earlier if BOTH urls and files are empty
                      logger.error("No input URLs or successfully saved files found.")
                      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid media sources found to process.")

            # Pass the instantiated 'form_data' object to helpers
            chunking_options_dict = _prepare_chunking_options_dict(form_data)
            common_processing_options = _prepare_common_options(form_data, chunking_options_dict)

            # Map input sources back to original refs (URL or original filename)
            # This helps in reporting results against the user's input identifier
            source_to_ref_map = {src: src for src in url_list} # URLs map to themselves
            source_to_ref_map.update({str(pf["path"]): pf["original_filename"] for pf in saved_files_info})

            # --- Get DB info from the dependency ---
            db_path_for_workers = db.db_path_str
            client_id_for_workers = db.client_id

            # --- 6. Process Media based on Type ---
            logging.info(f"Processing {len(all_valid_input_sources)} items of type '{form_data.media_type}'")

            if form_data.media_type in ['video', 'audio']:
                batch_results = await _process_batch_media(
                    media_type=form_data.media_type,
                    urls=url_list,
                    uploaded_file_paths=uploaded_file_paths,
                    source_to_ref_map=source_to_ref_map,
                    form_data=form_data,
                    chunk_options=chunking_options_dict,
                    loop=loop,
                    db_path=db_path_for_workers,
                    client_id=client_id_for_workers,
                    temp_dir=temp_dir_path
                )
                results.extend(batch_results)
            else:  # PDF/Document/Ebook
                tasks = [
                    _process_document_like_item(
                        item_input_ref=source_to_ref_map.get(source, source),
                        processing_source=source,
                        media_type=form_data.media_type,
                        is_url=(source in url_list),
                        form_data=form_data,
                        chunk_options=chunking_options_dict,
                        temp_dir=temp_dir_path,
                        loop=loop,
                        db_path=db_path_for_workers,
                        client_id=client_id_for_workers
                    )
                    for source in all_valid_input_sources
                ]
                individual_results = await asyncio.gather(*tasks)
                results.extend(individual_results)

        # --- 7. Determine Final Status Code and Return Response (Success Path) ---
        # TempDirManager handles cleanup automatically on exit from 'with' block
        final_status_code = _determine_final_status(results)
        log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
        logger.log(log_level, f"Request finished with status {final_status_code}. Results count: {len(results)}")

        # Successfully completed processing, return results
        return JSONResponse(status_code=final_status_code, content={"results": results})

    except HTTPException as e:
        # Log and re-raise HTTP exceptions
        logging.warning(f"HTTP Exception encountered: Status={e.status_code}, Detail={e.detail}")
        # Cleanup is handled by TempDirManager context exit
        raise e
    except OSError as e:
        # Handle potential errors during temp dir creation/management
        logging.error(f"OSError during processing setup: {e}", exc_info=True)
        # Cleanup is handled by TempDirManager context exit
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OS error during setup: {e}")
    except Exception as e:
        # Catch unexpected errors, ensure cleanup
        logging.error(f"Unhandled exception in add_media endpoint: {type(e).__name__} - {e}", exc_info=True)
        # Cleanup is handled by TempDirManager context exit
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected internal error: {type(e).__name__}")

    # No finally block needed for cleanup if using TempDirManager context

#
# End of General media ingestion and analysis
####################################################################################


######################## Video Processing Endpoint ###################################
#
# Video Processing Endpoint
# Endpoints:
# POST /api/v1/process-video

def get_process_videos_form(
    # Replicate Form(...) definitions from the original endpoint signature.
    # Use the field names from the Pydantic model where possible.
    # The 'alias' in Form(...) helps map incoming form keys.
    urls: Optional[List[str]] = Form(None, description="List of URLs of the video items"),
    title: Optional[str] = Form(None, description="Optional title (applied if only one item processed)"),
    author: Optional[str] = Form(None, description="Optional author (applied similarly to title)"),
    # Use the alias 'keywords' for the form field, matching AddMediaForm's alias for 'keywords_str'
    keywords: str = Form("", alias="keywords", description="Comma-separated keywords"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom prompt"),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt"),
    overwrite_existing: bool = Form(False, description="Overwrite existing media (Not used in this endpoint, but needed for model)"),
    perform_analysis: bool = Form(True, description="Perform analysis"),
    start_time: Optional[str] = Form(None, description="Optional start time (HH:MM:SS or seconds)"),
    end_time: Optional[str] = Form(None, description="Optional end time (HH:MM:SS or seconds)"),
    api_name: Optional[str] = Form(None, description="Optional API name"),
    api_key: Optional[str] = Form(None, description="Optional API key"), # Consider secure handling via settings
    use_cookies: bool = Form(False, description="Use cookies for URL download requests"),
    cookies: Optional[str] = Form(None, description="Cookie string if `use_cookies` is True"),
    transcription_model: str = Form("deepdml/faster-whisper-large-v3-turbo-ct2", description="Transcription model"),
    transcription_language: str = Form("en", description="Transcription language"),
    diarize: bool = Form(False, description="Enable speaker diarization"),
    timestamp_option: bool = Form(True, description="Include timestamps in transcription"),
    vad_use: bool = Form(False, description="Enable VAD filter"),
    perform_confabulation_check_of_analysis: bool = Form(False, description="Enable confabulation check"),
    pdf_parsing_engine: Optional[PdfEngine] = Form("pymupdf4llm", description="PDF parsing engine (for model compatibility)"),
    perform_chunking: bool = Form(True, description="Enable chunking"), # Default from ChunkingOptions
    chunk_method: Optional[ChunkMethod] = Form(None, description="Chunking method"),
    use_adaptive_chunking: bool = Form(False, description="Enable adaptive chunking"),
    use_multi_level_chunking: bool = Form(False, description="Enable multi-level chunking"),
    chunk_language: Optional[str] = Form(None, description="Chunking language override"),
    chunk_size: int = Form(500, description="Target chunk size"),
    chunk_overlap: int = Form(200, description="Chunk overlap size"),
    custom_chapter_pattern: Optional[str] = Form(None, description="Regex pattern for custom chapter splitting"),
    perform_rolling_summarization: bool = Form(False, description="Perform rolling summarization"),
    summarize_recursively: bool = Form(False, description="Perform recursive summarization"),
    # --- Keep Token and Files separate ---
    #token: str = Header(..., description="Authentication token"),  # Auth handled by get_db_for_user
    db=Depends(get_db_for_user)
) -> ProcessVideosForm:
    """
    Dependency function to parse form data and validate it
    against the ProcessVideosForm model.
    """
    try:
        # Create the Pydantic model instance using the parsed form data.
        form_instance = ProcessVideosForm(
            media_type="video", # Fixed by ProcessVideosForm
            urls=urls,
            title=title,
            author=author,
            keywords=keywords, # Pydantic handles mapping this to keywords_str via alias
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            overwrite_existing=overwrite_existing,
            keep_original_file=False, # Fixed by ProcessVideosForm
            perform_analysis=perform_analysis,
            start_time=start_time,
            end_time=end_time,
            api_name=api_name,
            api_key=api_key,
            use_cookies=use_cookies,
            cookies=cookies,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            diarize=diarize,
            timestamp_option=timestamp_option,
            vad_use=vad_use,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
            pdf_parsing_engine=pdf_parsing_engine,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            custom_chapter_pattern=custom_chapter_pattern,
            perform_rolling_summarization=perform_rolling_summarization,
            summarize_recursively=summarize_recursively,
        )
        return form_instance
    except ValidationError as e:
        # Process errors to make them JSON serializable by handling exceptions in 'ctx'
        serializable_errors = []
        for error in e.errors():
            serializable_error = error.copy()  # Work on a copy
            if 'ctx' in serializable_error and isinstance(serializable_error.get('ctx'), dict):
                # Create a new ctx dict, stringifying any exceptions
                new_ctx = {}
                for k, v in serializable_error['ctx'].items():
                    if isinstance(v, Exception):
                        new_ctx[k] = str(v)  # Convert Exception to string
                    else:
                        new_ctx[k] = v  # Keep other values as is
                serializable_error['ctx'] = new_ctx
                # Alternatively, if client doesn't need ctx, uncomment the next line:
                # del serializable_error['ctx']
            serializable_errors.append(serializable_error)

        logger.warning(f"Pydantic validation failed: {json.dumps(serializable_errors)}")
        # Raise HTTPException with the processed, serializable error details
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=serializable_errors,  # Pass the cleaned list
        ) from e
    except Exception as e: # Catch other potential errors during instantiation
        logger.error(f"Unexpected error creating ProcessVideosForm: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during form processing: {type(e).__name__}"
        )

# =============================================================================
# Video Processing Endpoint
# =============================================================================
@router.post(
    "/process-videos",
    # status_code=status.HTTP_200_OK, # Status determined dynamically
    summary="Transcribe / chunk / analyse videos and return the full artefacts (no DB write)",
    tags=["Media Processing (No DB)"],
)
async def process_videos_endpoint(
    # --- Dependencies ---
    background_tasks: BackgroundTasks,
    # 1. Auth + UserID Determined through `get_db_by_user`
    # Add check here for granular permissions if needed
    # 2. DB Dependency
    db: Database = Depends(get_db_for_user),
    # 3. Form Data Dependency: Parses form fields into the Pydantic model.
    form_data: ProcessVideosForm = Depends(get_process_videos_form),
    # 4. File Uploads
    files: Optional[List[UploadFile]] = File(None, description="Video file uploads"),
    # user_info: dict = Depends(verify_token), # Optional Auth
):
    """
    **Process Videos Endpoint (Fixed)**

    Transcribes, chunks, and analyses videos from URLs or uploaded files.
    Returns processing artifacts without saving to the database.
    Corrected the run_in_executor call and input_ref mapping.
    """
    # --- Validation and Logging ---
    logger.info("Request received for /process-videos. Form data validated via dependency.")

    if form_data.urls and form_data.urls == ['']:
        logger.info("Received urls=[''], treating as no URLs provided for video processing.")
        form_data.urls = None # Or []

    _validate_inputs("video", form_data.urls, files) # Keep basic input check

    # --- Setup ---
    loop = asyncio.get_running_loop()
    batch_result: Dict[str, Any] = {"processed_count": 0, "errors_count": 0, "errors": [], "results": [], "confabulation_results": None}
    file_handling_errors_structured: List[Dict[str, Any]] = []
    # --- Map to store temporary path -> original filename ---
    temp_path_to_original_name: Dict[str, str] = {}

    # --- Use TempDirManager for reliable cleanup ---
    with TempDirManager(cleanup=True, prefix="process_video_") as temp_dir:
        logger.info(f"Using temporary directory for /process-videos: {temp_dir}")

        # --- Save Uploads ---
        saved_files_info, file_handling_errors_raw = await _save_uploaded_files(files or [], temp_dir, validator=file_validator_instance,)

        # --- Populate the temp path to original name map ---
        for sf in saved_files_info:
            if sf.get("path") and sf.get("original_filename"):
                # Convert Path object to string for consistent dictionary keys
                temp_path_to_original_name[str(sf["path"])] = sf["original_filename"]
            else:
                logger.warning(f"Missing path or original_filename in saved_files_info item: {sf}")


        # --- Process File Handling Errors ---
        if file_handling_errors_raw:
            batch_result["errors_count"] += len(file_handling_errors_raw)
            batch_result["errors"].extend([err.get("error", "Unknown file save error") for err in file_handling_errors_raw])
            # Adapt raw file errors to the MediaItemProcessResponse structure
            for err in file_handling_errors_raw:
                 # *** Use original filename for input_ref here ***
                 original_filename = err.get("input", "Unknown Filename") # Assume 'input' holds original name from _save_uploaded_files error
                 file_handling_errors_structured.append({
                     "status": "Error",
                     "input_ref": err.get("input", "Unknown Filename"),
                     "processing_source": "N/A - File Save Failed",
                     "media_type": "video",
                     "metadata": {}, "content": "", "segments": None, "chunks": None,
                     "analysis": None, "analysis_details": {},
                     "error": err.get("error", "Failed to save uploaded file."), "warnings": None,
                     "db_id": None, "db_message": "Processing only endpoint.", "message": None,
                 })
            batch_result["results"].extend(file_handling_errors_structured) # Add structured errors

        # --- Prepare Inputs for Processing ---
        url_list = form_data.urls or []
        # Get the temporary paths (as strings) from saved_files_info
        uploaded_paths = [str(sf["path"]) for sf in saved_files_info if sf.get("path")]
        all_inputs_to_process = url_list + uploaded_paths

        # Check if there's anything left to process
        if not all_inputs_to_process:
            if file_handling_errors_raw: # Only file errors occurred
                logger.warning("No valid video sources to process after file saving errors.")
                # Return 207 with the structured file errors
                return JSONResponse(status_code=status.HTTP_207_MULTI_STATUS, content=batch_result)
            else: # No inputs provided at all
                logger.warning("No video sources provided.")
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "No valid video sources supplied.")

        # --- Call process_videos ---
        video_args = {
            "inputs": all_inputs_to_process,
            # Use form_data directly
            "start_time": form_data.start_time,
            "end_time": form_data.end_time,
            "diarize": form_data.diarize,
            "vad_use": form_data.vad_use,
            "transcription_model": form_data.transcription_model,
            "transcription_language": form_data.transcription_language, # Add language if process_videos needs it
            "perform_analysis": form_data.perform_analysis,
            "custom_prompt": form_data.custom_prompt,
            "system_prompt": form_data.system_prompt,
            "perform_chunking": form_data.perform_chunking,
            "chunk_method": form_data.chunk_method,
            "max_chunk_size": form_data.chunk_size,
            "chunk_overlap": form_data.chunk_overlap,
            "use_adaptive_chunking": form_data.use_adaptive_chunking,
            "use_multi_level_chunking": form_data.use_multi_level_chunking,
            "chunk_language": form_data.chunk_language,
            "summarize_recursively": form_data.summarize_recursively,
            "api_name": form_data.api_name if form_data.perform_analysis else None,
            "api_key": form_data.api_key,
            "use_cookies": form_data.use_cookies,
            "cookies": form_data.cookies,
            "timestamp_option": form_data.timestamp_option,
            "perform_confabulation_check": form_data.perform_confabulation_check_of_analysis,
            "temp_dir": str(temp_dir),  # Pass the managed temporary directory path
            # 'keep_original' might be relevant if library needs it, default is False
            # 'perform_diarization' seems redundant if 'diarize' is passed, check library usage
            # If perform_diarization is truly needed separately:
            # "perform_diarization": form_data.diarize, # Or map if different logic
        }

        try:
            logger.debug(f"Calling process_videos for /process-videos endpoint with {len(all_inputs_to_process)} inputs.")
            batch_func = functools.partial(process_videos, **video_args)

            processing_output = await loop.run_in_executor(None, batch_func)

            # Debug logging
            try:
                print(f"!!! DEBUG PRINT !!! My debug message: {json.dumps(processing_output, indent=2, default=str)}")
            except Exception as log_err:
                print(f"!!! DEBUG PRINT !!! My debug message: {log_err}")

            # --- Combine Processing Results ---
            # Reset results list if we only had file errors before, or append otherwise
            # Clear the specific counters before processing the library output
            batch_result["processed_count"] = 0
            batch_result["errors_count"] = 0
            batch_result["errors"] = []

            # Start with any structured file errors we recorded earlier
            final_results_list = list(file_handling_errors_structured)
            final_errors_list = [err.get("error", "File handling error") for err in file_handling_errors_structured]

            if isinstance(processing_output, dict):
                # Add results from the library processing
                processed_results_from_lib = processing_output.get("results", [])
                for res in processed_results_from_lib:
                    # *** Map input_ref back to original filename if applicable ***
                    current_input_ref = res.get("input_ref") # This is likely the temp path or URL
                    # If the current_input_ref is a key in our map, use the original name
                    # Otherwise, keep the current_input_ref (it's likely a URL)
                    res["input_ref"] = temp_path_to_original_name.get(current_input_ref, current_input_ref)

                    # Add endpoint-specific fields
                    res["db_id"] = None
                    res["db_message"] = "Processing only endpoint."
                    final_results_list.append(res) # Add the modified result

                # Add specific errors reported by the library
                final_errors_list.extend(processing_output.get("errors", []))

                # Handle confabulation results if present
                if "confabulation_results" in processing_output:
                    batch_result["confabulation_results"] = processing_output["confabulation_results"]

            else:
                # Handle unexpected output from process_videos library function
                logger.error(f"process_videos function returned unexpected type: {type(processing_output)}")
                general_error_msg = "Video processing library returned invalid data."
                final_errors_list.append(general_error_msg)
                # Create error entries for all inputs attempted in *this specific* processing call
                for input_src in all_inputs_to_process:
                    # *** Use original name for error input_ref if possible ***
                    original_ref_for_error = temp_path_to_original_name.get(input_src, input_src)
                    final_results_list.append({
                        "status": "Error",
                        "input_ref": original_ref_for_error, # Use original name/URL
                        "processing_source": input_src, # Show what was actually processed (temp path/URL)
                        "media_type": "video", "metadata": {}, "content": "", "segments": None,
                        "chunks": None, "analysis": None, "analysis_details": {},
                        "error": general_error_msg, "warnings": None, "db_id": None,
                        "db_message": "Processing only endpoint.", "message": None
                    })

            # --- Recalculate final counts based on the merged list ---
            batch_result["results"] = final_results_list
            batch_result["processed_count"] = sum(1 for r in final_results_list if r.get("status") == "Success")
            batch_result["errors_count"] = sum(1 for r in final_results_list if r.get("status") == "Error")
            # Remove duplicates from error messages list if desired
            # Make sure errors are strings before adding to set
            unique_errors = set(str(e) for e in final_errors_list if e is not None)
            batch_result["errors"] = list(unique_errors)


        except Exception as exec_err:
            # Catch errors during the library execution call itself
            logger.error(f"Error executing process_videos: {exec_err}", exc_info=True)
            error_msg = f"Error during video processing execution: {type(exec_err).__name__}"

            # Start with existing file errors
            final_results_list = list(file_handling_errors_structured)
            final_errors_list = [err.get("error", "File handling error") for err in file_handling_errors_structured]
            final_errors_list.append(error_msg)  # Add the execution error

            # Create error entries for all inputs attempted in this batch
            for input_src in all_inputs_to_process:
                 # *** Use original name for error input_ref if possible ***
                 original_ref_for_error = temp_path_to_original_name.get(input_src, input_src)
                 final_results_list.append({
                    "status": "Error",
                    "input_ref": original_ref_for_error, # Use original name/URL
                    "processing_source": input_src, # Show what was actually processed (temp path/URL)
                    "media_type": "video", "metadata": {}, "content": "", "segments": None,
                    "chunks": None, "analysis": None, "analysis_details": {},
                    "error": error_msg, "warnings": None, "db_id": None,
                    "db_message": "Processing only endpoint.", "message": None
                })

            # --- Update batch_result with merged errors ---
            batch_result["results"] = final_results_list
            batch_result["processed_count"] = 0 # Assume all failed if execution failed
            batch_result["errors_count"] = len(final_results_list) # Count all items as errors now
            unique_errors = set(str(e) for e in final_errors_list if e is not None)
            batch_result["errors"] = list(unique_errors)

        # --- Determine Final Status Code & Return ---
        # Base the status code *solely* on the final calculated errors_count
        final_error_count = batch_result.get("errors_count", 0)
        # Check if there are only warnings and no errors
        final_success_count = batch_result.get("processed_count", 0)
        total_items = len(batch_result.get("results", []))
        has_warnings = any(r.get("status") == "Warning" for r in batch_result.get("results", []))

        if total_items == 0: # Should not happen if validation passed, but handle defensively
            final_status_code = status.HTTP_400_BAD_REQUEST # Or 500?
            logger.error("No results generated despite processing attempt.")
        elif final_error_count == 0:
             final_status_code = status.HTTP_200_OK
        elif final_error_count == total_items:
             final_status_code = status.HTTP_207_MULTI_STATUS # All errors, could also be 4xx/5xx depending on cause
        else: # Mix of success/warnings/errors
             final_status_code = status.HTTP_207_MULTI_STATUS

        log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
        logger.log(log_level,
                   f"/process-videos request finished with status {final_status_code}. Results count: {len(batch_result.get('results', []))}, Errors: {final_error_count}")

        # --- TEMPORARY DEBUG ---
        try:
            logger.debug("Final batch_result before JSONResponse:")
            # Log only a subset if the full result is too large
            logged_result = batch_result.copy()
            if len(logged_result.get('results', [])) > 5: # Log details for first 5 results only
                 logged_result['results'] = logged_result['results'][:5] + [{"message": "... remaining results truncated for logging ..."}]
            logger.debug(json.dumps(logged_result, indent=2, default=str)) # Use default=str for non-serializable items

            success_item_debug = next((r for r in batch_result.get("results", []) if r.get("status") == "Success"), None)
            if success_item_debug:
                logger.debug(f"Value of input_ref for success item before return: {success_item_debug.get('input_ref')}")
            else:
                logger.debug("No success item found in final results before return.")
        except Exception as debug_err:
            logger.error(f"Error during debug logging: {debug_err}")
        # --- END TEMPORARY DEBUG ---

        return JSONResponse(status_code=final_status_code, content=batch_result)

#
# End of Video Processing
####################################################################################


######################## Audio Processing Endpoint ###################################
# Endpoints:
#   /process-audio

# =============================================================================
# Dependency Function for Audio Form Processing
# =============================================================================
def get_process_audios_form(
    # Replicate relevant Form(...) definitions for audio
    urls: Optional[List[str]] = Form(None, description="List of URLs of the audio items"),
    title: Optional[str] = Form(None, description="Optional title (applied if only one item processed)"),
    author: Optional[str] = Form(None, description="Optional author (applied similarly to title)"),
    keywords: str = Form("", alias="keywords", description="Comma-separated keywords"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom prompt"),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt"),
    overwrite_existing: bool = Form(False, description="Overwrite existing media (Not used in this endpoint, but needed for model)"),
    perform_analysis: bool = Form(True, description="Perform analysis"),
    api_name: Optional[str] = Form(None, description="Optional API name"),
    api_key: Optional[str] = Form(None, description="Optional API key"),
    use_cookies: bool = Form(False, description="Use cookies for URL download requests"),
    cookies: Optional[str] = Form(None, description="Cookie string if `use_cookies` is True"),
    transcription_model: str = Form("deepdml/faster-distil-whisper-large-v3.5", description="Transcription model"),
    transcription_language: str = Form("en", description="Transcription language"),
    diarize: bool = Form(False, description="Enable speaker diarization"),
    timestamp_option: bool = Form(True, description="Include timestamps in transcription"),
    vad_use: bool = Form(False, description="Enable VAD filter"),
    perform_confabulation_check_of_analysis: bool = Form(False, description="Enable confabulation check"),
    # Chunking options
    perform_chunking: bool = Form(True, description="Enable chunking"),
    chunk_method: Optional[ChunkMethod] = Form(None, description="Chunking method"),
    use_adaptive_chunking: bool = Form(False, description="Enable adaptive chunking"),
    use_multi_level_chunking: bool = Form(False, description="Enable multi-level chunking"),
    chunk_language: Optional[str] = Form(None, description="Chunking language override"),
    chunk_size: int = Form(500, description="Target chunk size"),
    chunk_overlap: int = Form(200, description="Chunk overlap size"),
    # Summarization options
    perform_rolling_summarization: bool = Form(False, description="Perform rolling summarization"), # Keep if AddMediaForm has it
    summarize_recursively: bool = Form(False, description="Perform recursive summarization"),
    # PDF options (Needed for AddMediaForm compatibility, ignored for audio)
    pdf_parsing_engine: Optional[PdfEngine] = Form("pymupdf4llm", description="PDF parsing engine (for model compatibility)"),
    custom_chapter_pattern: Optional[str] = Form(None, description="Regex pattern for custom chapter splitting (for model compatibility)"),
    # Audio/Video specific timing (Not applicable to audio-only usually, but keep for model compatibility if needed)
    start_time: Optional[str] = Form(None, description="Optional start time (HH:MM:SS or seconds)"),
    end_time: Optional[str] = Form(None, description="Optional end time (HH:MM:SS or seconds)"),

) -> ProcessAudiosForm:
    """
    Dependency function to parse form data and validate it
    against the ProcessAudiosForm model.
    """
    try:
        # Map form fields to ProcessAudiosForm fields
        form_instance = ProcessAudiosForm(
            urls=urls,
            title=title,
            author=author,
            keywords=keywords,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            overwrite_existing=overwrite_existing,
            keep_original_file=False,
            perform_analysis=perform_analysis,
            api_name=api_name,
            api_key=api_key,
            use_cookies=use_cookies,
            cookies=cookies,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            diarize=diarize,
            timestamp_option=timestamp_option,
            vad_use=vad_use,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            summarize_recursively=summarize_recursively,
            # Include fields inherited from AddMediaForm even if not directly used for audio
            perform_rolling_summarization=perform_rolling_summarization,
            pdf_parsing_engine=pdf_parsing_engine,
            custom_chapter_pattern=custom_chapter_pattern,
            start_time=start_time,
            end_time=end_time,
        )
        return form_instance
    except ValidationError as e:
        # Log the validation error details for debugging
        logger.warning(f"Form validation failed for /process-audios: {e.errors()}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating ProcessAudiosForm: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during form processing: {type(e).__name__}"
        )


# =============================================================================
# Audio Processing Endpoint (REFACTORED)
# =============================================================================
@router.post(
    "/process-audios",
    # status_code=status.HTTP_200_OK, # Status determined dynamically
    summary="Transcribe / chunk / analyse audio and return full artefacts (no DB write)",
    tags=["Media Processing (No DB)"],
    # Consider adding response models for better documentation and validation
    # response_model=YourBatchResponseModel,
    # responses={ # Example explicit responses
    #     200: {"description": "All items processed successfully."},
    #     207: {"description": "Partial success with some errors."},
    #     400: {"description": "Bad request (e.g., no input)."},
    #     422: {"description": "Validation error in form data."},
    #     500: {"description": "Internal server error."},
    # }
)
async def process_audios_endpoint(
    background_tasks: BackgroundTasks,
    # 1. Auth + UserID Determined through `get_db_by_user`
    # token: str = Header(None), # Use Header(None) for optional
    # 2. DB Dependency
    db: Database = Depends(get_db_for_user),
    # 3. Use Dependency Injection for Form Data
    form_data: ProcessAudiosForm = Depends(get_process_audios_form),
    # 4. File uploads remain separate
    files: Optional[List[UploadFile]] = File(None, description="Audio file uploads"),
):
    """
    **Process Audio Endpoint (Refactored)**

    Transcribes, chunks, and analyses audio from URLs or uploaded files.
    Returns processing artifacts without saving to the database. Uses dependency
    injection for form handling, consistent with the video endpoint.
    """
    # --- 0) Validation and Logging ---
    # Validation happened in the dependency. Log success or handle HTTPException.
    logger.info(f"Request received for /process-audios. Form data validated via dependency.")

    if form_data.urls and form_data.urls == ['']:
        logger.info("Received urls=[''], treating as no URLs provided for audio processing.")
        form_data.urls = None # Or []

    # Use the helper function from media_endpoints_utils
    try:
        _validate_inputs("audio", form_data.urls, files)
    except HTTPException as e:
         logger.warning(f"Input validation failed: {e.detail}")
         # Re-raise the HTTPException from _validate_inputs
         raise e

    # --- Rest of the logic using form_data ---
    loop = asyncio.get_running_loop()
    file_errors: List[Dict[str, Any]] = []
    # Initialize batch result structure
    batch_result: Dict[str, Any] = {"processed_count": 0, "errors_count": 0, "errors": [], "results": []}
    temp_path_to_original_name: Dict[str, str] = {}

    # ── 1) temp dir + uploads ────────────────────────────────────────────────
    with TempDirManager(cleanup=True, prefix="process_audio_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        ALLOWED_AUDIO_EXTENSIONS = ['.mp3', '.aac', '.flac', '.wav', '.ogg', '.m4a'] # Define allowed extensions
        saved_files, file_errors_raw = await _save_uploaded_files(
            files or [],
            temp_dir_path,
            validator=file_validator_instance,
            allowed_extensions=ALLOWED_AUDIO_EXTENSIONS # Pass allowed extensions
        )

        for sf in saved_files:
            if sf.get("path") and sf.get("original_filename"):
                temp_path_to_original_name[str(sf["path"])] = sf["original_filename"]
            else:
                logger.warning(f"Missing path or original_filename in saved_files_info item for audio: {sf}")

        # --- Adapt File Errors to Response Structure ---
        if file_errors:
            adapted_file_errors = []
            for err in file_errors:
                 # Ensure all necessary keys are present for consistency
                 original_filename = err.get("original_filename") or err.get("input", "Unknown Upload")
                 adapted_file_errors.append({
                     "status": "Error",
                     "input_ref": original_filename,
                     "processing_source": err.get("input", "Unknown Filename"),
                     "media_type": "audio",
                     "metadata": {},
                     "content": "",
                     "segments": None,
                     "chunks": None,      # Add chunks field
                     "analysis": None,    # Add analysis field
                     "analysis_details": {},
                     "error": err.get("error", "Failed to save uploaded file."),
                     "warnings": None,
                     "db_id": None,       # Explicitly None
                     "db_message": "Processing only endpoint.", # Explicit message
                     "message": "File saving failed.", # Optional general message
                 })
            batch_result["results"].extend(adapted_file_errors)
            batch_result["errors_count"] = len(file_errors)
            batch_result["errors"].extend([err["error"] for err in adapted_file_errors])

        url_list = form_data.urls or []
        uploaded_paths = [str(f["path"]) for f in saved_files]
        all_inputs = url_list + uploaded_paths

        # Check if there are any valid inputs *after* attempting saves
        if not all_inputs:
            # If only file errors occurred, return 207, otherwise 400
            status_code = status.HTTP_207_MULTI_STATUS if file_errors_raw else status.HTTP_400_BAD_REQUEST
            detail = "No valid audio sources supplied (or all uploads failed)."
            logger.warning(f"Request processing stopped: {detail}")
            if status_code == status.HTTP_400_BAD_REQUEST:
                 raise HTTPException(status_code=status_code, detail=detail)
            else:
                 return JSONResponse(status_code=status_code, content=batch_result)


        # ── 2) invoke library batch processor ────────────────────────────────
        # Use validated form_data directly
        audio_args = {
            "inputs": all_inputs,
            "transcription_model": form_data.transcription_model,
            "transcription_language": form_data.transcription_language,
            "perform_chunking": form_data.perform_chunking,
            "chunk_method": form_data.chunk_method if form_data.chunk_method else None, # Pass enum value
            "max_chunk_size": form_data.chunk_size, # Correct mapping
            "chunk_overlap": form_data.chunk_overlap,
            "use_adaptive_chunking": form_data.use_adaptive_chunking,
            "use_multi_level_chunking": form_data.use_multi_level_chunking,
            "chunk_language": form_data.chunk_language,
            "diarize": form_data.diarize,
            "vad_use": form_data.vad_use,
            "timestamp_option": form_data.timestamp_option,
            "perform_analysis": form_data.perform_analysis,
            "api_name": form_data.api_name if form_data.perform_analysis else None,
            "api_key": form_data.api_key,
            "custom_prompt_input": form_data.custom_prompt,
            "system_prompt_input": form_data.system_prompt,
            "summarize_recursively": form_data.summarize_recursively,
            "use_cookies": form_data.use_cookies,
            "cookies": form_data.cookies,
            "keep_original": False, # Explicitly false for this endpoint
            "custom_title": form_data.title,
            "author": form_data.author,
            "temp_dir": str(temp_dir_path), # Pass the managed temp dir path
        }

        processing_output = None
        try:
            logger.debug(f"Calling process_audio_files for /process-audios with {len(all_inputs)} inputs.")
            # Use functools.partial to pass arguments cleanly
            batch_func = functools.partial(process_audio_files, **audio_args)
            # Run the synchronous library function in an executor thread
            processing_output = await loop.run_in_executor(None, batch_func)

        except Exception as exec_err:
            # Catch errors during the execution setup or within the library if it raises unexpectedly
            logging.error(f"Error executing process_audio_files: {exec_err}", exc_info=True)
            error_msg = f"Error during audio processing execution: {type(exec_err).__name__}"
            # Calculate errors based on *attempted* inputs for this batch
            num_attempted = len(all_inputs)
            batch_result["errors_count"] += num_attempted # Assume all failed if executor errored
            batch_result["errors"].append(error_msg)
            # Create error entries for all inputs attempted in this batch
            error_results = []
            for input_src in all_inputs:
                original_ref = temp_path_to_original_name.get(str(input_src), str(input_src))
                if input_src in uploaded_paths:
                    for sf in saved_files:
                         if str(sf["path"]) == input_src:
                              original_ref = sf.get("original_filename", input_src)
                              break
                error_results.append({
                    "status": "Error",
                    "input_ref": original_ref,
                    "processing_source": input_src,
                    "media_type": "audio",
                    "error": error_msg,
                    "db_id": None,
                    "db_message": "Processing only endpoint.",
                    "metadata": {},
                    "content": "",
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "warnings": None,
                    "message": "Processing execution failed."
                })
            # Combine these errors with any previous file errors
            batch_result["results"].extend(error_results)
            # Fall through to return section

        # --- Merge Processing Results ---
        if processing_output and isinstance(processing_output, dict) and "results" in processing_output:
            # Update counts based on library's report
            batch_result["processed_count"] += processing_output.get("processed_count", 0)
            new_errors_count = processing_output.get("errors_count", 0)
            batch_result["errors_count"] += new_errors_count
            batch_result["errors"].extend(processing_output.get("errors", []))

            processed_items = processing_output.get("results", [])
            adapted_processed_items = []
            for item in processed_items:

                 identifier_from_lib = item.get("input_ref") or item.get("processing_source")
                 original_ref = temp_path_to_original_name.get(str(identifier_from_lib), str(identifier_from_lib))
                 item["input_ref"] = original_ref
                 # Keep processing_source as what library used
                 item["processing_source"] = identifier_from_lib or original_ref

                 # Ensure DB fields are set correctly and all expected fields exist
                 item["db_id"] = None
                 item["db_message"] = "Processing only endpoint."
                 item.setdefault("status", "Error") # Default status if missing
                 item.setdefault("input_ref", "Unknown")
                 item.setdefault("processing_source", "Unknown")
                 item.setdefault("media_type", "audio") # Ensure media type
                 item.setdefault("metadata", {})
                 item.setdefault("content", None) # Default content to None
                 item.setdefault("segments", None)
                 item.setdefault("chunks", None) # Add default for chunks
                 item.setdefault("analysis", None) # Add default for analysis
                 item.setdefault("analysis_details", {})
                 item.setdefault("error", None)
                 item.setdefault("warnings", None)
                 item.setdefault("message", None) # Optional message from library
                 adapted_processed_items.append(item)

            # Combine processing results with any previous file errors
            batch_result["results"].extend(adapted_processed_items)

        elif processing_output is None and not batch_result["results"]: # Handle case where executor failed AND no file errors
             # This case is now handled by the try/except around run_in_executor
             pass
        elif processing_output is not None:
            # Handle unexpected output format from the library function more gracefully
            logging.error(f"process_audio_files returned unexpected format: Type={type(processing_output)}")
            error_msg = "Audio processing library returned invalid data."
            num_attempted = len(all_inputs)
            batch_result["errors_count"] += num_attempted
            batch_result["errors"].append(error_msg)
            # Create error results for inputs if not already present
            existing_refs = {res.get("input_ref") for res in batch_result["results"]}
            error_results = []
            for input_src in all_inputs:
                original_ref = temp_path_to_original_name.get(str(input_src), str(input_src))
                if input_src in uploaded_paths:
                    for sf in saved_files:
                         if str(sf["path"]) == input_src:
                              original_ref = sf.get("original_filename", input_src)
                              break
                if original_ref not in existing_refs: # Only add errors for inputs not already covered (e.g., by file errors)
                    error_results.append({
                        "status": "Error",
                        "input_ref": original_ref,
                        "processing_source": input_src,
                        "media_type": "audio",
                        "error": error_msg,
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                        "metadata": {}, "content": "",
                        "segments": None,
                        "chunks": None,
                        "analysis": None,
                        "analysis_details": {},
                        "warnings": None,
                        "message": "Invalid processing result."
                    })
            batch_result["results"].extend(error_results)

    # TempDirManager cleans up the directory automatically here (unless keep_original=True passed to it)
    # ── 4) Determine Final Status Code ───────────────────────────────────────
    # Base final status on whether *any* errors occurred (file saving or processing)
    final_processed_count = sum(1 for r in batch_result["results"] if r.get("status") == "Success")
    final_error_count = sum(1 for r in batch_result["results"] if r.get("status") == "Error")
    batch_result["processed_count"] = final_processed_count
    batch_result["errors_count"] = final_error_count
    # Update errors list to avoid duplicates (optional)
    unique_errors = list(set(str(e) for e in batch_result["errors"] if e))
    batch_result["errors"] = unique_errors

    final_status_code = (
        status.HTTP_200_OK if batch_result.get("errors_count", 0) == 0 and batch_result.get("processed_count", 0) > 0
        else status.HTTP_207_MULTI_STATUS if batch_result.get("results") # Return 207 if there are *any* results (success, warning, or error)
        else status.HTTP_400_BAD_REQUEST # Only 400 if no inputs were ever processed (e.g., invalid initial request)
    )

    # --- Return Combined Results ---
    if final_status_code == status.HTTP_200_OK:
        logging.info("Congrats, all successful!")
        logger.info(
            f"/process-audios request finished with status {final_status_code}. Results count: {len(batch_result.get('results', []))}, Total Errors: {batch_result.get('errors_count', 0)}")
    else:
        logging.warning("Not all submissions were processed succesfully! Please Try Again!")
        logger.warning(f"/process-audios request finished with status {final_status_code}. Results count: {len(batch_result.get('results', []))}, Total Errors: {batch_result.get('errors_count', 0)}")

    return JSONResponse(status_code=final_status_code, content=batch_result)

#
# End of Audio Processing
##############################################################################################


######################## Ebook Processing Endpoint ###################################

# ─────────────────────── Form Model ─────────────────────────
class ProcessEbooksForm(AddMediaForm):
    media_type: Literal["ebook"] = "ebook"
    extraction_method: Literal['filtered', 'markdown', 'basic'] = Field('filtered', description="EPUB text extraction method ('filtered', 'markdown', 'basic')")
    keep_original_file: bool = False    # always cleanup tmp dir for this endpoint
    # Add any ebook specific options if needed, otherwise inherit from AddMediaForm

def _process_single_ebook(
    ebook_path: Path,
    original_ref: str, # Pass the original URL or filename
    # Pass necessary options from form_data
    title_override: Optional[str],
    author_override: Optional[str],
    keywords: Optional[List[str]],
    perform_chunking: bool,
    chunk_options: Optional[Dict[str, Any]],
    perform_analysis: bool,
    summarize_recursively: bool,
    api_name: Optional[str],
    api_key: Optional[str],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    extraction_method: str, # Pass selected method
) -> Dict[str, Any]:
    """
    Synchronous helper function to process one EPUB file using the library.
    Designed to be run in a thread executor.
    *No DB interaction.*
    """
    try:
        logger.info(f"Worker processing ebook: {original_ref} from path {ebook_path}")
        # Call the main library processing function
        result_dict = process_epub(
            file_path=str(ebook_path),
            title_override=title_override,
            author_override=author_override,
            keywords=keywords,
            perform_chunking=perform_chunking,
            chunk_options=chunk_options,
            perform_analysis=perform_analysis,
            api_name=api_name,
            api_key=api_key,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            summarize_recursively=summarize_recursively,
            extraction_method=extraction_method
        )
        # Ensure input_ref is set to the original URL/filename for consistency
        result_dict["input_ref"] = original_ref
        # processing_source is already set by process_epub to the actual path
        return result_dict

    except Exception as e:
        logger.error(f"_process_single_ebook error for {original_ref} ({ebook_path}): {e}", exc_info=True)
        # Return a standardized error dictionary consistent with process_epub
        return {
            "status": "Error",
            "input_ref": original_ref, # Use original ref for error reporting
            "processing_source": str(ebook_path),
            "media_type": "ebook",
            "error": f"Worker processing failed: {str(e)}",
            "content": None, "metadata": None, "chunks": None, "analysis": None,
            "keywords": keywords or [], "warnings": None, "analysis_details": None # Add analysis_details
        }


# ────────────────────── Dependency Function ───────────────────
def get_process_ebooks_form(
    # --- Inherited Fields from AddMediaForm ---
    urls: Optional[List[str]] = Form(None, description="List of URLs of the EPUB items"),
    title: Optional[str] = Form(None, description="Optional title override"),
    author: Optional[str] = Form(None, description="Optional author override"),
    keywords: str = Form("", alias="keywords_str", description="Comma-separated keywords"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom prompt for analysis"),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt for analysis"),
    overwrite_existing: bool = Form(False, description="Overwrite existing media (Not used, for model validation)"),
    perform_analysis: bool = Form(True, description="Perform analysis (summarization)"),
    api_name: Optional[str] = Form(None, description="Optional API name for analysis"),
    api_key: Optional[str] = Form(None, description="Optional API key for analysis"),
    use_cookies: bool = Form(False, description="Use cookies for URL download requests (Not implemented for ebooks)"),
    cookies: Optional[str] = Form(None, description="Cookie string (Not implemented for ebooks)"),
    summarize_recursively: bool = Form(False, description="Perform recursive summarization"),
    perform_rolling_summarization: bool = Form(False, description="Perform rolling summarization (Not applicable to ebooks)"),

    # --- Fields from ChunkingOptions ---
    perform_chunking: bool = Form(True, description="Enable chunking (default: by chapter)"),
    chunk_method: Optional[ChunkMethod] = Form('chapter', description="Chunking method (chapter, recursive, sentences etc.)"),
    chunk_language: Optional[str] = Form(None, description="Chunking language override (rarely needed for chapter)"),
    chunk_size: int = Form(1500, description="Target chunk size (used by non-chapter methods)"),
    chunk_overlap: int = Form(200, description="Chunk overlap size (used by non-chapter methods)"),
    custom_chapter_pattern: Optional[str] = Form(None, description="Regex pattern for custom chapter splitting (overrides method default)"),

    # --- Ebook Specific Options (Add if needed) ---
    extraction_method: Literal['filtered', 'markdown', 'basic'] = Form('filtered',
                                                                           description="EPUB text extraction method"),

    # --- Fields from other options (like AudioVideo) if needed for model validation ---
    # Include placeholders if AddMediaForm requires them, even if not used by ebooks
    start_time: Optional[str] = Form(None), end_time: Optional[str] = Form(None),
    transcription_model: Optional[str] = Form(None), transcription_language: Optional[str] = Form(None),
    diarize: Optional[bool] = Form(None), timestamp_option: Optional[bool] = Form(None),
    vad_use: Optional[bool] = Form(None), perform_confabulation_check_of_analysis: Optional[bool] = Form(None),
    # Include PDF options placeholder if AddMediaForm requires
    pdf_parsing_engine: Optional[PdfEngine] = Form(None),

    # --- Fields from ChunkingOptions - NOT used explicitly by ebooks but part of AddMediaForm ---
    use_adaptive_chunking: bool = Form(False, description="Enable adaptive chunking (Not applicable)"),
    use_multi_level_chunking: bool = Form(False, description="Enable multi-level chunking (Not applicable)"),
) -> ProcessEbooksForm:
    """
    Dependency function to parse form data and validate it
    against the ProcessEbooksForm model.
    """
    try:
        # --- MODIFIED: Only pass relevant fields explicitly ---
        ebook_form_data = {
            "media_type": "ebook", # Fixed
            "keep_original_file": False, # Fixed for this endpoint
            "urls": urls,
            "title": title,
            "author": author,
            "keywords": keywords, # Use alias mapping
            "custom_prompt": custom_prompt,
            "system_prompt": system_prompt,
            "overwrite_existing": overwrite_existing, # Keep for model validation if needed
            "perform_analysis": perform_analysis,
            "api_name": api_name,
            "api_key": api_key,
            "use_cookies": use_cookies, # Keep for model validation if needed
            "cookies": cookies, # Keep for model validation if needed
            "summarize_recursively": summarize_recursively,
            "perform_rolling_summarization": perform_rolling_summarization, # Keep for model validation
            # Chunking
            "perform_chunking": perform_chunking,
            "chunk_method": chunk_method,
            "chunk_language": chunk_language,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "custom_chapter_pattern": custom_chapter_pattern,
            # Ebook specific
            "extraction_method": extraction_method,

            # --- EXPLICITLY OMITTING irrelevant fields like: ---
            # "start_time": start_time,
            # "end_time": end_time,
            # "transcription_model": transcription_model, # DON'T PASS
            # "transcription_language": transcription_language, # DON'T PASS
            # "diarize": diarize, # DON'T PASS
            # "timestamp_option": timestamp_option, # DON'T PASS
            # "vad_use": vad_use, # DON'T PASS
            # "perform_confabulation_check_of_analysis": perform_confabulation_check_of_analysis, # DON'T PASS
            # "pdf_parsing_engine": pdf_parsing_engine, # DON'T PASS
            # "use_adaptive_chunking": use_adaptive_chunking, # Keep if needed by ChunkingOptions base
            # "use_multi_level_chunking": use_multi_level_chunking, # Keep if needed by ChunkingOptions base
        }

        # Filter out None values for optional fields if Pydantic requires non-None
        # (Might not be necessary if defaults are handled correctly, but safer)
        filtered_form_data = {k: v for k, v in ebook_form_data.items() if v is not None}
        # Ensure required fields are present even if None was filtered out (shouldn't happen for required ones)
        filtered_form_data["media_type"] = "ebook" # Re-add fixed fields
        filtered_form_data["keep_original_file"] = False

        form_instance = ProcessEbooksForm(**filtered_form_data)
        # ------------------------------------------------------
        return form_instance
    except ValidationError as e:
        # Keep existing detailed error handling
        serializable_errors = []
        for error in e.errors():
             serializable_error = error.copy()
             if 'ctx' in serializable_error and isinstance(serializable_error.get('ctx'), dict):
                 new_ctx = {}
                 for k, v in serializable_error['ctx'].items():
                     if isinstance(v, Exception): new_ctx[k] = str(v)
                     else: new_ctx[k] = v
                 serializable_error['ctx'] = new_ctx
             # Ensure 'input' exists for clarity, fallback to loc if missing
             serializable_error['input'] = serializable_error.get('input', serializable_error.get('loc'))
             serializable_errors.append(serializable_error)
        logger.warning(f"Pydantic validation failed for Ebook processing: {json.dumps(serializable_errors)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=serializable_errors,
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating ProcessEbooksForm: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during form processing: {type(e).__name__}"
        )


# ─────────────────────── Endpoint Implementation ────────────────
@router.post(
    "/process-ebooks",
    # status_code=status.HTTP_200_OK, # Determined dynamically
    summary="Extract, chunk, analyse EPUBs (NO DB Persistence)",
    tags=["Media Processing (No DB)"], # Separate tag maybe?
)
async def process_ebooks_endpoint(
    background_tasks: BackgroundTasks,
    # 1. Auth + UserID Determined through `get_db_by_user`
    # token: str = Header(None), # Use Header(None) for optional
    # 2. DB Dependency
    db: Database = Depends(get_db_for_user),
    # 3. Use Dependency Injection for Form Data
    form_data: ProcessEbooksForm = Depends(get_process_ebooks_form), # Use the dependency
    # 4. File uploads remain separate
    files: Optional[List[UploadFile]] = File(None, description="EPUB file uploads (.epub)"),
):
    """
    **Process Ebooks Endpoint (No Persistence)**

    Processes EPUB files (from uploaded files or URLs) by extracting content
    and metadata, optionally chunking, and optionally performing analysis
    (summarization). Returns the processing artifacts directly without saving
    to the database. Downloads URLs asynchronously. Accepts comma-separated keywords. Allows selection of extraction method.

    Supports `.epub` files.
    """
    logger.info("Request received for /process-ebooks (no persistence).")
    # Log form data safely (exclude sensitive fields)
    # Use .model_dump() for Pydantic v2
    logger.debug(f"Form data received: {form_data.model_dump(exclude={'api_key'})}")

    if form_data.urls and form_data.urls == ['']:
        logger.info("Received urls=[''], treating as no URLs provided for ebook processing.")
        form_data.urls = None # Or []

    _validate_inputs("ebook", form_data.urls, files)

    # --- Prepare result structure ---
    batch_result: Dict[str, Any] = {
        "processed_count": 0,
        "errors_count": 0,
        "errors": [],
        "results": []
    }
    # Map to track original ref -> temp path (still useful for context)
    source_map: Dict[str, str] = {}

    loop = asyncio.get_running_loop()
    temp_dir_manager = TempDirManager(cleanup=True) # Handles temp dir creation/cleanup

    local_paths_to_process: List[Tuple[str, Path]] = [] # (original_ref, local_path)

    # Use httpx.AsyncClient for concurrent downloads
    async with httpx.AsyncClient() as client:
        with temp_dir_manager as tmp_dir_path:
            temp_dir = Path(tmp_dir_path)
            logger.info(f"Using temporary directory: {temp_dir}")

            # --- Handle Uploads ---
            if files:
                saved_files, upload_errors = await _save_uploaded_files(
                    files,
                    temp_dir,
                    validator=file_validator_instance,
                    allowed_extensions=[".epub"]
                )
                # Add file saving/validation errors to batch_result
                for err_info in upload_errors:
                    # (Error handling for uploads remains the same as original)
                    err_detail = f"Upload error: {err_info['error']}"
                    batch_result["results"].append({
                        "status": "Error", "input_ref": err_info["original_filename"],
                        "error": err_detail, "media_type": "ebook",
                        "processing_source": None, "metadata": {}, "content": None, "chunks": None,
                        "analysis": None, "keywords": form_data.keywords, "warnings": None, # Use parsed keywords
                        "analysis_details": {}, "db_id": None, "db_message": "Processing only endpoint."
                    })
                    batch_result["errors_count"] += 1
                    batch_result["errors"].append(f"{err_info['original_filename']}: {err_detail}")

                for info in saved_files:
                    original_ref = info["original_filename"]
                    local_path = Path(info["path"])
                    local_paths_to_process.append((original_ref, local_path))
                    source_map[original_ref] = str(local_path)
                    logger.debug(f"Prepared uploaded file for processing: {original_ref} -> {local_path}")

            # --- Handle URLs (Asynchronously) ---
            if form_data.urls:
                logger.info(f"Attempting to download {len(form_data.urls)} URLs asynchronously...")
                download_tasks = [
                    _download_url_async(client, url, temp_dir, allowed_extensions = {".epub", ".pdf", ".mobi"})
                    for url in form_data.urls
                ]
                # Associate tasks with original URLs for error reporting
                url_task_map = {task: url for task, url in zip(download_tasks, form_data.urls)}

                # Gather results, return_exceptions=True prevents gather from stopping on first error
                download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

                for task, result in zip(download_tasks, download_results):
                    original_url = url_task_map[task] # Get URL associated with this task/result
                    if isinstance(result, Path):
                        # Success
                        downloaded_path = result
                        local_paths_to_process.append((original_url, downloaded_path))
                        source_map[original_url] = str(downloaded_path)
                        logger.debug(f"Prepared downloaded URL for processing: {original_url} -> {downloaded_path}")
                    elif isinstance(result, Exception):
                        # Failure
                        error = result
                        logger.error(f"Download or preparation failed for URL {original_url}: {error}", exc_info=False) # Log exception details separately if needed
                        err_detail = f"Download/preparation failed: {error}"
                        batch_result["results"].append({
                            "status": "Error", "input_ref": original_url, "error": err_detail,
                            "media_type": "ebook",
                            "processing_source": None, "metadata": {}, "content": None, "chunks": None,
                            "analysis": None, "keywords": form_data.keywords, "warnings": None, # Use parsed keywords
                            "analysis_details": {}, "db_id": None, "db_message": "Processing only endpoint."
                         })
                        batch_result["errors_count"] += 1
                        batch_result["errors"].append(f"{original_url}: {err_detail}")
                    else:
                         # Should not happen if _download_url_async returns Path or raises Exception
                         logger.error(f"Unexpected result type '{type(result)}' for URL download task: {original_url}")
                         err_detail = f"Unexpected download result type: {type(result).__name__}"
                         batch_result["results"].append({
                             "status": "Error", "input_ref": original_url, "error": err_detail,
                             "media_type": "ebook",
                             # Add default fields
                             "processing_source": None, "metadata": {}, "content": None, "chunks": None,
                             "analysis": None, "keywords": None, "warnings": None, "analysis_details": {},
                             "db_id": None, "db_message": "Processing only endpoint."
                         })
                         batch_result["errors_count"] += 1
                         batch_result["errors"].append(f"{original_url}: {err_detail}")


            # --- Check if any files are ready for processing ---
            if not local_paths_to_process:
                logger.warning("No valid EPUB sources found or prepared after handling uploads/URLs.")
                status_code = status.HTTP_207_MULTI_STATUS if batch_result["errors_count"] > 0 else status.HTTP_400_BAD_REQUEST
                return JSONResponse(status_code=status_code, content=batch_result)

            logger.info(f"Starting processing for {len(local_paths_to_process)} ebook(s).")

            # --- Prepare options for the worker ---
            chunk_options_dict = None
            if form_data.perform_chunking:
                 # Use form_data directly for chunk options
                 chunk_options_dict = {
                     'method': form_data.chunk_method, # Already defaults to 'chapter' in model
                     'max_size': form_data.chunk_size,
                     'overlap': form_data.chunk_overlap,
                     'language': form_data.chunk_language,
                     'custom_chapter_pattern': form_data.custom_chapter_pattern
                 }
                 chunk_options_dict = {k: v for k, v in chunk_options_dict.items() if v is not None}


            # --- Create and run processing tasks ---
            processing_tasks = []
            for original_ref, ebook_path in local_paths_to_process:
                partial_func = functools.partial(
                    _process_single_ebook, # Our sync helper
                    ebook_path=ebook_path,
                    original_ref=original_ref,
                    # Pass relevant options from form_data
                    title_override=form_data.title,
                    author_override=form_data.author,
                    keywords=form_data.keywords, # Pass the LIST validated by Pydantic
                    perform_chunking=form_data.perform_chunking,
                    chunk_options=chunk_options_dict,
                    perform_analysis=form_data.perform_analysis,
                    summarize_recursively=form_data.summarize_recursively,
                    api_name=form_data.api_name,
                    api_key=form_data.api_key,
                    custom_prompt=form_data.custom_prompt,
                    system_prompt=form_data.system_prompt,
                    # --- Pass the extraction method ---
                    extraction_method=form_data.extraction_method
                )
                processing_tasks.append(loop.run_in_executor(None, partial_func))

            # Gather results from processing tasks
            processing_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

    # --- Combine and Finalize Results (Outside temp dir and async client context) ---
    # (Result combination logic remains largely the same as original)
    for res in processing_results:
        if isinstance(res, dict):
            # Ensure mandatory fields and DB fields are null/default
            res["db_id"] = None
            res["db_message"] = "Processing only endpoint."
            res.setdefault("status", "Error") # Default if worker crashed badly
            res.setdefault("input_ref", "Unknown") # Should be set by worker
            res.setdefault("media_type", "ebook")
            res.setdefault("error", None)
            res.setdefault("warnings", None)
            res.setdefault("metadata", {})
            res.setdefault("content", None)
            res.setdefault("chunks", None)
            res.setdefault("analysis", None)
            res.setdefault("keywords", [])
            res.setdefault("analysis_details", {}) # Ensure exists

            batch_result["results"].append(res) # Add the processed/error dict

            # Update counts based on status
            if res["status"] == "Success" or res["status"] == "Warning":
                 batch_result["processed_count"] += 1
                 # Optionally add warnings to the main errors list or handle separately
                 if res["status"] == "Warning" and res.get("warnings"):
                     # Add warnings to the main list, prefixed by input ref?
                     for warn in res["warnings"]:
                          batch_result["errors"].append(f"{res.get('input_ref', 'Unknown')}: [Warning] {warn}")
                     # Don't increment errors_count for warnings
            else: # Status is Error
                 batch_result["errors_count"] += 1
                 error_msg = f"{res.get('input_ref', 'Unknown')}: {res.get('error', 'Unknown processing error')}"
                 if error_msg not in batch_result["errors"]: # Avoid duplicates if already added
                    batch_result["errors"].append(error_msg)

        elif isinstance(res, Exception): # Handle exceptions returned by asyncio.gather
             # Try to find original ref based on the exception context if possible (difficult)
             # For now, log and add a generic error
             logger.error(f"Task execution failed with exception: {res}", exc_info=res)
             error_detail = f"Task execution failed: {type(res).__name__}: {str(res)}"
             batch_result["results"].append({
                 "status": "Error", "input_ref": "Unknown Task", "error": error_detail,
                 "media_type": "ebook", "db_id": None, "db_message": "Processing only endpoint.",
                 "metadata": {}, "content": None, "chunks": None, "analysis": None,
                 "keywords": [], "warnings": None, "analysis_details": {},
             })
             batch_result["errors_count"] += 1
             if error_detail not in batch_result["errors"]:
                batch_result["errors"].append(error_detail)
        else: # Should not happen
             logger.error(f"Received unexpected result type from ebook worker task: {type(res)}")
             error_detail = "Invalid result type from ebook worker."
             batch_result["results"].append({
                 "status": "Error", "input_ref": "Unknown Task Type", "error": error_detail,
                 "media_type": "ebook", "db_id": None, "db_message": "Processing only endpoint.",
                 "metadata": {}, "content": None, "chunks": None, "analysis": None,
                 "keywords": [], "warnings": None, "analysis_details": {},
             })
             batch_result["errors_count"] += 1
             if error_detail not in batch_result["errors"]:
                 batch_result["errors"].append(error_detail)

    # --- Determine Final Status Code ---
    if batch_result["errors_count"] == 0 and batch_result["processed_count"] > 0:
        final_status_code = status.HTTP_200_OK
    elif batch_result["errors_count"] > 0 and batch_result["processed_count"] >= 0: # Allow 0 processed if all inputs failed
        # Includes cases: only input errors, only processing errors, mixed errors
        final_status_code = status.HTTP_207_MULTI_STATUS
    # Handle case where no inputs were valid / processed successfully or with error
    elif batch_result["processed_count"] == 0 and batch_result["errors_count"] == 0 and not local_paths_to_process:
         # This case should be caught earlier if no valid inputs were found
         final_status_code = status.HTTP_400_BAD_REQUEST # No valid input provided or prepared
    else: # Should ideally not be reached if logic above is sound
        logger.warning("Reached unexpected state for final status code determination.")
        final_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


    log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
    logger.log(log_level,
               f"/process-ebooks request finished with status {final_status_code}. "
               f"Processed: {batch_result['processed_count']}, Errors: {batch_result['errors_count']}")

    # --- Return Final Response ---
    return JSONResponse(status_code=final_status_code, content=batch_result)

#
# End of Ebook Processing Endpoint
#################################################################################################

######################## Document Processing Endpoint ###################################

# ─────────────────────── Form Model ─────────────────────────
class ProcessDocumentsForm(AddMediaForm):
    media_type: Literal["document"] = "document"
    keep_original_file: bool = False # Always cleanup tmp dir for this endpoint

    # Override chunking defaults if desired for documents
    perform_chunking: bool = True
    chunk_method: Optional[ChunkMethod] = Field('recursive', description="Default chunking method for documents")
    chunk_size: int = Field(1000, gt=0, description="Target chunk size for documents")
    chunk_overlap: int = Field(200, ge=0, description="Chunk overlap size for documents")

    # Note: No need for extraction_method specific to documents here

# ────────────────────── Dependency Function ───────────────────
def get_process_documents_form(
    # --- Inherited Fields from AddMediaForm ---
    # KEEP all Form(...) definitions to accept the data if sent by client
    urls: Optional[List[str]] = Form(None, description="List of URLs of the documents"),
    title: Optional[str] = Form(None, description="Optional title override"),
    author: Optional[str] = Form(None, description="Optional author override"),
    keywords: str = Form("", alias="keywords_str", description="Comma-separated keywords"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom prompt for analysis"),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt for analysis"),
    overwrite_existing: bool = Form(False), # Keep for model validation
    perform_analysis: bool = Form(True),
    api_name: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    use_cookies: bool = Form(False),
    cookies: Optional[str] = Form(None),
    summarize_recursively: bool = Form(False),
    perform_rolling_summarization: bool = Form(False), # Keep for model validation

    # --- Fields from ChunkingOptions ---
    perform_chunking: bool = Form(True), # Use default from ProcessDocumentsForm
    chunk_method: Optional[ChunkMethod] = Form('recursive'), # Use default from ProcessDocumentsForm
    chunk_language: Optional[str] = Form(None),
    chunk_size: int = Form(1000), # Use default from ProcessDocumentsForm
    chunk_overlap: int = Form(200), # Use default from ProcessDocumentsForm
    custom_chapter_pattern: Optional[str] = Form(None), # Less relevant but keep for model
    use_adaptive_chunking: bool = Form(False), # Keep for model validation
    use_multi_level_chunking: bool = Form(False), # Keep for model validation

    # --- Fields from other options (Audio/Video/PDF/Ebook) ---
    # KEEP Form() defs, but DON'T pass them explicitly to constructor below
    start_time: Optional[str] = Form(None), end_time: Optional[str] = Form(None),
    transcription_model: Optional[str] = Form(None), transcription_language: Optional[str] = Form(None),
    diarize: Optional[bool] = Form(None), timestamp_option: Optional[bool] = Form(None),
    vad_use: Optional[bool] = Form(None), perform_confabulation_check_of_analysis: Optional[bool] = Form(None),
    pdf_parsing_engine: Optional[Any] = Form(None), # Use Any if PdfEngine not imported/needed
    extraction_method: Optional[Any] = Form(None), # Keep placeholder

) -> ProcessDocumentsForm:
    """
    Dependency function to parse form data and validate it
    against the ProcessDocumentsForm model.
    """
    try:
        # Selectively create the data dict, omitting irrelevant fields
        doc_form_data = {
            "media_type": "document",
            "keep_original_file": False,
            "urls": urls,
            "title": title,
            "author": author,
            "keywords": keywords, # Pydantic handles alias mapping
            "custom_prompt": custom_prompt,
            "system_prompt": system_prompt,
            "overwrite_existing": overwrite_existing,
            "perform_analysis": perform_analysis,
            "api_name": api_name,
            "api_key": api_key,
            "use_cookies": use_cookies,
            "cookies": cookies,
            "summarize_recursively": summarize_recursively,
            "perform_rolling_summarization": perform_rolling_summarization,
            # Chunking
            "perform_chunking": perform_chunking,
            "chunk_method": chunk_method,
            "chunk_language": chunk_language,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "custom_chapter_pattern": custom_chapter_pattern,
            "use_adaptive_chunking": use_adaptive_chunking, # Keep if part of base ChunkingOptions
            "use_multi_level_chunking": use_multi_level_chunking, # Keep if part of base ChunkingOptions
            # Omit: start/end_time, transcription_*, diarize, timestamp_option, vad_use, pdf_*, ebook_*
        }

        # Filter out None values to allow Pydantic defaults to apply correctly
        filtered_form_data = {k: v for k, v in doc_form_data.items() if v is not None}
        # Re-add fixed fields that might have been filtered if None (shouldn't be)
        filtered_form_data["media_type"] = "document"
        filtered_form_data["keep_original_file"] = False

        form_instance = ProcessDocumentsForm(**filtered_form_data)
        return form_instance
    except ValidationError as e:
        # Use the detailed error handling from previous examples
        serializable_errors = []
        for error in e.errors():
             serializable_error = error.copy()
             # ... (copy the detailed error serialization logic here) ...
             if 'ctx' in serializable_error and isinstance(serializable_error.get('ctx'), dict):
                 new_ctx = {}
                 for k, v in serializable_error['ctx'].items():
                     if isinstance(v, Exception): new_ctx[k] = str(v)
                     else: new_ctx[k] = v
                 serializable_error['ctx'] = new_ctx
             serializable_error['input'] = serializable_error.get('input', serializable_error.get('loc'))
             serializable_errors.append(serializable_error)
        logger.warning(f"Pydantic validation failed for Document processing: {json.dumps(serializable_errors)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=serializable_errors,
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating ProcessDocumentsForm: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during form processing: {type(e).__name__}"
        )


# ─────────────────────── Endpoint Implementation ────────────────
@router.post(
    "/process-documents",
    # status_code=status.HTTP_200_OK, # Determined dynamically
    summary="Extract, chunk, analyse Documents (NO DB Persistence)",
    tags=["Media Processing (No DB)"],
    response_model=Dict[str, Any], # Define a response model if desired
)
async def process_documents_endpoint(
    # background_tasks: BackgroundTasks, # Remove if unused
    # 1. Auth + UserID Determined through `get_db_by_user`
    # token: str = Header(None), # Use Header(None) for optional
    # 2. DB Dependency
    db: Database = Depends(get_db_for_user),
    # 3. Form Data Dependency
    form_data: ProcessDocumentsForm = Depends(get_process_documents_form), # Use the dependency
    # 4. File Upload
    files: Optional[List[UploadFile]] = File(None, description="Document file uploads (.txt, .md, .docx, .rtf, .html, .xml)"),
):
    """
    **Process Documents Endpoint (No Persistence)**

    Processes document files (from uploaded files or URLs) by extracting content,
    optionally chunking, and optionally performing analysis.
    Returns the processing artifacts directly without saving to the database.

    Supports `.txt`, `.md`, `.docx`, `.rtf`, `.html`, `.htm`, `.xml`. Requires `pandoc` for `.rtf`.
    """
    logger.info("Request received for /process-documents (no persistence).")
    logger.debug(f"Form data received: {form_data.model_dump(exclude={'api_key'})}") # Use model_dump for Pydantic v2+

    # Define allowed extensions for this endpoint
    # Make sure these match what convert_document_to_text supports
    ALLOWED_DOC_EXTENSIONS = [".txt", ".md", ".docx", ".rtf", ".html", ".htm", ".xml"] # Add others if supported

    _validate_inputs("document", form_data.urls, files)

    # --- Prepare result structure ---
    batch_result: Dict[str, Any] = {
        "processed_count": 0,
        "errors_count": 0,
        "errors": [],
        "results": []
    }
    # Map to track original ref -> temp path
    source_map: Dict[str, Path] = {} # Store Path objects

    loop = asyncio.get_running_loop()
    # Use TempDirManager for reliable cleanup
    with TempDirManager(cleanup=(not form_data.keep_original_file), prefix="process_doc_") as temp_dir_path:
        temp_dir = Path(temp_dir_path)
        logger.info(f"Using temporary directory: {temp_dir}")

        local_paths_to_process: List[Tuple[str, Path]] = [] # (original_ref, local_path)

        # --- Handle Uploads ---
        if files:
            # Use specific allowed extensions for documents
            saved_files, upload_errors = await _save_uploaded_files(
                files,
                temp_dir,
                validator=file_validator_instance,
                allowed_extensions=ALLOWED_DOC_EXTENSIONS
            )
            # Add file saving/validation errors to batch_result
            for err_info in upload_errors:
                original_filename = err_info.get("input") or err_info.get("original_filename", "Unknown Upload")
                err_detail = f"Upload error: {err_info['error']}"
                batch_result["results"].append({
                    "status": "Error", "input_ref": original_filename,
                    "error": err_detail, "media_type": "document",
                    "processing_source": None, "metadata": {}, "content": None, "chunks": None,
                    "analysis": None, "keywords": form_data.keywords, "warnings": None,
                    "analysis_details": {}, "db_id": None, "db_message": "Processing only endpoint.",
                    "segments": None # Ensure all expected fields are present
                })
                batch_result["errors_count"] += 1
                batch_result["errors"].append(f"{original_filename}: {err_detail}")

            for info in saved_files:
                original_ref = info["original_filename"]
                local_path = Path(info["path"])
                local_paths_to_process.append((original_ref, local_path))
                source_map[original_ref] = local_path
                logger.debug(f"Prepared uploaded file for processing: {original_ref} -> {local_path}")

        # --- Handle URLs (Asynchronously) ---
        if form_data.urls:
            logger.info(f"Attempting to download {len(form_data.urls)} URLs asynchronously...")
            download_tasks = []  # Initialize outside the client block
            url_task_map = {}  # Initialize outside the client block

            # --- MODIFICATION: Create client first ---
            async with httpx.AsyncClient() as client:
                # --- MODIFICATION: Create tasks *inside* the client block ---
                allowed_ext_set = set(ALLOWED_DOC_EXTENSIONS)  # Convert to set once
                download_tasks = [
                    # Pass the client instance here
                    _download_url_async(
                        client=client,  # Pass the active client
                        url=url,
                        target_dir=temp_dir,
                        allowed_extensions=allowed_ext_set,  # Pass the set
                        check_extension=True  # Perform the check
                    )
                    for url in form_data.urls
                ]
                # --------------------------------------------------------

                # Create the map *after* tasks are created
                url_task_map = {task: url for task, url in zip(download_tasks, form_data.urls)}

                # Gather results (can stay inside or move just outside client block)
                # Keeping it inside is fine.
                if download_tasks:  # Only gather if there are tasks
                    download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
                else:
                    download_results = []  # No tasks to gather
            # --- End MODIFICATION ---

            # Process results (this loop remains largely the same)
            # Ensure download_tasks and download_results align if gather was conditional
            if download_tasks:  # Check if tasks were created/gathered
                for task, result in zip(download_tasks, download_results):
                    # Get original_url using the pre-built map
                    original_url = url_task_map.get(task, "Unknown URL")  # Use .get for safety

                    if isinstance(result, Path):
                        downloaded_path = result
                        local_paths_to_process.append((original_url, downloaded_path))
                        source_map[original_url] = downloaded_path  # Use original_url as key
                        logger.debug(f"Prepared downloaded URL for processing: {original_url} -> {downloaded_path}")
                    elif isinstance(result, Exception):
                        error = result
                        logger.error(f"Download or preparation failed for URL {original_url}: {error}", exc_info=False)
                        # Use the specific error message from the exception
                        err_detail = f"Download/preparation failed: {str(error)}"
                        batch_result["results"].append({
                            "status": "Error", "input_ref": original_url, "error": err_detail,
                            "media_type": "document",
                            "processing_source": None, "metadata": {}, "content": None, "chunks": None,
                            "analysis": None, "keywords": form_data.keywords, "warnings": None,
                            "analysis_details": {}, "db_id": None, "db_message": "Processing only endpoint.",
                            "segments": None
                        })
                        batch_result["errors_count"] += 1
                        batch_result["errors"].append(f"{original_url}: {err_detail}")
                    else:
                        logger.error(f"Unexpected result type '{type(result)}' for URL download task: {original_url}")
                        err_detail = f"Unexpected download result type: {type(result).__name__}"
                        batch_result["results"].append({
                            "status": "Error", "input_ref": original_url, "error": err_detail,
                            "media_type": "document",
                            "processing_source": None, "metadata": {}, "content": None, "chunks": None,
                            "analysis": None, "keywords": form_data.keywords, "warnings": None,
                            "analysis_details": {}, "db_id": None, "db_message": "Processing only endpoint.",
                            "segments": None
                        })
                        batch_result["errors_count"] += 1
                        batch_result["errors"].append(f"{original_url}: {err_detail}")


        # --- Check if any files are ready for processing ---
        if not local_paths_to_process:
            logger.warning("No valid document sources found or prepared after handling uploads/URLs.")
            status_code = status.HTTP_207_MULTI_STATUS if batch_result["errors_count"] > 0 else status.HTTP_400_BAD_REQUEST
            # Ensure results already added are returned
            return JSONResponse(status_code=status_code, content=batch_result)

        logger.info(f"Starting processing for {len(local_paths_to_process)} document(s).")

        # --- Prepare options for the worker ---
        # Use helper or form_data directly
        chunk_options_dict = _prepare_chunking_options_dict(form_data) if form_data.perform_chunking else None

        # --- Create and run processing tasks ---
        processing_tasks = []
        for original_ref, doc_path in local_paths_to_process:
            partial_func = functools.partial(
                process_document_content,
                doc_path=doc_path,
                # Pass relevant options from form_data
                perform_chunking=form_data.perform_chunking,
                chunk_options=chunk_options_dict,
                perform_analysis=form_data.perform_analysis,
                summarize_recursively=form_data.summarize_recursively,
                api_name=form_data.api_name,
                api_key=form_data.api_key,
                custom_prompt=form_data.custom_prompt,
                system_prompt=form_data.system_prompt,
                title_override=form_data.title,
                author_override=form_data.author,
                keywords=form_data.keywords, # Pass the LIST validated by Pydantic
            )
            processing_tasks.append(loop.run_in_executor(None, partial_func))

        # Gather results from processing tasks
        task_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

    # --- Combine and Finalize Results (Outside temp dir context) ---
    # Logic similar to ebook endpoint
    for i, res in enumerate(task_results):
        original_ref = local_paths_to_process[i][0] # Get corresponding original ref

        if isinstance(res, dict):
            # Ensure mandatory fields and DB fields are null/default
            res["input_ref"] = original_ref # Set input_ref to original URL/filename
            res["db_id"] = None
            res["db_message"] = "Processing only endpoint."
            res.setdefault("status", "Error")
            res.setdefault("media_type", "document")
            res.setdefault("error", None)
            res.setdefault("warnings", None)
            res.setdefault("metadata", {})
            res.setdefault("content", None)
            res.setdefault("chunks", None)
            res.setdefault("analysis", None)
            res.setdefault("keywords", [])
            res.setdefault("analysis_details", {})
            res.setdefault("segments", None) # Ensure segments field exists

            batch_result["results"].append(res) # Add the processed/error dict

            # Update counts based on status
            if res["status"] in ["Success", "Warning"]:
                 batch_result["processed_count"] += 1
                 if res["status"] == "Warning" and res.get("warnings"):
                     for warn in res["warnings"]:
                          batch_result["errors"].append(f"{original_ref}: [Warning] {warn}")
                     # Don't increment errors_count for warnings
            else: # Status is Error
                 batch_result["errors_count"] += 1
                 error_msg = f"{original_ref}: {res.get('error', 'Unknown processing error')}"
                 if error_msg not in batch_result["errors"]:
                    batch_result["errors"].append(error_msg)

        elif isinstance(res, Exception): # Handle exceptions returned by asyncio.gather
             logger.error(f"Task execution failed for {original_ref} with exception: {res}", exc_info=res)
             error_detail = f"Task execution failed: {type(res).__name__}: {str(res)}"
             batch_result["results"].append({
                 "status": "Error", "input_ref": original_ref, "error": error_detail,
                 "media_type": "document", "db_id": None, "db_message": "Processing only endpoint.",
                 "processing_source": str(local_paths_to_process[i][1]), # Include path if possible
                 "metadata": {}, "content": None, "chunks": None, "analysis": None,
                 "keywords": form_data.keywords, "warnings": None, "analysis_details": {}, "segments": None,
             })
             batch_result["errors_count"] += 1
             if error_detail not in batch_result["errors"]:
                batch_result["errors"].append(f"{original_ref}: {error_detail}")
        else: # Should not happen
             logger.error(f"Received unexpected result type from document worker task for {original_ref}: {type(res)}")
             error_detail = "Invalid result type from document worker."
             batch_result["results"].append({
                 "status": "Error", "input_ref": original_ref, "error": error_detail,
                 "media_type": "document", "db_id": None, "db_message": "Processing only endpoint.",
                 "processing_source": str(local_paths_to_process[i][1]),
                 "metadata": {}, "content": None, "chunks": None, "analysis": None,
                 "keywords": form_data.keywords, "warnings": None, "analysis_details": {}, "segments": None,
             })
             batch_result["errors_count"] += 1
             if error_detail not in batch_result["errors"]:
                 batch_result["errors"].append(f"{original_ref}: {error_detail}")

    # --- Determine Final Status Code ---
    # (Same logic as ebook endpoint)
    if batch_result["errors_count"] == 0 and batch_result["processed_count"] > 0:
        final_status_code = status.HTTP_200_OK
    elif batch_result["errors_count"] > 0: # Includes partial success/warnings and all errors
        final_status_code = status.HTTP_207_MULTI_STATUS
    elif batch_result["processed_count"] == 0 and batch_result["errors_count"] == 0:
         # This case means no valid inputs were processed or resulted in error state
         # Could happen if only upload errors occurred before processing started
         # Check if results list is non-empty (contains only upload errors)
         if batch_result["results"]:
              final_status_code = status.HTTP_207_MULTI_STATUS # Had only input errors
         else:
              final_status_code = status.HTTP_400_BAD_REQUEST # No valid input provided or prepared
    else:
        logger.warning("Reached unexpected state for final status code determination.")
        final_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
    logger.log(log_level,
               f"/process-documents request finished with status {final_status_code}. "
               f"Processed: {batch_result['processed_count']}, Errors: {batch_result['errors_count']}")

    # --- Return Final Response ---
    return JSONResponse(status_code=final_status_code, content=batch_result)

#
# End of Document Processing Endpoint
############################################################################################


######################## PDF Processing Endpoint ###################################
# Endpoints:
#

# ─────────────────────── form model (subset of AddMediaForm) ─────────────────
class ProcessPDFsForm(AddMediaForm):
    media_type: Literal["pdf"] = "pdf"
    keep_original_file: bool = False


def get_process_pdfs_form(
    # Include ALL fields defined in AddMediaForm and its parents
    # Use Form(...) for each
    urls: Optional[List[str]] = Form(None, description="List of URLs of the PDF items"),
    title: Optional[str] = Form(None, description="Optional title (applied if only one item processed)"),
    author: Optional[str] = Form(None, description="Optional author (applied similarly to title)"),
    keywords: str = Form("", alias="keywords", description="Comma-separated keywords"), # Use alias
    custom_prompt: Optional[str] = Form(None, description="Optional custom prompt"),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt"),
    overwrite_existing: bool = Form(False, description="Overwrite existing media (Not used, for model)"),
    keep_original_file: bool = Form(False, description="Retain original files (fixed in model)"), # Fixed by ProcessPDFsForm
    perform_analysis: bool = Form(True, description="Perform analysis"),
    api_name: Optional[str] = Form(None, description="Optional API name"), # Keep this
    api_key: Optional[str] = Form(None, description="Optional API key"),    # Keep this
    use_cookies: bool = Form(False, description="Use cookies for URL download requests"),
    cookies: Optional[str] = Form(None, description="Cookie string if `use_cookies` is True"),
    summarize_recursively: bool = Form(False, description="Perform recursive summarization"),
    perform_rolling_summarization: bool = Form(False, description="Perform rolling summarization"), # From AddMediaForm

    # --- Fields from PdfOptions ---
    pdf_parsing_engine: Optional[PdfEngine] = Form("pymupdf4llm", description="PDF parsing engine"),
    custom_chapter_pattern: Optional[str] = Form(None, description="Regex pattern for custom chapter splitting"),

    # --- Fields from ChunkingOptions ---
    perform_chunking: bool = Form(True, description="Enable chunking"),
    chunk_method: Optional[ChunkMethod] = Form(None, description="Chunking method"),
    use_adaptive_chunking: bool = Form(False, description="Enable adaptive chunking"),
    use_multi_level_chunking: bool = Form(False, description="Enable multi-level chunking"),
    chunk_language: Optional[str] = Form(None, description="Chunking language override"),
    chunk_size: int = Form(500, description="Target chunk size"),
    chunk_overlap: int = Form(200, description="Chunk overlap size"),

    # --- Fields from AudioVideoOptions (might be needed for AddMediaForm validation/defaults) ---
    start_time: Optional[str] = Form(None, description="Optional start time (HH:MM:SS or seconds)"),
    end_time: Optional[str] = Form(None, description="Optional end time (HH:MM:SS or seconds)"),
    transcription_model: str = Form("deepdml/faster-distil-whisper-large-v3.5", description="Transcription model"), # Get default from AddMediaForm if possible
    transcription_language: str = Form("en", description="Transcription language"),
    diarize: bool = Form(False, description="Enable speaker diarization"),
    timestamp_option: bool = Form(True, description="Include timestamps in transcription"),
    vad_use: bool = Form(False, description="Enable VAD filter"),
    perform_confabulation_check_of_analysis: bool = Form(False, description="Enable confabulation check"),

) -> ProcessPDFsForm:
    """
    Dependency function to parse form data and validate it
    against the ProcessPDFsForm model.
    """
    try:
        # Create the Pydantic model instance using the parsed form data.
        form_instance = ProcessPDFsForm(
            # --- Map all the parameters received by this function ---
            media_type="pdf", # Fixed by ProcessPDFsForm
            urls=urls,
            title=title,
            author=author,
            keywords=keywords, # Pydantic handles mapping this to keywords_str via alias
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            overwrite_existing=overwrite_existing,
            keep_original_file=keep_original_file, # Use arg
            perform_analysis=perform_analysis,
            api_name=api_name,   # Pass received arg
            api_key=api_key,     # Pass received arg
            use_cookies=use_cookies,
            cookies=cookies,
            summarize_recursively=summarize_recursively,
            perform_rolling_summarization=perform_rolling_summarization,
            pdf_parsing_engine=pdf_parsing_engine,
            custom_chapter_pattern=custom_chapter_pattern,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_time=start_time,
            end_time=end_time,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            diarize=diarize,
            timestamp_option=timestamp_option,
            vad_use=vad_use,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
        )
        return form_instance
    # --- Keep the exact same error handling as get_process_videos_form ---
    except ValidationError as e:
        serializable_errors = []
        for error in e.errors():
             serializable_error = error.copy()
             if 'ctx' in serializable_error and isinstance(serializable_error.get('ctx'), dict):
                 new_ctx = {}
                 for k, v in serializable_error['ctx'].items():
                     if isinstance(v, Exception):
                         new_ctx[k] = str(v)
                     else:
                         new_ctx[k] = v
                 serializable_error['ctx'] = new_ctx
             serializable_errors.append(serializable_error)
        logger.warning(f"Pydantic validation failed for PDF processing: {json.dumps(serializable_errors)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=serializable_errors,
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating ProcessPDFsForm: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during form processing: {type(e).__name__}"
        )

async def _single_pdf_worker(
    pdf_path: Path,
    form,                      # ProcessPDFsForm instance
    chunk_opts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    1) Read file bytes, 2) call process_pdf_task(), 3) normalise the result dict.
    """
    try:
        file_bytes = pdf_path.read_bytes()

        pdf_kwargs = {
            "file_bytes": file_bytes,
            "filename": pdf_path.name,
            "parser": form.pdf_parsing_engine,
            "custom_prompt": form.custom_prompt,
            "system_prompt": form.system_prompt,
            "api_name": form.api_name if form.perform_analysis else None,
            "api_key": form.api_key,
            "perform_analysis": form.perform_analysis,
            "keywords": form.keywords,
            "perform_chunking": form.perform_chunking and form.perform_analysis,
            "chunk_method":  chunk_opts["method"]      if form.perform_analysis else None,
            "max_chunk_size": chunk_opts["max_size"]   if form.perform_analysis else None,
            "chunk_overlap":  chunk_opts["overlap"]    if form.perform_analysis else None,
        }

        # process_pdf_task is async
        raw = await process_pdf_task(**pdf_kwargs)

        # Ensure minimal envelope consistency
        if isinstance(raw, dict):
            raw.setdefault("status", "Success")
            raw.setdefault("input", str(pdf_path))
            return raw
        else:
            return {"input": str(pdf_path), "status": "Error",
                    "error": f"Unexpected return type: {type(raw).__name__}"}

    except Exception as e:
        logging.error(f"PDF worker failed for {pdf_path}: {e}", exc_info=True)
        return {"input": str(pdf_path), "status": "Error", "error": str(e)}

def normalise_pdf_result(item: dict, original_ref: str) -> dict:
    """Ensure every required key is present and correctly typed for PDF results."""
    # Ensure base keys are present
    item.setdefault("status", "Error") # Default to Error if not set
    item["input_ref"] = original_ref   # Use the passed original ref
    # Add processing_source if missing, default to original ref
    item.setdefault("processing_source", original_ref)
    item.setdefault("media_type", "pdf")

    # Ensure metadata is a dict (can be empty)
    item["metadata"] = item.get("metadata") or {}
    if not isinstance(item["metadata"], dict):
        logger.warning(f"Normalizing non-dict metadata for {original_ref}: {item['metadata']}")
        item["metadata"] = {"original_metadata": item["metadata"]} # Wrap non-dict metadata

    # Keys that can be None
    item.setdefault("content", None)
    item.setdefault("chunks", None)
    item.setdefault("analysis", None)
    item.setdefault("warnings", None)
    item.setdefault("error", None)
    item.setdefault("segments", None) # Add segments default

    # Analysis details should be a dict
    item["analysis_details"] = item.get("analysis_details") or {}
    if not isinstance(item["analysis_details"], dict):
         logger.warning(f"Normalizing non-dict analysis_details for {original_ref}: {item['analysis_details']}")
         item["analysis_details"] = {"original_details": item["analysis_details"]}

    # Ensure keywords is a list (can be empty) - Use metadata keywords if present
    item.setdefault("keywords", item.get("metadata", {}).get("keywords"))
    if item["keywords"] is None:
        item["keywords"] = []
    elif not isinstance(item["keywords"], list):
        logger.warning(f"Normalizing non-list keywords for {original_ref}: {item['keywords']}")
        # Attempt to split if it's a comma-separated string, else wrap in list
        if isinstance(item["keywords"], str):
            item["keywords"] = [k.strip() for k in item["keywords"].split(',') if k.strip()]
        else:
            item["keywords"] = [str(item["keywords"])]


    # No persistence on this endpoint
    item["db_id"] = None
    item["db_message"] = "Processing only endpoint."

    return item

# ───────────────────────────── endpoint ──────────────────────────────────────
@router.post(
    "/process-pdfs",
    # status_code=status.HTTP_200_OK, # Determined dynamically
    summary="Extract, chunk, analyse PDFs (NO DB Persistence)",
    tags=["Media Processing (No DB)"],
)
async def process_pdfs_endpoint(
    background_tasks: BackgroundTasks,
    # 1. Auth + UserID Determined through `get_db_by_user`
    # token: str = Header(None), # Use Header(None) for optional
    # 2. DB Dependency
    db: Database = Depends(get_db_for_user),
    form_data: ProcessPDFsForm = Depends(get_process_pdfs_form),
    files: Optional[List[UploadFile]] = File(None,  description="PDF uploads"),
):
    """Process PDFs (No Persistence)"""
    logger.info("Request received for /process-pdfs (no persistence).")
    ALLOWED_PDF_EXTENSIONS = ['.pdf']
    _validate_inputs("pdf", form_data.urls, files)

    batch_result: Dict[str, Any] = {
        "processed_count": 0,
        "errors_count": 0,
        "errors": [],
        "results": []
    }

    loop = asyncio.get_running_loop()
    temp_dir_manager = TempDirManager(cleanup=True)

    # We need bytes for process_pdf_task, so handle uploads/downloads differently
    pdf_inputs_to_process: List[Tuple[str, bytes]] = [] # (original_ref, file_bytes)
    # Let's store tasks along with their original refs
    tasks_with_refs: List[Tuple[str, asyncio.Task]] = []
    source_to_ref_map = {} # original_ref -> temp_path (if created) or URL
    file_errors = []
    file_errors_encountered = False # Track if any file/download errors happened

    with temp_dir_manager as temp_dir: # Temp dir needed only if downloading URLs first
        # Handle Uploads (read bytes directly)
        if files:
            saved_files, upload_errors = await _save_uploaded_files(
                files,
                temp_dir=Path(temp_dir),  # Need Path object
                validator=file_validator_instance,
                allowed_extensions=ALLOWED_PDF_EXTENSIONS
            )

            for err_info in upload_errors:
                file_errors_encountered = True
                original_filename = err_info.get("original_filename") or err_info.get("input", "Unknown Upload")
                error_detail = f"Upload error: {err_info['error']}"
                # Add formatted error to results
                batch_result["results"].append(normalise_pdf_result({
                    "status": "Error", "error": error_detail, "processing_source": original_filename
                }, original_ref=original_filename))  # Use normalise for consistency
                batch_result["errors_count"] += 1
                batch_result["errors"].append(f"{original_filename}: {error_detail}")

            # Now read bytes for successfully saved files
            for info in saved_files:
                original_ref = info["original_filename"]
                local_path = Path(info["path"])
                try:
                    file_bytes = local_path.read_bytes()
                    pdf_inputs_to_process.append((original_ref, file_bytes))
                except Exception as read_err:
                    logger.error(f"Failed to read prepared PDF file {original_ref} from {local_path}: {read_err}")
                    file_errors_encountered = True
                    error_detail = f"Failed to read prepared file: {read_err}"
                    batch_result["results"].append(normalise_pdf_result({
                        "status": "Error", "error": error_detail, "processing_source": original_ref
                    }, original_ref=original_ref))
                    batch_result["errors_count"] += 1
                    batch_result["errors"].append(f"{original_ref}: {error_detail}")

        # Handle URLs (download bytes)
        if form_data.urls:
             async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                 download_tasks = []
                 for url in form_data.urls:
                     download_tasks.append(client.get(url)) # Gather response futures

                 download_responses = await asyncio.gather(*download_tasks, return_exceptions=True)

                 for i, res in enumerate(download_responses):
                     url = form_data.urls[i]
                     if isinstance(res, httpx.Response):
                         try:
                             res.raise_for_status()
                             content_type = res.headers.get('content-type', '').lower()
                             # Basic check for PDF content type or .pdf extension in final URL
                             final_url_path = Path(str(res.url)).name # Get filename from final redirected URL
                             if 'application/pdf' in content_type or final_url_path.endswith('.pdf'):
                                 file_bytes = res.content
                                 pdf_inputs_to_process.append((url, file_bytes)) # Use original URL as ref
                             else:
                                 raise ValueError(f"Downloaded content from {url} is not a PDF (Content-Type: {content_type}, Final URL: {res.url})")
                         except httpx.HTTPStatusError as status_err:
                             logger.error(f"HTTP error downloading {url}: {status_err}")
                             error_detail = f"Download failed (HTTP {status_err.response.status_code})"
                             batch_result["results"].append({
                                 "status": "Error", "input_ref": url,
                                 "processing_source": url,
                                 "error": error_detail,
                                 "media_type": "pdf", "db_id": None, "db_message": "Processing only endpoint.",
                                 "metadata": {},  # <-- CHANGED
                                 "content": None, "chunks": None,
                                 "analysis": None, "keywords": None, "warnings": None,
                                 "analysis_details": {}
                             })
                             batch_result["errors_count"] += 1
                             batch_result["errors"].append(error_detail)
                         except Exception as dl_err:
                             logger.error(f"Error processing download for {url}: {dl_err}")
                             error_detail = f"Download processing error: {dl_err}"
                             batch_result["results"].append({
                                 "status": "Error", "input_ref": url,
                                 "processing_source": url,
                                 "error": error_detail,
                                 "media_type": "pdf", "db_id": None, "db_message": "Processing only endpoint.",
                                 "metadata": {},  # <-- CHANGED
                                 "content": None, "chunks": None,
                                 "analysis": None, "keywords": None, "warnings": None,
                                 "analysis_details": {}
                             })
                             batch_result["errors_count"] += 1
                             batch_result["errors"].append(error_detail)
                     else:  # Handle exceptions during download (gather returned an exception)
                         logger.error(f"Download failed for {url}: {res}", exc_info=isinstance(res, Exception))
                         error_detail = f"Download failed: {res}"
                         batch_result["results"].append({
                             "status": "Error", "input_ref": url,
                             "processing_source": url,
                             "error": error_detail,
                             "media_type": "pdf", "db_id": None, "db_message": "Processing only endpoint.",
                             "metadata": {}, # <-- CHANGED
                             "content": None, "chunks": None,
                             "analysis": None, "keywords": None, "warnings": None,
                             "analysis_details": {}
                         })
                         batch_result["errors_count"] += 1
                         batch_result["errors"].append(error_detail)

        if not pdf_inputs_to_process:
            # Determine status based on whether *any* errors occurred during input handling
            status_code = status.HTTP_207_MULTI_STATUS if batch_result["errors_count"] > 0 else status.HTTP_400_BAD_REQUEST
            return JSONResponse(status_code=status_code, content=batch_result)

        logger.debug(f"ENDPOINT: #1 Passing to task -> api_name='{form_data.api_name}', api_key='{form_data.api_key}'")
        # --- Call process_pdf_task for each input ---
        tasks = []
        for original_ref, file_bytes in pdf_inputs_to_process:
             # --- Pass chunk options correctly ---
             chunk_opts_for_task = {
                 'method': form_data.chunk_method if form_data.chunk_method else 'sentences', # Use enum value or default
                 'max_size': form_data.chunk_size,
                 'overlap': form_data.chunk_overlap
             }
             logger.debug(
                 f"ENDPOINT: #2 Passing to task -> api_name='{form_data.api_name}', api_key='{form_data.api_key}'")
             # Create the async task
             task = asyncio.create_task(
                 process_pdf_task(
                     file_bytes=file_bytes,
                     filename=original_ref, # Use original ref as filename hint
                     parser=str(form_data.pdf_parsing_engine) or "pymupdf4llm",
                     # Pass options from form
                     title_override=form_data.title,
                     author_override=form_data.author,
                     keywords=form_data.keywords, # Pass list
                     perform_chunking=form_data.perform_chunking or None,
                     # Pass individual chunk params from form model
                     chunk_method=chunk_opts_for_task['method'],
                     max_chunk_size=chunk_opts_for_task['max_size'],
                     chunk_overlap=chunk_opts_for_task['overlap'],
                     perform_analysis=form_data.perform_analysis,
                     api_name=form_data.api_name,
                     api_key=form_data.api_key,
                     custom_prompt=form_data.custom_prompt,
                     system_prompt=form_data.system_prompt,
                     summarize_recursively=form_data.summarize_recursively,
                 )
             )
             tasks_with_refs.append((original_ref, task))

        # Gather results from processing tasks
        gathered_results = await asyncio.gather(*[task for _, task in tasks_with_refs], return_exceptions=True)

        # Add processing results, ensuring no DB fields
        for i, (original_ref, _) in enumerate(tasks_with_refs):
            res = gathered_results[i]  # Get corresponding result/exception

            if isinstance(res, dict):
                # Normalize the result dictionary using the correct original_ref
                normalized_res = normalise_pdf_result(res, original_ref=original_ref)
                batch_result["results"].append(normalized_res)

                # Update counts based on normalized status
                if normalized_res["status"] in ["Success", "Warning"]:
                    batch_result["processed_count"] += 1
                    if normalized_res["status"] == "Warning" and normalized_res.get("warnings"):
                        for warn in normalized_res["warnings"]:
                            batch_result["errors"].append(f"{original_ref}: [Warning] {warn}")
                else:  # Status is Error
                    batch_result["errors_count"] += 1
                    error_msg = f"{original_ref}: {normalized_res.get('error', 'Unknown processing error')}"
                    if error_msg not in batch_result["errors"]:
                        batch_result["errors"].append(error_msg)

            elif isinstance(res, Exception):  # Handle exceptions returned by gather
                logger.error(f"PDF processing task for {original_ref} failed with exception: {res}", exc_info=res)
                error_detail = f"Task execution failed: {type(res).__name__}: {str(res)}"
                # Normalize the error result
                normalized_err = normalise_pdf_result({
                    "status": "Error", "error": error_detail, "processing_source": original_ref
                }, original_ref=original_ref)
                batch_result["results"].append(normalized_err)
                batch_result["errors_count"] += 1
                error_msg = f"{original_ref}: {error_detail}"
                if error_msg not in batch_result["errors"]:
                    batch_result["errors"].append(error_msg)
            else:  # Should not happen
                logger.error(f"Received unexpected result type from PDF worker task for {original_ref}: {type(res)}")
                error_detail = "Invalid result type from PDF worker."
                normalized_err = normalise_pdf_result({
                    "status": "Error", "error": error_detail, "processing_source": original_ref
                }, original_ref=original_ref)
                batch_result["results"].append(normalized_err)
                batch_result["errors_count"] += 1
                error_msg = f"{original_ref}: {error_detail}"
                if error_msg not in batch_result["errors"]:
                    batch_result["errors"].append(error_msg)

    # --- Determine Final Status Code & Return ---
    final_processed_count = sum(1 for r in batch_result["results"] if r.get("status") == "Success")
    final_error_count = sum(1 for r in batch_result["results"] if r.get("status") == "Error")
    batch_result["processed_count"] = final_processed_count
    batch_result["errors_count"] = final_error_count
    # Update errors list to avoid duplicates (optional)
    unique_errors = list(set(str(e) for e in batch_result["errors"] if e))
    batch_result["errors"] = unique_errors

    if batch_result["errors_count"] == 0 and batch_result["processed_count"] > 0:
        final_status_code = status.HTTP_200_OK
    elif batch_result.get("results"): # Any result (success, warning, error) -> 207
        final_status_code = status.HTTP_207_MULTI_STATUS
    else: # No results -> likely means no valid input provided
        final_status_code = status.HTTP_400_BAD_REQUEST

    log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
    logger.log(log_level,
               f"/process-pdfs request finished with status {final_status_code}. "
               f"Results: {len(batch_result['results'])}, Processed: {batch_result['processed_count']}, Errors: {batch_result['errors_count']}")

    return JSONResponse(status_code=final_status_code, content=batch_result)

#
# End of PDF Processing Endpoint
############################################################################################


######################## XML Processing Endpoint ###################################
# Endpoints:
# FIXME

#XML File handling
# /Server_API/app/api/v1/endpoints/media.py

class XMLIngestRequest(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[List[str]] = []
    system_prompt: Optional[str] = None
    custom_prompt: Optional[str] = None
    auto_summarize: bool = False
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    mode: str = "persist"  # or "ephemeral"

# @router.post("/process-xml")
# async def process_xml_endpoint(
#     payload: XMLIngestRequest = Form(...),
#     file: UploadFile = File(...)
# ):
#     """
#     Ingest an XML file, optionally summarize it,
#     then either store ephemeral or persist in DB.
#     """
#     try:
#         file_bytes = await file.read()
#         filename = file.filename
#
#         # 1) call the service
#         result_data = await process_xml_task(
#             file_bytes=file_bytes,
#             filename=filename,
#             title=payload.title,
#             author=payload.author,
#             keywords=payload.keywords or [],
#             system_prompt=payload.system_prompt,
#             custom_prompt=payload.custom_prompt,
#             auto_summarize=payload.auto_summarize,
#             api_name=payload.api_name,
#             api_key=payload.api_key
#         )
#
#         # 2) ephemeral vs. persist
#         if payload.mode == "ephemeral":
#             ephemeral_id = ephemeral_storage.store_data(result_data)
#             return {
#                 "status": "ephemeral-ok",
#                 "media_id": ephemeral_id,
#                 "title": result_data["info_dict"]["title"]
#             }
#         else:
#             # store in DB
#             info_dict = result_data["info_dict"]
#             summary = result_data["summary"]
#             segments = result_data["segments"]
#             combined_prompt = (payload.system_prompt or "") + "\n\n" + (payload.custom_prompt or "")
#
#             media_id = add_media_to_database(
#                 url=filename,
#                 info_dict=info_dict,
#                 segments=segments,
#                 summary=summary,
#                 keywords=",".join(payload.keywords or []),
#                 custom_prompt_input=combined_prompt,
#                 whisper_model="xml-import",
#                 media_type="xml_document",
#                 overwrite=False
#             )
#
#             return {
#                 "status": "persist-ok",
#                 "media_id": str(media_id),
#                 "title": info_dict["title"]
#             }
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Your gradio_xml_ingestion_tab.py is already set up to call import_xml_handler(...) directly. If you’d prefer to unify it with the new approach, you can simply have your Gradio UI call the new POST /process-xml route, sending the file as UploadFile plus all your form fields. The existing code is fine for a local approach, but if you want your new single endpoint approach, you might adapt the code in the click() callback to do an HTTP request to /process-xml with the “mode” param, etc.
#
# End of XML Ingestion
############################################################################################################


######################## Web Scraping & URL Ingestion Endpoint ###################################
# Endpoints:
#

@router.post("/ingest-web-content")
async def ingest_web_content(
    request: IngestWebContentRequest,
    background_tasks: BackgroundTasks,
    token: str = Header(..., description="Authentication token"),
    db=Depends(get_db_for_user),
):
    """
    A single endpoint that supports multiple advanced scraping methods:
      - individual: Each item in 'urls' is scraped individually
      - sitemap:    Interprets the first 'url' as a sitemap, scrapes it
      - url_level:  Scrapes all pages up to 'url_level' path segments from the first 'url'
      - recursive:  Scrapes up to 'max_pages' links, up to 'max_depth' from the base 'url'

    Also supports content analysis, translation, chunking, DB ingestion, etc.
    """

    # 1) Basic checks
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")

    # If any array is shorter than # of URLs, pad it so we can zip them easily
    num_urls = len(request.urls)
    titles = request.titles or []
    authors = request.authors or []
    keywords = request.keywords or []

    if len(titles) < num_urls:
        titles += ["Untitled"] * (num_urls - len(titles))
    if len(authors) < num_urls:
        authors += ["Unknown"] * (num_urls - len(authors))
    if len(keywords) < num_urls:
        keywords += ["no_keyword_set"] * (num_urls - len(keywords))

    # 2) Parse cookies if needed
    custom_cookies_list = None
    if request.use_cookies and request.cookies:
        try:
            parsed = json.loads(request.cookies)
            # if it's a dict, wrap in a list
            if isinstance(parsed, dict):
                custom_cookies_list = [parsed]
            elif isinstance(parsed, list):
                custom_cookies_list = parsed
            else:
                raise ValueError("Cookies must be a dict or list of dicts.")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON for cookies: {e}")

    # 3) Choose the appropriate scraping method
    scrape_method = request.scrape_method
    logging.info(f"Selected scrape method: {scrape_method}")

    # We'll accumulate all “raw” results (scraped data) in a list of dicts
    raw_results = []

    # Helper function to perform summarization (if needed)
    async def maybe_summarize_one(article: dict) -> dict:
        if not request.perform_analysis:
            article["analysis"] = None
            return article

        content = article.get("content", "")
        if not content:
            article["analysis"] = "No content to analyze."
            return article

        # Analyze
        analysis_results = analyze(
            input_data=content,
            custom_prompt_arg=request.custom_prompt or "Summarize this article.",
            api_name=request.api_name,
            api_key=request.api_key,
            temp=0.7,
            system_message=request.system_prompt or "Act as a professional summarizer."
        )
        article["analysis"] = analysis_results

        # Rolling summarization or confab check
        if request.perform_rolling_summarization:
            logging.info("Performing rolling summarization (placeholder).")
            # Insert logic for multi-step summarization if needed
        if request.perform_confabulation_check_of_analysis:
            logging.info("Performing confabulation check of analysis (placeholder).")

        return article

    #####################################################################
    # INDIVIDUAL
    #####################################################################
    if scrape_method == ScrapeMethod.INDIVIDUAL:
        # Possibly multiple URLs
        # You already have a helper: scrape_and_summarize_multiple(...),
        # but we can do it manually to show the synergy with your “titles/authors” approach:
        # If you’d rather skip multiple loops, you can rely on your library.
        # For example, your library already can handle “custom_article_titles” as strings.
        # But here's a direct approach:

        for i, url in enumerate(request.urls):
            title_ = titles[i]
            author_ = authors[i]
            kw_ = keywords[i]

            # Scrape one URL
            article_data = await scrape_article(url, custom_cookies=custom_cookies_list)
            if not article_data or not article_data.get("extraction_successful"):
                logging.warning(f"Failed to scrape: {url}")
                continue

            # Overwrite metadata with user-supplied fields
            article_data["title"] = title_ or article_data["title"]
            article_data["author"] = author_ or article_data["author"]
            article_data["keywords"] = kw_

            # Summarize if requested
            article_data = await maybe_summarize_one(article_data)
            raw_results.append(article_data)

    #####################################################################
    # SITEMAP
    #####################################################################
    elif scrape_method == ScrapeMethod.SITEMAP:
        # Typically the user will supply only 1 URL in request.urls[0]
        sitemap_url = request.urls[0]
        # Sync approach vs. async approach: your library’s `scrape_from_sitemap`
        # is a synchronous function that returns a list of articles or partial results.

        # You might want to run it in a thread if it’s truly blocking:
        def scrape_in_thread():
            return scrape_from_sitemap(sitemap_url)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, scrape_in_thread)

        # The “scrape_from_sitemap” function might return partial dictionaries
        # that do not have the final summarization. Let’s handle summarization next:
        # We unify everything to raw_results.
        if not results:
            logging.warning("No articles returned from sitemap scraping.")
        else:
            # Each item is presumably a dict with at least {url, title, content}
            for r in results:
                # Summarize if needed
                r = await maybe_summarize_one(r)
                raw_results.append(r)

    #####################################################################
    # URL LEVEL
    #####################################################################
    elif scrape_method == ScrapeMethod.URL_LEVEL:
        # Typically the user will supply only 1 base URL
        base_url = request.urls[0]
        level = request.url_level or 2

        # `scrape_by_url_level(base_url, level)` is presumably synchronous in your code.
        def scrape_in_thread():
            return scrape_by_url_level(base_url, level)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, scrape_in_thread)

        if not results:
            logging.warning("No articles returned from URL-level scraping.")
        else:
            for r in results:
                # Summarize if needed
                r = await maybe_summarize_one(r)
                raw_results.append(r)

    #####################################################################
    # RECURSIVE SCRAPING
    #####################################################################
    elif scrape_method == ScrapeMethod.RECURSIVE:
        base_url = request.urls[0]
        max_pages = request.max_pages or 10
        max_depth = request.max_depth or 3

        # The function is already async, so we can call it directly
        # You also have `progress_callback` in your code.
        # For an API scenario, we might skip progress callbacks or store them in logs.
        results = await recursive_scrape(
            base_url=base_url,
            max_pages=max_pages,
            max_depth=max_depth,
            progress_callback=logging.info,  # or None if you want silent
            custom_cookies=custom_cookies_list
        )

        if not results:
            logging.warning("No articles returned from recursive scraping.")
        else:
            for r in results:
                # Summarize if needed
                r = await maybe_summarize_one(r)
                raw_results.append(r)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scrape method: {scrape_method}"
        )

    # 4) If we have nothing so far, exit
    if not raw_results:
        return {
            "status": "warning",
            "message": "No articles were successfully scraped for this request.",
            "results": []
        }

    # 5) Perform optional translation (if the user wants it *after* scraping)
    if request.perform_translation:
        logging.info(f"Translating to {request.translation_language} (placeholder).")
        # Insert your real translation code here:
        # for item in raw_results:
        #   item["content"] = translator.translate(item["content"], to_lang=request.translation_language)
        #   if item.get("analysis"):
        #       item["analysis"] = translator.translate(item["analysis"], to_lang=request.translation_language)

    # 6) Perform optional chunking
    if request.perform_chunking:
        logging.info("Performing chunking on each article (placeholder).")
        # Insert chunking logic here. For example:
        # for item in raw_results:
        #     chunks = chunk_text(
        #         text=item["content"],
        #         chunk_size=request.chunk_size,
        #         overlap=request.chunk_overlap,
        #         method=request.chunk_method,
        #         ...
        #     )
        #     item["chunks"] = chunks

    # 7) Timestamp or Overwrite
    if request.timestamp_option:
        timestamp_str = datetime.now().isoformat()
        for item in raw_results:
            item["ingested_at"] = timestamp_str

    # If overwriting existing is set, you’d query the DB here to see if the article already exists, etc.

    # 8) Optionally store results in DB
    # For each article, do something like:
    # media_ids = []
    # for r in raw_results:
    #     media_id = ingest_article_to_db(
    #         url=r["url"],
    #         title=r.get("title", "Untitled"),
    #         author=r.get("author", "Unknown"),
    #         content=r.get("content", ""),
    #         keywords=r.get("keywords", ""),
    #         ingestion_date=r.get("ingested_at", ""),
    #         analysis=r.get("analysis", None),
    #         chunking_data=r.get("chunks", [])
    #     )
    #     media_ids.append(media_id)
    #
    # return {
    #     "status": "success",
    #     "message": "Web content processed and added to DB",
    #     "count": len(raw_results),
    #     "media_ids": media_ids
    # }

    # If you prefer to just return everything as JSON:
    return {
        "status": "success",
        "message": "Web content processed",
        "count": len(raw_results),
        "results": raw_results
    }

# Web Scraping
#     Accepts JSON body describing the scraping method, URL(s), etc.
#     Calls process_web_scraping_task(...).
#     Returns ephemeral or persistent results.
# POST /api/v1/media/process-web-scraping
# that takes a JSON body in the shape of WebScrapingRequest and uses your same “Gradio logic” behind the scenes, but in an API-friendly manner.
#
# Clients can now POST JSON like:
# {
#   "scrape_method": "Individual URLs",
#   "url_input": "https://example.com/article1\nhttps://example.com/article2",
#   "url_level": null,
#   "max_pages": 10,
#   "max_depth": 3,
#   "summarize_checkbox": true,
#   "custom_prompt": "Please summarize with bullet points only.",
#   "api_name": "openai",
#   "api_key": "sk-1234",
#   "keywords": "web, scraping, example",
#   "custom_titles": "Article 1 Title\nArticle 2 Title",
#   "system_prompt": "You are a bulleted-notes specialist...",
#   "temperature": 0.7,
#   "custom_cookies": [{"name":"mycookie", "value":"abc", "domain":".example.com"}],
#   "mode": "ephemeral"
# }
#
#     scrape_method can be "Individual URLs", "Sitemap", "URL Level", or "Recursive Scraping".
#     url_input is either:
#         Multi-line list of URLs (for "Individual URLs"),
#         A single sitemap URL (for "Sitemap"),
#         A single base URL (for "URL Level" or "Recursive Scraping"),
#     url_level only matters if scrape_method="URL Level".
#     max_pages and max_depth matter if scrape_method="Recursive Scraping".
#     summarize_checkbox indicates if you want to run summarization afterwards.
#     api_name + api_key for whichever LLM you want to do summarization.
#     custom_cookies is an optional list of cookie dicts for e.g. paywalls or login.
#     mode can be "ephemeral" or "persist".
#
# The endpoint returns a structure describing ephemeral or persisted results, consistent with your other ingestion endpoints.

# FIXME

# /Server_API/app/api/v1/endpoints/media.py
class WebScrapingRequest(BaseModel):
    scrape_method: str  # "Individual URLs", "Sitemap", "URL Level", "Recursive Scraping"
    url_input: str
    url_level: Optional[int] = None
    max_pages: int = 10
    max_depth: int = 3
    summarize_checkbox: bool = False
    custom_prompt: Optional[str] = None
    api_name: Optional[str] = None
    api_key: Optional[str] = None
    keywords: Optional[str] = "default,no_keyword_set"
    custom_titles: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    custom_cookies: Optional[List[Dict[str, Any]]] = None  # e.g. [{"name":"mycookie","value":"abc"}]
    mode: str = "persist"  # or "ephemeral"

@router.post("/process-web-scraping")
async def process_web_scraping_endpoint(
        payload: WebScrapingRequest,
        # 1. Auth + UserID Determined through `get_db_by_user`
        # token: str = Header(None), # Use Header(None) for optional
        # 2. DB Dependency
        db: Database = Depends(get_db_for_user),
    ):
    """
    Ingest / scrape data from websites or sitemaps, optionally summarize,
    then either store ephemeral or persist in DB.
    """
    try:
        # Delegates to the service
        result = await process_web_scraping_task(
            scrape_method=payload.scrape_method,
            url_input=payload.url_input,
            url_level=payload.url_level,
            max_pages=payload.max_pages,
            max_depth=payload.max_depth,
            summarize_checkbox=payload.summarize_checkbox,
            custom_prompt=payload.custom_prompt,
            api_name=payload.api_name,
            api_key=payload.api_key,
            keywords=payload.keywords or "",
            custom_titles=payload.custom_titles,
            system_prompt=payload.system_prompt,
            temperature=payload.temperature,
            custom_cookies=payload.custom_cookies,
            mode=payload.mode
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#
# End of Web Scraping Ingestion
#####################################################################################



######################## Debugging and Diagnostics ###################################
# Endpoints:
#     GET /api/v1/media/debug/schema
# Debugging and Diagnostics
@router.get("/debug/schema",)
async def debug_schema(
        # 1. Auth + UserID Determined through `get_db_by_user`
        # token: str = Header(None), # Use Header(None) for optional
        # 2. DB Dependency
        db: Database = Depends(get_db_for_user),
    ):
    """Diagnostic endpoint to check database schema."""
    try:
        schema_info = {}

        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            schema_info["tables"] = [table[0] for table in cursor.fetchall()]

            # Get Media table columns
            cursor.execute("PRAGMA table_info(Media)")
            schema_info["media_columns"] = [col[1] for col in cursor.fetchall()]

            # Get MediaModifications table columns
            cursor.execute("PRAGMA table_info(MediaModifications)")
            schema_info["media_mods_columns"] = [col[1] for col in cursor.fetchall()]

            # Count media rows
            cursor.execute("SELECT COUNT(*) FROM Media")
            schema_info["media_count"] = cursor.fetchone()[0]

        return schema_info
    except Exception as e:
        return {"error": str(e)}

#
# End of Debugging and Diagnostics
#####################################################################################

async def _download_url_async(
        client: httpx.AsyncClient,
        url: str,
        target_dir: Path,
        allowed_extensions: Optional[Set[str]] = None,  # Use a Set for faster lookups
        check_extension: bool = True  # Flag to enable/disable check
) -> Path:
    """
    Downloads a URL asynchronously and saves it to the target directory.
    Optionally validates the file extension against a set of allowed extensions.
    """
    if allowed_extensions is None:
        allowed_extensions = set()  # Default to empty set if None

    # Generate a safe filename
    try:
        # Basic filename extraction - consider more robust libraries if needed
        try:
            url_path_segment = httpx.URL(url).path.split('/')[-1]
            if url_path_segment:
                # Basic sanitization (replace potentially invalid chars) - enhance if needed
                safe_segment = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in url_path_segment)
                filename = safe_segment
            else:
                # Fallback if no path segment
                filename = f"downloaded_{hash(url)}.tmp"
        except Exception:  # Broad catch for URL parsing issues
            filename = f"downloaded_{hash(url)}.tmp"

        target_path = target_dir / filename
        # Simple collision avoidance (add number if exists) - improve if high concurrency expected
        counter = 1
        base_name = target_path.stem
        suffix = target_path.suffix
        while target_path.exists():
            target_path = target_dir / f"{base_name}_{counter}{suffix}"
            counter += 1

        async with client.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()  # Raise HTTPStatusError for 4xx/5xx

            if check_extension and allowed_extensions:
                # Get the actual suffix from the final target path
                actual_suffix = target_path.suffix.lower()  # Use the generated path's suffix
                if not actual_suffix:
                    # Try getting from Content-Disposition header if available
                    content_disposition = response.headers.get('content-disposition')
                    if content_disposition:
                        match = re.search(r'filename=["\'](.*?)["\']', content_disposition)
                        disp_filename = match.group(1) if match else None
                        if disp_filename:
                            actual_suffix = Path(disp_filename).suffix.lower()

                if not actual_suffix or actual_suffix not in allowed_extensions:
                    raise ValueError(
                        f"Downloaded file '{target_path.name}' from {url} does not have an allowed extension (allowed: {', '.join(allowed_extensions)})")

            async with aiofiles.open(target_path, 'wb') as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    await f.write(chunk)

            logger.info(f"Successfully downloaded {url} to {target_path}")
            return target_path

    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error downloading {url}: {e.response.status_code} - {e.response.text[:200]}...")  # Log snippet of text
        # Attempt cleanup of potentially partially downloaded file
        if 'target_path' in locals() and target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise ConnectionError(f"HTTP error {e.response.status_code} for {url}") from e
    except httpx.RequestError as e:
        logger.error(f"Request error downloading {url}: {e}")
        raise ConnectionError(f"Network/request error for {url}: {e}") from e
    except ValueError as e:  # Catch our specific extension validation error
        logger.error(f"Validation error for {url}: {e}")
        # Attempt cleanup
        if 'target_path' in locals() and target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise ValueError(str(e)) from e  # Re-raise the specific error
    except Exception as e:
        logger.error(f"Error processing download for {url}: {e}", exc_info=True)
        # Attempt cleanup
        if 'target_path' in locals() and target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to download or save {url}: {e}") from e  # Use RuntimeError for unexpected

#
# End of media.py
#######################################################################################################################
