# Server_API/app/api/v1/endpoints/text_processing.py
# Description: This code provides FastAPI endpoints for text processing, including chunking.
#
# Imports
import asyncio
from typing import List, Optional, Dict, Any
#
# Third-party Libraries
from pydantic import BaseModel, Field, field_validator
from loguru import logger
from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Depends # If you add auth/DB dependencies later
)
from fastapi.encoders import jsonable_encoder

from tldw_Server_API.app.api.v1.schemas.chunking_schema import ChunkingResponse, ChunkingTextRequest, \
    ChunkingOptionsRequest
#
# Local Imports
from tldw_Server_API.app.core.Utils.Chunk_Lib import (
    improved_chunking_process,
    chunk_options as default_chunk_options_from_lib, # Default options from your library's config
    # Import any specific exception types if you define them in Chunk_Lib
)
#
#######################################################################################################################
#
# Functions:

# --- FastAPI Router ---
router = APIRouter()

# --- Endpoint to Chunk Text (JSON input) ---
@router.post(
    "/chunk_text",
    summary="Chunks provided text content based on specified options.",
    tags=["Text Processing", "Chunking"],
    response_model=ChunkingResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input, options, or chunking error (e.g., invalid JSON in text for 'json' method)."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error in request payload."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during chunking."},
    }
)
async def process_text_for_chunking_json(
    request_data: ChunkingTextRequest = Body(...)
    # Add Depends(get_current_user) or similar if you need authentication
):
    """
    Accepts text content and chunking options in a JSON body.
    Returns the text divided into chunks with associated metadata.

    - **text_content**: The raw string data to be chunked.
    - **file_name**: (Optional) A nominal filename for context.
    - **options**: (Optional) A dictionary specifying chunking parameters:
        - **method**: e.g., 'words', 'sentences', 'json', 'semantic', 'xml', 'ebook_chapters'.
        - **max_size**: Max size for chunks (depends on method).
        - **overlap**: Overlap between chunks.
        - **language**: e.g., 'en'. Auto-detected if None.
        - **adaptive**: (bool) For methods that support it.
        - **multi_level**: (bool) For methods that support it.
        - **custom_chapter_pattern**: (str) Regex for 'ebook_chapters' method.
        *(Refer to ChunkingOptionsRequest schema for all parameters and defaults)*
    """
    logger.info(f"Received chunking request for '{request_data.file_name}'. Method: {request_data.options.method if request_data.options else 'default'}.")

    # Prepare effective chunking options: start with library defaults, then override with request options
    effective_options = default_chunk_options_from_lib.copy()
    if request_data.options:
        # model_dump(exclude_unset=True) ensures only provided fields are used for update
        request_options_dict = request_data.options.model_dump(exclude_unset=True)
        effective_options.update(request_options_dict)
        logger.debug(f"Request options provided: {request_options_dict}")

    # Ensure critical options have valid types (Pydantic validator on model helps, but good to be defensive)
    try:
        effective_options['max_size'] = int(effective_options.get('max_size', default_chunk_options_from_lib.get('max_size')))
        effective_options['overlap'] = int(effective_options.get('overlap', default_chunk_options_from_lib.get('overlap')))
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid type for max_size or overlap in effective_options: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"max_size and overlap must be integers. Error: {e}"
        )

    logger.debug(f"Effective chunking options: {effective_options}")

    loop = asyncio.get_running_loop()
    try:
        # improved_chunking_process is synchronous and can be CPU-intensive
        # Run it in a thread pool to avoid blocking the event loop
        chunk_results: List[Dict[str, Any]] = await loop.run_in_executor(
            None,  # Uses the default executor (ThreadPoolExecutor)
            improved_chunking_process,
            request_data.text_content,
            effective_options  # Pass the dictionary of options
        )
    except ValueError as ve: # Catch specific errors from chunk_lib if possible
        logger.warning(f"ValueError during chunking process for '{request_data.file_name}': {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during chunking process for '{request_data.file_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An internal error occurred during text chunking: {type(e).__name__}")

    if not chunk_results:
        logger.warning(f"Chunking produced no results for '{request_data.file_name}'.")
        # Depending on desired behavior, you might return empty list or an error
        # For now, return empty list, client can decide how to handle.

    # Prepare response: Convert dicts to Pydantic models for validation and schema generation (optional, but good practice)
    # For simplicity here, we'll trust the structure from improved_chunking_process if it matches ChunkedContentResponse.
    # If you want strict validation of the library's output:
    # validated_chunks = [ChunkedContentResponse(**chunk) for chunk in chunk_results]

    return ChunkingResponse(
        chunks=chunk_results, # Directly use if structure matches ChunkedContentResponse
        original_file_name=request_data.file_name,
        applied_options=ChunkingOptionsRequest(**effective_options) # Show what was actually used
    )


# --- Endpoint to Chunk Uploaded File ---
# (Optional - if you want a separate endpoint for file uploads which is often cleaner)
@router.post(
    "/chunk_file",
    summary="Uploads a file, chunks its content, and returns the chunks.",
    tags=["Text Processing", "Chunking"],
    response_model=ChunkingResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "No file uploaded, or chunking error."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error in form parameters."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during chunking."},
    }
)
async def process_file_for_chunking(
    file: UploadFile = File(...),
    # Chunking options can be sent as form data.
    # Using `Form` for each option. FastAPI will populate them.
    # The Pydantic model `ChunkingOptionsRequest` can be used with `Depends` for complex form data,
    # or manually construct the options dict.
    method: Optional[str] = Form(default_chunk_options_from_lib.get('method'), description="Chunking method."),
    max_size: Optional[int] = Form(default_chunk_options_from_lib.get('max_size'), description="Max chunk size."),
    overlap: Optional[int] = Form(default_chunk_options_from_lib.get('overlap'), description="Chunk overlap."),
    language: Optional[str] = Form(default_chunk_options_from_lib.get('language'), description="Text language."),
    adaptive: Optional[bool] = Form(default_chunk_options_from_lib.get('adaptive')),
    multi_level: Optional[bool] = Form(default_chunk_options_from_lib.get('multi_level')),
    custom_chapter_pattern: Optional[str] = Form(None)
):
    """
    Accepts a file upload and chunking options via form data.
    Returns the file content divided into chunks with associated metadata.
    """
    logger.info(f"Received file upload for chunking: '{file.filename}'. Method: {method}.")

    if not file.filename: # Should not happen with File(...) but good check
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided or filename is missing.")

    try:
        text_content_bytes = await file.read()
        text_content = text_content_bytes.decode('utf-8') # Assuming UTF-8, add error handling or encoding detection if needed
    except Exception as e:
        logger.error(f"Error reading uploaded file '{file.filename}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read or decode file: {e}")
    finally:
        await file.close()

    # Consolidate form options into a dictionary
    effective_options_dict = {
        'method': method,
        'max_size': int(max_size) if max_size is not None else default_chunk_options_from_lib.get('max_size'),
        'overlap': int(overlap) if overlap is not None else default_chunk_options_from_lib.get('overlap'),
        'language': language,
        'adaptive': adaptive,
        'multi_level': multi_level,
        'custom_chapter_pattern': custom_chapter_pattern,
    }
    # Filter out None values if improved_chunking_process expects them to be absent vs. None
    effective_options = {k: v for k, v in effective_options_dict.items() if v is not None}
    # Merge with library defaults for any options not provided in the form
    final_processing_options = default_chunk_options_from_lib.copy()
    final_processing_options.update(effective_options)

    logger.debug(f"Effective chunking options for file: {final_processing_options}")

    loop = asyncio.get_running_loop()
    try:
        chunk_results: List[Dict[str, Any]] = await loop.run_in_executor(
            None,
            improved_chunking_process,
            text_content,
            final_processing_options
        )
    except ValueError as ve:
        logger.warning(f"ValueError during chunking file '{file.filename}': {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during chunking file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An internal error occurred during file chunking: {type(e).__name__}")

    return ChunkingResponse(
        chunks=chunk_results,
        original_file_name=file.filename,
        applied_options=ChunkingOptionsRequest(**final_processing_options)
    )

# To include this router in your main FastAPI app:
# from .endpoints import text_processing
# app.include_router(text_processing.router, prefix="/api/v1/process", tags=["Text Processing"])