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

# Local Imports
from tldw_Server_API.app.core.Utils.Chunk_Lib import (
    improved_chunking_process,
    DEFAULT_CHUNK_OPTIONS as default_chunk_options_from_lib,
    ChunkingError, # Import custom exceptions from Chunk_Lib
    InvalidInputError,
    InvalidChunkingMethodError
)
from tldw_Server_API.app.api.v1.schemas.chunking_schema import ChunkingResponse, ChunkingTextRequest, \
    ChunkingOptionsRequest
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze as general_llm_analyzer
from tldw_Server_API.app.core.config import load_and_log_configs as load_server_configs
#
#######################################################################################################################
#
# Functions:

# --- FastAPI Router ---
router = APIRouter()

# --- Pydantic Schemas for Request and Response ---

class LLMOptionsForChunking(BaseModel):
    """Optional LLM parameters if a chunking method uses an LLM (e.g., rolling_summarize)."""
    provider: Optional[str] = Field(None, description="LLM provider for internal chunking steps (e.g., 'openai', 'anthropic'). Server default if None.")
    model: Optional[str] = Field(None, description="LLM model for internal chunking steps. Server default if None.")
    temperature: Optional[float] = Field(None, description="Temperature for LLM. Server default if None.")
    # Add other relevant LLM params you might want to control, e.g., system_prompt for summarization step

class ChunkingOptionsRequest(BaseModel):
    method: Optional[str] = Field(default_chunk_options_from_lib.get('method'),
                                  description="Chunking method (e.g., 'words', 'sentences', 'json', 'semantic', 'xml', 'ebook_chapters', 'rolling_summarize').")
    max_size: Optional[int] = Field(default_chunk_options_from_lib.get('max_size'),
                                   description="Max size of chunks (meaning depends on method).")
    overlap: Optional[int] = Field(default_chunk_options_from_lib.get('overlap'),
                                  description="Overlap between chunks (meaning depends on method).")
    language: Optional[str] = Field(default_chunk_options_from_lib.get('language'),
                                    description="Language of the text (e.g., 'en', 'zh'). Auto-detected if None.")
    adaptive: Optional[bool] = Field(default_chunk_options_from_lib.get('adaptive'),
                                     description="Enable adaptive chunking.")
    multi_level: Optional[bool] = Field(default_chunk_options_from_lib.get('multi_level'),
                                        description="Enable multi-level chunking.")
    custom_chapter_pattern: Optional[str] = Field(None,
                                                  description="Custom regex pattern for 'ebook_chapters' method.")
    tokenizer_name_or_path: Optional[str] = Field("gpt2",
                                                  description="Tokenizer model name or path (e.g., 'gpt2', 'bert-base-uncased').")
    # Options for LLM-dependent chunking methods like 'rolling_summarize'
    llm_options_for_internal_steps: Optional[LLMOptionsForChunking] = Field(None,
                                                                         description="LLM configurations if the chunking method itself uses an LLM.")

    # Add other specific options from DEFAULT_CHUNK_OPTIONS if you want them client-configurable
    semantic_similarity_threshold: Optional[float] = Field(default_chunk_options_from_lib.get('semantic_similarity_threshold'), description="Threshold for semantic chunking breaks.")
    semantic_overlap_sentences: Optional[int] = Field(default_chunk_options_from_lib.get('semantic_overlap_sentences'), description="Sentence overlap for semantic_chunking.")
    json_chunkable_data_key: Optional[str] = Field(default_chunk_options_from_lib.get('json_chunkable_data_key', 'data'), description="Key in JSON dict to chunk.")
    # Options for rolling_summarize if not covered by llm_options_for_internal_steps
    summarization_detail: Optional[float] = Field(default_chunk_options_from_lib.get('summarization_detail'), ge=0, le=1, description="Detail level for rolling_summarize (0.0-1.0).")


    @field_validator('max_size', 'overlap', 'semantic_overlap_sentences', mode='before')
    @classmethod
    def ensure_int_type(cls, value: Any, field) -> Optional[int]: # Added field argument for context
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            # Pydantic will raise its own validation error with field name, which is good.
            # This custom validator mostly handles explicit type conversion if string numbers are sent.
            raise ValueError(f"{field.name} must be an integer or convertible to an integer")

    @field_validator('semantic_similarity_threshold', 'summarization_detail', mode='before')
    @classmethod
    def ensure_float_type(cls, value: Any, field) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{field.name} must be a float or convertible to a float")

class ChunkingTextRequest(BaseModel):
    text_content: str = Field(..., description="Text content to be chunked.")
    file_name: Optional[str] = Field("input_text.txt",
                                     description="Optional name for the input, used in some metadata/logging.")
    options: Optional[ChunkingOptionsRequest] = Field(None, description="Chunking parameters. Library defaults will be used if not provided or partially provided.")

class ChunkedContentResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]

class ChunkingResponse(BaseModel):
    chunks: List[ChunkedContentResponse]
    original_file_name: Optional[str]
    applied_options: ChunkingOptionsRequest # Shows the actual options used

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
    logger.info(f"Received chunking request for '{request_data.file_name}'. Method: {request_data.options.method if request_data.options else 'default from library'}.")

    # Prepare effective chunking options
    effective_options = default_chunk_options_from_lib.copy()
    if request_data.options:
        request_options_dict = request_data.options.model_dump(exclude_unset=True) # Only use fields explicitly set by client
        # Special handling for nested llm_options
        if 'llm_options_for_internal_steps' in request_options_dict and request_options_dict['llm_options_for_internal_steps'] is not None:
            # If llm_options are provided, update them carefully
            # Assuming direct update is fine if Pydantic model is structured well
            pass # Pydantic's model_dump should handle this nesting.
        effective_options.update(request_options_dict)
        logger.debug(f"Request options provided: {request_options_dict}")
    else: # No options provided in request, log that we are using library defaults
        logger.debug(f"No request options provided. Using default library options: {effective_options}")


    # Type conversions for max_size and overlap are now better handled by Pydantic model's field_validators
    # Ensure required integer options are indeed integers if they came from dict update
    for key_to_check in ['max_size', 'overlap']:
        if key_to_check in effective_options and effective_options[key_to_check] is not None:
            try:
                effective_options[key_to_check] = int(effective_options[key_to_check])
            except (ValueError, TypeError) as e:
                 raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Option '{key_to_check}' must be an integer. Value: {effective_options[key_to_check]}. Error: {e}"
                )


    logger.debug(f"Effective chunking options before LLM setup: {effective_options}")

    # --- LLM Configuration for specific chunking methods ---
    llm_call_func_to_use = None
    llm_api_config_to_use = None
    # Tokenizer is now part of effective_options, to be read by Chunker init
    tokenizer_for_chunker = effective_options.get("tokenizer_name_or_path", "gpt2") # Default if not set

    current_chunking_method = effective_options.get('method')
    if current_chunking_method == 'rolling_summarize': # Or other methods you add that need LLM
        llm_call_func_to_use = general_llm_analyzer # Your Summarization_General_Lib.analyze

        # Load server's comprehensive configuration to get API keys and defaults
        server_configs = load_server_configs()
        if not server_configs:
            logger.error("Server configuration could not be loaded. LLM-dependent chunking may fail.")
            # Depending on policy, you might raise an error here if the method *requires* LLM
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Server configuration error, cannot perform LLM-dependent chunking.")

        # Determine LLM provider and model for the summarization steps
        # Priority: Request's llm_options -> Server default for summarization -> Hardcoded default
        requested_llm_options = effective_options.get('llm_options_for_internal_steps', {}) # This is a dict now
        if requested_llm_options is None: requested_llm_options = {}


        default_summarization_provider = server_configs.get('llm_api_settings', {}).get('default_api', 'openai')
        summarization_provider = requested_llm_options.get('provider') or default_summarization_provider

        provider_specific_config_key = f"{summarization_provider}_api" # e.g., "openai_api"
        api_details_from_server_config = server_configs.get(provider_specific_config_key, {})

        # Build llm_api_config for the call to `general_llm_analyzer`
        llm_api_config_to_use = {
            "api_name": summarization_provider,
            "model": requested_llm_options.get('model') or api_details_from_server_config.get('model'),
            "api_key": api_details_from_server_config.get('api_key'), # CRITICAL: Key comes from server config
            "temp": requested_llm_options.get('temperature'), # If None, general_llm_analyzer will use its own default/config
            # Add other parameters that general_llm_analyzer might expect or that need to be overridden
            # e.g., "system_message" could be derived from effective_options.get('summarize_system_prompt')
        }

        if not llm_api_config_to_use.get("api_key"):
            logger.error(f"API key for '{summarization_provider}' for internal summarization step not found in server configuration.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Configuration error: Missing API key for {summarization_provider} for internal LLM step.")
        if not llm_api_config_to_use.get("model"):
            logger.error(f"Model for '{summarization_provider}' for internal summarization step not determined.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Configuration error: Missing model for {summarization_provider} for internal LLM step.")

        logger.info(f"'{current_chunking_method}' will use LLM provider: {summarization_provider}, Model: {llm_api_config_to_use['model']}")


    # --- Perform Chunking ---
    loop = asyncio.get_running_loop()
    try:
        chunk_results: List[Dict[str, Any]] = await loop.run_in_executor(
            None,
            improved_chunking_process,
            request_data.text_content,
            effective_options,  # Pass the full dict of resolved options
            tokenizer_for_chunker, # Pass the selected tokenizer
            llm_call_func_to_use,      # Pass the prepared LLM function
            llm_api_config_to_use      # Pass the config for that LLM function
        )
    except (ChunkingError, InvalidInputError, InvalidChunkingMethodError) as lib_error: # Catch specific errors from Chunk_Lib
        logger.warning(f"Chunking library error for '{request_data.file_name}': {lib_error}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(lib_error))
    except ValueError as ve: # General value errors (e.g., from Pydantic or type conversions if not caught earlier)
        logger.warning(f"ValueError during chunking setup or process for '{request_data.file_name}': {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during chunking process for '{request_data.file_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An internal error occurred during text chunking: {type(e).__name__}")

    if not chunk_results:
        logger.info(f"Chunking produced no results for '{request_data.file_name}'. Returning empty list.")

    return ChunkingResponse(
        chunks=chunk_results,
        original_file_name=request_data.file_name,
        applied_options=ChunkingOptionsRequest(**effective_options) # Show what was actually used
    )


# --- Endpoint to Chunk Uploaded File ---
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
    # Form fields for chunking options
    method: Optional[str] = Form(default_chunk_options_from_lib.get('method')),
    max_size: Optional[int] = Form(default_chunk_options_from_lib.get('max_size')),
    overlap: Optional[int] = Form(default_chunk_options_from_lib.get('overlap')),
    language: Optional[str] = Form(default_chunk_options_from_lib.get('language')),
    adaptive: Optional[bool] = Form(default_chunk_options_from_lib.get('adaptive')),
    multi_level: Optional[bool] = Form(default_chunk_options_from_lib.get('multi_level')),
    custom_chapter_pattern: Optional[str] = Form(None),
    tokenizer_name_or_path: Optional[str] = Form("gpt2"),
    # LLM options for methods like rolling_summarize (optional, server defaults often preferred)
    llm_provider_for_internal: Optional[str] = Form(None),
    llm_model_for_internal: Optional[str] = Form(None),
    llm_temperature_for_internal: Optional[float] = Form(None),
    summarization_detail_for_internal: Optional[float] = Form(None),
    # Add other form fields corresponding to ChunkingOptionsRequest if needed
):
    """
    Accepts a file upload and chunking options via form data.
    Returns the file content divided into chunks with associated metadata.
    """
    logger.info(f"Received file upload for chunking: '{file.filename}'. Method from form: {method}.")

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

    # Consolidate form options into a dictionary, similar to how Pydantic model would be built
    form_options = {
        'method': method,
        'max_size': max_size,
        'overlap': overlap,
        'language': language,
        'adaptive': adaptive,
        'multi_level': multi_level,
        'custom_chapter_pattern': custom_chapter_pattern,
        'tokenizer_name_or_path': tokenizer_name_or_path,
        'llm_options_for_internal_steps': { # Construct the nested dict
            'provider': llm_provider_for_internal,
            'model': llm_model_for_internal,
            'temperature': llm_temperature_for_internal,
        } if llm_provider_for_internal or llm_model_for_internal or llm_temperature_for_internal else None, # Only add if any LLM option is given
        'summarization_detail': summarization_detail_for_internal,
    }
    # Filter out None values from the top level to allow library defaults to take precedence cleanly
    form_options_cleaned = {k: v for k, v in form_options.items() if v is not None}
    if form_options_cleaned.get('llm_options_for_internal_steps') is not None:
        form_options_cleaned['llm_options_for_internal_steps'] = {
            k:v for k,v in form_options_cleaned['llm_options_for_internal_steps'].items() if v is not None
        }
        if not form_options_cleaned['llm_options_for_internal_steps']: # If all nested are None
            del form_options_cleaned['llm_options_for_internal_steps']


    # Prepare effective_options: Start with library defaults, then update with form options
    effective_processing_options = default_chunk_options_from_lib.copy()
    effective_processing_options.update(form_options_cleaned)
    logger.debug(f"Effective chunking options from form data: {effective_processing_options}")

    # Similar LLM config setup as in the JSON endpoint
    llm_call_func_to_use = None
    llm_api_config_to_use = None
    tokenizer_for_chunker_file = effective_processing_options.get("tokenizer_name_or_path", "gpt2")

    current_chunking_method_file = effective_processing_options.get('method')
    if current_chunking_method_file == 'rolling_summarize':
        llm_call_func_to_use = general_llm_analyzer
        server_configs = load_server_configs()
        if not server_configs:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error for LLM step.")

        requested_llm_options_file = effective_processing_options.get('llm_options_for_internal_steps', {})
        if requested_llm_options_file is None: requested_llm_options_file = {}

        default_summarization_provider_file = server_configs.get('llm_api_settings', {}).get('default_api', 'openai')
        summarization_provider_file = requested_llm_options_file.get('provider') or default_summarization_provider_file
        provider_specific_config_key_file = f"{summarization_provider_file}_api"
        api_details_from_server_config_file = server_configs.get(provider_specific_config_key_file, {})

        llm_api_config_to_use = {
            "api_name": summarization_provider_file,
            "model": requested_llm_options_file.get('model') or api_details_from_server_config_file.get('model'),
            "api_key": api_details_from_server_config_file.get('api_key'),
            "temp": requested_llm_options_file.get('temperature'),
        }
        if not llm_api_config_to_use.get("api_key"):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server config missing API key for {summarization_provider_file}.")
        if not llm_api_config_to_use.get("model"):
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server config missing model for {summarization_provider_file}.")
        logger.info(f"'{current_chunking_method_file}' for file will use LLM: {summarization_provider_file}, Model: {llm_api_config_to_use['model']}")


    loop = asyncio.get_running_loop()
    try:
        chunk_results: List[Dict[str, Any]] = await loop.run_in_executor(
            None,
            improved_chunking_process,
            text_content,
            effective_processing_options, # Pass the constructed options dict
            tokenizer_for_chunker_file,
            llm_call_func_to_use,
            llm_api_config_to_use
        )
    except (ChunkingError, InvalidInputError, InvalidChunkingMethodError) as lib_error:
        logger.warning(f"Chunking library error for file '{file.filename}': {lib_error}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(lib_error))
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
        applied_options=ChunkingOptionsRequest(**effective_processing_options) # Show actual options used
    )