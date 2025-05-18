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
chunking_router = APIRouter()

# --- Endpoint to Chunk Text (JSON input) ---
@chunking_router.post(
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

        server_task_specific_model = api_details_from_server_config.get('model_for_summarization')
        logger.debug(f"TEMP DEBUG: server_task_specific_model = {server_task_specific_model}")

        server_general_model = api_details_from_server_config.get('model')
        logger.debug(f"TEMP DEBUG: server_general_model = {server_general_model}")

        final_model_for_step = server_task_specific_model or server_general_model
        logger.debug(f"TEMP DEBUG: final_model_for_step = {final_model_for_step}")

        # System Prompt for internal LLM steps:
        # Priority: Client suggested -> Server default for rolling_summarize method -> General LLM default
        client_suggested_system_prompt = requested_llm_options.get('system_prompt_for_step')
        # Get the method-specific default from Chunker's options (which came from global config)
        method_default_system_prompt = effective_options.get('summarize_system_prompt')  # Specific to rolling_summarize

        final_system_prompt_for_step = client_suggested_system_prompt or method_default_system_prompt
        # If still None, your general_llm_analyzer might have its own ultimate default.

        # Max tokens per LLM step:
        client_suggested_max_tokens = requested_llm_options.get('max_tokens_per_step')
        # Server might have a cap or default for this specific internal operation
        server_default_max_tokens_step = api_details_from_server_config.get('max_tokens_for_summarization_step',
                                                                            1024)  # Example key in your config

        final_max_tokens_for_step = client_suggested_max_tokens or server_default_max_tokens_step
        # Optional: Apply a server-enforced cap
        # server_cap_max_tokens = 2048
        # if final_max_tokens_for_step > server_cap_max_tokens:
        #     logger.warning(f"Client suggested max_tokens_per_step {final_max_tokens_for_step} capped to {server_cap_max_tokens}")
        #     final_max_tokens_for_step = server_cap_max_tokens

        # Build llm_api_config for the call to `general_llm_analyzer`
        llm_api_config_to_use = {
            "api_name": summarization_provider,
            "model": final_model_for_step or api_details_from_server_config.get('model'),
            "api_key": api_details_from_server_config.get('api_key'),
            "temp": requested_llm_options.get('temperature'), # If None, general_llm_analyzer will use its own default/config
            "system_message": final_system_prompt_for_step,
            "max_tokens": final_max_tokens_for_step,
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

    # FIXME chunks incorrect type
    return ChunkingResponse(
        chunks=chunk_results,
        original_file_name=request_data.file_name,
        applied_options=ChunkingOptionsRequest(**effective_options) # Show what was actually used
    )


# --- Endpoint to Chunk Uploaded File ---
@chunking_router.post(
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
    language: Optional[str] = Form(None), # Default to None for auto-detection
    tokenizer_name_or_path: Optional[str] = Form(default_chunk_options_from_lib.get('tokenizer_name_or_path', "gpt2")),
    adaptive: Optional[bool] = Form(default_chunk_options_from_lib.get('adaptive')),
    multi_level: Optional[bool] = Form(default_chunk_options_from_lib.get('multi_level')),
    custom_chapter_pattern: Optional[str] = Form(None),
    semantic_similarity_threshold: Optional[float] = Form(default_chunk_options_from_lib.get('semantic_similarity_threshold')),
    semantic_overlap_sentences: Optional[int] = Form(default_chunk_options_from_lib.get('semantic_overlap_sentences')),
    json_chunkable_data_key: Optional[str] = Form(default_chunk_options_from_lib.get('json_chunkable_data_key', 'data')),
    summarization_detail: Optional[float] = Form(default_chunk_options_from_lib.get('summarization_detail')),
    # Flattened client suggestions for LLM options for internal steps
    llm_step_temperature: Optional[float] = Form(None, description="Client suggested temp for internal LLM steps."),
    llm_step_system_prompt: Optional[str] = Form(None, description="Client suggested system prompt for internal LLM steps."),
    llm_step_max_tokens: Optional[int] = Form(None, description="Client suggested max tokens for internal LLM steps."),
):
    logger.info(f"Received file upload for chunking: '{file.filename}'. Method from form: {method}.")

    if not file.filename: # Should not happen with File(...) but good check
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided or filename is missing.")

    try:
        text_content_bytes = await file.read()
        text_content = text_content_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading uploaded file '{file.filename}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read or decode file: {e}")
    finally:
        await file.close()

    # Consolidate form options
    form_options_dict = {
        'method': method, 'max_size': max_size, 'overlap': overlap, 'language': language,
        'tokenizer_name_or_path': tokenizer_name_or_path, 'adaptive': adaptive, 'multi_level': multi_level,
        'custom_chapter_pattern': custom_chapter_pattern,
        'semantic_similarity_threshold': semantic_similarity_threshold,
        'semantic_overlap_sentences': semantic_overlap_sentences,
        'json_chunkable_data_key': json_chunkable_data_key,
        'summarization_detail': summarization_detail,
    }
    # Build the nested llm_options_for_internal_steps from flattened form fields
    internal_llm_opts_from_form = {}
    if llm_step_temperature is not None: internal_llm_opts_from_form['temperature'] = llm_step_temperature
    if llm_step_system_prompt is not None: internal_llm_opts_from_form['system_prompt_for_step'] = llm_step_system_prompt
    if llm_step_max_tokens is not None: internal_llm_opts_from_form['max_tokens_per_step'] = llm_step_max_tokens

    if internal_llm_opts_from_form:
        form_options_dict['llm_options_for_internal_steps'] = internal_llm_opts_from_form

    # Filter out None values from the top level to allow library defaults
    form_options_cleaned = {k: v for k, v in form_options_dict.items() if v is not None}

    effective_processing_options = default_chunk_options_from_lib.copy()
    effective_processing_options.update(form_options_cleaned)
    logger.debug(f"Effective chunking options from form data for file: {effective_processing_options}")

    # LLM config setup for file endpoint (mirroring the JSON endpoint logic)
    llm_call_func_to_use_file = None
    llm_api_config_to_use_file = None
    tokenizer_for_chunker_file = effective_processing_options.get("tokenizer_name_or_path", "gpt2")

    current_chunking_method_file = effective_processing_options.get('method')
    if current_chunking_method_file == 'rolling_summarize':
        llm_call_func_to_use_file = general_llm_analyzer
        server_configs_file = load_server_configs()
        if not server_configs_file:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error for LLM step (file).")

        internal_llm_provider_file = server_configs_file.get('llm_api_settings', {}).get('default_api_for_tasks',
                                          server_configs_file.get('llm_api_settings', {}).get('default_api', 'openai'))
        provider_specific_config_key_file = f"{internal_llm_provider_file}_api"
        api_details_server_file = server_configs_file.get(provider_specific_config_key_file, {})

        server_task_specific_model_file = api_details_server_file.get('model_for_summarization')
        server_general_model_file = api_details_server_file.get('model')
        internal_llm_model_file = server_task_specific_model_file or server_general_model_file

        if not internal_llm_model_file:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server config missing model for {internal_llm_provider_file} (file).")
        api_key_server_file = api_details_server_file.get('api_key')
        if not api_key_server_file:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Server config missing API key for {internal_llm_provider_file} (file).")

        requested_llm_params_file = effective_processing_options.get('llm_options_for_internal_steps', {})
        if requested_llm_params_file is None: requested_llm_params_file = {}

        client_suggested_system_prompt_file = requested_llm_params_file.get('system_prompt_for_step')
        method_default_system_prompt_file = effective_processing_options.get('summarize_system_prompt')
        final_system_prompt_step_file = client_suggested_system_prompt_file or method_default_system_prompt_file

        client_suggested_max_tokens_file = requested_llm_params_file.get('max_tokens_per_step')
        server_default_max_tokens_step_file = int(api_details_server_file.get('max_tokens_for_summarization_step', 1024))
        final_max_tokens_step_file = client_suggested_max_tokens_file or server_default_max_tokens_step_file

        llm_api_config_to_use_file = {
            "api_name": internal_llm_provider_file, "model": internal_llm_model_file,
            "api_key": api_key_server_file,
            "temp": requested_llm_params_file.get('temperature'),
            "system_message": final_system_prompt_step_file,
            "max_tokens": final_max_tokens_step_file,
        }
        logger.info(f"'{current_chunking_method_file}' for file will use server LLM: {internal_llm_provider_file}, Model: {internal_llm_model_file}.")


    loop = asyncio.get_running_loop()
    try:
        chunk_results: List[Dict[str, Any]] = await loop.run_in_executor(
            None, improved_chunking_process, text_content,
            effective_processing_options, tokenizer_for_chunker_file,
            llm_call_func_to_use_file, llm_api_config_to_use_file
        )
    except (ChunkingError, InvalidInputError, InvalidChunkingMethodError) as lib_error: # Catch specific errors
        logger.warning(f"Chunking library error for file '{file.filename}': {lib_error}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(lib_error))
    except ValueError as ve: # General value errors
        logger.warning(f"ValueError during chunking file '{file.filename}': {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during chunking file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error during file chunking: {type(e).__name__}")

    # FIXME chunks incorrect type
    return ChunkingResponse(
        chunks=chunk_results,
        original_file_name=file.filename,
        applied_options=ChunkingOptionsRequest(**effective_processing_options)
    )