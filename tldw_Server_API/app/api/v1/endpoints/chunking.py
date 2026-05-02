# Server_API/app/api/v1/endpoints/text_processing.py
# Description: This code provides FastAPI endpoints for text processing, including chunking.
#
# Imports
import asyncio
from typing import Any, Optional, get_args

from fastapi import (
    APIRouter,
    Body,
    Depends,  # Added for dependency injection
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

#
# Third-party Libraries
from loguru import logger

# Local Imports
from tldw_Server_API.app.core.Chunking import (
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
    improved_chunking_process,
)

# Default chunking options
default_chunk_options_from_lib = {
    'method': 'words',
    'max_size': 400,
    'overlap': 200,
    'language': 'en',
    'adaptive': False,
    'multi_level': False,
    'semantic_similarity_threshold': 0.7,
    'semantic_overlap_sentences': 2,
    'json_chunkable_data_key': 'data',
    'summarization_detail': 0.5,
    'tokenizer_name_or_path': 'gpt2'
}
# Dependencies for user-specific database access
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.chunking_schema import (
    ChunkedContentResponse,
    ChunkingCapabilitiesResponse,
    ChunkingOptionsRequest,
    ChunkingResponse,
    ChunkingTextRequest,
    CodeMethodOptions,
    MethodSpecificOptions,
    build_chunking_options_schema,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import PdfEngine
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    record_byok_missing_credentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking.base import ChunkingMethod
from tldw_Server_API.app.core.config import load_and_log_configs as load_server_configs
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze as general_llm_analyzer

#
#######################################################################################################################
#
# Functions:

async def _resolve_chunking_byok(
    provider: str,
    *,
    current_user: User,
    request: Request,
    fallback_key: Optional[str],
):
    try:
        user_id_int = int(getattr(current_user, "id", None))
    except (TypeError, AttributeError, ValueError):
        user_id_int = None

    def _fallback_resolver(_name: str) -> Optional[str]:
        return fallback_key

    return await resolve_byok_credentials(
        provider,
        user_id=user_id_int,
        request=request,
        fallback_resolver=_fallback_resolver,
    )


def _raise_missing_chunking_key(provider: str) -> None:
    record_byok_missing_credentials(provider, operation="chunking")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error_code": "missing_provider_credentials",
            "message": f"Provider '{provider}' requires an API key for chunking.",
        },
    )


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
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error in request payload."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during chunking."},
    }
)
async def process_text_for_chunking_json(
    request_data: ChunkingTextRequest = Body(...),
    *,
    http_request: Request,
    current_user: User = Depends(get_request_user),
    media_db: Optional[Any] = Depends(try_get_media_db_for_user),
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

    # Check if a template was specified
    template_used = False
    if request_data.options and request_data.options.template_name:
        # Import necessary modules for template support
        import json

        from tldw_Server_API.app.core.Chunking.templates import ChunkingTemplate, TemplateProcessor, TemplateStage

        try:
            # Use the injected user-specific database instance
            # Get template from database
            if not media_db:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail="Chunking template requested but user database is unavailable.")
            template_data = media_db.get_chunking_template(name=request_data.options.template_name)
            if template_data:
                logger.info(f"Using chunking template: {request_data.options.template_name}")
                template_used = True

                # Parse template JSON and prepare options
                template_config = json.loads(template_data['template_json'])

                # Start with template's chunking config
                effective_options = template_config.get('chunking', {}).get('config', {}).copy()
                effective_options['method'] = template_config.get('chunking', {}).get('method', 'words')

                # Allow explicit request options to override template defaults
                request_options_dict = request_data.options.model_dump(exclude_unset=True)
                # Remove template_name from override options
                request_options_dict.pop('template_name', None)
                effective_options.update(request_options_dict)

                logger.debug(f"Template-based effective options: {effective_options}")

                # Build Template object (DB schema -> stages)
                stages = []
                if 'preprocessing' in template_config:
                    stages.append(TemplateStage(
                        name='preprocess',
                        operations=template_config['preprocessing'],
                        enabled=True
                    ))
                stages.append(TemplateStage(
                    name='chunk',
                    operations=[template_config['chunking']],
                    enabled=True
                ))
                if 'postprocessing' in template_config:
                    stages.append(TemplateStage(
                        name='postprocess',
                        operations=template_config['postprocessing'],
                        enabled=True
                    ))

                template_obj = ChunkingTemplate(
                    name=template_data['name'],
                    description=template_data['description'] or "",
                    base_method=template_config['chunking']['method'],
                    stages=stages,
                    default_options=template_config['chunking'].get('config', {}),
                    metadata={'tags': template_data['tags']}
                )

                # If template uses LLM-heavy methods, prepare a configured Chunker
                configured_chunker = None
                byok_resolution_tmp = None
                current_chunking_method_tmp = effective_options.get('method')
                if current_chunking_method_tmp == 'rolling_summarize':
                    # Reuse the same provider/model selection logic from below
                    server_configs = load_server_configs()
                    if not server_configs:
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                            detail="Server configuration error, cannot perform LLM-dependent chunking.")
                    requested_llm_options = effective_options.get('llm_options_for_internal_steps', {}) or {}
                    default_summarization_provider = server_configs.get('llm_api_settings', {}).get('default_api', 'openai')
                    summarization_provider = requested_llm_options.get('provider') or default_summarization_provider
                    provider_specific_config_key = f"{summarization_provider}_api"
                    api_details_from_server_config = server_configs.get(provider_specific_config_key, {})
                    byok_resolution_tmp = await _resolve_chunking_byok(
                        summarization_provider,
                        current_user=current_user,
                        request=http_request,
                        fallback_key=api_details_from_server_config.get('api_key'),
                    )
                    api_key_value = byok_resolution_tmp.api_key
                    final_model_for_step = api_details_from_server_config.get('model_for_summarization') or api_details_from_server_config.get('model')
                    provider_key = (summarization_provider or "").strip().lower()
                    if not api_key_value and provider_requires_api_key(provider_key):
                        _raise_missing_chunking_key(provider_key)
                    if not final_model_for_step:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Configuration error: Missing model for {summarization_provider}.",
                        )
                    client_suggested_system_prompt = requested_llm_options.get('system_prompt_for_step')
                    method_default_system_prompt = effective_options.get('summarize_system_prompt')
                    final_system_prompt_for_step = client_suggested_system_prompt or method_default_system_prompt
                    final_max_tokens_for_step = requested_llm_options.get('max_tokens_per_step') or int(api_details_from_server_config.get('max_tokens_for_summarization_step', 1024))
                    llm_api_config_to_use_tmp = {
                        "api_name": summarization_provider,
                        "model": final_model_for_step,
                        "api_key": api_key_value,
                        "temp": requested_llm_options.get('temperature'),
                        "system_message": final_system_prompt_for_step,
                        "max_tokens": final_max_tokens_for_step,
                    }
                    configured_chunker = Chunker(llm_call_func=general_llm_analyzer, llm_config=llm_api_config_to_use_tmp)

                # Process via TemplateProcessor (with optional configured chunker)
                processor = TemplateProcessor(chunker=configured_chunker)
                chunks = processor.process_template(
                    text=request_data.text_content,
                    template=template_obj,
                    **{k: v for k, v in effective_options.items() if k not in {"method"}}
                )

                # Build minimal metadata for response
                total = len(chunks)
                chunked_responses = []
                for idx, chunk in enumerate(chunks):
                    meta = {
                        "chunk_index": idx,
                        "total_chunks": total,
                        "chunk_method": template_obj.base_method,
                        "max_size": effective_options.get('max_size'),
                        "overlap": effective_options.get('overlap'),
                        "language": effective_options.get('language'),
                        "relative_position": 0.0 if total <= 1 else idx/(total-1),
                        "template_applied": template_obj.name,
                        "template_version": template_data.get('version', 1),
                    }
                    chunked_responses.append(ChunkedContentResponse(text=chunk, metadata=meta))

                # Applied options include template_name for clarity
                applied_opts = dict(effective_options)
                applied_opts['template_name'] = request_data.options.template_name
                if byok_resolution_tmp and byok_resolution_tmp.uses_byok:
                    await byok_resolution_tmp.touch_last_used()
                return ChunkingResponse(
                    chunks=chunked_responses,
                    original_file_name=request_data.file_name,
                    applied_options=ChunkingOptionsRequest(**applied_opts)
                )
            else:
                logger.warning(f"Template '{request_data.options.template_name}' not found, falling back to regular options")
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            # Fall back to regular processing

    if not template_used:
        # Prepare effective chunking options (original logic)
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
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Option '{key_to_check}' must be an integer. Value: {effective_options[key_to_check]}. Error: {e}"
                ) from e


    # Filename-based language hint if language not set/empty
    try:
        if (not effective_options.get('language')) and request_data.file_name:
            fname = str(request_data.file_name)
            ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''
            ext_map = {
                'py': 'python', 'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript', 'tsx': 'typescript',
                'java': 'java', 'rb': 'ruby', 'rs': 'rust', 'go': 'go', 'kt': 'kotlin', 'swift': 'swift',
                'c': 'c', 'cc': 'cpp', 'cxx': 'cpp', 'cpp': 'cpp', 'hpp': 'cpp', 'h': 'c',
            }
            if ext in ext_map:
                effective_options['language'] = ext_map[ext]
    except Exception as ext_detect_error:
        logger.debug(
            "Failed to infer code language from file extension ({})",
            type(ext_detect_error).__name__,
        )

    logger.debug(f"Effective chunking options before LLM setup: {effective_options}")

    # --- LLM Configuration for specific chunking methods ---
    llm_call_func_to_use = None
    llm_api_config_to_use = None
    byok_resolution = None
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
        if requested_llm_options is None:
            requested_llm_options = {}


        default_summarization_provider = server_configs.get('llm_api_settings', {}).get('default_api', 'openai')
        summarization_provider = requested_llm_options.get('provider') or default_summarization_provider

        provider_specific_config_key = f"{summarization_provider}_api" # e.g., "openai_api"
        api_details_from_server_config = server_configs.get(provider_specific_config_key, {})
        byok_resolution = await _resolve_chunking_byok(
            summarization_provider,
            current_user=current_user,
            request=http_request,
            fallback_key=api_details_from_server_config.get('api_key'),
        )
        api_key_value = byok_resolution.api_key

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
            "api_key": api_key_value,
            "temp": requested_llm_options.get('temperature'), # If None, general_llm_analyzer will use its own default/config
            "system_message": final_system_prompt_for_step,
            "max_tokens": final_max_tokens_for_step,
        }

        provider_key = (summarization_provider or "").strip().lower()
        if not llm_api_config_to_use.get("api_key") and provider_requires_api_key(provider_key):
            logger.error(
                "API key for '{}' for internal summarization step not found in server configuration.",
                summarization_provider,
            )
            _raise_missing_chunking_key(provider_key)
        if not llm_api_config_to_use.get("model"):
            logger.error(f"Model for '{summarization_provider}' for internal summarization step not determined.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Configuration error: Missing model for {summarization_provider} for internal LLM step.")

        logger.info(f"'{current_chunking_method}' will use LLM provider: {summarization_provider}, Model: {llm_api_config_to_use['model']}")


    # --- Perform Chunking ---
    loop = asyncio.get_running_loop()
    try:
        chunk_results: list[dict[str, Any]] = await loop.run_in_executor(
            None,
            improved_chunking_process,
            request_data.text_content,
            effective_options,  # Pass the full dict of resolved options
            tokenizer_for_chunker, # Pass the selected tokenizer
            llm_call_func_to_use,      # Pass the prepared LLM function
            llm_api_config_to_use      # Pass the config for that LLM function
        )
    except (ChunkingError, InvalidInputError, InvalidChunkingMethodError) as lib_error: # Catch specific errors from chunker
        logger.warning(f"Chunking library error for '{request_data.file_name}': {lib_error}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(lib_error)) from lib_error
    except ValueError as ve: # General value errors (e.g., from Pydantic or type conversions if not caught earlier)
        logger.warning(f"ValueError during chunking setup or process for '{request_data.file_name}': {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except Exception as e:
        logger.error("Unexpected error during chunking process ({})", type(e).__name__)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An internal error occurred during text chunking") from e

    if not chunk_results:
        logger.info(f"Chunking produced no results for '{request_data.file_name}'. Returning empty list.")

    # Convert chunk_results to ChunkedContentResponse objects
    chunked_responses = [
        ChunkedContentResponse(
            text=chunk['text'],
            metadata=chunk['metadata']
        )
        for chunk in chunk_results
    ]

    if byok_resolution and byok_resolution.uses_byok:
        await byok_resolution.touch_last_used()

    return ChunkingResponse(
        chunks=chunked_responses,
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
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error in form parameters."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during chunking."},
    }
)
async def process_file_for_chunking(
    http_request: Request,
    file: UploadFile = File(...),
    # Form fields for chunking options
    method: Optional[str] = Form(default_chunk_options_from_lib.get('method')),
    max_size: Optional[int] = Form(default_chunk_options_from_lib.get('max_size')),
    overlap: Optional[int] = Form(default_chunk_options_from_lib.get('overlap')),
    language: Optional[str] = Form(None), # Default to None for auto-detection
    tokenizer_name_or_path: Optional[str] = Form(default_chunk_options_from_lib.get('tokenizer_name_or_path', "gpt2")),
    code_mode: Optional[str] = Form(None, description="For method='code': 'auto' | 'ast' | 'heuristic'"),
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
    current_user: User = Depends(get_request_user),
):
    logger.info(f"Received file upload for chunking: '{file.filename}'. Method from form: {method}.")

    if not file.filename: # Should not happen with File(...) but good check
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided or filename is missing.")

    try:
        text_content_bytes = await file.read()
        text_content = text_content_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading uploaded file '{file.filename}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read or decode file: {e}") from e
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
    if llm_step_temperature is not None:
        internal_llm_opts_from_form['temperature'] = llm_step_temperature
    if llm_step_system_prompt is not None:
        internal_llm_opts_from_form['system_prompt_for_step'] = llm_step_system_prompt
    if llm_step_max_tokens is not None:
        internal_llm_opts_from_form['max_tokens_per_step'] = llm_step_max_tokens

    if internal_llm_opts_from_form:
        form_options_dict['llm_options_for_internal_steps'] = internal_llm_opts_from_form

    # Filter out None values from the top level to allow library defaults
    form_options_cleaned = {k: v for k, v in form_options_dict.items() if v is not None}
    # Filename-based language hint if not provided
    try:
        if not form_options_cleaned.get('language') and file and file.filename:
            ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
            ext_map = {
                'py': 'python', 'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript', 'tsx': 'typescript',
                'java': 'java', 'rb': 'ruby', 'rs': 'rust', 'go': 'go', 'kt': 'kotlin', 'swift': 'swift',
                'c': 'c', 'cc': 'cpp', 'cxx': 'cpp', 'cpp': 'cpp', 'hpp': 'cpp', 'h': 'c',
            }
            if ext in ext_map:
                form_options_cleaned['language'] = ext_map[ext]
    except Exception as ext_detect_error:
        logger.debug(
            "Failed to infer cleaned form language from file extension ({})",
            type(ext_detect_error).__name__,
        )
    if code_mode is not None:
        form_options_cleaned['code_mode'] = code_mode

    effective_processing_options = default_chunk_options_from_lib.copy()
    effective_processing_options.update(form_options_cleaned)
    logger.debug(f"Effective chunking options from form data for file: {effective_processing_options}")

    # LLM config setup for file endpoint (mirroring the JSON endpoint logic)
    llm_call_func_to_use_file = None
    llm_api_config_to_use_file = None
    byok_resolution_file = None
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
        byok_resolution_file = await _resolve_chunking_byok(
            internal_llm_provider_file,
            current_user=current_user,
            request=http_request,
            fallback_key=api_details_server_file.get('api_key'),
        )
        api_key_server_file = byok_resolution_file.api_key
        provider_key_file = (internal_llm_provider_file or "").strip().lower()
        if not api_key_server_file and provider_requires_api_key(provider_key_file):
            _raise_missing_chunking_key(provider_key_file)

        requested_llm_params_file = effective_processing_options.get('llm_options_for_internal_steps', {})
        if requested_llm_params_file is None:
            requested_llm_params_file = {}

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
        chunk_results: list[dict[str, Any]] = await loop.run_in_executor(
            None, improved_chunking_process, text_content,
            effective_processing_options, tokenizer_for_chunker_file,
            llm_call_func_to_use_file, llm_api_config_to_use_file
        )
    except (ChunkingError, InvalidInputError, InvalidChunkingMethodError) as lib_error: # Catch specific errors
        logger.warning(f"Chunking library error for file '{file.filename}': {lib_error}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(lib_error)) from lib_error
    except ValueError as ve: # General value errors
        logger.warning(f"ValueError during chunking file '{file.filename}': {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except Exception as e:
        logger.error("Unexpected error during chunking file ({})", type(e).__name__)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error during file chunking") from e

    # Convert chunk_results to ChunkedContentResponse objects
    chunked_responses = [
        ChunkedContentResponse(
            text=chunk['text'],
            metadata=chunk['metadata']
        )
        for chunk in chunk_results
    ]

    if byok_resolution_file and byok_resolution_file.uses_byok:
        await byok_resolution_file.touch_last_used()

    return ChunkingResponse(
        chunks=chunked_responses,
        original_file_name=file.filename,
        applied_options=ChunkingOptionsRequest(**effective_processing_options)
    )


# --- Capabilities Endpoint ---
@chunking_router.get(
    "/capabilities",
    summary="List chunking methods and defaults",
    tags=["Text Processing", "Chunking"],
    response_model=ChunkingCapabilitiesResponse,
)
async def get_chunking_capabilities(
    current_user: User = Depends(get_request_user),
):
    # Prefer runtime registry from Chunker to include non-enum methods like 'structure_aware'
    try:
        runtime_methods = Chunker().get_available_methods()
    except Exception:
        runtime_methods = []
    enum_methods = [m.value for m in ChunkingMethod]
    # Merge and de-duplicate
    methods = sorted(set(enum_methods + runtime_methods))
    llm_required = [m for m in ["rolling_summarize", "propositions"] if m in methods]
    pdf_engines = list(get_args(PdfEngine))
    return {
        "methods": methods,
        "default_options": default_chunk_options_from_lib,
        "llm_required_methods": llm_required,
        "hierarchical_support": True,
        "notes": "Text chunking capabilities. For method='code', the option 'code_mode' controls routing: 'auto' (default), 'ast' (Python), or 'heuristic'. Ingestion-specific chunkers are configured via templates or step config.",
        "pdf_parsing_engines": pdf_engines,
        "options_schema": build_chunking_options_schema(),
        "method_specific_options": MethodSpecificOptions(
            code=CodeMethodOptions(
                code_mode=["auto", "ast", "heuristic"],
                language_hints={
                    "py": "python", "js": "javascript", "jsx": "javascript", "ts": "typescript", "tsx": "typescript"
                },
            )
        ),
    }
