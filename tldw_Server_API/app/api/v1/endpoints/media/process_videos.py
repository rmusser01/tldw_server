from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from loguru import logger
from starlette.responses import JSONResponse
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import propagate_billing_headers, require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import (
    get_process_videos_form,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.API_Deps.validations_deps import file_validator_instance
from tldw_Server_API.app.api.v1.endpoints import media as media_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessVideosForm
from tldw_Server_API.app.core.AuthNZ.permissions import (
    MEDIA_CREATE,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    apply_chunking_template_if_any,
    async_resolve_chunking_for_result,
    attach_chunking_plan_to_result,
    resolve_chunking_options_and_plan,
    uses_hierarchical_chunking,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    TempDirManager,
    save_uploaded_files,
)
from tldw_Server_API.app.api.v1.endpoints.media.input_contracts import (
    normalize_urls_field,
    validate_media_inputs,
)
from tldw_Server_API.app.api.v1.endpoints.media.deprecation_signals import (
    apply_media_legacy_headers,
    build_media_legacy_signal,
)

router = APIRouter()


@router.post(
    "/process-videos",
    summary="Transcribe / chunk / analyse videos and return the full artefacts (no DB write)",
    tags=["Media Processing (No DB)"],
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
        Depends(guard_storage_quota),
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def process_videos_endpoint(
    background_tasks: BackgroundTasks,
    injected_response: Response,
    db: Any = Depends(get_media_db_for_user),
    form_data: ProcessVideosForm = Depends(get_process_videos_form),
    files: list[UploadFile] | None = File(
        None,
        description="Video file uploads",
    ),
    current_user: User = Depends(get_request_user),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    """
    Process videos without persisting to the Media DB.

    This endpoint mirrors the legacy `/process-videos` behavior while routing
    through the modular `media` package and using shared helpers for input
    handling and batch orchestration.
    """

    # --- Validation and Logging ---
    logger.info("Request received for /process-videos. Form data validated via dependency.")

    # Lazy import to avoid import-time hard failures from optional transcriber backends.
    from tldw_Server_API.app.core.Ingestion_Media_Processing.video_batch import (
        run_video_batch,
    )

    try:
        usage_log.log_event(
            "media.process.video",
            tags=["no_db"],
            metadata={"has_urls": bool(form_data.urls), "has_files": bool(files)},
        )
    except Exception:
        # Usage logging is best-effort; do not fail the request.
        logger.debug("Video process endpoint usage logging failed")

    legacy_urls_empty_sentinel_used = bool(form_data.urls and form_data.urls == [""])
    if legacy_urls_empty_sentinel_used:
        logger.info("Received urls=[''], treating as no URLs provided for video processing.")
    form_data.urls = normalize_urls_field(form_data.urls)
    legacy_signal = (
        build_media_legacy_signal(
            successor="/api/v1/media/process-videos",
            warning_code="legacy_urls_empty_sentinel",
        )
        if legacy_urls_empty_sentinel_used
        else None
    )

    # Reuse shared validation so that error messages and 400 semantics match
    # the legacy implementation (including "No valid media sources supplied").
    validate_media_inputs(
        media_mod._validate_inputs,
        "video",
        form_data.urls,
        files,
    )

    batch_result: dict[str, Any] = {
        "processed_count": 0,
        "errors_count": 0,
        "errors": [],
        "results": [],
        "confabulation_results": None,
    }
    file_handling_errors_structured: list[dict[str, Any]] = []
    # Map temporary path -> original filename
    temp_path_to_original_name: dict[str, str] = {}
    chunk_options_dict: dict[str, Any] | None = None
    chunking_plan: dict[str, Any] | None = None

    # --- Use TempDirManager for reliable cleanup ---
    with TempDirManager(cleanup=True, prefix="process_video_") as temp_dir:
        logger.info(f"Using temporary directory for /process-videos: {temp_dir}")
        temp_dir_path = Path(temp_dir)

        # Preserve test-time monkeypatching of `media.file_validator_instance`
        # by resolving the validator from the media module export.
        validator = getattr(
            media_mod,
            "file_validator_instance",
            file_validator_instance,
        )

        # --- Save Uploads ---
        saved_files_info, file_handling_errors_raw = await save_uploaded_files(
            files or [],
            temp_dir=temp_dir_path,
            validator=validator,
        )

        # Populate the temp path to original name map.
        for sf in saved_files_info:
            if sf.get("path") and sf.get("original_filename"):
                temp_path_to_original_name[str(sf["path"])] = sf["original_filename"]
            else:
                logger.warning(f"Missing path or original_filename in saved_files_info item: {sf}")

        # Process file-handling errors into the response structure.
        if file_handling_errors_raw:
            batch_result["errors_count"] += len(file_handling_errors_raw)
            batch_result["errors"].extend(
                [err.get("error", "Unknown file save error") for err in file_handling_errors_raw]
            )
            for err in file_handling_errors_raw:
                input_ref = err.get("input_ref") or err.get("original_filename") or err.get("input") or "Unknown Upload"
                file_handling_errors_structured.append(
                    {
                        "status": "Error",
                        "input_ref": input_ref,
                        "processing_source": "N/A - File Save Failed",
                        "media_type": "video",
                        "metadata": {},
                        "content": "",
                        "segments": None,
                        "chunks": None,
                        "analysis": None,
                        "analysis_details": {},
                        "error": err.get("error", "Failed to save uploaded file."),
                        "warnings": None,
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                        "message": None,
                    }
                )
            batch_result["results"].extend(file_handling_errors_structured)

        # --- Prepare Inputs for Processing ---
        url_list = form_data.urls or []
        uploaded_paths = [str(sf["path"]) for sf in saved_files_info if sf.get("path")]
        all_inputs_to_process = url_list + uploaded_paths

        # Check if there's anything left to process.
        if not all_inputs_to_process:
            if file_handling_errors_raw:
                logger.warning("No valid video sources to process after file saving errors.")
                # Return 207 with the structured file errors.
                response = JSONResponse(
                    status_code=status.HTTP_207_MULTI_STATUS,
                    content=batch_result,
                )
                if legacy_signal is not None:
                    apply_media_legacy_headers(response, legacy_signal)
                propagate_billing_headers(injected_response, response)
                return response

            logger.warning("No video sources provided.")
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "No valid video sources supplied.",
            )

        # --- Call process_videos via helper ---
        batch_result = await run_video_batch(
            all_inputs_to_process=all_inputs_to_process,
            form_data=form_data,
            current_user=current_user,
            temp_dir=str(temp_dir_path),
            temp_path_to_original_name=temp_path_to_original_name,
            file_handling_errors_structured=file_handling_errors_structured,
        )

    # --- Determine Final Status Code & Return ---
    final_error_count = batch_result.get("errors_count", 0)
    final_success_count = batch_result.get("processed_count", 0)
    total_items = len(batch_result.get("results", []))
    has_warnings = any(r.get("status") == "Warning" for r in batch_result.get("results", []))
    # NOTE: `has_warnings` is currently unused but kept for parity/debugging.
    _ = has_warnings, final_success_count

    if total_items == 0:
        # Should not happen if validation passed, but handle defensively.
        final_status_code = status.HTTP_400_BAD_REQUEST
        logger.error("No results generated despite processing attempt.")
    elif final_error_count == 0:
        final_status_code = status.HTTP_200_OK
    elif final_error_count == total_items:
        # All errors, could also be 4xx/5xx depending on cause; keep legacy 207.
        final_status_code = status.HTTP_207_MULTI_STATUS
    else:
        # Mix of success/warnings/errors.
        final_status_code = status.HTTP_207_MULTI_STATUS

    log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
    logger.log(
        log_level,
        "/process-videos request finished with status {}. Results count: {}, " "Errors: {}",
        final_status_code,
        total_items,
        final_error_count,
    )

    # TEMPORARY DEBUG (kept for parity with legacy implementation).
    try:
        logger.debug("Final batch_result before JSONResponse:")
        logged_result = batch_result.copy()
        if len(logged_result.get("results", [])) > 5:
            logged_result["results"] = logged_result["results"][:5] + [
                {"message": "... remaining results truncated for logging ..."}
            ]
        logger.debug(
            "{}",
            logged_result,
        )

        success_item_debug = next(
            (r for r in batch_result.get("results", []) if r.get("status") == "Success"),
            None,
        )
        if success_item_debug:
            logger.debug(
                "Value of input_ref for success item before return: {}",
                success_item_debug.get("input_ref"),
            )
        else:
            logger.debug("No success item found in final results before return.")
    except Exception:  # pragma: no cover - defensive logging
        logger.error("Video process endpoint debug logging failed")

    # Optional template/hierarchical re-chunking of video transcripts (best-effort).
    try:
        if form_data.perform_chunking:
            first_url = (form_data.urls or [None])[0]
            first_filename = None
            try:
                if saved_files_info:
                    first_filename = saved_files_info[0].get("original_filename")
            except Exception:
                first_filename = None

            chunk_options_dict, chunking_plan = resolve_chunking_options_and_plan(
                form_data,
                media_type="video",
                source_name=first_filename or first_url,
            )
            try:
                TemplateClassifier = getattr(media_mod, "TemplateClassifier", None)
            except Exception:
                TemplateClassifier = None

            if chunk_options_dict is not None and chunking_plan is None:
                chunk_options_dict = apply_chunking_template_if_any(
                    form_data=form_data,
                    db=db,
                    chunking_options_dict=chunk_options_dict,
                    TemplateClassifier=TemplateClassifier,
                    first_url=first_url,
                    first_filename=first_filename,
                )

        if form_data.perform_chunking and chunk_options_dict:
            from tldw_Server_API.app.core.Chunking import (  # type: ignore
                improved_chunking_process as _improved_chunking_process,
            )
            from tldw_Server_API.app.core.Chunking.chunker import (  # type: ignore
                Chunker as _Chunker,
            )

            ck: _Chunker | None = None
            batch_auto_chunk_options = chunk_options_dict
            batch_auto_chunking_plan = chunking_plan
            batch_llm_chunking_resolved = False

            for res in batch_result.get("results", []):
                if not isinstance(res, dict):
                    continue
                status_value = str(res.get("status", "")).lower()
                if status_value not in {"success", "warning"}:
                    continue
                text = res.get("content")
                if not isinstance(text, str) or not text.strip():
                    attach_chunking_plan_to_result(res, chunking_plan)
                    continue

                result_chunk_options, result_chunking_plan = await async_resolve_chunking_for_result(
                    form_data,
                    res,
                    media_type="video",
                    default_chunk_options=batch_auto_chunk_options,
                    default_chunking_plan=batch_auto_chunking_plan,
                    allow_llm_assist=not batch_llm_chunking_resolved,
                )
                attach_chunking_plan_to_result(res, result_chunking_plan)
                if getattr(form_data, "auto_chunking_use_llm", False) and result_chunking_plan:
                    batch_auto_chunk_options = result_chunk_options
                    batch_auto_chunking_plan = result_chunking_plan
                    batch_llm_chunking_resolved = True
                if not result_chunk_options:
                    continue

                if uses_hierarchical_chunking(result_chunk_options):
                    ck = ck or _Chunker()
                    chunks = ck.chunk_text_hierarchical_flat(
                        text,
                        method=result_chunk_options.get("method") or "sentences",
                        max_size=result_chunk_options.get("max_size") or 500,
                        overlap=result_chunk_options.get("overlap") or 200,
                        language=result_chunk_options.get("language"),
                        template=(
                            result_chunk_options.get("hierarchical_template")
                            if isinstance(result_chunk_options.get("hierarchical_template"), dict)
                            else None
                        ),
                    )
                else:
                    chunks = _improved_chunking_process(text, result_chunk_options)

                res["chunks"] = chunks
    except Exception:
        logger.debug("Video process endpoint rechunking failed; returning original result")

    response = JSONResponse(status_code=final_status_code, content=batch_result)
    if legacy_signal is not None:
        apply_media_legacy_headers(response, legacy_signal)
    propagate_billing_headers(injected_response, response)
    return response


__all__ = ["router"]
