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

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import (
    propagate_billing_headers,
    require_within_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import (
    get_process_audios_form,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.API_Deps.validations_deps import file_validator_instance
from tldw_Server_API.app.api.v1.endpoints import media as media_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessAudiosForm
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
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.api.v1.endpoints.media.input_contracts import (
    normalize_urls_field,
    validate_media_inputs,
)
from tldw_Server_API.app.api.v1.endpoints.media.deprecation_signals import (
    apply_media_legacy_headers,
    build_media_legacy_signal,
)

router = APIRouter()

_propagate_billing_headers = propagate_billing_headers


@router.post(
    "/process-audios",
    summary="Transcribe / chunk / analyse audio and return full artefacts (no DB write)",
    tags=["Media Processing (No DB)"],
    dependencies=[
        Depends(guard_storage_quota),
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def process_audios_endpoint(
    background_tasks: BackgroundTasks,
    injected_response: Response,
    db: Any = Depends(get_media_db_for_user),
    form_data: ProcessAudiosForm = Depends(get_process_audios_form),
    files: list[UploadFile] | None = File(
        None,
        description="Audio file uploads",
    ),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    """
    Process audio inputs (URLs and uploads) without persisting to the Media DB.

    This endpoint mirrors the legacy `/process-audios` behavior while routing
    through the modular `media` package and using shared helpers for input
    handling and batch orchestration.
    """

    logger.info("Request received for /process-audios. Form data validated via dependency.")
    try:
        usage_log.log_event(
            "media.process.audio",
            tags=["no_db"],
            metadata={"has_urls": bool(form_data.urls), "has_files": bool(files)},
        )
    except Exception:
        # Usage logging is best-effort; do not fail the request.
        logger.debug("Audio process endpoint usage logging failed")

    # Normalize the "urls=['']" sentinel used by some clients.
    legacy_urls_empty_sentinel_used = bool(form_data.urls and form_data.urls == [""])
    if legacy_urls_empty_sentinel_used:
        logger.info("Received urls=[''], treating as no URLs provided for audio processing.")
    form_data.urls = normalize_urls_field(form_data.urls)
    legacy_signal = (
        build_media_legacy_signal(
            successor="/api/v1/media/process-audios",
            warning_code="legacy_urls_empty_sentinel",
        )
        if legacy_urls_empty_sentinel_used
        else None
    )

    # Reuse shared validation so that error messages and 400 semantics match
    # the legacy implementation (including "No valid media sources supplied").
    try:
        validate_media_inputs(
            media_mod._validate_inputs,
            "audio",
            form_data.urls,
            files,
        )
    except HTTPException as exc:
        logger.warning("Input validation failed for /process-audios: {}", exc.detail)
        raise

    # Base batch structure used when we need to return an empty 207.
    empty_batch: dict[str, Any] = {
        "processed_count": 0,
        "errors_count": 0,
        "errors": [],
        "results": [],
    }

    # Lazy import to avoid import-time hard failures from optional STT backends.
    from tldw_Server_API.app.core.Ingestion_Media_Processing.audio_batch import (
        run_audio_batch,
    )

    # Map temporary path -> original filename for uploads.
    temp_path_to_original_name: dict[str, str] = {}
    saved_files: list[dict[str, Any]] = []
    chunk_options_dict: dict[str, Any] | None = None
    chunking_plan: dict[str, Any] | None = None

    with TempDirManager(cleanup=True, prefix="process_audio_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        logger.info("Using temporary directory for /process-audios: {}", temp_dir_path.as_posix())

        # Preserve test-time monkeypatching of `media.file_validator_instance`
        # by resolving the validator from the media module export.
        validator = getattr(
            media_mod,
            "file_validator_instance",
            file_validator_instance,
        )

        # Allowed audio file extensions (mirrors legacy implementation).
        allowed_audio_extensions = [
            ".mp3",
            ".aac",
            ".flac",
            ".wav",
            ".ogg",
            ".m4a",
        ]

        saved_files, file_errors_raw = await save_uploaded_files(
            files or [],
            temp_dir=temp_dir_path,
            validator=validator,
            allowed_extensions=allowed_audio_extensions,
        )

        # Build combined input list: URLs + uploaded temp paths.
        url_list = form_data.urls or []
        uploaded_paths = [str(sf["path"]) for sf in saved_files if sf.get("path")]
        all_inputs = url_list + uploaded_paths

        # If there are no inputs after attempting to save uploads, we may still
        # need to return a 207 based on how uploads were rejected.
        if not all_inputs:
            detail = "No valid audio sources supplied (or all uploads failed)."
            logger.warning("Request processing stopped: {}", detail)

            if file_errors_raw:
                # Determine if any error is a security block. For those cases
                # we surface structured error entries; otherwise we return an
                # empty batch with 207 to indicate the request was handled.
                security_block = any(
                    isinstance(err, dict)
                    and isinstance(err.get("error"), str)
                    and "security reasons" in err.get("error")
                    for err in file_errors_raw
                )

                # Use the batch helper to normalize file errors into the
                # standard batch shape for security-blocked uploads.
                batch_result = await run_audio_batch(
                    all_inputs=[],
                    form_data=form_data,
                    temp_dir=str(temp_dir_path),
                    temp_path_to_original_name=temp_path_to_original_name,
                    saved_files=saved_files,
                    file_errors_raw=file_errors_raw,
                )

                if security_block:
                    response = JSONResponse(
                        status_code=status.HTTP_207_MULTI_STATUS,
                        content=batch_result,
                    )
                    if legacy_signal is not None:
                        apply_media_legacy_headers(response, legacy_signal)
                    _propagate_billing_headers(injected_response, response)
                    return response

                response = JSONResponse(
                    status_code=status.HTTP_207_MULTI_STATUS,
                    content=empty_batch,
                )
                if legacy_signal is not None:
                    apply_media_legacy_headers(response, legacy_signal)
                _propagate_billing_headers(injected_response, response)
                return response

            # Otherwise, no inputs at all -> 400.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail,
            )

        # Normal path: we have at least one URL or uploaded file to process.
        batch_result = await run_audio_batch(
            all_inputs=all_inputs,
            form_data=form_data,
            temp_dir=str(temp_dir_path),
            temp_path_to_original_name=temp_path_to_original_name,
            saved_files=saved_files,
            file_errors_raw=file_errors_raw,
        )

    # Determine final HTTP status code based on the batch outcome.
    final_processed_count = batch_result.get("processed_count", 0)
    final_error_count = batch_result.get("errors_count", 0)
    total_items = len(batch_result.get("results", []))

    if total_items == 0:
        final_status_code = status.HTTP_400_BAD_REQUEST
        logger.error("No results generated for /process-audios despite processing attempt.")
    elif final_error_count == 0 and final_processed_count > 0:
        final_status_code = status.HTTP_200_OK
        logger.info(
            "/process-audios request finished with status {}. Results: {}, Errors: {}",
            final_status_code,
            total_items,
            final_error_count,
        )
    else:
        # Mixed or all-error batches return 207.
        final_status_code = status.HTTP_207_MULTI_STATUS
        logger.warning(
            "/process-audios request finished with status {}. Results: {}, Errors: {}",
            final_status_code,
            total_items,
            final_error_count,
        )

        # In TEST_MODE, emit a more explicit debug log when the CDN-hosted
        # audio URL used in tests fails due to egress/DNS issues so that
        # environment-induced skips can be distinguished from real regressions.
        if is_test_mode():
            try:
                errors_joined = " | ".join(str(e) for e in batch_result.get("errors", []) if e)
                if "Download failed" in errors_joined or "Host could not be resolved" in errors_joined:
                    logger.debug(
                        "TEST_MODE: /process-audios returned 207 due to audio " "download/egress error: {}",
                        errors_joined,
                    )
            except Exception:
                # Logging must never affect endpoint behavior.
                logger.debug("Audio process endpoint warning log formatting failed")

    # Optional template/hierarchical re-chunking of transcripts (best-effort).
    try:
        if form_data.perform_chunking:
            first_url = (form_data.urls or [None])[0]
            first_filename = None
            try:
                if saved_files:
                    first_filename = saved_files[0].get("original_filename")
            except Exception:
                first_filename = None
            first_input_source = str(all_inputs[0]) if all_inputs else first_url or first_filename

            chunk_options_dict, chunking_plan = resolve_chunking_options_and_plan(
                form_data,
                media_type="audio",
                source_name=first_input_source,
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
                # Audio results may store transcript under 'content'
                text = res.get("content")
                if not isinstance(text, str) or not text.strip():
                    attach_chunking_plan_to_result(res, chunking_plan)
                    continue

                result_chunk_options, result_chunking_plan = await async_resolve_chunking_for_result(
                    form_data,
                    res,
                    media_type="audio",
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
        logger.warning("Best-effort audio chunking post-processing failed; leaving results unchunked")

    response = JSONResponse(status_code=final_status_code, content=batch_result)
    if legacy_signal is not None:
        apply_media_legacy_headers(response, legacy_signal)
    _propagate_billing_headers(injected_response, response)
    return response


__all__ = ["router"]
