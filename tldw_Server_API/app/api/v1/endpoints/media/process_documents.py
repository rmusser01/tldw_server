from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Response, UploadFile, status
from loguru import logger
from starlette.responses import JSONResponse

import tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files as docs
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import propagate_billing_headers, require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import (
    get_process_documents_form,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.API_Deps.validations_deps import file_validator_instance
from tldw_Server_API.app.api.v1.endpoints import media as media_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessDocumentsForm
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    apply_chunking_template_if_any,
    async_resolve_chunking_for_result,
    attach_chunking_plan_to_result,
    resolve_chunking_options_and_plan,
    uses_hierarchical_chunking,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    TempDirManager,
    save_uploaded_files as core_save_uploaded_files,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils import (
    download_url_async as core_download_url_async,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.pipeline import (
    ProcessItem,
    run_batch_processor,
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


ALLOWED_DOC_EXTENSIONS = [
    ".txt",
    ".md",
    ".docx",
    ".rtf",
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".json",
]


@router.post(
    "/process-documents",
    summary="Extract, chunk, analyse Documents (NO DB Persistence)",
    tags=["Media Processing (No DB)"],
    dependencies=[
        Depends(guard_storage_quota),
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def process_documents_endpoint(
    injected_response: Response,
    db: Any = Depends(get_media_db_for_user),
    form_data: ProcessDocumentsForm = Depends(get_process_documents_form),
    files: list[UploadFile] | None = File(
        None,
        description="Document file uploads (.txt, .md, .docx, .rtf, .html, .xml)",
    ),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    """
    Process Documents (No Persistence).

    This is a modularized version of the original legacy implementation,
    preserving behavior while
    routing through the `media` package with core ingestion helpers.
    """

    logger.info("Request received for /process-documents (no persistence).")
    try:
        usage_log.log_event(
            "media.process.document",
            tags=["no_db"],
            metadata={"has_urls": bool(form_data.urls), "has_files": bool(files)},
        )
    except Exception:
        # Usage logging is best-effort; never fail the request.
        logger.debug("Document process endpoint usage logging failed")
    logger.debug(
        "Form data for /process-documents: has_urls={}, has_files={}, " "perform_analysis={}, perform_chunking={}",
        bool(form_data.urls),
        bool(files),
        form_data.perform_analysis,
        form_data.perform_chunking,
    )
    legacy_urls_empty_sentinel_used = bool(form_data.urls and form_data.urls == [""])
    if legacy_urls_empty_sentinel_used:
        logger.info("Received urls=[''], treating as no URLs provided for document processing.")
    form_data.urls = normalize_urls_field(form_data.urls)
    legacy_signal = (
        build_media_legacy_signal(
            successor="/api/v1/media/process-documents",
            warning_code="legacy_urls_empty_sentinel",
        )
        if legacy_urls_empty_sentinel_used
        else None
    )

    # Guardrails: restrict to a known set of document extensions for this endpoint.
    validate_media_inputs(
        media_mod._validate_inputs,
        "document",
        form_data.urls,
        files,
    )

    # --- Prepare result structure ---
    batch_result: dict[str, Any] = {
        "errors": [],
        "results": [],
    }
    saved_files_info: list[dict[str, Any]] = []
    # Map to track original ref -> temp path
    source_map: dict[str, Path] = {}

    asyncio.get_running_loop()
    # Use TempDirManager for reliable cleanup
    with TempDirManager(
        cleanup=(not form_data.keep_original_file),
        prefix="process_doc_",
    ) as temp_dir_path:
        temp_dir = Path(temp_dir_path)
        logger.info("Using temporary directory for /process-documents: {}", temp_dir)

        local_paths_to_process: list[tuple[str, Path]] = []

        # --- Handle Uploads ---
        if files:
            # Preserve test-time monkeypatching of `media.file_validator_instance`
            # while using the canonical core upload helper.
            save_uploaded_files = core_save_uploaded_files
            validator = getattr(
                media_mod,
                "file_validator_instance",
                file_validator_instance,
            )

            saved_files, upload_errors = await save_uploaded_files(
                files,
                temp_dir,
                validator=validator,
                allowed_extensions=ALLOWED_DOC_EXTENSIONS,
            )
            saved_files_info = list(saved_files)
            # Add file saving/validation errors to batch_result
            for err_info in upload_errors:
                original_filename = err_info.get("input") or err_info.get("original_filename", "Unknown Upload")
                err_detail = f"Upload error: {err_info['error']}"
                batch_result["results"].append(
                    {
                        "status": "Error",
                        "input_ref": original_filename,
                        "error": err_detail,
                        "media_type": "document",
                        "processing_source": None,
                        "metadata": {},
                        "content": None,
                        "chunks": None,
                        "analysis": None,
                        "keywords": form_data.keywords,
                        "warnings": None,
                        "analysis_details": {},
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                        "segments": None,
                    }
                )
                batch_result["errors"].append(f"{original_filename}: {err_detail}")

            for info in saved_files:
                original_ref = info["original_filename"]
                local_path = Path(info["path"])
                local_paths_to_process.append((original_ref, local_path))
                source_map[original_ref] = local_path
                logger.debug(
                    "Prepared uploaded file for processing: {} -> {}",
                    original_ref,
                    local_path,
                )

        # --- Handle URLs (asynchronously) ---
        if form_data.urls:
            logger.info(
                "Attempting to download {} document URLs asynchronously...",
                len(form_data.urls),
            )
            download_tasks: list[asyncio.Task[Path]] = []
            url_task_map: dict[asyncio.Task[Path], str] = {}

            allowed_ext_set = set(ALLOWED_DOC_EXTENSIONS)

            download_url_async = core_download_url_async

            download_tasks = [
                download_url_async(
                    client=None,
                    url=url,
                    target_dir=temp_dir,
                    allowed_extensions=allowed_ext_set,
                    check_extension=True,
                    # Disallow only clearly unsupported/generic types. Allow HTML/XHTML/XML
                    # types here because this endpoint handles .html/.htm/.xml content.
                    disallow_content_types={
                        "application/msword",
                        "application/octet-stream",
                    },
                )
                for url in form_data.urls
            ]

            url_task_map = dict(zip(download_tasks, form_data.urls))

            if download_tasks:
                download_results = await asyncio.gather(
                    *download_tasks,
                    return_exceptions=True,
                )
            else:
                download_results: list[Any] = []

            if download_tasks:
                for task, result in zip(download_tasks, download_results):
                    original_url = url_task_map.get(task, "Unknown URL")

                    if isinstance(result, Path):
                        downloaded_path = result
                        local_paths_to_process.append((original_url, downloaded_path))
                        source_map[original_url] = downloaded_path
                        logger.debug(
                            "Prepared downloaded URL for processing: {} -> {}",
                            original_url,
                            downloaded_path,
                        )
                    elif isinstance(result, Exception):
                        error = result
                        logger.error(
                            "Download or preparation failed for URL {}: {}",
                            original_url,
                            error,
                            exc_info=False,
                        )
                        err_detail = "Download/preparation failed"
                        batch_result["results"].append(
                            {
                                "status": "Error",
                                "input_ref": original_url,
                                "error": err_detail,
                                "media_type": "document",
                                "processing_source": None,
                                "metadata": {},
                                "content": None,
                                "chunks": None,
                                "analysis": None,
                                "keywords": form_data.keywords,
                                "warnings": None,
                                "analysis_details": {},
                                "db_id": None,
                                "db_message": "Processing only endpoint.",
                                "segments": None,
                            }
                        )
                        batch_result["errors"].append(f"{original_url}: {err_detail}")
                    else:
                        logger.error(
                            "Unexpected result type '{}' for URL download task: {}",
                            type(result),
                            original_url,
                        )
                        err_detail = f"Unexpected download result type: {type(result).__name__}"
                        batch_result["results"].append(
                            {
                                "status": "Error",
                                "input_ref": original_url,
                                "error": err_detail,
                                "media_type": "document",
                                "processing_source": None,
                                "metadata": {},
                                "content": None,
                                "chunks": None,
                                "analysis": None,
                                "keywords": form_data.keywords,
                                "warnings": None,
                                "analysis_details": {},
                                "db_id": None,
                                "db_message": "Processing only endpoint.",
                                "segments": None,
                            }
                        )
                        batch_result["errors"].append(f"{original_url}: {err_detail}")

        # --- Check if any files are ready for processing ---
        if not local_paths_to_process:
            logger.warning("No valid document sources found or prepared after handling uploads/URLs.")
            # When uploads/URLs were rejected, surface counts like other process-* endpoints.
            if batch_result["results"]:
                batch_result["errors_count"] = sum(
                    1 for r in batch_result["results"] if str(r.get("status", "")).lower() == "error"
                )
                batch_result["processed_count"] = 0
                status_code = status.HTTP_207_MULTI_STATUS
            else:
                batch_result["errors_count"] = 0
                batch_result["processed_count"] = 0
                status_code = status.HTTP_400_BAD_REQUEST

            response = JSONResponse(status_code=status_code, content=batch_result)
            if legacy_signal is not None:
                apply_media_legacy_headers(response, legacy_signal)
            propagate_billing_headers(injected_response, response)
            return response

        logger.info("Starting processing for {} document(s).", len(local_paths_to_process))

        # --- Prepare options for the worker ---
        chunking_plan: dict[str, Any] | None = None
        if form_data.perform_chunking:
            first_url = (form_data.urls or [None])[0]
            first_filename = None
            if saved_files_info:
                first_filename = saved_files_info[0].get("original_filename")

            chunk_options_dict, chunking_plan = resolve_chunking_options_and_plan(
                form_data,
                media_type="document",
                source_name=first_filename or first_url,
            )
            TemplateClassifier = getattr(media_mod, "TemplateClassifier", None)

            if chunk_options_dict is not None and chunking_plan is None:
                chunk_options_dict = apply_chunking_template_if_any(
                    form_data=form_data,
                    db=db,
                    chunking_options_dict=chunk_options_dict,
                    TemplateClassifier=TemplateClassifier,
                    first_url=first_url,
                    first_filename=first_filename,
                )
        else:
            chunk_options_dict = None

        # --- Build ProcessItem list and run batch processor ---
        items: list[ProcessItem] = [
            ProcessItem(
                input_ref=original_ref,
                local_path=doc_path,
                media_type="document",
                metadata={},
            )
            for original_ref, doc_path in local_paths_to_process
        ]

        async def _document_batch_processor(
            process_items: list[ProcessItem],
        ) -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            loop = asyncio.get_running_loop()

            tasks: list[asyncio.Future] = []
            for item in process_items:
                partial_func = functools.partial(
                    docs.process_document_content,
                    doc_path=item.local_path,
                    perform_chunking=form_data.perform_chunking,
                    chunk_options=chunk_options_dict,
                    perform_analysis=form_data.perform_analysis,
                    summarize_recursively=form_data.summarize_recursively,
                    api_name=form_data.api_name,
                    api_key=None,
                    custom_prompt=form_data.custom_prompt,
                    system_prompt=form_data.system_prompt,
                    title_override=form_data.title,
                    author_override=form_data.author,
                    keywords=form_data.keywords,
                    base_dir=temp_dir,
                )
                tasks.append(loop.run_in_executor(None, partial_func))

            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for item, res in zip(process_items, task_results):
                original_ref = item.input_ref

                if isinstance(res, dict):
                    res["input_ref"] = original_ref
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
                    res.setdefault("segments", None)

                    results.append(res)
                elif isinstance(res, Exception):
                    logger.error(
                        "Task execution failed for {} with exception: {}",
                        original_ref,
                        res,
                        exc_info=res,
                    )
                    error_detail = "Document processing failed"
                    results.append(
                        {
                            "status": "Error",
                            "input_ref": original_ref,
                            "error": error_detail,
                            "media_type": "document",
                            "processing_source": str(item.local_path),
                            "metadata": {},
                            "content": None,
                            "chunks": None,
                            "analysis": None,
                            "keywords": form_data.keywords,
                            "warnings": None,
                            "analysis_details": {},
                            "db_id": None,
                            "db_message": "Processing only endpoint.",
                            "segments": None,
                        }
                    )
                else:
                    logger.error(
                        "Received unexpected result type from document worker task for {}: {}",
                        original_ref,
                        type(res),
                    )
                    error_detail = "Invalid result type from document worker."
                    results.append(
                        {
                            "status": "Error",
                            "input_ref": original_ref,
                            "error": error_detail,
                            "media_type": "document",
                            "processing_source": str(item.local_path),
                            "metadata": {},
                            "content": None,
                            "chunks": None,
                            "analysis": None,
                            "keywords": form_data.keywords,
                            "warnings": None,
                            "analysis_details": {},
                            "db_id": None,
                            "db_message": "Processing only endpoint.",
                            "segments": None,
                        }
                    )

            return results

        base_batch: dict[str, Any] = {
            "results": list(batch_result["results"]),
            "errors": list(batch_result["errors"]),
        }
        batch_result = await run_batch_processor(
            items=items,
            processor=_document_batch_processor,
            base_batch=base_batch,
        )

    # --- Determine final status code ---
    if batch_result.get("errors_count", 0) == 0 and batch_result.get("processed_count", 0) > 0:
        final_status_code = status.HTTP_200_OK
    elif batch_result.get("errors_count", 0) > 0:
        final_status_code = status.HTTP_207_MULTI_STATUS
    elif batch_result.get("processed_count", 0) == 0 and batch_result.get("errors_count", 0) == 0:
        final_status_code = status.HTTP_207_MULTI_STATUS if batch_result["results"] else status.HTTP_400_BAD_REQUEST
    else:
        logger.warning("Reached unexpected state for final status code determination " "in /process-documents.")
        final_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
    logger.log(
        log_level,
        "/process-documents request finished with status {}. " "Processed: {}, Errors: {}",
        final_status_code,
        batch_result.get("processed_count", 0),
        batch_result.get("errors_count", 0),
    )

    # --- Optional template/hierarchical re-chunking of results ---
    try:
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
                    media_type="document",
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
        logger.debug("Re-chunking failed during metadata normalization")

    response = JSONResponse(status_code=final_status_code, content=batch_result)
    if legacy_signal is not None:
        apply_media_legacy_headers(response, legacy_signal)
    propagate_billing_headers(injected_response, response)
    return response


__all__ = ["router"]
