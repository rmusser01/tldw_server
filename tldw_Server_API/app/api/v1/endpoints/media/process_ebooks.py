from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Response,
    UploadFile,
    status,
)
from loguru import logger
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import propagate_billing_headers, require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import (
    get_process_ebooks_form,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.endpoints import media as media_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessEbooksForm
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
from tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils import (
    download_url_async as core_download_url_async,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.pipeline import (
    ProcessItem,
    run_batch_processor,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator
from tldw_Server_API.app.api.v1.endpoints.media.input_contracts import (
    normalize_urls_field,
    validate_media_inputs,
)
from tldw_Server_API.app.api.v1.endpoints.media.deprecation_signals import (
    apply_media_legacy_headers,
    build_media_legacy_signal,
)

router = APIRouter()


ALLOWED_EBOOK_EXTENSIONS = [".epub"]


def _process_single_ebook(
    ebook_path: Path,
    original_ref: str,
    title_override: str | None,
    author_override: str | None,
    keywords: list[str] | None,
    perform_chunking: bool,
    chunk_options: dict[str, Any] | None,
    perform_analysis: bool,
    summarize_recursively: bool,
    api_name: str | None,
    custom_prompt: str | None,
    system_prompt: str | None,
    extraction_method: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    """Synchronous worker for EPUB processing (mirrors legacy helper)."""
    try:
        result_dict = media_mod.books.process_epub(
            file_path=str(ebook_path),
            title_override=title_override,
            author_override=author_override,
            keywords=keywords,
            perform_chunking=perform_chunking,
            chunk_options=chunk_options,
            perform_analysis=perform_analysis,
            api_name=api_name,
            api_key=None,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            summarize_recursively=summarize_recursively,
            extraction_method=extraction_method,
            base_dir=base_dir,
        )
        result_dict["input_ref"] = original_ref
        # Ensure overrides and derived fields are present even if the library
        # omitted them (legacy parity expectations in tests).
        result_dict.setdefault("metadata", {})
        if title_override:
            result_dict["metadata"]["title"] = title_override
        if author_override:
            result_dict["metadata"]["author"] = author_override
        result_dict["keywords"] = keywords or result_dict.get("keywords") or []
        result_dict.setdefault("analysis", None)
        result_dict.setdefault("analysis_details", {})
        return result_dict
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.error(
            "_process_single_ebook error for {} ({}): {}",
            original_ref,
            ebook_path,
            exc,
            exc_info=True,
        )
        return {
            "status": "Error",
            "input_ref": original_ref,
            "processing_source": str(ebook_path),
            "media_type": "ebook",
            "error": "Ebook processing failed",
            "content": None,
            "metadata": None,
            "chunks": None,
            "analysis": None,
            "keywords": keywords or [],
            "warnings": None,
            "analysis_details": {},
        }


@router.post(
    "/process-ebooks",
    summary="Extract, chunk, analyse EPUBs (NO DB Persistence)",
    tags=["Media Processing (No DB)"],
    dependencies=[
        Depends(guard_storage_quota),
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def process_ebooks_endpoint(
    injected_response: Response,
    db: Any = Depends(get_media_db_for_user),
    form_data: ProcessEbooksForm = Depends(get_process_ebooks_form),
    files: list[UploadFile] | None = File(None, description="EPUB file uploads (.epub)"),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    """
    Process EPUBs without persisting to the Media DB.

    This endpoint mirrors the legacy `/process-ebooks` behavior while routing
    through the modular `media` package and using the shared input_sourcing
    and pipeline helpers for input handling and batch orchestration.
    """

    logger.info("Request received for /process-ebooks (no persistence).")
    try:
        usage_log.log_event(
            "media.process.ebook",
            tags=["no_db"],
            metadata={"has_urls": bool(form_data.urls), "has_files": bool(files)},
        )
    except Exception:
        # Usage logging is best-effort; do not fail the request.
        logger.debug("Ebook process endpoint usage logging failed")

    # Legacy endpoint treats urls=[""] as "no URLs".
    legacy_urls_empty_sentinel_used = bool(form_data.urls and form_data.urls == [""])
    if legacy_urls_empty_sentinel_used:
        logger.info("Received urls=[''], treating as no URLs provided for ebook processing.")
    form_data.urls = normalize_urls_field(form_data.urls)
    legacy_signal = (
        build_media_legacy_signal(
            successor="/api/v1/media/process-ebooks",
            warning_code="legacy_urls_empty_sentinel",
        )
        if legacy_urls_empty_sentinel_used
        else None
    )

    # Reuse shared validation so that error messages and 400 semantics match
    # the legacy implementation (including the "At least one 'url'..." detail).
    validate_media_inputs(
        media_mod._validate_inputs,
        "ebook",
        form_data.urls,
        files,
    )

    batch: dict[str, Any] = {"results": [], "errors": []}
    items: list[ProcessItem] = []
    saved_files_info: list[dict[str, Any]] = []
    chunk_options_dict: dict[str, Any] | None = None
    chunking_plan: dict[str, Any] | None = None

    with TempDirManager(prefix="process_ebooks_") as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Preserve test-time monkeypatching of `media.file_validator_instance`
        # by resolving the validator from the media module export.
        validator: FileValidator = getattr(
            media_mod,
            "file_validator_instance",
            FileValidator(),
        )

        # ---- Handle uploads via shared input_sourcing helper ----
        if files:
            saved_files, upload_errors = await save_uploaded_files(
                files,
                temp_dir=temp_dir_path,
                validator=validator,
                allowed_extensions=ALLOWED_EBOOK_EXTENSIONS,
            )
            saved_files_info = list(saved_files)

            for err_info in upload_errors:
                original_filename = err_info.get("original_filename") or err_info.get("input") or "Unknown Upload"
                error_detail = str(err_info.get("error") or "Upload error")
                batch["results"].append(
                    {
                        "status": "Error",
                        "input_ref": original_filename,
                        "processing_source": None,
                        "media_type": "ebook",
                        "error": f"Upload error: {error_detail}",
                        "metadata": {},
                        "content": None,
                        "chunks": None,
                        "analysis": None,
                        "keywords": form_data.keywords,
                        "warnings": None,
                        "analysis_details": {},
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                    }
                )
                batch["errors"].append(f"{original_filename}: {error_detail}")

            for info in saved_files:
                local_path = Path(info["path"])
                input_ref = info["original_filename"]
                items.append(
                    ProcessItem(
                        input_ref=input_ref,
                        local_path=local_path,
                        media_type="ebook",
                        metadata={"source": "upload"},
                    )
                )

        # ---- Handle URL inputs via shared downloader ----
        if form_data.urls:
            logger.info(
                "Attempting to download {} EPUB URL(s) asynchronously...",
                len(form_data.urls),
            )
            download_url_async = core_download_url_async
            download_tasks = [
                download_url_async(
                    client=None,
                    url=url,
                    target_dir=temp_dir_path,
                    allowed_extensions={".epub"},
                )
                for url in form_data.urls
            ]
            download_results = await asyncio.gather(
                *download_tasks,
                return_exceptions=True,
            )

            for url, result in zip(form_data.urls, download_results):
                if isinstance(result, Path):
                    items.append(
                        ProcessItem(
                            input_ref=url,
                            local_path=result,
                            media_type="ebook",
                            metadata={"source": "url"},
                        )
                    )
                else:
                    error_detail = "Download/preparation failed"
                    batch["results"].append(
                        {
                            "status": "Error",
                            "input_ref": url,
                            "processing_source": None,
                            "media_type": "ebook",
                            "error": error_detail,
                            "metadata": {},
                            "content": None,
                            "chunks": None,
                            "analysis": None,
                            "keywords": form_data.keywords,
                            "warnings": None,
                            "analysis_details": {},
                            "db_id": None,
                            "db_message": "Processing only endpoint.",
                        }
                    )
                    batch["errors"].append(f"{url}: {error_detail}")

        # If we ended up with no valid inputs, mirror the legacy behavior:
        # - 207 when we have result entries (all errors)
        # - 400 when no inputs at all (handled by _validate_inputs above)
        if not items:

            async def _noop_processor(_: list[ProcessItem]) -> list[dict[str, Any]]:
                return []

            batch = await run_batch_processor(
                items=[],
                processor=_noop_processor,
                base_batch=batch,
            )
            status_code = status.HTTP_207_MULTI_STATUS if batch.get("results") else status.HTTP_400_BAD_REQUEST
            response = JSONResponse(status_code=status_code, content=batch)
            if legacy_signal is not None:
                apply_media_legacy_headers(response, legacy_signal)
            propagate_billing_headers(injected_response, response)
            return response

        # Prepare chunking options (with optional templates/hierarchical rules).
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
                media_type="ebook",
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
        else:
            chunk_options_dict = None

        async def _ebook_batch_processor(
            process_items: list[ProcessItem],
        ) -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            loop = asyncio.get_running_loop()

            for item in process_items:
                original_ref = item.input_ref
                ebook_path = item.local_path

                try:
                    partial_func = functools.partial(
                        _process_single_ebook,
                        ebook_path=ebook_path,
                        original_ref=original_ref,
                        title_override=form_data.title,
                        author_override=form_data.author,
                        keywords=form_data.keywords,
                        perform_chunking=form_data.perform_chunking,
                        chunk_options=chunk_options_dict,
                        perform_analysis=form_data.perform_analysis,
                        summarize_recursively=form_data.summarize_recursively,
                        api_name=form_data.api_name,
                        custom_prompt=form_data.custom_prompt,
                        system_prompt=form_data.system_prompt,
                        extraction_method=form_data.extraction_method,
                        base_dir=temp_dir,
                    )
                    res = await loop.run_in_executor(None, partial_func)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.error(
                        "Task execution failed for {}: {}",
                        original_ref,
                        exc,
                        exc_info=True,
                    )
                    error_detail = "Ebook processing failed"
                    res = {
                        "status": "Error",
                        "input_ref": original_ref,
                        "processing_source": str(ebook_path),
                        "media_type": "ebook",
                        "error": error_detail,
                        "content": None,
                        "metadata": {},
                        "chunks": None,
                        "analysis": None,
                        "keywords": form_data.keywords or [],
                        "warnings": None,
                        "analysis_details": {},
                    }

                # Ensure mandatory fields and DB fields match the legacy endpoint.
                res["db_id"] = None
                res["db_message"] = "Processing only endpoint."
                res.setdefault("status", "Error")
                res.setdefault("input_ref", original_ref or "Unknown")
                res.setdefault("media_type", "ebook")
                res.setdefault("error", None)
                res.setdefault("warnings", None)
                res.setdefault("metadata", {})
                res.setdefault("content", None)
                res.setdefault("chunks", None)
                res.setdefault("analysis", None)
                res.setdefault("keywords", [])
                res.setdefault("analysis_details", {})

                status_value = str(res.get("status", "")).lower()
                if status_value == "warning":
                    warnings_list = res.get("warnings") or []
                    for warn in warnings_list:
                        msg = f"{res.get('input_ref', 'Unknown')}: [Warning] {warn}"
                        batch["errors"].append(msg)
                elif status_value == "error":
                    error_msg = f"{res.get('input_ref', 'Unknown')}: " f"{res.get('error', 'Unknown processing error')}"
                    if error_msg not in batch["errors"]:
                        batch["errors"].append(error_msg)

                results.append(res)

            return results

        batch = await run_batch_processor(
            items=items,
            processor=_ebook_batch_processor,
            base_batch=batch,
        )

    # Final HTTP status logic matches the legacy endpoint.
    processed_count = int(batch.get("processed_count") or 0)
    errors_count = int(batch.get("errors_count") or 0)
    if errors_count == 0 and processed_count > 0:
        final_status_code = status.HTTP_200_OK
    elif batch.get("results"):
        final_status_code = status.HTTP_207_MULTI_STATUS
    else:
        final_status_code = status.HTTP_400_BAD_REQUEST

    log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
    logger.log(
        log_level,
        "/process-ebooks request finished with status {}. Results: {}, " "Processed: {}, Errors: {}",
        final_status_code,
        len(batch.get("results", [])),
        processed_count,
        errors_count,
    )

    # Optional template/hierarchical re-chunking of results (best-effort).
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

            for res in batch.get("results", []):
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
                    media_type="ebook",
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
                        max_size=result_chunk_options.get("max_size") or 1000,
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
        logger.debug("Ebook post-processing re-chunking skipped/failed")

    response = JSONResponse(status_code=final_status_code, content=batch)
    if legacy_signal is not None:
        apply_media_legacy_headers(response, legacy_signal)
    propagate_billing_headers(injected_response, response)
    return response


__all__ = ["router"]
