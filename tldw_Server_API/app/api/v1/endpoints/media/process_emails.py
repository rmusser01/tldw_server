from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger
from starlette.responses import JSONResponse

import tldw_Server_API.app.core.Ingestion_Media_Processing.Email.Email_Processing_Lib as email_lib  # type: ignore
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import (
    get_process_emails_form,
)
from tldw_Server_API.app.api.v1.endpoints import media as media_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessEmailsForm
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
from tldw_Server_API.app.core.Ingestion_Media_Processing.pipeline import (
    ProcessItem,
    run_batch_processor,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

router = APIRouter()


@router.post(
    "/process-emails",
    summary="Extract, chunk, analyse Emails (NO DB Persistence)",
    tags=["Media Processing (No DB)"],
    dependencies=[Depends(guard_storage_quota)],
)
async def process_emails_endpoint(
    db: Any = Depends(get_media_db_for_user),
    form_data: ProcessEmailsForm = Depends(get_process_emails_form),
    files: list[UploadFile] | None = File(None),
):
    """
    Modularized wrapper for the legacy /process-emails endpoint.

    Uses TempDirManager, save_uploaded_files, and run_batch_processor for input
    handling and batch orchestration while preserving the legacy response shape
    and status-code semantics.
    """

    if not files:
        # Preserve legacy 400 behavior when no files are provided.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one EML file must be uploaded.",
        )

    logger.info("Request received for /process-emails (no persistence).")

    batch: dict[str, Any] = {
        "results": [],
        "errors": [],
    }
    items: list[ProcessItem] = []
    saved_files_info: list[dict[str, Any]] = []
    chunk_options_dict: dict[str, Any] | None = None
    chunking_plan: dict[str, Any] | None = None

    # Prepare base chunking options before batch processing so the worker
    # closure can use them when invoking the email processing library.
    if form_data.perform_chunking:
        chunk_options_dict, chunking_plan = resolve_chunking_options_and_plan(
            form_data,
            media_type="email",
        )

    # Resolve validator via media exports so tests that monkeypatch
    # media.file_validator_instance continue to work.
    validator: FileValidator = getattr(
        media_mod,
        "file_validator_instance",
        FileValidator(),
    )

    # Determine allowed extensions based on form toggles.
    allowed_exts: list[str] = [".eml"]
    if form_data.accept_archives:
        allowed_exts.append(".zip")
    if getattr(form_data, "accept_mbox", False):
        allowed_exts.append(".mbox")
    if getattr(form_data, "accept_pst", False):
        allowed_exts.extend([".pst", ".ost"])

    with TempDirManager(prefix="email_process_", cleanup=True) as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Save uploaded files via shared helper.
        saved_files_info, file_errors = await save_uploaded_files(
            files or [],
            temp_dir=temp_dir_path,
            validator=validator,
            allowed_extensions=allowed_exts,
        )

        for err in file_errors:
            batch["results"].append(
                {
                    "status": "Error",
                    "input_ref": err.get("input_ref"),
                    "processing_source": None,
                    "media_type": "email",
                    "error": err.get("error", "File save failed"),
                    "metadata": {},
                    "content": None,
                    "chunks": None,
                    "analysis": None,
                    "keywords": None,
                    "warnings": None,
                    "analysis_details": {},
                    "db_id": None,
                    "db_message": "Processing only endpoint.",
                }
            )
            if err.get("error"):
                batch["errors"].append(err.get("error"))

        for pf in saved_files_info:
            path = Path(pf["path"])
            items.append(
                ProcessItem(
                    input_ref=pf.get("original_filename") or path.name,
                    local_path=path,
                    media_type="email",
                    metadata={},
                )
            )

        async def _email_batch_processor(
            process_items: list[ProcessItem],
        ) -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            loop = asyncio.get_running_loop()

            for item in process_items:
                pf = {
                    "path": str(item.local_path),
                    "original_filename": item.input_ref,
                }
                try:
                    path = Path(pf["path"]).resolve()
                    # Read bytes
                    async with media_mod.aiofiles.open(path, "rb") as f:
                        file_bytes = await f.read()

                    if form_data.perform_chunking and chunk_options_dict:
                        chunk_opts = {
                            "method": chunk_options_dict.get("method")
                            or (form_data.chunk_method if form_data.chunk_method else "sentences"),
                            "max_size": chunk_options_dict.get("max_size") or form_data.chunk_size,
                            "overlap": chunk_options_dict.get("overlap") or form_data.chunk_overlap,
                        }
                    else:
                        chunk_opts = {
                            "method": (form_data.chunk_method if form_data.chunk_method else "sentences"),
                            "max_size": form_data.chunk_size,
                            "overlap": form_data.chunk_overlap,
                        }

                    name_lower = (pf.get("original_filename") or path.name).lower()

                    if name_lower.endswith(".zip") and form_data.accept_archives:
                        arch_name = pf.get("original_filename") or path.name
                        processor = functools.partial(
                            email_lib.process_eml_archive_bytes,
                            file_bytes=file_bytes,
                            archive_name=arch_name,
                            title_override=form_data.title,
                            author_override=form_data.author,
                            keywords=form_data.keywords,
                            perform_chunking=form_data.perform_chunking,
                            chunk_options=chunk_opts,
                            perform_analysis=form_data.perform_analysis,
                            api_name=form_data.api_name,
                            api_key=None,
                            custom_prompt=form_data.custom_prompt,
                            system_prompt=form_data.system_prompt,
                            summarize_recursively=form_data.summarize_recursively,
                            ingest_attachments=form_data.ingest_attachments,
                            max_depth=form_data.max_depth,
                        )
                        res_list = await loop.run_in_executor(None, processor)
                        for r_item in res_list:
                            r_item.setdefault("media_type", "email")
                            r_item.setdefault("processing_source", f"archive:{str(path)}")
                            r_item.setdefault(
                                "input_ref",
                                r_item.get("input_ref") or arch_name,
                            )
                            r_item.update(
                                {
                                    "db_id": None,
                                    "db_message": "Processing only endpoint.",
                                }
                            )
                            results.append(r_item)
                    elif name_lower.endswith(".mbox") and getattr(form_data, "accept_mbox", False):
                        mbox_name = pf.get("original_filename") or path.name
                        processor = functools.partial(
                            email_lib.process_mbox_bytes,
                            file_bytes=file_bytes,
                            mbox_name=mbox_name,
                            title_override=form_data.title,
                            author_override=form_data.author,
                            keywords=form_data.keywords,
                            perform_chunking=form_data.perform_chunking,
                            chunk_options=chunk_opts,
                            perform_analysis=form_data.perform_analysis,
                            api_name=form_data.api_name,
                            api_key=None,
                            custom_prompt=form_data.custom_prompt,
                            system_prompt=form_data.system_prompt,
                            summarize_recursively=form_data.summarize_recursively,
                            ingest_attachments=form_data.ingest_attachments,
                            max_depth=form_data.max_depth,
                        )
                        res_list = await loop.run_in_executor(None, processor)
                        for r_item in res_list:
                            r_item.setdefault("media_type", "email")
                            r_item.setdefault("processing_source", f"mbox:{str(path)}")
                            r_item.setdefault(
                                "input_ref",
                                r_item.get("input_ref") or mbox_name,
                            )
                            r_item.update(
                                {
                                    "db_id": None,
                                    "db_message": "Processing only endpoint.",
                                }
                            )
                            results.append(r_item)
                    elif (name_lower.endswith(".pst") or name_lower.endswith(".ost")) and getattr(
                        form_data, "accept_pst", False
                    ):
                        pst_name = pf.get("original_filename") or path.name
                        processor = functools.partial(
                            email_lib.process_pst_bytes,
                            file_bytes=file_bytes,
                            pst_name=pst_name,
                            title_override=form_data.title,
                            author_override=form_data.author,
                            keywords=form_data.keywords,
                            perform_chunking=form_data.perform_chunking,
                            chunk_options=chunk_opts,
                            perform_analysis=form_data.perform_analysis,
                            api_name=form_data.api_name,
                            api_key=None,
                            custom_prompt=form_data.custom_prompt,
                            system_prompt=form_data.system_prompt,
                            summarize_recursively=form_data.summarize_recursively,
                            ingest_attachments=form_data.ingest_attachments,
                            max_depth=form_data.max_depth,
                        )
                        res_list = await loop.run_in_executor(None, processor)
                        for r_item in res_list:
                            r_item.setdefault("media_type", "email")
                            r_item.setdefault("processing_source", f"pst:{str(path)}")
                            r_item.setdefault(
                                "input_ref",
                                r_item.get("input_ref") or pst_name,
                            )
                            r_item.update(
                                {
                                    "db_id": None,
                                    "db_message": "Processing only endpoint.",
                                }
                            )
                            results.append(r_item)
                    else:
                        processor = functools.partial(
                            email_lib.process_email_task,
                            file_bytes=file_bytes,
                            filename=pf.get("original_filename") or path.name,
                            title_override=form_data.title,
                            author_override=form_data.author,
                            keywords=form_data.keywords,
                            perform_chunking=form_data.perform_chunking,
                            chunk_options=chunk_opts,
                            perform_analysis=form_data.perform_analysis,
                            api_name=form_data.api_name,
                            api_key=None,
                            custom_prompt=form_data.custom_prompt,
                            system_prompt=form_data.system_prompt,
                            summarize_recursively=form_data.summarize_recursively,
                            ingest_attachments=form_data.ingest_attachments,
                            max_depth=form_data.max_depth,
                        )
                        res = await loop.run_in_executor(None, processor)
                        res.setdefault("media_type", "email")
                        res.setdefault("processing_source", str(path))
                        res.setdefault(
                            "input_ref",
                            pf.get("original_filename") or path.name,
                        )
                        res.update(
                            {
                                "db_id": None,
                                "db_message": "Processing only endpoint.",
                            }
                        )
                        results.append(res)
                except Exception as exc:  # pragma: no cover - defensive
                    results.append(
                        {
                            "status": "Error",
                            "input_ref": pf.get("original_filename"),
                            "processing_source": str(pf.get("path")),
                            "media_type": "email",
                            "error": "Email processing failed",
                            "metadata": {},
                            "content": None,
                            "chunks": None,
                            "analysis": None,
                            "keywords": None,
                            "warnings": None,
                            "analysis_details": {},
                            "db_id": None,
                            "db_message": "Processing only endpoint.",
                        }
                    )

            return results

        batch = await run_batch_processor(
            items=items,
            processor=_email_batch_processor,
            base_batch=batch,
        )

    processed_count = int(batch.get("processed_count") or 0)
    errors_count = int(batch.get("errors_count") or 0)
    if processed_count > 0 and errors_count == 0:
        final_status = status.HTTP_200_OK
    elif batch.get("results"):
        final_status = status.HTTP_207_MULTI_STATUS
    else:
        final_status = status.HTTP_400_BAD_REQUEST

    # Optional template/hierarchical re-chunking (best-effort).
    try:
        if form_data.perform_chunking:
            # Build chunk options once using shared helper + templates.
            first_filename = None
            try:
                if saved_files_info:
                    first_filename = saved_files_info[0].get("original_filename")
            except Exception:
                logger.debug("Could not determine first filename")
                first_filename = None

            chunk_options_dict, chunking_plan = resolve_chunking_options_and_plan(
                form_data,
                media_type="email",
                source_name=first_filename,
            )
            try:
                TemplateClassifier = getattr(media_mod, "TemplateClassifier", None)
            except Exception:
                logger.debug("TemplateClassifier not available")
                TemplateClassifier = None

            if chunk_options_dict is not None and chunking_plan is None:
                chunk_options_dict = apply_chunking_template_if_any(
                    form_data=form_data,
                    db=db,
                    chunking_options_dict=chunk_options_dict,
                    TemplateClassifier=TemplateClassifier,
                    first_url=None,
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
                    media_type="email",
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
        logger.debug("Optional email re-chunking failed")

    return JSONResponse(status_code=final_status, content=batch)


__all__ = ["router"]
