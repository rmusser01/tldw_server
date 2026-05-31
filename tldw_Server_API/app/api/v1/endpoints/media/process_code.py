from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Response, UploadFile, status
from loguru import logger
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import propagate_billing_headers, require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_code_deps import get_process_code_form
from tldw_Server_API.app.api.v1.endpoints import media as media_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessCodeForm
from tldw_Server_API.app.core.Ingestion_Media_Processing.code_utils import (
    chunk_code_lines,
    detect_code_language,
    read_text_safe,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    TempDirManager,
    save_uploaded_files,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils import (
    download_url_async,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.pipeline import (
    ProcessItem,
    run_batch_processor,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    CODE_FILE_EXTENSIONS,
)
from tldw_Server_API.app.core.testing import is_test_mode

router = APIRouter()


@router.post(
    "/process-code",
    summary="Process code files (NO DB Persistence)",
    tags=["Media Processing (No DB)"],
    dependencies=[
        Depends(guard_storage_quota),
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def process_code_endpoint(
    injected_response: Response,
    db: Any = Depends(get_media_db_for_user),  # Parity with legacy signature; unused
    form_data: ProcessCodeForm = Depends(get_process_code_form),
    files: list[UploadFile] | None = File(
        None,
        description="Code uploads (.py, .c, .cpp, .java, .ts, etc.)",
    ),
) -> JSONResponse:
    """
    Reads uploaded or downloaded code files as text, optionally chunks by
    lines or structure-aware code chunking, and returns artifacts without
    DB writes.

    This mirrors legacy `process-code` behavior while routing through the
    modular `media` package.
    """

    # Reuse shared validation so that error messages and 400 semantics match
    # the legacy implementation (including URL/file presence checks).
    media_mod._validate_inputs("code", form_data.urls or [], files)  # type: ignore[arg-type]

    urls = form_data.urls or []
    # Do not preemptively hard-fail entire batch on URL policy here.
    # URL safety is handled during per-item download to allow partial 207
    # batches in tests.
    batch: dict[str, Any] = {"errors": [], "results": []}

    with TempDirManager(cleanup=True, prefix="process_code_") as temp_dir_path:
        temp_dir = Path(temp_dir_path)
        items: list[ProcessItem] = []

        # Handle uploads
        if files:
            # Preserve test-time monkeypatching of `file_validator_instance`
            # while using the canonical core upload helper.
            validator = media_mod.file_validator_instance

            saved, upload_errors = await save_uploaded_files(
                files,
                temp_dir,
                validator=validator,
                allowed_extensions=sorted(CODE_FILE_EXTENSIONS),
                skip_archive_scanning=False,
                expected_media_type_key="code",
            )

            # TEST_MODE diagnostics for upload validation behavior
            try:
                if is_test_mode() and upload_errors:
                    logger.warning("TEST_MODE: process-code upload errors")
            except Exception:
                logger.debug("Failed to emit TEST_MODE upload diagnostics")

            for err in upload_errors:
                batch["results"].append(
                    {
                        "status": "Error",
                        "input_ref": err.get("original_filename", "Unknown Upload"),
                        # Normalize message for tests: map any disallowed-type to a
                        # standard phrase.
                        "error": (
                            "Invalid file type"
                            if isinstance(err.get("error"), str)
                            and (
                                "not allowed for security"
                                in err.get("error", "").lower()
                                or "invalid file type"
                                in err.get("error", "").lower()
                            )
                            else f"Upload error: {err.get('error')}"
                        ),
                        "media_type": "code",
                        "processing_source": None,
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

            for info in saved:
                filename = info["original_filename"]
                local_path = Path(info["path"])
                items.append(
                    ProcessItem(
                        input_ref=filename,
                        local_path=local_path,
                        media_type="code",
                        metadata={"source": "upload"},
                    )
                )

        # Handle URLs
        if urls:
            tasks = [
                download_url_async(
                    client=None,
                    url=u,
                    target_dir=temp_dir,
                    allowed_extensions=CODE_FILE_EXTENSIONS,
                    check_extension=True,
                )
                for u in urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, res in zip(urls, results):
                if isinstance(res, Exception):
                    batch["results"].append(
                        {
                            "status": "Error",
                            "input_ref": url,
                            "processing_source": None,
                            "media_type": "code",
                            "error": "Download/preparation failed",
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
                    continue

                local_path = Path(res)
                items.append(
                    ProcessItem(
                        input_ref=url,
                        local_path=local_path,
                        media_type="code",
                        metadata={"source": "url"},
                    )
                )

        if not items and not batch["results"]:
            # No valid inputs at all
            resp = JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={**batch, "processed_count": 0, "errors_count": 0},
            )
            propagate_billing_headers(injected_response, resp)
            return resp
        if not items and batch["results"]:
            # Only validation/download errors; treat as partial failure
            resp = JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content={**batch, "processed_count": 0, "errors_count": len(batch["results"])},
            )
            propagate_billing_headers(injected_response, resp)
            return resp

        async def _code_batch_processor(
            process_items: list[ProcessItem],
        ) -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for item in process_items:
                local_path = item.local_path
                input_ref = item.input_ref
                filename = local_path.name
                language = detect_code_language(filename)
                try:
                    text = read_text_safe(local_path)
                except Exception as exc:
                    # TEST_MODE diagnostics for read errors after successful save
                    try:
                        if is_test_mode():
                            logger.warning("TEST_MODE: process-code read error")
                    except Exception:
                        logger.debug("Failed to emit TEST_MODE read-error diagnostics")
                    results.append(
                        {
                            "status": "Error",
                            "input_ref": input_ref,
                            "processing_source": str(local_path),
                            "media_type": "code",
                            "error": "Failed to read code file",
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
                    continue

                chunks = []
                op_status = "Success"
                op_warnings: list[str] | None = None

                if form_data.perform_chunking:
                    if str(form_data.chunk_method or "code").lower() == "lines":
                        chunks = chunk_code_lines(
                            text,
                            form_data.chunk_size,
                            form_data.chunk_overlap,
                            language,
                        )
                    else:
                        from tldw_Server_API.app.core.Chunking.chunker import (  # noqa: WPS433
                            Chunker,
                            ChunkerConfig,
                        )

                        try:
                            chunker = Chunker(
                                config=ChunkerConfig(
                                    default_method="code",
                                    default_max_size=form_data.chunk_size,
                                    default_overlap=form_data.chunk_overlap,
                                )
                            )
                            crs = chunker.chunk_text_with_metadata(
                                text,
                                method="code",
                                max_size=form_data.chunk_size,
                                overlap=form_data.chunk_overlap,
                                language=language,
                            )
                            total = len(crs)
                            chunks = []
                            for idx, cr in enumerate(crs):
                                md = asdict(cr.metadata)
                                opts = md.pop("options", {}) or {}
                                md.update(opts)
                                md.setdefault("chunk_method", "code")
                                md.setdefault("language", language)
                                if md.get("start_line") is None or md.get("end_line") is None:
                                    try:
                                        blocks = md.get("blocks") or []
                                        starts = [
                                            b.get("start_line")
                                            for b in blocks
                                            if isinstance(b, dict) and b.get("start_line") is not None
                                        ]
                                        ends = [
                                            b.get("end_line")
                                            for b in blocks
                                            if isinstance(b, dict) and b.get("end_line") is not None
                                        ]
                                        if starts:
                                            md["start_line"] = int(min(starts))
                                        if ends:
                                            md["end_line"] = int(max(ends))
                                    except Exception:
                                        logger.debug("Failed to derive code chunk line bounds")
                                md["chunk_index"] = idx + 1
                                md["total_chunks"] = total
                                chunks.append({"text": cr.text, "metadata": md})
                        except Exception as code_chunk_err:  # pragma: no cover - rare fallback
                            chunks = chunk_code_lines(
                                text,
                                form_data.chunk_size,
                                form_data.chunk_overlap,
                                language,
                            )
                            op_status = "Warning"
                            op_warnings = [
                                "Structure-aware code chunker failed; "
                                f"fell back to line chunking: {code_chunk_err}"
                            ]

                results.append(
                    {
                        "status": op_status,
                        "input_ref": input_ref,
                        "processing_source": str(local_path),
                        "media_type": "code",
                        "content": text,
                        "metadata": {
                            "language": language,
                            "filename": filename,
                            "lines": text.count("\n") + 1,
                        },
                        "chunks": chunks,
                        "analysis": None,
                        "keywords": None,
                        "warnings": op_warnings,
                        "analysis_details": {},
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                    }
                )

            return results

        # Use shared batch processor to compute counts and ordering while the
        # temporary directory is still available.
        batch = await run_batch_processor(
            items=items,
            processor=_code_batch_processor,
            base_batch={"results": batch["results"], "errors": batch["errors"]},
        )

    final_status = (
        status.HTTP_200_OK
        if (batch["processed_count"] > 0 and batch["errors_count"] == 0)
        else (
            status.HTTP_207_MULTI_STATUS if batch["results"] else status.HTTP_400_BAD_REQUEST
        )
    )
    resp = JSONResponse(status_code=final_status, content=batch)
    propagate_billing_headers(injected_response, resp)
    return resp


__all__ = ["router"]
