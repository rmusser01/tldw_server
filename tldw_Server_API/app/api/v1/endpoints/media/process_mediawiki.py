from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger
from pydantic import ValidationError
from starlette.responses import StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.backpressure import (
    guard_backpressure_and_quota,
)
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.api.v1.API_Deps.media_mediawiki_deps import (
    get_mediawiki_form_data,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import (
    MediaWikiDumpOptionsForm,
    ProcessedMediaWikiPage,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    TempDirManager,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki.Media_Wiki import (
    ALLOWED_MEDIAWIKI_DUMP_EXTENSIONS,
    MAX_MEDIAWIKI_FILE_SIZE_BYTES,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki.Media_Wiki import (
    import_mediawiki_dump as core_import_mediawiki_dump,
)
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename

router = APIRouter()

ALLOWED_DUMP_MIME_TYPES = frozenset(
    {
        "application/xml",
        "text/xml",
        "application/x-xml",
        "text/plain",
        "application/octet-stream",
        "application/gzip",
        "application/x-gzip",
        "application/x-gzip-compressed",
        "application/bzip2",
        "application/x-bzip2",
        "application/x-bzip",
    }
)


def _validate_dump_filename(filename: str) -> None:
    """Validate dump file has an allowed extension."""
    base_name = Path(filename).name
    lower = base_name.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_MEDIAWIKI_DUMP_EXTENSIONS):
        allowed = ", ".join(sorted(ALLOWED_MEDIAWIKI_DUMP_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {allowed}",
        )


def _validate_dump_content_type(content_type: str | None) -> None:
    """Validate dump file content type when provided."""
    if not content_type:
        return
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized and normalized not in ALLOWED_DUMP_MIME_TYPES:
        allowed = ", ".join(sorted(ALLOWED_DUMP_MIME_TYPES))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{content_type}'. Allowed: {allowed}",
        )


def _validate_dump_magic_bytes(first_chunk: bytes, filename: str) -> None:
    """Validate file headers for XML, gzip, or bzip2 dumps."""
    lower = Path(filename).name.lower()
    if lower.endswith((".gz", ".xml.gz")):
        if len(first_chunk) < 2 or not first_chunk.startswith(b"\x1f\x8b"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gzip dump file has invalid header.",
            )
        return
    if lower.endswith((".bz2", ".xml.bz2")):
        if len(first_chunk) < 3 or not first_chunk.startswith(b"BZh"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bzip2 dump file has invalid header.",
            )
        return
    stripped = first_chunk.lstrip(b"\xef\xbb\xbf \t\r\n")
    if not stripped.startswith(b"<"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="XML dump file has invalid header.",
        )


def _parse_namespaces(namespaces_str: str | None) -> list[int] | None:
    if not namespaces_str:
        return None
    return [int(ns.strip()) for ns in namespaces_str.split(",")]


async def _process_mediawiki_dump(
    *,
    form_data: MediaWikiDumpOptionsForm,
    dump_file: UploadFile,
    store_to_db: bool,
    store_to_vector_db: bool,
    filter_item_results: bool,
) -> StreamingResponse:
    """Shared ingestion/processing helper."""
    namespaces = _parse_namespaces(form_data.namespaces_str)
    chunk_options_override = {"max_size": form_data.chunk_max_size}

    def _raise_file_too_large() -> None:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large",
        )

    if core_import_mediawiki_dump is None:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="MediaWiki processing module not loaded.",
        )

    if not dump_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dump file has no filename.",
        )
    _validate_dump_filename(dump_file.filename)
    _validate_dump_content_type(dump_file.content_type)
    dump_file_size = getattr(dump_file, "size", None)
    if isinstance(dump_file_size, int) and dump_file_size > MAX_MEDIAWIKI_FILE_SIZE_BYTES:
        _raise_file_too_large()

    prefix = "mediawiki_ingest_" if store_to_db or store_to_vector_db else "mediawiki_process_"
    with TempDirManager(prefix=prefix, cleanup=False) as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_file_path = temp_dir_path / sanitize_filename(dump_file.filename)
        try:
            async with aiofiles.open(temp_file_path, "wb") as f:
                bytes_written = 0
                first_chunk = await dump_file.read(8192)
                if not first_chunk:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Dump file is empty.",
                    )
                _validate_dump_magic_bytes(first_chunk, dump_file.filename)
                bytes_written += len(first_chunk)
                if bytes_written > MAX_MEDIAWIKI_FILE_SIZE_BYTES:
                    _raise_file_too_large()
                await f.write(first_chunk)
                while chunk := await dump_file.read(8192):
                    bytes_written += len(chunk)
                    if bytes_written > MAX_MEDIAWIKI_FILE_SIZE_BYTES:
                        _raise_file_too_large()
                    await f.write(chunk)
        except HTTPException:
            shutil.rmtree(temp_dir_path, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save uploaded MediaWiki dump")
            shutil.rmtree(temp_dir_path, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save uploaded file",
            ) from exc
        finally:
            with contextlib.suppress(Exception):
                await dump_file.close()

        logger.info("MediaWiki dump saved to temporary path: {}", temp_file_path)

        async def stream_ingestion_results() -> AsyncGenerator[str, None]:
            try:
                for result_event in core_import_mediawiki_dump(
                    file_path=str(temp_file_path),
                    wiki_name=form_data.wiki_name,
                    namespaces=namespaces,
                    skip_redirects=form_data.skip_redirects,
                    chunk_options_override=chunk_options_override,
                    store_to_db=store_to_db,
                    store_to_vector_db=store_to_vector_db,
                    api_name_vector_db=form_data.api_name_vector_db,
                    api_key_vector_db=form_data.api_key_vector_db,
                    allowed_dir=temp_dir_path,
                ):
                    if filter_item_results and result_event.get("type") == "item_result":
                        page_data = result_event.get("data", {})
                        try:
                            processed_page_model = ProcessedMediaWikiPage(**page_data)
                            yield json.dumps(processed_page_model.model_dump()) + "\n"
                        except ValidationError as ve:
                            logger.error(
                                "Validation error for processed MediaWiki page "
                                "'{}': {}",
                                page_data.get("title", "Unknown"),
                                ve.errors(),
                            )
                            error_output = {
                                "type": "validation_error",
                                "title": page_data.get("title", "Unknown"),
                                "page_id": page_data.get("page_id"),
                                "detail": ve.errors(),
                            }
                            yield json.dumps(error_output) + "\n"
                    else:
                        yield json.dumps(result_event) + "\n"

                    await asyncio.sleep(0.01)
            finally:
                try:
                    shutil.rmtree(temp_dir_path, ignore_errors=True)
                    logger.info("Cleaned up temporary directory: {}", temp_dir_path)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to cleanup temporary directory")

        return StreamingResponse(
            stream_ingestion_results(),
            media_type="application/x-ndjson",
        )


@router.post(
    "/mediawiki/ingest-dump",
    summary="Ingest and process a MediaWiki XML dump, storing results to database and vector store.",
    tags=["MediaWiki Processing"],
    dependencies=[Depends(guard_backpressure_and_quota), Depends(guard_storage_quota)],
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "NDJSON stream of MediaWiki ingest events",
            "content": {"application/x-ndjson": {}},
        },
    },
)
async def ingest_mediawiki_dump_endpoint(
    form_data: MediaWikiDumpOptionsForm = Depends(get_mediawiki_form_data),
    dump_file: UploadFile = File(
        ...,
        description="MediaWiki XML dump file (.xml, .xml.bz2, .xml.gz).",
    ),
) -> StreamingResponse:
    """
    MediaWiki ingest endpoint (streaming).

    Streams ingestion events while processing a MediaWiki XML dump and
    persisting results to the primary database and vector store.
    """
    return await _process_mediawiki_dump(
        form_data=form_data,
        dump_file=dump_file,
        store_to_db=True,
        store_to_vector_db=True,
        filter_item_results=False,
    )


@router.post(
    "/mediawiki/process-dump",
    summary="Process a MediaWiki XML dump and return structured content without database storage.",
    tags=["MediaWiki Processing"],
    dependencies=[Depends(guard_backpressure_and_quota), Depends(guard_storage_quota)],
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "NDJSON stream of processed MediaWiki pages",
            "content": {"application/x-ndjson": {}},
        },
    },
)
async def process_mediawiki_dump_ephemeral_endpoint(
    form_data: MediaWikiDumpOptionsForm = Depends(get_mediawiki_form_data),
    dump_file: UploadFile = File(
        ...,
        description="MediaWiki XML dump file (.xml, .xml.bz2, .xml.gz).",
    ),
) -> StreamingResponse:
    """
    MediaWiki processing endpoint (ephemeral, streaming).

    Streams processed items from a MediaWiki XML dump without saving to the
    main database or vector store. Each line in the response is a JSON object.
    """
    return await _process_mediawiki_dump(
        form_data=form_data,
        dump_file=dump_file,
        store_to_db=False,
        store_to_vector_db=False,
        filter_item_results=True,
    )


__all__ = ["router"]
