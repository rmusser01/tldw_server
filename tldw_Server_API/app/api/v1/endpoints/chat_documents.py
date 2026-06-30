from __future__ import annotations

import asyncio
import contextlib
import datetime
import os
import sqlite3
import time
from collections.abc import AsyncIterator
from typing import Any, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from loguru import logger
from starlette.responses import StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.chat_documents_deps import get_document_generator_service
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.document_generator_schemas import (
    AsyncGenerationResponse,
    BulkGenerateRequest,
    BulkGenerateResponse,
    DocumentListResponse,
    GeneratedDocument,
    GenerateDocumentRequest,
    GenerateDocumentResponse,
    GenerationStatistics,
    JobStatusResponse,
    PromptConfigResponse,
    SavePromptConfigRequest,
)
from tldw_Server_API.app.api.v1.schemas.document_generator_schemas import (
    DocumentType as DocType,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    record_byok_missing_credentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key
from tldw_Server_API.app.core.Chat.document_generator import (
    DocumentGeneratorService,
    DocumentType,
)
from tldw_Server_API.app.core.Chat.document_generator import (
    GenerationStatus as GenStatus,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode

router = APIRouter()
MAX_GENERATED_DOCUMENTS_OFFSET = 10_000

_CHAT_DOCS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


@router.post(
    "/documents/generate",
    response_model=Union[GenerateDocumentResponse, AsyncGenerationResponse],
    responses={
        200: {
            "description": "Generated document result, async job metadata, or SSE stream.",
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            {"$ref": "#/components/schemas/GenerateDocumentResponse"},
                            {"$ref": "#/components/schemas/AsyncGenerationResponse"},
                        ]
                    }
                },
                "text/event-stream": {},
            },
        },
    },
    summary="Generate a document from conversation",
    description="Generate a document using conversation content and a template. May return async job metadata.",
    tags=["chat-documents"],
)
async def generate_document(
    request: GenerateDocumentRequest,
    http_request: Request,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> GenerateDocumentResponse | AsyncGenerationResponse:
    """Generate a document from a conversation."""
    try:
        service = service_cls(db)

        doc_type = DocumentType(request.document_type.value)

        provider_name = (request.provider or "").strip()
        if not provider_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provider is required")
        provider_key = provider_name.lower()

        # Resolve provider key requirements
        try:
            from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
        except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS:
            def provider_requires_api_key(_provider: str) -> bool:  # type: ignore[misc]
                return True

        try:
            _is_pytest = bool(os.getenv("PYTEST_CURRENT_TEST"))
        except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS:
            _is_pytest = False
        _is_test_mode = is_test_mode()

        if request.api_key:
            logger.debug("Ignoring per-request api_key override for provider={}", provider_name)
        explicit_key = None
        provider_api_key = None
        byok_resolution = None
        app_config_override = None

        # When no explicit key is provided, resolve BYOK first and fall back to server defaults.
        if not provider_api_key:
            def _fallback_resolver(name: str) -> str | None:
                key_val, _ = resolve_provider_api_key(
                    name,
                    prefer_module_keys_in_tests=True,
                )
                return key_val

            user_id_int = None
            try:
                if principal.user_id is not None:
                    user_id_int = int(principal.user_id)
            except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS:
                user_id_int = None

            byok_resolution = await resolve_byok_credentials(
                provider_key,
                user_id=user_id_int,
                request=http_request,
                fallback_resolver=_fallback_resolver,
            )
            provider_api_key = byok_resolution.api_key
            app_config_override = byok_resolution.app_config

        if provider_requires_api_key(provider_key) and not provider_api_key:
            if (_is_pytest or _is_test_mode) and bool(request.stream):
                logger.debug(
                    'Bypassing provider API key requirement for streaming document generation during tests (provider={})',
                    provider_name,
                )
                provider_api_key = None
            else:
                record_byok_missing_credentials(provider_key, operation="chat_documents")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error_code": "missing_provider_credentials",
                        "message": f"Provider '{provider_name}' requires an API key.",
                    },
                )

        if request.async_generation:
            job_id = service.create_generation_job(
                conversation_id=request.conversation_id,
                document_type=doc_type,
                provider=provider_name,
                model=request.model,
                prompt_config={
                    "specific_message": request.specific_message,
                    "custom_prompt": request.custom_prompt,
                },
            )

            return AsyncGenerationResponse(
                job_id=job_id,
                status=GenStatus.PENDING,
                conversation_id=request.conversation_id,
                document_type=request.document_type,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                message="Document generation job created",
            )

        def _generate_doc(stream: bool) -> str | Any:
            return service.generate_document(
                conversation_id=request.conversation_id,
                document_type=doc_type,
                provider=provider_name,
                model=request.model,
                api_key=provider_api_key or "",
                app_config=app_config_override,
                specific_message=request.specific_message,
                custom_prompt=request.custom_prompt,
                stream=stream,
            )

        content = await asyncio.to_thread(_generate_doc, request.stream)

        if isinstance(content, dict):
            if content.get("success") is False:
                detail = content.get("error") or "Document generation failed"
                logger.warning(
                    'Document generation failed for conversation {}: {}',
                    request.conversation_id,
                    detail,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail,
                )
            logger.error(
                'Unexpected document generation payload for conversation {}: {}',
                request.conversation_id,
                type(content).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected document generation response format",
            )

        if request.stream:
            streaming_source = content

            def _normalize_chunk(chunk: Any) -> str:
                if chunk is None:
                    return ""
                if isinstance(chunk, (bytes, bytearray)):
                    try:
                        return chunk.decode("utf-8")
                    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS:
                        return chunk.decode("utf-8", errors="ignore")
                return str(chunk)

            def _encode_sse(text: str) -> str:
                lines = text.splitlines() or [""]
                return "".join(f"data: {line}\n" for line in lines) + "\n"

            async def _iter_stream() -> AsyncIterator[Any]:
                nonlocal streaming_source
                if hasattr(streaming_source, "__aiter__"):
                    async for chunk in streaming_source:  # type: ignore[attr-defined]
                        yield chunk
                    return
                if hasattr(streaming_source, "__iter__") and not isinstance(streaming_source, (str, bytes, bytearray)):
                    iterator = iter(streaming_source)  # type: ignore[arg-type]
                    while True:
                        try:
                            chunk = await asyncio.to_thread(next, iterator)
                        except StopIteration:
                            break
                        yield chunk
                    return
                yield streaming_source

            stream_started_at = time.perf_counter()
            collected_chunks: list[str] = []

            if env_flag_enabled("STREAMS_UNIFIED"):
                from tldw_Server_API.app.core.Streaming.streams import SSEStream

                stream = SSEStream(labels={"component": "chat", "endpoint": "chat_doc_stream"})

                async def _produce() -> None:
                    try:
                        async for chunk in _iter_stream():
                            payload = _normalize_chunk(chunk)
                            if not payload:
                                continue
                            collected_chunks.append(payload)
                            for line in payload.splitlines() or [""]:
                                if line.strip().lower() == "[done]":
                                    continue
                                await stream.send_raw_sse_line(f"data: {line}")
                        await stream.done()
                    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as exc:
                        await stream.error("internal_error", f"{exc}")

                async def _gen() -> AsyncIterator[str]:
                    prod = asyncio.create_task(_produce())
                    try:
                        async for ln in stream.iter_sse():
                            yield ln
                    except asyncio.CancelledError:
                        if not prod.done():
                            with contextlib.suppress(_CHAT_DOCS_NONCRITICAL_EXCEPTIONS):
                                prod.cancel()
                            with contextlib.suppress(_CHAT_DOCS_NONCRITICAL_EXCEPTIONS):
                                await prod
                        raise
                    else:
                        if not prod.done():
                            with contextlib.suppress(_CHAT_DOCS_NONCRITICAL_EXCEPTIONS):
                                await prod
                        try:
                            document_body = "".join(collected_chunks).strip()
                            if document_body:
                                generation_time_ms = int((time.perf_counter() - stream_started_at) * 1000)
                                await asyncio.to_thread(
                                    service.record_streamed_document,
                                    conversation_id=request.conversation_id,
                                    document_type=doc_type,
                                    content=document_body,
                                    provider=provider_name,
                                    model=request.model,
                                    generation_time_ms=generation_time_ms,
                                )
                            else:
                                logger.info(
                                    'Streamed document produced no content for conversation {}; skipping persistence',
                                    request.conversation_id,
                                )
                            if byok_resolution and byok_resolution.uses_byok and not explicit_key:
                                await byok_resolution.touch_last_used()
                        except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as persist_exc:
                            logger.error(
                                'Failed to persist streamed document for conversation {}: {}',
                                request.conversation_id,
                                persist_exc,
                            )

                headers = {
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
                return StreamingResponse(_gen(), media_type="text/event-stream", headers=headers)

            async def _sse_stream() -> AsyncIterator[str]:
                try:
                    async for chunk in _iter_stream():
                        payload = _normalize_chunk(chunk)
                        if payload:
                            collected_chunks.append(payload)
                            yield _encode_sse(payload)
                except asyncio.CancelledError:
                    logger.info(
                        'Document generation stream cancelled for conversation {}',
                        request.conversation_id,
                    )
                    raise
                finally:
                    try:
                        document_body = "".join(collected_chunks).strip()
                        if document_body:
                            generation_time_ms = int((time.perf_counter() - stream_started_at) * 1000)
                            await asyncio.to_thread(
                                service.record_streamed_document,
                                conversation_id=request.conversation_id,
                                document_type=doc_type,
                                content=document_body,
                                provider=provider_name,
                                model=request.model,
                                generation_time_ms=generation_time_ms,
                            )
                        else:
                            logger.info(
                                'Streamed document produced no content for conversation {}; skipping persistence',
                                request.conversation_id,
                            )
                        if byok_resolution and byok_resolution.uses_byok and not explicit_key:
                            await byok_resolution.touch_last_used()
                    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as persist_exc:
                        logger.error(
                            'Failed to persist streamed document for conversation {}: {}',
                            request.conversation_id,
                            persist_exc,
                        )

            headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
            return StreamingResponse(_sse_stream(), media_type="text/event-stream", headers=headers)

        docs = service.get_generated_documents(
            conversation_id=request.conversation_id,
            document_type=doc_type,
            limit=1,
        )
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document generated but could not be retrieved",
            )

        doc = docs[0]
        if byok_resolution and byok_resolution.uses_byok and not explicit_key:
            await byok_resolution.touch_last_used()
        return GenerateDocumentResponse(
            document_id=doc["id"],
            conversation_id=doc["conversation_id"],
            document_type=request.document_type,
            title=doc["title"],
            content=doc["content"],
            provider=doc["provider"],
            model=doc["model"],
            generation_time_ms=doc["generation_time_ms"],
            created_at=doc["created_at"],
        )

    except InputError as e:
        logger.warning(f"Input error generating document: {e}")
        raise map_db_error_to_http(e) from e
    except ChatAPIError as e:
        logger.error(f"API error generating document: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The chat service provider is currently unavailable.",
        ) from e
    except HTTPException:
        raise
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error generating document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred.",
        ) from e


@router.get(
    "/documents/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get generation job status",
    description="Check the current status and progress of a document generation job.",
    tags=["chat-documents"],
)
async def get_job_status(
    job_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> JobStatusResponse:
    """Get the status of a document generation job."""
    try:
        service = service_cls(db)
        job = service.get_job_status(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        progress = 0
        if job["status"] == GenStatus.PENDING.value:
            progress = 0
        elif job["status"] == GenStatus.IN_PROGRESS.value:
            progress = 50
        elif job["status"] in [
            GenStatus.COMPLETED.value,
            GenStatus.FAILED.value,
            GenStatus.CANCELLED.value,
        ]:
            progress = 100

        return JobStatusResponse(
            job_id=job["job_id"],
            conversation_id=job["conversation_id"],
            document_type=DocType(job["document_type"]),
            status=GenStatus(job["status"]),
            provider=job["provider"],
            model=job["model"],
            result_content=job["result_content"],
            error_message=job["error_message"],
            created_at=job["created_at"],
            started_at=job["started_at"],
            completed_at=job["completed_at"],
            progress_percentage=progress,
        )
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to get generation job status") from e
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generation job status",
        ) from e


@router.delete(
    "/documents/jobs/{job_id}",
    summary="Cancel generation job",
    description="Cancel a pending or running document generation job.",
    tags=["chat-documents"],
)
async def cancel_job(
    job_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> dict[str, str]:
    """Cancel a document generation job."""
    try:
        service = service_cls(db)

        job = service.get_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        if job["status"] in [
            GenStatus.COMPLETED.value,
            GenStatus.FAILED.value,
            GenStatus.CANCELLED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is already {job['status']}",
            )

        success = service.update_job_status(job_id, GenStatus.CANCELLED)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel job",
            )

        return {"message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to cancel generation job") from e
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel generation job",
        ) from e


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List generated documents",
    description="List previously generated documents for the current user.",
    tags=["chat-documents"],
)
async def list_generated_documents(
    conversation_id: str | None = Query(None, min_length=1, description="Filter by conversation ID"),
    document_type: DocType | None = Query(None, description="Filter by document type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of documents"),
    offset: int = Query(
        0,
        ge=0,
        le=MAX_GENERATED_DOCUMENTS_OFFSET,
        description="Zero-based pagination offset",
    ),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> DocumentListResponse:
    """List previously generated documents."""
    try:
        service = service_cls(db)

        doc_type = DocumentType(document_type.value) if document_type else None

        paged_documents = service.get_generated_documents(
            conversation_id=conversation_id,
            document_type=doc_type,
            limit=limit,
            offset=offset,
        )

        count_documents = getattr(service, "count_generated_documents", None)
        if callable(count_documents):
            total = count_documents(conversation_id=conversation_id, document_type=doc_type)
        else:
            total = offset + len(paged_documents)

        doc_responses = [GeneratedDocument(**doc) for doc in paged_documents]
        pagination = build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(doc_responses),
        )

        return DocumentListResponse(
            documents=doc_responses,
            total=total,
            pagination=pagination,
            conversation_id=conversation_id,
            document_type=document_type,
        )
    except CharactersRAGDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to list generated documents") from e
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error listing generated documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list generated documents",
        ) from e


@router.get(
    "/documents/{document_id}",
    response_model=GeneratedDocument,
    summary="Get generated document",
    description="Retrieve a generated document by its identifier.",
    tags=["chat-documents"],
)
async def get_generated_document(
    document_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> GeneratedDocument:
    """Get a specific generated document."""
    try:
        service = service_cls(db)

        doc = service.get_generated_document_by_id(document_id)

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        return GeneratedDocument(**doc)
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to get generated document") from e
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generated document",
        ) from e


@router.delete(
    "/documents/{document_id}",
    summary="Delete generated document",
    description="Delete a generated document by its identifier.",
    tags=["chat-documents"],
)
async def delete_generated_document(
    document_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> dict[str, str]:
    """Delete a generated document."""
    try:
        service = service_cls(db)

        success = service.delete_generated_document(document_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        return {"message": f"Document {document_id} deleted successfully"}
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to delete generated document") from e
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete generated document",
        ) from e


@router.post(
    "/documents/prompts",
    response_model=PromptConfigResponse,
    summary="Save custom prompt configuration",
    description="Save a custom prompt configuration for a given document type.",
    tags=["chat-documents"],
)
async def save_prompt_config(
    config: SavePromptConfigRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> PromptConfigResponse:
    """Save a custom prompt configuration for a document type."""
    try:
        service = service_cls(db)

        doc_type = DocumentType(config.document_type.value)

        success = service.save_user_prompt_config(
            document_type=doc_type,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save prompt configuration",
            )

        return PromptConfigResponse(
            document_type=config.document_type,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            is_custom=True,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            updated_at=datetime.datetime.now(datetime.timezone.utc),
        )
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to save prompt configuration") from e
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error saving prompt config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save prompt configuration",
        ) from e


@router.get(
    "/documents/prompts/{document_type}",
    response_model=PromptConfigResponse,
    summary="Get prompt configuration",
    description="Retrieve the saved prompt configuration for a document type.",
    tags=["chat-documents"],
)
async def get_prompt_config(
    document_type: DocType,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> PromptConfigResponse:
    """Get the prompt configuration for a document type."""
    try:
        service = service_cls(db)

        doc_type = DocumentType(document_type.value)

        config = service.get_user_prompt_config(doc_type)

        is_custom = False
        try:
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM user_prompts WHERE document_type = ? AND is_active = 1",
                    (doc_type.value,),
                )
                is_custom = cursor.fetchone() is not None
        except sqlite3.OperationalError as e:
            logger.warning(f"Database operational error checking custom prompts: {e}")
            is_custom = False
        except sqlite3.DatabaseError as e:
            logger.error(f"Database error checking custom prompts for doc_type={doc_type.value}: {e}")
            is_custom = False
        except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Unexpected error checking custom prompts: {type(e).__name__}: {e}", exc_info=True)
            is_custom = False

        return PromptConfigResponse(
            document_type=document_type,
            system_prompt=config["system"],
            user_prompt=config["user"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            is_custom=is_custom,
            created_at=None,
            updated_at=None,
        )
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting prompt config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get prompt configuration",
        ) from e


@router.post(
    "/documents/bulk",
    response_model=BulkGenerateResponse,
    summary="Bulk generate documents",
    description="Submit multiple document generations in one request. May return async job IDs.",
    tags=["chat-documents"],
)
async def bulk_generate_documents(
    request: BulkGenerateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> BulkGenerateResponse:
    """Generate multiple documents in bulk (async)."""
    try:
        service = service_cls(db)

        job_ids: list[str] = []
        total_jobs = len(request.conversation_ids) * len(request.document_types)

        for conv_id in request.conversation_ids:
            for doc_type_str in request.document_types:
                doc_type = DocumentType(doc_type_str.value)

                job_id = service.create_generation_job(
                    conversation_id=conv_id,
                    document_type=doc_type,
                    provider=request.provider,
                    model=request.model,
                    prompt_config={},
                )
                job_ids.append(job_id)

        estimated_time = total_jobs * 10

        return BulkGenerateResponse(
            total_jobs=total_jobs,
            job_ids=job_ids,
            estimated_time_seconds=estimated_time,
            message=f"Created {total_jobs} generation jobs",
        )
    except CharactersRAGDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to create bulk generation jobs") from e
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error creating bulk generation jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create bulk generation jobs",
        ) from e


@router.get(
    "/documents/statistics",
    response_model=GenerationStatistics,
    summary="Get generation statistics",
    description="Aggregate statistics across generated documents (counts, durations, errors).",
    tags=["chat-documents"],
)
async def get_generation_statistics(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    service_cls: type[DocumentGeneratorService] = Depends(get_document_generator_service),
) -> GenerationStatistics:
    """Get statistics about document generation."""
    try:
        service = service_cls(db)

        all_docs = service.get_generated_documents(limit=1000)

        if not all_docs:
            return GenerationStatistics(
                total_documents=0,
                by_type={},
                by_provider={},
                average_generation_time_ms=0,
                total_tokens_used=None,
                last_generated=None,
                most_used_model=None,
            )

        by_type: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        total_time = 0
        total_tokens = 0
        models: dict[str, int] = {}

        for doc in all_docs:
            doc_type = doc["document_type"]
            by_type[doc_type] = by_type.get(doc_type, 0) + 1

            provider = doc["provider"]
            by_provider[provider] = by_provider.get(provider, 0) + 1

            total_time += doc.get("generation_time_ms", 0)

            if doc.get("token_count"):
                total_tokens += doc["token_count"]

            model = doc["model"]
            models[model] = models.get(model, 0) + 1

        most_used_model = max(models, key=models.get) if models else None

        last_doc = max(all_docs, key=lambda d: d["created_at"])

        return GenerationStatistics(
            total_documents=len(all_docs),
            by_type=by_type,
            by_provider=by_provider,
            average_generation_time_ms=total_time / len(all_docs) if all_docs else 0,
            total_tokens_used=total_tokens if total_tokens > 0 else None,
            last_generated=last_doc["created_at"],
            most_used_model=most_used_model,
        )
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting generation statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generation statistics",
        ) from e
