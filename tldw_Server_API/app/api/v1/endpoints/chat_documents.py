from __future__ import annotations

import asyncio
import contextlib
import datetime
import inspect
import json
import sqlite3
import time
from collections.abc import AsyncIterator
from typing import Any, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.chat_documents_deps import get_document_generator_service
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
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
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    derive_trusted_credential_scope,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    record_byok_missing_credentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    capture_provider_override_call_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    DaemonCapacityError,
    await_bounded_sync_call,
    await_owned_worker,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.Chat.document_generator import (
    DocumentGeneratorService,
    DocumentType,
)
from tldw_Server_API.app.core.Chat.document_generator import (
    GenerationStatus as GenStatus,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    invoke_stream_close_bounded,
    normalize_provider_stream_error,
    provider_stream_error_payload,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import provider_auth_is_resolved
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.testing import env_flag_enabled

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


def _is_lazy_document_stream(value: Any) -> bool:
    """Return whether *value* is a deferred provider stream."""
    return hasattr(value, "__aiter__") or (
        hasattr(value, "__iter__")
        and not isinstance(value, (str, bytes, bytearray, dict, list, tuple))
    )


def _document_nonstream_result_is_successful(value: Any) -> bool:
    """Return whether a completed factory result is valid provider output."""

    if _is_lazy_document_stream(value) or normalize_provider_stream_error(value) is not None:
        return False
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return bool(value.strip()) and not value.lstrip().lower().startswith("error:")
    return False


def _normalize_document_stream_chunk(chunk: Any) -> str:
    """Normalize one document-provider chunk for classification and delivery."""

    if chunk is None:
        return ""
    if isinstance(chunk, (bytes, bytearray)):
        return chunk.decode("utf-8", errors="replace")
    return str(chunk)


def _classify_document_stream_chunk(chunk: Any) -> tuple[str, bool, str | None]:
    """Return safe payload, semantic-success flag, and terminal kind."""

    normalized_error = normalize_provider_stream_error(chunk)
    if normalized_error is not None:
        payload = json.dumps(
            provider_stream_error_payload(normalized_error),
            separators=(",", ":"),
        )
        return payload, False, "error"

    payload = _normalize_document_stream_chunk(chunk)
    control = payload.strip().lower()
    if control == "[done]" or (
        control.startswith("data:") and control.removeprefix("data:").strip() == "[done]"
    ):
        return "[DONE]", False, "done"
    return payload, bool(control), None


def _record_document_stream_chunk_result(
    success_state: dict[str, bool],
    chunk: Any,
) -> None:
    """Apply the shared fail-closed stream-result accounting contract."""

    _payload, successful, terminal_kind = _classify_document_stream_chunk(chunk)
    if terminal_kind == "error":
        _record_document_stream_terminal_failure(success_state)
    elif not success_state.get("terminal_error") and successful:
        success_state["successful"] = True


def _record_document_stream_terminal_failure(
    success_state: dict[str, bool],
) -> None:
    """Invalidate all partial output after a terminal stream failure."""

    success_state["successful"] = False
    success_state["terminal_error"] = True


def _next_document_stream_chunk(iterator: Any) -> tuple[bool, Any]:
    """Read one sync chunk without crossing asyncio with StopIteration."""
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


async def _iterate_document_stream(
    source: Any,
    *,
    resource_holder: dict[str, Any],
    success_state: dict[str, bool],
) -> AsyncIterator[Any]:
    """Iterate provider output while retaining ownership through cancellation."""

    def record_cancelled_chunk(chunk: Any) -> None:
        _record_document_stream_chunk_result(success_state, chunk)

    def record_cancelled_sync_result(result: Any) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            return
        finished, chunk = result
        if not finished:
            record_cancelled_chunk(chunk)

    if hasattr(source, "__aiter__"):
        iterator = source.__aiter__()
        resource_holder["iterator"] = iterator
        while True:
            try:
                chunk = await await_owned_worker(
                    iterator.__anext__(),
                    on_cancel_result=record_cancelled_chunk,
                )
            except StopAsyncIteration:
                return
            yield chunk

    if hasattr(source, "__iter__") and not isinstance(
        source,
        (str, bytes, bytearray),
    ):
        def retain_cancelled_iterator(iterator: Any) -> None:
            resource_holder["iterator"] = iterator

        iterator = await await_owned_worker(
            await_bounded_sync_call(
                lambda: iter(source),
                pool=SYNC_ADAPTER_CALL_POOL,
                exhaustion_message="Document provider capacity is exhausted",
            ),
            on_cancel_result=retain_cancelled_iterator,
        )
        resource_holder["iterator"] = iterator
        while True:
            finished, chunk = await await_owned_worker(
                await_bounded_sync_call(
                    lambda: _next_document_stream_chunk(iterator),
                    pool=SYNC_ADAPTER_CALL_POOL,
                    exhaustion_message="Document provider capacity is exhausted",
                ),
                on_cancel_result=record_cancelled_sync_result,
            )
            if finished:
                return
            yield chunk

    yield source


async def _close_document_stream(
    source: Any,
    resource_holder: dict[str, Any] | None = None,
) -> None:
    """Close provider output before its credential runtime is released."""
    iterator = (resource_holder or {}).get("iterator")
    candidates = (iterator, source) if iterator is not source else (source,)
    for candidate in candidates:
        if candidate is None:
            continue
        close = getattr(candidate, "aclose", None)
        if not callable(close):
            close = getattr(candidate, "close", None)
        if not callable(close):
            continue
        try:
            await await_owned_worker(invoke_stream_close_bounded(close))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - cleanup remains best effort
            logger.debug(
                "Document provider stream cleanup failed error_type={}",
                type(exc).__name__,
            )
        return


def _build_document_stream_cleanup(
    *,
    runtime: ProviderCredentialRuntime,
    credentials: Any,
    source: Any,
    resource_holder: dict[str, Any],
    success_state: dict[str, bool],
):
    """Build idempotent stream cleanup with usage, source, runtime ordering."""
    lock = asyncio.Lock()
    cleanup_done = False

    async def cleanup_once() -> None:
        nonlocal cleanup_done
        async with lock:
            if cleanup_done:
                return
            try:
                if success_state.get("successful"):
                    try:
                        await runtime.mark_used(credentials)
                    except Exception as exc:  # noqa: BLE001 - usage tracking is best effort
                        logger.debug(
                            "Document credential usage tracking failed error_type={}",
                            type(exc).__name__,
                        )
            finally:
                try:
                    await _close_document_stream(source, resource_holder)
                finally:
                    await runtime.close()
                    cleanup_done = True

    async def cleanup() -> None:
        await await_owned_worker(cleanup_once())

    return cleanup


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
    credential_runtime: ProviderCredentialRuntime | None = None
    credential_runtime_transferred = False
    stream_cleanup: Any = None
    try:
        service = service_cls(db)

        doc_type = DocumentType(request.document_type.value)

        provider_name = canonical_provider_name(request.provider or "")
        if not provider_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provider is required")
        provider_key = provider_name

        # Resolve provider key requirements
        try:
            from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
        except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS:
            def provider_requires_api_key(_provider: str) -> bool:  # type: ignore[misc]
                return True

        if request.api_key:
            logger.debug("Ignoring per-request api_key override for provider={}", provider_name)
        user_id_int, team_ids, org_ids, trusted_base_url_override = (
            derive_trusted_credential_scope(http_request, principal)
        )
        credential_runtime = ProviderCredentialRuntime(
            user_id=user_id_int,
            team_ids=team_ids,
            org_ids=org_ids,
            trusted_base_url_override=trusted_base_url_override,
            override_snapshot_resolver=capture_provider_override_call_snapshot,
        )
        provider_credentials = await credential_runtime.resolve(
            provider_key,
            model=request.model,
        )
        provider_api_key = provider_credentials.api_key
        app_config_override = provider_credentials.app_config

        if provider_requires_api_key(provider_key) and not provider_auth_is_resolved(
            provider_key,
            api_key=provider_api_key,
            app_config=app_config_override,
            credentials_resolved=provider_credentials.credentials_resolved,
        ):
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
                credentials_resolved=True,
                provider_credentials=provider_credentials,
                specific_message=request.specific_message,
                custom_prompt=request.custom_prompt,
                stream=stream,
            )

        async def _execute_generation() -> Any:
            result = await await_bounded_sync_call(
                lambda: _generate_doc(request.stream),
                pool=SYNC_ADAPTER_CALL_POOL,
                exhaustion_message="Document provider capacity is exhausted",
            )
            if inspect.isawaitable(result):
                result = await result
            return result

        async def _finalize_cancelled_generation(result: Any) -> None:
            if _is_lazy_document_stream(result):
                await _close_document_stream(result)
            elif _document_nonstream_result_is_successful(result):
                await credential_runtime.mark_used(provider_credentials)

        content = await await_owned_worker(
            _execute_generation(),
            on_cancel_result=_finalize_cancelled_generation,
        )

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
            stream_resource_holder: dict[str, Any] = {}
            stream_success_state = {"successful": False}
            stream_cleanup = _build_document_stream_cleanup(
                runtime=credential_runtime,
                credentials=provider_credentials,
                source=streaming_source,
                resource_holder=stream_resource_holder,
                success_state=stream_success_state,
            )

            def _encode_sse(text: str) -> str:
                lines = text.splitlines() or [""]
                return "".join(f"data: {line}\n" for line in lines) + "\n"

            async def _iter_stream() -> AsyncIterator[Any]:
                async for chunk in _iterate_document_stream(
                    streaming_source,
                    resource_holder=stream_resource_holder,
                    success_state=stream_success_state,
                ):
                    yield chunk

            stream_started_at = time.perf_counter()
            collected_chunks: list[str] = []

            if env_flag_enabled("STREAMS_UNIFIED"):
                from tldw_Server_API.app.core.Streaming.streams import SSEStream

                stream = SSEStream(labels={"component": "chat", "endpoint": "chat_doc_stream"})

                async def _produce() -> None:
                    try:
                        async for chunk in _iter_stream():
                            payload, successful, terminal_kind = (
                                _classify_document_stream_chunk(chunk)
                            )
                            if terminal_kind == "error":
                                _record_document_stream_terminal_failure(
                                    stream_success_state
                                )
                                await stream.send_raw_sse_line(f"data: {payload}")
                                await stream.done()
                                return
                            if terminal_kind == "done":
                                await stream.done()
                                return
                            if successful and not stream_success_state.get("terminal_error"):
                                stream_success_state["successful"] = True
                            if not payload:
                                continue
                            collected_chunks.append(payload)
                            for line in payload.splitlines() or [""]:
                                await stream.send_raw_sse_line(f"data: {line}")
                        await stream.done()
                    except asyncio.CancelledError:
                        raise
                    except ChatAPIError as exc:
                        _record_document_stream_terminal_failure(stream_success_state)
                        logger.debug(
                            "Document stream provider failure error_type={}",
                            type(exc).__name__,
                        )
                        await stream.error("provider_error", "Chat provider error")
                    except Exception as exc:  # noqa: BLE001 - lazy failures need a terminal frame
                        _record_document_stream_terminal_failure(stream_success_state)
                        logger.debug(
                            "Document stream failure error_type={}",
                            type(exc).__name__,
                        )
                        await stream.error("internal_error", "An internal error has occurred.")

                async def _gen() -> AsyncIterator[str]:
                    prod = asyncio.create_task(_produce())
                    try:
                        async for ln in stream.iter_sse():
                            yield ln
                    except asyncio.CancelledError:
                        if not prod.done():
                            prod.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await prod
                        raise
                    else:
                        if not prod.done():
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await prod
                        try:
                            document_body = "".join(collected_chunks).strip()
                            if (
                                document_body
                                and stream_success_state.get("successful")
                                and not stream_success_state.get("terminal_error")
                            ):
                                generation_time_ms = int((time.perf_counter() - stream_started_at) * 1000)
                                await await_owned_worker(
                                    asyncio.to_thread(
                                        service.record_streamed_document,
                                        conversation_id=request.conversation_id,
                                        document_type=doc_type,
                                        content=document_body,
                                        provider=provider_name,
                                        model=request.model,
                                        generation_time_ms=generation_time_ms,
                                    )
                                )
                            else:
                                logger.info(
                                    'Streamed document produced no content for conversation {}; skipping persistence',
                                    request.conversation_id,
                                )
                        except asyncio.CancelledError:
                            raise
                        except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as persist_exc:
                            logger.error(
                                'Failed to persist streamed document for conversation {} error_type={}',
                                request.conversation_id,
                                type(persist_exc).__name__,
                            )
                    finally:
                        if not prod.done():
                            prod.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await prod
                        await stream_cleanup()

                headers = {
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
                response = StreamingResponse(
                    _gen(),
                    media_type="text/event-stream",
                    headers=headers,
                    background=BackgroundTask(stream_cleanup),
                )
                credential_runtime_transferred = True
                return response

            async def _sse_stream() -> AsyncIterator[str]:
                try:
                    async for chunk in _iter_stream():
                        payload, successful, terminal_kind = (
                            _classify_document_stream_chunk(chunk)
                        )
                        if terminal_kind == "error":
                            _record_document_stream_terminal_failure(
                                stream_success_state
                            )
                            yield _encode_sse(payload)
                            return
                        if terminal_kind == "done":
                            yield _encode_sse(payload)
                            return
                        if successful and not stream_success_state.get("terminal_error"):
                            stream_success_state["successful"] = True
                        if payload:
                            collected_chunks.append(payload)
                            yield _encode_sse(payload)
                except asyncio.CancelledError:
                    logger.info(
                        'Document generation stream cancelled for conversation {}',
                        request.conversation_id,
                    )
                    raise
                except ChatAPIError as exc:
                    _record_document_stream_terminal_failure(stream_success_state)
                    logger.debug(
                        "Document stream provider failure error_type={}",
                        type(exc).__name__,
                    )
                    yield _encode_sse("Chat provider error")
                except Exception as exc:  # noqa: BLE001 - lazy failures need a terminal frame
                    _record_document_stream_terminal_failure(stream_success_state)
                    logger.debug(
                        "Document stream failure error_type={}",
                        type(exc).__name__,
                    )
                    yield _encode_sse("An internal error has occurred.")
                finally:
                    try:
                        document_body = "".join(collected_chunks).strip()
                        if (
                            document_body
                            and stream_success_state.get("successful")
                            and not stream_success_state.get("terminal_error")
                        ):
                            generation_time_ms = int((time.perf_counter() - stream_started_at) * 1000)
                            await await_owned_worker(
                                asyncio.to_thread(
                                    service.record_streamed_document,
                                    conversation_id=request.conversation_id,
                                    document_type=doc_type,
                                    content=document_body,
                                    provider=provider_name,
                                    model=request.model,
                                    generation_time_ms=generation_time_ms,
                                )
                            )
                        else:
                            logger.info(
                                'Streamed document produced no content for conversation {}; skipping persistence',
                                request.conversation_id,
                            )
                    except asyncio.CancelledError:
                        raise
                    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as persist_exc:
                        logger.error(
                            'Failed to persist streamed document for conversation {} error_type={}',
                            request.conversation_id,
                            type(persist_exc).__name__,
                        )
                    finally:
                        await stream_cleanup()

            headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
            response = StreamingResponse(
                _sse_stream(),
                media_type="text/event-stream",
                headers=headers,
                background=BackgroundTask(stream_cleanup),
            )
            credential_runtime_transferred = True
            return response

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
        await credential_runtime.mark_used(provider_credentials)
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

    except asyncio.CancelledError:
        raise
    except DaemonCapacityError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "provider_capacity_exhausted",
                "message": "The chat service provider is temporarily busy.",
            },
        ) from None
    except InputError as e:
        logger.warning(
            "Input error generating document error_type={}",
            type(e).__name__,
        )
        raise map_db_error_to_http(e) from e
    except ChatAPIError as e:
        logger.error(
            "API error generating document error_type={}",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The chat service provider is currently unavailable.",
        ) from e
    except HTTPException:
        raise
    except ByokResolutionError as e:
        code = getattr(e, "policy_code", e.code)
        if code in {"provider_disabled", "model_not_allowed", "credential_scope_revoked"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": code,
                    "message": (
                        "The active credential scope is no longer available."
                        if code == "credential_scope_revoked"
                        else "The selected provider or model is disabled by administrator policy."
                    ),
                },
            ) from None
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": code,
                "message": "Provider credentials are temporarily unavailable.",
            },
        ) from None
    except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(
            "Unexpected error generating document error_type={}",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred.",
        ) from e
    finally:
        if credential_runtime is not None and not credential_runtime_transferred:
            if stream_cleanup is not None:
                await stream_cleanup()
            else:
                await await_owned_worker(credential_runtime.close())


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
        logger.error("Error getting job status error_type={}", type(e).__name__)
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
        logger.error("Error cancelling job error_type={}", type(e).__name__)
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
        logger.error(
            "Error listing generated documents error_type={}",
            type(e).__name__,
        )
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
        logger.error(
            "Error getting document {} error_type={}",
            document_id,
            type(e).__name__,
        )
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
        logger.error(
            "Error deleting document {} error_type={}",
            document_id,
            type(e).__name__,
        )
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
        logger.error("Error saving prompt config error_type={}", type(e).__name__)
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
            logger.warning(
                "Database operational error checking custom prompts error_type={}",
                type(e).__name__,
            )
            is_custom = False
        except sqlite3.DatabaseError as e:
            logger.error(
                "Database error checking custom prompts doc_type={} error_type={}",
                doc_type.value,
                type(e).__name__,
            )
            is_custom = False
        except _CHAT_DOCS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(
                "Unexpected error checking custom prompts error_type={}",
                type(e).__name__,
            )
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
        logger.error("Error getting prompt config error_type={}", type(e).__name__)
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
        logger.error(
            "Error creating bulk generation jobs error_type={}",
            type(e).__name__,
        )
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
        logger.error(
            "Error getting generation statistics error_type={}",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generation statistics",
        ) from e
