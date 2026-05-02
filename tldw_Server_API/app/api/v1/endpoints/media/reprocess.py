from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import media_embeddings as embeddings_endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import ReprocessMediaRequest
from tldw_Server_API.app.api.v1.schemas.media_response_models import ReprocessMediaResponse
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.utils.rag_cache import invalidate_rag_caches
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_UPDATE
from tldw_Server_API.app.core.Chunking import improved_chunking_process
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.Chunking.templates import TemplateClassifier
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.DB_Manager import mark_media_as_processed
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    apply_chunking_template_if_any,
    prepare_chunking_options_dict,
)

router = APIRouter(tags=["Media Management"])
_MARK_PROCESSED_CONFLICT_RETRIES = 3
_MARK_PROCESSED_CONFLICT_BACKOFF_SECONDS = 0.05


def _normalize_chunks(raw_chunks: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, str):
            text = chunk
            meta: dict[str, Any] | None = None
            start_char = None
            end_char = None
            chunk_type = None
            chunk_index = idx
        elif isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("chunk_text")
            meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else None
            start_char = chunk.get("start_char") or chunk.get("start")
            end_char = chunk.get("end_char") or chunk.get("end")
            chunk_type = chunk.get("chunk_type") or (meta or {}).get("chunk_type")
            raw_index = chunk.get("chunk_index")
            if raw_index is None:
                raw_index = (meta or {}).get("chunk_index")
            if raw_index is None:
                raw_index = (meta or {}).get("index")
            try:
                chunk_index = int(raw_index) if raw_index is not None else idx
            except (TypeError, ValueError):
                chunk_index = idx

            if start_char is None:
                start_char = (meta or {}).get("start_char")
            if start_char is None:
                start_char = (meta or {}).get("start_index")
            if start_char is None:
                start_char = (meta or {}).get("start_offset")

            if end_char is None:
                end_char = (meta or {}).get("end_char")
            if end_char is None:
                end_char = (meta or {}).get("end_index")
            if end_char is None:
                end_char = (meta or {}).get("end_offset")
        else:
            continue

        if not isinstance(text, str) or not text.strip():
            continue

        normalized.append(
            {
                "chunk_text": text,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
                "chunk_type": chunk_type,
                "metadata": meta if isinstance(meta, dict) else None,
            }
        )
    return normalized


async def _mark_embeddings_complete_with_retry(*, db: MediaDatabase, media_id: int, context: str) -> bool:
    for attempt in range(1, _MARK_PROCESSED_CONFLICT_RETRIES + 1):
        try:
            mark_media_as_processed(db_instance=db, media_id=media_id)
            return True
        except ConflictError as exc:
            if attempt >= _MARK_PROCESSED_CONFLICT_RETRIES:
                logger.warning(
                    "Embeddings completed for media {} but the ready-state update still conflicted after {} attempts in {}: {}",
                    media_id,
                    attempt,
                    context,
                    exc,
                )
                return False
            await asyncio.sleep(_MARK_PROCESSED_CONFLICT_BACKOFF_SECONDS * attempt)
    return False


def _delete_embeddings_for_media(media_id: int, user_id: str) -> None:
    manager = embeddings_endpoint.ChromaDBManager(
        user_id=user_id,
        user_embedding_config=embeddings_endpoint._user_embedding_config(),
    )
    collection_name = f"user_{user_id}_media_embeddings"
    collection = manager.get_or_create_collection(collection_name)
    try:
        collection.delete(where={"media_id": str(media_id)})
    except Exception:
        logger.warning("Where-delete failed for media embeddings, falling back to id delete")
        data = collection.get(where={"media_id": str(media_id)}, include=["metadatas"], limit=100000)
        ids = (data or {}).get("ids") or []
        if ids:
            collection.delete(ids=ids)
    try:
        remaining = collection.get(where={"media_id": str(media_id)}, include=["metadatas"], limit=1)
        remaining_ids = (remaining or {}).get("ids") or []
        if remaining_ids:
            collection.delete(ids=remaining_ids)
    except Exception:
        logger.warning("Failed to verify embeddings delete")


async def _generate_embeddings(
    *,
    media_id: int,
    media_payload: dict[str, Any],
    request: ReprocessMediaRequest,
    user_id: str,
    db: Any,
    cache_namespaces: list[str] | None = None,
) -> None:
    try:
        embedding_model = request.embedding_model or settings.get(
            "embedding_model",
            embeddings_endpoint.DEFAULT_EMBEDDING_MODEL,
        )
        embedding_provider = request.embedding_provider or settings.get(
            "embedding_provider",
            embeddings_endpoint.DEFAULT_EMBEDDING_PROVIDER,
        )
        result = await embeddings_endpoint.generate_embeddings_for_media(
            media_id=media_id,
            media_content=media_payload,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            user_id=user_id,
        )
        allow_zero = bool(result.get("allow_zero_embeddings"))
        if result.get("status") == "success" or allow_zero:
            await _mark_embeddings_complete_with_retry(
                db=db,
                media_id=media_id,
                context="media.reprocess",
            )
        else:
            error_detail = str(result.get("error") or result.get("message") or "Embedding generation failed")
            raise RuntimeError(error_detail)
        invalidate_rag_caches(None, namespaces=cache_namespaces, media_id=media_id)
    except Exception:
        logger.error("Embeddings regeneration failed")
        try:
            db.mark_embeddings_error(media_id, "Embeddings regeneration failed")
        except Exception:
            logger.error("Failed to mark embeddings error")
        raise


@router.post(
    "/{media_id:int}/reprocess",
    dependencies=[
        Depends(RequirePermission(MEDIA_UPDATE)),
        Depends(rbac_rate_limit("media.update")),
    ],
    status_code=status.HTTP_200_OK,
    response_model=ReprocessMediaResponse,
    summary="Reprocess media chunks and embeddings",
)
async def reprocess_media_item(
    payload: ReprocessMediaRequest,
    background_tasks: BackgroundTasks,
    media_id: int = Path(..., description="The ID of the media item"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> ReprocessMediaResponse:
    """
    Reprocess stored media content by rebuilding chunks and/or regenerating embeddings.
    """
    logger.info(
        "Reprocess request received for media_id={} (chunking={}, embeddings={})",
        media_id,
        payload.perform_chunking,
        payload.generate_embeddings,
    )
    media_item = db.get_media_by_id(media_id)
    if not media_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media item not found.")

    content = media_item.get("content") or ""
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Media content is empty.")

    if not payload.perform_chunking and not payload.generate_embeddings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nothing to reprocess (chunking and embeddings both disabled).",
        )

    chunks_created: int | None = None
    if payload.perform_chunking:
        chunk_request = SimpleNamespace(**payload.model_dump())
        chunk_request.media_type = media_item.get("type")
        chunk_request.title = media_item.get("title")
        chunk_request.url = media_item.get("url")

        chunk_options = prepare_chunking_options_dict(chunk_request)
        chunk_options = apply_chunking_template_if_any(
            chunk_request,
            db,
            chunk_options,
            TemplateClassifier=TemplateClassifier,
            first_url=media_item.get("url"),
            first_filename=media_item.get("filename"),
        )
        chunk_options = chunk_options or {}

        use_hier = bool(
            chunk_options.get("hierarchical") or isinstance(chunk_options.get("hierarchical_template"), dict)
        )
        try:
            if use_hier:
                ck = Chunker()
                raw_chunks = ck.chunk_text_hierarchical_flat(
                    content,
                    method=chunk_options.get("method") or "sentences",
                    max_size=chunk_options.get("max_size") or 500,
                    overlap=chunk_options.get("overlap") or 200,
                    language=chunk_options.get("language"),
                    template=(
                        chunk_options.get("hierarchical_template")
                        if isinstance(chunk_options.get("hierarchical_template"), dict)
                        else None
                    ),
                )
            else:
                raw_chunks = improved_chunking_process(content, chunk_options)
        except Exception as exc:
            logger.error(
                "Chunking failed for media {}",
                media_id,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to re-chunk media content.",
            ) from exc

        normalized_chunks = _normalize_chunks(raw_chunks)
        try:
            db.clear_unvectorized_chunks(media_id)
        except (InputError, ConflictError, DatabaseError) as exc:
            raise map_db_error_to_http(exc, default_detail="Failed to update media chunks") from exc
        if normalized_chunks:
            try:
                db.process_unvectorized_chunks(media_id, normalized_chunks)
            except (InputError, ConflictError, DatabaseError) as exc:
                raise map_db_error_to_http(exc, default_detail="Failed to update media chunks") from exc
        chunks_created = len(normalized_chunks)

        try:
            db.update_media_reprocess_state(
                media_id,
                chunking_status="completed",
                reset_vector_processing=bool(payload.generate_embeddings),
            )
        except (InputError, ConflictError, DatabaseError) as exc:
            raise map_db_error_to_http(exc, default_detail="Failed to update media reprocess state") from exc
    elif payload.generate_embeddings:
        try:
            db.update_media_reprocess_state(
                media_id,
                chunking_status=None,
                reset_vector_processing=True,
            )
        except (InputError, ConflictError, DatabaseError) as exc:
            raise map_db_error_to_http(exc, default_detail="Failed to update media reprocess state") from exc

    cache_namespaces: list[str] = []
    try:
        username = getattr(current_user, "username", None)
        if username:
            cache_namespaces.append(str(username))
        user_id_val = getattr(current_user, "id", None)
        if user_id_val is not None and user_id_val != "":
            cache_namespaces.append(str(user_id_val))
    except Exception:
        cache_namespaces = []

    embeddings_started = False
    if payload.generate_embeddings:
        raw_user_id = getattr(current_user, "id", None)
        if raw_user_id is None or raw_user_id == "":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated user id missing.",
            )
        user_id = str(raw_user_id)
        if payload.force_regenerate_embeddings:
            try:
                _delete_embeddings_for_media(media_id, user_id)
            except Exception:
                logger.warning("Failed to delete embeddings before regeneration")

        embeddings_started = True
        media_payload = {"media_item": media_item, "content": media_item}
        background_tasks.add_task(
            _generate_embeddings,
            media_id=media_id,
            media_payload=media_payload,
            request=payload,
            user_id=user_id,
            db=db,
            cache_namespaces=cache_namespaces,
        )

    message = "Reprocess completed."
    if payload.generate_embeddings:
        message = "Reprocess completed; embeddings regeneration started."

    invalidate_rag_caches(current_user, media_id=media_id)

    return ReprocessMediaResponse(
        media_id=media_id,
        status="completed",
        message=message,
        chunks_created=chunks_created,
        embeddings_started=embeddings_started,
        job_id=None,
    )


__all__ = ["router"]
