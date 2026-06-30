# media_embeddings.py
# Description: Endpoints for managing embeddings for media items
#
# This module provides endpoints to:
# - Check if embeddings exist for a media item
# - Generate embeddings for uploaded media
# - Delete embeddings for a media item

import json
import os
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status as http_status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, resolve_user_id_for_request, User
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta


# Local imports
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.utils.rag_cache import invalidate_rag_caches
from tldw_Server_API.app.core.Chunking.base import ChunkerConfig
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_by_id
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Embeddings.jobs_adapter import EmbeddingsJobsAdapter

router = APIRouter(prefix="/media", tags=["media-embeddings"])

# Default embedding settings
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_PROVIDER = "huggingface"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_MEDIA_EMBEDDING_JOBS_OFFSET = 10_000

_MEDIA_EMBEDDINGS_PARSE_EXCEPTIONS = (TypeError, ValueError, UnicodeError, json.JSONDecodeError)
_MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeError,
    json.JSONDecodeError,
)


def _user_embedding_config() -> dict[str, Any]:
    cfg = settings.get("EMBEDDING_CONFIG", {}).copy()
    try:
        user_db_base_dir = str(DatabasePaths.get_user_db_base_dir())
    except Exception:
        logger.warning("Falling back to USER_DB_BASE_DIR setting after user DB base resolution failed")
        user_db_base_dir = settings.get("USER_DB_BASE_DIR")
    cfg["USER_DB_BASE_DIR"] = user_db_base_dir
    return cfg


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _parse_media_type_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip().lower() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    return []


def _allow_zero_embeddings_for_media(media_item: dict[str, Any]) -> bool:
    media_type = str(media_item.get("media_type") or media_item.get("type") or "").strip().lower()
    if not media_type:
        return False

    cfg = settings.get("EMBEDDING_CONFIG", {}) or {}
    raw = cfg.get("allow_zero_embeddings_media_types")
    if raw is None:
        raw = cfg.get("skip_media_types")
    if raw is None:
        raw = cfg.get("no_embeddings_media_types")
    if raw is None:
        raw = os.getenv("ALLOW_ZERO_EMBEDDINGS_MEDIA_TYPES") or os.getenv("EMBEDDINGS_SKIP_MEDIA_TYPES")

    allowed = {"audio", "video"} if raw is None else set(_parse_media_type_list(raw))

    return media_type in allowed


def _resolve_model_provider(
    embedding_model: Optional[str],
    embedding_provider: Optional[str],
) -> tuple[str, str]:
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            _resolve_model_and_provider,
        )

        return _resolve_model_and_provider(embedding_model, embedding_provider)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        default_model = embedding_model or settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        resolved_provider = embedding_provider or settings.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
        if isinstance(default_model, str) and ":" in default_model and not embedding_provider:
            resolved_provider = default_model.split(":", 1)[0]
        return default_model, resolved_provider


def _embeddings_jobs_backend() -> str:
    raw = (os.getenv("EMBEDDINGS_JOBS_BACKEND") or os.getenv("TLDW_JOBS_BACKEND") or "").strip().lower()
    if raw in {"jobs", "core", ""}:
        return "jobs"
    logger.warning("Embeddings jobs backend override ignored; core Jobs is the only backend")
    return "jobs"


class GenerateEmbeddingsRequest(BaseModel):
    """Request model for generating embeddings"""
    embedding_model: Optional[str] = Field(
        None,
        description="Specific embedding model to use (defaults to Qwen3-Embedding-4B-GGUF)"
    )
    embedding_provider: Optional[str] = Field(
        None,
        description="Embedding provider (huggingface, openai, etc)"
    )
    chunk_size: int = Field(
        1000,
        description="Size of text chunks for embedding"
    )
    chunk_overlap: int = Field(
        200,
        description="Overlap between chunks"
    )
    force_regenerate: bool = Field(
        False,
        description="Force regeneration even if embeddings exist"
    )
    priority: int = Field(
        50,
        ge=0,
        le=100,
        description="Embedding job priority (0-100)"
    )

class EmbeddingsStatusResponse(BaseModel):
    """Response model for embedding status"""
    media_id: int
    has_embeddings: bool
    embedding_count: Optional[int] = None
    embedding_model: Optional[str] = None
    last_generated: Optional[str] = None

class GenerateEmbeddingsResponse(BaseModel):
    """Response model for embedding generation"""
    media_id: int
    status: str
    message: str
    embedding_count: Optional[int] = None
    embedding_model: str
    chunks_processed: Optional[int] = None
    job_id: Optional[str] = None


class BatchMediaEmbeddingsRequest(BaseModel):
    media_ids: list[int] = Field(..., min_length=1, description="List of media IDs to embed")
    embedding_model: Optional[str] = Field(None, alias="model")
    embedding_provider: Optional[str] = Field(None, alias="provider")
    chunk_size: int = Field(1000, description="Chunk size to use for each media item")
    chunk_overlap: int = Field(200, description="Chunk overlap to use for each media item")
    force_regenerate: bool = Field(False, description="Force regeneration even if embeddings exist")
    priority: int = Field(50, ge=0, le=100, description="Embedding job priority (0-100)")

    model_config = ConfigDict(populate_by_name=True)


class BatchMediaEmbeddingsResponse(BaseModel):
    status: Literal["accepted", "partial"]
    job_ids: list[str]
    submitted: int
    failed_media_ids: list[int] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class EmbeddingsSearchRequest(BaseModel):
    query: str = Field(..., description="Query text to embed and search with")
    top_k: int = Field(5, gt=0, le=100, description="Number of nearest results to return")
    collection: Optional[str] = Field(None, description="Target collection to search")
    embedding_model: Optional[str] = Field(None, alias="model")
    embedding_provider: Optional[str] = Field(None, alias="provider")
    filters: Optional[dict[str, Any]] = Field(None, description="Optional metadata filters")

    model_config = ConfigDict(populate_by_name=True)


class EmbeddingsSearchResult(BaseModel):
    id: Optional[str]
    document: Optional[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = None


class EmbeddingsSearchResponse(BaseModel):
    results: list[EmbeddingsSearchResult]
    count: int


async def get_media_content(media_id: int, db: Any) -> dict[str, Any]:
    """Retrieve media content from database"""
    try:
        # Get media item details
        media_item = get_media_by_id(db, media_id)
        if not media_item:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Media item {media_id} not found"
            )

        # Fall back to latest document version content when Media.content is empty.
        try:
            if isinstance(media_item, dict) and not (media_item.get("content") or "").strip():
                from tldw_Server_API.app.core.DB_Management.media_db.api import (
                    get_document_version,
                )
                latest = get_document_version(db, media_id=media_id, version_number=None, include_content=True)
                if latest and latest.get("content"):
                    media_item = dict(media_item)
                    media_item["content"] = latest["content"]
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            logger.warning("Failed to load fallback document content")

        # Get content
        content = media_item  # The get_media_by_id returns all data including content
        if not content:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"No content found for media item {media_id}"
            )

        return {
            "media_item": media_item,
            "content": content
        }
    except HTTPException:
        # Propagate explicit HTTP errors (e.g., 404 Not Found)
        raise
    except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error retrieving media content")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving media content"
        ) from e


def chunk_media_content(text: str, chunk_size: int = 1000, overlap: int = 200, method: str = "words") -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks using the Chunking module.

    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks
        method: Chunking method to use (words, sentences, tokens, etc.)

    Returns:
        List of chunk dictionaries with text and metadata
    """
    # Initialize chunker with configuration
    config = ChunkerConfig(
        default_method=method,
        default_max_size=chunk_size,
        default_overlap=overlap,
        language="en"  # TODO: Detect language from content
    )

    chunker = Chunker(config=config)

    # Use chunk_text_with_metadata to get detailed results
    chunk_results = chunker.chunk_text_with_metadata(
        text=text,
        method=method,
        max_size=chunk_size,
        overlap=overlap
    )

    # Convert ChunkResult objects to our format
    chunks = []
    for i, result in enumerate(chunk_results):
        # ChunkMetadata is an object, not a dict
        chunks.append({
            "text": result.text,
            "index": i,
            "start": result.metadata.start_char if hasattr(result.metadata, 'start_char') else i * (chunk_size - overlap),
            "end": result.metadata.end_char if hasattr(result.metadata, 'end_char') else (i + 1) * chunk_size,
            "metadata": {
                "word_count": result.metadata.word_count if hasattr(result.metadata, 'word_count') else None,
                "language": result.metadata.language if hasattr(result.metadata, 'language') else "en"
            }
        })

    return chunks


async def generate_embeddings_for_media(
    media_id: int,
    media_content: dict[str, Any],
    embedding_model: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
    user_id: str = "1"
) -> dict[str, Any]:
    """Generate embeddings for media content"""
    request_metadata = {"user_id": str(user_id)}
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import create_embeddings_batch_async

        # Extract text content
        allow_zero_embeddings = _allow_zero_embeddings_for_media(media_content.get("media_item", {}))
        content_payload = media_content.get("content")
        if isinstance(content_payload, dict):
            content_text = content_payload.get("content") or content_payload.get("text") or ""
        elif isinstance(content_payload, str):
            content_text = content_payload
        else:
            content_text = ""
        if not content_text or not content_text.strip():
            msg = "No text content to generate embeddings from"
            if allow_zero_embeddings:
                return {
                    "status": "success",
                    "message": msg,
                    "embedding_count": 0,
                    "chunks_processed": 0,
                    "allow_zero_embeddings": True,
                }
            return {
                "status": "error",
                "message": msg,
                "error": msg,
                "embedding_count": 0,
                "chunks_processed": 0,
            }

        # Chunk the text using the Chunking module
        chunks = chunk_media_content(content_text, chunk_size, chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks for media {media_id}")
        if not chunks:
            msg = "No chunks produced from media content"
            if allow_zero_embeddings:
                return {
                    "status": "success",
                    "message": msg,
                    "embedding_count": 0,
                    "chunks_processed": 0,
                    "allow_zero_embeddings": True,
                }
            return {
                "status": "error",
                "message": msg,
                "error": msg,
                "embedding_count": 0,
                "chunks_processed": 0,
            }

        # Extract chunk texts for embedding
        chunk_texts = [chunk["text"] for chunk in chunks]
        if not chunk_texts:
            msg = "No chunk text available to embed"
            if allow_zero_embeddings:
                return {
                    "status": "success",
                    "message": msg,
                    "embedding_count": 0,
                    "chunks_processed": 0,
                    "allow_zero_embeddings": True,
                }
            return {
                "status": "error",
                "message": msg,
                "error": msg,
                "embedding_count": 0,
                "chunks_processed": 0,
            }

        def _chunk_type_for_metadata(chunk: dict[str, Any]) -> str:
            chunk_metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            raw_chunk_type = (
                chunk.get("chunk_type")
                or chunk_metadata.get("chunk_type")
                or chunk_metadata.get("paragraph_kind")
                or chunk_metadata.get("type")
                or chunk_metadata.get("kind")
            )
            try:
                return Chunker.normalize_chunk_type(raw_chunk_type) or "text"
            except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                return "text"

        def _validate_embeddings_result(embeddings, expected_count: int) -> Optional[str]:
            if not embeddings:
                return "Embedding service returned no embeddings"
            if len(embeddings) != expected_count:
                return f"Embedding service returned {len(embeddings)} embeddings for {expected_count} chunks"
            for emb in embeddings:
                if emb is None:
                    return "Embedding service returned empty embedding vectors"
                try:
                    if len(emb) == 0:
                        return "Embedding service returned empty embedding vectors"
                except TypeError:
                    return "Embedding service returned invalid embedding vectors"
            return None

        def _storage_failure_result(exc: Exception, embedding_count: int) -> dict[str, Any]:
            message = f"Storage failure while saving embeddings to ChromaDB: {exc}"
            return {
                "status": "error",
                "message": message,
                "error": message,
                "embedding_count": embedding_count,
                "chunks_processed": len(chunks),
            }

        def _store_embeddings(
            *,
            model_name: str,
            provider_name: str,
            embeddings_to_store: list[Any],
        ) -> None:
            collection_name = f"user_{user_id}_media_embeddings"

            extra_metadata = {}
            try:
                media_item_meta = media_content.get("media_item", {})
                if isinstance(media_item_meta, dict):
                    extra_metadata = media_item_meta.get("metadata") or {}
            except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                extra_metadata = {}

            metadatas = []
            for _i, chunk in enumerate(chunks):
                metadata = {
                    "media_id": str(media_id),
                    "chunk_index": chunk["index"],
                    "chunk_start": chunk["start"],
                    "chunk_end": chunk["end"],
                    "chunk_type": _chunk_type_for_metadata(chunk),
                    "title": media_content["media_item"].get("title", ""),
                    "author": media_content["media_item"].get("author", ""),
                    "embedding_model": model_name,
                    "embedding_provider": provider_name,
                }
                if isinstance(extra_metadata, dict) and extra_metadata:
                    metadata["extra"] = dict(extra_metadata)
                metadatas.append(metadata)

            ids = [f"media_{media_id}_chunk_{i}" for i in range(len(chunks))]

            logger.info(
                f"Embeddings type: {type(embeddings_to_store)}, first item type: {type(embeddings_to_store[0]) if embeddings_to_store else 'None'}"
            )
            if embeddings_to_store and hasattr(embeddings_to_store[0], "tolist"):
                embeddings_list = [emb.tolist() for emb in embeddings_to_store]
            else:
                embeddings_list = embeddings_to_store

            logger.info(
                f"After conversion - embeddings_list type: {type(embeddings_list)}, first item type: {type(embeddings_list[0]) if embeddings_list else 'None'}"
            )
            logger.info(
                f"First embedding length: {len(embeddings_list[0]) if embeddings_list and embeddings_list[0] else 0}"
            )

            manager = ChromaDBManager(
                user_id=str(user_id),
                user_embedding_config=_user_embedding_config(),
            )
            manager.store_in_chroma(
                collection_name=collection_name,
                texts=chunk_texts,
                embeddings=embeddings_list,
                ids=ids,
                metadatas=metadatas,
                embedding_model_id_for_dim_check=model_name,
            )

        try:
            embeddings = await create_embeddings_batch_async(
                texts=chunk_texts,
                provider=embedding_provider,
                model_id=embedding_model,
                metadata=request_metadata,
            )
        except Exception:
            if embedding_model != FALLBACK_EMBEDDING_MODEL:
                logger.warning("Primary embedding generation failed; trying fallback model")
                try:
                    embeddings = await create_embeddings_batch_async(
                        texts=chunk_texts,
                        provider="huggingface",
                        model_id=FALLBACK_EMBEDDING_MODEL,
                        metadata=request_metadata,
                    )
                except Exception as exc:
                    logger.error("Fallback embedding generation failed")
                    return {
                        "status": "error",
                        "message": "Failed to generate embeddings",
                        "error": "Embedding generation failed",
                        "embedding_count": 0,
                        "chunks_processed": len(chunks),
                    }

                validation_error = _validate_embeddings_result(embeddings, len(chunk_texts))
                if validation_error:
                    return {
                        "status": "error",
                        "message": validation_error,
                        "error": validation_error,
                        "embedding_count": len(embeddings) if embeddings else 0,
                        "chunks_processed": len(chunks),
                    }

                try:
                    _store_embeddings(
                        model_name=FALLBACK_EMBEDDING_MODEL,
                        provider_name="huggingface",
                        embeddings_to_store=embeddings,
                    )
                except Exception as exc:
                    logger.error("Error storing fallback embeddings")
                    return _storage_failure_result(exc, len(embeddings) if embeddings else 0)

                return {
                    "status": "success",
                    "message": f"Generated embeddings using fallback model {FALLBACK_EMBEDDING_MODEL}",
                    "embedding_count": len(embeddings),
                    "chunks_processed": len(chunks),
                }
            raise

        validation_error = _validate_embeddings_result(embeddings, len(chunk_texts))
        if validation_error:
            return {
                "status": "error",
                "message": validation_error,
                "error": validation_error,
                "embedding_count": len(embeddings) if embeddings else 0,
                "chunks_processed": len(chunks),
            }

        try:
            _store_embeddings(
                model_name=embedding_model,
                provider_name=embedding_provider,
            embeddings_to_store=embeddings,
        )
        except Exception as exc:
            logger.error("Error storing primary embeddings")
            return _storage_failure_result(exc, len(embeddings) if embeddings else 0)

        return {
            "status": "success",
            "message": f"Successfully generated {len(embeddings)} embeddings",
            "embedding_count": len(embeddings),
            "chunks_processed": len(chunks),
        }

    except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        logger.error("Error generating embeddings")
        return {
            "status": "error",
            "message": "Failed to generate embeddings",
            "error": "Embedding generation failed",
            "embedding_count": 0,
            "chunks_processed": 0,
        }


@router.get(
    "/{media_id}/embeddings/status",
    response_model=EmbeddingsStatusResponse,
    dependencies=[Depends(rbac_rate_limit("embeddings.status"))],
)
async def get_embeddings_status(
    media_id: int,
    db: Annotated[Any, Depends(get_media_db_for_user)],
    current_user: Annotated[User, Depends(get_request_user)],
) -> EmbeddingsStatusResponse:
    """Check if embeddings exist for a media item"""
    try:
        # Check if media exists
        media_item = get_media_by_id(db, media_id)
        if not media_item:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Media item {media_id} not found"
            )

        # Check if embeddings exist by querying the per-user collection in ChromaDB
        user_id = resolve_user_id_for_request(
            current_user,
            error_status=http_status.HTTP_400_BAD_REQUEST,
        )
        manager = ChromaDBManager(user_id=user_id, user_embedding_config=_user_embedding_config())
        collection_name = f"user_{user_id}_media_embeddings"

        has_embeddings = False
        embedding_count: Optional[int] = None
        embedding_model: Optional[str] = None
        last_generated: Optional[str] = None

        try:
            collection = manager.get_or_create_collection(collection_name)
            # Try a filtered get to see if any vectors exist for this media_id
            data = collection.get(where={"media_id": str(media_id)}, include=["metadatas"], limit=100000)
            ids = (data or {}).get("ids") or []
            has_embeddings = len(ids) > 0
            if has_embeddings:
                embedding_count = len(ids)
                # Try to infer embedding model from first metadata
                md_list = (data or {}).get("metadatas") or []
                if md_list:
                    first_md = md_list[0]
                    embedding_model = first_md.get("embedding_model") if isinstance(first_md, dict) else None
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            logger.warning("ChromaDB status check failed")

        return EmbeddingsStatusResponse(
            media_id=media_id,
            has_embeddings=has_embeddings,
            embedding_count=embedding_count,
            embedding_model=embedding_model,
            last_generated=last_generated,
        )

    except HTTPException:
        raise
    except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error checking embeddings status")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking embeddings status"
        ) from e


@router.post(
    "/{media_id}/embeddings",
    response_model=GenerateEmbeddingsResponse,
    dependencies=[Depends(rbac_rate_limit("embeddings.create"))],
)
async def generate_embeddings(
    media_id: int,
    request: GenerateEmbeddingsRequest,
    db: Annotated[Any, Depends(get_media_db_for_user)],
    current_user: Annotated[User, Depends(get_request_user)],
) -> GenerateEmbeddingsResponse:
    """Generate embeddings for a media item"""

    embedding_model, embedding_provider = _resolve_model_provider(
        request.embedding_model,
        request.embedding_provider,
    )

    try:
        # Generate embeddings in per-user collection
        user_id = resolve_user_id_for_request(
            current_user,
            error_status=http_status.HTTP_400_BAD_REQUEST,
        )

        if _embeddings_jobs_backend() != "jobs":
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embeddings jobs backend is unavailable",
            )

        adapter = EmbeddingsJobsAdapter()
        media_item = get_media_by_id(db, media_id)
        if not media_item:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Media item {media_id} not found"
            )

        try:
            job_row = adapter.create_job(
                user_id=user_id,
                media_id=media_id,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                request_source="media",
                force_regenerate=request.force_regenerate,
                stage="chunking",
                embedding_priority=request.priority,
            )
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Failed to persist media embedding job")
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to queue embedding job",
            ) from e

        job_id = str((job_row or {}).get("uuid") or (job_row or {}).get("id") or "").strip()
        if not job_id:
            logger.error("Embeddings job creation returned no job id")
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to queue embedding job",
            )
        return GenerateEmbeddingsResponse(
            media_id=media_id,
            status="accepted",
            message="Embedding generation started",
            embedding_count=None,
            embedding_model=embedding_model,
            chunks_processed=None,
            job_id=job_id
        )

    except HTTPException:
        raise
    except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error generating embeddings")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating embeddings"
        ) from e


@router.post(
    "/embeddings/batch",
    response_model=BatchMediaEmbeddingsResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rbac_rate_limit("embeddings.create"))],
)
async def generate_embeddings_batch(
    request: BatchMediaEmbeddingsRequest,
    db: Annotated[Any, Depends(get_media_db_for_user)],
    current_user: Annotated[User, Depends(get_request_user)],
) -> BatchMediaEmbeddingsResponse:
    """Launch embedding jobs for multiple media items."""

    media_ids = list(dict.fromkeys(request.media_ids))
    if not media_ids:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="media_ids must not be empty")

    embedding_model, embedding_provider = _resolve_model_provider(
        request.embedding_model,
        request.embedding_provider,
    )

    user_id = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_400_BAD_REQUEST,
    )
    if _embeddings_jobs_backend() != "jobs":
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings jobs backend is unavailable",
        )

    adapter = EmbeddingsJobsAdapter()
    job_ids: list[str] = []
    failed_media_ids: list[int] = []
    failure_reasons: list[str] = []
    for media_id in media_ids:
        media_item = get_media_by_id(db, media_id)
        if not media_item:
            failed_media_ids.append(int(media_id))
            failure_reasons.append(f"media_id={media_id}: not found")
            continue
        job_id: Optional[str] = None
        try:
            job_row = adapter.create_job(
                user_id=user_id,
                media_id=media_id,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                request_source="media_batch",
                force_regenerate=request.force_regenerate,
                stage="chunking",
                embedding_priority=request.priority,
            )
            job_id = str((job_row or {}).get("uuid") or (job_row or {}).get("id") or "").strip()
            if not job_id:
                raise ValueError("create_job returned no uuid/id")
            job_ids.append(job_id)
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            failed_media_ids.append(int(media_id))
            failure_reasons.append(f"media_id={media_id}: {type(exc).__name__}")
            logger.error("Failed to persist batch embedding job")

    if failed_media_ids and not job_ids:
        detail = {
            "error": "batch_enqueue_failed",
            "message": "Failed to queue one or more embedding jobs",
            "submitted": 0,
            "failed_media_ids": failed_media_ids,
            "failure_reasons": failure_reasons,
        }
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )

    if failed_media_ids:
        return BatchMediaEmbeddingsResponse(
            status="partial",
            job_ids=job_ids,
            submitted=len(job_ids),
            failed_media_ids=failed_media_ids,
            failure_reasons=failure_reasons,
        )

    return BatchMediaEmbeddingsResponse(
        status="accepted",
        job_ids=job_ids,
        submitted=len(job_ids),
    )


@router.post(
    "/embeddings/search",
    response_model=EmbeddingsSearchResponse,
    status_code=http_status.HTTP_200_OK,
    dependencies=[Depends(rbac_rate_limit("embeddings.search"))],
)
async def search_embeddings(
    request: EmbeddingsSearchRequest,
    current_user: Annotated[User, Depends(get_request_user)],
) -> EmbeddingsSearchResponse:
    """Search stored embeddings using the provided query text."""

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="query must not be empty")

    user_id = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_400_BAD_REQUEST,
    )
    embedding_model, embedding_provider = _resolve_model_provider(
        request.embedding_model,
        request.embedding_provider,
    )

    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import create_embeddings_batch_async
        query_metadata = {"user_id": user_id}

        query_embeddings = await create_embeddings_batch_async(
            texts=[request.query],
            provider=embedding_provider,
            model_id=embedding_model,
            metadata=query_metadata,
        )
    except Exception as exc:
        logger.error("Failed to embed search query")
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        ) from exc

    if not query_embeddings or not query_embeddings[0]:
        return EmbeddingsSearchResponse(results=[], count=0)

    manager = ChromaDBManager(user_id=user_id, user_embedding_config=_user_embedding_config())
    collection_name = request.collection or f"user_{user_id}_media_embeddings"

    try:
        collection = manager.client.get_collection(name=collection_name)
    except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found",
        ) from None

    include = ["metadatas", "documents", "distances"]
    try:
        query_result = collection.query(
            query_embeddings=[query_embeddings[0]],
            n_results=request.top_k,
            include=include,
            where=request.filters if request.filters else None
        )
    except Exception as exc:
        logger.error("Chroma query failed")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        ) from exc

    ids = (query_result.get("ids") or [[]])[0]
    documents = (query_result.get("documents") or [[]])[0]
    metadatas = (query_result.get("metadatas") or [[]])[0]
    distances = (query_result.get("distances") or [[]])[0]

    results: list[EmbeddingsSearchResult] = []
    for idx, item_id in enumerate(ids):
        metadata_obj: dict[str, Any] = {}
        if idx < len(metadatas) and isinstance(metadatas[idx], dict):
            metadata_obj = metadatas[idx]
        document_text = documents[idx] if idx < len(documents) else None
        distance_val = distances[idx] if idx < len(distances) else None
        results.append(
            EmbeddingsSearchResult(
                id=item_id,
                document=document_text,
                metadata=metadata_obj,
                distance=distance_val
            )
        )

    return EmbeddingsSearchResponse(results=results, count=len(results))


@router.delete(
    "/{media_id}/embeddings",
    dependencies=[Depends(rbac_rate_limit("embeddings.delete"))],
)
async def delete_embeddings(
    media_id: int,
    db: Annotated[Any, Depends(get_media_db_for_user)],
    current_user: Annotated[User, Depends(get_request_user)],
) -> dict[str, Any]:
    """Delete embeddings for a media item"""
    try:
        # Check if media exists
        media_item = get_media_by_id(db, media_id)
        if not media_item:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Media item {media_id} not found"
            )

        # Delete embeddings from per-user collection using a where filter
        user_id = resolve_user_id_for_request(
            current_user,
            error_status=http_status.HTTP_400_BAD_REQUEST,
        )
        manager = ChromaDBManager(user_id=user_id, user_embedding_config=_user_embedding_config())
        collection_name = f"user_{user_id}_media_embeddings"
        collection = manager.get_or_create_collection(collection_name)
        try:
            # Use where-based delete if supported
            collection.delete(where={"media_id": str(media_id)})
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Fall back to fetching IDs then deleting by ids
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
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            logger.warning("Failed to verify embeddings delete")

        invalidate_rag_caches(current_user, media_id=media_id)

        return {
            "status": "success",
            "message": f"Embeddings deleted for media item {media_id}"
        }
    except HTTPException:
        raise
    except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error deleting embeddings")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting embeddings"
        ) from e


@router.get(
    "/embeddings/jobs/{job_id}",
    dependencies=[Depends(rbac_rate_limit("embeddings.jobs.get"))],
)
async def get_media_embedding_job(
    job_id: str,
    current_user: Annotated[User, Depends(get_request_user)],
):
    user_id = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_400_BAD_REQUEST,
    )
    adapter = EmbeddingsJobsAdapter()
    rec = adapter.get_job(job_id, user_id)
    if not rec:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Job not found")
    return rec


@router.get(
    "/embeddings/jobs",
    dependencies=[Depends(rbac_rate_limit("embeddings.jobs.list"))],
)
async def list_media_embedding_jobs(
    current_user: Annotated[User, Depends(get_request_user)],
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0, le=MAX_MEDIA_EMBEDDING_JOBS_OFFSET),
):
    user_id = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_400_BAD_REQUEST,
    )
    adapter = EmbeddingsJobsAdapter()
    fetched = adapter.list_jobs(user_id=user_id, status=status, limit=limit + 1, offset=offset)
    rows = fetched[:limit]
    has_more = len(fetched) > limit
    pagination = build_offset_pagination_meta(
        total=None,
        limit=limit,
        offset=offset,
        count=len(rows),
        has_more=has_more,
    ).model_dump(mode="json")
    pagination["count"] = len(rows)
    return {
        "data": rows,
        "pagination": pagination,
        "has_more": pagination["has_more"],
        "next_offset": pagination["next_offset"],
    }
