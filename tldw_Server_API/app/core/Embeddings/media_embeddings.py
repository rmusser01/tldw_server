# media_embeddings.py
# Description: Media-embedding helpers shared by the media-embeddings endpoint, the embeddings jobs
# worker, ingestion persistence and the media-ingest worker. Moved verbatim from
# api/v1/endpoints/media_embeddings.py (TASK-13950.20).

import json
import os
from typing import Any, Optional

from fastapi import HTTPException
from fastapi import status as http_status
from loguru import logger

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_by_id
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

# Default embedding settings
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_PROVIDER = "huggingface"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
