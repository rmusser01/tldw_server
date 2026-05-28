"""
Embeddings Jobs worker (legacy):

- Consumes core Jobs entries for legacy embeddings jobs.
- Stage handlers are reused by the Redis Streams worker.
- Root Jobs remain the status/billing record.

Job contract (domain/queue/job_type):
- domain = "embeddings"
- stage queue = os.getenv("EMBEDDINGS_JOBS_QUEUE", "default")
- job_types:
  - embeddings_chunking -> embeddings_embedding -> embeddings_storage (legacy stage jobs)
  - embeddings_pipeline (root job; not consumed by workers)
  - content_embeddings (legacy inline content)

Payload fields (media jobs):
- media_id: int (required)
- embedding_model: str
- embedding_provider: str
- chunk_size: int
- chunk_overlap: int
- request_source: str (optional)
- root_job_uuid: str (optional)
- parent_job_uuid: str (optional)
- config_version: str (optional)

Usage (legacy only):
  python -m tldw_Server_API.app.core.Embeddings.services.jobs_worker
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.endpoints.media_embeddings import (
    FALLBACK_EMBEDDING_MODEL,
    _allow_zero_embeddings_for_media,
    _resolve_model_provider,
    chunk_media_content,
    generate_embeddings_for_media,
)
from tldw_Server_API.app.api.v1.utils.rag_cache import invalidate_rag_caches
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    get_user_media_db_path,
)
from tldw_Server_API.app.core.DB_Management.DB_Manager import mark_media_as_processed
from tldw_Server_API.app.core.DB_Management.Kanban_DB import _kanban_card_indexable
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_by_id
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.testing import is_truthy

_EMBEDDINGS_DOMAIN = "embeddings"
_EMBEDDINGS_ROOT_JOB_TYPE = "embeddings_pipeline"
_EMBEDDINGS_CHUNKING_JOB_TYPE = "embeddings_chunking"
_EMBEDDINGS_EMBEDDING_JOB_TYPE = "embeddings_embedding"
_EMBEDDINGS_STORAGE_JOB_TYPE = "embeddings_storage"
_CONTENT_JOB_TYPE = "content_embeddings"
_STAGE_JOB_TYPES = {
    "chunking": _EMBEDDINGS_CHUNKING_JOB_TYPE,
    "embedding": _EMBEDDINGS_EMBEDDING_JOB_TYPE,
    "storage": _EMBEDDINGS_STORAGE_JOB_TYPE,
}
_EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


class EmbeddingsJobError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


def _jobs_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _get_user_id(job: dict[str, Any], payload: dict[str, Any]) -> str:
    owner = job.get("owner_user_id") or payload.get("user_id")
    if owner is None or str(owner).strip() == "":
        return str(DatabasePaths.get_single_user_id())
    return str(owner)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return is_truthy(str(value).strip().lower())


def _normalize_chunk_type(value: Any) -> str | None:
    """Normalize a chunk-type candidate while surfacing non-fatal normalization failures."""

    try:
        return Chunker.normalize_chunk_type(value)
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Failed to normalize chunk type candidate {!r}: {}", value, exc)
        return None


def _root_job_uuid(payload: dict[str, Any]) -> str | None:
    root = payload.get("root_job_uuid") or payload.get("parent_job_uuid")
    if root is None:
        return None
    return str(root)


def _resolve_chunk_type(*candidates: Any) -> str:
    """Return the first normalized chunk type from the candidate list, else fall back to ``text``."""

    for candidate in candidates:
        normalized = _normalize_chunk_type(candidate)
        if normalized:
            return normalized
    return "text"


def _update_root_job(
    root_uuid: str | None,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if not root_uuid:
        return
    jm = _jobs_manager()
    try:
        root = jm.get_job_by_uuid(str(root_uuid))
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS:
        return
    if not root:
        return
    root_id = int(root.get("id"))
    if status == "completed":
        jm.update_job_progress(root_id, progress_percent=100.0)
        jm.complete_job(root_id, result=result, enforce=False)
    elif status == "failed":
        message = error or "Embeddings stage failed"
        jm.fail_job(root_id, error=message, retryable=False, enforce=False)


def _should_track_media_state(job_type: str | None, payload: dict[str, Any]) -> bool:
    """Return whether this embeddings job should update Media DB readiness state."""

    if job_type in {
        _EMBEDDINGS_CHUNKING_JOB_TYPE,
        _EMBEDDINGS_EMBEDDING_JOB_TYPE,
        _EMBEDDINGS_STORAGE_JOB_TYPE,
    }:
        return True
    if job_type != _CONTENT_JOB_TYPE:
        return False
    return not (payload.get("collection_name") and payload.get("document_id"))


def _mark_media_embeddings_complete(*, user_id: str, media_id: int) -> None:
    """Mark a media item as vector-ready after embeddings storage completes."""

    db_path = get_user_media_db_path(user_id)
    with managed_media_database(
        client_id="embeddings_jobs_worker",
        db_path=db_path,
        initialize=False,
    ) as db:
        mark_media_as_processed(db_instance=db, media_id=int(media_id))


def _mark_media_embeddings_error(*, user_id: str, media_id: int, error_message: str) -> None:
    """Mark embeddings processing as failed for a media item."""

    db_path = get_user_media_db_path(user_id)
    with managed_media_database(
        client_id="embeddings_jobs_worker",
        db_path=db_path,
        initialize=False,
    ) as db:
        db.mark_embeddings_error(int(media_id), str(error_message))


def _mark_media_embeddings_error_safely(
    *, user_id: str, media_id: int, error_message: str
) -> None:
    """Best-effort failure-state marker for worker exception paths."""

    try:
        _mark_media_embeddings_error(
            user_id=user_id,
            media_id=media_id,
            error_message=error_message,
        )
    except Exception as exc:
        logger.warning(
            "Failed to mark media embeddings error state "
            f"(user_id={user_id}, media_id={media_id}): {exc}"
        )


def _load_media_content(media_id: int, user_id: str) -> dict[str, Any]:
    db_path = get_user_media_db_path(user_id)
    with managed_media_database(
        client_id="embeddings_jobs_worker",
        db_path=db_path,
        initialize=False,
    ) as db:
        media_item = get_media_by_id(db, media_id)
        if not media_item:
            raise EmbeddingsJobError(f"Media item {media_id} not found", retryable=False)

        try:
            if isinstance(media_item, dict) and not (media_item.get("content") or "").strip():
                from tldw_Server_API.app.core.DB_Management.media_db.api import (
                    get_document_version,
                )

                latest = get_document_version(db, media_id=media_id, version_number=None, include_content=True)
                if latest and latest.get("content"):
                    media_item = dict(media_item)
                    media_item["content"] = latest["content"]
        except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Failed to load fallback document content for media {media_id}: {exc}")

        if not media_item:
            raise EmbeddingsJobError(f"No content found for media item {media_id}", retryable=False)

        return {
            "media_item": media_item,
            "content": media_item,
        }


def _update_root_progress(
    root_uuid: str | None,
    *,
    progress_percent: float | None,
    progress_message: str | None = None,
) -> None:
    if not root_uuid:
        return
    jm = _jobs_manager()
    try:
        root = jm.get_job_by_uuid(str(root_uuid))
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS:
        return
    if not root:
        return
    root_id = int(root.get("id"))
    jm.update_job_progress(
        root_id,
        progress_percent=progress_percent,
        progress_message=progress_message,
    )


def _update_root_result(
    root_uuid: str | None,
    *,
    result: dict[str, Any],
) -> None:
    if not root_uuid:
        return
    jm = _jobs_manager()
    try:
        root = jm.get_job_by_uuid(str(root_uuid))
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS:
        return
    if not root:
        return
    root_id = int(root.get("id"))
    try:
        jm.update_job_result(root_id, result=result, merge=True)
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS:
        return


def _artifact_dir(
    user_id: str,
    root_uuid: str | None,
    media_id: int,
    job_uuid: str | None,
) -> Path:
    base_dir = DatabasePaths.get_user_vector_store_dir(user_id) / "embeddings_jobs"
    base_dir.mkdir(parents=True, exist_ok=True)
    name = root_uuid or job_uuid or f"media_{media_id}"
    path = base_dir / str(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chunk_artifact_path(base_dir: Path) -> Path:
    return base_dir / "chunks.json"


def _embedding_artifact_path(base_dir: Path) -> Path:
    return base_dir / "embeddings.json"


def _storage_artifact_path(base_dir: Path) -> Path:
    return base_dir / "storage.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_content_text(media_content: dict[str, Any]) -> str:
    content_payload = media_content.get("content")
    if isinstance(content_payload, dict):
        return content_payload.get("content") or content_payload.get("text") or ""
    if isinstance(content_payload, str):
        return content_payload
    return ""


def _config_version(
    embedding_model: str,
    embedding_provider: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> str:
    return ":".join(
        [
            str(embedding_model or "").strip(),
            str(embedding_provider or "").strip(),
            str(chunk_size or ""),
            str(chunk_overlap or ""),
        ]
    )


def _embedding_config_for_user() -> dict[str, Any]:
    cfg = settings.get("EMBEDDING_CONFIG", {}).copy()
    try:
        user_db_base_dir = str(DatabasePaths.get_user_db_base_dir())
    except Exception:
        user_db_base_dir = settings.get("USER_DB_BASE_DIR")
    cfg["USER_DB_BASE_DIR"] = user_db_base_dir
    return cfg


def _stage_idempotency_key(media_id: int, stage: str, version: str) -> str:
    return f"{media_id}:{stage}:{version}"


def _map_priority(embedding_priority: int | None) -> int:
    try:
        raw = int(embedding_priority) if embedding_priority is not None else 50
    except (TypeError, ValueError):
        raw = 50
    return max(1, min(10, int(raw / 10)))


def _normalize_embeddings(embeddings: Any) -> list[list[float]]:
    if not embeddings:
        return []
    if hasattr(embeddings[0], "tolist"):
        return [emb.tolist() for emb in embeddings]
    return embeddings


def _validate_embeddings_result(embeddings: Any, expected_count: int) -> str | None:
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


def _resolve_config_version(
    payload: dict[str, Any],
    embedding_model: str,
    embedding_provider: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    version = payload.get("config_version")
    if isinstance(version, str) and version.strip():
        return version
    return _config_version(embedding_model, embedding_provider, chunk_size, chunk_overlap)


def _enqueue_stage_job(
    *,
    job: dict[str, Any],
    payload: dict[str, Any],
    stage: str,
    media_id: int,
    user_id: str,
    embedding_model: str,
    embedding_provider: str | None,
    chunk_size: int,
    chunk_overlap: int,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    jm = _jobs_manager()
    stage_queue = str(job.get("queue") or "").strip() or (os.getenv("EMBEDDINGS_JOBS_QUEUE") or "default").strip() or "default"
    root_uuid = _root_job_uuid(payload)
    parent_uuid = str(job.get("uuid") or job.get("id"))
    stage_payload = dict(payload)
    if artifacts:
        stage_payload.update(artifacts)
    stage_payload["current_stage"] = stage
    stage_payload["root_job_uuid"] = root_uuid
    stage_payload["parent_job_uuid"] = parent_uuid
    stage_payload["embedding_model"] = embedding_model
    stage_payload["embedding_provider"] = embedding_provider

    idempotency_key = None
    if not _coerce_bool(payload.get("force_regenerate")):
        version = _resolve_config_version(payload, embedding_model, embedding_provider, chunk_size, chunk_overlap)
        stage_payload["config_version"] = version
        idempotency_key = _stage_idempotency_key(media_id, stage, version)

    priority = job.get("priority")
    if priority is None:
        priority = _map_priority(payload.get("embedding_priority"))

    return jm.create_job(
        domain=_EMBEDDINGS_DOMAIN,
        queue=stage_queue,
        job_type=_STAGE_JOB_TYPES[stage],
        payload=stage_payload,
        owner_user_id=str(user_id) if user_id is not None else None,
        idempotency_key=idempotency_key,
        priority=int(priority),
        max_retries=0,
        request_id=job.get("request_id"),
        trace_id=job.get("trace_id"),
    )


async def _handle_chunking_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    media_id: int,
    user_id: str,
    chunk_size: int,
    chunk_overlap: int,
    root_uuid: str | None,
) -> tuple[dict[str, Any], bool]:
    force_regenerate = _coerce_bool(payload.get("force_regenerate"))
    artifact_dir = _artifact_dir(user_id, root_uuid, media_id, str(job.get("uuid") or job.get("id")))
    chunks_path = Path(payload.get("chunks_path") or _chunk_artifact_path(artifact_dir))

    if chunks_path.exists() and not force_regenerate:
        chunks = _read_json(chunks_path)
        count = len(chunks) if isinstance(chunks, list) else 0
        _update_root_result(root_uuid, result={"total_chunks": count})
        _update_root_progress(root_uuid, progress_percent=33.0, progress_message="chunking completed")
        return {
            "chunks_path": str(chunks_path),
            "total_chunks": count,
            "chunks_processed": count,
            "idempotent": True,
        }, False

    media_content = _load_media_content(media_id, user_id)
    allow_zero = _allow_zero_embeddings_for_media(media_content.get("media_item", {}))
    content_text = _extract_content_text(media_content)
    if not content_text or not content_text.strip():
        if allow_zero:
            _write_json(chunks_path, [])
            _update_root_result(root_uuid, result={"total_chunks": 0})
            _update_root_progress(root_uuid, progress_percent=33.0, progress_message="chunking completed")
            return {
                "chunks_path": str(chunks_path),
                "total_chunks": 0,
                "chunks_processed": 0,
                "allow_zero_embeddings": True,
                "skip_pipeline": True,
            }, True
        raise EmbeddingsJobError("No text content to generate embeddings from", retryable=False)

    chunks = chunk_media_content(content_text, chunk_size, chunk_overlap)
    if not chunks:
        if allow_zero:
            _write_json(chunks_path, [])
            _update_root_result(root_uuid, result={"total_chunks": 0})
            _update_root_progress(root_uuid, progress_percent=33.0, progress_message="chunking completed")
            return {
                "chunks_path": str(chunks_path),
                "total_chunks": 0,
                "chunks_processed": 0,
                "allow_zero_embeddings": True,
                "skip_pipeline": True,
            }, True
        raise EmbeddingsJobError("No chunks produced from media content", retryable=False)

    _write_json(chunks_path, chunks)
    _update_root_result(root_uuid, result={"total_chunks": len(chunks)})
    _update_root_progress(root_uuid, progress_percent=33.0, progress_message="chunking completed")
    return {
        "chunks_path": str(chunks_path),
        "total_chunks": len(chunks),
        "chunks_processed": len(chunks),
    }, False


async def _handle_embedding_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    media_id: int,
    user_id: str,
    embedding_model: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
    root_uuid: str | None,
) -> dict[str, Any]:
    force_regenerate = _coerce_bool(payload.get("force_regenerate"))
    artifact_dir = _artifact_dir(user_id, root_uuid, media_id, str(job.get("uuid") or job.get("id")))
    chunks_path = Path(payload.get("chunks_path") or _chunk_artifact_path(artifact_dir))
    if not chunks_path.exists():
        raise EmbeddingsJobError("Chunk artifacts missing for embedding stage", retryable=False)

    embeddings_path = Path(payload.get("embeddings_path") or _embedding_artifact_path(artifact_dir))
    if embeddings_path.exists() and not force_regenerate:
        stored = _read_json(embeddings_path)
        if isinstance(stored, dict):
            embeddings = stored.get("embeddings") or []
            embedding_model = stored.get("embedding_model") or embedding_model
            embedding_provider = stored.get("embedding_provider") or embedding_provider
            embedding_count = stored.get("embedding_count")
        else:
            embeddings = stored
            embedding_count = None
        if embedding_count is None:
            embedding_count = len(embeddings) if embeddings else 0
        return {
            "embeddings_path": str(embeddings_path),
            "chunks_path": str(chunks_path),
            "embedding_count": embedding_count,
            "total_chunks": embedding_count,
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "idempotent": True,
        }

    chunks = _read_json(chunks_path)
    if not isinstance(chunks, list):
        raise EmbeddingsJobError("Chunk payload invalid for embedding stage", retryable=False)
    chunk_texts = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise EmbeddingsJobError("Chunk payload invalid for embedding stage", retryable=False)
        text = chunk.get("text")
        if text is None:
            raise EmbeddingsJobError("Chunk payload missing text for embedding stage", retryable=False)
        chunk_texts.append(text)
    if not chunk_texts:
        raise EmbeddingsJobError("No chunk text available to embed", retryable=False)

    request_metadata = {"user_id": str(user_id)}
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            create_embeddings_batch_async,
        )

        embeddings = await create_embeddings_batch_async(
            texts=chunk_texts,
            provider=embedding_provider,
            model_id=embedding_model,
            metadata=request_metadata,
        )
        validation_error = _validate_embeddings_result(embeddings, len(chunk_texts))
        if validation_error:
            raise EmbeddingsJobError(validation_error, retryable=False)
    except EmbeddingsJobError:
        raise
    except Exception as exc:
        if embedding_model != FALLBACK_EMBEDDING_MODEL:
            logger.warning(f"Failed with {embedding_model}, trying fallback {FALLBACK_EMBEDDING_MODEL}")
            try:
                embeddings = await create_embeddings_batch_async(
                    texts=chunk_texts,
                    provider="huggingface",
                    model_id=FALLBACK_EMBEDDING_MODEL,
                    metadata=request_metadata,
                )
            except Exception as fallback_exc:
                raise EmbeddingsJobError(str(fallback_exc), retryable=True) from fallback_exc
            validation_error = _validate_embeddings_result(embeddings, len(chunk_texts))
            if validation_error:
                raise EmbeddingsJobError(validation_error, retryable=False) from exc
            embedding_model = FALLBACK_EMBEDDING_MODEL
            embedding_provider = "huggingface"
        else:
            raise EmbeddingsJobError(str(exc), retryable=True) from exc

    embeddings_list = _normalize_embeddings(embeddings)
    payload_to_store = {
        "embeddings": embeddings_list,
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
        "embedding_count": len(embeddings_list),
    }
    _write_json(embeddings_path, payload_to_store)
    return {
        "embeddings_path": str(embeddings_path),
        "chunks_path": str(chunks_path),
        "embedding_count": len(embeddings_list),
        "total_chunks": len(chunk_texts),
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
    }


async def _handle_storage_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    media_id: int,
    user_id: str,
    embedding_model: str,
    embedding_provider: str,
    root_uuid: str | None,
) -> dict[str, Any]:
    force_regenerate = _coerce_bool(payload.get("force_regenerate"))
    artifact_dir = _artifact_dir(user_id, root_uuid, media_id, str(job.get("uuid") or job.get("id")))
    chunks_path = Path(payload.get("chunks_path") or _chunk_artifact_path(artifact_dir))
    embeddings_path = Path(payload.get("embeddings_path") or _embedding_artifact_path(artifact_dir))
    storage_path = Path(payload.get("storage_path") or _storage_artifact_path(artifact_dir))

    if storage_path.exists() and not force_regenerate:
        stored = _read_json(storage_path)
        if isinstance(stored, dict):
            return stored
        raise EmbeddingsJobError(
            "Storage artifact invalid for idempotent reuse",
            retryable=False,
        )

    if not chunks_path.exists() or not embeddings_path.exists():
        raise EmbeddingsJobError("Embeddings artifacts missing for storage stage", retryable=False)

    chunks = _read_json(chunks_path)
    if not isinstance(chunks, list):
        raise EmbeddingsJobError("Chunk payload invalid for storage stage", retryable=False)
    stored_embeddings = _read_json(embeddings_path)
    if isinstance(stored_embeddings, dict):
        embeddings = stored_embeddings.get("embeddings") or []
        embedding_model = stored_embeddings.get("embedding_model") or embedding_model
        embedding_provider = stored_embeddings.get("embedding_provider") or embedding_provider
    else:
        embeddings = stored_embeddings
    if not isinstance(embeddings, list):
        raise EmbeddingsJobError("Embeddings payload invalid for storage stage", retryable=False)

    if len(embeddings) != len(chunks):
        raise EmbeddingsJobError(
            f"Embeddings count {len(embeddings)} does not match chunks {len(chunks)}",
            retryable=False,
        )

    chunk_texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    media_content = _load_media_content(media_id, user_id)
    extra_metadata: dict[str, Any] = {}
    try:
        media_item_meta = media_content.get("media_item", {})
        if isinstance(media_item_meta, dict):
            extra_metadata = media_item_meta.get("metadata") or {}
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS:
        extra_metadata = {}

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise EmbeddingsJobError("Chunk payload invalid for storage stage", retryable=False)
        text = chunk.get("text")
        if text is None:
            raise EmbeddingsJobError("Chunk payload missing text for storage stage", retryable=False)
        chunk_texts.append(text)
        chunk_metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        metadata = {
            "media_id": str(media_id),
            "chunk_index": chunk.get("index", idx),
            "chunk_start": chunk.get("start"),
            "chunk_end": chunk.get("end"),
            "chunk_type": _resolve_chunk_type(
                chunk.get("chunk_type"),
                chunk_metadata.get("chunk_type"),
                chunk_metadata.get("paragraph_kind"),
                chunk_metadata.get("type"),
                chunk_metadata.get("kind"),
            ),
            "title": media_content["media_item"].get("title", ""),
            "author": media_content["media_item"].get("author", ""),
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
        }
        if isinstance(extra_metadata, dict) and extra_metadata:
            metadata["extra"] = dict(extra_metadata)
        metadatas.append(metadata)

    ids = [f"media_{media_id}_chunk_{i}" for i in range(len(chunks))]
    manager = ChromaDBManager(
        user_id=str(user_id),
        user_embedding_config=_embedding_config_for_user(),
    )
    manager.store_in_chroma(
        collection_name=f"user_{user_id}_media_embeddings",
        texts=chunk_texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
        embedding_model_id_for_dim_check=embedding_model,
    )

    try:
        invalidate_rag_caches(None, namespaces=[user_id], media_id=int(media_id))
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            f"Failed to invalidate RAG caches after embeddings storage "
            f"(user_id={user_id}, media_id={media_id}): {exc}"
        )

    result = {
        "embedding_count": len(embeddings),
        "chunks_processed": len(chunks),
        "total_chunks": len(chunks),
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
    }
    _write_json(storage_path, result)
    return result


async def _handle_content_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    media_id: int,
    user_id: str,
    embedding_model: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
    root_uuid: str | None,
) -> dict[str, Any]:
    collection_name = payload.get("collection_name")
    document_id = payload.get("document_id")
    has_collection_name = bool(collection_name)
    has_document_id = bool(document_id)
    if has_collection_name != has_document_id:
        missing = "collection_name" if not has_collection_name else "document_id"
        raise EmbeddingsJobError(
            f"Custom content job requires both collection_name and document_id; missing {missing}",
            retryable=False,
        )
    if has_collection_name and has_document_id:
        return await _handle_custom_content_job(
            job=job,
            payload=payload,
            media_id=media_id,
            user_id=user_id,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            root_uuid=root_uuid,
        )
    raw_content = payload.get("content") or payload.get("text")
    if not raw_content or not str(raw_content).strip():
        raise EmbeddingsJobError("Missing content for content_embeddings job", retryable=False)
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    title = payload.get("title") or meta.get("title") or ""
    author = payload.get("author") or meta.get("author") or ""
    media_content = {
        "media_item": {
            "title": title,
            "author": author,
            "metadata": meta,
        },
        "content": {"content": str(raw_content)},
    }

    result = await generate_embeddings_for_media(
        media_id=int(media_id),
        media_content=media_content,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        user_id=user_id,
    )

    allow_zero = bool(result.get("allow_zero_embeddings"))
    if result.get("status") != "success" and not allow_zero:
        error = result.get("error") or result.get("message") or "Embedding generation failed"
        retryable = _coerce_bool(result.get("retryable"))
        backoff_seconds = result.get("backoff_seconds")
        if backoff_seconds is not None:
            backoff_seconds = _coerce_int(backoff_seconds, 0)
            if backoff_seconds <= 0:
                backoff_seconds = None
        raise EmbeddingsJobError(str(error), retryable=retryable, backoff_seconds=backoff_seconds)

    try:
        invalidate_rag_caches(None, namespaces=[user_id], media_id=int(media_id))
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            f"Failed to invalidate RAG caches after content embeddings "
            f"(user_id={user_id}, media_id={media_id}): {exc}"
        )

    payload_result = {
        "embedding_count": result.get("embedding_count"),
        "chunks_processed": result.get("chunks_processed"),
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
    }
    return payload_result


async def _handle_custom_content_job(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    media_id: int,
    user_id: str,
    embedding_model: str,
    embedding_provider: str,
    root_uuid: str | None,
) -> dict[str, Any]:
    """Generate embeddings for a single custom content payload and store in Chroma."""
    del job, media_id  # kept for signature compatibility; intentionally unused
    raw_content = payload.get("content") or payload.get("text")
    if not raw_content or not str(raw_content).strip():
        raise EmbeddingsJobError("Missing content for content_embeddings job", retryable=False)
    collection_name = payload.get("collection_name")
    document_id = payload.get("document_id")
    if not collection_name or not document_id:
        raise EmbeddingsJobError(
            "Custom content job missing collection_name or document_id",
            retryable=False,
        )

    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        meta = {}

    if payload.get("request_source") == "kanban":
        card_id = payload.get("card_id") or meta.get("card_id")
        expected_version = payload.get("card_version")
        int_card_id = None
        if card_id is not None:
            try:
                int_card_id = int(card_id)
            except (TypeError, ValueError):
                logger.warning(
                    "Kanban content job has non-numeric card_id; skipping indexability check "
                    f"(user_id={user_id}, card_id={card_id})"
                )
        if card_id is not None and int_card_id is None:
            result = {
                "embedding_count": 0,
                "chunks_processed": 0,
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
                "skipped": True,
                "skip_reason": "card_not_indexable",
            }
            _update_root_job(root_uuid, status="completed", result=result)
            return result
        if int_card_id is not None and not _kanban_card_indexable(
            user_id=str(user_id),
            card_id=int_card_id,
            expected_version=expected_version,
        ):
            result = {
                "embedding_count": 0,
                "chunks_processed": 0,
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
                "skipped": True,
                "skip_reason": "card_not_indexable",
            }
            _update_root_job(root_uuid, status="completed", result=result)
            return result

    request_metadata = {"user_id": str(user_id)}
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            create_embeddings_batch_async,
        )

        embeddings = await create_embeddings_batch_async(
            texts=[str(raw_content)],
            provider=embedding_provider,
            model_id=embedding_model,
            metadata=request_metadata,
        )
        validation_error = _validate_embeddings_result(embeddings, 1)
        if validation_error:
            raise EmbeddingsJobError(validation_error, retryable=False)
    except EmbeddingsJobError:
        raise
    except Exception as exc:
        if embedding_model != FALLBACK_EMBEDDING_MODEL:
            logger.warning(f"Failed with {embedding_model}, trying fallback {FALLBACK_EMBEDDING_MODEL}")
            try:
                embeddings = await create_embeddings_batch_async(
                    texts=[str(raw_content)],
                    provider="huggingface",
                    model_id=FALLBACK_EMBEDDING_MODEL,
                    metadata=request_metadata,
                )
            except Exception as fallback_exc:
                raise EmbeddingsJobError(str(fallback_exc), retryable=True) from fallback_exc
            validation_error = _validate_embeddings_result(embeddings, 1)
            if validation_error:
                raise EmbeddingsJobError(validation_error, retryable=False) from exc
            embedding_model = FALLBACK_EMBEDDING_MODEL
            embedding_provider = "huggingface"
        else:
            raise EmbeddingsJobError(str(exc), retryable=True) from exc

    embeddings_list = _normalize_embeddings(embeddings)
    metadata = dict(meta)
    metadata["embedding_model"] = embedding_model
    metadata["embedding_provider"] = embedding_provider

    manager = ChromaDBManager(
        user_id=str(user_id),
        user_embedding_config=_embedding_config_for_user(),
    )
    manager.store_in_chroma(
        collection_name=str(collection_name),
        texts=[str(raw_content)],
        embeddings=embeddings_list,
        ids=[str(document_id)],
        metadatas=[metadata],
        embedding_model_id_for_dim_check=embedding_model,
    )

    try:
        invalidate_rag_caches(None, namespaces=[user_id])
    except _EMBEDDINGS_JOB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to invalidate RAG caches for custom content (user={user_id}): {exc}")

    result = {
        "embedding_count": len(embeddings_list),
        "chunks_processed": len(embeddings_list),
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
    }
    return result


async def _handle_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    job_type = job.get("job_type")
    if job_type not in {
        _CONTENT_JOB_TYPE,
        _EMBEDDINGS_CHUNKING_JOB_TYPE,
        _EMBEDDINGS_EMBEDDING_JOB_TYPE,
        _EMBEDDINGS_STORAGE_JOB_TYPE,
    }:
        raise EmbeddingsJobError(
            f"Unsupported embeddings job type: {job_type}",
            retryable=False,
        )

    media_id = payload.get("media_id")
    if media_id is None and job_type == _CONTENT_JOB_TYPE:
        media_id = payload.get("item_id") or payload.get("content_id")
    if media_id is None:
        raise EmbeddingsJobError("Missing media_id in job payload", retryable=False)
    media_id = int(media_id)

    user_id = _get_user_id(job, payload)
    root_uuid = _root_job_uuid(payload)

    embedding_model = payload.get("embedding_model")
    embedding_provider = payload.get("embedding_provider")
    embedding_model, embedding_provider = _resolve_model_provider(embedding_model, embedding_provider)
    payload["embedding_model"] = embedding_model
    payload["embedding_provider"] = embedding_provider

    chunk_size = _coerce_int(payload.get("chunk_size"), 1000)
    chunk_overlap = _coerce_int(payload.get("chunk_overlap"), 200)

    try:
        if job_type == _CONTENT_JOB_TYPE:
            result = await _handle_content_job(
                job,
                payload,
                media_id=media_id,
                user_id=user_id,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                root_uuid=root_uuid,
            )
            _update_root_job(root_uuid, status="completed", result=result)
            if _should_track_media_state(job_type, payload):
                _mark_media_embeddings_complete(user_id=user_id, media_id=media_id)
            return result

        if job_type == _EMBEDDINGS_CHUNKING_JOB_TYPE:
            result, skip = await _handle_chunking_job(
                job,
                payload,
                media_id=media_id,
                user_id=user_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                root_uuid=root_uuid,
            )
            if skip:
                payload_result = {
                    "embedding_count": 0,
                    "chunks_processed": 0,
                    "embedding_model": embedding_model,
                    "embedding_provider": embedding_provider,
                    "total_chunks": 0,
                }
                _update_root_job(root_uuid, status="completed", result=payload_result)
                if _should_track_media_state(job_type, payload):
                    _mark_media_embeddings_complete(user_id=user_id, media_id=media_id)
                return result
            _enqueue_stage_job(
                job=job,
                payload=payload,
                stage="embedding",
                media_id=media_id,
                user_id=user_id,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                artifacts=result,
            )
            return result

        if job_type == _EMBEDDINGS_EMBEDDING_JOB_TYPE:
            result = await _handle_embedding_job(
                job,
                payload,
                media_id=media_id,
                user_id=user_id,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                root_uuid=root_uuid,
            )
            _update_root_result(
                root_uuid,
                result={
                    "total_chunks": result.get("total_chunks"),
                    "embedding_count": result.get("embedding_count"),
                },
            )
            _update_root_progress(root_uuid, progress_percent=66.0, progress_message="embedding completed")
            _enqueue_stage_job(
                job=job,
                payload=payload,
                stage="storage",
                media_id=media_id,
                user_id=user_id,
                embedding_model=result.get("embedding_model") or embedding_model,
                embedding_provider=result.get("embedding_provider") or embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                artifacts=result,
            )
            return result

        if job_type == _EMBEDDINGS_STORAGE_JOB_TYPE:
            result = await _handle_storage_job(
                job,
                payload,
                media_id=media_id,
                user_id=user_id,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                root_uuid=root_uuid,
            )
            _update_root_job(root_uuid, status="completed", result=result)
            if _should_track_media_state(job_type, payload):
                _mark_media_embeddings_complete(user_id=user_id, media_id=media_id)
            return result

        raise EmbeddingsJobError(
            f"Unsupported embeddings stage job type: {job_type}",
            retryable=False,
        )
    except EmbeddingsJobError as exc:
        if not getattr(exc, "retryable", False):
            _update_root_job(root_uuid, status="failed", error=str(exc))
            if _should_track_media_state(job_type, payload):
                _mark_media_embeddings_error_safely(
                    user_id=user_id,
                    media_id=media_id,
                    error_message=str(exc),
                )
        raise
    except Exception as exc:
        _update_root_job(root_uuid, status="failed", error=str(exc))
        if _should_track_media_state(job_type, payload):
            _mark_media_embeddings_error_safely(
                user_id=user_id,
                media_id=media_id,
                error_message=str(exc),
            )
        raise


async def main() -> None:
    """Run the legacy embeddings jobs worker loop."""
    worker_id = (os.getenv("EMBEDDINGS_JOBS_WORKER_ID") or f"embeddings-jobs-{os.getpid()}").strip()
    queue = (os.getenv("EMBEDDINGS_JOBS_QUEUE") or "default").strip() or "default"

    cfg = WorkerConfig(
        domain=_EMBEDDINGS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=_coerce_int(os.getenv("EMBEDDINGS_JOBS_LEASE_SECONDS"), 60),
        renew_jitter_seconds=_coerce_int(os.getenv("EMBEDDINGS_JOBS_RENEW_JITTER_SECONDS"), 5),
        renew_threshold_seconds=_coerce_int(os.getenv("EMBEDDINGS_JOBS_RENEW_THRESHOLD_SECONDS"), 10),
        backoff_base_seconds=_coerce_int(os.getenv("EMBEDDINGS_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=_coerce_int(os.getenv("EMBEDDINGS_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("EMBEDDINGS_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )

    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    logger.info(f"Embeddings Jobs worker starting (queue={queue}, worker_id={worker_id})")
    await sdk.run(handler=_handle_job)


if __name__ == "__main__":
    asyncio.run(main())
