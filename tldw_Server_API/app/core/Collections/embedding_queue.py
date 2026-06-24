from __future__ import annotations

import contextlib
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Embeddings import redis_pipeline
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Collections.utils import truncate_text_hard


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid integer for {}: {!r}; using default={}", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("Out-of-range integer for {}: {}; minimum={}; using default={}", name, value, minimum, default)
        return default
    return value


EMBEDDING_JOB_CONTENT_MAX_CHARS = _env_int("EMBEDDING_JOB_CONTENT_MAX_CHARS", 200_000, minimum=0)


def _jobs_queue() -> str:
    queue = (os.getenv("EMBEDDINGS_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def _root_jobs_queue(stage_queue: str) -> str:
    root_queue = (os.getenv("EMBEDDINGS_ROOT_JOBS_QUEUE") or "").strip()
    if root_queue:
        return root_queue
    return "low" if stage_queue != "low" else "default"


def _jobs_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _map_priority(priority: int) -> int:
    # Map 0-100 legacy priority to 1-10 Jobs priority.
    try:
        val = int(priority)
    except (TypeError, ValueError):
        val = 50
    mapped = max(1, min(10, int(val / 10)))
    return mapped or 5


async def enqueue_embeddings_job_for_item(
    *,
    user_id: int | str,
    item_id: int,
    content: str,
    metadata: dict[str, Any] | None = None,
    priority: int = 50,
) -> None:
    """Best-effort queueing of an embedding job for a collections item."""
    if is_test_mode():
        return

    text = (content or "").strip()
    if not text:
        return
    bounded_text, content_truncated, original_len = truncate_text_hard(text, EMBEDDING_JOB_CONTENT_MAX_CHARS)
    if not bounded_text:
        return

    try:
        jm = _jobs_manager()
        stage_queue = _jobs_queue()
        root_queue = _root_jobs_queue(stage_queue)
        payload_metadata = dict(metadata or {})
        if content_truncated:
            payload_metadata["content_truncated"] = True
            payload_metadata["content_char_count"] = original_len
            payload_metadata["content_max_chars"] = EMBEDDING_JOB_CONTENT_MAX_CHARS
        payload = {
            "item_id": int(item_id),
            "content": bounded_text,
            "metadata": payload_metadata,
            "current_stage": "content",
            "request_source": "collections",
        }
        root_job = jm.create_job(
            domain="embeddings",
            queue=root_queue,
            job_type="embeddings_pipeline",
            payload=payload,
            owner_user_id=str(user_id),
            priority=_map_priority(priority),
            max_retries=0,
        )
        stage_payload = dict(payload)
        stage_payload["root_job_uuid"] = root_job.get("uuid")
        stage_payload["parent_job_uuid"] = root_job.get("uuid")
        stage_payload["user_id"] = str(user_id)

        try:
            redis_pipeline.enqueue_content_job(
                payload=stage_payload,
                root_job_uuid=str(root_job.get("uuid") or ""),
                force_regenerate=False,
                require_redis=not redis_pipeline.allow_stub(),
            )
        except Exception:
            with contextlib.suppress(Exception):
                jm.fail_job(
                    int(root_job["id"]),
                    error="Failed to enqueue content embeddings job to Redis",
                    retryable=False,
                    enforce=False,
                )
            raise
    except Exception as exc:
        logger.debug(f"Embedding job enqueue failed for item {item_id}: {exc}")
