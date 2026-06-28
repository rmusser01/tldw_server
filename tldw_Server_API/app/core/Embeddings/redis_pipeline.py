"""Embeddings Redis Streams pipeline helpers."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Infrastructure.redis_factory import create_sync_redis_client
from tldw_Server_API.app.core.testing import is_truthy

_STAGE_STREAM_DEFAULTS = {
    "chunking": "embeddings:chunking",
    "embedding": "embeddings:embedding",
    "storage": "embeddings:storage",
    "content": "embeddings:content",
}
_STAGE_GROUP_DEFAULTS = {
    "chunking": "chunking-workers",
    "embedding": "embedding-workers",
    "storage": "storage-workers",
    "content": "content-workers",
}


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return is_truthy(str(raw).strip().lower())


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    try:
        return int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _sanitize_stream_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (bytes, str, int, float)):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=True)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _sanitize_stream_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        clean_value = _sanitize_stream_value(value)
        if clean_value is None:
            continue
        sanitized[str(key)] = clean_value
    return sanitized


def _stage_key(stage: str) -> str:
    stage_norm = str(stage or "").strip().lower()
    if stage_norm not in _STAGE_STREAM_DEFAULTS:
        raise ValueError(f"Unsupported embeddings stage: {stage}")
    return stage_norm


def stream_name(stage: str) -> str:
    stage_norm = _stage_key(stage)
    env_key = f"EMBEDDINGS_REDIS_STREAM_{stage_norm.upper()}"
    return (os.getenv(env_key) or _STAGE_STREAM_DEFAULTS[stage_norm]).strip()


def group_name(stage: str) -> str:
    stage_norm = _stage_key(stage)
    env_key = f"EMBEDDINGS_REDIS_GROUP_{stage_norm.upper()}"
    return (os.getenv(env_key) or _STAGE_GROUP_DEFAULTS[stage_norm]).strip()


def dlq_prefix() -> str:
    return (os.getenv("EMBEDDINGS_REDIS_DLQ_PREFIX") or "embeddings:dlq").strip()


def allow_stub() -> bool:
    return _env_bool("EMBEDDINGS_REDIS_ALLOW_STUB", False)


def idempotency_ttl_seconds() -> int:
    return _env_int("EMBEDDINGS_REDIS_IDEMPOTENCY_TTL", 86400)


@dataclass(frozen=True)
class RedisEmbeddingsQueues:
    streams: dict[str, str]
    groups: dict[str, str]
    dlq_prefix: str


class RedisEnqueueError(RuntimeError):
    """Raised when an embeddings Redis enqueue operation cannot be completed."""


def load_queues() -> RedisEmbeddingsQueues:
    return RedisEmbeddingsQueues(
        streams={stage: stream_name(stage) for stage in _STAGE_STREAM_DEFAULTS},
        groups={stage: group_name(stage) for stage in _STAGE_GROUP_DEFAULTS},
        dlq_prefix=dlq_prefix(),
    )


def enqueue_stage(
    *,
    stage: str,
    payload: dict[str, Any],
    redis_client: Any | None = None,
    require_redis: bool = True,
) -> str | None:
    """Enqueue a stage payload into Redis Streams."""
    stage_norm = _stage_key(stage)
    client = redis_client
    created_client = False
    if client is None:
        client = create_sync_redis_client(
            context="embeddings-redis-enqueue",
            fallback_to_fake=not require_redis and allow_stub(),
        )
        created_client = True

    stream = stream_name(stage_norm)
    try:
        return client.xadd(stream, _sanitize_stream_payload(payload))
    finally:
        if created_client:
            with contextlib.suppress(Exception):
                client.close()


def enqueue_chunking_job(
    *,
    payload: dict[str, Any],
    root_job_uuid: str | None,
    force_regenerate: bool,
    redis_client: Any | None = None,
    require_redis: bool = True,
) -> str | None:
    """Enqueue the chunking stage with idempotency guard."""
    if not root_job_uuid:
        raise ValueError("root_job_uuid is required for embeddings chunking enqueue")

    client = redis_client
    created_client = False
    if client is None:
        client = create_sync_redis_client(
            context="embeddings-redis-enqueue",
            fallback_to_fake=not require_redis and allow_stub(),
        )
        created_client = True

    stream = stream_name("chunking")
    ttl = idempotency_ttl_seconds()
    idempotency_key = f"embeddings:root:{root_job_uuid}:enqueued"
    try:
        if not force_regenerate:
            try:
                try:
                    inserted = client.set(idempotency_key, "1", nx=True, ex=ttl)
                except TypeError:
                    if client.get(idempotency_key) is not None:
                        inserted = False
                    else:
                        client.set(idempotency_key, "1", ex=ttl)
                        inserted = True
                if not inserted:
                    return None
            except Exception as exc:
                logger.warning(f"Failed to set idempotency key for embeddings root {root_job_uuid}: {exc}")
                raise RedisEnqueueError(
                    f"Failed to set Redis idempotency key for embeddings root {root_job_uuid}"
                ) from exc
        return client.xadd(stream, _sanitize_stream_payload(payload))
    finally:
        if created_client:
            with contextlib.suppress(Exception):
                client.close()


def enqueue_content_job(
    *,
    payload: dict[str, Any],
    root_job_uuid: str | None,
    force_regenerate: bool,
    redis_client: Any | None = None,
    require_redis: bool = True,
) -> str | None:
    """Enqueue the content stage with idempotency guard."""
    if not root_job_uuid:
        raise ValueError("root_job_uuid is required for embeddings content enqueue")

    client = redis_client
    created_client = False
    if client is None:
        client = create_sync_redis_client(
            context="embeddings-redis-enqueue",
            fallback_to_fake=not require_redis and allow_stub(),
        )
        created_client = True

    stream = stream_name("content")
    ttl = idempotency_ttl_seconds()
    idempotency_key = f"embeddings:root:{root_job_uuid}:content_enqueued"
    try:
        if not force_regenerate:
            try:
                try:
                    inserted = client.set(idempotency_key, "1", nx=True, ex=ttl)
                except TypeError:
                    if client.get(idempotency_key) is not None:
                        inserted = False
                    else:
                        client.set(idempotency_key, "1", ex=ttl)
                        inserted = True
                if not inserted:
                    return None
            except Exception as exc:
                logger.warning(f"Failed to set content idempotency key for embeddings root {root_job_uuid}: {exc}")
                raise RedisEnqueueError(
                    f"Failed to set Redis idempotency key for embeddings root {root_job_uuid}"
                ) from exc
        return client.xadd(stream, _sanitize_stream_payload(payload))
    finally:
        if created_client:
            with contextlib.suppress(Exception):
                client.close()


__all__ = [
    "RedisEmbeddingsQueues",
    "allow_stub",
    "dlq_prefix",
    "enqueue_chunking_job",
    "enqueue_content_job",
    "enqueue_stage",
    "group_name",
    "load_queues",
    "stream_name",
]
