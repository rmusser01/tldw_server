"""Embeddings Redis Streams worker (minimal pipeline)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Embeddings import redis_pipeline
from tldw_Server_API.app.core.Embeddings.services import jobs_worker
from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client

_STAGE_JOB_TYPES = {
    "chunking": "embeddings_chunking",
    "embedding": "embeddings_embedding",
    "storage": "embeddings_storage",
    "content": "content_embeddings",
}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    try:
        return int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _retry_limits() -> dict[str, int]:
    return {
        "max_retries": _env_int("EMBEDDINGS_REDIS_MAX_RETRIES", 2),
        "base_backoff": _env_int("EMBEDDINGS_REDIS_RETRY_BACKOFF_BASE", 2),
        "max_backoff": _env_int("EMBEDDINGS_REDIS_RETRY_BACKOFF_MAX", 30),
    }


async def _ensure_group(client: Any, *, stream: str, group: str) -> None:
    try:
        try:
            await client.xgroup_create(stream, group, id="0", mkstream=True)
        except TypeError:
            await client.xgroup_create(stream, group, id="0")
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def _normalize_payload(fields: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in (fields or {}).items():
        payload[str(key)] = value
    return payload


async def _send_dlq(
    client: Any,
    *,
    stage: str,
    payload: dict[str, Any],
    error: str,
) -> None:
    dlq_stream = f"{redis_pipeline.dlq_prefix()}:{stage}"
    dlq_payload = dict(payload)
    dlq_payload["error"] = error
    dlq_payload["failed_stage"] = stage
    dlq_payload["failed_at"] = int(time.time())
    try:
        await client.xadd(dlq_stream, dlq_payload)
    except Exception as exc:
        logger.warning(f"Failed to enqueue embeddings DLQ entry: {exc}")


async def _handle_stage_message(
    *,
    stage: str,
    message_id: str,
    fields: dict[str, Any],
    client: Any,
    streams: redis_pipeline.RedisEmbeddingsQueues,
) -> None:
    payload = _normalize_payload(fields)
    payload["current_stage"] = stage

    root_uuid = payload.get("root_job_uuid") or payload.get("job_id") or payload.get("root_job_id")
    if root_uuid is None:
        logger.warning(f"Embeddings redis worker missing root_job_uuid for stage {stage}")
        return
    root_uuid = str(root_uuid)
    payload["root_job_uuid"] = root_uuid

    media_id = payload.get("media_id")
    if media_id is None and stage == "content":
        media_id = payload.get("item_id") or payload.get("content_id")
    if media_id is None:
        raise jobs_worker.EmbeddingsJobError("Missing media_id in embeddings payload", retryable=False)
    media_id = int(media_id)

    stage_job_type = _STAGE_JOB_TYPES.get(stage)
    job = {
        "uuid": message_id,
        "queue": streams.streams.get(stage),
        "job_type": stage_job_type,
        "owner_user_id": payload.get("user_id"),
        "request_id": payload.get("request_id"),
        "trace_id": payload.get("trace_id"),
    }
    user_id = jobs_worker._get_user_id(job, payload)

    embedding_model = payload.get("embedding_model")
    embedding_provider = payload.get("embedding_provider")
    embedding_model, embedding_provider = jobs_worker._resolve_model_provider(embedding_model, embedding_provider)
    payload["embedding_model"] = embedding_model
    payload["embedding_provider"] = embedding_provider

    chunk_size = jobs_worker._coerce_int(payload.get("chunk_size"), 1000)
    chunk_overlap = jobs_worker._coerce_int(payload.get("chunk_overlap"), 200)

    try:
        if stage == "chunking":
            jobs_worker._update_root_progress(root_uuid, progress_percent=1.0, progress_message="chunking started")
            result, skip = await jobs_worker._handle_chunking_job(
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
                jobs_worker._update_root_job(root_uuid, status="completed", result=payload_result)
                if jobs_worker._should_track_media_state(stage_job_type, payload):
                    jobs_worker._mark_media_embeddings_complete(user_id=user_id, media_id=media_id)
                return
            next_payload = dict(payload)
            next_payload.update(result)
            await client.xadd(streams.streams["embedding"], next_payload)
            return

        if stage == "embedding":
            result = await jobs_worker._handle_embedding_job(
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
            jobs_worker._update_root_result(
                root_uuid,
                result={
                    "total_chunks": result.get("total_chunks"),
                    "embedding_count": result.get("embedding_count"),
                },
            )
            jobs_worker._update_root_progress(root_uuid, progress_percent=66.0, progress_message="embedding completed")
            next_payload = dict(payload)
            next_payload.update(result)
            next_payload["embedding_model"] = result.get("embedding_model") or embedding_model
            next_payload["embedding_provider"] = result.get("embedding_provider") or embedding_provider
            await client.xadd(streams.streams["storage"], next_payload)
            return

        if stage == "storage":
            result = await jobs_worker._handle_storage_job(
                job,
                payload,
                media_id=media_id,
                user_id=user_id,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                root_uuid=root_uuid,
            )
            jobs_worker._update_root_job(root_uuid, status="completed", result=result)
            if jobs_worker._should_track_media_state(stage_job_type, payload):
                jobs_worker._mark_media_embeddings_complete(user_id=user_id, media_id=media_id)
            return

        if stage == "content":
            jobs_worker._update_root_progress(
                root_uuid,
                progress_percent=1.0,
                progress_message="content embedding started",
            )
            result = await jobs_worker._handle_content_job(
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
            jobs_worker._update_root_job(root_uuid, status="completed", result=result)
            if jobs_worker._should_track_media_state(stage_job_type, payload):
                jobs_worker._mark_media_embeddings_complete(user_id=user_id, media_id=media_id)
            return

        raise jobs_worker.EmbeddingsJobError(f"Unsupported embeddings stage {stage}", retryable=False)
    except jobs_worker.EmbeddingsJobError as exc:
        retry_cfg = _retry_limits()
        payload_val = locals().get("payload") or {}
        retry_count = jobs_worker._coerce_int(payload_val.get("retry_count"), 0)
        if exc.retryable and retry_count < retry_cfg["max_retries"]:
            backoff = getattr(exc, "backoff_seconds", None)
            if backoff is None:
                backoff = retry_cfg["base_backoff"] * (2 ** retry_count)
            backoff = min(float(backoff), float(retry_cfg["max_backoff"]))
            await asyncio.sleep(backoff)
            retry_payload = dict(payload_val)
            retry_payload["retry_count"] = retry_count + 1
            retry_payload["last_error"] = str(exc)
            await client.xadd(streams.streams[stage], retry_payload)
            return
        root_uuid_val = locals().get("root_uuid")
        if root_uuid_val:
            jobs_worker._update_root_job(root_uuid_val, status="failed", error=str(exc))
        if payload_val and jobs_worker._should_track_media_state(stage_job_type, payload_val):
            user_id_val = locals().get("user_id")
            media_id_val = locals().get("media_id")
            if user_id_val is not None and media_id_val is not None:
                jobs_worker._mark_media_embeddings_error_safely(
                    user_id=user_id_val,
                    media_id=media_id_val,
                    error_message=str(exc),
                )
        await _send_dlq(client, stage=stage, payload=payload_val, error=str(exc))
    except Exception as exc:
        payload_val = locals().get("payload") or {}
        root_uuid_val = locals().get("root_uuid")
        if root_uuid_val:
            jobs_worker._update_root_job(root_uuid_val, status="failed", error=str(exc))
        if payload_val and jobs_worker._should_track_media_state(stage_job_type, payload_val):
            user_id_val = locals().get("user_id")
            media_id_val = locals().get("media_id")
            if user_id_val is not None and media_id_val is not None:
                jobs_worker._mark_media_embeddings_error_safely(
                    user_id=user_id_val,
                    media_id=media_id_val,
                    error_message=str(exc),
                )
        await _send_dlq(client, stage=stage, payload=payload_val, error=str(exc))


async def _worker_loop(stage: str, worker_id: str, stop_event: asyncio.Event) -> None:
    streams = redis_pipeline.load_queues()
    stream = streams.streams[stage]
    group = streams.groups[stage]
    batch_size = _env_int("EMBEDDINGS_REDIS_BATCH_SIZE", 1)
    block_ms = _env_int("EMBEDDINGS_REDIS_POLL_INTERVAL_MS", 1000)
    require_redis = not redis_pipeline.allow_stub()

    client = await create_async_redis_client(
        context=f"embeddings-redis-{stage}",
        fallback_to_fake=not require_redis,
    )
    try:
        await _ensure_group(client, stream=stream, group=group)
        while not stop_event.is_set():
            try:
                response = await client.xreadgroup(
                    group,
                    worker_id,
                    streams={stream: ">"},
                    count=batch_size,
                    block=block_ms,
                )
            except Exception as exc:
                logger.warning(f"Embeddings redis worker read error stage={stage}: {exc}")
                await asyncio.sleep(1)
                continue
            if not response:
                continue
            for _stream_name, messages in response:
                for message_id, fields in messages:
                    await _handle_stage_message(
                        stage=stage,
                        message_id=message_id,
                        fields=fields,
                        client=client,
                        streams=streams,
                    )
                    try:
                        await client.xack(stream, group, message_id)
                    except Exception as exc:
                        logger.warning(f"Embeddings redis worker ack failed stage={stage} id={message_id}: {exc}")
    finally:
        with contextlib.suppress(Exception):
            await client.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Embeddings Redis Streams worker")
    parser.add_argument("--stage", choices=("chunking", "embedding", "storage", "content", "all"), default="all")
    args = parser.parse_args()

    stop_event = asyncio.Event()
    stages = ("chunking", "embedding", "storage", "content") if args.stage == "all" else (args.stage,)

    tasks = []
    for stage in stages:
        count = _env_int(f"EMBEDDINGS_REDIS_WORKERS_{stage.upper()}", 1)
        for idx in range(max(1, count)):
            worker_id = (os.getenv("EMBEDDINGS_REDIS_WORKER_ID") or f"embeddings-redis-{stage}-{os.getpid()}-{idx}").strip()
            tasks.append(asyncio.create_task(_worker_loop(stage, worker_id, stop_event)))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        stop_event.set()
    except Exception:
        stop_event.set()
        raise


if __name__ == "__main__":
    asyncio.run(main())
