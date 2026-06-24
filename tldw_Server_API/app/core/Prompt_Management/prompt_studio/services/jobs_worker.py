"""
Prompt Studio Jobs worker (Phase 2):

- Consumes core Jobs entries for Prompt Studio jobs.
- Executes prompt studio job handlers via JobProcessor.
- Updates Jobs status/result via the core JobManager.

Job contract (domain/queue/job_type):
- domain = "prompt_studio"
- queue = os.getenv("PROMPT_STUDIO_JOBS_QUEUE", "default")
- job_type = "optimization" | "evaluation" | "generation"

Payload fields (examples):
- optimization_id / evaluation_id / project_id (generation)
- entity_id (generic id for job processor)
- prompt_id, test_case_ids, model_configs, optimization_config, optimizer_type
- request_id (optional)

Usage:
  python -m tldw_Server_API.app.core.Prompt_Management.prompt_studio.services.jobs_worker
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
from collections import OrderedDict
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_prompt_studio_database,
    get_content_backend_instance,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import JobProcessor
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.quota_config import (
    apply_prompt_studio_quota_defaults,
    apply_prompt_studio_quota_policy,
)

_PROMPT_STUDIO_DOMAIN = "prompt_studio"

if os.getenv("PROMPT_STUDIO_JOBS_BACKEND") not in {"", "core"}:
    logger.warning("PROMPT_STUDIO_JOBS_BACKEND is not core; forcing core backend for prompt studio jobs worker")
    os.environ["PROMPT_STUDIO_JOBS_BACKEND"] = "core"


class PromptStudioJobError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


_DB_CACHE: OrderedDict[str, Any] = OrderedDict()
_PROCESSOR_CACHE: OrderedDict[str, JobProcessor] = OrderedDict()
_ACTIVE_USER_COUNTS: dict[str, int] = {}
_PENDING_CLOSE: dict[str, list[Any]] = {}
_CACHE_LOCK = threading.RLock()


def _jobs_manager() -> JobManager:
    apply_prompt_studio_quota_defaults()
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


_MAX_CACHE_ENTRIES = max(1, _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_CACHE_MAX_USERS"), 20))


def _normalize_user_id(value: Any, *, allow_default: bool = False) -> str:
    if value is None or str(value).strip() == "":
        if not allow_default:
            raise PromptStudioJobError("Missing owner_user_id for prompt studio job", retryable=False)
        return str(DatabasePaths.get_single_user_id())
    return str(value)


def _close_db(db: Any, *, user_id: str | None = None) -> None:
    for method_name in ("close_connection", "close"):
        method = getattr(db, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                logger.opt(exception=exc).warning(
                    "Failed to close Prompt Studio DB for user {} via {}",
                    user_id or "<unknown>",
                    method_name,
                )
            return


def _evict_cache_entries_if_needed() -> None:
    while len(_DB_CACHE) > _MAX_CACHE_ENTRIES:
        user_id, db = _DB_CACHE.popitem(last=False)
        _PROCESSOR_CACHE.pop(user_id, None)
        if _ACTIVE_USER_COUNTS.get(user_id, 0) > 0:
            _PENDING_CLOSE.setdefault(user_id, []).append(db)
            logger.debug("Deferred Prompt Studio DB close for active user {}", user_id)
            continue
        _close_db(db, user_id=user_id)


@contextlib.contextmanager
def _active_user_cache_scope(user_id: str):
    with _CACHE_LOCK:
        _ACTIVE_USER_COUNTS[user_id] = _ACTIVE_USER_COUNTS.get(user_id, 0) + 1
    try:
        yield
    finally:
        pending_close: list[Any] = []
        with _CACHE_LOCK:
            remaining = _ACTIVE_USER_COUNTS.get(user_id, 0) - 1
            if remaining > 0:
                _ACTIVE_USER_COUNTS[user_id] = remaining
            else:
                _ACTIVE_USER_COUNTS.pop(user_id, None)
                pending_close = _PENDING_CLOSE.pop(user_id, [])
        for db in pending_close:
            _close_db(db, user_id=user_id)


def _normalize_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _build_worker_config(*, worker_id: str, queue: str) -> WorkerConfig:
    lease_seconds = _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_LEASE_SECONDS"), 60)
    renew_jitter_seconds = _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_RENEW_JITTER_SECONDS"), 5)
    renew_threshold_seconds = _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_RENEW_THRESHOLD_SECONDS"), 10)

    heartbeat_raw = (os.getenv("TLDW_PS_HEARTBEAT_SECONDS") or "").strip()
    if heartbeat_raw:
        heartbeat_seconds = _coerce_int(heartbeat_raw, 0)
        if heartbeat_seconds > 0:
            max_threshold = max(1, lease_seconds - 1) if lease_seconds > 1 else 1
            desired_threshold = max(1, lease_seconds - heartbeat_seconds)
            renew_threshold_seconds = min(max_threshold, desired_threshold)

    return WorkerConfig(
        domain=_PROMPT_STUDIO_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter_seconds,
        renew_threshold_seconds=renew_threshold_seconds,
        backoff_base_seconds=_coerce_int(os.getenv("PROMPT_STUDIO_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=_coerce_int(os.getenv("PROMPT_STUDIO_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("PROMPT_STUDIO_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )


def _get_db(user_id: str):
    with _CACHE_LOCK:
        cached = _DB_CACHE.get(user_id)
        if cached is not None:
            _DB_CACHE.move_to_end(user_id)
            return cached
        backend = get_content_backend_instance()
        db_path = DatabasePaths.get_prompt_studio_db_path(user_id)
        client_id = f"prompt_studio_jobs_worker:{user_id}"
        db = create_prompt_studio_database(
            client_id=client_id,
            db_path=db_path,
            backend=backend,
        )
        _DB_CACHE[user_id] = db
        _DB_CACHE.move_to_end(user_id)
        _evict_cache_entries_if_needed()
        return db


def _get_processor(user_id: str) -> JobProcessor:
    with _CACHE_LOCK:
        cached = _PROCESSOR_CACHE.get(user_id)
        if cached is not None:
            _PROCESSOR_CACHE.move_to_end(user_id)
            if user_id in _DB_CACHE:
                _DB_CACHE.move_to_end(user_id, last=True)
            return cached
        db = _get_db(user_id)
        processor = JobProcessor(db)
        _PROCESSOR_CACHE[user_id] = processor
        _PROCESSOR_CACHE.move_to_end(user_id)
        _evict_cache_entries_if_needed()
        return processor


def _resolve_entity_id(job_type: str, payload: dict[str, Any]) -> int:
    if job_type == "optimization":
        value = payload.get("optimization_id") or payload.get("entity_id")
    elif job_type == "evaluation":
        value = payload.get("evaluation_id") or payload.get("entity_id")
    elif job_type == "generation":
        value = payload.get("project_id") or payload.get("entity_id")
    else:
        value = payload.get("entity_id")
    if value is None:
        raise PromptStudioJobError(f"Missing entity id for {job_type} job", retryable=False)
    return _coerce_int(value, 0)


async def _handle_job(job: dict[str, Any]) -> dict[str, Any]:
    job_type = str(job.get("job_type") or "").strip().lower()
    if job_type not in {"optimization", "evaluation", "generation"}:
        raise PromptStudioJobError(f"Unsupported prompt studio job type: {job_type}", retryable=False)

    payload = _normalize_payload(job.get("payload"))
    payload["job_id"] = str(job.get("uuid") or job.get("id"))
    payload.setdefault("request_id", job.get("request_id"))

    entity_id = _resolve_entity_id(job_type, payload)
    if job_type == "optimization":
        payload.setdefault("optimization_id", entity_id)
    elif job_type == "evaluation":
        payload.setdefault("evaluation_id", entity_id)
    elif job_type == "generation":
        payload.setdefault("project_id", entity_id)

    owner_user_id = job.get("owner_user_id")
    owner_missing = owner_user_id is None or str(owner_user_id).strip() == ""
    if owner_missing and payload.get("user_id") is not None:
        raise PromptStudioJobError("Missing owner_user_id for prompt studio job", retryable=False)
    user_id = _normalize_user_id(owner_user_id, allow_default=True)
    with _active_user_cache_scope(user_id):
        processor = _get_processor(user_id)

        if job_type == "optimization":
            return await processor.process_optimization_job(payload, entity_id)
        if job_type == "evaluation":
            return await processor.process_evaluation_job(payload, entity_id)
        return await processor.process_generation_job(payload, entity_id)


async def _inflight_quota_guard(job: dict[str, Any], jm: JobManager) -> bool:
    owner = job.get("owner_user_id")
    if owner is None or str(owner).strip() == "":
        return True
    owner_id = str(owner)
    try:
        await apply_prompt_studio_quota_policy(owner_id)
    except Exception as exc:
        logger.debug("Prompt Studio quota policy lookup failed for {}: {}", owner_id, exc)
    try:
        max_inflight = jm._quota_get("JOBS_QUOTA_MAX_INFLIGHT", _PROMPT_STUDIO_DOMAIN, owner_id)
    except Exception:
        max_inflight = 0
    if not max_inflight:
        return True
    current = jm.count_processing_for_owner(domain=_PROMPT_STUDIO_DOMAIN, owner_user_id=owner_id)
    if current > int(max_inflight):
        logger.info("Prompt Studio inflight quota reached for user {}; requeueing job {}", owner_id, job.get("id"))
        return False
    return True


async def run_prompt_studio_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("PROMPT_STUDIO_JOBS_WORKER_ID") or f"prompt-studio-jobs-{os.getpid()}").strip()
    queue = (os.getenv("PROMPT_STUDIO_JOBS_QUEUE") or "default").strip() or "default"
    cfg = _build_worker_config(worker_id=worker_id, queue=queue)

    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    stop_task: asyncio.Task[None] | None = None
    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(_watch_stop())
    logger.info(f"Prompt Studio Jobs worker starting (queue={queue}, worker_id={worker_id})")
    try:
        await sdk.run(handler=_handle_job, acquire_guard=lambda job: _inflight_quota_guard(job, jm))
    finally:
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


async def main() -> None:
    await run_prompt_studio_jobs_worker()


if __name__ == "__main__":
    asyncio.run(main())
