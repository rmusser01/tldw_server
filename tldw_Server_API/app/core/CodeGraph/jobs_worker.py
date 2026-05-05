"""Jobs worker entrypoint for native CodeGraph indexing."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager

from .config import CodeGraphSettings
from .indexer import CodeGraphIndexer, IndexingResult
from .jobs import CODEGRAPH_INDEX_JOB_TYPE, CODEGRAPH_JOBS_DOMAIN, codegraph_jobs_queue
from .language_registry import CodeGraphLanguageRegistry


class CodeGraphJobError(RuntimeError):
    """Non-retryable CodeGraph Jobs validation or execution error."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


async def handle_codegraph_index_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one CodeGraph index or sync job and return serialized result data."""
    return await asyncio.to_thread(_handle_codegraph_index_job_sync, job)


def _handle_codegraph_index_job_sync(job: dict[str, Any]) -> dict[str, Any]:
    """Synchronous implementation for one CodeGraph index or sync job."""
    job_type = str(job.get("job_type") or "").strip()
    if job_type != CODEGRAPH_INDEX_JOB_TYPE:
        raise CodeGraphJobError(f"unsupported job_type: {job_type or '<missing>'}", retryable=False)

    payload = _coerce_payload(job.get("payload"))
    operation = str(payload.get("operation") or "").strip().lower()
    if operation not in {"index", "sync"}:
        raise CodeGraphJobError(f"unsupported operation: {operation or '<missing>'}", retryable=False)

    workspace_root = _required_path(payload, "workspace_root").resolve(strict=False)
    workspace_key = str(payload.get("workspace_key") or "").strip()
    if not workspace_key:
        raise CodeGraphJobError("missing workspace_key", retryable=False)

    settings = CodeGraphSettings.from_mapping(_coerce_mapping(payload.get("settings"), "settings"))
    index_db_path = _required_path(payload, "index_db_path").resolve(strict=False)
    _validate_index_path(index_db_path=index_db_path, settings=settings)

    repository = CodeGraphRepository(index_db_path)
    indexer = CodeGraphIndexer(settings=settings, registry=CodeGraphLanguageRegistry())
    languages = _coerce_languages(payload.get("languages"))
    max_files = _coerce_optional_int(payload.get("max_files"), "max_files")

    if operation == "index":
        result = indexer.index_workspace(
            workspace_root,
            workspace_key,
            repository,
            force=bool(payload.get("force", False)),
            languages=languages,
            max_files=max_files,
        )
    else:
        result = indexer.sync_workspace(
            workspace_root,
            workspace_key,
            repository,
            languages=languages,
            max_files=max_files,
        )

    return _job_result_to_dict(
        result,
        operation=operation,
        workspace_key=workspace_key,
        index_db_path=index_db_path,
    )


async def run_codegraph_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the CodeGraph Jobs worker loop until stopped."""
    worker_id = (os.getenv("CODEGRAPH_JOBS_WORKER_ID") or f"codegraph-jobs-{os.getpid()}").strip()
    queue = codegraph_jobs_queue()
    cfg = WorkerConfig(
        domain=CODEGRAPH_JOBS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=_coerce_int(os.getenv("CODEGRAPH_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60),
        renew_jitter_seconds=_coerce_int(
            os.getenv("CODEGRAPH_JOBS_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
            5,
        ),
        renew_threshold_seconds=_coerce_int(
            os.getenv("CODEGRAPH_JOBS_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
            10,
        ),
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher_task = asyncio.create_task(_watch_stop())

    logger.info("CodeGraph Jobs worker starting: queue={} worker_id={}", queue, worker_id)
    try:
        await sdk.run(handler=handle_codegraph_index_job)
    finally:
        if stop_watcher_task is not None and not stop_watcher_task.done():
            stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher_task


def _coerce_payload(value: Any) -> dict[str, Any]:
    """Return a JSON payload mapping from a Jobs row payload value."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CodeGraphJobError(f"invalid payload json: {exc}", retryable=False) from exc
        if isinstance(loaded, dict):
            return loaded
    raise CodeGraphJobError("payload must be an object", retryable=False)


def _coerce_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return a dict payload field or raise a non-retryable worker error."""
    if isinstance(value, dict):
        return dict(value)
    raise CodeGraphJobError(f"{field_name} must be an object", retryable=False)


def _required_path(payload: dict[str, Any], field_name: str) -> Path:
    """Return a non-empty payload path field."""
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise CodeGraphJobError(f"missing {field_name}", retryable=False)
    return Path(value).expanduser()


def _validate_index_path(*, index_db_path: Path, settings: CodeGraphSettings) -> None:
    """Ensure the job can only write below the configured CodeGraph index base."""
    index_base = settings.index_base_dir.expanduser().resolve(strict=False)
    if index_base not in index_db_path.parents:
        raise CodeGraphJobError("index_db_path_outside_index_base", retryable=False)


def _coerce_languages(value: Any) -> list[str] | None:
    """Return optional language filters from a Jobs payload."""
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise CodeGraphJobError("languages must be an array of strings", retryable=False)


def _coerce_optional_int(value: Any, field_name: str) -> int | None:
    """Return an optional integer payload field."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise CodeGraphJobError(f"{field_name} must be an integer", retryable=False)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CodeGraphJobError(f"{field_name} must be an integer", retryable=False) from exc
    if parsed < 1:
        raise CodeGraphJobError(f"{field_name} must be positive", retryable=False)
    return parsed


def _job_result_to_dict(
    result: IndexingResult,
    *,
    operation: str,
    workspace_key: str,
    index_db_path: Path,
) -> dict[str, Any]:
    """Serialize a CodeGraph indexing result for Jobs completion storage."""
    return {
        "operation": operation,
        "workspace_key": workspace_key,
        "index_db_path": str(index_db_path),
        "status": result.status,
        "counters": dict(result.counters),
        "errors": list(result.errors),
    }


if __name__ == "__main__":
    asyncio.run(run_codegraph_jobs_worker())
