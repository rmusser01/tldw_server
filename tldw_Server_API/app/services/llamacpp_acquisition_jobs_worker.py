"""Jobs worker for safe llama.cpp asset acquisition downloads."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.Local_LLM import llamacpp_acquisition_service as acquisition_service
from tldw_Server_API.app.core.Local_LLM.llamacpp_acquisition_jobs import (
    LLAMACPP_ACQUISITION_DOMAIN,
    LLAMACPP_ACQUISITION_QUEUE,
    LLAMACPP_DOWNLOAD_JOB_TYPE,
)
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b((?:bearer|token|api[_-]?key|password|secret|signature)\s*[:=]\s*)[^\s,;]+"
)
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)[^/@\s]+@")
_CANCEL_CHECK_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_PROGRESS_DB_UPDATE_INTERVAL_SECONDS = 1.0
_PROGRESS_DB_UPDATE_PERCENT_DELTA = 1.0
_PROGRESS_DB_UPDATE_BYTES_DELTA = 5 * 1024 * 1024
_CANCELLATION_DB_POLL_INTERVAL_SECONDS = 1.0


@dataclass
class _ProgressState:
    """Mutable progress snapshot reported back to the Jobs worker SDK."""

    percent: float | None = None
    message: str | None = None
    bytes_downloaded: int = 0
    total_bytes: int | None = None


@dataclass
class _ProgressUpdateThrottle:
    """Rate-limit persisted progress updates while keeping in-memory progress fresh."""

    interval_seconds: float = _PROGRESS_DB_UPDATE_INTERVAL_SECONDS
    percent_delta: float = _PROGRESS_DB_UPDATE_PERCENT_DELTA
    bytes_delta: int = _PROGRESS_DB_UPDATE_BYTES_DELTA
    _last_updated_at: float | None = None
    _last_percent: float | None = None
    _last_bytes: int = 0
    _last_message: str | None = None

    def should_update(self, *, bytes_downloaded: int, percent: float | None, message: str) -> bool:
        """Return whether a progress row should be persisted for this update."""

        now = time.monotonic()
        if self._last_updated_at is None:
            self._record(now=now, bytes_downloaded=bytes_downloaded, percent=percent, message=message)
            return True
        if message != self._last_message:
            self._record(now=now, bytes_downloaded=bytes_downloaded, percent=percent, message=message)
            return True
        if percent is not None and self._last_percent is not None:
            if abs(percent - self._last_percent) >= self.percent_delta:
                self._record(now=now, bytes_downloaded=bytes_downloaded, percent=percent, message=message)
                return True
        if bytes_downloaded - self._last_bytes >= self.bytes_delta:
            self._record(now=now, bytes_downloaded=bytes_downloaded, percent=percent, message=message)
            return True
        if now - self._last_updated_at >= self.interval_seconds:
            self._record(now=now, bytes_downloaded=bytes_downloaded, percent=percent, message=message)
            return True
        return False

    def _record(self, *, now: float, bytes_downloaded: int, percent: float | None, message: str) -> None:
        self._last_updated_at = now
        self._last_percent = percent
        self._last_bytes = bytes_downloaded
        self._last_message = message


@dataclass
class _CancellationPoller:
    """Poll injected cancellation each chunk and Jobs status at a bounded rate."""

    job_manager: Any
    job_id: int
    cancel_check: acquisition_service.CancelCheck | None
    interval_seconds: float = _CANCELLATION_DB_POLL_INTERVAL_SECONDS
    _last_status_poll_at: float | None = None

    def __call__(self) -> bool:
        if self._cancel_check_requested():
            return True
        now = time.monotonic()
        if self._last_status_poll_at is not None and now - self._last_status_poll_at < self.interval_seconds:
            return False
        self._last_status_poll_at = now
        return _job_status_cancelled(self.job_manager, self.job_id)

    def _cancel_check_requested(self) -> bool:
        if self.cancel_check is None:
            return False
        try:
            return bool(self.cancel_check())
        except _CANCEL_CHECK_EXCEPTIONS as exc:
            logger.debug(f"llama.cpp acquisition cancel_check failed for job {self.job_id}: {exc}")
            self._last_status_poll_at = time.monotonic()
            return _job_status_cancelled(self.job_manager, self.job_id)


class LlamaCppAcquisitionJobError(RuntimeError):
    """Normalized worker error with retry metadata for llama.cpp acquisition."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        backoff_seconds: int | None = None,
        failure_code: str = "llamacpp_acquisition_failed",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.failure_code = failure_code
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


def _build_worker_config(*, worker_id: str, queue: str) -> WorkerConfig:
    """Build the worker SDK configuration for llama.cpp acquisition jobs."""

    return WorkerConfig(
        domain=LLAMACPP_ACQUISITION_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=_coerce_int(os.getenv("LLAMACPP_ACQUISITION_JOBS_LEASE_SECONDS"), 120),
        renew_jitter_seconds=_coerce_int(
            os.getenv("LLAMACPP_ACQUISITION_JOBS_RENEW_JITTER_SECONDS"),
            5,
        ),
        renew_threshold_seconds=_coerce_int(
            os.getenv("LLAMACPP_ACQUISITION_JOBS_RENEW_THRESHOLD_SECONDS"),
            15,
        ),
        backoff_base_seconds=_coerce_int(
            os.getenv("LLAMACPP_ACQUISITION_JOBS_BACKOFF_BASE_SECONDS"),
            2,
        ),
        backoff_max_seconds=_coerce_int(
            os.getenv("LLAMACPP_ACQUISITION_JOBS_BACKOFF_MAX_SECONDS"),
            30,
        ),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(
            os.getenv("LLAMACPP_ACQUISITION_JOBS_RETRY_BACKOFF_SECONDS"),
            10,
        ),
    )


def _resolve_queue_name() -> str:
    """Return the configured acquisition queue name."""

    return (os.getenv("LLAMACPP_ACQUISITION_JOBS_QUEUE") or LLAMACPP_ACQUISITION_QUEUE).strip() or (
        LLAMACPP_ACQUISITION_QUEUE
    )


def _normalize_payload(value: Any) -> dict[str, Any]:
    """Normalize stored job payloads into a dictionary."""

    return dict(value) if isinstance(value, dict) else {}


def _should_cancel(job_manager: Any, job_id: int, cancel_check: acquisition_service.CancelCheck | None) -> bool:
    """Return whether the current job has been cancelled."""

    if cancel_check is not None:
        try:
            if bool(cancel_check()):
                return True
        except _CANCEL_CHECK_EXCEPTIONS as exc:
            logger.debug(f"llama.cpp acquisition cancel_check failed for job {job_id}: {exc}")
    return _job_status_cancelled(job_manager, job_id)


def _job_status_cancelled(job_manager: Any, job_id: int) -> bool:
    try:
        current = job_manager.get_job(int(job_id)) or {}
    except _CANCEL_CHECK_EXCEPTIONS as exc:
        logger.debug(f"llama.cpp acquisition Jobs status check failed for job {job_id}: {exc}")
        return False
    return str(current.get("status") or "").lower() == "cancelled"


def _finalize_cancelled(job_manager: Any, job_id: int, *, reason: str) -> None:
    with contextlib.suppress(Exception):
        job_manager.finalize_cancelled(int(job_id), reason=reason)


def _safe_error_message(value: Any) -> str:
    """Redact URL credentials and secret-looking assignments from worker errors."""

    text = str(value or "llama.cpp acquisition job failed")
    text = _URL_USERINFO_RE.sub(r"\1", text)
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1[redacted]", text)
    return " ".join(text.split())[:1000]


async def process_llamacpp_acquisition_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager,
    worker_id: str = "llamacpp-acquisition-worker",
    progress: _ProgressState | None = None,
    stream_factory: acquisition_service.DownloadStreamFactory | None = None,
    cancel_check: acquisition_service.CancelCheck | None = None,
) -> dict[str, Any]:
    """Download, validate, promote, and optionally register one llama.cpp asset job."""

    del worker_id
    try:
        job_id = int(job.get("id"))
    except (TypeError, ValueError) as exc:
        raise LlamaCppAcquisitionJobError("invalid job id", retryable=False) from exc
    if str(job.get("job_type") or "").lower() != LLAMACPP_DOWNLOAD_JOB_TYPE:
        raise LlamaCppAcquisitionJobError("unsupported job_type", retryable=False)

    payload = _normalize_payload(job.get("payload"))
    partial_path = None
    try:
        validated = acquisition_service.validate_download_payload(payload)
        final_path = validated.destination_path
        if final_path.exists() and not validated.overwrite:
            raise ServerError("Destination file already exists; set overwrite=true to replace it.")
        partial_path = acquisition_service.partial_download_path(final_path, str(job_id))

        if _should_cancel(job_manager, job_id, cancel_check):
            acquisition_service.cleanup_partial_if_needed(partial_path)
            _finalize_cancelled(job_manager, job_id, reason="cancel requested before download")
            return {}

        if progress is not None:
            progress.percent = 0.0
            progress.message = "queued"

        progress_throttle = _ProgressUpdateThrottle()

        async def _progress_callback(update: dict[str, Any]) -> None:
            bytes_downloaded = int(update.get("bytes_downloaded") or 0)
            percent_value = update.get("progress_percent")
            percent = float(percent_value) if percent_value is not None else None
            message = str(update.get("progress_message") or "downloading")
            if progress is not None:
                progress.bytes_downloaded = bytes_downloaded
                total = update.get("total_bytes")
                progress.total_bytes = int(total) if total is not None else None
                progress.percent = percent if percent is not None else progress.percent
                progress.message = message
            persisted_percent = progress.percent if progress is not None else percent
            if not progress_throttle.should_update(
                bytes_downloaded=bytes_downloaded,
                percent=persisted_percent,
                message=message,
            ):
                return
            job_manager.update_job_progress(
                job_id,
                progress_percent=persisted_percent,
                progress_message=message,
            )

        cancellation_poller = _CancellationPoller(job_manager, job_id, cancel_check)
        bytes_written = await acquisition_service.download_to_partial(
            validated,
            partial_path,
            progress_callback=_progress_callback,
            cancel_check=cancellation_poller,
            stream_factory=stream_factory,
        )

        if _should_cancel(job_manager, job_id, cancel_check):
            acquisition_service.cleanup_partial_if_needed(partial_path)
            _finalize_cancelled(job_manager, job_id, reason="cancel requested during download")
            return {}

        warnings = list(validated.warnings)
        warnings.extend(
            acquisition_service.validate_completed_download(
                partial_path,
                validated.expected_sha256,
                validated.expected_size_bytes,
            )
        )
        final_path = acquisition_service.promote_partial_download(
            partial_path,
            final_path,
            overwrite=validated.overwrite,
        )
        partial_path = None

        asset_id = None
        if validated.register_asset:
            asset = acquisition_service.register_completed_download(final_path)
            asset_id = _asset_value(asset, "asset_id")
            warnings.extend(_asset_warnings(asset))

        if progress is not None:
            progress.percent = 100.0
            progress.message = "completed"
            progress.bytes_downloaded = bytes_written
            progress.total_bytes = progress.total_bytes or validated.expected_size_bytes
        job_manager.update_job_progress(
            job_id,
            progress_percent=100.0,
            progress_message="completed",
        )
        result_progress: dict[str, Any] = {
            "bytes_downloaded": bytes_written,
            "progress_percent": 100.0,
            "progress_message": "completed",
        }
        total_bytes = progress.total_bytes if progress is not None else validated.expected_size_bytes
        if total_bytes is not None:
            result_progress["total_bytes"] = total_bytes
        return {
            "status": "ready",
            "asset_id": asset_id,
            "destination_path": str(final_path),
            "bytes": bytes_written,
            "warnings": warnings,
            "progress": result_progress,
        }
    except acquisition_service.LlamaCppDownloadCancelled:
        if partial_path is not None:
            acquisition_service.cleanup_partial_if_needed(partial_path)
        _finalize_cancelled(job_manager, job_id, reason="cancel requested during download")
        return {}
    except acquisition_service.LlamaCppDownloadError as exc:
        if partial_path is not None:
            acquisition_service.cleanup_partial_if_needed(partial_path)
        raise LlamaCppAcquisitionJobError(
            _safe_error_message(exc),
            retryable=True,
            failure_code="download_failed",
        ) from exc
    except ServerError as exc:
        if partial_path is not None:
            acquisition_service.cleanup_partial_if_needed(partial_path)
        raise LlamaCppAcquisitionJobError(
            _safe_error_message(exc),
            retryable=False,
            failure_code="validation_failed",
        ) from exc


def _asset_value(asset: Any, key: str) -> str | None:
    if isinstance(asset, dict):
        value = asset.get(key)
    else:
        value = getattr(asset, key, None)
    if value is None:
        return None
    text = str(value)
    return text or None


def _asset_warnings(asset: Any) -> list[str]:
    if isinstance(asset, dict):
        value = asset.get("warnings")
    else:
        value = getattr(asset, "warnings", None)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


async def run_llamacpp_acquisition_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the long-lived llama.cpp acquisition download worker until stopped."""

    queue_name = _resolve_queue_name()
    worker_id = (
        os.getenv("LLAMACPP_ACQUISITION_JOBS_WORKER_ID")
        or f"llamacpp-acquisition-worker-{os.getpid()}"
    ).strip()
    jm = _jobs_manager()
    sdk = WorkerSDK(jm, _build_worker_config(worker_id=worker_id, queue=queue_name))
    progress = _ProgressState()

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        return await process_llamacpp_acquisition_job(
            job,
            job_manager=jm,
            worker_id=worker_id,
            progress=progress,
        )

    async def _cancel_check(job: dict[str, Any]) -> bool:
        try:
            return _should_cancel(jm, int(job.get("id")), None)
        except (TypeError, ValueError):
            return False

    def _progress_cb() -> dict[str, Any]:
        update: dict[str, Any] = {}
        if progress.percent is not None:
            update["progress_percent"] = progress.percent
        if progress.message is not None:
            update["progress_message"] = progress.message
        return update

    async def _watch_stop() -> None:
        if stop_event is None:
            return
        await stop_event.wait()
        sdk.stop()

    logger.info("Starting llama.cpp Acquisition Jobs worker (queue={})", queue_name)
    watcher = asyncio.create_task(_watch_stop())
    try:
        await sdk.run(handler=_handler, cancel_check=_cancel_check, progress_cb=_progress_cb)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_llamacpp_acquisition_jobs_worker())
