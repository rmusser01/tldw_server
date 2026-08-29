from __future__ import annotations

import asyncio
import contextlib
import copy
import hmac
import os
import re
import secrets
import sqlite3
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.testing import is_test_mode, is_truthy

from .manager import JobManager
from .operations.contracts import (
    ApplyPreparedDispositionCommand,
    EnsureLeaseHorizonCommand,
    LeaseHorizonResult,
    OperationOutcome,
    PreparedDispositionOrigin,
    PreparedDispositionResult,
    PreparedJobDisposition,
)

try:
    import psycopg  # type: ignore

    _WORKER_SDK_BACKEND_EXCEPTIONS: tuple[type[BaseException], ...] = (
        sqlite3.Error,
        psycopg.Error,
    )
except ImportError:
    _WORKER_SDK_BACKEND_EXCEPTIONS = (sqlite3.Error,)

CancelCheck = Callable[[dict[str, Any]], Awaitable[bool]]
CompletionCallback = Callable[
    [dict[str, Any], dict[str, Any]],
    Awaitable[None],
]
FailureCallback = Callable[
    [dict[str, Any], Exception],
    Awaitable[None],
]
_SLIDES_JOBS_KEY_RE = re.compile(r"slides:v1:[0-9a-f]{64}\Z")


def _same_slides_jobs_key(left: object, right: object) -> bool:
    return bool(
        isinstance(left, str)
        and _SLIDES_JOBS_KEY_RE.fullmatch(left) is not None
        and isinstance(right, str)
        and _SLIDES_JOBS_KEY_RE.fullmatch(right) is not None
        and hmac.compare_digest(left, right)
    )


class WorkerTerminalizationConflict(Exception):
    """Raised when the exact terminal CAS cannot bind to the acquired job."""


@dataclass(frozen=True, slots=True)
class WorkerTerminalOutcome:
    """Closed handler-returned terminal outcome for failed or cancelled work."""

    status: str
    error_code: str
    message: str

    def __post_init__(self) -> None:
        if self.status not in {"failed", "cancelled"}:
            raise ValueError("terminal status must be failed or cancelled")
        if re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", self.error_code) is None:
            raise ValueError("terminal error_code is invalid")
        if not isinstance(self.message, str) or len(self.message) > 1024:
            raise ValueError("terminal message exceeds 1024 characters")
        if any(ord(character) < 32 for character in self.message):
            raise ValueError("terminal message contains control characters")


JobHandler = Callable[
    [dict[str, Any]],
    Awaitable[dict[str, Any] | WorkerTerminalOutcome | None],
]


@dataclass(frozen=True, slots=True)
class WorkerLeaseSnapshot:
    """Read-only lease evidence visible to one prepared handler."""

    worker_id: str
    lease_id: str
    leased_until: datetime | None
    renewal_lost: bool


def _lease_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class WorkerExecutionContext:
    """Prepared-handler access to current exact lease evidence."""

    __slots__ = (
        "_domain",
        "_job_id",
        "_job_type",
        "_jm",
        "_lease_id",
        "_leased_until",
        "_payload",
        "_queue",
        "_renewal_lost",
        "_worker_id",
    )

    def __init__(
        self,
        jm: JobManager,
        job: dict[str, Any],
        *,
        worker_id: str,
    ) -> None:
        self._jm = jm
        self._job_id = int(job["id"])
        self._domain = str(job["domain"])
        self._queue = str(job["queue"])
        self._job_type = str(job["job_type"])
        payload = job["payload"]
        if not isinstance(payload, dict):
            raise TypeError("prepared job payload must be an object")
        self._payload = copy.deepcopy(payload)
        self._worker_id = worker_id
        self._lease_id = str(job["lease_id"])
        self._leased_until = _lease_datetime(job.get("leased_until"))
        self._renewal_lost = False

    @property
    def renewal_lost(self) -> bool:
        return self._renewal_lost

    def _mark_renewal_lost(self) -> None:
        self._renewal_lost = True

    def snapshot(self) -> WorkerLeaseSnapshot:
        return WorkerLeaseSnapshot(
            worker_id=self._worker_id,
            lease_id=self._lease_id,
            leased_until=self._leased_until,
            renewal_lost=self._renewal_lost,
        )

    async def ensure_lease_horizon(self, seconds: int) -> bool:
        try:
            result = self._jm.ensure_lease_horizon(
                EnsureLeaseHorizonCommand(
                    job_id=self._job_id,
                    domain=self._domain,
                    queue=self._queue,
                    job_type=self._job_type,
                    expected_payload=self._payload,
                    worker_id=self._worker_id,
                    lease_id=self._lease_id,
                    minimum_seconds=seconds,
                )
            )
        except Exception as exc:  # noqa: BLE001 - backend isolation boundary
            self._renewal_lost = True
            logger.bind(error_type=type(exc).__name__).warning(
                "Jobs prepared lease horizon failed"
            )
            return False

        if isinstance(result, LeaseHorizonResult) and result.leased_until is not None:
            self._leased_until = result.leased_until
        ensured = bool(
            isinstance(result, LeaseHorizonResult)
            and result.outcome is OperationOutcome.APPLIED
            and result.ensured
            and result.leased_until is not None
        )
        if not ensured:
            self._renewal_lost = True
        return ensured


PreparedJobHandler = Callable[
    [dict[str, Any], WorkerExecutionContext],
    Awaitable[PreparedJobDisposition],
]
PreAcquireGuard = Callable[[], Awaitable[bool]]
PreparedHandlerErrorDisposition = Callable[
    [dict[str, Any], type[BaseException]],
    PreparedJobDisposition,
]
PreparedDispositionCallback = Callable[
    [dict[str, Any], PreparedJobDisposition, PreparedDispositionResult],
    Awaitable[None],
]

_WORKER_SDK_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    *_WORKER_SDK_BACKEND_EXCEPTIONS,
)


@dataclass
class WorkerConfig:
    domain: str
    queue: str
    worker_id: str
    lease_seconds: int = 30
    renew_jitter_seconds: int = 5
    renew_threshold_seconds: int = 10
    backoff_base_seconds: int = 2
    backoff_max_seconds: int = 30
    # Retry on handler exception
    retry_on_exception: bool = True
    retry_backoff_seconds: int = 10
    completion_callback_timeout_seconds: float = 30.0
    completion_callback_max_detached_tasks: int = 32
    bind_completion_token: bool = False


class WorkerSDK:
    """Lightweight worker helper: acquisition, auto-renew, progress heartbeats, and cancellation checks.

    Example:
        sdk = WorkerSDK(JobManager(), WorkerConfig(domain='prompt_studio', queue='default', worker_id='w1'))
        await sdk.run(handler=my_handler)
    """

    def __init__(self, jm: JobManager, cfg: WorkerConfig):
        self.jm = jm
        self.cfg = cfg
        self._stop = asyncio.Event()
        # Allow test overrides without monkeypatching global asyncio.sleep
        # (keeps event loop behavior stable under tests)
        self._sleep = asyncio.sleep
        self._detached_completion_callbacks: set[asyncio.Task[None]] = set()
        # Detect test mode for more responsive sleeps and optional iteration caps
        try:
            self._test_mode = is_test_mode()
        except (TypeError, ValueError):
            self._test_mode = False
        try:
            self._max_iters = int(os.getenv("JOBS_WORKER_MAX_ITERATIONS", "0") or "0")
        except (TypeError, ValueError):
            self._max_iters = 0

    def _observe_detached_completion_callback(
        self,
        task: asyncio.Task[None],
    ) -> None:
        """Release and consume a detached callback task's eventual outcome."""

        self._detached_completion_callbacks.discard(task)
        if not task.cancelled():
            task.exception()

    def _detach_completion_callback(self, task: asyncio.Task[None]) -> None:
        """Track a cancellation-resistant callback without awaiting it."""

        if task.done():
            self._observe_detached_completion_callback(task)
            return
        self._detached_completion_callbacks.add(task)
        task.add_done_callback(self._observe_detached_completion_callback)

    async def _invoke_completion_callback(
        self,
        callback: CompletionCallback | FailureCallback,
        job: dict[str, Any],
        callback_value: dict[str, Any] | Exception,
        *,
        callback_name: str,
    ) -> None:
        await self._invoke_callback(
            callback,
            job,
            callback_value,
            callback_name=callback_name,
        )

    async def _invoke_prepared_disposition_callback(
        self,
        callback: PreparedDispositionCallback,
        job: dict[str, Any],
        disposition: PreparedJobDisposition,
        result: PreparedDispositionResult,
        *,
        callback_name: str,
    ) -> None:
        await self._invoke_callback(
            callback,
            job,
            disposition,
            result,
            callback_name=callback_name,
        )

    async def _invoke_callback(
        self,
        callback: Callable[..., Awaitable[None]],
        *callback_args: object,
        callback_name: str,
    ) -> None:
        """Run bounded post-finalization work without re-finalizing the job.

        The deadline cannot preempt callback code that blocks the event loop
        before reaching an await point.
        """

        try:
            max_detached = max(
                0,
                int(self.cfg.completion_callback_max_detached_tasks),
            )
        except (TypeError, ValueError):
            max_detached = 0
        if len(self._detached_completion_callbacks) >= max_detached:
            logger.warning(
                "Jobs worker {} callback skipped: detached callback capacity reached",
                callback_name,
            )
            return

        async def invoke() -> None:
            await callback(*callback_args)

        callback_task = asyncio.create_task(invoke())
        try:
            done, _pending = await asyncio.wait(
                {callback_task},
                timeout=max(
                    0.01,
                    float(self.cfg.completion_callback_timeout_seconds),
                ),
            )
        except asyncio.CancelledError:
            callback_task.cancel()
            self._detach_completion_callback(callback_task)
            raise
        except Exception as exc:  # noqa: BLE001 - callback is an isolation boundary
            callback_task.cancel()
            self._detach_completion_callback(callback_task)
            logger.bind(error_type=type(exc).__name__).warning(
                "Jobs worker {} callback failed",
                callback_name,
            )
            return

        if callback_task not in done:
            callback_task.cancel()
            self._detach_completion_callback(callback_task)
            logger.bind(error_type=TimeoutError.__name__).warning(
                "Jobs worker {} callback failed",
                callback_name,
            )
            return

        try:
            callback_task.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - callback is an isolation boundary
            logger.bind(error_type=type(exc).__name__).warning(
                "Jobs worker {} callback failed",
                callback_name,
            )

    async def _sleep_chunked(self, total_seconds: float) -> None:
        """Sleep until the delay elapses or stop() is requested.

        The worker loop spends most of its idle time in backoff sleeps. When
        shutdown calls stop(), we need those sleeps to wake promptly instead of
        waiting out the full backoff interval.
        """
        delay = max(0.0, float(total_seconds))
        if delay <= 0 or self._stop.is_set():
            return

        sleep_task = asyncio.create_task(self._sleep(delay))
        stop_task = asyncio.create_task(self._stop.wait())
        try:
            done, _ = await asyncio.wait(
                {sleep_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if sleep_task in done:
                await sleep_task
        finally:
            for task in (sleep_task, stop_task):
                if task.done():
                    continue
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def stop(self) -> None:
        self._stop.set()

    async def _auto_renew(self, job: dict[str, Any], progress_cb: Callable[[], dict[str, Any]] | None = None) -> None:
        lease = int(max(1, self.cfg.lease_seconds))
        jitter = max(0, int(self.cfg.renew_jitter_seconds))
        threshold = max(1, int(self.cfg.renew_threshold_seconds))
        job_id = int(job.get('id'))
        lease_id = job.get('lease_id')
        iters = 0
        while True:
            # Sleep for lease - threshold, plus small jitter
            sleep_for = max(1, lease - threshold) + (secrets.randbelow(jitter + 1) if jitter else 0)
            await self._sleep(float(sleep_for))
            kwargs = {"job_id": job_id, "seconds": lease, "worker_id": self.cfg.worker_id, "lease_id": lease_id}
            if progress_cb:
                try:
                    upd = progress_cb() or {}
                    if 'progress_percent' in upd:
                        kwargs['progress_percent'] = float(upd['progress_percent'])
                    if 'progress_message' in upd:
                        kwargs['progress_message'] = str(upd['progress_message'])
                except _WORKER_SDK_NONCRITICAL_EXCEPTIONS:
                    pass
            try:
                ok = self.jm.renew_job_lease(**kwargs)
                if not ok:
                    logger.debug(f"Auto-renew failed for job {job_id}; stopping renew loop")
                    return
            except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Auto-renew error for job {job_id}: {e}")
                return
            iters += 1
            if self._max_iters and iters >= self._max_iters:
                logger.debug("Auto-renew reached max iterations; exiting loop")
                return

    async def _auto_renew_prepared(
        self,
        context: WorkerExecutionContext,
        stop_requested: asyncio.Event,
    ) -> None:
        iters = 0
        while True:
            try:
                requested_lease = max(1, int(self.cfg.lease_seconds))
                lease_cap = max(
                    1,
                    int(os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600"),
                )
                effective_lease = max(1, min(requested_lease, lease_cap))
                jitter = max(0, int(self.cfg.renew_jitter_seconds))
                threshold = max(1, int(self.cfg.renew_threshold_seconds))
                renewal_margin = min(float(threshold), effective_lease / 2.0)
                base_sleep = float(effective_lease) - renewal_margin
            except Exception as exc:  # noqa: BLE001 - configuration isolation boundary
                context._mark_renewal_lost()
                logger.bind(error_type=type(exc).__name__).warning(
                    "Jobs prepared renewal configuration failed"
                )
                return

            if not await context.ensure_lease_horizon(effective_lease):
                return
            iters += 1
            if self._max_iters and iters >= self._max_iters:
                return

            earlier_jitter = float(secrets.randbelow(jitter + 1)) if jitter else 0.0
            sleep_for = max(
                base_sleep / 2.0,
                base_sleep - earlier_jitter,
            )
            await self._sleep(sleep_for)
            if stop_requested.is_set():
                return

    async def _cleanup_prepared_renewal(
        self,
        renew_task: asyncio.Task[None],
        stop_requested: asyncio.Event,
    ) -> None:
        """Stop renewal and normalize every child outcome to normal completion."""

        stop_requested.set()
        renew_task.cancel()
        try:
            await renew_task
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - renewal isolation boundary
            logger.bind(error_type=type(exc).__name__).warning(
                "Jobs prepared renewal task failed"
            )

    async def _stop_prepared_renewal(
        self,
        renew_task: asyncio.Task[None],
        stop_requested: asyncio.Event,
    ) -> None:
        """Cancel and consume renewal while preserving new outer cancellation."""

        outer_cancellation: asyncio.CancelledError | None = None
        cleanup_task = asyncio.create_task(
            self._cleanup_prepared_renewal(renew_task, stop_requested)
        )
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                outer_cancellation = exc
        cleanup_task.result()
        if outer_cancellation is not None:
            raise outer_cancellation

    async def run_prepared(
        self,
        *,
        handler: PreparedJobHandler,
        pre_acquire_guard: PreAcquireGuard,
        handler_error_disposition: PreparedHandlerErrorDisposition,
        owner_user_id: str | None = None,
        job_type: str | None = None,
        on_disposition_applied: PreparedDispositionCallback | None = None,
        on_disposition_rejected: PreparedDispositionCallback | None = None,
    ) -> None:
        """Run a fail-closed worker loop with one exact typed disposition."""

        backoff = max(1, int(self.cfg.backoff_base_seconds))
        backoff_max = max(backoff, int(self.cfg.backoff_max_seconds))
        while not self._stop.is_set():
            try:
                guard_ok = await pre_acquire_guard()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - guard is an isolation boundary
                logger.bind(error_type=type(exc).__name__).warning(
                    "Jobs prepared pre-acquire guard failed"
                )
                guard_ok = False
            if not guard_ok:
                await self._sleep_chunked(float(min(backoff, backoff_max)))
                if self._stop.is_set():
                    break
                backoff = min(backoff * 2, backoff_max)
                continue
            if self._stop.is_set():
                break

            try:
                job = self.jm.acquire_next_job(
                    domain=self.cfg.domain,
                    queue=self.cfg.queue,
                    lease_seconds=self.cfg.lease_seconds,
                    worker_id=self.cfg.worker_id,
                    owner_user_id=owner_user_id,
                    job_type=job_type,
                )
            except Exception as exc:  # noqa: BLE001 - backend isolation boundary
                logger.bind(error_type=type(exc).__name__).warning(
                    "Jobs prepared acquisition failed"
                )
                job = None
            if not job:
                await self._sleep_chunked(float(min(backoff, backoff_max)))
                if self._stop.is_set():
                    break
                backoff = min(backoff * 2, backoff_max)
                continue
            backoff = max(1, int(self.cfg.backoff_base_seconds))

            acquired_job = copy.deepcopy(job)
            try:
                job_id = int(copy.deepcopy(acquired_job["id"]))
                domain = str(copy.deepcopy(acquired_job["domain"]))
                queue = str(copy.deepcopy(acquired_job["queue"]))
                job_type_name = str(copy.deepcopy(acquired_job["job_type"]))
                expected_payload = copy.deepcopy(acquired_job["payload"])
                worker_id = str(copy.deepcopy(self.cfg.worker_id))
                lease_id = str(copy.deepcopy(acquired_job["lease_id"]))
                context = WorkerExecutionContext(
                    self.jm,
                    copy.deepcopy(acquired_job),
                    worker_id=worker_id,
                )
            except Exception as exc:  # noqa: BLE001 - acquired row boundary
                logger.bind(error_type=type(exc).__name__).warning(
                    "Jobs prepared execution context failed"
                )
                continue

            renewal_stop = asyncio.Event()
            renew_task = asyncio.create_task(
                self._auto_renew_prepared(context, renewal_stop)
            )
            try:
                error_class: type[BaseException] | None = None
                try:
                    disposition = await handler(copy.deepcopy(acquired_job), context)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - handler isolation boundary
                    error_class = type(exc)
                    disposition = None

                if not isinstance(disposition, PreparedJobDisposition):
                    error_class = error_class or TypeError
                if error_class is not None:
                    try:
                        disposition = handler_error_disposition(
                            copy.deepcopy(acquired_job),
                            error_class,
                        )
                    except Exception as exc:  # noqa: BLE001 - factory boundary
                        logger.bind(error_type=type(exc).__name__).warning(
                            "Jobs prepared handler error disposition failed"
                        )
                        continue
                    if not isinstance(disposition, PreparedJobDisposition):
                        logger.bind(error_type=TypeError.__name__).warning(
                            "Jobs prepared handler error disposition failed"
                        )
                        continue

                try:
                    result = self.jm.apply_prepared_disposition(
                        ApplyPreparedDispositionCommand(
                            job_id=job_id,
                            domain=domain,
                            queue=queue,
                            job_type=job_type_name,
                            expected_payload=expected_payload,
                            worker_id=worker_id,
                            lease_id=lease_id,
                            disposition=disposition,
                        )
                    )
                    if not isinstance(result, PreparedDispositionResult) or not isinstance(
                        result.outcome,
                        OperationOutcome,
                    ):
                        raise TypeError
                except Exception as exc:  # noqa: BLE001 - typed apply boundary
                    logger.bind(error_type=type(exc).__name__).warning(
                        "Jobs prepared disposition apply failed"
                    )
                    continue

                if result.outcome is OperationOutcome.APPLIED:
                    if (
                        disposition.origin is PreparedDispositionOrigin.AUTHNZ
                        and on_disposition_applied is not None
                    ):
                        await self._invoke_prepared_disposition_callback(
                            on_disposition_applied,
                            copy.deepcopy(acquired_job),
                            disposition,
                            result,
                            callback_name="prepared-disposition-applied",
                        )
                elif on_disposition_rejected is not None:
                    await self._invoke_prepared_disposition_callback(
                        on_disposition_rejected,
                        copy.deepcopy(acquired_job),
                        disposition,
                        result,
                        callback_name="prepared-disposition-rejected",
                    )
            finally:
                await self._stop_prepared_renewal(renew_task, renewal_stop)

    async def run(
        self,
        *,
        handler: JobHandler,
        cancel_check: CancelCheck | None = None,
        progress_cb: Callable[[], dict[str, Any]] | None = None,
        acquire_guard: Callable[[dict[str, Any]], Awaitable[bool]] | None = None,
        owner_user_id: str | None = None,
        job_type: str | None = None,
        on_completed: CompletionCallback | None = None,
        on_completion_rejected: CompletionCallback | None = None,
        on_failed: FailureCallback | None = None,
    ) -> None:
        """Run the worker loop until stop() is called.

        handler should accept a job dict and return a result dict (or None) to finalize.
        Completion callbacks run after the durable completion attempt and receive
        the acquired job plus the normalized result.
        """
        backoff = max(1, int(self.cfg.backoff_base_seconds))
        backoff_max = max(backoff, int(self.cfg.backoff_max_seconds))
        enforce = self.jm.should_enforce_leases()
        while not self._stop.is_set():
            try:
                job = self.jm.acquire_next_job(
                    domain=self.cfg.domain,
                    queue=self.cfg.queue,
                    lease_seconds=self.cfg.lease_seconds,
                    worker_id=self.cfg.worker_id,
                    owner_user_id=owner_user_id,
                    job_type=job_type,
                )
            except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Acquire error: {e}")
                job = None
            if not job:
                # Sleep with backoff
                await self._sleep_chunked(float(min(backoff, backoff_max)))
                if self._stop.is_set():
                    break
                backoff = min(backoff * 2, backoff_max)
                continue
            backoff = max(1, int(self.cfg.backoff_base_seconds))

            job_id = int(job.get('id'))
            lease_id = job.get('lease_id')
            lease_id_str = str(lease_id) if lease_id is not None else None
            # Only start auto-renew after we know we will actually handle the job
            renew_task = None
            failure_job_items = tuple(job.items())

            async def _finalize_failure(
                exc: Exception,
                job_id: int = job_id,
                lease_id_str: str | None = lease_id_str,
                job_items: tuple[tuple[str, Any], ...] = failure_job_items,
            ) -> None:
                job_row = dict(job_items)
                retryable = self.cfg.retry_on_exception and bool(getattr(exc, "retryable", True))
                backoff_s = int(getattr(exc, "backoff_seconds", self.cfg.retry_backoff_seconds))
                error_code = str(getattr(exc, "failure_code", "worker_exception") or "worker_exception")
                try:
                    finalized = self.jm.fail_job(
                        job_id,
                        error=str(exc),
                        retryable=retryable,
                        backoff_seconds=backoff_s,
                        worker_id=self.cfg.worker_id,
                        lease_id=lease_id_str,
                        completion_token=(
                            lease_id_str
                            if self.cfg.bind_completion_token
                            or is_truthy(os.getenv("JOBS_REQUIRE_COMPLETION_TOKEN"))
                            else None
                        ),
                        enforce=enforce,
                        error_code=error_code,
                        error_class=type(exc).__name__,
                    )
                except _WORKER_SDK_NONCRITICAL_EXCEPTIONS:
                    logger.debug(f"Fail finalize error for job {job_id}")
                    return
                if not finalized or on_failed is None:
                    return
                try:
                    stored = self.jm.get_job_or_archived_by_uuid(
                        str(job_row.get("uuid") or ""),
                        domain=str(job_row.get("domain") or ""),
                        owner_user_id=str(job_row.get("owner_user_id") or ""),
                    )
                except _WORKER_SDK_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Failed to verify durable failure for job {}", job_id)
                    return
                exact_terminal_failure = bool(
                    stored
                    and stored.get("uuid") == job_row.get("uuid")
                    and stored.get("owner_user_id") == job_row.get("owner_user_id")
                    and stored.get("domain") == job_row.get("domain")
                    and stored.get("queue") == job_row.get("queue")
                    and stored.get("job_type") == job_row.get("job_type")
                    and stored.get("batch_group") == job_row.get("batch_group")
                    and stored.get("status") in {"failed", "quarantined"}
                    and (
                        stored.get("archived")
                        or stored.get("id") == job_row.get("id")
                    )
                )
                if exact_terminal_failure:
                    await self._invoke_completion_callback(
                        on_failed,
                        job_row,
                        exc,
                        callback_name="failed",
                    )

            try:
                if acquire_guard is not None:
                    try:
                        guard_ok = await acquire_guard(job)
                    except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug("Acquire guard failed for job {}: {}", job_id, exc)
                        guard_ok = True
                    if not guard_ok:
                        try:
                            self.jm.release_job(
                                job_id,
                                worker_id=self.cfg.worker_id,
                                lease_id=lease_id_str,
                                reason="guard_reject",
                                enforce=enforce,
                            )
                        except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as exc:
                            logger.debug("Release job failed for {}: {}", job_id, exc)
                        with contextlib.suppress(_WORKER_SDK_NONCRITICAL_EXCEPTIONS):
                            await self._sleep(0)
                        continue
                # Cancellation check (optional)
                if cancel_check is not None:
                    should_cancel = False
                    try:
                        should_cancel = await cancel_check(job)
                    except _WORKER_SDK_NONCRITICAL_EXCEPTIONS:
                        pass
                    if should_cancel:
                        # The acquired UUID and lease form a compare-and-set boundary:
                        # a stale worker must not cancel a reassigned or reused row.
                        try:
                            self.jm.finalize_cancelled(
                                job_id,
                                reason="requested",
                                expected_uuid=str(job.get("uuid") or ""),
                                worker_id=self.cfg.worker_id,
                                lease_id=lease_id_str,
                            )
                        except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as exc:
                            logger.debug("Cancel finalize error for job {}: {}", job_id, exc)
                        with contextlib.suppress(_WORKER_SDK_NONCRITICAL_EXCEPTIONS):
                            await self._sleep(0)
                        continue
                # Start auto-renew task only if not cancelled
                renew_task = asyncio.create_task(self._auto_renew(job, progress_cb=progress_cb))
                # Handle job
                try:
                    result = await handler(job)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Handler failures are expected control-flow for retry/fail semantics.
                    await _finalize_failure(exc)
                    continue
                if isinstance(result, WorkerTerminalOutcome):
                    try:
                        current = self.jm.resolve_slides_generation_job(
                            job_uuid=str(job.get("uuid") or ""),
                            owner_user_id=str(job.get("owner_user_id") or ""),
                            idempotency_key=str(job.get("idempotency_key") or ""),
                        )
                        if (
                            current is None
                            or any(
                                current.get(field) != job.get(field)
                                for field in (
                                    "uuid",
                                    "owner_user_id",
                                    "domain",
                                    "queue",
                                    "job_type",
                                )
                            )
                            or (not current.get("archived") and current.get("id") != job.get("id"))
                            or not _same_slides_jobs_key(
                                current.get("idempotency_key"),
                                job.get("idempotency_key"),
                            )
                        ):
                            raise WorkerTerminalizationConflict(f"terminal identity changed for job {job_id}")
                        current_status = current.get("status")
                        if current_status in {"failed", "cancelled", "quarantined"}:
                            continue
                        if current_status != "processing":
                            raise WorkerTerminalizationConflict(
                                f"terminal status changed to {current_status} for job {job_id}"
                            )
                        terminal_result = self.jm.terminalize_job_from_worker(
                            job_id=job_id,
                            job_uuid=str(job.get("uuid") or ""),
                            owner_user_id=str(job.get("owner_user_id") or ""),
                            domain=str(job.get("domain") or ""),
                            queue=str(job.get("queue") or ""),
                            job_type=str(job.get("job_type") or ""),
                            worker_id=self.cfg.worker_id,
                            lease_id=str(lease_id_str or ""),
                            completion_token=str(lease_id_str or ""),
                            status=result.status,
                            error_code=result.error_code,
                            error_message=result.message,
                        )
                    except WorkerTerminalizationConflict:
                        raise
                    except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as exc:
                        raise WorkerTerminalizationConflict(f"terminal CAS failed for job {job_id}") from exc
                    if terminal_result not in {"APPLIED", "IDEMPOTENT", "ALREADY_TERMINAL"}:
                        raise WorkerTerminalizationConflict(f"terminal CAS returned {terminal_result} for job {job_id}")
                    continue
                if result is None:
                    # No result; treat as success with empty result
                    result = {}
                ok = self.jm.complete_job(
                    job_id,
                    result=result,
                    worker_id=self.cfg.worker_id,
                    lease_id=lease_id_str,
                    completion_token=(
                        lease_id_str
                        if self.cfg.bind_completion_token
                        or is_truthy(os.getenv("JOBS_REQUIRE_COMPLETION_TOKEN"))
                        else None
                    ),
                    enforce=enforce,
                )
                if not ok:
                    logger.debug(f"Complete returned False for job {job_id}")
                    if on_completion_rejected is not None:
                        await self._invoke_completion_callback(
                            on_completion_rejected,
                            job,
                            result,
                            callback_name="completion-rejected",
                        )
                elif on_completed is not None:
                    await self._invoke_completion_callback(
                        on_completed,
                        job,
                        result,
                        callback_name="completed",
                    )
            except asyncio.CancelledError:
                raise
            except WorkerTerminalizationConflict:
                raise
            except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as e:
                await _finalize_failure(e)
            finally:
                if renew_task is not None:
                    renew_task.cancel()
                    try:
                        await renew_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:  # noqa: BLE001 - renewal is an isolation boundary
                        logger.debug("Auto-renew task failed for job {}: {}", job_id, exc)
