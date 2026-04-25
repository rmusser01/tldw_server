from __future__ import annotations

import asyncio
import contextlib
import os
import random
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.testing import is_test_mode, is_truthy

from .manager import JobManager

CancelCheck = Callable[[dict[str, Any]], Awaitable[bool]]
JobHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]

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
        # Detect test mode for more responsive sleeps and optional iteration caps
        try:
            self._test_mode = is_test_mode()
        except (TypeError, ValueError):
            self._test_mode = False
        try:
            self._max_iters = int(os.getenv("JOBS_WORKER_MAX_ITERATIONS", "0") or "0")
        except (TypeError, ValueError):
            self._max_iters = 0

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
        while not self._stop.is_set():
            # Sleep for lease - threshold, plus small jitter
            sleep_for = max(1, lease - threshold) + (random.randint(0, jitter) if jitter else 0)
            await self._sleep_chunked(float(sleep_for))
            if self._stop.is_set():
                return
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

    async def run(
        self,
        *,
        handler: JobHandler,
        cancel_check: CancelCheck | None = None,
        progress_cb: Callable[[], dict[str, Any]] | None = None,
        acquire_guard: Callable[[dict[str, Any]], Awaitable[bool]] | None = None,
        owner_user_id: str | None = None,
    ) -> None:
        """Run the worker loop until stop() is called.

        handler should accept a job dict and return a result dict (or None) to finalize.
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

            def _finalize_failure(exc: Exception) -> None:
                retryable = self.cfg.retry_on_exception and bool(getattr(exc, "retryable", True))
                backoff_s = int(getattr(exc, "backoff_seconds", self.cfg.retry_backoff_seconds))
                try:
                    self.jm.fail_job(
                        job_id,
                        error=str(exc),
                        retryable=retryable,
                        backoff_seconds=backoff_s,
                        worker_id=self.cfg.worker_id,
                        lease_id=lease_id_str,
                        completion_token=(lease_id_str if is_truthy(os.getenv("JOBS_REQUIRE_COMPLETION_TOKEN")) else None),
                        enforce=enforce,
                        error_code="worker_exception",
                        error_class=type(exc).__name__,
                    )
                except _WORKER_SDK_NONCRITICAL_EXCEPTIONS:
                    logger.debug(f"Fail finalize error for job {job_id}")

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
                    try:
                        if await cancel_check(job):
                            # Respect cancellation request; finalize and yield once to avoid tight spin
                            self.jm.cancel_job(job_id, reason="requested")
                            with contextlib.suppress(_WORKER_SDK_NONCRITICAL_EXCEPTIONS):
                                await self._sleep(0)
                            continue
                    except _WORKER_SDK_NONCRITICAL_EXCEPTIONS:
                        pass
                # Start auto-renew task only if not cancelled
                renew_task = asyncio.create_task(self._auto_renew(job, progress_cb=progress_cb))
                # Handle job
                try:
                    result = await handler(job)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Handler failures are expected control-flow for retry/fail semantics.
                    _finalize_failure(exc)
                    continue
                if result is None:
                    # No result; treat as success with empty result
                    result = {}
                ok = self.jm.complete_job(
                    job_id,
                    result=result,
                    worker_id=self.cfg.worker_id,
                    lease_id=lease_id_str,
                    completion_token=(lease_id_str if is_truthy(os.getenv("JOBS_REQUIRE_COMPLETION_TOKEN")) else None),
                    enforce=enforce,
                )
                if not ok:
                    logger.debug(f"Complete returned False for job {job_id}")
            except asyncio.CancelledError:
                raise
            except _WORKER_SDK_NONCRITICAL_EXCEPTIONS as e:
                _finalize_failure(e)
            finally:
                try:
                    if renew_task is not None:
                        renew_task.cancel()
                except _WORKER_SDK_NONCRITICAL_EXCEPTIONS:
                    pass
