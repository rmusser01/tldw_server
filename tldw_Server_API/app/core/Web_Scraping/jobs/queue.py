"""Single-event-loop ownership of bounded process-local scraping jobs.

Admission, cancellation and terminalization never suspend. Only injected execution
and physical drain suspend, outside lifecycle transitions. No shared worker is
cancelled to cancel one job, and retries never enqueue a second admission.
"""

from __future__ import annotations

import asyncio
import builtins
import contextvars
import heapq
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import count
from threading import Lock
from typing import Any

from loguru import logger

from ..contracts.execution import PageData
from ..runtime.controls import ParentControls, ParentStop, WorkHandle, positive_integer
from .admission import Admission, AdmissionLease, JobPayload, QueueRejected, SnapshotTooLarge
from .models import JobPriority, JobStatus, QueueLimits, QueueStopReport, ScrapingJob
from .projections import History

TERMINAL = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})
PageExecutor = Callable[[JobPayload, ParentControls | None], WorkHandle[PageData]]


def _observe(future: asyncio.Future) -> None:
    """Detached subscribers must not create unobserved-exception diagnostics."""
    if not future.cancelled():
        future.exception()


def _clear_exception_frames(error: BaseException) -> None:
    """Release execution frames without changing exception identities or links."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, builtins.BaseExceptionGroup):
            pending.extend(current.exceptions)


@dataclass(repr=False)
class _Entry:
    job: ScrapingJob
    payload: JobPayload | None
    controls: ParentControls | None
    context: contextvars.Context | None
    lease: AdmissionLease
    completion: asyncio.Future | None
    drained: asyncio.Future
    cancelled: asyncio.Future
    execute: PageExecutor | None
    subscribers: list[tuple[str, asyncio.Future]] = field(default_factory=list)
    handle: WorkHandle[PageData] | None = None
    started: bool = False
    cancel_sent: bool = False


class ProcessLocalJobQueue:
    """Bounded queue with separately owned logical completion and physical drain."""

    def __init__(
        self,
        *,
        execute: Callable[[JobPayload, ParentControls | None], WorkHandle[PageData]],
        max_concurrent: int = 5,
        completed_retention: int = 1000,
        limits: QueueLimits = QueueLimits(),
    ) -> None:
        positive_integer(max_concurrent, "max_concurrent")
        if type(completed_retention) is not int or completed_retention < 0:
            raise ValueError("completed_retention must be a nonnegative integer")
        if max_concurrent > min(limits.max_pending, limits.max_live):
            raise ValueError("worker count exceeds pending/live capacity")
        self.max_concurrent = max_concurrent
        self._execute = execute
        self._limits = limits
        self._admission = Admission(limits)
        self._history = History(completed_retention, limits.max_history_bytes)
        self._entries: dict[str, _Entry] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._sequence = count()
        self._tasks: set[asyncio.Task] = set()
        self._occupied = 0
        self._state = "new"
        self._loop: asyncio.AbstractEventLoop | None = None

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("queue belongs to a different event loop")
        return loop

    async def start(self) -> None:
        """Start once; stopping/stopped queues cannot reopen admission."""
        self._bind_loop()
        if self._state == "new":
            self._state = "running"
            self._pump()

    @property
    def producer_window(self) -> int:
        """Read-only production bound; neither an admission grant nor a waiter."""
        return min(self.max_concurrent, self._limits.max_pending, self._limits.max_live)

    def _admit(
        self, job: ScrapingJob, controls: ParentControls | None, kind: str, execute: PageExecutor | None = None
    ) -> tuple[_Entry, asyncio.Future]:
        loop = self._bind_loop()
        if self._state in {"stopping", "stopped"}:
            raise QueueRejected("queue_stopping")
        if type(job.job_id) is not str or job.job_id in self._entries:
            raise ValueError("invalid or duplicate live job id")
        try:
            payload = JobPayload.from_job(job, self._limits.max_payload_bytes)
        except SnapshotTooLarge:
            raise QueueRejected("job_payload_too_large") from None
        lease = self._admission.acquire(payload.size_bytes)
        # All subscriptions are created before this entry can become runnable.
        completion, drained, subscriber = loop.create_future(), loop.create_future(), loop.create_future()
        cancelled = loop.create_future()
        for future in (completion, drained, subscriber, cancelled):
            future.add_done_callback(_observe)
        public = replace(job, metadata={}, context=None, result=None, error=None)
        entry = _Entry(
            public,
            payload,
            controls,
            contextvars.copy_context(),
            lease,
            completion,
            drained,
            cancelled,
            self._execute if execute is None else execute,
            [(kind, subscriber)],
        )
        self._history.remove(job.job_id)
        self._entries[job.job_id] = entry
        heapq.heappush(self._heap, (-public.priority.value, next(self._sequence), public.job_id))
        self._pump()
        return entry, subscriber

    async def add_job(self, job: ScrapingJob) -> asyncio.Future[dict[str, Any]]:
        """Immediate admission; cancelling this future detaches only its caller."""
        _, subscriber = self._admit(job, None, "article")
        return subscriber

    async def submit_page(
        self, job: ScrapingJob, *, controls: ParentControls, execute: PageExecutor | None = None
    ) -> WorkHandle[PageData]:
        """Atomically subscribe to live page/discovery and this job's own drain."""
        entry, subscriber = self._admit(job, controls, "page", execute)
        loop = self._bind_loop()
        drained = loop.create_future()
        drained.add_done_callback(_observe)

        def deliver_drain(_future: asyncio.Future) -> None:
            if not drained.done():
                drained.set_result(None)

        entry.drained.add_done_callback(deliver_drain)
        cancel_lock = Lock()
        requested = False

        def cancel() -> None:
            nonlocal requested
            with cancel_lock:
                if requested:
                    return
                requested = True
            loop.call_soon_threadsafe(self._cancel_entry, entry)

        return WorkHandle(subscriber, drained, cancel)

    def _pump(self) -> None:
        while self._state == "running" and self._heap and self._occupied < self.max_concurrent:
            _, _, job_id = heapq.heappop(self._heap)
            entry = self._entries[job_id]
            if entry.job.cancel_requested:
                self._cancel_entry(entry)
                continue
            entry.started = True
            entry.job.status = JobStatus.IN_PROGRESS
            entry.job.started_at = datetime.now()
            if entry.job.retries >= entry.job.max_retries:
                entry.lease.release_pending()
            self._occupied += 1
            task = asyncio.create_task(self._run(entry), context=entry.context)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(_observe)

    async def _run(self, entry: _Entry) -> None:
        retry = False
        try:
            if entry.job.status in TERMINAL:
                return
            if entry.controls is not None:
                entry.controls.check()
            if entry.payload is None or entry.execute is None:
                raise RuntimeError("execution payload already released")
            entry.handle = entry.execute(entry.payload, entry.controls)
            entry.handle.result.add_done_callback(_observe)
            entry.handle.drained.add_done_callback(_observe)
            await asyncio.wait((entry.handle.result, entry.cancelled), return_when=asyncio.FIRST_COMPLETED)
            if entry.job.status in TERMINAL:
                return
            value = entry.handle.result.result()
            if not isinstance(value, PageData) or type(value.article) is not dict:
                raise ValueError("executor did not return PageData")
            article = value.article
            if article.get("error") or article.get("extraction_successful") is False:
                entry.job.error = "source_extraction_failed"
                retry = entry.job.retries < entry.job.max_retries and self._state == "running"
                if not retry:
                    self._terminal(
                        entry, JobStatus.FAILED, value, Exception(article.get("error") or "Source extraction failed")
                    )
            else:
                self._terminal(entry, JobStatus.COMPLETED, value)
        except asyncio.CancelledError:
            self._terminal(entry, JobStatus.CANCELLED)
        except Exception as error:  # noqa: BLE001 - injected execution boundary; settle every accepted subscription
            self._terminal(entry, JobStatus.FAILED, error=error)
        finally:
            if entry.handle is not None:
                # Cancellation and logical timeout must not cancel this physical future.
                await asyncio.shield(entry.handle.drained)
            self._occupied -= 1
            entry.started = False
            entry.handle = None
            if retry and entry.job.status not in TERMINAL and self._state == "running":
                entry.job.retries += 1
                entry.job.status = JobStatus.PENDING
                heapq.heappush(self._heap, (-entry.job.priority.value, next(self._sequence), entry.job.job_id))
            else:
                self._release(entry)
            self._pump()

    def _terminal(
        self,
        entry: _Entry,
        status: JobStatus,
        value: PageData | None = None,
        error: Exception | None = None,
        *,
        shutdown: bool = False,
    ) -> None:
        if entry.job.status in TERMINAL:
            return
        entry.job.status = status
        entry.job.completed_at = datetime.now()
        if status is JobStatus.CANCELLED:
            if not entry.cancelled.done():
                entry.cancelled.set_result(None)
            entry.job.cancel_requested = True
            entry.job.error = "Job cancelled due to shutdown" if shutdown else "Job cancelled"
            error = Exception(entry.job.error)
        elif error is not None:
            entry.job.error = (
                "source_extraction_failed"
                if value is not None
                else (error.reason if isinstance(error, ParentStop) else "execution_failed")
            )
            # Causes, contexts and group members can hold the same private frames.
            _clear_exception_frames(error)
        else:
            entry.job.error = None
        self._history.record(entry.job, value.article if value else None)
        targets = [("page", entry.completion), *entry.subscribers]
        entry.subscribers.clear()
        entry.completion = None
        for kind, future in targets:
            if future is None or future.done():
                continue
            if value is not None and (status is JobStatus.COMPLETED or kind == "page"):
                # Internal crawl subscribers need the failed page's compatibility
                # payload/sidecar after retries; legacy article futures still fail.
                future.set_result(value if kind == "page" else value.article)
            elif kind == "page" and status is JobStatus.CANCELLED:
                future.cancel()
            else:
                future.set_exception(error or Exception("Job failed"))

    def _cancel_handle(self, entry: _Entry) -> None:
        if entry.handle is not None and not entry.cancel_sent:
            entry.cancel_sent = True
            try:
                entry.handle.cancel()
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 - isolate untrusted cancellation callbacks
                # Callback failures are secondary; never print private exception text.
                logger.warning("queue_cancel_callback_failed")

    def _cancel_entry(self, entry: _Entry, *, shutdown: bool = False) -> None:
        if self._entries.get(entry.job.job_id) is not entry:
            return
        if entry.job.status in TERMINAL:
            # Explicit ownership can still interrupt physical work after logical
            # delivery, without rewriting the already committed public outcome.
            self._cancel_handle(entry)
            return
        self._terminal(entry, JobStatus.CANCELLED, shutdown=shutdown)
        self._cancel_handle(entry)
        if not entry.started:
            self._heap = [item for item in self._heap if item[2] != entry.job.job_id]
            heapq.heapify(self._heap)
            self._release(entry)

    def _release(self, entry: _Entry) -> None:
        if entry.lease.released:
            return
        entry.lease.release()
        self._entries.pop(entry.job.job_id, None)
        entry.payload = None
        entry.controls = None
        entry.context = None
        entry.execute = None
        if not entry.drained.done():
            entry.drained.set_result(None)
        if self._state == "stopping" and not self._entries:
            self._state = "stopped"

    async def cancel_job(self, job_id: str) -> ScrapingJob | None:
        """Cancel only this job; never a worker or another caller's children."""
        self._bind_loop()
        entry = self._entries.get(job_id)
        if entry is None:
            return self.get_job(job_id)
        self._cancel_entry(entry)
        return deepcopy(entry.job)

    async def stop(self) -> QueueStopReport:
        """Close admission before cancelling pending/active owned jobs, without waiting."""
        self._bind_loop()
        self._state = "stopping"
        for entry in tuple(self._entries.values()):
            self._cancel_entry(entry, shutdown=True)
            # Already-terminal physical work also receives cooperative shutdown.
            self._cancel_handle(entry)
        if not self._entries:
            self._state = "stopped"
        return QueueStopReport(self._state, len(self._entries))

    def get_job(self, job_id: str) -> ScrapingJob | None:
        """Return only a safe public copy; no payload or controls escape polling."""
        retained = self._history.get(job_id)
        if retained is not None:
            return retained
        entry = self._entries.get(job_id)
        return deepcopy(entry.job) if entry is not None else None

    def get_status(self) -> dict[str, Any]:
        """Public counts plus bounded ownership diagnostics for later adapters."""
        return {
            "active_jobs": self._occupied,
            "completed_jobs": len(self._history.entries),
            "pending_by_priority": {
                priority.name: sum(item[0] == -priority.value for item in self._heap) for priority in JobPriority
            },
            "state": self._state,
            "live_jobs": self._admission.live,
            "pending_reservations": self._admission.pending,
            "payload_bytes": self._admission.payload_bytes,
            "history_bytes": self._history.bytes,
        }
