"""Lazy bounded batch production and atomic, privately subscribed page ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..contracts.execution import CrawlOutcome, CrawlReport, PageData, successful_article
from ..crawl.models import CrawlPlan, FrontierEntry
from ..jobs.admission import QueueRejected
from ..jobs.models import ScrapingJob
from ..jobs.queue import PageExecutor, ProcessLocalJobQueue
from ..runtime.controls import ParentStop, WorkHandle
from .crawl_context import CrawlContext

_CLEANUP_SECONDS = 2.0


async def queued_page(
    entry: FrontierEntry,
    *,
    context: CrawlContext,
    queue: ProcessLocalJobQueue,
    make_job: Callable[[str], ScrapingJob],
    execute: PageExecutor | None = None,
    owned: Iterable[WorkHandle[PageData]] = (),
) -> WorkHandle[PageData]:
    """Admit and subscribe without a scheduling gap or polling public history."""
    context.check()
    job = make_job(entry.url)
    draining = {handle.drained for handle in owned if not handle.drained.done()}
    while True:
        context.check()
        try:
            handle = await queue.submit_page(job, controls=context, execute=execute)
            break
        except QueueRejected as error:
            if error.code != "queue_overloaded" or not draining:
                raise
            done = await _wait_owned(draining, context)
            draining.difference_update(done)
    context.attach(handle)
    return handle


@dataclass
class _Child:
    url: str
    handle: WorkHandle[PageData]
    collected: bool = False


def _collect(children: dict[int, _Child], articles: dict[int, dict[str, Any]], context: CrawlContext | None) -> None:
    """Commit completed compatibility payloads; actual data is a separate flag."""
    for index, child in children.items():
        if child.collected or not child.handle.result.done():
            continue
        try:
            value = child.handle.result.result().article
        except ParentStop:
            if context is not None:
                raise
            continue
        except asyncio.CancelledError:
            # A private child subscription is cancelled by explicit cancel_job.
            # That is an item failure, not cancellation of this task or parent.
            value = {"url": child.url, "error": "Job cancelled", "extraction_successful": False}
        except Exception:  # noqa: BLE001 - safe per-input failure, never private exception prose
            value = {"url": child.url, "error": "execution_failed", "extraction_successful": False}
        if context is not None:
            context.check()
            if successful_article(value):
                context.note_successful_result()
        # No await between the actual-data flag and logical payload commit.
        articles[index] = value
        child.collected = True


async def _wait_owned(futures: set[asyncio.Future], context: CrawlContext) -> set[asyncio.Future]:
    """Wake on owned progress or parent stop; neither cancel futures nor infer drain."""
    stop = context.stop_signal()
    done, _ = await asyncio.wait(
        futures | {stop}, timeout=context.cap_timeout(None), return_when=asyncio.FIRST_COMPLETED
    )
    context.check()
    if not done:
        raise ParentStop("deadline_exhausted")
    return done - {stop}


async def _progress(children: dict[int, _Child], context: CrawlContext) -> None:
    """Wait only on accepted owned progress, never on a competing queue occupant."""
    futures = {child.handle.drained if child.collected else child.handle.result for child in children.values()}
    await _wait_owned(futures, context)


def _cancel_children(children: dict[int, _Child]) -> None:
    """Cancel only this producer's accepted physical-live entries."""
    for child in children.values():
        if not child.handle.drained.done():
            child.handle.cancel()


async def _cleanup(children: dict[int, _Child], *, abort: bool) -> asyncio.CancelledError | None:
    """One aggregate allowance; the queue/context retain physical ownership later."""
    if abort:
        _cancel_children(children)
    pending = [asyncio.shield(child.handle.drained) for child in children.values() if not child.handle.drained.done()]
    if pending:
        try:
            await asyncio.wait_for(asyncio.gather(*pending), _CLEANUP_SECONDS)
        except TimeoutError:
            # Only shield proxies are cancelled. Actual drain and admission leases
            # are still owned by the queue, not abandoned by this logical return.
            pass
        except asyncio.CancelledError as error:
            _cancel_children(children)
            return error
    return None


async def execute_batch(
    urls: Iterable[str],
    *,
    plan: CrawlPlan,
    context: CrawlContext,
    queue: ProcessLocalJobQueue,
    make_job: Callable[[str], ScrapingJob],
    execute: PageExecutor | None = None,
) -> CrawlReport:
    """Fill a finite owned window lazily and return caller-required input order.

    Queue pressure permits one retry only after an owned completion/drain event;
    no ownerless admission waiters, task-per-input fanout or second payload list.
    """
    children: dict[int, _Child] = {}
    articles: dict[int, dict[str, Any]] = {}
    source = None
    next_job = None
    next_url = None
    index = 0
    exhausted = False
    reason = "frontier_exhausted"
    cancellation = None
    try:
        context.check()
        source = iter(urls)
        while True:
            context.check()
            _collect(children, articles, context)
            children = {
                key: child for key, child in children.items() if not (child.collected and child.handle.drained.done())
            }
            remaining = plan.limits.page_attempts - context.snapshot().page_attempts
            window = min(queue.producer_window, remaining)
            while not exhausted and len(children) < window:
                context.check()
                if next_job is None:
                    if next_url is None:
                        try:
                            next_url = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                    next_job = make_job(next_url)
                try:
                    handle = await queue.submit_page(next_job, controls=context, execute=execute)
                except QueueRejected as error:
                    if error.code != "queue_overloaded" or not children:
                        raise
                    break
                context.attach(handle)
                children[index] = _Child(next_url, handle)
                index += 1
                next_job = next_url = None
            if not children:
                if exhausted:
                    break
                if remaining <= 0:
                    # A single lookahead distinguishes exact-limit completion
                    # from truncation. Never construct an over-budget job.
                    if next_job is None and next_url is None:
                        try:
                            next(source)
                        except StopIteration:
                            break
                    raise ParentStop("budget_exhausted")
            if all(child.collected for child in children.values()):
                # Once logical progress is collected, a single URL lookahead can
                # identify the last window without materializing another job or
                # waiting for final physical drain under the execution deadline.
                if not exhausted and next_job is None and next_url is None:
                    try:
                        next_url = next(source)
                    except StopIteration:
                        exhausted = True
                if exhausted:
                    break
            await _progress(children, context)
    except asyncio.CancelledError as error:
        cancellation, reason = error, "cancelled"
    except ParentStop as error:
        reason = error.reason
    except QueueRejected as error:
        reason = error.code
    except Exception:  # noqa: BLE001 - input factory/execution boundary, safe diagnostics only
        reason = "execution_failed"
    finally:
        if reason != "frontier_exhausted":
            _collect(children, articles, None)
        cleanup_cancel = await _cleanup(children, abort=reason != "frontier_exhausted")
        if cleanup_cancel is not None and cancellation is None:
            cancellation, reason = cleanup_cancel, "cancelled"
        _collect(children, articles, None)
        # Retained cancellation tracebacks must not own execution capabilities.
        children.clear()
        source = next_job = next_url = execute = make_job = urls = queue = handle = None
    values = tuple(articles[key] for key in sorted(articles))
    normal = reason == "frontier_exhausted"
    outcome = context.publish(
        CrawlOutcome("completed", reason, False) if normal else CrawlReport.interrupted(values, reason).outcome
    )
    report = CrawlReport(values, outcome)
    if outcome.state == "cancelled" and cancellation is None:
        cancellation = asyncio.CancelledError()
    if cancellation is not None:
        cancellation.report = report
        raise cancellation
    return report
