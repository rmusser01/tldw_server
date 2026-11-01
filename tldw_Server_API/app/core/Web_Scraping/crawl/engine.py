"""Bounded profile traversal over injected acquisition and policy protocols."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from time import monotonic
from types import MappingProxyType
from typing import Any, TypeVar
from urllib.parse import urlsplit

from ..contracts.execution import CrawlOutcome, CrawlReport, DiscoveryResult, PageData, successful_article
from ..runtime.controls import ParentStop, PhysicalWork, WorkHandle, owned_to_thread, physical_work
from .discovery import charge, url_bytes
from .frontier import Frontier
from .models import (
    CheckpointPort,
    CheckpointState,
    CrawlControls,
    CrawlHooks,
    CrawlPlan,
    FrontierEntry,
    TraversalDecision,
    TraversalPolicy,
)

_RECURSIVE = {"legacy_recursive", "enhanced_recursive", "enhanced_url_level"}
_CLEANUP_SECONDS = 2.0
_T = TypeVar("_T")


def _checkpoint_operation(
    context: CrawlControls, function: Callable[..., _T], *args: Any, **kwargs: Any
) -> WorkHandle[_T]:
    """Register before default-pool dispatch, including under eager factories."""
    owner = PhysicalWork()
    result = owner.loop.create_future()
    # File operations are not cancelable once dispatched. Their real completion
    # (not this awaiter's state) releases the port and private crawl graph.
    handle = WorkHandle(result, owner.drained, lambda: None)
    context.attach(handle)

    async def run():
        token = physical_work.set(owner)
        try:
            value = await owned_to_thread(function, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - preserve cancellation on the owned result
            result.set_exception(exc)
        else:
            result.set_result(value)
        finally:
            physical_work.reset(token)

    def completed(done):
        # A task canceled before its first step never enters run's finally.
        # Its initial registration still has to drain, even in that case.
        if done.cancelled() and not result.done():
            result.set_exception(asyncio.CancelledError())
        elif not done.cancelled():
            done.exception()
        owner.finish()

    operation = run()
    try:
        task = asyncio.create_task(operation, name="crawl-checkpoint-operation")
    except BaseException:
        operation.close()
        owner.finish()
        raise
    task.add_done_callback(completed)
    result.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    return handle


def _retained(plan: CrawlPlan, values: tuple[dict, ...], entries: tuple[FrontierEntry, ...] = ()) -> tuple[dict, ...]:
    """Compatibility retention, deliberately distinct from actual-data success."""
    if plan.profile in {"enhanced_recursive", "enhanced_url_level"}:
        return tuple(item for item in values if item.get("extraction_successful"))
    if plan.profile == "legacy_recursive":
        return tuple(
            item
            for item in values
            if item.get("extraction_successful") or item.get("error") == "browser_transport_unavailable"
        )
    if plan.profile == "legacy_url_level" and entries:
        # The old collector returns a visited set, then filters it for extraction.
        # Include non-extracted visits when reproducing that set iteration order.
        visited, articles = set(), {}
        for entry, item in zip(entries, values):
            visited.add(entry.url)
            if item:
                articles[entry.url] = item
        return tuple(articles[url] for url in visited if url in articles)
    return tuple(item for item in values if item)


async def _bounded(operation: Awaitable, context: CrawlControls):
    """Bound policy waits and shielded results by the active parent deadline."""
    try:
        return await asyncio.wait_for(operation, context.cap_timeout(None))
    except TimeoutError:
        context.check()
        raise ParentStop("deadline_exhausted") from None


async def _readmit_restored(
    entry: FrontierEntry, plan: CrawlPlan, evaluate: Callable[..., Awaitable[TraversalDecision]]
) -> TraversalDecision:
    """Stored dispositions are not grants to bypass current profile admission."""
    if entry.url == plan.request.base_url:
        return await evaluate(entry, "seed")
    if plan.profile == "legacy_recursive":
        source, target = urlsplit(plan.request.base_url), urlsplit(entry.url)
        if (source.scheme, source.hostname, source.port) != (target.scheme, target.hostname, target.port):
            return TraversalDecision(False, reason="filtered")
    depth = plan.options.get("max_depth")
    if depth is not None and entry.depth > depth:
        return TraversalDecision(False, reason="filtered")
    return await evaluate(entry, "discovery")


async def _within_cleanup(operation: Awaitable[_T], deadline: float) -> _T:
    """Spend only the remaining aggregate finalization allowance."""
    return await asyncio.wait_for(operation, max(0.0, deadline - monotonic()))


async def _checkpoint_call(
    context: CrawlControls,
    handles: list[WorkHandle[Any]],
    function: Callable[..., _T],
    *args: Any,
    loading: bool = False,
    background: bool = False,
    cleanup_deadline: float | None = None,
    **kwargs: Any,
) -> _T:
    """Await an owned file operation without conflating cancellation and drain."""
    if not loading and not background and cleanup_deadline is None:
        raise ValueError("checkpoint cleanup deadline required")
    owned = _checkpoint_operation(context, function, *args, **kwargs)
    handles.append(owned)

    async def wait(future):
        operation = asyncio.shield(future)
        if background:
            return await operation
        if loading:
            return await _bounded(operation, context)
        return await _within_cleanup(operation, cleanup_deadline)

    try:
        value = await wait(owned.result)
    except asyncio.CancelledError:
        if owned.result.done() and not asyncio.current_task().cancelling():
            # Port cancellation is a checkpoint error, not parent authority.
            raise ValueError("checkpoint_failed") from None
        raise
    # A failed/canceled result leaves physical ownership with the registered
    # handle. Successful result/drain waits spend the same deadline as each other.
    await wait(owned.drained)
    # A resolved future can precede its scheduled release callback. Prune through
    # the owner's structural API, not executor-global or private context state.
    context.owned_handles()
    return value


async def _release_crawl(
    context: CrawlControls,
    handles: list[WorkHandle[Any]],
    checkpoint: CheckpointPort | None,
    continuation: Callable[[], CheckpointState] | None,
    owned: tuple[object, ...],
) -> None:
    """Retain the private graph and writer until every actual operation exits."""
    await asyncio.gather(*(asyncio.shield(item.drained) for item in handles if not item.drained.done()))
    if checkpoint is not None:
        try:
            if continuation is not None:
                await _checkpoint_call(context, handles, checkpoint.save, continuation(), background=True)
        finally:
            await asyncio.gather(*(asyncio.shield(item.drained) for item in handles if not item.drained.done()))
            await _checkpoint_call(context, handles, checkpoint.close, background=True)


async def _finalize_crawl(
    plan: CrawlPlan,
    *,
    context: CrawlControls,
    frontier: Frontier,
    handles: list[WorkHandle[Any]],
    checkpoint: CheckpointPort | None,
    loading: bool,
    count: int,
    reason: str,
    cancellation: asyncio.CancelledError | None,
    diagnostics: list[str],
    owned: tuple[object, ...] | None,
) -> tuple[CrawlReport, asyncio.CancelledError | None]:
    """Finalize checkpoint and writer ownership before the single publication."""
    cleanup_deadline = monotonic() + _CLEANUP_SECONDS
    # Policy/robots work may outlive awaiters without ever acquiring a page.
    handles = list({id(item): item for item in (*handles, *context.owned_handles())}.values())
    normal = reason in {"frontier_exhausted", "page_cap_reached"}
    if not normal and cancellation is None:
        for handle in handles:
            try:
                handle.cancel()
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - preserve primary stop
                diagnostics.append("execution_failed")
    snapshot = frontier.snapshot()
    deferred_save = False

    def continuation():
        budget = context.snapshot()
        return CheckpointState(
            checkpoint.scope,
            plan,
            snapshot,
            budget,
            reason,
            budget.accounting_complete and not any(not item.drained.done() for item in handles),
            pages_scraped=count,
        )

    if checkpoint is not None and not loading:
        try:
            if normal:
                # Bound logical cleanup but retain writer/private graph if actual
                # drain is delayed. No normal publication before primary failure.
                draining = tuple(item.drained for item in handles if not item.drained.done())
                if draining:
                    await _within_cleanup(
                        asyncio.gather(*(asyncio.shield(future) for future in draining)), cleanup_deadline
                    )
                await _checkpoint_call(context, handles, checkpoint.clear, cleanup_deadline=cleanup_deadline)
            else:
                await _checkpoint_call(
                    context, handles, checkpoint.save, continuation(), cleanup_deadline=cleanup_deadline
                )
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation, reason = exc, "cancelled"
                context.cancel()
            deferred_save = True
        except Exception:  # noqa: BLE001 - secondary failure cannot mask stop
            diagnostics.append("checkpoint_failed")
            if normal:
                reason, deferred_save = "checkpoint_failed", True
    pending = tuple(item.drained for item in handles if not item.drained.done())
    if pending or deferred_save:
        task = asyncio.create_task(
            _release_crawl(context, handles, checkpoint, continuation if deferred_save else None, owned),
            name="crawl-owned-drain",
        )
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    elif checkpoint is not None:
        try:
            await _checkpoint_call(context, handles, checkpoint.close, cleanup_deadline=cleanup_deadline)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation, reason = exc, "cancelled"
                context.cancel()
        except Exception:  # noqa: BLE001 - preserve cancellation/exhaustion
            diagnostics.append("checkpoint_failed")
            if normal:
                reason = "checkpoint_failed"
        if any(not item.drained.done() for item in handles):
            # A timed-out/canceled close is already dispatched. Retain its graph
            # until actual return, without releasing or closing the port twice.
            task = asyncio.create_task(_release_crawl(context, handles, None, None, owned), name="crawl-close-drain")
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    owned = None
    articles = _retained(plan, snapshot.results, snapshot.completed)
    normal = reason in {"frontier_exhausted", "page_cap_reached"}
    outcome = context.publish(
        CrawlOutcome("completed", reason, False) if normal else CrawlReport.interrupted(articles, reason).outcome
    )
    return CrawlReport(articles, outcome, tuple(diagnostics)), cancellation


async def run_crawl(
    plan: CrawlPlan,
    *,
    context: CrawlControls,
    acquire: Callable[[FrontierEntry], Awaitable[WorkHandle[PageData]]],
    checkpoint: CheckpointPort | None = None,
    hooks: CrawlHooks | None = None,
    seeds: Iterable[str] | None = None,
    policy: TraversalPolicy | None = None,
) -> CrawlReport:
    """Evaluate expansion before atomic parent/result/frontier disposition.

    Production supplies policy. Acquisition alone owns page attempts and retries.
    """
    frontier = Frontier(plan)
    handles: list[WorkHandle[PageData]] = []
    reason, diagnostics = "frontier_exhausted", []
    cancellation = None
    count = sequence = 0
    recursive = plan.profile in _RECURSIVE
    legacy = plan.profile == "legacy_recursive"
    maximum = plan.options.get("max_pages") if recursive else None
    depth_limit = plan.options.get("max_depth") if recursive else None
    loading = checkpoint is not None
    source = iter((plan.request.base_url,) if seeds is None else seeds)
    input_done = False
    restored_sequences = frozenset()

    async def evaluate(entry, phase):
        context.check()
        decision = await _bounded(policy.evaluate(entry, phase=phase), context) if policy else TraversalDecision()
        context.check()
        return decision

    def commit_denial(entry, decision):
        # Orchestration supplies the profile's established safe failure payload;
        # pure policy fakes can omit details, but cannot manufacture success data.
        failure = {**dict(decision.metadata), "url": entry.url, "extraction_successful": False}
        failure.setdefault("error", decision.reason)
        context.check()
        frontier.commit(entry, PageData(failure, DiscoveryResult(entry.url, entry.url, ())))
        notify("in_progress", entry)

    def notify(status, entry=None):
        if hooks and hooks.on_progress:
            hooks.on_progress(
                MappingProxyType(
                    {
                        "status": status,
                        "pages_scraped": count,
                        "current_url": entry.url if entry else None,
                        "queue_size": frontier.queued_count,
                        "visited": frontier.completed_count,
                    }
                )
            )

    try:
        state = None
        if checkpoint is not None:
            if seeds is not None or plan.profile == "enhanced_batch":
                raise ValueError("live multi-input continuation is unsupported")
            state = await _checkpoint_call(
                context, handles, checkpoint.load, plan=plan, scope=checkpoint.scope, loading=True
            )
            if state is not None:
                if state.scope != checkpoint.scope or state.plan != plan or state.schema_version != 1:
                    raise ValueError("checkpoint binding mismatch")
                context.restore_budget(state.budget, accounting_complete=state.accounting_complete)
                frontier = Frontier.from_snapshot(plan, state.frontier)
                frontier.restore_unfinished()
                restored_sequences = frozenset(
                    item.sequence for item in state.frontier.queued + state.frontier.inflight
                )
                count = (
                    state.pages_scraped
                    if state.pages_scraped is not None
                    else sum(bool(item.get("extraction_successful")) for item in state.frontier.results)
                )
                sequence = (
                    max(
                        (
                            item.sequence
                            for item in state.frontier.queued + state.frontier.inflight + state.frontier.completed
                        ),
                        default=-1,
                    )
                    + 1
                )
                input_done = True
                if any(successful_article(item) for item in _retained(plan, state.frontier.results)):
                    context.note_successful_result()
        loading = False
        while True:
            context.check()
            at_cap = maximum is not None and count >= maximum
            if not frontier.queued_count and not input_done and (not at_cap or sequence == 0):
                try:
                    url = next(source)
                except StopIteration:
                    input_done = True
                else:
                    charge(context, "discovered_candidates")
                    charge(context, "discovered_url_bytes", url_bytes(url, plan))
                    entry = FrontierEntry(url, None, 0, sequence, 0.0)
                    sequence += 1
                    if not frontier.admit(entry):
                        continue
                    decision = await evaluate(entry, "seed")
                    if not decision.allowed:
                        if plan.profile == "enhanced_batch":
                            frontier.take()
                            commit_denial(entry, decision)
                            continue
                        reason = (
                            decision.reason
                            if decision.reason in {"policy_denied", "transport_unavailable", "execution_failed"}
                            else "policy_denied"
                        )
                        if plan.profile == "legacy_recursive" and decision.reason == "transport_unavailable":
                            frontier.take()
                            context.check()
                            frontier.commit(
                                entry, PageData(dict(decision.metadata), DiscoveryResult(entry.url, entry.url, ()))
                            )
                        break
                    frontier.prioritize(entry, decision.ordering_score)
            if at_cap:
                reason = "page_cap_reached"
                break
            # Preserve enhanced traversal's ten-item selection window; newly
            # discovered descendants cannot jump ahead of selected siblings.
            window = min(10, maximum - count) if recursive and not legacy and maximum is not None else 1
            batch = []
            while len(batch) < window and (entry := frontier.take()) is not None:
                batch.append(entry)
            if not batch:
                if input_done:
                    break
                continue
            for entry in batch:
                context.check()
                if maximum is not None and count >= maximum:
                    reason = "page_cap_reached"
                    break
                if entry.sequence in restored_sequences:
                    admission = await _readmit_restored(entry, plan, evaluate)
                    context.check()
                    if not admission.allowed:
                        reason = "policy_denied" if admission.reason == "filtered" else admission.reason
                        break
                visit = await evaluate(entry, "visit")
                if not visit.allowed:
                    independent = plan.profile == "enhanced_batch" or (
                        plan.profile.endswith("sitemap") and entry.parent_url is not None
                    )
                    if independent:
                        commit_denial(entry, visit)
                        continue
                    reason = visit.reason if visit.reason != "filtered" else "policy_denied"
                    break
                if legacy:
                    if hooks and hooks.progress_callback:
                        hooks.progress_callback(f"Scraping page {count + 1}/{maximum}: {entry.url}")
                    delay = plan.options.get("delay", 1.0)
                    if delay:
                        await _bounded(asyncio.sleep(delay), context)
                # Immediate admission is awaited directly: no proxy task can
                # strand a just-accepted handle before ownership attaches.
                handle = await acquire(entry)
                context.attach(handle)
                handles = [item for item in handles if not item.drained.done()]
                handles.append(handle)
                page = await _bounded(asyncio.shield(handle.result), context)
                article = dict(page.article)
                if article and visit.metadata:
                    article["metadata"] = {**dict(article.get("metadata") or {}), **dict(visit.metadata)}
                page = PageData(article, page.discovery)
                success = bool(article.get("extraction_successful"))
                next_count = count + int(success)
                collection_root = plan.profile.endswith("sitemap") and not article
                expands = collection_root or plan.profile == "legacy_url_level" or (recursive and (legacy or success))
                expands &= depth_limit is None or entry.depth < depth_limit
                expands &= maximum is None or next_count < maximum
                children, identities = [], set()
                if expands:
                    remaining = maximum - next_count if maximum is not None and not legacy else None
                    for link in page.discovery.links:
                        if remaining is not None and remaining <= 0:
                            break
                        child = FrontierEntry(
                            link.url, link.parent_url, 0 if collection_root else entry.depth + 1, sequence, 0.0
                        )
                        sequence += 1
                        identity = frontier._identity(child)
                        if frontier.contains(child) or identity in identities:
                            continue
                        decision = await evaluate(child, "discovery")
                        if not decision.allowed:
                            continue
                        if (
                            frontier.queued_count + frontier.inflight_count - 1 + len(children) + 1
                            > plan.limits.frontier_entries
                        ):
                            raise ParentStop("budget_exhausted")
                        children.append(replace(child, score=decision.ordering_score))
                        identities.add(identity)
                        if remaining is not None:
                            remaining -= 1
                if frontier.queued_count + frontier.inflight_count - 1 + len(children) > plan.limits.frontier_entries:
                    raise ParentStop("budget_exhausted")
                context.check()
                # All async policy finished. No await before the complete article,
                # disposition and admitted-child transition.
                if successful_article(article) and _retained(plan, (article,)):
                    context.note_successful_result()
                frontier.commit(entry, page)
                for child in children:
                    frontier.admit(child)
                count = next_count
                notify("in_progress", entry)
                if article.get("error") == "browser_transport_unavailable":
                    reason = "transport_unavailable"
                    break
            if reason != "frontier_exhausted":
                break
    except asyncio.CancelledError as exc:
        cancellation, reason = exc, "cancelled"
        context.cancel()
    except ParentStop as exc:
        reason = exc.reason
    except Exception as exc:  # noqa: BLE001 - only machine codes cross this boundary
        reason = "checkpoint_failed" if loading else getattr(exc, "code", "execution_failed")
        if reason not in {
            "checkpoint_failed",
            "queue_overloaded",
            "queue_stopping",
            "job_payload_too_large",
            "execution_failed",
        }:
            reason = "execution_failed"
    finally:
        report, cancellation = await _finalize_crawl(
            plan,
            context=context,
            frontier=frontier,
            handles=handles,
            checkpoint=checkpoint,
            loading=loading,
            count=count,
            reason=reason,
            cancellation=cancellation,
            diagnostics=diagnostics,
            owned=(hooks, policy, acquire),
        )
    if cancellation is not None:
        acquire = policy = hooks = seeds = handle = source = None
        handles.clear()
        cancellation.__traceback__ = None
        raise cancellation
    if report.outcome.state == "completed":
        if legacy and hooks and hooks.progress_callback:
            hooks.progress_callback(f"Scraping completed. Total pages scraped: {count}")
        notify("completed")
    return report
