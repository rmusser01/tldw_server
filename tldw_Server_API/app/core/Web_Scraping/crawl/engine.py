"""One crawl lifecycle over injected, owned page acquisition handles."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import MappingProxyType

from ..contracts.execution import CrawlOutcome, CrawlReport, successful_article
from ..runtime.controls import ParentStop, WorkHandle
from .discovery import charge, url_bytes
from .frontier import Frontier
from .models import CheckpointPort, CheckpointState, CrawlControls, CrawlHooks, CrawlPlan, FrontierEntry


async def run_crawl(
    plan: CrawlPlan,
    *,
    context: CrawlControls,
    acquire: Callable[[FrontierEntry], Awaitable[WorkHandle]],
    checkpoint: CheckpointPort | None = None,
    hooks: CrawlHooks | None = None,
) -> CrawlReport:
    """Commit results atomically and publish only after checkpoint finalization.

    Acquisition owns physical page attempts (including retries). This owner only
    charges the seed and attaches accepted work; discovered encounters were already
    admitted by discovery, not by frontier occupancy.
    """
    frontier = Frontier(plan)
    handles = []
    reason = "frontier_exhausted"
    cancellation = None
    diagnostics = []
    count = 0
    sequence = 1
    maximum = plan.options.get("max_pages")
    depth_limit = plan.options.get("max_depth")
    legacy = plan.profile == "legacy_recursive"
    loading_checkpoint = checkpoint is not None
    try:
        state = None
        if checkpoint is not None:
            state = checkpoint.load(plan=plan, scope=checkpoint.scope)
            if state is not None:
                if state.scope != checkpoint.scope or state.plan != plan or state.schema_version != 1:
                    raise ValueError("checkpoint binding mismatch")
                context.restore_budget(state.budget, accounting_complete=state.accounting_complete)
                frontier = Frontier.from_snapshot(plan, state.frontier)
                frontier.restore_unfinished()
                count = sum(bool(item.get("extraction_successful")) for item in state.frontier.results)
                entries = state.frontier.queued + state.frontier.inflight + state.frontier.completed
                sequence = max((entry.sequence for entry in entries), default=0) + 1
                if any(successful_article(item) for item in state.frontier.results):
                    context.note_successful_result()
        loading_checkpoint = False
        if state is None:
            charge(context, "discovered_candidates")
            charge(context, "discovered_url_bytes", url_bytes(plan.request.base_url, plan))
            frontier.admit(FrontierEntry(plan.request.base_url, None, 0, 0, 0.0))
        while True:
            context.check()
            if maximum is not None and count >= maximum:
                reason = "page_cap_reached"
                break
            entry = frontier.take()
            if entry is None:
                break
            if legacy and hooks and hooks.progress_callback:
                hooks.progress_callback(f"Scraping page {count + 1}/{maximum}: {entry.url}")
            handle = await acquire(entry)
            handles.append(handle)
            context.attach(handle)
            try:
                page = await asyncio.wait_for(asyncio.shield(handle.result), context.cap_timeout(None))
            except TimeoutError:
                context.check()
                raise ParentStop("deadline_exhausted") from None
            context.check()
            # No await between the stop/publication-lock marker and article /
            # frontier disposition commit. Profile counting is deliberately looser.
            if successful_article(page.article):
                context.note_successful_result()
            frontier.commit(entry, page)
            profile_success = bool(page.article.get("extraction_successful"))
            count += profile_success
            if not legacy and not profile_success:
                diagnostics.append("execution_failed")
            expands = legacy or profile_success
            if expands and (depth_limit is None or entry.depth < depth_limit):
                remaining = maximum - count if maximum is not None else None
                for link in page.discovery.links:
                    context.check()
                    if remaining is not None and remaining <= 0 and not legacy:
                        break
                    child = FrontierEntry(
                        link.url, page.article.get("url") or entry.url, entry.depth + 1, sequence, 0.0
                    )
                    if frontier.admit(child):
                        sequence += 1
                        if remaining is not None:
                            remaining -= 1
            if hooks and hooks.on_progress:
                snapshot = frontier.snapshot()
                hooks.on_progress(
                    MappingProxyType(
                        {
                            "status": "in_progress",
                            "pages_scraped": count,
                            "current_url": entry.url,
                            "queue_size": len(snapshot.queued),
                            "visited": len(snapshot.completed),
                        }
                    )
                )
    except asyncio.CancelledError as exc:
        cancellation = exc
        reason = "cancelled"
        context.cancel()
    except ParentStop as exc:
        reason = exc.reason
    except Exception:  # noqa: BLE001 - sanitize internal execution failures at the report boundary
        reason = "checkpoint_failed" if loading_checkpoint else "execution_failed"
    finally:
        normal = reason in {"frontier_exhausted", "page_cap_reached"}
        if not normal:
            for handle in handles:
                try:
                    handle.cancel()
                except (asyncio.CancelledError, Exception):  # noqa: BLE001 - preserve primary reason
                    diagnostics.append("execution_failed")
        snapshot = frontier.snapshot()
        if checkpoint is not None:
            try:
                if normal:
                    checkpoint.clear()
                else:
                    budget = context.snapshot()
                    checkpoint.save(
                        CheckpointState(checkpoint.scope, plan, snapshot, budget, reason, budget.accounting_complete)
                    )
            except Exception:  # noqa: BLE001 - safe checkpoint failure precedence
                diagnostics.append("checkpoint_failed")
                if normal:
                    reason = "checkpoint_failed"
            pending = tuple(handle.drained for handle in handles if not handle.drained.done())
            if pending:

                async def close_after_drain():
                    await asyncio.gather(*(asyncio.shield(future) for future in pending))
                    checkpoint.close()

                task = asyncio.create_task(close_after_drain(), name="crawl-checkpoint-drain")
                task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            else:
                try:
                    checkpoint.close()
                except Exception:  # noqa: BLE001 - close cannot mask cancellation/exhaustion
                    diagnostics.append("checkpoint_failed")
                    if normal:
                        reason = "checkpoint_failed"
        articles = tuple(item for item in snapshot.results if successful_article(item))
        normal = reason in {"frontier_exhausted", "page_cap_reached"}
        outcome = (
            CrawlOutcome("completed", reason, False) if normal else CrawlReport.interrupted(articles, reason).outcome
        )
        outcome = context.publish(outcome)
    if cancellation is not None:
        raise cancellation
    if outcome.state == "completed" and hooks:
        if legacy and hooks.progress_callback:
            hooks.progress_callback(f"Scraping completed. Total pages scraped: {count}")
        if hooks.on_progress:
            hooks.on_progress(MappingProxyType({"status": "completed", "pages_scraped": count}))
    return CrawlReport(articles, outcome, tuple(diagnostics))
