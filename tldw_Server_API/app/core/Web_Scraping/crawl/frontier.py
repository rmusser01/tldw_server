"""Bounded occupancy and atomic dispositions, independent of page acquisition."""

from __future__ import annotations

from ..contracts.execution import PageData, _safe_article
from ..runtime.controls import ParentStop
from ..url_utils import normalize_for_crawl
from .models import CrawlPlan, FrontierEntry, FrontierSnapshot


class Frontier:
    """Own queued/in-flight capacity; candidate accounting belongs to discovery."""

    def __init__(self, plan: CrawlPlan) -> None:
        self.plan = plan
        self._queued: list[FrontierEntry] = []
        self._inflight: list[FrontierEntry] = []
        self._completed: list[FrontierEntry] = []
        self._results: list[dict] = []
        self._seen: set[str] = set()

    @classmethod
    def from_snapshot(cls, plan: CrawlPlan, snapshot: FrontierSnapshot) -> Frontier:
        """Restore validated dispositions; storage validation remains Task 7's job."""
        frontier = cls(plan)
        if len(snapshot.results) != len(snapshot.completed):
            raise ValueError("checkpoint dispositions and results differ")
        if len(snapshot.queued) + len(snapshot.inflight) > plan.limits.frontier_entries:
            raise ValueError("checkpoint frontier exceeds allowance")
        for entry in snapshot.completed + snapshot.inflight + snapshot.queued:
            identity = frontier._identity(entry)
            if identity in frontier._seen:
                raise ValueError("duplicate checkpoint disposition")
            frontier._seen.add(identity)
        frontier._queued = list(snapshot.queued)
        frontier._inflight = list(snapshot.inflight)
        frontier._completed = list(snapshot.completed)
        frontier._results = [_safe_article(item) for item in snapshot.results]
        return frontier

    def _identity(self, entry: FrontierEntry) -> str:
        if self.plan.profile.startswith("enhanced_"):
            return normalize_for_crawl(entry.url, self.plan.request.base_url)
        return entry.url

    def admit(self, entry: FrontierEntry) -> bool:
        """Admit once or reject duplicate; never debit candidate encounters here."""
        identity = self._identity(entry)
        if identity in self._seen:
            return False
        if len(self._queued) + len(self._inflight) >= self.plan.limits.frontier_entries:
            raise ParentStop("budget_exhausted")
        self._seen.add(identity)
        self._queued.append(entry)
        return True

    def take(self) -> FrontierEntry | None:
        """Move the next strategy-ordered entry to in-flight without freeing space."""
        if not self._queued:
            return None
        strategy = self.plan.options.get("crawl_strategy", "default")
        if self.plan.profile in {"enhanced_recursive", "enhanced_url_level"} and strategy in {
            "best_first",
            "best-first",
            "bestfirst",
        }:
            from urllib.parse import urlsplit

            self._queued.sort(
                key=lambda item: (
                    -item.score,
                    item.depth,
                    -len([part for part in urlsplit(item.url).path.split("/") if part]),
                    item.url,
                    item.parent_url or "",
                )
            )
        entry = self._queued.pop(0)
        self._inflight.append(entry)
        return entry

    def commit(self, entry: FrontierEntry, page: PageData) -> None:
        """Commit the article and disposition as one synchronous state transition."""
        if entry not in self._inflight:
            raise ValueError("entry is not in flight")
        article = _safe_article(page.article)
        self._results.append(article)
        self._completed.append(entry)
        self._inflight.remove(entry)

    def restore_unfinished(self) -> None:
        """Restore unfinished entries in their original admission order."""
        self._queued = sorted(self._queued + self._inflight, key=lambda item: item.sequence)
        self._inflight.clear()

    def snapshot(self) -> FrontierSnapshot:
        """Return detached committed payloads and ordered immutable dispositions."""
        return FrontierSnapshot(
            tuple(self._queued),
            tuple(self._inflight),
            tuple(self._completed),
            tuple(_safe_article(item) for item in self._results),
        )
