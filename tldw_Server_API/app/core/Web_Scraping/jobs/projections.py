"""Count/byte-bounded allowlisted polling history, independent of live delivery."""

import sys
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..contracts.execution import CrawlOutcome
from .admission import _PRIVATE_STATE_KEYS as _PRIVATE_STATE_KEYS
from .admission import _URL_IN_TEXT as _URL_IN_TEXT
from .admission import PUBLIC_CONTENT_KEYS as PUBLIC_CONTENT_KEYS
from .admission import bounded_snapshot, public_diagnostic
from .admission import private_state_key as private_state_key
from .admission import public_state_string as public_state_string
from .models import ScrapingJob

ARTICLE_FIELDS = frozenset(
    {
        "url",
        "title",
        "author",
        "date",
        "content",
        "extraction_successful",
        "error",
        "metadata",
        "preflight_analysis",
        "summary",
        "description",
        "language",
        "word_count",
        "extraction_method",
        "method",
        "duplicate",
        "is_duplicate",
        "content_hash",
        "raw_html",
        "raw_content",
    }
)

_PROGRESS_FIELDS = frozenset(
    {
        "status",
        "total_urls",
        "processed_urls",
        "current_url",
        "started_at",
        "updated_at",
        "completed_at",
        "pages_scraped",
        "total_pages",
        "queue_size",
        "visited",
        "remaining",
        "remaining_urls",
        "max_pages",
        "max_depth",
        "current_depth",
        "error",
        "crawl_outcome",
        "service_status",
    }
)


def public_crawl_outcome(outcome: CrawlOutcome) -> dict[str, Any]:
    """Project terminal crawl facts, never execution state or credentials."""
    return {
        "state": outcome.state,
        "stop_reason": outcome.stop_reason,
        "has_partial_results": outcome.has_partial_results,
    }


def public_job(job: ScrapingJob, *, crawl_outcome: CrawlOutcome | None = None) -> dict[str, Any]:
    """Allowlist polling state for both legacy and process-local job values."""
    snapshot = {
        "job_id": job.job_id,
        "url": job.url,
        "method": job.method,
        "user_id": job.user_id,
        "priority": job.priority.value,
        "status": job.status.name,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "retries": job.retries,
        "result": job.result,
        "error": public_diagnostic(job.error),
        "metadata": {},
        "cancel_requested": job.cancel_requested,
    }
    if crawl_outcome is not None:
        snapshot["crawl_outcome"] = public_crawl_outcome(crawl_outcome)
    return bounded_snapshot(snapshot, sys.maxsize, redact=True, article_path=("result",))[0]


def public_progress(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return an allowlisted detached progress snapshot, never resume state."""
    return bounded_snapshot(
        {key: value for key, value in snapshot.items() if key in _PROGRESS_FIELDS}, sys.maxsize, redact=True
    )[0]


class History:
    """Retain only bounded public snapshots; oversize delivery still succeeds."""

    def __init__(self, count: int, max_bytes: int) -> None:
        self.count = count
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries: OrderedDict[str, tuple[ScrapingJob, int]] = OrderedDict()

    def project(self, job: ScrapingJob, article: dict[str, Any] | None) -> tuple[ScrapingJob, int] | None:
        """Size as we copy, stopping before building an oversized result clone."""
        candidate = replace(job, metadata={}, context=None, result=None, error=public_diagnostic(job.error))
        if article is not None:
            candidate.result = {key: value for key, value in article.items() if key in ARTICLE_FIELDS}
            if article.get("error"):
                candidate.result["error"] = "source_extraction_failed"
        try:
            # Keep sizing incremental; do not build an unbounded public copy first.
            state = candidate.to_dict()
            state["url"] = public_state_string(candidate.url)
            snapshot, size = bounded_snapshot(state, self.max_bytes, redact=True, article_path=("result",))
        except ValueError:
            return None
        candidate.result = snapshot["result"]
        candidate.url = snapshot["url"]
        return candidate, size

    def record(self, job: ScrapingJob, article: dict[str, Any] | None) -> None:
        if not self.count:
            return
        projection = self.project(job, article)
        if projection is None:
            return
        self.remove(job.job_id)
        candidate, size = projection
        while self.entries and (len(self.entries) >= self.count or self.bytes + size > self.max_bytes):
            _, (_, old_size) = self.entries.popitem(last=False)
            self.bytes -= old_size
        self.entries[job.job_id] = (candidate, size)
        self.bytes += size

    def remove(self, job_id: str) -> None:
        old = self.entries.pop(job_id, None)
        if old is not None:
            self.bytes -= old[1]

    def get(self, job_id: str) -> ScrapingJob | None:
        value = self.entries.get(job_id)
        return deepcopy(value[0]) if value else None
