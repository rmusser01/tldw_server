"""Count/byte-bounded allowlisted polling history, independent of live delivery."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .admission import bounded_snapshot
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
    }
)


class History:
    """Retain only bounded public snapshots; oversize delivery still succeeds."""

    def __init__(self, count: int, max_bytes: int) -> None:
        self.count = count
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries: OrderedDict[str, tuple[ScrapingJob, int]] = OrderedDict()

    def project(self, job: ScrapingJob, article: dict[str, Any] | None) -> tuple[ScrapingJob, int] | None:
        """Size as we copy, stopping before building an oversized result clone."""
        candidate = replace(job, metadata={}, context=None, result=None)
        if article is not None:
            candidate.result = {key: value for key, value in article.items() if key in ARTICLE_FIELDS}
        try:
            snapshot, size = bounded_snapshot(candidate.to_dict(), self.max_bytes, redact=True)
        except ValueError:
            return None
        candidate.result = snapshot["result"]
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
