"""Best-effort, bounded observability for one complete quiz generation attempt."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.quizzes import QuizGenerationProfile, QuizSourceType
from tldw_Server_API.app.core.exceptions import BadRequestError, OsceProviderError, OsceVerificationError
from tldw_Server_API.app.core.Metrics import metrics_manager

_PROFILES = frozenset(profile.value for profile in QuizGenerationProfile)
_SOURCE_TYPES = frozenset(source.value for source in QuizSourceType)


@contextlib.contextmanager
def _optional_metrics() -> Iterator[None]:
    """Isolate telemetry failures, including logging outages, from generation."""
    try:
        yield
    except Exception:  # noqa: BLE001 - Observability must never replace a generation result or error.
        with contextlib.suppress(Exception):
            logger.debug("Quiz generation metrics unavailable")


def _record(name: str, labels: dict[str, str], value: float = 1, *, histogram: bool = False) -> None:
    """Isolate each registry write so one failed metric does not prevent another."""
    with _optional_metrics():
        registry = metrics_manager.get_metrics_registry()
        if histogram:
            registry.observe(name, value, labels=labels)
        else:
            registry.increment(name, value, labels=labels)


@dataclass
class QuizGenerationMetrics:
    """Record one request/outcome pair, preserving returns, errors, and cancellation.

    The caller supplies already-normalized profile and source types. Phase markers
    distinguish provider/persistence exceptions from genuine validation failures;
    neither phases nor exception messages become metric labels.
    """

    profile: str = "unknown"
    phase: Literal["validation", "provider", "runtime", "osce"] = "validation"
    _source_type: str = field(default="unknown", init=False)
    _started_at: float | None = field(default=None, init=False)
    _request_recorded: bool = field(default=False, init=False)

    def __enter__(self) -> QuizGenerationMetrics:
        """Start elapsed timing before source validation or resolution."""
        with _optional_metrics():
            self._started_at = time.perf_counter()
        return self

    def start(self, sources: Sequence[dict[str, str]]) -> None:
        """Count normalized requests before I/O, collapsing duplicate and mixed types."""
        with _optional_metrics():
            source_types = {source["source_type"] for source in sources}
            if source_types and source_types <= _SOURCE_TYPES:
                self._source_type = next(iter(source_types)) if len(source_types) == 1 else "mixed"
            self._record_request()

    def _labels(self) -> dict[str, str]:
        """Return only allowlisted dimensions; never retain source IDs or content."""
        return {
            "profile": self.profile if self.profile in _PROFILES else "unknown",
            "source_type": self._source_type,
        }

    def _record_request(self) -> None:
        """Attempt request recording once, including requests rejected before start."""
        if not self._request_recorded:
            self._request_recorded = True
            _record("quiz_generation_requests_total", self._labels())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Observe the terminal outcome without suppressing or replacing the error."""
        with _optional_metrics():
            if error is None:
                outcome = "success"
            elif isinstance(error, asyncio.CancelledError):
                outcome = "cancelled"
            elif self.phase == "provider" or (
                self.phase == "osce"
                and (
                    isinstance(error, OsceProviderError)
                    or (isinstance(error, OsceVerificationError) and error.__cause__ is not None)
                )
            ):
                outcome = "provider_error"
            elif self.phase != "runtime" and isinstance(error, (ValueError, BadRequestError)):
                outcome = "validation_error"
            else:
                outcome = "runtime_error"
            self._record_request()
            labels = {**self._labels(), "outcome": outcome}
            _record("quiz_generation_outcomes_total", labels)
            if self._started_at is not None:
                _record(
                    "quiz_generation_duration_seconds",
                    labels,
                    max(0.0, time.perf_counter() - self._started_at),
                    histogram=True,
                )
        return False
