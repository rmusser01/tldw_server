"""Progress and cancellation callbacks scoped to one PDF ingest job."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

PDF_SAVE_PROGRESS_PERCENT = 85


class PDFProcessingCancelled(Exception):
    """The current PDF job cannot start another processing step."""


_job_progress: ContextVar[Callable[[float | None, str], bool] | None] = ContextVar(
    "pdf_ingest_job_progress", default=None
)


@contextmanager
def pdf_job_progress(
    cancel_check: Callable[[], bool], report: Callable[[float, str], bool] | None = None
) -> Iterator[None]:
    def admit_step(percent: float | None, message: str) -> bool:
        if percent is not None and report is not None:
            return report(percent, message)
        return not cancel_check()

    token = _job_progress.set(admit_step)
    try:
        yield
    finally:
        _job_progress.reset(token)


def pdf_job_step_allowed(percent: float | None = None, message: str = "") -> bool:
    callback = _job_progress.get()
    return callback is None or callback(percent, message)


def check_pdf_job(percent: float | None = None, message: str = "") -> None:
    if not pdf_job_step_allowed(percent, message):
        raise PDFProcessingCancelled()
