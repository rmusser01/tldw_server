"""
Lightweight logging context helpers for propagating request/job identifiers.

Usage:

    from tldw_Server_API.app.core.Logging.log_context import log_context, new_request_id

    req_id = new_request_id()
    with log_context(request_id=req_id, job_id=job_id, ps_component="job_processor") as log:
        log.info("Starting work")
        ...

The context manager both contextualizes the base logger (so nested logs inherit
the fields) and returns a bound logger for convenience.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Security.request_id_middleware import _clean_request_id

try:
    # Optional import for FastAPI Request type annotation
    from fastapi import Request  # type: ignore
except ImportError:  # pragma: no cover - typing aid only
    Request = None  # type: ignore

_LOG_CONTEXT_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)

_TRACEPARENT_RE = re.compile(
    r"^00-"
    r"(?P<trace_id>[0-9a-fA-F]{32})-"
    r"(?P<span_id>[0-9a-fA-F]{16})-"
    r"(?P<trace_flags>[0-9a-fA-F]{2})$"
)


def new_request_id() -> str:
    """Return a new opaque request identifier."""
    return _clean_request_id(None)


@contextmanager
def log_context(**fields: Any) -> Iterator[Any]:
    """Context manager that sets structured logging fields and yields a bound logger.

    - Adds fields to the logger context (via logger.contextualize) so that any
      logs emitted inside the context inherit them.
    - Yields a logger bound with the same fields for direct use.
    """
    clean = {k: v for k, v in fields.items() if v is not None}
    with logger.contextualize(**clean):
        bound = logger.bind(**clean)
        yield bound


def ensure_request_id(request: Any) -> str:
    """Return a request_id from FastAPI Request or synthesize one.

    - Prefers `request.state.request_id` (set by RequestIDMiddleware).
    - Falls back to `X-Request-ID` header if present.
    - Generates a new request_id if none is found and attaches it to `request.state`.
    """
    try:
        req_id = getattr(getattr(request, "state", None), "request_id", None)
        if not req_id:
            headers = getattr(request, "headers", {}) or {}
            header_value = headers.get("X-Request-ID") or headers.get("x-request-id")
            req_id = _clean_request_id(header_value)
            try:
                request.state.request_id = req_id  # type: ignore[attr-defined]
            except _LOG_CONTEXT_NONCRITICAL_EXCEPTIONS:
                pass
        return str(req_id)
    except _LOG_CONTEXT_NONCRITICAL_EXCEPTIONS:
        return new_request_id()


def ensure_traceparent(request: Any) -> str:
    """Extract a W3C traceparent header, attach to request.state, and return it.

    This is a best-effort helper for environments without active tracing. The
    main log patcher already populates trace fields from the tracer when
    available; this simply surfaces inbound traceparent for correlation.
    """
    try:
        headers = getattr(request, "headers", {}) or {}
        tp = (
            headers.get("traceparent")
            or headers.get("Traceparent")
            or headers.get("TRACEPARENT")
            or ""
        )
        tp = _clean_traceparent(tp)
        if tp:
            try:
                request.state.traceparent = tp  # type: ignore[attr-defined]
            except _LOG_CONTEXT_NONCRITICAL_EXCEPTIONS:
                pass
        return tp
    except _LOG_CONTEXT_NONCRITICAL_EXCEPTIONS:
        return ""


def _clean_traceparent(value: Any) -> str:
    """Validate and normalize an inbound W3C traceparent header value."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = _TRACEPARENT_RE.fullmatch(raw)
    if not match:
        return ""
    trace_id = match.group("trace_id")
    span_id = match.group("span_id")
    if int(trace_id, 16) == 0 or int(span_id, 16) == 0:
        return ""
    return raw.lower()


def get_ps_logger(
    *,
    evaluation_id: int | None = None,
    prompt_id: int | None = None,
    optimization_id: int | None = None,
    project_id: int | None = None,
    request_id: str | None = None,
    job_id: int | None = None,
    ps_component: str | None = None,
    ps_job_kind: str | None = None,
    traceparent: str | None = None,
):
    """Return a bound logger with common Prompt Studio fields.

    Centralizes field names so direct logs outside of log_context have
    consistent structured metadata.
    """
    fields: dict[str, Any] = {}
    if evaluation_id is not None:
        fields["evaluation_id"] = evaluation_id
    if prompt_id is not None:
        fields["prompt_id"] = prompt_id
    if optimization_id is not None:
        fields["optimization_id"] = optimization_id
    if project_id is not None:
        fields["project_id"] = project_id
    if request_id is not None:
        fields["request_id"] = request_id
    if job_id is not None:
        fields["job_id"] = job_id
    if ps_component is not None:
        fields["ps_component"] = ps_component
    if ps_job_kind is not None:
        fields["ps_job_kind"] = ps_job_kind
    if traceparent is not None:
        fields["traceparent"] = traceparent
    return logger.bind(**fields)
