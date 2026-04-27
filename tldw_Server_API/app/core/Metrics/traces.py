"""
Distributed tracing support for the tldw_server application.

Provides easy-to-use tracing utilities for tracking operations
across service boundaries and async operations.
"""

import asyncio
import contextvars
import functools
import json
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from .telemetry import OTEL_AVAILABLE, get_telemetry_manager

if OTEL_AVAILABLE:
    from opentelemetry import baggage, context, trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
else:
    Status = None  # type: ignore
    StatusCode = None  # type: ignore


# Type variable for decorators
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class TraceContext:
    """Context information for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: dict[str, str] = None

    def to_headers(self) -> dict[str, str]:
        """Convert trace context to HTTP headers."""
        headers = {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01"
        }

        if self.baggage:
            baggage_items = [f"{k}={v}" for k, v in self.baggage.items()]
            headers["baggage"] = ",".join(baggage_items)

        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional['TraceContext']:
        """Extract trace context from HTTP headers."""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) >= 4:
                trace_id = parts[1]
                span_id = parts[2]

                # Parse baggage if present
                baggage_str = headers.get("baggage", "")
                baggage_dict = {}
                if baggage_str:
                    for item in baggage_str.split(","):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            baggage_dict[k.strip()] = v.strip()

                return cls(
                    trace_id=trace_id,
                    span_id=span_id,
                    baggage=baggage_dict
                )
        except Exception as e:
            logger.error(f"Failed to parse trace context: {e}")

        return None


class TracingManager:
    """Manager for distributed tracing operations."""

    def __init__(self):
        """Initialize the tracing manager."""
        self.active_spans = {}
        # Local baggage store when OpenTelemetry baggage is unavailable
        self._local_baggage = contextvars.ContextVar("tldw_local_baggage", default=None)

    @property
    def telemetry(self):
        """Always return the *current* global telemetry manager so that a
        shutdown/re-init cycle is picked up automatically."""
        return get_telemetry_manager()

    @property
    def tracer(self):
        """Return a tracer from the current telemetry manager."""
        return self.telemetry.get_tracer("tldw_server.tracing")

    @contextmanager
    def span(
        self,
        name: str,
        kind: Optional['SpanKind'] = None,
        attributes: Optional[dict[str, Any]] = None,
        links: Optional[list] = None
    ):
        """
        Create a new span.

        Args:
            name: Name of the span
            kind: Type of span (CLIENT, SERVER, INTERNAL, PRODUCER, CONSUMER)
            attributes: Initial attributes for the span
            links: Links to other spans

        Yields:
            The span object
        """
        if not OTEL_AVAILABLE:
            yield None
            return

        with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes,
            links=links,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span_id: Optional[str] = None
            try:
                # Track active span
                if hasattr(span, "get_span_context"):
                    span_context = span.get_span_context()
                    span_id = format(span_context.span_id, '016x')
                    self.active_spans[span_id] = span

                yield span

            except Exception as e:
                if span and hasattr(span, "record_exception"):
                    span.record_exception(e)
                if span and hasattr(span, "set_status") and Status and StatusCode:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                # Remove from active spans
                if span_id and span_id in self.active_spans:
                    del self.active_spans[span_id]

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: Optional['SpanKind'] = None,
        attributes: Optional[dict[str, Any]] = None,
        links: Optional[list] = None
    ):
        """
        Create a new span for async operations.

        Args:
            name: Name of the span
            kind: Type of span
            attributes: Initial attributes for the span
            links: Links to other spans

        Yields:
            The span object
        """
        if not OTEL_AVAILABLE:
            yield None
            return

        with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes,
            links=links,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span_id: Optional[str] = None
            try:
                # Track active span
                if hasattr(span, "get_span_context"):
                    span_context = span.get_span_context()
                    span_id = format(span_context.span_id, '016x')
                    self.active_spans[span_id] = span

                yield span

            except Exception as e:
                if span and hasattr(span, "record_exception"):
                    span.record_exception(e)
                if span and hasattr(span, "set_status") and Status and StatusCode:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                # Remove from active spans
                if span_id and span_id in self.active_spans:
                    del self.active_spans[span_id]

    def get_current_span(self):
        """Get the current active span."""
        if not OTEL_AVAILABLE:
            return None
        return trace.get_current_span()

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None):
        """
        Add an event to the current span.

        Args:
            name: Name of the event
            attributes: Event attributes
        """
        span = self.get_current_span()
        if span:
            span.add_event(name, attributes=attributes)

    def set_attribute(self, key: str, value: Any):
        """
        Set an attribute on the current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        span = self.get_current_span()
        if span:
            span.set_attribute(key, value)

    def set_attributes(self, attributes: dict[str, Any]):
        """
        Set multiple attributes on the current span.

        Args:
            attributes: Dictionary of attributes
        """
        span = self.get_current_span()
        if span:
            for key, value in attributes.items():
                span.set_attribute(key, value)

    def record_exception(self, exception: Exception, escaped: bool = True):
        """
        Record an exception in the current span.

        Args:
            exception: The exception to record
            escaped: Whether the exception escaped the span
        """
        span = self.get_current_span()
        if span:
            span.record_exception(exception, escaped=escaped)
            if escaped:
                span.set_status(Status(StatusCode.ERROR, str(exception)))

    def set_status(self, code: 'StatusCode', description: Optional[str] = None):
        """
        Set the status of the current span.

        Args:
            code: Status code
            description: Optional status description
        """
        span = self.get_current_span()
        if span:
            span.set_status(Status(code, description))

    def set_baggage(self, key: str, value: str):
        """
        Set a baggage item that propagates to child spans.

        Args:
            key: Baggage key
            value: Baggage value
        """
        if OTEL_AVAILABLE:
            ctx = baggage.set_baggage(key, value)
            context.attach(ctx)
        else:
            # Fallback: store in local in-memory baggage
            bag = dict(self._local_baggage.get() or {})
            bag[key] = value
            self._local_baggage.set(bag)

    def get_baggage(self, key: str) -> Optional[str]:
        """
        Get a baggage item.

        Args:
            key: Baggage key

        Returns:
            Baggage value or None
        """
        if OTEL_AVAILABLE:
            return baggage.get_baggage(key)
        return (self._local_baggage.get() or {}).get(key)

    def extract_context(self, carrier: dict[str, str]) -> Optional[TraceContext]:
        """
        Extract trace context from a carrier (e.g., HTTP headers).

        Args:
            carrier: Dictionary containing trace context

        Returns:
            TraceContext or None
        """
        return TraceContext.from_headers(carrier)

    def inject_context(self, carrier: dict[str, str]):
        """
        Inject current trace context into a carrier.

        Args:
            carrier: Dictionary to inject context into
        """
        span = self.get_current_span()
        if span and OTEL_AVAILABLE:
            span_context = span.get_span_context()
            if span_context.is_valid:
                trace_id = format(span_context.trace_id, '032x')
                span_id = format(span_context.span_id, '016x')

                carrier["traceparent"] = f"00-{trace_id}-{span_id}-01"

                # Add baggage
                baggage_items = []
                for key in ["user_id", "session_id", "request_id"]:
                    value = self.get_baggage(key)
                    if value:
                        baggage_items.append(f"{key}={value}")

                if baggage_items:
                    carrier["baggage"] = ",".join(baggage_items)


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """
    Get or create the global tracing manager.

    Returns:
        TracingManager instance
    """
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


# Decorator functions
def trace_operation(
    name: Optional[str] = None,
    kind: Optional['SpanKind'] = None,
    attributes: Optional[dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False
) -> Callable[[F], F]:
    """
    Decorator to trace a function execution.

    Args:
        name: Span name (defaults to function name)
        kind: Type of span
        attributes: Initial span attributes
        record_args: Whether to record function arguments
        record_result: Whether to record function result

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                manager = get_tracing_manager()

                # Build attributes
                span_attributes = dict(attributes) if attributes else {}
                span_attributes["function"] = func.__name__
                span_attributes["module"] = func.__module__

                if record_args:
                    # Safely serialize arguments
                    try:
                        span_attributes["args"] = json.dumps(str(args)[:1000])
                        span_attributes["kwargs"] = json.dumps(str(kwargs)[:1000])
                    except Exception:
                        logger.debug("trace_operation arg serialization failed")

                async with manager.async_span(span_name, kind=kind, attributes=span_attributes) as span:
                    result = await func(*args, **kwargs)

                    if record_result and span:
                        try:
                            span.set_attribute("result", json.dumps(str(result)[:1000]))
                        except Exception:
                            logger.debug("trace_operation result serialization failed")

                    return result

            return async_wrapper

        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                manager = get_tracing_manager()

                # Build attributes
                span_attributes = dict(attributes) if attributes else {}
                span_attributes["function"] = func.__name__
                span_attributes["module"] = func.__module__

                if record_args:
                    try:
                        span_attributes["args"] = json.dumps(str(args)[:1000])
                        span_attributes["kwargs"] = json.dumps(str(kwargs)[:1000])
                    except Exception:
                        logger.debug("trace_operation arg serialization failed")

                with manager.span(span_name, kind=kind, attributes=span_attributes) as span:
                    result = func(*args, **kwargs)

                    if record_result and span:
                        try:
                            span.set_attribute("result", json.dumps(str(result)[:1000]))
                        except Exception:
                            logger.debug("trace_operation result serialization failed")

                    return result

            return sync_wrapper

    return decorator


def trace_method(
    name: Optional[str] = None,
    kind: Optional['SpanKind'] = None,
    attributes: Optional[dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator to trace a class method.

    Similar to trace_operation but includes class name in span name.
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                span_name = name or f"{self.__class__.__name__}.{func.__name__}"
                manager = get_tracing_manager()
                span_attributes = dict(attributes) if attributes else {}
                span_attributes["class"] = self.__class__.__name__
                span_attributes["method"] = func.__name__
                async with manager.async_span(span_name, kind=kind, attributes=span_attributes):
                    return await func(self, *args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            span_name = name or f"{self.__class__.__name__}.{func.__name__}"
            manager = get_tracing_manager()
            span_attributes = dict(attributes) if attributes else {}
            span_attributes["class"] = self.__class__.__name__
            span_attributes["method"] = func.__name__
            with manager.span(span_name, kind=kind, attributes=span_attributes):
                return func(self, *args, **kwargs)

        return sync_wrapper

    return decorator


# Convenience functions
def start_span(name: str, **kwargs):
    """Start a new span."""
    return get_tracing_manager().span(name, **kwargs)


def start_async_span(name: str, **kwargs):
    """Start a new async span."""
    return get_tracing_manager().async_span(name, **kwargs)


def add_span_event(name: str, attributes: Optional[dict[str, Any]] = None):
    """Add an event to the current span."""
    get_tracing_manager().add_event(name, attributes)


def set_span_attribute(key: str, value: Any):
    """Set an attribute on the current span."""
    get_tracing_manager().set_attribute(key, value)


def set_span_attributes(attributes: dict[str, Any]):
    """Set multiple attributes on the current span."""
    get_tracing_manager().set_attributes(attributes)


def record_span_exception(exception: Exception, escaped: bool = True):
    """Record an exception in the current span."""
    get_tracing_manager().record_exception(exception, escaped)


def set_span_status(code: 'StatusCode', description: Optional[str] = None):
    """Set the status of the current span."""
    get_tracing_manager().set_status(code, description)
