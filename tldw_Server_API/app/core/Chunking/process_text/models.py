"""Data models and protocols for the internal process-text pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


@dataclass(frozen=True)
class PreparedText:
    """Text and options after frontmatter/header preparation."""

    original_text: str
    processed_text: str
    prefix_offset: int
    json_meta: dict[str, Any]
    header_text: str
    options: dict[str, Any]


@dataclass(frozen=True)
class ResolvedProcessOptions:
    """Validated and derived options for one process-text invocation."""

    method: Any
    method_lower: str
    max_size: int
    overlap: int
    language: Any
    adaptive: bool
    hierarchical: bool
    hier_template: dict[str, Any] | None
    multi_level: bool
    code_mode_for_method: str | None
    method_options_for_chunk: dict[str, Any]
    align_text_to_source: bool = True


@dataclass(frozen=True)
class NormalizedChunk:
    """Strategy output normalized to text plus mutable metadata."""

    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TelemetryHooks:
    """Telemetry callbacks injected to preserve chunker monkeypatch compatibility."""

    increment_counter: Callable[..., Any]
    observe_histogram: Callable[..., Any]
    set_gauge: Callable[..., Any]
    start_span: Callable[..., Any]
    set_span_attribute: Callable[..., Any]
    add_span_event: Callable[..., Any]
    record_span_exception: Callable[..., Any]


class ProcessTextContext(Protocol):
    """Structural contract required from ``Chunker`` by ``ProcessTextPipeline``."""

    config: ChunkerConfig
    _thread_local: Any

    def _enforce_text_size(self, text: str, *, source: str) -> None:
        """Raise if ``text`` exceeds configured process-text limits."""
        ...

    def _normalize_method_argument(self, method: Any) -> Any:
        """Normalize public method input into the chunker's canonical form."""
        ...

    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any]) -> Any:
        """Resolve the effective strategy method for a prepared input."""
        ...

    def _compute_paragraph_spans(self, text: str, template: Any = None) -> list[tuple[int, int, str]]:
        """Return source spans used by multi-level paragraph dispatch."""
        ...

    def chunk_text(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]:
        """Chunk text and return the strategy's standard output shape."""
        ...

    def chunk_text_with_metadata(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]:
        """Chunk text and return metadata-bearing chunk result objects."""
        ...

    def chunk_text_hierarchical_flat(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        template: Any = None,
        method_options: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk text with hierarchical/template awareness into flat rows."""
        ...
