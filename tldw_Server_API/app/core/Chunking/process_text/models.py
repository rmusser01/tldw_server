from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


@dataclass(frozen=True)
class PreparedText:
    original_text: str
    processed_text: str
    prefix_offset: int
    json_meta: dict[str, Any]
    header_text: str
    options: dict[str, Any]


@dataclass(frozen=True)
class ResolvedProcessOptions:
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
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TelemetryHooks:
    increment_counter: Callable[..., Any]
    observe_histogram: Callable[..., Any]
    set_gauge: Callable[..., Any]
    start_span: Callable[..., Any]
    set_span_attribute: Callable[..., Any]
    add_span_event: Callable[..., Any]
    record_span_exception: Callable[..., Any]


class ProcessTextContext(Protocol):
    config: ChunkerConfig
    _thread_local: Any

    def _enforce_text_size(self, text: str, *, source: str) -> None: ...
    def _normalize_method_argument(self, method: Any) -> Any: ...
    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any]) -> Any: ...
    def _compute_paragraph_spans(self, text: str, template: Any = None) -> list[tuple[int, int, str]]: ...
    def chunk_text(self, text: str, method: Any = None, max_size: Any = None, overlap: Any = None, language: Any = None, **options: Any) -> list[Any]: ...
    def chunk_text_with_metadata(self, text: str, method: Any = None, max_size: Any = None, overlap: Any = None, language: Any = None, **options: Any) -> list[Any]: ...
    def chunk_text_hierarchical_flat(self, text: str, method: Any = None, max_size: Any = None, overlap: Any = None, language: Any = None, template: Any = None, method_options: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
