"""Live context contracts and shallow-frozen per-call hierarchy values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


class LeafChunkingContext(Protocol):
    """Strategy entry points used by leaves without owning orchestration state."""

    def chunk_text(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]:
        """Chunk source text with the resolved method, limits, and forwarded options."""
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
        """Return strategy results whose metadata can provide source character offsets."""
        ...


class HierarchyContext(LeafChunkingContext, Protocol):
    """Call-time configuration and hooks supplied by the public chunking context."""

    config: ChunkerConfig

    def _enforce_text_size(self, text: str, *, source: str) -> None:
        """Reject oversized text with source-specific diagnostics before chunking."""
        ...

    def _normalize_method_argument(self, method: Any) -> Any:
        """Convert a supplied method representation before resolving its default."""
        ...

    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any]) -> Any:
        """Select the effective strategy using the language and per-call options."""
        ...

    def _sanitize_input(self, text: str, *, suppress_security_log: bool = False) -> str:
        """Return the sanitized text view, optionally suppressing duplicate diagnostics."""
        ...

    def normalize_chunk_type(self, value: Any) -> str | None:
        """Map an output kind to its canonical chunk type, or return no match."""
        ...


@dataclass(frozen=True)
class ResolvedHierarchyOptions:
    """Resolved per-call settings; nested method options retain shallow identity."""

    method: Any
    max_size: Any
    overlap: Any
    language: Any
    method_options: dict[str, Any]
    sanitize_output: bool


@dataclass(frozen=True)
class HierarchyTextViews:
    """Original span source, sanitized lookup text, and selected output slice text."""

    original: str
    sanitized: str
    output: str
