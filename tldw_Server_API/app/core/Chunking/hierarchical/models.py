from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


class LeafChunkingContext(Protocol):
    def chunk_text(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]: ...

    def chunk_text_with_metadata(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]: ...


class HierarchyContext(LeafChunkingContext, Protocol):
    config: ChunkerConfig

    def _enforce_text_size(self, text: str, *, source: str) -> None: ...

    def _normalize_method_argument(self, method: Any) -> Any: ...

    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any]) -> Any: ...

    def _sanitize_input(self, text: str, *, suppress_security_log: bool = False) -> str: ...

    def normalize_chunk_type(self, value: Any) -> str | None: ...


@dataclass(frozen=True)
class ResolvedHierarchyOptions:
    method: Any
    max_size: Any
    overlap: Any
    language: Any
    method_options: dict[str, Any]
    sanitize_output: bool


@dataclass(frozen=True)
class HierarchyTextViews:
    original: str
    sanitized: str
    output: str
