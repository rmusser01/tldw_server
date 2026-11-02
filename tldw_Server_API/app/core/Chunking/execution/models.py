"""Live execution dependencies supplied by the facade, without reverse imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..base import ChunkerConfig
from ..llm_context import _LLM_UNSET


class ResultCache(Protocol):
    """Result cache operations used by text execution without owning the cache."""

    def get(self, key: str) -> Any:
        """Return a cached value under the cache's existing copy policy."""
        ...

    def put(self, key: str, value: Any) -> None:
        """Store a value under the cache's existing copy policy."""
        ...


@dataclass(frozen=True)
class ExecutionHooks:
    """Forwarders resolve current facade observability functions for each event."""

    get_logger: Callable[[], Any]
    increment_counter: Callable[..., Any]


ExecutionKind = Literal["text", "metadata", "generator"]


@dataclass(frozen=True)
class PreparedExecution:
    """Resolved per-call values; option dictionaries retain shallow identities."""

    method: Any
    max_size: Any
    overlap: Any
    language: Any
    tokenizer_override: Any
    options_raw: dict[str, Any]
    strategy_options: dict[str, Any]
    use_per_call_strategy: bool
    align_text_to_source: bool


class ExecutionContext(Protocol):
    """Read live facilities and retain callable override seams on the facade."""

    @property
    def config(self) -> ChunkerConfig:
        """Current configuration, not a construction-time snapshot."""
        ...

    @property
    def _cache(self) -> ResultCache | None:
        """Current optional result cache."""
        ...

    @property
    def _strategy_cache_mode(self) -> str:
        """Current strategy reuse mode."""
        ...

    @property
    def _strategy_lock(self) -> Any:
        """Existing lock; execution, not strategy preparation, owns its scope."""
        ...

    def _sanitize_input(self, text: str, *, suppress_security_log: bool = False) -> str:
        """Apply the existing sanitization and security logging policy."""
        ...

    def _enforce_text_size(self, text: str, *, source: str) -> None:
        """Apply the existing input limit with its path-specific source label."""
        ...

    def _normalize_method_argument(self, method: Any) -> Any:
        """Normalize a method before resolving its default."""
        ...

    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any] | None = None) -> str:
        """Resolve strategy selection with live facade policy."""
        ...

    def _apply_option_aliases(self, method: str, options: dict[str, Any]) -> dict[str, Any]:
        """Apply existing method-specific option aliases."""
        ...

    def _override_overlap_for_method(self, method: str, overlap: Any, options: dict[str, Any]) -> Any:
        """Resolve an existing method-specific overlap override."""
        ...

    def _get_effective_llm_hooks(self) -> tuple[Any, Any]:
        """Resolve thread-local LLM overrides once at the path's original point."""
        ...

    def _get_llm_signature(self, *, llm_call_func: Any = _LLM_UNSET, llm_config: Any = _LLM_UNSET) -> str:
        """Build the existing signature from supplied effective hooks."""
        ...

    def _get_cache_key(
        self,
        text: str,
        method: str,
        max_size: int,
        overlap: int,
        language: str,
        options: dict,
        *,
        llm_signature: str = "",
    ) -> str:
        """Build the existing result-cache key, retaining facade overrides."""
        ...

    def _create_strategy_instance(self, method: str) -> Any:
        """Create an uncached strategy through the existing registry."""
        ...

    def _get_strategy_for_call(self, method: str) -> Any:
        """Look up a strategy using existing injection and cache-mode policy."""
        ...

    def _sync_strategy_llm(
        self, strategy: Any, *, llm_call_func: Any = _LLM_UNSET, llm_config: Any = _LLM_UNSET
    ) -> None:
        """Apply supplied effective LLM hooks under the existing copy policy."""
        ...

    def _apply_strategy_language(self, strategy: Any, language: str) -> None:
        """Apply the existing language-resource synchronization policy."""
        ...
