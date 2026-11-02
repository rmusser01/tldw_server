"""Deterministic execution-boundary doubles; no facade dependency."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig, ChunkMetadata, ChunkResult
from tldw_Server_API.app.core.Chunking.llm_context import _LLM_UNSET


class RecordingLock:
    """A real reentrant lock with per-thread observable ownership depth."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self._lock = threading.RLock()
        self._local = threading.local()

    @property
    def depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def __enter__(self) -> RecordingLock:
        self._lock.acquire()
        self._local.depth = self.depth + 1
        self.events.append(("lock_enter", self.depth))
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.events.append(("lock_exit", self.depth))
        self._local.depth = self.depth - 1
        self._lock.release()


class RecordingCache:
    """Expose cache identity, lookup failure and noncritical write failure."""

    def __init__(self, value: Any = None) -> None:
        self.value = value
        self.events: list[Any] = []
        self.get_error: Exception | None = None
        self.put_error: Exception | None = None

    def get(self, key: str) -> Any:
        self.events.append(("get", key))
        if self.get_error is not None:
            raise self.get_error
        return self.value

    def put(self, key: str, value: Any) -> None:
        self.events.append(("put", key, value))
        if self.put_error is not None:
            raise self.put_error
        self.value = value


class RecordingStrategy:
    """Return controlled real payloads while recording strategy boundaries."""

    def __init__(self, events: list[Any], lock: RecordingLock) -> None:
        self.events = events
        self.lock = lock
        self.calls: list[tuple[str, str, Any, Any, dict[str, Any]]] = []
        self.language = "en"
        self.llm_call_func: Any = None
        self.llm_config: Any = None
        self.tokenizer_name: Any = "old"
        self._tokenizer: Any = object()
        self._tokenizer_init_attempted = True
        self.text_result: Any = ["strategy text"]
        self.metadata_result: Any = [ChunkResult("strategy metadata", ChunkMetadata(0, 0, 4, 1))]
        self.iterator_factory: Callable[[], Any] | None = None
        self.on_call: Callable[[str, dict[str, Any]], Any] | None = None

    def _record(self, path: str, text: str, size: Any, overlap: Any, options: dict[str, Any]) -> None:
        self.calls.append((path, text, size, overlap, options))
        self.events.append((path, self.lock.depth))
        if self.on_call is not None:
            self.on_call(path, options)

    def chunk(self, text: str, max_size: Any, overlap: Any, **options: Any) -> Any:
        self._record("text", text, max_size, overlap, options)
        return self.text_result

    def chunk_with_metadata(self, text: str, max_size: Any, overlap: Any, **options: Any) -> Any:
        self._record("metadata", text, max_size, overlap, options)
        return self.metadata_result

    def chunk_generator(self, text: str, max_size: Any, overlap: Any, **options: Any) -> Iterator[str]:
        self._record("generator", text, max_size, overlap, options)
        return self.iterator_factory() if self.iterator_factory else iter(self.text_result)


class FakeExecutionContext:
    """Small live context for direct service tests, independent of Chunker."""

    def __init__(self) -> None:
        self.config = ChunkerConfig(enable_cache=False, default_max_size=8, default_overlap=0, language="en")
        self.events: list[Any] = []
        self._strategy_lock = RecordingLock(self.events)
        self._strategy_cache_mode = "shared"
        self._cache: RecordingCache | None = None
        self.strategy = RecordingStrategy(self.events, self._strategy_lock)
        self.effective_llm: tuple[Any, Any] = (None, {})

    def _sanitize_input(self, text: str, *, suppress_security_log: bool = False) -> str:
        self.events.append(("sanitize", text))
        return text

    def _enforce_text_size(self, text: str, *, source: str) -> None:
        self.events.append(("size", source))

    def _normalize_method_argument(self, method: Any) -> Any:
        self.events.append(("normalize", method))
        return method

    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any] | None = None) -> str:
        self.events.append(("resolve", method, language, options))
        return method

    def _apply_option_aliases(self, method: str, options: dict[str, Any]) -> dict[str, Any]:
        self.events.append(("aliases", method))
        return dict(options)

    def _override_overlap_for_method(self, method: str, overlap: Any, options: dict[str, Any]) -> Any:
        self.events.append(("overlap", method))
        return overlap

    def _get_effective_llm_hooks(self) -> tuple[Any, Any]:
        self.events.append(("llm",))
        return self.effective_llm

    def _get_llm_signature(self, *, llm_call_func: Any = _LLM_UNSET, llm_config: Any = _LLM_UNSET) -> str:
        self.events.append(("signature", llm_call_func, llm_config))
        return "signature"

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
        self.events.append(("key", text, method, max_size, overlap, language, options, llm_signature))
        return "key"

    def _create_strategy_instance(self, method: str) -> Any:
        self.events.append(("create", self._strategy_lock.depth))
        return self.strategy

    def _get_strategy_for_call(self, method: str) -> Any:
        self.events.append(("lookup", self._strategy_lock.depth))
        return self.strategy

    def _sync_strategy_llm(
        self, strategy: Any, *, llm_call_func: Any = _LLM_UNSET, llm_config: Any = _LLM_UNSET
    ) -> None:
        self.events.append(("sync", llm_call_func, llm_config))
        strategy.llm_call_func = llm_call_func
        strategy.llm_config = llm_config

    def _apply_strategy_language(self, strategy: Any, language: str) -> None:
        self.events.append(("language", language))
        strategy.language = language
