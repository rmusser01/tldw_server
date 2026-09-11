"""Canonical cache lifecycle for extraction strategies."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from typing import Any, Callable

from ..selectors import clear_selector_caches, get_selector_cache_stats
from .metrics import emit_callback_counter
from .throttles import get_throttle_stats

_CLUSTER_EMBED_CACHE_MAX = 512
_SCHEMA_RESULT_CACHE_MAX = 128
_CLUSTER_EMBED_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_SCHEMA_RESULT_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CLUSTER_CACHE_LOCK = Lock()
_SCHEMA_CACHE_LOCK = Lock()
_SELECTOR_CACHE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)


def _schema_cache_get(key: str) -> dict[str, Any] | None:
    with _SCHEMA_CACHE_LOCK:
        value = _SCHEMA_RESULT_CACHE.get(key)
        if value is None:
            return None
        _SCHEMA_RESULT_CACHE.move_to_end(key)
    return deepcopy(value)


def _is_schema_result_cacheable(value: dict[str, Any]) -> bool:
    if not value.get("extraction_successful"):
        return False
    warnings = value.get("schema_selector_warnings")
    return not (
        isinstance(warnings, list)
        and any(isinstance(warning, dict) and warning.get("warning") == "no_matches" for warning in warnings)
    )


def _schema_cache_put(key: str, value: dict[str, Any]) -> None:
    if not _is_schema_result_cacheable(value):
        return
    stored = deepcopy(value)
    with _SCHEMA_CACHE_LOCK:
        _SCHEMA_RESULT_CACHE[key] = stored
        _SCHEMA_RESULT_CACHE.move_to_end(key)
        while len(_SCHEMA_RESULT_CACHE) > _SCHEMA_RESULT_CACHE_MAX:
            _SCHEMA_RESULT_CACHE.popitem(last=False)


def _cluster_cache_get(
    key: str,
    *,
    increment_counter: Callable[..., None] | None = None,
) -> list[float] | None:
    result: list[float] | None
    with _CLUSTER_CACHE_LOCK:
        value = _CLUSTER_EMBED_CACHE.get(key)
        if value is None:
            result = None
        else:
            _CLUSTER_EMBED_CACHE.move_to_end(key)
            result = list(value)
    if increment_counter is not None:
        emit_callback_counter(
            increment_counter,
            "extraction_cluster_cache_total",
            labels={"cache": "embedding", "result": "hit" if result is not None else "miss"},
        )
    return result


def _cluster_cache_put(key: str, value: list[float]) -> None:
    with _CLUSTER_CACHE_LOCK:
        _CLUSTER_EMBED_CACHE[key] = list(value)
        _CLUSTER_EMBED_CACHE.move_to_end(key)
        while len(_CLUSTER_EMBED_CACHE) > _CLUSTER_EMBED_CACHE_MAX:
            _CLUSTER_EMBED_CACHE.popitem(last=False)


def get_extraction_cache_stats() -> dict[str, int]:
    with _CLUSTER_CACHE_LOCK:
        cluster_size = len(_CLUSTER_EMBED_CACHE)
    with _SCHEMA_CACHE_LOCK:
        schema_size = len(_SCHEMA_RESULT_CACHE)
    stats = {
        "cluster_embedding_cache_size": cluster_size,
        "schema_result_cache_size": schema_size,
    }
    stats.update(get_throttle_stats())
    try:
        stats.update(get_selector_cache_stats())
    except _SELECTOR_CACHE_NONCRITICAL_EXCEPTIONS:
        return stats
    return stats


def clear_extraction_caches() -> None:
    """Clear extraction data caches without replacing live throttle objects."""

    with _CLUSTER_CACHE_LOCK:
        _CLUSTER_EMBED_CACHE.clear()
    with _SCHEMA_CACHE_LOCK:
        _SCHEMA_RESULT_CACHE.clear()
    try:
        clear_selector_caches()
    except _SELECTOR_CACHE_NONCRITICAL_EXCEPTIONS:
        return
