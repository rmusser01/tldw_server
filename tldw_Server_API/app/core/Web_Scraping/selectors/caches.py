"""Bounded thread-safe caches for compiled CSS and XPath selectors."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

_SELECTOR_CACHE_MAX = 512
_XPATH_SELECTOR_CACHE: OrderedDict[str, Any] = OrderedDict()
_CSS_SELECTOR_CACHE: OrderedDict[str, Any] = OrderedDict()
_SELECTOR_CACHE_LOCK = Lock()


def get_selector_cache_stats() -> dict[str, int]:
    with _SELECTOR_CACHE_LOCK:
        return {
            "selector_xpath_cache_size": len(_XPATH_SELECTOR_CACHE),
            "selector_css_cache_size": len(_CSS_SELECTOR_CACHE),
        }


def clear_selector_caches() -> None:
    with _SELECTOR_CACHE_LOCK:
        _XPATH_SELECTOR_CACHE.clear()
        _CSS_SELECTOR_CACHE.clear()


def _selector_cache_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
    with _SELECTOR_CACHE_LOCK:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value


def _selector_cache_put(cache: OrderedDict[str, Any], key: str, value: Any) -> None:
    with _SELECTOR_CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _SELECTOR_CACHE_MAX:
            cache.popitem(last=False)


def _get_xpath_selector(key: str) -> Any | None:
    return _selector_cache_get(_XPATH_SELECTOR_CACHE, key)


def _put_xpath_selector(key: str, value: Any) -> None:
    _selector_cache_put(_XPATH_SELECTOR_CACHE, key, value)


def _get_css_selector(key: str) -> Any | None:
    return _selector_cache_get(_CSS_SELECTOR_CACHE, key)


def _put_css_selector(key: str, value: Any) -> None:
    _selector_cache_put(_CSS_SELECTOR_CACHE, key, value)
