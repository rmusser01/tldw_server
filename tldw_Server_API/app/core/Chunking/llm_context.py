"""Scoped LLM override helpers used by the chunking pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from .error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS

_LLM_UNSET = object()


@contextmanager
def llm_override_scope(context: Any, llm_call_func: Any = None, llm_config: Any = None) -> Iterator[None]:
    """Temporarily install per-call LLM overrides on a chunker context."""
    previous = getattr(context._thread_local, "llm_overrides", _LLM_UNSET)
    apply_overrides = (llm_call_func is not None) or (llm_config is not None)
    if apply_overrides:
        override_func = llm_call_func if llm_call_func is not None else _LLM_UNSET
        override_config = llm_config if llm_config is not None else _LLM_UNSET
        context._thread_local.llm_overrides = (override_func, override_config)
    try:
        yield
    finally:
        if apply_overrides:
            if previous is _LLM_UNSET:
                with suppress(CHUNKER_NONCRITICAL_EXCEPTIONS):
                    delattr(context._thread_local, "llm_overrides")
            else:
                context._thread_local.llm_overrides = previous
