"""Strategy setup through supplied decisions; no cache or lock ownership."""

from __future__ import annotations

from typing import Any

from ..base import ChunkingMethod
from ..error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from .models import ExecutionContext, ExecutionHooks, PreparedExecution


def prepare_strategy(
    context: ExecutionContext,
    prepared: PreparedExecution,
    *,
    effective_llm: tuple[Any, Any],
    ephemeral: bool,
    hooks: ExecutionHooks,
) -> Any:
    """Select and synchronize without rereading mode or resolving LLM hooks."""
    method = prepared.method
    tokenizer_override = prepared.tokenizer_override
    language = prepared.language
    strategy = (
        context._create_strategy_instance(method)
        if (prepared.use_per_call_strategy or ephemeral)
        else context._get_strategy_for_call(method)
    )
    effective_llm_call, effective_llm_cfg = effective_llm
    context._sync_strategy_llm(strategy, llm_call_func=effective_llm_call, llm_config=effective_llm_cfg)
    if method == ChunkingMethod.TOKENS.value and tokenizer_override:
        try:
            if getattr(strategy, "tokenizer_name", None) != tokenizer_override:
                strategy.tokenizer_name = tokenizer_override
                if hasattr(strategy, "_tokenizer"):
                    strategy._tokenizer = None
                if hasattr(strategy, "_tokenizer_init_attempted"):
                    strategy._tokenizer_init_attempted = False
        except CHUNKER_NONCRITICAL_EXCEPTIONS:
            hooks.get_logger().debug("Failed to update tokenizer override", exc_info=True)
    if language != strategy.language:
        context._apply_strategy_language(strategy, language)
    return strategy
