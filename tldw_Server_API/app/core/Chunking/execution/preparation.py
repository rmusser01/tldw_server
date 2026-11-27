"""Resolve equivalent setup while retaining the three option policies."""

from __future__ import annotations

from typing import Any

from ..base import ChunkingMethod
from ..error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from ..option_utils import _coerce_bool_option
from .models import ExecutionContext, ExecutionHooks, ExecutionKind, PreparedExecution


def prepare_execution(
    context: ExecutionContext,
    *,
    method: Any,
    max_size: Any,
    overlap: Any,
    language: Any,
    user_options: dict[str, Any],
    kind: ExecutionKind,
    hooks: ExecutionHooks,
) -> PreparedExecution:
    """Prepare supplied options without sanitization, cache or strategy work."""
    options_raw: dict[str, Any] = dict(user_options)
    align_text_to_source = False
    align_text_to_source_opt = None
    if kind == "text":
        options_raw.pop("align_text_to_source", None)
    elif kind == "metadata":
        align_text_to_source_opt = options_raw.pop("align_text_to_source", None)
        align_text_to_source = _coerce_bool_option(align_text_to_source_opt, False)
    method = context._normalize_method_argument(method) or context.config.default_method.value
    max_size = max_size if max_size is not None else context.config.default_max_size
    overlap = overlap if overlap is not None else context.config.default_overlap
    language = language or context.config.language
    method = context._resolve_method(method, language, options_raw)
    options_raw = context._apply_option_aliases(method, options_raw)
    overlap = context._override_overlap_for_method(method, overlap, options_raw)
    try:
        if isinstance(max_size, int) and isinstance(overlap, int):
            if overlap < 0:
                hooks.get_logger().warning(f"Negative overlap ({overlap}) adjusted to 0")
                overlap = 0
            if max_size <= 0:
                pass
            elif overlap >= max_size:
                hooks.get_logger().warning(f"Overlap ({overlap}) >= max_size ({max_size}); adjusting to max_size - 1")
                overlap = max_size - 1
    except CHUNKER_NONCRITICAL_EXCEPTIONS as error:
        if kind == "text":
            hooks.get_logger().debug(f"Overlap adjustment check failed: {error}")

    tokenizer_override = options_raw.get("tokenizer_name")
    if tokenizer_override is None:
        tokenizer_override = options_raw.get("tokenizer_name_or_path")
    if isinstance(tokenizer_override, str):
        tokenizer_override = tokenizer_override.strip() or None
    use_per_call_strategy = bool(tokenizer_override) and method == ChunkingMethod.TOKENS.value
    strategy_options: dict[str, Any] = dict(options_raw)
    if kind == "metadata" and align_text_to_source_opt is not None:
        strategy_options["align_text_to_source"] = align_text_to_source
    strategy_options.pop("code_mode", None)
    strategy_options.pop("tokenizer_name", None)
    strategy_options.pop("tokenizer_name_or_path", None)
    return PreparedExecution(
        method=method,
        max_size=max_size,
        overlap=overlap,
        language=language,
        tokenizer_override=tokenizer_override,
        options_raw=options_raw,
        strategy_options=strategy_options,
        use_per_call_strategy=use_per_call_strategy,
        align_text_to_source=align_text_to_source,
    )
