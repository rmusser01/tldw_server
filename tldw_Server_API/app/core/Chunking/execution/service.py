"""Path-specific execution sequencing; mutable facilities remain on the context."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import nullcontext, suppress
from typing import Any

from ..base import ChunkingMethod, ChunkResult
from ..error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS as _CHUNKER_NONCRITICAL_EXCEPTIONS
from ..exceptions import ChunkingError
from ..option_utils import _coerce_bool_option
from .models import ExecutionContext, ExecutionHooks

# Keep each suppress tuple nested, as in the facade baseline: expanding it
# changes ExceptionGroup handling. Narrow arg-type ignores preserve that policy.


class ChunkExecutionService:
    """Execute a call through live context facilities and observability hooks."""

    def __init__(self, context: ExecutionContext, hooks: ExecutionHooks) -> None:
        """Retain references only; validation and all effects belong to execution."""
        self.context = context
        self.hooks = hooks

    def chunk_text(
        self,
        text: str,
        method: str | None = None,
        max_size: int | None = None,
        overlap: int | None = None,
        language: str | None = None,
        **options: Any,
    ) -> list[str]:
        """
        Chunk text using the specified method.

        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options

        Returns:
            List of text chunks

        Raises:
            InvalidInputError: If input validation fails
            InvalidChunkingMethodError: If method is not supported
            ChunkingError: For other chunking errors
        """
        # Sanitize and validate input
        text = self.context._sanitize_input(text)

        if not text.strip():
            self.hooks.get_logger().debug("Empty text provided, returning empty list")
            return []

        self.context._enforce_text_size(text, source="chunk_text")

        # Use defaults if not specified
        options_raw: dict[str, Any] = dict(options)
        options_raw.pop("align_text_to_source", None)
        method = self.context._normalize_method_argument(method) or self.context.config.default_method.value
        max_size = max_size if max_size is not None else self.context.config.default_max_size
        overlap = overlap if overlap is not None else self.context.config.default_overlap
        # Harden overlap to avoid non-progressing loops in strategies
        language = language or self.context.config.language
        method = self.context._resolve_method(method, language, options_raw)
        options_raw = self.context._apply_option_aliases(method, options_raw)
        overlap = self.context._override_overlap_for_method(method, overlap, options_raw)
        try:
            if isinstance(max_size, int) and isinstance(overlap, int):
                if overlap < 0:
                    self.hooks.get_logger().warning(f"Negative overlap ({overlap}) adjusted to 0")
                    overlap = 0
                if max_size <= 0:
                    # Strategy will raise on invalid max_size; leave as-is here
                    pass
                elif overlap >= max_size:
                    self.hooks.get_logger().warning(
                        f"Overlap ({overlap}) >= max_size ({max_size}); adjusting to max_size - 1"
                    )
                    overlap = max_size - 1
        except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:
            self.hooks.get_logger().debug(f"Overlap adjustment check failed: {e}")

        # Configuration-only flags (removed before invoking strategy)
        tokenizer_override = options_raw.get("tokenizer_name")
        if tokenizer_override is None:
            tokenizer_override = options_raw.get("tokenizer_name_or_path")
        if isinstance(tokenizer_override, str):
            tokenizer_override = tokenizer_override.strip() or None
        use_per_call_strategy = bool(tokenizer_override) and method == ChunkingMethod.TOKENS.value
        strategy_options: dict[str, Any] = dict(options_raw)
        strategy_options.pop("code_mode", None)
        strategy_options.pop("tokenizer_name", None)
        strategy_options.pop("tokenizer_name_or_path", None)

        # Check cache if enabled
        cache_key = None
        cache_allowed = False
        cache_reason = "disabled"
        if self.context._cache is not None:
            try:
                tlen = len(text)
                min_len = getattr(self.context.config, "min_text_length_to_cache", 0)
                max_len = getattr(
                    self.context.config,
                    "max_text_length_to_cache",
                    getattr(self.context.config, "cache_max_text_length", 2_000_000),
                )
                if tlen < int(min_len):
                    cache_reason = "skipped_min_len"
                elif tlen > int(max_len):
                    cache_reason = "skipped_max_len"
                else:
                    cache_allowed = True
                    cache_reason = "allowed"
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                cache_reason = "policy_error"
        # Attempt get if allowed
        effective_llm_call, effective_llm_cfg = self.context._get_effective_llm_hooks()
        llm_signature = self.context._get_llm_signature(
            llm_call_func=effective_llm_call,
            llm_config=effective_llm_cfg,
        )
        if self.context._cache is not None and cache_allowed:
            cache_key = self.context._get_cache_key(
                text,
                method,
                max_size,
                overlap,
                language,
                options_raw,
                llm_signature=llm_signature,
            )
            cached_result: list[str] | None = self.context._cache.get(cache_key)
            if cached_result is not None:
                with suppress(_CHUNKER_NONCRITICAL_EXCEPTIONS):  # type: ignore[arg-type]
                    self.hooks.increment_counter(
                        "chunker_cache_get_total", labels={"result": "hit", "reason": cache_reason}
                    )
                self.hooks.get_logger().debug("Returning cached result")
                return cached_result
            else:
                with suppress(_CHUNKER_NONCRITICAL_EXCEPTIONS):  # type: ignore[arg-type]
                    self.hooks.increment_counter(
                        "chunker_cache_get_total", labels={"result": "miss", "reason": cache_reason}
                    )
        else:
            # cache disabled or not allowed by policy
            with suppress(_CHUNKER_NONCRITICAL_EXCEPTIONS):  # type: ignore[arg-type]
                self.hooks.increment_counter(
                    "chunker_cache_get_total", labels={"result": "skip", "reason": cache_reason}
                )

        strategy_lock = self.context._strategy_lock if self.context._strategy_cache_mode == "shared" else nullcontext()
        try:
            with strategy_lock:
                # Get strategy lazily per cache mode; force per-call strategy for tokenizer overrides
                strategy = (
                    self.context._create_strategy_instance(method)
                    if use_per_call_strategy
                    else self.context._get_strategy_for_call(method)
                )
                self.context._sync_strategy_llm(
                    strategy, llm_call_func=effective_llm_call, llm_config=effective_llm_cfg
                )
                # Allow token strategy to switch tokenizer dynamically
                if method == ChunkingMethod.TOKENS.value and tokenizer_override:
                    try:
                        if getattr(strategy, "tokenizer_name", None) != tokenizer_override:
                            strategy.tokenizer_name = tokenizer_override
                            if hasattr(strategy, "_tokenizer"):
                                strategy._tokenizer = None
                            if hasattr(strategy, "_tokenizer_init_attempted"):
                                strategy._tokenizer_init_attempted = False
                    except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                        self.hooks.get_logger().debug("Failed to update tokenizer override", exc_info=True)

                # Update strategy language if different
                if language != strategy.language:
                    self.context._apply_strategy_language(strategy, language)

                # Perform chunking
                self.hooks.get_logger().debug(
                    f"Chunking with method={method}, max_size={max_size}, " f"overlap={overlap}, language={language}"
                )

                chunks: list[str] = strategy.chunk(text, max_size, overlap, **strategy_options)

            # Cache result if enabled
            if self.context._cache is not None and len(chunks) > 0 and cache_allowed and cache_key is not None:
                try:
                    self.context._cache.put(cache_key, chunks)
                    with suppress(_CHUNKER_NONCRITICAL_EXCEPTIONS):  # type: ignore[arg-type]
                        self.hooks.increment_counter(
                            "chunker_cache_put_total",
                            labels={"result": "stored", "reason": cache_reason or "allowed"},
                        )
                except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                    # Defensive: never fail chunking due to cache policy
                    with suppress(_CHUNKER_NONCRITICAL_EXCEPTIONS):  # type: ignore[arg-type]
                        self.hooks.increment_counter(
                            "chunker_cache_put_total",
                            labels={"result": "error", "reason": cache_reason or "allowed"},
                        )
            elif self.context._cache is not None and len(chunks) > 0:
                with suppress(_CHUNKER_NONCRITICAL_EXCEPTIONS):  # type: ignore[arg-type]
                    self.hooks.increment_counter(
                        "chunker_cache_put_total", labels={"result": "skipped", "reason": cache_reason}
                    )

            if getattr(self.context.config, "verbose_logging", False):
                self.hooks.get_logger().info(f"Created {len(chunks)} chunks using {method} method")
            else:
                self.hooks.get_logger().debug(f"Created {len(chunks)} chunks using {method} method")
            return chunks

        except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:
            self.hooks.get_logger().error(f"Chunking failed: {e}")
            if isinstance(e, ChunkingError):
                raise
            raise ChunkingError(f"Chunking failed: {str(e)}") from e

    def chunk_text_with_metadata(
        self,
        text: str,
        method: str | None = None,
        max_size: int | None = None,
        overlap: int | None = None,
        language: str | None = None,
        **options: Any,
    ) -> list[ChunkResult]:
        """
        Chunk text and return results with metadata.

        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options

        Returns:
            List of ChunkResult objects with text and metadata.
            Note: Extractive strategies (words/sentences/paragraphs) return source-aligned
            text by default. Set `align_text_to_source=False` to keep normalized text
            when supported, or `align_text_to_source=True` to force source slicing for
            transforming strategies.
        """
        # Sanitize input
        text = self.context._sanitize_input(text)
        if not text.strip():
            self.hooks.get_logger().debug("Empty text provided, returning empty list")
            return []
        self.context._enforce_text_size(text, source="chunk_text_with_metadata")

        # Use defaults if not specified
        options_raw: dict[str, Any] = dict(options)
        align_text_to_source_opt = options_raw.pop("align_text_to_source", None)
        align_text_to_source = _coerce_bool_option(align_text_to_source_opt, False)
        method = self.context._normalize_method_argument(method) or self.context.config.default_method.value
        max_size = max_size if max_size is not None else self.context.config.default_max_size
        overlap = overlap if overlap is not None else self.context.config.default_overlap
        language = language or self.context.config.language
        method = self.context._resolve_method(method, language, options_raw)
        options_raw = self.context._apply_option_aliases(method, options_raw)
        overlap = self.context._override_overlap_for_method(method, overlap, options_raw)
        # Harden overlap to avoid non-progressing loops in strategies
        try:
            if isinstance(max_size, int) and isinstance(overlap, int):
                if overlap < 0:
                    self.hooks.get_logger().warning(f"Negative overlap ({overlap}) adjusted to 0")
                    overlap = 0
                if max_size <= 0:
                    # Strategy will raise on invalid max_size; leave as-is here
                    pass
                elif overlap >= max_size:
                    self.hooks.get_logger().warning(
                        f"Overlap ({overlap}) >= max_size ({max_size}); adjusting to max_size - 1"
                    )
                    overlap = max_size - 1
        except _CHUNKER_NONCRITICAL_EXCEPTIONS:
            pass

        tokenizer_override = options_raw.get("tokenizer_name")
        if tokenizer_override is None:
            tokenizer_override = options_raw.get("tokenizer_name_or_path")
        if isinstance(tokenizer_override, str):
            tokenizer_override = tokenizer_override.strip() or None
        use_per_call_strategy = bool(tokenizer_override) and method == ChunkingMethod.TOKENS.value
        strategy_options: dict[str, Any] = dict(options_raw)
        if align_text_to_source_opt is not None:
            # Allow strategies to opt into/out of source-aligned text
            strategy_options["align_text_to_source"] = align_text_to_source
        strategy_options.pop("code_mode", None)
        strategy_options.pop("tokenizer_name", None)
        strategy_options.pop("tokenizer_name_or_path", None)

        effective_llm_call, effective_llm_cfg = self.context._get_effective_llm_hooks()
        strategy_lock = self.context._strategy_lock if self.context._strategy_cache_mode == "shared" else nullcontext()
        try:
            with strategy_lock:
                # Get strategy lazily (supports factory registration); force per-call strategy for tokenizer overrides
                strategy = (
                    self.context._create_strategy_instance(method)
                    if use_per_call_strategy
                    else self.context._get_strategy_for_call(method)
                )
                self.context._sync_strategy_llm(
                    strategy, llm_call_func=effective_llm_call, llm_config=effective_llm_cfg
                )
                if method == ChunkingMethod.TOKENS.value and tokenizer_override:
                    try:
                        if getattr(strategy, "tokenizer_name", None) != tokenizer_override:
                            strategy.tokenizer_name = tokenizer_override
                            if hasattr(strategy, "_tokenizer"):
                                strategy._tokenizer = None
                            if hasattr(strategy, "_tokenizer_init_attempted"):
                                strategy._tokenizer_init_attempted = False
                    except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                        self.hooks.get_logger().debug("Failed to update tokenizer override", exc_info=True)

                # Update strategy language if different
                if language != strategy.language:
                    self.context._apply_strategy_language(strategy, language)

                # Get chunks with metadata
                results: list[ChunkResult] = strategy.chunk_with_metadata(text, max_size, overlap, **strategy_options)
        except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:
            self.hooks.get_logger().error(f"Chunking with metadata failed: {e}")
            if isinstance(e, ChunkingError):
                raise
            raise ChunkingError(f"Chunking failed: {str(e)}") from e

        # Optionally align emitted text with the source span (opt-in)
        if align_text_to_source and isinstance(results, list):
            text_len = len(text)
            for item in results:
                try:
                    metadata = getattr(item, "metadata", None)
                    if metadata is None:
                        continue
                    start_char = getattr(metadata, "start_char", None)
                    end_char = getattr(metadata, "end_char", None)
                    if not isinstance(start_char, int) or not isinstance(end_char, int):
                        continue
                    start_char = max(0, min(start_char, text_len))
                    end_char = max(start_char, min(end_char, text_len))
                    item.text = text[start_char:end_char]
                except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                    continue
        if getattr(self.context.config, "verbose_logging", False):
            self.hooks.get_logger().info(f"Created {len(results)} chunks with metadata using {method} method")
        else:
            self.hooks.get_logger().debug(f"Created {len(results)} chunks with metadata using {method} method")
        return results

    def iter_chunks(
        self,
        text: str,
        method: str | None = None,
        max_size: int | None = None,
        overlap: int | None = None,
        language: str | None = None,
        **options: Any,
    ) -> Generator[str, None, None]:
        """
        Memory-efficient generator for chunking large texts.

        Args:
            text: Text to chunk
            method: Chunking method (uses default if not specified)
            max_size: Maximum chunk size (uses default if not specified)
            overlap: Overlap between chunks (uses default if not specified)
            language: Language of the text (uses default if not specified)
            **options: Additional method-specific options

        Yields:
            Individual text chunks
        """
        # Sanitize input
        text = self.context._sanitize_input(text)
        self.context._enforce_text_size(text, source="chunk_text_generator")

        # Use defaults if not specified
        options_raw: dict[str, Any] = dict(options)
        method = self.context._normalize_method_argument(method) or self.context.config.default_method.value
        max_size = max_size if max_size is not None else self.context.config.default_max_size
        overlap = overlap if overlap is not None else self.context.config.default_overlap
        language = language or self.context.config.language
        method = self.context._resolve_method(method, language, options_raw)
        options_raw = self.context._apply_option_aliases(method, options_raw)
        overlap = self.context._override_overlap_for_method(method, overlap, options_raw)
        # Align overlap handling with primary chunk_text path
        try:
            if isinstance(max_size, int) and isinstance(overlap, int):
                if overlap < 0:
                    self.hooks.get_logger().warning(f"Negative overlap ({overlap}) adjusted to 0")
                    overlap = 0
                if max_size <= 0:
                    pass
                elif overlap >= max_size:
                    self.hooks.get_logger().warning(
                        f"Overlap ({overlap}) >= max_size ({max_size}); adjusting to max_size - 1"
                    )
                    overlap = max_size - 1
        except _CHUNKER_NONCRITICAL_EXCEPTIONS:
            pass

        tokenizer_override = options_raw.get("tokenizer_name")
        if tokenizer_override is None:
            tokenizer_override = options_raw.get("tokenizer_name_or_path")
        if isinstance(tokenizer_override, str):
            tokenizer_override = tokenizer_override.strip() or None
        use_per_call_strategy = bool(tokenizer_override) and method == ChunkingMethod.TOKENS.value
        strategy_options: dict[str, Any] = dict(options_raw)
        strategy_options.pop("code_mode", None)
        strategy_options.pop("tokenizer_name", None)
        strategy_options.pop("tokenizer_name_or_path", None)

        effective_llm_call, effective_llm_cfg = self.context._get_effective_llm_hooks()
        use_ephemeral_strategy = self.context._strategy_cache_mode == "shared"
        strategy_lock = self.context._strategy_lock if self.context._strategy_cache_mode == "shared" else nullcontext()
        with strategy_lock:
            # A yielded generator must not keep the shared strategy locked across
            # consumer-controlled pauses, so shared mode uses a per-call instance here.
            strategy = (
                self.context._create_strategy_instance(method)
                if (use_per_call_strategy or use_ephemeral_strategy)
                else self.context._get_strategy_for_call(method)
            )
            self.context._sync_strategy_llm(strategy, llm_call_func=effective_llm_call, llm_config=effective_llm_cfg)
            if method == ChunkingMethod.TOKENS.value and tokenizer_override:
                try:
                    if getattr(strategy, "tokenizer_name", None) != tokenizer_override:
                        strategy.tokenizer_name = tokenizer_override
                        if hasattr(strategy, "_tokenizer"):
                            strategy._tokenizer = None
                        if hasattr(strategy, "_tokenizer_init_attempted"):
                            strategy._tokenizer_init_attempted = False
                except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                    self.hooks.get_logger().debug("Failed to update tokenizer override", exc_info=True)

            # Update strategy language if different
            if language != strategy.language:
                self.context._apply_strategy_language(strategy, language)

            # Use generator method when available, otherwise fall back to eager chunking
            chunk_gen = getattr(strategy, "chunk_generator", None)
            if callable(chunk_gen):
                chunk_iter = chunk_gen(text, max_size, overlap, **strategy_options)
            else:
                self.hooks.get_logger().debug(
                    f"{strategy.__class__.__name__} lacks chunk_generator; falling back to chunk()"
                )
                chunk_iter = iter(strategy.chunk(text, max_size, overlap, **strategy_options))

        yield from chunk_iter
