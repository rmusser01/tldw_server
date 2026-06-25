from __future__ import annotations

import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chunking.error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.Chunking.exceptions import InvalidInputError
from tldw_Server_API.app.core.Chunking.option_utils import _coerce_bool_option
from tldw_Server_API.app.core.Chunking.process_text.models import (
    ProcessTextContext,
    ResolvedProcessOptions,
)


METHOD_OPTION_EXCLUDES = {
    "method",
    "max_size",
    "overlap",
    "language",
    "hierarchical",
    "hierarchical_template",
    "multi_level",
    "timecode_map",
    "enable_frontmatter_parsing",
    "frontmatter_sentinel_key",
    "adaptive",
    "base_adaptive_chunk_size",
    "min_adaptive_chunk_size",
    "max_adaptive_chunk_size",
    "adaptive_overlap",
    "base_overlap",
    "max_adaptive_overlap",
    "code_mode",
    "align_text_to_source",
}


def resolve_process_options(
    context: ProcessTextContext,
    processed_text: str,
    options: dict[str, Any],
) -> ResolvedProcessOptions:
    requested_method = context._normalize_method_argument(options.get("method"))
    method = requested_method or context.config.default_method.value

    max_size_opt = options.get("max_size")
    if max_size_opt is None:
        max_size = context.config.default_max_size
    else:
        try:
            max_size = int(max_size_opt)
        except CHUNKER_NONCRITICAL_EXCEPTIONS as exc:
            raise InvalidInputError(f"Invalid max_size value: {max_size_opt}") from exc
        if max_size <= 0:
            raise InvalidInputError(f"max_size must be positive, got {max_size}")

    overlap_opt = options.get("overlap")
    if overlap_opt is None:
        overlap = context.config.default_overlap
    else:
        try:
            overlap = int(overlap_opt)
        except CHUNKER_NONCRITICAL_EXCEPTIONS as exc:
            raise InvalidInputError(f"Invalid overlap value: {overlap_opt}") from exc
        if overlap < 0:
            logger.warning(f"Negative overlap ({overlap}) adjusted to 0 in process_text")
            overlap = 0

    language = options.get("language")
    # Support explicit auto/detect override and default autodetect when not provided
    if (not language) or (isinstance(language, str) and language.strip().lower() in {"auto", "detect"}):
        # Lightweight language detection by Unicode script ranges
        try:
            if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", processed_text):
                language = "ja"       # Hiragana/Katakana (Japanese)
            elif re.search(r"[\u4e00-\u9fff]", processed_text):
                language = "zh"       # CJK Unified Ideographs (Chinese)
            elif re.search(r"[\u0e00-\u0e7f]", processed_text):
                language = "th"       # Thai
            elif re.search(r"[\u0900-\u097f]", processed_text):
                language = "hi"       # Devanagari (Hindi)
            elif re.search(r"[\u0400-\u04ff]", processed_text):
                language = "ru"       # Cyrillic (Russian)
            elif re.search(r"[\uac00-\ud7af]", processed_text):
                language = "ko"       # Hangul (Korean)
            elif re.search(r"[\u0600-\u06ff]", processed_text):
                language = "ar"       # Arabic
            else:
                language = context.config.language
        except CHUNKER_NONCRITICAL_EXCEPTIONS:
            language = context.config.language

    method = context._resolve_method(method, language, options)
    method_lower = str(method).lower() if method is not None else ""
    method_options = {
        k: v for k, v in options.items() if k not in METHOD_OPTION_EXCLUDES
    }
    code_mode_for_method: str | None = None
    if "code_mode" in options:
        try:
            cm_val = options.get("code_mode")
            if cm_val is not None:
                code_mode_for_method = str(cm_val).lower()
        except CHUNKER_NONCRITICAL_EXCEPTIONS:
            code_mode_for_method = None
    elif method_lower == "code_ast":
        code_mode_for_method = "ast"
    elif method_lower == "code":
        code_mode_for_method = "auto"
    method_options_for_chunk: dict[str, Any] = dict(method_options)
    if code_mode_for_method is not None and method_lower in ("code", "code_ast"):
        method_options_for_chunk["code_mode"] = code_mode_for_method

    adaptive = _coerce_bool_option(options.get("adaptive"), False)
    if adaptive and method not in ("semantic", "json", "xml", "ebook_chapters", "rolling_summarize"):
        try:
            base_adaptive = int(options.get("base_adaptive_chunk_size") or max_size)
            min_adaptive = int(options.get("min_adaptive_chunk_size") or max_size)
            max_adaptive_hi = int(options.get("max_adaptive_chunk_size") or max_size)
            # Very rough heuristic: scale with document size
            density = max(0.0, min(3.0, len(processed_text) / 10000.0))
            scaled = int(base_adaptive * (1.0 + 0.2 * density))
            max_size = max(min_adaptive, min(max_adaptive_hi, scaled))
            # Optional adaptive overlap tuned by density
            if _coerce_bool_option(options.get("adaptive_overlap"), False):
                try:
                    base_overlap = int(options.get("base_overlap") or overlap or 0)
                    max_overlap = int(options.get("max_adaptive_overlap") or max(0, base_overlap + 100))
                    # Increase overlap slightly for denser/longer docs; cap to avoid waste
                    tuned = int(base_overlap + (density * 10))
                    overlap = max(0, min(max_overlap, tuned))
                except CHUNKER_NONCRITICAL_EXCEPTIONS:
                    pass
        except CHUNKER_NONCRITICAL_EXCEPTIONS:
            pass

    hierarchical = _coerce_bool_option(options.get("hierarchical"), False)
    hier_template = options.get("hierarchical_template") if isinstance(options.get("hierarchical_template"), dict) else None
    multi_level = _coerce_bool_option(options.get("multi_level"), False) and method in ("words", "sentences") and not (hierarchical or hier_template)
    align_text_to_source = _coerce_bool_option(options.get("align_text_to_source"), True)

    return ResolvedProcessOptions(
        method=method,
        method_lower=method_lower,
        max_size=max_size,
        overlap=overlap,
        language=language,
        adaptive=adaptive,
        hierarchical=hierarchical,
        hier_template=hier_template,
        multi_level=multi_level,
        code_mode_for_method=code_mode_for_method,
        method_options_for_chunk=method_options_for_chunk,
        align_text_to_source=align_text_to_source,
    )
