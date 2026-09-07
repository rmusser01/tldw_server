"""Paragraph span detection shared by hierarchical and process-text chunking."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from ..error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS as _CHUNKER_NONCRITICAL_EXCEPTIONS


def compute_paragraph_spans(
    text: str,
    template: dict[str, Any] | None = None,
) -> list[tuple[int, int, str]]:
    """Compute paragraph/block spans with kinds and optional template boundaries.

    Lightweight port of the structure detection used by the legacy utility to
    avoid maintaining two libraries. Recognizes blank lines, ATX headers, hrules,
    simple lists, code fences, markdown tables, and optional custom boundary rules.
    """
    spans: list[tuple[int, int, str]] = []
    if not text:
        return spans

    # Compile template boundary patterns if provided, with safety limits
    template_patterns: list[tuple[str, re.Pattern]] = []
    try:
        boundaries = (template or {}).get("boundaries") or []
        # Safety caps aligned with API validator: at most 20 rules
        MAX_RULES = 20
        MAX_PATTERN_LEN = 256
        from ..regex_safety import check_pattern, compile_flags

        for rule in boundaries[:MAX_RULES]:
            try:
                kind = str(rule.get("kind") or "template")
                pattern = str(rule.get("pattern") or "")
                if not pattern:
                    continue
                if len(pattern) > MAX_PATTERN_LEN:
                    logger.warning(f"Skipping overlong boundary pattern (>{MAX_PATTERN_LEN} chars)")
                    continue
                # Safety check
                err = check_pattern(pattern, max_len=MAX_PATTERN_LEN)
                if err:
                    logger.warning(f"Skipping boundary pattern due to safety check: {err}")
                    continue
                flags_val, ferr = compile_flags(str(rule.get("flags") or ""))
                flags = flags_val if ferr is None else 0
                compiled = re.compile(pattern, flags)
                template_patterns.append((kind, compiled))
            except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Ignoring invalid boundary rule: {e}")
    except _CHUNKER_NONCRITICAL_EXCEPTIONS:
        template_patterns = []

    lines = text.splitlines(keepends=True)
    offsets: list[tuple[int, int, str]] = []
    pos = 0
    for line in lines:
        start, end = pos, pos + len(line)
        offsets.append((start, end, line))
        pos = end
    if pos < len(text):
        offsets.append((pos, len(text), text[pos : len(text)]))

    def match_template(s: str) -> str | None:
        # Use safe search with optional timeouts and RE2 when available
        try:
            from ..regex_safety import safe_search
        except _CHUNKER_NONCRITICAL_EXCEPTIONS:
            safe_search = None  # type: ignore
        for kind, pat in template_patterns:
            try:
                ok = False
                ok = bool(safe_search(pat, s)) if safe_search is not None else pat.search(s) is not None
                if ok:
                    return kind
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                continue
        return None

    def classify_line(s: str) -> str | None:
        k = match_template(s)
        if k:
            # Custom kinds (e.g., "bold_subsection") are template-driven; no built-in detection.
            return k
        if re.match(r"^\s*$", s):
            return "blank"
        if re.match(r"^\s*#{1,6}\s", s):
            return "header_atx"
        if re.match(r"^\s*(\*{3,}|-{3,}|_{3,})\s*$", s):
            return "hr"
        if re.match(r"^\s*(`{3,}|~{3,})", s):
            return "code_fence"
        if re.match(r"^\s*([-+*])\s+\S", s):
            return "list_unordered"
        if re.match(r"^\s*\d+[\.)]\s+\S", s):
            return "list_ordered"
        if re.match(r"^\s*\|.*\|\s*$", s):
            return "table_md"
        return None

    buf_start: int | None = None
    code_fence_start: int | None = None
    code_fence_marker: str | None = None
    for start, end, content in offsets:
        if code_fence_start is not None:
            try:
                marker = code_fence_marker or ""
                if marker:
                    stripped = content.strip()
                    # Closing fence must use the same fence char and be at least as long as the opener.
                    if stripped and all(ch == marker[0] for ch in stripped) and len(stripped) >= len(marker):
                        spans.append((code_fence_start, end, "code_fence"))
                        code_fence_start = None
                        code_fence_marker = None
                continue
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                spans.append((code_fence_start, end, "code_fence"))
                code_fence_start = None
                code_fence_marker = None
                continue

        kind = classify_line(content)
        if kind == "blank":
            if buf_start is not None:
                spans.append((buf_start, start, "paragraph"))
                buf_start = None
            spans.append((start, end, "blank"))
        elif kind == "code_fence":
            if buf_start is not None:
                spans.append((buf_start, start, "paragraph"))
                buf_start = None
            try:
                marker_match = re.match(r"^\s*(`{3,}|~{3,})", content)
                code_fence_marker = marker_match.group(1) if marker_match else "```"
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                code_fence_marker = "```"
            code_fence_start = start
        elif kind is not None:
            if buf_start is not None:
                spans.append((buf_start, start, "paragraph"))
                buf_start = None
            spans.append((start, end, kind))
        else:
            if buf_start is None:
                buf_start = start
    if buf_start is not None:
        spans.append((buf_start, len(text), "paragraph"))
    if code_fence_start is not None:
        spans.append((code_fence_start, len(text), "code_fence"))
    return spans
