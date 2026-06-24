from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse


READING_ALLOWED_STATUSES = frozenset({"saved", "reading", "read", "archived"})


def truncate_text(value: str | None, limit: int = 400) -> str | None:
    if not value:
        return None
    stripped = str(value).strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: max(0, limit - 3)].rstrip() + "..."


def truncate_text_hard(value: str | None, limit: int) -> tuple[str | None, bool, int]:
    """Return text capped to limit without exceeding the requested length."""
    if value is None:
        return None, False, 0
    text = str(value)
    original_len = len(text)
    if limit <= 0:
        return None, bool(text), original_len
    if original_len <= limit:
        return text, False, original_len
    return text[:limit], True, original_len


def normalize_reading_status(value: str | None, default: str = "saved") -> str:
    """Normalize untrusted reading status values to the allowed set."""
    text = str(value or default).strip().lower()
    if text in READING_ALLOWED_STATUSES:
        return text
    fallback = str(default or "saved").strip().lower()
    return fallback if fallback in READING_ALLOWED_STATUSES else "saved"


def is_supported_reading_url(value: str | None) -> bool:
    """Return True when value is an absolute http(s) reading URL."""
    if not value:
        return False
    parsed = urlparse(str(value).strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def hash_text_sha256(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    except Exception:
        return None


def word_count(value: str | None) -> int | None:
    if not value:
        return None
    words = [w for w in str(value).split() if w]
    return len(words) if words else None


HIGHLIGHT_CONTEXT_WINDOW = 64


def build_highlight_context(text: str, start_offset: int, end_offset: int, window: int = HIGHLIGHT_CONTEXT_WINDOW) -> tuple[str, str]:
    if not text:
        return "", ""
    start_offset = max(0, min(start_offset, len(text)))
    end_offset = max(start_offset, min(end_offset, len(text)))
    before_start = max(0, start_offset - window)
    after_end = min(len(text), end_offset + window)
    return text[before_start:start_offset], text[end_offset:after_end]


def find_highlight_span(
    text: str,
    quote: str,
    *,
    start_offset: int | None = None,
    end_offset: int | None = None,
    context_before: str | None = None,
    context_after: str | None = None,
    anchor_strategy: str = "fuzzy_quote",
) -> tuple[int, int] | None:
    if not text or not quote:
        return None

    quote_len = len(quote)
    if start_offset is not None:
        candidate_end = end_offset if end_offset is not None else start_offset + quote_len
        if 0 <= start_offset <= candidate_end <= len(text):
            candidate = text[start_offset:candidate_end]
            if candidate == quote:
                return start_offset, candidate_end
        if anchor_strategy == "exact_offset":
            return None

    matches = [m.start() for m in re.finditer(re.escape(quote), text)]
    case_insensitive = False
    if not matches:
        matches = [m.start() for m in re.finditer(re.escape(quote), text, flags=re.IGNORECASE)]
        case_insensitive = True
    if not matches:
        return None

    if len(matches) == 1:
        start = matches[0]
        return start, start + quote_len

    if not context_before and not context_after:
        start = matches[0]
        return start, start + quote_len

    def score_match(start_idx: int) -> int:
        score = 0
        end_idx = start_idx + quote_len
        if context_before:
            ctx_len = len(context_before)
            candidate_before = text[max(0, start_idx - ctx_len):start_idx]
            if candidate_before == context_before:
                score += 2
            elif candidate_before.endswith(context_before):
                score += 1
        if context_after:
            ctx_len = len(context_after)
            candidate_after = text[end_idx:end_idx + ctx_len]
            if candidate_after == context_after:
                score += 2
            elif candidate_after.startswith(context_after):
                score += 1
        return score

    best_start = matches[0]
    best_score = score_match(best_start)
    for start_idx in matches[1:]:
        score = score_match(start_idx)
        if score > best_score:
            best_score = score
            best_start = start_idx

    if case_insensitive and best_start + quote_len > len(text):
        return None
    return best_start, best_start + quote_len
