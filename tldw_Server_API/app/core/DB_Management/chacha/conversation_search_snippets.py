"""Build the short message excerpt shown under a chat that matched a search by content."""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_SNIPPET_MAX_CHARS = 160
_ELLIPSIS = "…"
_FTS_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})
_TERM_RE = re.compile(r"\w+(?:'\w+)*")


def extract_search_terms(query: str | None) -> list[str]:
    """Return the words of a search query, lower-cased and in order, without FTS operators.

    The excerpt only needs somewhere to centre on, so this is a plain word scan
    rather than a parse of the FTS grammar: quotes, ``*`` and parentheses are
    dropped, and the upper-case operators ``AND``/``OR``/``NOT``/``NEAR`` are skipped.
    """
    terms: list[str] = []
    for word in _TERM_RE.findall(query or ""):
        if word in _FTS_OPERATORS:
            continue
        term = word.lower()
        if term not in terms:
            terms.append(term)
    return terms


def build_match_snippet(
    content: str | None,
    terms: Sequence[str],
    *,
    max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
) -> str:
    """Return a one-line excerpt of ``content`` around the longest of ``terms`` found in it.

    The longest query word is the most specific one, so a common short word
    earlier in the message does not pull the excerpt away from the real match.
    The excerpt is at most ``max_chars`` characters, plus an ellipsis on each
    side that was cut. When no term can be located (the index matched a stemmed
    or differently tokenised form) the excerpt starts at the beginning, so a
    matched message always yields some context.
    """
    text = " ".join((content or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    match_start = 0
    for term in sorted((term for term in terms if term), key=len, reverse=True):
        found = re.search(re.escape(term), text, re.IGNORECASE)
        if found is not None:
            match_start = found.start()
            break

    # Keep about a third of the window before the match so its lead-in is readable.
    start = max(0, match_start - max_chars // 3)
    if start > 0:
        # Start on a word boundary when one is close, without stepping past the match.
        boundary = text.find(" ", start, match_start)
        if boundary != -1:
            start = boundary + 1
    end = min(len(text), start + max_chars)
    if end == len(text):
        start = max(0, end - max_chars)

    excerpt = text[start:end].strip()
    prefix = _ELLIPSIS if start > 0 else ""
    suffix = _ELLIPSIS if end < len(text) else ""
    return f"{prefix}{excerpt}{suffix}"
