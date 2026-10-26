"""The one keyword and tag identity rule (TASK-13950.5).

Keywords and tags are case-insensitive product-wide with display casing preserved:
stores keep the first-seen casing, and every comparison, dedupe or lookup compares
``normalize_keyword_identity`` values. ``tests/lint/test_keyword_identity_ratchet.py``
fails on new ad hoc ``.lower()``/``.casefold()`` keyword normalization.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable


def normalize_keyword_identity(display: str) -> str:
    """Return the portable keyword identity used by every database backend: NFC plus casefold."""

    return unicodedata.normalize("NFC", display.strip()).casefold()


def dedupe_keywords(values: Iterable[object]) -> list[str]:
    """Strip values and drop empties and identity duplicates, keeping the first-seen display casing."""

    unique: dict[str, str] = {}
    for value in values:
        text = "" if value is None else str(value).strip()
        if text:
            unique.setdefault(normalize_keyword_identity(text), text)
    return list(unique.values())
