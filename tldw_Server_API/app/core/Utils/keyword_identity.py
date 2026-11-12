"""The one keyword and tag identity rule (TASK-13950.5).

Keywords and tags are case-insensitive product-wide with display casing preserved:
stores keep the first-seen casing, and every comparison, dedupe or lookup compares
``normalize_keyword_identity`` values. ``tests/lint/test_keyword_identity_ratchet.py``
fails on new ad hoc ``.lower()``/``.casefold()`` keyword normalization.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Collection, Iterable, Mapping
from typing import Any


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


def rank_identity_matches(wanted: Collection[str], rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group keyword ``rows`` under the wanted identity of their ``keyword``.

    Each group lists active rows before soft-deleted ones, then by ``id``; a repeated ``id`` is kept once.
    """
    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        identity = normalize_keyword_identity(str(row["keyword"]))
        if identity in wanted:
            groups.setdefault(identity, {}).setdefault(int(row["id"]), dict(row))
    return {
        identity: sorted(found.values(), key=lambda row: (bool(row["deleted"]), int(row["id"])))
        for identity, found in groups.items()
    }
