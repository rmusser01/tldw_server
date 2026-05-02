"""Springer Nature Metadata API adapter.

Return signature follows project convention. Requires `SPRINGER_NATURE_API_KEY`.
"""
from __future__ import annotations

import os
from typing import Any

from tldw_Server_API.app.core.http_client import fetch_json


def _missing_key_error() -> str:
    return "Springer Nature API key not configured. Set SPRINGER_NATURE_API_KEY."


BASE_URL = "https://api.springernature.com/metadata/json"


def _join_authors(creators: Any) -> str | None:
    try:
        names = []
        for c in creators or []:
            name = c.get("creator")
            if name:
                names.append(name)
        return ", ".join(names) if names else None
    except Exception:
        return None


def _pdf_url(urls: Any) -> str | None:
    for u in urls or []:
        if (u.get("format") or "").lower() == "pdf" and u.get("value"):
            return u.get("value")
    return None


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    doi = rec.get("doi")
    return {
        "id": doi or rec.get("identifier"),
        "title": rec.get("title") or "",
        "authors": _join_authors(rec.get("creators")),
        "journal": rec.get("publicationName") or None,
        "pub_date": rec.get("publicationDate") or None,
        "abstract": rec.get("abstract") or None,
        "doi": doi,
        "url": (rec.get("url") or [{}])[0].get("value") if isinstance(rec.get("url"), list) else None,
        "pdf_url": _pdf_url(rec.get("url")),
        "provider": "springer",
    }


def search_springer(
    q: str | None,
    offset: int,
    limit: int,
    journal: str | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
) -> tuple[list[dict] | None, int, str | None]:
    api_key = os.getenv("SPRINGER_NATURE_API_KEY")
    if not api_key:
        return None, 0, _missing_key_error()
    try:
        # Springer uses 'q' with fielded query; 'p' page size, 's' start index
        q_parts: list[str] = []
        if q:
            q_parts.append(q)
        if journal:
            # Field often called 'journal' in examples; alternatively 'publicationName'
            q_parts.append(f"journal:\"{journal}\"")
        if from_year or to_year:
            lo = from_year or to_year
            hi = to_year or from_year
            if lo and hi and hi < lo:
                lo, hi = hi, lo
            if lo and hi:
                q_parts.append(f"year:{lo}-{hi}")
            elif lo:
                q_parts.append(f"year:{lo}")

        params: dict[str, Any] = {
            "api_key": api_key,
            "p": limit,
            "s": offset,
            "q": " AND ".join(q_parts) if q_parts else "*",
        }
        data = fetch_json(method="GET", url=BASE_URL, params=params, timeout=20)
        message = data.get("result") or []
        # total items can be found in the first element's 'total' typically
        total = 0
        if message and isinstance(message, list):
            try:
                total = int((message[0] or {}).get("total") or 0)
            except Exception:
                total = 0
        records = data.get("records") or []
        items = [_normalize_record(rec) for rec in records]
        return items, total, None
    except TimeoutError:
        return None, 0, "Springer request timed out."
    except Exception:
        return None, 0, "Springer request failed."


def get_springer_by_doi(doi: str) -> tuple[dict | None, str | None]:
    api_key = os.getenv("SPRINGER_NATURE_API_KEY")
    if not api_key:
        return None, _missing_key_error()
    try:
        params = {
            "api_key": api_key,
            "p": 1,
            "s": 0,
            "q": f"doi:{doi}",
        }
        data = fetch_json(method="GET", url=BASE_URL, params=params, timeout=20)
        records = data.get("records") or []
        if not records:
            return None, None
        return _normalize_record(records[0]), None
    except TimeoutError:
        return None, "Springer request timed out."
    except Exception:
        return None, "Springer request failed."
