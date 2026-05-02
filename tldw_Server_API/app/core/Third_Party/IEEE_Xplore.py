"""IEEE Xplore API adapter.

Implements search and lookup with the project’s standard return signatures.
Requires `IEEE_API_KEY` in environment.
"""
from __future__ import annotations

import os
from typing import Any

from tldw_Server_API.app.core.http_client import fetch_json


def _missing_key_error() -> str:
    return "IEEE API key not configured. Set IEEE_API_KEY to enable this provider."


BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"


def _join_authors(authors_block: Any) -> str | None:
    try:
        auths = ((authors_block or {}).get("authors") or [])
        names = []
        for a in auths:
            name = a.get("preferred_name") or a.get("full_name")
            if name:
                names.append(name)
        return ", ".join(names) if names else None
    except Exception:
        return None


def _pdf_url(article: dict[str, Any]) -> str | None:
    # Prefer pdf_url when present
    if article.get("pdf_url"):
        return article.get("pdf_url")
    # Some responses contain 'html_url' only
    return None


def _normalize_article(article: dict[str, Any]) -> dict[str, Any]:
    doi = article.get("doi")
    return {
        "id": str(article.get("article_number") or ""),
        "title": article.get("title") or "",
        "authors": _join_authors(article.get("authors")),
        "journal": article.get("publication_title") or None,
        "pub_date": str(article.get("publication_year") or ""),
        "abstract": article.get("abstract") or None,
        "doi": doi,
        "url": article.get("html_url") or (f"https://doi.org/{doi}" if doi else None),
        "pdf_url": _pdf_url(article),
        "provider": "ieee",
    }


def search_ieee(
    q: str | None,
    offset: int,
    limit: int,
    from_year: int | None = None,
    to_year: int | None = None,
    publication_title: str | None = None,
    authors: str | None = None,
) -> tuple[list[dict] | None, int, str | None]:
    api_key = os.getenv("IEEE_API_KEY")
    if not api_key:
        return None, 0, _missing_key_error()
    try:
        # IEEE uses 1-based start_record; limit via max_records
        start_record = max(1, offset + 1)
        params: dict[str, Any] = {
            "apikey": api_key,
            "format": "json",
            "start_record": start_record,
            "max_records": limit,
        }
        # Build querytext combining free text and author/publication if provided
        q_parts: list[str] = []
        if q:
            q_parts.append(q)
        if authors:
            q_parts.append(f"authors:{authors}")
        if publication_title:
            q_parts.append(f"publication_title:{publication_title}")
        if q_parts:
            params["querytext"] = " AND ".join(q_parts)
        # Year range
        if from_year or to_year:
            lo = from_year or to_year
            hi = to_year or from_year
            if lo and hi and hi < lo:
                lo, hi = hi, lo
            if lo and hi:
                params["publication_year"] = f"{lo}_{hi}"
            elif lo:
                params["publication_year"] = f"{lo}_{lo}"

        data = fetch_json(method="GET", url=BASE_URL, params=params, timeout=20)
        total = int(data.get("total_records") or 0)
        articles = data.get("articles") or []
        items = [_normalize_article(it) for it in articles]
        return items, total, None
    except TimeoutError:
        return None, 0, "IEEE Xplore request timed out."
    except Exception:
        return None, 0, "IEEE Xplore request failed."


def get_ieee_by_doi(doi: str) -> tuple[dict | None, str | None]:
    api_key = os.getenv("IEEE_API_KEY")
    if not api_key:
        return None, _missing_key_error()
    try:
        params = {
            "apikey": api_key,
            "format": "json",
            "max_records": 1,
            # Use querytext targeting DOI; IEEE search supports doi in querytext
            "querytext": f"doi:{doi}",
        }
        data = fetch_json(method="GET", url=BASE_URL, params=params, timeout=20)
        articles = data.get("articles") or []
        if not articles:
            return None, None
        return _normalize_article(articles[0]), None
    except TimeoutError:
        return None, "IEEE Xplore request timed out."
    except Exception:
        return None, "IEEE Xplore request failed."


def get_ieee_by_id(article_number: str) -> tuple[dict | None, str | None]:
    api_key = os.getenv("IEEE_API_KEY")
    if not api_key:
        return None, _missing_key_error()
    try:
        params = {
            "apikey": api_key,
            "format": "json",
            "max_records": 1,
            # arnumber is the field commonly used for IEEE article number
            "querytext": f"arnumber:{article_number}",
        }
        data = fetch_json(method="GET", url=BASE_URL, params=params, timeout=20)
        articles = data.get("articles") or []
        if not articles:
            return None, None
        return _normalize_article(articles[0]), None
    except TimeoutError:
        return None, "IEEE Xplore request timed out."
    except Exception:
        return None, "IEEE Xplore request failed."
