"""Elsevier Scopus adapter.

Uses API key when present. Implements basic search and DOI lookup with retries
and normalization. Some features may require institutional entitlements.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any

from tldw_Server_API.app.core.http_client import fetch, fetch_json


def _missing_key_error() -> str:
    return "Elsevier API key not configured. Set ELSEVIER_API_KEY to enable Scopus provider."


BASE_URL = "https://api.elsevier.com/content/search/scopus"





def _headers() -> dict[str, str]:
    h = {
        "X-ELS-APIKey": os.getenv("ELSEVIER_API_KEY", ""),
        "Accept": "application/json",
    }
    inst = os.getenv("ELSEVIER_INST_TOKEN")
    if inst:
        h["X-ELS-Insttoken"] = inst
    return h


def _join_authors(entry: dict[str, Any]) -> str | None:
    # Scopus search returns 'dc:creator' (first author) and 'author' array in detailed endpoints.
    # Here we best-effort use 'dc:creator'.
    name = entry.get("dc:creator")
    return name if name else None


def _pdf_url(entry: dict[str, Any]) -> str | None:
    # Scopus search usually does not return direct PDFs
    return None


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    doi = entry.get("prism:doi")
    return {
        "id": entry.get("eid") or entry.get("dc:identifier"),
        "title": entry.get("dc:title") or "",
        "authors": _join_authors(entry),
        "journal": entry.get("prism:publicationName") or None,
        "pub_date": entry.get("prism:coverDate") or None,
        "abstract": None,
        "doi": doi,
        "url": entry.get("prism:url") or (f"https://doi.org/{doi}" if doi else None),
        "pdf_url": _pdf_url(entry),
        "provider": "scopus",
    }


def search_scopus(
    q: str | None,
    offset: int,
    limit: int,
    from_year: int | None = None,
    to_year: int | None = None,
    open_access_only: bool = False,
) -> tuple[list[dict] | None, int, str | None]:
    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key:
        return None, 0, _missing_key_error()
    try:
        query_parts: list[str] = []
        if q:
            query_parts.append(q)
        if from_year or to_year:
            lo = from_year or to_year
            hi = to_year or from_year
            if lo and hi and hi < lo:
                lo, hi = hi, lo
            if lo and hi:
                query_parts.append(f"PUBYEAR >={lo} AND PUBYEAR <={hi}")
            elif lo:
                query_parts.append(f"PUBYEAR = {lo}")
        if open_access_only:
            # Best-effort OA filter in Scopus advanced query
            query_parts.append("OPENACCESS(1)")

        params: dict[str, Any] = {
            "query": " AND ".join(query_parts) if query_parts else "ALL(*)",
            "start": offset,
            "count": limit,
            "view": "STANDARD",
        }
        data = fetch_json(method="GET", url=BASE_URL, headers=_headers(), params=params, timeout=20)
        sr = data.get("search-results") or {}
        total = int(sr.get("opensearch:totalResults") or 0)
        entries = sr.get("entry") or []
        items = [_normalize_entry(e) for e in entries]
        return items, total, None
    except TimeoutError:
        return None, 0, "Scopus request timed out."
    except Exception:
        return None, 0, "Scopus request failed."


def get_scopus_by_doi(doi: str) -> tuple[dict | None, str | None]:
    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key:
        return None, _missing_key_error()
    try:
        params = {
            "query": f"DOI({doi})",
            "start": 0,
            "count": 1,
            "view": "STANDARD",
        }
        r = fetch(method="GET", url=BASE_URL, headers=_headers(), params=params, timeout=20)
        if r.status_code == 404:
            with contextlib.suppress(Exception):
                r.close()
            return None, None
        data = r.json() or {}
        sr = data.get("search-results") or {}
        entries = sr.get("entry") or []
        if not entries:
            return None, None
        return _normalize_entry(entries[0]), None
    except TimeoutError:
        return None, "Scopus request timed out."
    except Exception:
        return None, "Scopus request failed."
