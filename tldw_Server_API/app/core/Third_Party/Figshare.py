"""Figshare Public API adapter.

Implements minimal search, by-id, by-doi lookup, OAI-PMH passthrough, and
PDF extraction for ingestion when available.

Docs:
 - REST: https://api.figshare.com/v2 (articles, files)
 - Search: POST /articles/search?page=..&page_size=.. with JSON body
 - By-id: GET /articles/{id}
 - Files: GET /articles/{id}/files (each item has download_url)
 - OAI-PMH: https://api.figshare.com/v2/oai?verb=Identify ...
"""
from __future__ import annotations

import contextlib
from typing import Any

from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    JSONDecodeError,
    NetworkError,
    RetryExhaustedError,
)
from tldw_Server_API.app.core.http_client import fetch, fetch_json

BASE_URL = "https://api.figshare.com/v2"
OAI_BASE = f"{BASE_URL}/oai"
_FIGSHARE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    EgressPolicyError,
    JSONDecodeError,
    LookupError,
    NetworkError,
    OSError,
    RetryExhaustedError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _join_authors(item: dict[str, Any]) -> str | None:
    try:
        authors = item.get("authors") or []
        names = []
        for a in authors:
            nm = (a or {}).get("full_name") or (a or {}).get("name") or ""
            if nm:
                names.append(nm)
        return ", ".join(names) if names else None
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None


def _pick_pdf_from_files(files: list[dict[str, Any]]) -> str | None:
    try:
        for f in files or []:
            name = (f or {}).get("name") or ""
            url = (f or {}).get("download_url") or (f or {}).get("url")
            if name.lower().endswith(".pdf") and url:
                return url
        # Fallback: try any file ending in .pdf via file id
        for f in files or []:
            fid = (f or {}).get("id")
            name = (f or {}).get("name") or ""
            if fid and name.lower().endswith(".pdf"):
                return f"https://ndownloader.figshare.com/files/{fid}"
        return None
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None


def _normalize_article(item: dict[str, Any]) -> dict[str, Any]:
    doi = item.get("doi") or None
    title = item.get("title") or ""
    url = item.get("url_public_html") or item.get("figshare_url") or item.get("url_public_api") or item.get("url")
    pdf_url = None
    files = item.get("files") or []
    if isinstance(files, list) and files:
        pdf_url = _pick_pdf_from_files(files)
    return {
        "id": str(item.get("id") or ""),
        "title": title,
        "authors": _join_authors(item),
        "journal": None,
        "pub_date": item.get("published_date") or (item.get("timeline") or {}).get("firstOnline"),
        "abstract": item.get("description") or None,
        "doi": doi,
        "url": url,
        "pdf_url": pdf_url,
        "provider": "figshare",
    }


def search_articles(
    q: str | None,
    page: int,
    page_size: int,
    order: str | None = None,
    order_direction: str | None = None,
    search_for: str | None = None,
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    """Search Figshare records. Uses POST /articles/search with JSON body.

    Returns normalized GenericPaper-like records (without expensive file lookups).
    """
    try:
        params: dict[str, Any] = {
            "page": max(1, page),
            "page_size": max(1, min(page_size, 1000)),
        }
        body: dict[str, Any] = {}
        # Figshare defaults to all metadata fields; allow either raw text query or fielded search_for
        if search_for:
            body["search_for"] = search_for
        elif q:
            body["search_for"] = q
        if order:
            body["order"] = order
        if order_direction:
            body["order_direction"] = order_direction

        data = fetch_json(
            method="POST",
            url=f"{BASE_URL}/articles/search",
            params=params,
            json=body or {},
            headers={"Accept": "application/json"},
            timeout=20,
        )
        # Some deployments may wrap the results; handle minimally
        items_raw = data.get("items") or data.get("results") or data or []
        items = []
        for it in items_raw:
            if isinstance(it, dict):
                items.append(_normalize_article(it))
        # Figshare search response does not return a total count directly; approximate from length
        total = len(items)
        return items, total, None
    except TimeoutError:
        return None, 0, "Figshare request timed out."
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None, 0, "Figshare request failed."


def get_article_by_id(article_id: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        r = fetch(method="GET", url=f"{BASE_URL}/articles/{article_id}", headers={"Accept": "application/json"}, timeout=20)
        if r.status_code == 404:
            with contextlib.suppress(_FIGSHARE_NONCRITICAL_EXCEPTIONS):
                r.close()
            return None, None
        data = r.json() or {}
        return _normalize_article(data), None
    except TimeoutError:
        return None, "Figshare article request timed out."
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None, "Figshare article request failed."


def get_article_raw(article_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return raw Figshare article JSON for inspection."""
    try:
        data = fetch_json(method="GET", url=f"{BASE_URL}/articles/{article_id}", headers={"Accept": "application/json"}, timeout=20)
        return data or {}, None
    except TimeoutError:
        return None, "Figshare raw article request timed out."
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None, "Figshare raw article request failed."


def get_article_files(article_id: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        data = fetch_json(method="GET", url=f"{BASE_URL}/articles/{article_id}/files", headers={"Accept": "application/json"}, timeout=20)
        if not isinstance(data, list):
            return [], None
        return data, None
    except TimeoutError:
        return None, "Figshare file request timed out."
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None, "Figshare file request failed."


def get_article_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort DOI lookup via search_for fielded query (:doi: DOI)."""
    try:
        items, _, err = search_articles(None, page=1, page_size=1, search_for=f":doi: {doi}")
        if err:
            return None, err
        if items:
            return items[0], None
        return None, None
    except TimeoutError:
        return None, "Figshare DOI request timed out."
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None, "Figshare DOI request failed."


def oai_raw(params: dict[str, Any]) -> tuple[bytes | None, str | None, str | None]:
    """Raw OAI-PMH passthrough to Figshare OAI endpoint."""
    try:
        r = fetch(method="GET", url=OAI_BASE, params=params, headers={"Accept": "application/xml"}, timeout=20)
        if r.status_code >= 400:
            return None, None, f"Figshare OAI-PMH HTTP error: {r.status_code}"
        ct = r.headers.get("content-type") or "application/xml"
        return r.content, ct.split(";")[0], None
    except TimeoutError:
        return None, None, "Figshare OAI-PMH request timed out."
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None, None, "Figshare OAI-PMH request failed."


def extract_pdf_download_url(article: dict[str, Any]) -> str | None:
    """Get a direct download URL for a PDF file from a raw article JSON if available."""
    try:
        files = article.get("files") or []
        url = _pick_pdf_from_files(files)
        if url:
            return url
        # Fallback to files endpoint
        aid = article.get("id") or None
        if not aid:
            return None
        files2, _ = get_article_files(str(aid))
        if files2:
            return _pick_pdf_from_files(files2)
        return None
    except _FIGSHARE_NONCRITICAL_EXCEPTIONS:
        return None
