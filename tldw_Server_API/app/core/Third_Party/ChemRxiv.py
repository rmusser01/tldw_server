"""ChemRxiv Public API adapter.

Docs (Swagger 2): host chemrxiv.org, basePath /engage/chemrxiv/public-api/v1
Endpoints used:
 - GET /items (search)
 - GET /items/{itemId}
 - GET /items/doi/{doi}
 - GET /categories
 - GET /licenses
 - GET /version
 - GET /oai (raw, OAI-PMH XML)
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote as urlquote

from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    JSONDecodeError,
    NetworkError,
    RetryExhaustedError,
)
from tldw_Server_API.app.core.http_client import fetch, fetch_json

BASE_URL = "https://chemrxiv.org/engage/chemrxiv/public-api/v1"
_CHEMRXIV_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
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


def _join_authors(authors: Any) -> str | None:
    try:
        names = []
        for a in authors or []:
            first = (a or {}).get("firstName") or ""
            last = (a or {}).get("lastName") or ""
            nm = (first + " " + last).strip()
            if nm:
                names.append(nm)
        return ", ".join(names) if names else None
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    doi = item.get("doi")
    title = item.get("title") or ""
    url = None
    # Prefer webLinks.url if present
    links = item.get("webLinks") or []
    if links and isinstance(links, list):
        url = (links[0] or {}).get("url")
    if not url and doi:
        url = f"https://doi.org/{doi}"
    return {
        "id": item.get("id") or item.get("legacyId"),
        "title": title,
        "authors": _join_authors(item.get("authors")),
        "journal": None,
        "pub_date": item.get("publishedDate") or item.get("submittedDate"),
        "abstract": item.get("abstract"),
        "doi": doi,
        "url": url,
        "pdf_url": None,
        "provider": "chemrxiv",
    }


def search_items(
    term: str | None,
    skip: int,
    limit: int,
    sort: str | None = None,
    author: str | None = None,
    searchDateFrom: str | None = None,
    searchDateTo: str | None = None,
    searchLicense: str | None = None,
    categoryIds: list[str] | None = None,
    subjectIds: list[str] | None = None,
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    try:
        url = f"{BASE_URL}/items"
        params: dict[str, Any] = {
            "skip": max(0, skip),
            "limit": min(max(1, limit), 50),
        }
        if term:
            params["term"] = term
        if sort:
            params["sort"] = sort
        if author:
            params["author"] = author
        if searchDateFrom:
            params["searchDateFrom"] = searchDateFrom
        if searchDateTo:
            params["searchDateTo"] = searchDateTo
        if searchLicense:
            params["searchLicense"] = searchLicense
        if categoryIds:
            for cid in categoryIds:
                params.setdefault("categoryIds", []).append(cid)
        if subjectIds:
            for sid in subjectIds:
                params.setdefault("subjectIds", []).append(sid)

        data = fetch_json(method="GET", url=url, params=params, timeout=20)
        total = int(data.get("totalCount") or 0)
        hits = data.get("itemHits") or []
        # Each hit may wrap details or already be the item; best-effort unwrap
        items = []
        for h in hits:
            if isinstance(h, dict) and "title" in h:
                items.append(_normalize_item(h))
            elif isinstance(h, dict):
                # Fallback if nested under a key
                for v in h.values():
                    if isinstance(v, dict) and "title" in v:
                        items.append(_normalize_item(v))
                        break
        return items, total, None
    except TimeoutError:
        return None, 0, "ChemRxiv request timed out."
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None, 0, "ChemRxiv request failed."


def get_item_by_id(item_id: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        url = f"{BASE_URL}/items/{item_id}"
        r = fetch(method="GET", url=url, timeout=20)
        if r.status_code == 410:
            return None, None
        if r.status_code >= 400:
            return None, f"ChemRxiv HTTP error: {r.status_code}"
        data = r.json() or {}
        return _normalize_item(data), None
    except TimeoutError:
        return None, "ChemRxiv item request timed out."
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None, "ChemRxiv item request failed."


def get_item_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        doi_enc = urlquote(doi.strip(), safe="/")
        url = f"{BASE_URL}/items/doi/{doi_enc}"
        r = fetch(method="GET", url=url, timeout=20)
        if r.status_code == 410:
            return None, None
        if r.status_code >= 400:
            return None, f"ChemRxiv HTTP error: {r.status_code}"
        data = r.json() or {}
        return _normalize_item(data), None
    except TimeoutError:
        return None, "ChemRxiv DOI request timed out."
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None, "ChemRxiv DOI request failed."


def get_categories() -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = fetch_json(method="GET", url=f"{BASE_URL}/categories", timeout=20)
        return data, None
    except TimeoutError:
        return None, "ChemRxiv categories request timed out."
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None, "ChemRxiv categories request failed."


def get_licenses() -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = fetch_json(method="GET", url=f"{BASE_URL}/licenses", timeout=20)
        return data, None
    except TimeoutError:
        return None, "ChemRxiv licenses request timed out."
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None, "ChemRxiv licenses request failed."


def get_version() -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = fetch_json(method="GET", url=f"{BASE_URL}/version", timeout=20)
        return data, None
    except TimeoutError:
        return None, "ChemRxiv version request timed out."
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None, "ChemRxiv version request failed."


def oai_raw(params: dict[str, Any]) -> tuple[bytes | None, str | None, str | None]:
    """Raw OAI-PMH passthrough. Returns (content, media_type, error)."""
    r = None
    try:
        url = f"{BASE_URL}/oai"
        r = fetch(method="GET", url=url, params=params, timeout=20)
        if r.status_code >= 400:
            return None, None, f"ChemRxiv HTTP error: {r.status_code}"
        ct = r.headers.get("content-type") or "application/xml"
        return r.content, ct.split(";")[0], None
    except TimeoutError:
        return None, None, "ChemRxiv OAI-PMH request timed out."
    except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
        return None, None, "ChemRxiv OAI-PMH request failed."
    finally:
        try:
            if r is not None:
                r.close()
        except _CHEMRXIV_NONCRITICAL_EXCEPTIONS:
            pass
