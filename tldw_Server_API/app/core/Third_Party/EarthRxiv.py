"""EarthArXiv (EarthRxiv) adapter via OSF Preprints API.

Docs: https://api.osf.io/
Provider key: 'eartharxiv'

We implement simple search and lookup helpers that return a normalized
GenericPaper-like dict. We avoid per-item subrequests; DOI and authors
may be missing in list results. We provide stable html/pdf links that
enable ingest-by-pdf.
"""
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.http_client import fetch, fetch_json

BASE_URL = "https://api.osf.io/v2/preprints/"
PROVIDER = "eartharxiv"





def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    """Map OSF preprint item to GenericPaper fields.

    We compute EarthArXiv landing and download links from the OSF id.
    """
    osf_id = item.get("id") or ""
    attrs = item.get("attributes") or {}
    title = attrs.get("title") or ""
    abstract = attrs.get("description") or None
    pub_date = attrs.get("date_published") or attrs.get("date_created")
    # DOI can appear in different places depending on OSF; may be missing
    doi = attrs.get("doi") or attrs.get("preprint_doi") or None
    # Build EarthArXiv URLs
    url = f"https://eartharxiv.org/{osf_id}/" if osf_id else None
    pdf_url = f"https://eartharxiv.org/{osf_id}/download" if osf_id else None
    return {
        "id": osf_id,
        "title": title,
        "authors": None,  # Omitted to avoid extra calls; UI still supports ingest
        "journal": None,
        "pub_date": pub_date,
        "abstract": abstract,
        "doi": doi,
        "url": url,
        "pdf_url": pdf_url,
        "provider": "eartharxiv",
    }


def search_items(
    term: str | None,
    page: int,
    results_per_page: int,
    from_date: str | None = None,
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    try:
        params: dict[str, Any] = {
            "filter[provider]": PROVIDER,
            "page[size]": max(1, min(results_per_page, 100)),
            "page[number]": max(1, page),
        }
        if term:
            # OSF supports 'q' across common text fields
            params["q"] = term
        if from_date:
            params["filter[date_created][gte]"] = from_date
        data = fetch_json(method="GET", url=BASE_URL, params=params, headers={"Accept": "application/json"}, timeout=20)
        items = [
            _normalize_item(it)
            for it in (data.get("data") or [])
            if isinstance(it, dict)
        ]
        # OSF meta may include total; if missing, approximate
        meta = (data.get("links") or {}).get("meta") or {}
        total = int(meta.get("total") or 0)
        if total == 0:
            total = len(items)
        return items, total, None
    except TimeoutError:
        return None, 0, "EarthArXiv request timed out."
    except Exception:
        return None, 0, "EarthArXiv request failed."


def get_item_by_id(osf_id: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        url = f"{BASE_URL}{osf_id}"
        data = fetch_json(method="GET", url=url, headers={"Accept": "application/json"}, timeout=20)
        if isinstance(data.get("data"), dict):
            return _normalize_item(data["data"]), None
        return None, None
    except TimeoutError:
        return None, "EarthArXiv request timed out."
    except Exception:
        return None, "EarthArXiv request failed."


def get_item_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        # Attempt direct DOI filter; fallback to search query
        params: dict[str, Any] = {
            "filter[provider]": PROVIDER,
            "filter[doi]": doi,
            "page[size]": 1,
        }
        r = fetch(method="GET", url=BASE_URL, params=params, headers={"Accept": "application/json"}, timeout=20)
        if r.status_code == 200:
            data = r.json() or {}
            items = data.get("data") or []
            if items:
                return _normalize_item(items[0]), None
        # Fallback: query by DOI string
        data2 = fetch_json(
            method="GET",
            url=BASE_URL,
            params={"filter[provider]": PROVIDER, "q": doi, "page[size]": 1},
            headers={"Accept": "application/json"},
            timeout=20,
        )
        items2 = data2.get("data") or []
        if items2:
            return _normalize_item(items2[0]), None
        return None, None
    except TimeoutError:
        return None, "EarthArXiv request timed out."
    except Exception:
        return None, "EarthArXiv request failed."
