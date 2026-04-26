"""OSF (Open Science Framework) Preprints API adapter.

Docs: https://developer.osf.io/
Base: https://api.osf.io/v2/preprints/

Supports:
- Search preprints across all providers or a specific provider via filter[provider]
- Lookup by OSF preprint ID or by DOI (best-effort)
- Raw passthrough helpers where needed
- Primary file download URL resolution for ingestion

Notes:
- We avoid per-item subrequests during search for performance; PDF URL is
  resolved on-demand by ingest via primary_file relationship.
"""
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.http_client import fetch, fetch_json

BASE_URL = "https://api.osf.io/v2/preprints/"


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    osf_id = item.get("id") or ""
    attrs = item.get("attributes") or {}
    links = item.get("links") or {}
    title = attrs.get("title") or ""
    abstract = attrs.get("description") or None
    pub_date = attrs.get("date_published") or attrs.get("date_created")
    doi = (
        attrs.get("doi")
        or attrs.get("preprint_doi")
        or attrs.get("article_doi")
        or None
    )
    url = links.get("html") or (f"https://osf.io/preprints/{osf_id}")
    return {
        "id": osf_id,
        "title": title,
        "authors": None,  # author expansion requires includes; omitted to avoid N+1
        "journal": None,
        "pub_date": pub_date,
        "abstract": abstract,
        "doi": doi,
        "url": url,
        "pdf_url": None,  # resolve via primary file at ingest time
        "provider": "osf",
    }


def search_preprints(
    term: str | None,
    page: int,
    results_per_page: int,
    provider: str | None = None,
    from_date: str | None = None,
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    """Search OSF preprints, optionally narrowing to a specific provider.

    - `term` maps to OSF's `q` parameter
    - `provider` maps to `filter[provider]`
    - `from_date` maps to `filter[date_created][gte]`
    """
    try:
        params: dict[str, Any] = {
            "page[size]": max(1, min(results_per_page, 100)),
            "page[number]": max(1, page),
        }
        if term:
            params["q"] = term
        if provider:
            params["filter[provider]"] = provider
        if from_date:
            params["filter[date_created][gte]"] = from_date

        data = fetch_json(method="GET", url=BASE_URL, params=params, headers={"Accept": "application/json"}, timeout=20)
        items = [
            _normalize_item(it)
            for it in (data.get("data") or [])
            if isinstance(it, dict)
        ]
        # Total may be present under links.meta.total; otherwise approximate
        meta = (data.get("links") or {}).get("meta") or {}
        total = int(meta.get("total") or 0)
        if total == 0:
            total = len(items)
        return items, total, None
    except TimeoutError:
        return None, 0, "OSF request timed out."
    except Exception:
        return None, 0, "OSF request failed."


def get_preprint_by_id(osf_id: str) -> tuple[dict[str, Any] | None, str | None]:
    r = None
    try:
        url = f"{BASE_URL}{osf_id}"
        r = fetch(method="GET", url=url, headers={"Accept": "application/json"}, timeout=20)
        if r.status_code == 404:
            return None, None
        if r.status_code >= 400:
            return None, f"OSF HTTP error: {r.status_code}"
        data = r.json() or {}
        if isinstance(data.get("data"), dict):
            return _normalize_item(data["data"]), None
        return None, None
    except TimeoutError:
        return None, "OSF preprint request timed out."
    except Exception:
        return None, "OSF preprint request failed."
    finally:
        try:
            if r is not None:
                r.close()
        except Exception as close_error:
            _ = close_error  # best-effort response close


def get_preprint_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    """Lookup preprint by DOI using direct filter and fallback query."""
    try:
        if not doi or not doi.strip():
            return None, "DOI cannot be empty"
        # Try exact filter on doi and article_doi
        for field in ("doi", "article_doi"):
            params = {"page[size]": 1, f"filter[{field}]": doi}
            r = fetch(method="GET", url=BASE_URL, params=params, headers={"Accept": "application/json"}, timeout=20)
            if r.status_code == 200:
                data = r.json() or {}
                items = data.get("data") or []
                if items:
                    return _normalize_item(items[0]), None
        # Fallback free-text search
        r2 = fetch(method="GET", url=BASE_URL, params={"q": doi, "page[size]": 1}, headers={"Accept": "application/json"}, timeout=20)
        if r2.status_code == 404:
            return None, None
        if r2.status_code >= 400:
            return None, f"OSF HTTP error: {r2.status_code}"
        data2 = r2.json() or {}
        items2 = data2.get("data") or []
        if items2:
            return _normalize_item(items2[0]), None
        return None, None
    except TimeoutError:
        return None, "OSF DOI request timed out."
    except Exception:
        return None, "OSF DOI request failed."


def get_primary_file_download_url(osf_id: str) -> tuple[str | None, str | None]:
    """Resolve the primary file's direct download URL for a given preprint id.

    Returns (download_url, error).
    """
    try:
        # 1) Fetch preprint with primary_file relationship link
        data = fetch_json(method="GET", url=f"{BASE_URL}{osf_id}", headers={"Accept": "application/json"}, timeout=20)
        rel = ((data.get("data") or {}).get("relationships") or {}).get("primary_file") or {}
        rel_links = (rel.get("links") or {}).get("related") or {}
        href = rel_links.get("href")
        if not href:
            # try embeds via include
            r2 = fetch(method="GET", url=f"{BASE_URL}{osf_id}?include=primary_file", headers={"Accept": "application/json"}, timeout=20)
            if r2.status_code == 200:
                d2 = r2.json() or {}
                included = d2.get("included") or []
                for inc in included:
                    if inc.get("type") == "files" and inc.get("id"):
                        links = inc.get("links") or {}
                        dl = links.get("download") or links.get("meta", {}).get("download")
                        if dl:
                            return dl, None
            return None, None
        # 2) Follow file endpoint to get download link
        fdata = fetch_json(method="GET", url=href, headers={"Accept": "application/json"}, timeout=20)
        links = (fdata.get("data") or {}).get("links") or {}
        dl_url = links.get("download") or links.get("meta", {}).get("download")
        return (dl_url or None), None
    except TimeoutError:
        return None, "OSF primary file request timed out."
    except Exception:
        return None, "OSF primary file request failed."


def raw_preprints(params: dict[str, Any]) -> tuple[bytes | None, str | None, str | None]:
    """Raw passthrough for OSF preprints list endpoint.

    Accepts a dict of query parameters (e.g., q, filter[provider], page[size], page[number], filter[date_created][gte]).
    Returns (content, media_type, error).
    """
    try:
        r = fetch(method="GET", url=BASE_URL, params=params, headers={"Accept": "application/json"}, timeout=25)
        if r.status_code >= 400:
            return None, None, f"OSF HTTP error: {r.status_code}"
        ct = r.headers.get("content-type") or "application/json"
        return r.content, ct.split(";")[0], None
    except TimeoutError:
        return None, None, "OSF raw preprints request timed out."
    except Exception:
        return None, None, "OSF raw preprints request failed."


def raw_by_id(osf_id: str) -> tuple[bytes | None, str | None, str | None]:
    """Raw passthrough for a single OSF preprint by id."""
    try:
        r = fetch(method="GET", url=f"{BASE_URL}{osf_id}", headers={"Accept": "application/json"}, timeout=25)
        if r.status_code == 404:
            return None, "application/json", None
        if r.status_code >= 400:
            return None, None, f"OSF HTTP error: {r.status_code}"
        ct = r.headers.get("content-type") or "application/json"
        return r.content, ct.split(";")[0], None
    except TimeoutError:
        return None, None, "OSF raw preprint request timed out."
    except Exception:
        return None, None, "OSF raw preprint request failed."
