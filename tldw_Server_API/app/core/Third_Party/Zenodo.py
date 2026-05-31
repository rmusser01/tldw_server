"""Zenodo API adapter (Records + OAI-PMH passthrough).

Search published records via https://zenodo.org/api/records and normalize to
GenericPaper-like objects. Also provides by-id and by-doi helpers and an
OAI-PMH raw passthrough for XML.
"""
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.exceptions import (
    DownloadError,
    EgressPolicyError,
    JSONDecodeError,
    NetworkError,
    RetryExhaustedError,
    StreamingProtocolError,
)
from tldw_Server_API.app.core.http_client import fetch, fetch_json

RECORDS_URL = "https://zenodo.org/api/records"
OAI_BASE = "https://zenodo.org/oai2d"

_ZENODO_PARSE_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)
_ZENODO_REQUEST_EXCEPTIONS = (
    ConnectionError,
    DownloadError,
    EgressPolicyError,
    JSONDecodeError,
    NetworkError,
    OSError,
    RetryExhaustedError,
    RuntimeError,
    StreamingProtocolError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _join_authors(meta: dict[str, Any]) -> str | None:
    try:
        creators = meta.get("creators") or []
        names = []
        for c in creators:
            nm = (c or {}).get("name") or ""
            if nm:
                names.append(nm)
        return ", ".join(names) if names else None
    except _ZENODO_PARSE_EXCEPTIONS:
        return None


def _pick_pdf_url(rec: dict[str, Any]) -> str | None:
    try:
        files = rec.get("files") or []
        # New API variants may nest under rec["files"][i]["links"]["self"]
        for f in files:
            key = (f or {}).get("key") or (f or {}).get("filename") or ""
            mimetype = (f or {}).get("mimetype") or (f or {}).get("type") or ""
            links = (f or {}).get("links") or {}
            href = links.get("self") or links.get("download")
            if (key and key.lower().endswith(".pdf")) or (mimetype and "pdf" in mimetype.lower()):
                return href or None
        return None
    except _ZENODO_PARSE_EXCEPTIONS:
        return None


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    meta = rec.get("metadata") or {}
    doi = meta.get("doi") or None
    title = meta.get("title") or rec.get("title") or ""
    url = (rec.get("links") or {}).get("html") or (f"https://doi.org/{doi}" if doi else None)
    pdf_url = _pick_pdf_url(rec)
    return {
        "id": str(rec.get("id") or rec.get("record_id") or ""),
        "title": title,
        "authors": _join_authors(meta),
        "journal": meta.get("journal_title") or None,
        "pub_date": meta.get("publication_date") or None,
        "abstract": meta.get("description") or None,
        "doi": doi,
        "url": url,
        "pdf_url": pdf_url,
        "provider": "zenodo",
    }


def search_records(
    q: str | None,
    page: int,
    size: int,
    type_: str | None = None,
    subtype: str | None = None,
    communities: str | None = None,
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    try:
        params: dict[str, Any] = {
            "page": max(1, page),
            "size": max(1, min(size, 100)),
        }
        if q:
            params["q"] = q
        if type_:
            params["type"] = type_
        if subtype:
            params["subtype"] = subtype
        if communities:
            params["communities"] = communities
        data = fetch_json(method="GET", url=RECORDS_URL, params=params, timeout=20)
        # Zenodo may return plain list or hits dict depending on version
        hits_block = data.get("hits") if isinstance(data, dict) else None
        if hits_block and isinstance(hits_block, dict):
            hits = hits_block.get("hits") or []
            total = int((hits_block.get("total") or {}).get("value") or hits_block.get("total") or len(hits))
        elif isinstance(data, list):
            hits = data
            total = len(hits)
        else:
            # Fallback: assume top-level has 'hits'
            hits = data.get("hits", []) if isinstance(data, dict) else []
            total = len(hits)
        items = [_normalize_record(h) for h in hits if isinstance(h, dict)]
        return items, total, None
    except TimeoutError:
        return None, 0, "Zenodo request timed out."
    except _ZENODO_REQUEST_EXCEPTIONS:
        return None, 0, "Zenodo request failed."


def get_record_by_id(record_id: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = fetch_json(method="GET", url=f"{RECORDS_URL}/{record_id}", timeout=20)
        return _normalize_record(data), None
    except TimeoutError:
        return None, "Zenodo record request timed out."
    except _ZENODO_REQUEST_EXCEPTIONS:
        return None, "Zenodo record request failed."


def get_record_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        # Search by DOI string
        params = {"q": doi, "size": 1}
        data = fetch_json(method="GET", url=RECORDS_URL, params=params, timeout=20)
        hits_block = data.get("hits") if isinstance(data, dict) else None
        items = []
        if hits_block and isinstance(hits_block, dict):
            items = hits_block.get("hits") or []
        elif isinstance(data, list):
            items = data
        if items:
            return _normalize_record(items[0]), None
        return None, None
    except TimeoutError:
        return None, "Zenodo DOI request timed out."
    except _ZENODO_REQUEST_EXCEPTIONS:
        return None, "Zenodo DOI request failed."


def oai_raw(params: dict[str, Any]) -> tuple[bytes | None, str | None, str | None]:
    try:
        r = fetch(method="GET", url=OAI_BASE, params=params, headers={"Accept": "application/xml"}, timeout=20)
        if r.status_code >= 400:
            return None, None, f"Zenodo OAI-PMH HTTP error: {r.status_code}"
        ct = r.headers.get("content-type") or "application/xml"
        return r.content, ct.split(";")[0], None
    except TimeoutError:
        return None, None, "Zenodo OAI-PMH request timed out."
    except _ZENODO_REQUEST_EXCEPTIONS:
        return None, None, "Zenodo OAI-PMH request failed."


def get_record_raw(record_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the raw Zenodo record JSON for advanced inspection (e.g., files)."""
    try:
        data = fetch_json(method="GET", url=f"{RECORDS_URL}/{record_id}", timeout=20)
        return data or {}, None
    except TimeoutError:
        return None, "Zenodo raw record request timed out."
    except _ZENODO_REQUEST_EXCEPTIONS:
        return None, "Zenodo raw record request failed."


def extract_pdf_from_raw(rec: dict[str, Any]) -> str | None:
    """Find a PDF download link from raw record JSON if present."""
    try:
        files = rec.get("files") or []
        for f in files:
            links = (f or {}).get("links") or {}
            href = links.get("self") or links.get("download")
            mimetype = (f or {}).get("mimetype") or (f or {}).get("type") or ""
            key = (f or {}).get("key") or (f or {}).get("filename") or ""
            if ((key and key.lower().endswith('.pdf')) or (mimetype and 'pdf' in mimetype.lower())) and href:
                return href
        return None
    except _ZENODO_PARSE_EXCEPTIONS:
        return None
