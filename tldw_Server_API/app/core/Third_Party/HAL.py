"""HAL.science Search API adapter (Solr-like).

API docs: https://api.archives-ouvertes.fr/docs/search
Entry point (default): https://api.archives-ouvertes.fr/search/

Supports:
 - JSON search with normalization to GenericPaper-like items
 - Raw passthrough for multiple formats via `wt` (json, xml, xml-tei, csv, bibtex, endnote, atom, rss)
 - By-docid lookup (best-effort)
 - PDF URL extraction from common HAL fields
"""
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.http_client import fetch, fetch_json

BASE_URL = "https://api.archives-ouvertes.fr/search/"


def _build_url(scope: str | None) -> str:
    if not scope:
        return BASE_URL
    s = str(scope).strip().strip('/')
    if not s:
        return BASE_URL
    return f"{BASE_URL}{s}/"




DEFAULT_FL = (
    "docid,label_s,title_s,authFullName_s,doiId_s,abstract_s,"
    "producedDate_tdate,producedDateY_i,uri_s,fileMain_s,files_s,linkExtUrl_s"
)


def _join_authors(doc: dict[str, Any]) -> str | None:
    try:
        auths = doc.get("authFullName_s")
        if isinstance(auths, list):
            names = [str(a) for a in auths if a]
            return ", ".join(names) if names else None
        if isinstance(auths, str):
            return auths
        return None
    except Exception:
        return None


def _pick_pdf_url(doc: dict[str, Any]) -> str | None:
    """Heuristically pick a PDF URL from HAL document fields."""
    try:
        # Common fields that may contain URLs
        candidates: list[str] = []
        for key in ("fileMain_s", "linkExtUrl_s", "uri_s"):
            val = doc.get(key)
            if isinstance(val, str):
                candidates.append(val)
        # Multi-valued files
        for key in ("files_s", "enclosures_s"):
            val = doc.get(key)
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, str):
                        candidates.append(v)
        for url in candidates:
            u = url.strip()
            if u.lower().endswith(".pdf"):
                return u
        return None
    except Exception:
        return None


def _normalize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    title = doc.get("title_s") or doc.get("label_s") or ""
    doi = doc.get("doiId_s") or None
    url = doc.get("uri_s") or (f"https://doi.org/{doi}" if doi else None)
    return {
        "id": str(doc.get("docid") or ""),
        "title": title,
        "authors": _join_authors(doc),
        "journal": None,
        "pub_date": doc.get("producedDate_tdate") or None,
        "abstract": doc.get("abstract_s") or None,
        "doi": doi,
        "url": url,
        "pdf_url": _pick_pdf_url(doc),
        "provider": "hal",
    }


def search(
    q: str,
    start: int,
    rows: int,
    fl: str | None = None,
    fq: list[str] | None = None,
    sort: str | None = None,
    scope: str | None = None,
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    try:
        params_list: list[tuple[str, str]] = [
            ("q", (q or "*:*")),
            ("wt", "json"),
            ("start", str(max(0, start))),
            ("rows", str(max(1, min(rows, 1000)))),
            ("fl", fl or DEFAULT_FL),
        ]
        if sort:
            params_list.append(("sort", sort))
        if fq:
            for f in fq:
                params_list.append(("fq", f))
        url = _build_url(scope)
        data = fetch_json(method="GET", url=url, params=params_list, headers={"Accept": "application/json"}, timeout=20)
        resp = (data.get("response") or {}) if isinstance(data, dict) else {}
        total = int(resp.get("numFound") or 0)
        docs = resp.get("docs") or []
        items = []
        for d in docs:
            if isinstance(d, dict):
                items.append(_normalize_doc(d))
        return items, total, None
    except TimeoutError:
        return None, 0, "HAL request timed out."
    except Exception:
        return None, 0, "HAL request failed."


def by_docid(docid: str, fl: str | None = None, scope: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        items, _, err = search(q=f"docid:{docid}", start=0, rows=1, fl=fl or DEFAULT_FL, scope=scope)
        if err:
            return None, err
        if items:
            return items[0], None
        return None, None
    except TimeoutError:
        return None, "HAL request timed out."
    except Exception:
        return None, "HAL request failed."


def raw(params: dict[str, Any], scope: str | None = None) -> tuple[bytes | None, str | None, str | None]:
    """Raw passthrough. Accepts wt in params and returns (content, media_type, error)."""
    try:
        wt = (params.get("wt") or "json").lower()
        # set Accept to something reasonable; we return upstream content-type anyway
        if wt in ("xml", "xml-tei", "atom", "rss"):
            accept = "application/xml"
        elif wt in ("csv", "bibtex", "endnote"):
            accept = "text/plain"
        else:
            accept = "application/json"

        url = _build_url(scope)
        r = fetch(method="GET", url=url, params=params, headers={"Accept": accept}, timeout=25)
        ct = r.headers.get("content-type") or "application/octet-stream"
        return r.content, ct.split(";")[0], None
    except TimeoutError:
        return None, None, "HAL request timed out."
    except Exception:
        return None, None, "HAL request failed."


def extract_pdf_from_raw_doc(doc: dict[str, Any]) -> str | None:
    return _pick_pdf_url(doc)
