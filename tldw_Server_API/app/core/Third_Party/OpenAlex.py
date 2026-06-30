"""OpenAlex client for venue-constrained searches (ACM/Wiley/etc.).

No API key required. Implements basic search and DOI lookup with retries and
standardized return signatures expected by the API layer.
"""
from __future__ import annotations

import contextlib
import math
import os
from typing import Any

from tldw_Server_API.app.core.http_client import fetch, fetch_json

BASE_URL = "https://api.openalex.org"


def _norm_authors(authorships: Any) -> str | None:
    try:
        names = []
        for a in authorships or []:
            author = (a or {}).get("author") or {}
            name = author.get("display_name")
            if name:
                names.append(name)
        return ", ".join(names) if names else None
    except Exception:
        return None


def _norm_venue(result: dict[str, Any]) -> str | None:
    hv = result.get("host_venue") or {}
    name = hv.get("display_name")
    if name:
        return name
    pl = result.get("primary_location") or {}
    src = pl.get("source") or {}
    name = src.get("display_name")
    return name or None


def _norm_pdf_url(result: dict[str, Any]) -> str | None:
    # OpenAlex has open_access.oa_url or best_oa_location.url
    oa = result.get("open_access") or {}
    if oa.get("oa_url"):
        return oa.get("oa_url")
    bol = result.get("best_oa_location") or {}
    url = bol.get("url_for_pdf") or bol.get("url")
    return url or None


def _norm_url(result: dict[str, Any]) -> str | None:
    # Prefer DOI link; fallback landing page
    doi = result.get("doi")
    if isinstance(doi, str) and doi:
        return f"https://doi.org/{doi.split('doi.org/')[-1] if 'doi.org' in doi else doi}"
    pl = result.get("primary_location") or {}
    return pl.get("landing_page_url") or None


def _normalize_openalex_work(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "title": result.get("title") or "",
        "authors": _norm_authors(result.get("authorships")),
        "journal": _norm_venue(result),
        "pub_date": result.get("publication_date") or str(result.get("publication_year") or ""),
        "abstract": None,  # abstract_inverted_index is non-trivial to reconstruct
        "doi": (result.get("doi") or "").replace("https://doi.org/", ""),
        "url": _norm_url(result),
        "pdf_url": _norm_pdf_url(result),
        "provider": "openalex",
    }


def search_openalex(
    q: str | None,
    offset: int,
    limit: int,
    filter_venue: str | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
) -> tuple[list[dict] | None, int, str | None]:
    try:
        url = f"{BASE_URL}/works"
        page = math.floor(offset / max(1, limit)) + 1
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if filter_venue:
            filters.append(f"host_venue.display_name.search:{filter_venue}")

        params: dict[str, Any] = {"per-page": limit, "page": page}
        if q:
            params["search"] = q
        if filters:
            params["filter"] = ",".join(filters)
        # Add mailto param if provided via env (improves reliability and rate limits)
        mailto = os.getenv("OPENALEX_MAILTO")
        if mailto:
            params["mailto"] = mailto

        headers = {"Accept": "application/json", "User-Agent": "tldw_server/0.1 (+https://github.com/openai/tldw_server)"}
        data = fetch_json(method="GET", url=url, params=params, headers=headers, timeout=20)
        results = data.get("results") or []
        total = (data.get("meta") or {}).get("count") or 0
        items = [_normalize_openalex_work(it) for it in results]
        return items, int(total), None
    except TimeoutError:
        return None, 0, "OpenAlex request timed out."
    except Exception:
        return None, 0, "OpenAlex request failed."


def get_openalex_by_doi(doi: str) -> tuple[dict | None, str | None]:
    try:
        doi_clean = doi.strip()
        url = f"{BASE_URL}/works/doi:{doi_clean}"
        r = fetch(method="GET", url=url, timeout=20, headers={"Accept": "application/json", "User-Agent": "tldw_server/0.1 (+https://github.com/openai/tldw_server)"})
        if r.status_code == 404:
            with contextlib.suppress(Exception):
                r.close()
            return None, None
        data = r.json()
        return _normalize_openalex_work(data), None
    except TimeoutError:
        return None, "OpenAlex request timed out."
    except Exception:
        return None, "OpenAlex request failed."


# Remove duplicate stubs (implementation above)
