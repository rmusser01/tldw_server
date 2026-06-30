"""
PubMed.py

Provider adapter for PubMed search using NCBI E-utilities (ESearch + ESummary).

This module exposes a single primary function `search_pubmed` which takes a query and
returns normalized items with pagination metadata and robust error handling. It follows
the design patterns used by other provider adapters in this project (retries, timeouts,
and conservative parsing).

Notes:
- We use ESearch to get PMIDs matching the query and ESummary to fetch metadata.
- Optional year range is mapped to ESearch mindate/maxdate with datetype=pdat.
- Optional free-full-text filter is added to the query term.
- URLs are constructed to PubMed and PMC when available; PDF links provided for PMC articles.

Limits:
- Abstract text is not included (ESummary does not return it). Fetching abstracts would require
  EFetch (XML) and additional parsing; we keep search lightweight here.
"""

from __future__ import annotations

import contextlib
from typing import Any

from defusedxml import ElementTree as DET
from tldw_Server_API.app.core.http_client import fetch, fetch_json

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"




def _build_term(query: str, free_full_text: bool) -> str:
    term = query.strip()
    if free_full_text:
        # PubMed filter for free full text
        # Appending with AND to respect user query
        term = f"({term}) AND free full text[Filter]"
    return term


def _normalize_authors(author_list: list[dict[str, Any]] | None) -> str | None:
    if not author_list:
        return None
    names: list[str] = []
    for a in author_list:
        name = a.get("name") or a.get("authtype")
        if name:
            names.append(str(name))
    return ", ".join(names) if names else None


def _extract_article_ids(articleids: list[dict[str, Any]] | None) -> dict[str, str | None]:
    out: dict[str, str | None] = {"doi": None, "pmcid": None}
    if not articleids:
        return out
    for it in articleids:
        idtype = (it.get("idtype") or "").lower()
        val = it.get("value") or it.get("id")
        if not val:
            continue
        if idtype == "doi":
            out["doi"] = val
        elif idtype == "pmc":
            # Value typically is like "PMC1234567"
            out["pmcid"] = val.replace("PMC", "") if val.startswith("PMC") else val
    return out


def _normalize_item(uid: str, raw: dict[str, Any]) -> dict[str, Any]:
    title = raw.get("title") or "N/A"
    journal = raw.get("fulljournalname") or raw.get("source")
    pubdate = raw.get("pubdate") or raw.get("epubdate") or raw.get("sortpubdate")
    authors = _normalize_authors(raw.get("authors"))
    ids = _extract_article_ids(raw.get("articleids"))
    doi = ids.get("doi")
    pmcid = ids.get("pmcid")
    pmid = str(uid)
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    pmc_url = f"https://pmc.ncbi.nlm.nih.gov/{pmcid}/" if pmcid else None
    pdf_url = f"https://pmc.ncbi.nlm.nih.gov/{pmcid}/pdf" if pmcid else None

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "journal": journal,
        "pub_date": pubdate,
        "doi": doi,
        "url": url,
        "pmcid": pmcid,
        "pmc_url": pmc_url,
        "pdf_url": pdf_url,
        # abstract intentionally omitted (would require EFetch)
    }


def search_pubmed(
    query: str,
    offset: int = 0,
    limit: int = 10,
    from_year: int | None = None,
    to_year: int | None = None,
    free_full_text: bool = False,
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    """
    Search PubMed via ESearch and hydrate items via ESummary.

    Returns (items, total_results, error_message).
    """
    try:
        if not query or not query.strip():
            return [], 0, None

        # 1) ESearch: find PMIDs
        term = _build_term(query, free_full_text)
        esearch_params: dict[str, Any] = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retstart": str(max(0, int(offset))),
            "retmax": str(max(1, min(int(limit), 200))),  # keep reasonable
            "sort": "relevance",
        }
        if from_year or to_year:
            # Use publication date limits
            # PubMed expects YYYY; if only one bound, set both to that year for a single-year filter
            fy = int(from_year) if from_year else int(to_year)  # type: ignore[arg-type]
            ty = int(to_year) if to_year else int(from_year)   # type: ignore[arg-type]
            esearch_params.update({
                "datetype": "pdat",
                "mindate": str(fy),
                "maxdate": str(ty),
            })

        esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
        data = fetch_json(method="GET", url=esearch_url, params=esearch_params, timeout=15)
        esr = data.get("esearchresult") or {}
        idlist: list[str] = esr.get("idlist") or []
        total = int(esr.get("count") or 0)
        if not idlist:
            return [], total, None

        # 2) ESummary for the returned PMIDs
        ids = ",".join(idlist)
        esum_params = {"db": "pubmed", "id": ids, "retmode": "json"}
        esum_url = f"{EUTILS_BASE}/esummary.fcgi"
        j = fetch_json(method="GET", url=esum_url, params=esum_params, timeout=15)
        result = j.get("result") or {}
        # UIDs list in j['result']['uids'] may be present
        uids = result.get("uids") or idlist
        items: list[dict[str, Any]] = []
        for uid in uids:
            raw = result.get(str(uid))
            if not isinstance(raw, dict):
                continue
            items.append(_normalize_item(str(uid), raw))

        return items, total, None
    except TimeoutError:
        return None, 0, "PubMed request timed out."
    except Exception:
        return None, 0, "PubMed request failed."


def get_pubmed_by_id(pmid: str) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch PubMed record details by PMID, including abstract via EFetch XML.

    Returns (item_dict, error_message). Shape aligns with PubMedPaper.
    """
    try:
        if not pmid or not str(pmid).strip():
            return None, "PMID cannot be empty"
        pmid_str = str(pmid).strip()

        # Summary first (JSON) for structured metadata
        esum_url = f"{EUTILS_BASE}/esummary.fcgi"
        esum_params = {"db": "pubmed", "id": pmid_str, "retmode": "json"}
        j = fetch_json(method="GET", url=esum_url, params=esum_params, timeout=15)
        result = j.get("result") or {}
        rec = result.get(pmid_str)
        if not isinstance(rec, dict):
            return None, None
        base = _normalize_item(pmid_str, rec)

        # EFetch for abstract (XML)
        efetch_url = f"{EUTILS_BASE}/efetch.fcgi"
        efetch_params = {"db": "pubmed", "id": pmid_str, "retmode": "xml"}
        rf = fetch(method="GET", url=efetch_url, params=efetch_params, timeout=15)
        try:
            xml_text = rf.text
        finally:
            with contextlib.suppress(Exception):
                rf.close()
        # Simple XML parsing for AbstractText nodes
        abstract_text = None
        try:
            root = DET.fromstring(xml_text)
            # Path: PubmedArticle/MedlineCitation/Article/Abstract/AbstractText
            for abst in root.findall('.//Abstract/AbstractText'):
                part = ''.join(abst.itertext()).strip()
                if part:
                    abstract_text = (abstract_text + "\n\n" + part) if abstract_text else part
        except Exception:
            # If XML parse fails, leave abstract as None
            abstract_text = None

        if abstract_text:
            base["abstract"] = abstract_text
        return base, None
    except TimeoutError:
        return None, "PubMed by-id request timed out."
    except Exception:
        return None, "PubMed by-id request failed."
