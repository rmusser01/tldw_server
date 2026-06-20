"""First-slice provider adapters for research discovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .models import ResearchSourceCatalogEntry
from .router import DiscoveryProviderAdapter


SearchFunction = Callable[..., object]


class OpenAlexDiscoveryAdapter:
    """Discovery adapter for the existing OpenAlex helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_openalex_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            query,
            0,
            limit,
            filter_venue=None,
            from_year=filters.get("from_year"),
            to_year=filters.get("to_year"),
        )
        return _normalize_records(_items_from_tuple_result(result), provider="openalex")


class SemanticScholarDiscoveryAdapter:
    """Discovery adapter for the existing Semantic Scholar helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_semantic_scholar_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            query,
            offset=0,
            limit=limit,
            fields_of_study=None,
            publication_types=None,
            year_range=filters.get("year_range"),
            venue=None,
            min_citations=None,
            open_access_only=False,
        )
        return _normalize_records(
            _items_from_semantic_scholar_result(result),
            provider="semantic_scholar",
        )


class CrossrefDiscoveryAdapter:
    """Discovery adapter for the existing Crossref helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_crossref_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            query,
            0,
            limit,
            filter_venue=None,
            from_year=filters.get("from_year"),
            to_year=filters.get("to_year"),
        )
        return _normalize_records(_items_from_tuple_result(result), provider="crossref")


class ArxivDiscoveryAdapter:
    """Discovery adapter for the existing arXiv helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_arxiv_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            query,
            author=None,
            year=filters.get("year"),
            start_index=0,
            page_size=limit,
        )
        return _normalize_records(_items_from_tuple_result(result), provider="arxiv")


class PubMedDiscoveryAdapter:
    """Discovery adapter for the existing PubMed helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_pubmed_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            query,
            offset=0,
            limit=limit,
            from_year=filters.get("from_year"),
            to_year=filters.get("to_year"),
            free_full_text=False,
        )
        return _normalize_records(_items_from_tuple_result(result), provider="pubmed")


class ZenodoDiscoveryAdapter:
    """Discovery adapter for the existing Zenodo helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_zenodo_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            query,
            page=1,
            size=limit,
            type_=None,
            subtype=None,
            communities=None,
        )
        return _normalize_records(_items_from_tuple_result(result), provider="zenodo")


class FigshareDiscoveryAdapter:
    """Discovery adapter for the existing Figshare helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_figshare_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            query,
            page=1,
            page_size=limit,
            order=None,
            order_direction=None,
            search_for=None,
        )
        return _normalize_records(_items_from_tuple_result(result), provider="figshare")


class OSFDiscoveryAdapter:
    """Discovery adapter for the existing OSF helper."""

    def __init__(self, search_fn: SearchFunction | None = None) -> None:
        self._search_fn = search_fn or _default_osf_search

    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._search_fn,
            term=query,
            page=1,
            results_per_page=limit,
            provider=None,
            from_date=filters.get("from_date"),
        )
        return _normalize_records(_items_from_tuple_result(result), provider="osf")


def default_discovery_adapters() -> dict[str, DiscoveryProviderAdapter]:
    return {
        "openalex": OpenAlexDiscoveryAdapter(),
        "semantic_scholar": SemanticScholarDiscoveryAdapter(),
        "crossref": CrossrefDiscoveryAdapter(),
        "arxiv": ArxivDiscoveryAdapter(),
        "pubmed": PubMedDiscoveryAdapter(),
        "zenodo": ZenodoDiscoveryAdapter(),
        "figshare": FigshareDiscoveryAdapter(),
        "osf": OSFDiscoveryAdapter(),
    }


def _items_from_tuple_result(result: object) -> list[dict[str, Any]]:
    if not isinstance(result, tuple):
        return _items_from_payload(result)

    if len(result) == 3:
        payload, _total, error = result
    elif len(result) == 2:
        payload, error = result
    else:
        payload = result[0] if result else None
        error = None

    if error:
        raise RuntimeError("Provider request failed.")
    return _items_from_payload(payload)


def _items_from_semantic_scholar_result(result: object) -> list[dict[str, Any]]:
    if isinstance(result, tuple):
        if len(result) == 2:
            payload, error = result
        elif len(result) == 3:
            payload, _total, error = result
        else:
            payload = result[0] if result else None
            error = None
        if error:
            raise RuntimeError("Provider request failed.")
    else:
        payload = result

    return _items_from_payload(payload)


def _items_from_payload(payload: object) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, Mapping):
        for key in ("data", "items", "results", "hits"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, Mapping):
                nested_hits = nested.get("hits")
                if isinstance(nested_hits, list):
                    return [item for item in nested_hits if isinstance(item, dict)]
        return [dict(payload)]
    return []


def _normalize_records(
    records: list[dict[str, Any]],
    *,
    provider: str,
) -> list[dict[str, Any]]:
    return [_normalize_record(record, provider=provider) for record in records]


def _normalize_record(record: dict[str, Any], *, provider: str) -> dict[str, Any]:
    provider_ids = _provider_ids(record, provider=provider)
    abstract = _text(
        record.get("abstract")
        or record.get("snippet")
        or record.get("summary")
        or record.get("description")
    )

    arxiv_id = _text(
        record.get("arxiv_id")
        or record.get("arxiv")
        or _semantic_external_id(record, "ArXiv")
    )
    if provider == "arxiv" and not arxiv_id:
        arxiv_id = _text(record.get("id"))

    pmid = _text(record.get("pmid") or _semantic_external_id(record, "PubMed"))
    pmcid = _text(
        record.get("pmcid")
        or _semantic_external_id(record, "PubMedCentral")
        or _semantic_external_id(record, "PMCID")
    )
    doi = _text(record.get("doi") or _semantic_external_id(record, "DOI"))
    url = _text(record.get("url"))
    if provider == "arxiv" and not url and arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"

    return {
        "title": _text(record.get("title")) or "",
        "authors": _authors(record.get("authors")),
        "abstract": abstract,
        "snippet": _text(record.get("snippet")) or abstract,
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "arxiv_id": arxiv_id,
        "url": url,
        "pdf_url": _text(record.get("pdf_url") or _open_access_pdf_url(record)),
        "provider": provider,
        "provider_ids": provider_ids,
    }


def _provider_ids(record: dict[str, Any], *, provider: str) -> dict[str, str]:
    provider_ids: dict[str, str] = {}
    record_id = _text(record.get("id") or record.get("paperId") or record.get("uid"))
    if record_id:
        provider_ids["id"] = record_id
        provider_ids[f"{provider}_id"] = record_id

    semantic_id = _text(record.get("paperId"))
    if semantic_id:
        provider_ids["semantic_scholar_id"] = semantic_id

    doi = _text(record.get("doi") or _semantic_external_id(record, "DOI"))
    if doi:
        provider_ids["doi"] = doi

    pmid = _text(record.get("pmid") or _semantic_external_id(record, "PubMed"))
    if pmid:
        provider_ids["pmid"] = pmid

    pmcid = _text(
        record.get("pmcid")
        or _semantic_external_id(record, "PubMedCentral")
        or _semantic_external_id(record, "PMCID")
    )
    if pmcid:
        provider_ids["pmcid"] = pmcid

    arxiv_id = _text(
        record.get("arxiv_id")
        or record.get("arxiv")
        or _semantic_external_id(record, "ArXiv")
    )
    if provider == "arxiv" and not arxiv_id:
        arxiv_id = record_id
    if arxiv_id:
        provider_ids["arxiv_id"] = arxiv_id

    return dict(sorted(provider_ids.items()))


def _authors(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        authors: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                name = _text(
                    item.get("name")
                    or item.get("full_name")
                    or " ".join(
                        part
                        for part in (
                            _text(item.get("given")),
                            _text(item.get("family")),
                        )
                        if part
                    )
                )
            else:
                name = _text(item)
            if name:
                authors.append(name)
        return tuple(authors)
    return ()


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _semantic_external_id(record: dict[str, Any], key: str) -> object:
    external_ids = record.get("externalIds")
    if not isinstance(external_ids, Mapping):
        return None
    return external_ids.get(key)


def _open_access_pdf_url(record: dict[str, Any]) -> object:
    open_access_pdf = record.get("openAccessPdf")
    if isinstance(open_access_pdf, Mapping):
        return open_access_pdf.get("url")
    return None


def _default_openalex_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.OpenAlex import search_openalex

    return search_openalex(*args, **kwargs)


def _default_semantic_scholar_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.Semantic_Scholar import (
        search_papers_semantic_scholar,
    )

    return search_papers_semantic_scholar(*args, **kwargs)


def _default_crossref_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.Crossref import search_crossref

    return search_crossref(*args, **kwargs)


def _default_arxiv_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.Arxiv import search_arxiv_custom_api

    return search_arxiv_custom_api(*args, **kwargs)


def _default_pubmed_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.PubMed import search_pubmed

    return search_pubmed(*args, **kwargs)


def _default_zenodo_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.Zenodo import search_records

    return search_records(*args, **kwargs)


def _default_figshare_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.Figshare import search_articles

    return search_articles(*args, **kwargs)


def _default_osf_search(*args: Any, **kwargs: Any) -> object:
    from tldw_Server_API.app.core.Third_Party.OSF import search_preprints

    return search_preprints(*args, **kwargs)
