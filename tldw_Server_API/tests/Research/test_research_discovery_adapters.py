import pytest


pytestmark = pytest.mark.unit


def _source(source_id):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    return default_source_catalog().get_source(source_id)


@pytest.mark.asyncio
async def test_openalex_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter

    captured = {}

    def fake_search_openalex(q, offset, limit, filter_venue=None, from_year=None, to_year=None):
        captured.update(
            {
                "q": q,
                "offset": offset,
                "limit": limit,
                "filter_venue": filter_venue,
                "from_year": from_year,
                "to_year": to_year,
            }
        )
        return (
            [
                {
                    "id": "https://openalex.org/W1",
                    "title": "OpenAlex Paper",
                    "authors": "Ada Lovelace",
                    "abstract": "OpenAlex abstract",
                    "doi": "10.1000/openalex",
                    "url": "https://doi.org/10.1000/openalex",
                    "pdf_url": "https://example.test/openalex.pdf",
                }
            ],
            1,
            None,
        )

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)
    records = await adapter.search(
        query="graph",
        source=_source("openalex"),
        limit=2,
        filters={"from_year": 2020, "to_year": 2024},
    )

    assert captured == {
        "q": "graph",
        "offset": 0,
        "limit": 2,
        "filter_venue": None,
        "from_year": 2020,
        "to_year": 2024,
    }
    assert records[0]["title"] == "OpenAlex Paper"
    assert records[0]["authors"] == ("Ada Lovelace",)
    assert records[0]["abstract"] == "OpenAlex abstract"
    assert records[0]["doi"] == "10.1000/openalex"
    assert records[0]["url"] == "https://doi.org/10.1000/openalex"
    assert records[0]["pdf_url"] == "https://example.test/openalex.pdf"
    assert records[0]["provider"] == "openalex"
    assert records[0]["provider_ids"]["openalex_id"] == "https://openalex.org/W1"


@pytest.mark.asyncio
async def test_adapter_provider_error_tuple_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return None, 0, "secret token /private/key"

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "secret token" not in str(exc_info.value)
    assert "/private/key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_none_payload_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return None, 0, None

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."


@pytest.mark.asyncio
async def test_adapter_helper_exception_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        raise RuntimeError("secret token /private/key")

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "secret token" not in str(exc_info.value)
    assert "/private/key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_error_shaped_mapping_payload_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return {"error": "secret token /private/key"}, 0, None

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "secret token" not in str(exc_info.value)
    assert "/private/key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_error_shaped_mapping_precedes_nested_results():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return (
            {
                "error": "secret token /private/key",
                "results": [{"title": "Should Not Pass"}],
            },
            1,
            None,
        )

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "secret token" not in str(exc_info.value)
    assert "/private/key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_unrecognized_mapping_payload_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return {"total": 1, "count": 1}, 1, None

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "total" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_scalar_payload_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return "not-json", 1, None

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "not-json" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_list_with_non_record_items_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return ["not-a-record"], 1, None

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "not-a-record" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_list_with_non_record_mapping_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return [{"total": 1}], 1, None

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."
    assert "total" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_list_with_empty_mapping_raises_sanitized_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.router import DiscoveryProviderError

    def fake_search_openalex(*_args, **_kwargs):
        return [{}], 1, None

    adapter = OpenAlexDiscoveryAdapter(search_fn=fake_search_openalex)

    with pytest.raises(DiscoveryProviderError) as exc_info:
        await adapter.search(
            query="graph",
            source=_source("openalex"),
            limit=2,
            filters={},
        )

    assert str(exc_info.value) == "Provider request failed."


@pytest.mark.asyncio
async def test_semantic_scholar_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import (
        SemanticScholarDiscoveryAdapter,
    )

    captured = {}

    def fake_search_semantic_scholar(
        query,
        offset=0,
        limit=10,
        fields_of_study=None,
        publication_types=None,
        year_range=None,
        venue=None,
        min_citations=None,
        open_access_only=False,
    ):
        captured.update(
            {
                "query": query,
                "offset": offset,
                "limit": limit,
                "fields_of_study": fields_of_study,
                "publication_types": publication_types,
                "year_range": year_range,
                "venue": venue,
                "min_citations": min_citations,
                "open_access_only": open_access_only,
            }
        )
        return (
            {
                "total": 1,
                "data": [
                    {
                        "paperId": "S2-1",
                        "title": "Semantic Scholar Paper",
                        "abstract": "Semantic Scholar abstract",
                        "authors": [{"name": "Grace Hopper"}],
                        "externalIds": {
                            "DOI": "10.1000/s2",
                            "PubMed": "12345",
                            "ArXiv": "2401.12345",
                        },
                        "url": "https://www.semanticscholar.org/paper/S2-1",
                        "openAccessPdf": {"url": "https://example.test/s2.pdf"},
                    }
                ],
            },
            None,
        )

    adapter = SemanticScholarDiscoveryAdapter(search_fn=fake_search_semantic_scholar)
    records = await adapter.search(
        query="citations",
        source=_source("semantic_scholar"),
        limit=5,
        filters={"year_range": "2020-2024"},
    )

    assert captured == {
        "query": "citations",
        "offset": 0,
        "limit": 5,
        "fields_of_study": None,
        "publication_types": None,
        "year_range": "2020-2024",
        "venue": None,
        "min_citations": None,
        "open_access_only": False,
    }
    assert records[0]["title"] == "Semantic Scholar Paper"
    assert records[0]["authors"] == ("Grace Hopper",)
    assert records[0]["abstract"] == "Semantic Scholar abstract"
    assert records[0]["doi"] == "10.1000/s2"
    assert records[0]["pmid"] == "12345"
    assert records[0]["arxiv_id"] == "2401.12345"
    assert records[0]["pdf_url"] == "https://example.test/s2.pdf"
    assert records[0]["provider"] == "semantic_scholar"
    assert records[0]["provider_ids"]["semantic_scholar_id"] == "S2-1"


@pytest.mark.asyncio
async def test_crossref_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import CrossrefDiscoveryAdapter

    captured = {}

    def fake_search_crossref(q, offset, limit, filter_venue=None, from_year=None, to_year=None):
        captured.update(
            {
                "q": q,
                "offset": offset,
                "limit": limit,
                "filter_venue": filter_venue,
                "from_year": from_year,
                "to_year": to_year,
            }
        )
        return (
            [
                {
                    "id": "10.1000/crossref",
                    "title": "Crossref Paper",
                    "authors": "Katherine Johnson",
                    "abstract": "Crossref abstract",
                    "doi": "10.1000/crossref",
                    "url": "https://doi.org/10.1000/crossref",
                    "pdf_url": "https://example.test/crossref.pdf",
                }
            ],
            1,
            None,
        )

    adapter = CrossrefDiscoveryAdapter(search_fn=fake_search_crossref)
    records = await adapter.search(
        query="metadata",
        source=_source("crossref"),
        limit=4,
        filters={"from_year": 2019, "to_year": 2023},
    )

    assert captured == {
        "q": "metadata",
        "offset": 0,
        "limit": 4,
        "filter_venue": None,
        "from_year": 2019,
        "to_year": 2023,
    }
    assert records[0]["title"] == "Crossref Paper"
    assert records[0]["authors"] == ("Katherine Johnson",)
    assert records[0]["abstract"] == "Crossref abstract"
    assert records[0]["doi"] == "10.1000/crossref"
    assert records[0]["url"] == "https://doi.org/10.1000/crossref"
    assert records[0]["pdf_url"] == "https://example.test/crossref.pdf"
    assert records[0]["provider"] == "crossref"
    assert records[0]["provider_ids"]["crossref_id"] == "10.1000/crossref"


@pytest.mark.asyncio
async def test_arxiv_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import ArxivDiscoveryAdapter

    captured = {}

    def fake_search_arxiv(query, author=None, year=None, start_index=0, page_size=10):
        captured.update(
            {
                "query": query,
                "author": author,
                "year": year,
                "start_index": start_index,
                "page_size": page_size,
            }
        )
        return (
            [
                {
                    "id": "2401.12345v2",
                    "title": "arXiv Paper",
                    "authors": "Barbara Liskov",
                    "abstract": "arXiv abstract",
                    "pdf_url": "https://arxiv.org/pdf/2401.12345v2",
                }
            ],
            1,
            None,
        )

    adapter = ArxivDiscoveryAdapter(search_fn=fake_search_arxiv)
    records = await adapter.search(
        query="types",
        source=_source("arxiv"),
        limit=6,
        filters={"year": "2024"},
    )

    assert captured == {
        "query": "types",
        "author": None,
        "year": "2024",
        "start_index": 0,
        "page_size": 6,
    }
    assert records[0]["title"] == "arXiv Paper"
    assert records[0]["authors"] == ("Barbara Liskov",)
    assert records[0]["abstract"] == "arXiv abstract"
    assert records[0]["arxiv_id"] == "2401.12345v2"
    assert records[0]["url"] == "https://arxiv.org/abs/2401.12345v2"
    assert records[0]["pdf_url"] == "https://arxiv.org/pdf/2401.12345v2"
    assert records[0]["provider"] == "arxiv"
    assert records[0]["provider_ids"]["arxiv_id"] == "2401.12345v2"


@pytest.mark.asyncio
async def test_pubmed_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import PubMedDiscoveryAdapter

    captured = {}

    def fake_search_pubmed(
        query,
        offset=0,
        limit=10,
        from_year=None,
        to_year=None,
        free_full_text=False,
    ):
        captured.update(
            {
                "query": query,
                "offset": offset,
                "limit": limit,
                "from_year": from_year,
                "to_year": to_year,
                "free_full_text": free_full_text,
            }
        )
        return (
            [
                {
                    "pmid": "98765",
                    "pmcid": "PMC123",
                    "title": "PubMed Paper",
                    "authors": "Rosalind Franklin",
                    "abstract": "PubMed abstract",
                    "doi": "10.1000/pubmed",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/98765/",
                    "pdf_url": "https://pmc.ncbi.nlm.nih.gov/PMC123/pdf",
                }
            ],
            1,
            None,
        )

    adapter = PubMedDiscoveryAdapter(search_fn=fake_search_pubmed)
    records = await adapter.search(
        query="medicine",
        source=_source("pubmed"),
        limit=7,
        filters={"from_year": 2018, "to_year": 2024},
    )

    assert captured == {
        "query": "medicine",
        "offset": 0,
        "limit": 7,
        "from_year": 2018,
        "to_year": 2024,
        "free_full_text": False,
    }
    assert records[0]["title"] == "PubMed Paper"
    assert records[0]["authors"] == ("Rosalind Franklin",)
    assert records[0]["abstract"] == "PubMed abstract"
    assert records[0]["doi"] == "10.1000/pubmed"
    assert records[0]["pmid"] == "98765"
    assert records[0]["pmcid"] == "PMC123"
    assert records[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/98765/"
    assert records[0]["pdf_url"] == "https://pmc.ncbi.nlm.nih.gov/PMC123/pdf"
    assert records[0]["provider"] == "pubmed"
    assert records[0]["provider_ids"]["pmid"] == "98765"


@pytest.mark.asyncio
async def test_zenodo_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import ZenodoDiscoveryAdapter

    captured = {}

    def fake_search_zenodo(q, page=1, size=10, type_=None, subtype=None, communities=None):
        captured.update(
            {
                "q": q,
                "page": page,
                "size": size,
                "type_": type_,
                "subtype": subtype,
                "communities": communities,
            }
        )
        return (
            [
                {
                    "id": "zenodo-1",
                    "title": "Zenodo Record",
                    "authors": "Edsger Dijkstra",
                    "abstract": "Zenodo abstract",
                    "doi": "10.5281/zenodo.1",
                    "url": "https://zenodo.org/records/1",
                    "pdf_url": "https://zenodo.org/api/records/1/files/paper.pdf",
                }
            ],
            1,
            None,
        )

    adapter = ZenodoDiscoveryAdapter(search_fn=fake_search_zenodo)
    records = await adapter.search(
        query="dataset",
        source=_source("zenodo"),
        limit=8,
        filters={},
    )

    assert captured == {
        "q": "dataset",
        "page": 1,
        "size": 8,
        "type_": None,
        "subtype": None,
        "communities": None,
    }
    assert records[0]["title"] == "Zenodo Record"
    assert records[0]["authors"] == ("Edsger Dijkstra",)
    assert records[0]["abstract"] == "Zenodo abstract"
    assert records[0]["doi"] == "10.5281/zenodo.1"
    assert records[0]["url"] == "https://zenodo.org/records/1"
    assert records[0]["pdf_url"] == "https://zenodo.org/api/records/1/files/paper.pdf"
    assert records[0]["provider"] == "zenodo"
    assert records[0]["provider_ids"]["zenodo_id"] == "zenodo-1"


@pytest.mark.asyncio
async def test_figshare_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import FigshareDiscoveryAdapter

    captured = {}

    def fake_search_figshare(
        q,
        page=1,
        page_size=10,
        order=None,
        order_direction=None,
        search_for=None,
    ):
        captured.update(
            {
                "q": q,
                "page": page,
                "page_size": page_size,
                "order": order,
                "order_direction": order_direction,
                "search_for": search_for,
            }
        )
        return (
            [
                {
                    "id": "figshare-1",
                    "title": "Figshare Article",
                    "authors": "Margaret Hamilton",
                    "abstract": "Figshare abstract",
                    "doi": "10.6084/figshare.1",
                    "url": "https://figshare.com/articles/1",
                    "pdf_url": "https://ndownloader.figshare.com/files/1",
                }
            ],
            1,
            None,
        )

    adapter = FigshareDiscoveryAdapter(search_fn=fake_search_figshare)
    records = await adapter.search(
        query="artifact",
        source=_source("figshare"),
        limit=9,
        filters={},
    )

    assert captured == {
        "q": "artifact",
        "page": 1,
        "page_size": 9,
        "order": None,
        "order_direction": None,
        "search_for": None,
    }
    assert records[0]["title"] == "Figshare Article"
    assert records[0]["authors"] == ("Margaret Hamilton",)
    assert records[0]["abstract"] == "Figshare abstract"
    assert records[0]["doi"] == "10.6084/figshare.1"
    assert records[0]["url"] == "https://figshare.com/articles/1"
    assert records[0]["pdf_url"] == "https://ndownloader.figshare.com/files/1"
    assert records[0]["provider"] == "figshare"
    assert records[0]["provider_ids"]["figshare_id"] == "figshare-1"


@pytest.mark.asyncio
async def test_osf_adapter_wraps_existing_function_and_normalizes_record():
    from tldw_Server_API.app.core.Research.discovery.adapters import OSFDiscoveryAdapter

    captured = {}

    def fake_search_osf(
        term,
        page=1,
        results_per_page=10,
        provider=None,
        from_date=None,
    ):
        captured.update(
            {
                "term": term,
                "page": page,
                "results_per_page": results_per_page,
                "provider": provider,
                "from_date": from_date,
            }
        )
        return (
            [
                {
                    "id": "osf-1",
                    "title": "OSF Preprint",
                    "authors": "Radia Perlman",
                    "abstract": "OSF abstract",
                    "doi": "10.31219/osf.io/abcde",
                    "url": "https://osf.io/preprints/osf-1",
                    "pdf_url": "https://osf.io/download/osf-1",
                }
            ],
            1,
            None,
        )

    adapter = OSFDiscoveryAdapter(search_fn=fake_search_osf)
    records = await adapter.search(
        query="preprint",
        source=_source("osf"),
        limit=10,
        filters={"from_date": "2024-01-01"},
    )

    assert captured == {
        "term": "preprint",
        "page": 1,
        "results_per_page": 10,
        "provider": None,
        "from_date": "2024-01-01",
    }
    assert records[0]["title"] == "OSF Preprint"
    assert records[0]["authors"] == ("Radia Perlman",)
    assert records[0]["abstract"] == "OSF abstract"
    assert records[0]["doi"] == "10.31219/osf.io/abcde"
    assert records[0]["url"] == "https://osf.io/preprints/osf-1"
    assert records[0]["pdf_url"] == "https://osf.io/download/osf-1"
    assert records[0]["provider"] == "osf"
    assert records[0]["provider_ids"]["osf_id"] == "osf-1"


def test_default_discovery_adapters_contains_first_slice_sources():
    from tldw_Server_API.app.core.Research.discovery.adapters import default_discovery_adapters

    adapters = default_discovery_adapters()

    assert {
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    }.issubset(adapters)
