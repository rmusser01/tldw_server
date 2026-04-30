import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_paper_search_arxiv_success(monkeypatch, paper_search_app):
    def _fake_arxiv(query, author, year, start_index, page_size):
        items = [
            {
                "id": "1234.5678v1",
                "title": "Attention Is All You Need",
                "authors": "Vaswani, A.; Shazeer, N.; et al.",
                "published_date": "2017-06-01",
                "abstract": "We propose the Transformer...",
                "pdf_url": "http://arxiv.org/pdf/1234.5678.pdf",
            },
            {
                "id": "2345.6789v1",
                "title": "Transformers in Vision",
                "authors": "Dosovitskiy, A.; et al.",
                "published_date": "2020-10-01",
                "abstract": "We explore ViT...",
                "pdf_url": "http://arxiv.org/pdf/2345.6789.pdf",
            },
        ]
        return items, 2, None

    from tldw_Server_API.app.core.Third_Party import Arxiv as _Arxiv
    monkeypatch.setattr(_Arxiv, "search_arxiv_custom_api", _fake_arxiv)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/arxiv",
            params={"query": "transformer", "page": 1, "results_per_page": 2},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] == 2
        assert len(data["items"]) == 2
        assert data["pagination"] == {
            "mode": "page",
            "page": 1,
            "per_page": 2,
            "total": 2,
            "total_pages": 1,
            "has_more": False,
        }


@pytest.mark.asyncio
async def test_paper_search_biorxiv_success(monkeypatch, paper_search_app):

    def _fake_bio(q, server, f, t, category, offset, limit, recent_days=None, recent_count=None):

        items = [
            {
                "doi": "10.1101/2020.01.01.123456",
                "title": "Sample BioRxiv Paper",
                "authors": "Doe, J.; Roe, R.",
                "category": "bioinformatics",
                "date": "2020-01-01",
                "abstract": "This is a test abstract.",
                "server": server,
                "version": 1,
                "url": "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1",
                "pdf_url": "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1.full.pdf",
            }
        ]
        return items, 1, None

    from tldw_Server_API.app.core.Third_Party import BioRxiv as _Bio
    monkeypatch.setattr(_Bio, "search_biorxiv", _fake_bio)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/biorxiv",
            params={"q": "genomics", "server": "biorxiv", "page": 1, "results_per_page": 1},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] == 1
        assert len(data["items"]) == 1
        assert data["pagination"] == {
            "mode": "page",
            "page": 1,
            "per_page": 1,
            "total": 1,
            "total_pages": 1,
            "has_more": False,
        }


@pytest.mark.asyncio
async def test_paper_search_semantic_scholar_success(monkeypatch, paper_search_app):

    def _fake_s2(query, offset, limit, fos, pub_types, year_range, venue, min_citations, open_access_only, fields_to_return=None):

        return (
            {
                "total": 1,
                "offset": 0,
                "next": None,
                "data": [
                    {
                        "paperId": "paper-1",
                        "title": "Graph Neural Networks",
                        "authors": [{"authorId": "1", "name": "A. Researcher"}],
                    }
                ],
            },
            None,
        )

    from tldw_Server_API.app.core.Third_Party import Semantic_Scholar as _S2
    monkeypatch.setattr(_S2, "search_papers_semantic_scholar", _fake_s2)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/semantic-scholar",
            params={"query": "graph neural networks", "page": 1, "results_per_page": 2},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] == 1
        assert len(data["items"]) == 1
        assert data["pagination"] == {
            "mode": "page",
            "page": 1,
            "per_page": 2,
            "total": 1,
            "total_pages": 1,
            "has_more": False,
        }


@pytest.mark.asyncio
async def test_paper_search_semantic_scholar_error_mapping(monkeypatch, paper_search_app):

    def _fake_s2(query, offset, limit, fos, pub_types, year_range, venue, min_citations, open_access_only, fields_to_return=None):

        return None, "Semantic Scholar API HTTP Error: 429 - Too Many Requests."

    from tldw_Server_API.app.core.Third_Party import Semantic_Scholar as _S2
    monkeypatch.setattr(_S2, "search_papers_semantic_scholar", _fake_s2)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/semantic-scholar",
            params={"query": "graph neural networks", "page": 1, "results_per_page": 2},
        )
        # Expect 502 mapped from HTTP error
        assert r.status_code in (429, 502)


@pytest.mark.asyncio
async def test_paper_search_pubmed_success(monkeypatch, paper_search_app):

    def _fake_pubmed(q, offset, limit, from_year, to_year, free_full_text):

        items = [
            {
                "pmid": "12345678",
                "title": "Sample PubMed Article",
                "authors": "Doe J, Roe R",
                "journal": "Sample Journal",
                "pub_date": "2021 Jan 01",
                "doi": "10.1000/example.doi",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "pmcid": "7654321",
                "pmc_url": "https://pmc.ncbi.nlm.nih.gov/7654321/",
                "pdf_url": "https://pmc.ncbi.nlm.nih.gov/7654321/pdf",
            }
        ]
        return items, 1, None

    from tldw_Server_API.app.core.Third_Party import PubMed as _Pub
    monkeypatch.setattr(_Pub, "search_pubmed", _fake_pubmed)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/pubmed",
            params={"q": "cancer immunotherapy", "page": 1, "results_per_page": 1, "free_full_text": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] == 1
        assert len(data["items"]) == 1
        assert data["pagination"] == {
            "mode": "page",
            "page": 1,
            "per_page": 1,
            "total": 1,
            "total_pages": 1,
            "has_more": False,
        }


@pytest.mark.asyncio
async def test_paper_search_pubmed_error_mapping(monkeypatch, paper_search_app):

    def _fake_pubmed_err(q, offset, limit, from_year, to_year, free_full_text):

        return None, 0, "PubMed API HTTP Error: 429"

    from tldw_Server_API.app.core.Third_Party import PubMed as _Pub
    monkeypatch.setattr(_Pub, "search_pubmed", _fake_pubmed_err)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/pubmed",
            params={"q": "cancer", "page": 1, "results_per_page": 1},
        )
    assert r.status_code in (429, 502)


@pytest.mark.asyncio
async def test_biorxiv_by_doi_success(monkeypatch, paper_search_app):

    def _fake_by_doi(doi, server):

        return {
            "doi": doi,
            "title": "A test preprint",
            "authors": "A. Author; B. Author",
            "category": "bioinformatics",
            "date": "2024-09-01",
            "abstract": "Test abstract",
            "server": server,
            "version": 1,
            "url": f"https://www.{server}.org/content/{doi}v1",
            "pdf_url": f"https://www.{server}.org/content/{doi}v1.full.pdf",
        }, None

    from tldw_Server_API.app.core.Third_Party import BioRxiv as _Bio
    monkeypatch.setattr(_Bio, "get_biorxiv_by_doi", _fake_by_doi)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/biorxiv/by-doi",
            params={"doi": "10.1101/2021.11.09.467936", "server": "biorxiv"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["doi"].startswith("10.1101/")


@pytest.mark.asyncio
async def test_biorxiv_by_doi_not_found(monkeypatch, paper_search_app):

    def _fake_by_doi_notfound(doi, server):

        return None, None

    from tldw_Server_API.app.core.Third_Party import BioRxiv as _Bio
    monkeypatch.setattr(_Bio, "get_biorxiv_by_doi", _fake_by_doi_notfound)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/biorxiv/by-doi",
            params={"doi": "10.1101/does.not.exist", "server": "biorxiv"},
        )
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_arxiv_by_id_success(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_arxiv_by_id(paper_id):

        return {
            "id": paper_id,
            "title": "Attention Is All You Need",
            "authors": "Vaswani, A.; Shazeer, N.; et al.",
            "published_date": "2017-06-01",
            "abstract": "We propose the Transformer...",
            "pdf_url": "http://arxiv.org/pdf/1706.03762.pdf",
        }, None

    from tldw_Server_API.app.core.Third_Party import Arxiv as _Arxiv
    monkeypatch.setattr(_Arxiv, "get_arxiv_by_id", _fake_arxiv_by_id)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/arxiv/by-id",
            params={"id": "1706.03762"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "1706.03762"


@pytest.mark.asyncio
async def test_arxiv_by_id_not_found(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_arxiv_by_id_notfound(paper_id):

        return None, None

    from tldw_Server_API.app.core.Third_Party import Arxiv as _Arxiv
    monkeypatch.setattr(_Arxiv, "get_arxiv_by_id", _fake_arxiv_by_id_notfound)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/arxiv/by-id",
            params={"id": "0000.00000"},
        )
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_semantic_scholar_by_id_success(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_s2_details(paper_id, fields_to_return='paperId,title,abstract,year,citationCount,authors,venue,openAccessPdf,url,publicationTypes,publicationDate,externalIds'):

        return {
            "paperId": paper_id,
            "title": "Graph Neural Networks",
            "abstract": "An overview of GNNs...",
            "year": 2020,
            "citationCount": 1234,
            "authors": [{"name": "A. Author"}],
            "venue": "NeurIPS",
            "openAccessPdf": {"url": "https://example/pdf", "status": "GREEN"},
            "url": "https://www.semanticscholar.org/paper/abcdef",
            "publicationTypes": ["JournalArticle"],
            "publicationDate": "2020-12-01",
            "externalIds": {"DOI": "10.1000/xyz"}
        }, None

    from tldw_Server_API.app.core.Third_Party import Semantic_Scholar as _S2
    monkeypatch.setattr(_S2, "get_paper_details_semantic_scholar", _fake_s2_details)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/semantic-scholar/by-id",
            params={"paper_id": "abcdef"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["paperId"] == "abcdef"


@pytest.mark.asyncio
async def test_semantic_scholar_by_id_not_found(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_s2_notfound(paper_id, fields_to_return='paperId,title,abstract,year,citationCount,authors,venue,openAccessPdf,url,publicationTypes,publicationDate,externalIds'):

        return None, "Semantic Scholar API HTTP Error: 404 - Not Found"

    from tldw_Server_API.app.core.Third_Party import Semantic_Scholar as _S2
    monkeypatch.setattr(_S2, "get_paper_details_semantic_scholar", _fake_s2_notfound)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/semantic-scholar/by-id",
            params={"paper_id": "notfound"},
        )
        assert r.status_code in (404, 502)


@pytest.mark.asyncio
async def test_pubmed_by_id_success(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_pubmed_by_id(pmid):

        return {
            "pmid": pmid,
            "title": "A Sample PubMed Record",
            "authors": "Doe J, Roe R",
            "journal": "Journal of Testing",
            "pub_date": "2022 Jan 01",
            "abstract": "This is a test abstract from PubMed.",
            "doi": "10.1000/test",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "pmcid": None,
            "pmc_url": None,
            "pdf_url": None,
        }, None

    from tldw_Server_API.app.core.Third_Party import PubMed as _Pub
    monkeypatch.setattr(_Pub, "get_pubmed_by_id", _fake_pubmed_by_id)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/pubmed/by-id",
            params={"pmid": "12345678"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["pmid"] == "12345678"
        assert data.get("abstract") is not None


@pytest.mark.asyncio
async def test_pubmed_by_id_not_found(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_pubmed_by_id_notfound(pmid):

        return None, None

    from tldw_Server_API.app.core.Third_Party import PubMed as _Pub
    monkeypatch.setattr(_Pub, "get_pubmed_by_id", _fake_pubmed_by_id_notfound)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/pubmed/by-id",
            params={"pmid": "99999999"},
        )
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_biorxiv_pubs_search_success(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_pubs(server, f, t, offset, limit, recent_days, recent_count, q):

        items = [{
            "biorxiv_doi": "10.1101/2021.11.09.467936",
            "published_doi": "10.7554/eLife.75393",
            "published_journal": "eLife",
            "preprint_platform": server,
            "preprint_title": "A test preprint",
            "preprint_authors": "Doe, J.; Roe, R.",
            "preprint_category": "cell biology",
            "preprint_date": "2024-09-01",
            "published_date": "2024-11-01",
            "preprint_abstract": "Test",
            "preprint_author_corresponding": "Doe",
            "preprint_author_corresponding_institution": "Uni"
        }]
        return items, 1, None

    from tldw_Server_API.app.core.Third_Party import BioRxiv as _Bio
    monkeypatch.setattr(_Bio, "search_biorxiv_pubs", _fake_pubs)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/biorxiv-pubs",
            params={"server": "biorxiv", "recent_days": 7, "page": 1, "results_per_page": 10},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_results"] == 1
        # Now request compact (no abstracts)
        r2 = await client.get(
            "/api/v1/paper-search/biorxiv-pubs",
            params={"server": "biorxiv", "recent_days": 7, "page": 1, "results_per_page": 10, "include_abstracts": False},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["items"][0].get("preprint_abstract") is None


@pytest.mark.asyncio
async def test_biorxiv_pubs_by_doi_success(monkeypatch):
    from tldw_Server_API.app.main import app

    def _fake_pub_by_doi(doi, server):

        return {
            "biorxiv_doi": doi,
            "published_doi": "10.7554/eLife.75393",
            "published_journal": "eLife",
            "preprint_platform": server,
        }, None

    from tldw_Server_API.app.core.Third_Party import BioRxiv as _Bio
    monkeypatch.setattr(_Bio, "get_biorxiv_published_by_doi", _fake_pub_by_doi)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/paper-search/biorxiv-pubs/by-doi",
            params={"doi": "10.1101/2021.11.09.467936", "server": "biorxiv"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["published_doi"] is not None
        # Compact include_abstracts=false
        r2 = await client.get(
            "/api/v1/paper-search/biorxiv-pubs/by-doi",
            params={"doi": "10.1101/2021.11.09.467936", "server": "biorxiv", "include_abstracts": False},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2.get("preprint_abstract") is None
