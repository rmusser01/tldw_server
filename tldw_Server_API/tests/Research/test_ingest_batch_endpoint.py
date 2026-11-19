from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


class _MediaDb:
    backend = object()


class _FakeRepo:
    def __init__(self, media_uuid: str) -> None:
        self.calls: list[dict[str, object]] = []
        self.media_uuid = media_uuid

    def add_media_with_keywords(self, **kwargs):
        self.calls.append(kwargs)
        return 41, self.media_uuid, "stored"


def _override_media_repo(paper_search_app, monkeypatch, media_uuid: str):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    import tldw_Server_API.app.api.v1.endpoints.paper_search as paper_search

    media_db = _MediaDb()
    fake_repo = _FakeRepo(media_uuid)
    paper_search_app.dependency_overrides[get_media_db_for_user] = lambda: media_db
    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.media_db.api.get_media_repository", lambda db: fake_repo, raising=False)
    return fake_repo, get_media_db_for_user, paper_search


@pytest.mark.asyncio
async def test_ingest_batch_routes_direct_pdf_url_items_through_media_repository(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "11111111-1111-4111-8111-111111111111",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    async def _fake_process_pdf_task(**kwargs):
        return {
            "text": "batch pdf text",
            "analysis": "batch pdf summary",
            "metadata": {"title": "Meta Title", "author": "Meta Author"},
        }

    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_process_pdf_task)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/ingest/batch",
            json={
                "items": [
                    {
                        "pdf_url": "https://example.com/direct.pdf",
                        "title": "Direct PDF",
                        "author": "Batch Author",
                        "keywords": ["direct"],
                    }
                ],
                "perform_chunking": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert data["results"][0]["success"] is True
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "https://example.com/direct.pdf"
        assert saved["title"] == "Direct PDF"
        assert saved["author"] == "Batch Author"
        assert saved["content"] == "batch pdf text"
        assert saved["analysis_content"] == "batch pdf summary"
        assert '"provider": "batch"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_ingest_batch_routes_pmcid_items_through_media_repository(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, _paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "22222222-2222-4222-8222-222222222222",
    )

    def _fake_download_pmc_pdf(pmcid):
        return b"%PDF-1.5\n...", "paper.pdf", None

    async def _fake_process_pdf_task(**kwargs):
        return {
            "text": "pmcid batch text",
            "analysis": "pmcid batch summary",
            "metadata": {"title": "PMC Title", "author": "PMC Author"},
        }

    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA

    monkeypatch.setattr(_OA, "download_pmc_pdf", _fake_download_pmc_pdf)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_process_pdf_task)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/ingest/batch",
            json={
                "items": [
                    {
                        "pmcid": "123",
                        "keywords": ["pmc"],
                    }
                ],
                "perform_chunking": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert data["results"][0]["success"] is True
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "pmcid:PMC123"
        assert saved["title"] == "PMC Title"
        assert saved["author"] == "PMC Author"
        assert saved["content"] == "pmcid batch text"
        assert saved["analysis_content"] == "pmcid batch summary"
        assert '"pmcid": "PMC123"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_ingest_batch_routes_arxiv_items_through_media_repository(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "33333333-3333-4333-8333-333333333333",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    def _fake_fetch_arxiv_xml(arxiv_id):
        return "<feed></feed>"

    def _fake_parse_arxiv_feed(xml_content: bytes):
        return [
            {
                "id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "published_date": "2017-06-01",
                "pdf_url": None,
            }
        ]

    async def _fake_process_pdf_task(**kwargs):
        return {
            "text": "arxiv batch text",
            "analysis": "arxiv batch summary",
            "metadata": {},
        }

    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    from tldw_Server_API.app.core.Third_Party import Arxiv as _Arxiv

    monkeypatch.setattr(_Arxiv, "fetch_arxiv_xml", _fake_fetch_arxiv_xml)
    monkeypatch.setattr(_Arxiv, "parse_arxiv_feed", _fake_parse_arxiv_feed)
    monkeypatch.setattr(_Arxiv, "fetch_arxiv_pdf_url", lambda arxiv_id: None)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_process_pdf_task)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/ingest/batch",
            json={
                "items": [
                    {
                        "arxiv_id": "1706.03762",
                        "keywords": ["transformers"],
                    }
                ],
                "perform_chunking": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert data["results"][0]["success"] is True
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "arxiv:1706.03762"
        assert saved["title"] == "Attention Is All You Need"
        assert saved["author"] == "Vaswani et al."
        assert saved["content"] == "arxiv batch text"
        assert saved["analysis_content"] == "arxiv batch summary"
        assert '"arxiv_id": "1706.03762"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_ingest_batch_routes_doi_items_through_media_repository(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "44444444-4444-4444-8444-444444444444",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    async def _fake_process_pdf_task(**kwargs):
        return {
            "text": "doi batch text",
            "analysis": "doi batch summary",
        }

    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    from tldw_Server_API.app.core.Third_Party import Unpaywall as _Unpaywall

    monkeypatch.setattr(_Unpaywall, "resolve_oa_pdf", lambda doi: ("https://example.com/doi.pdf", None))

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_process_pdf_task)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/ingest/batch",
            json={
                "items": [
                    {
                        "doi": "10.1000/xyz",
                        "title": "DOI Batch",
                        "author": "DOI Author",
                    }
                ],
                "perform_chunking": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert data["results"][0]["success"] is True
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "doi:10.1000/xyz"
        assert saved["title"] == "DOI Batch"
        assert saved["author"] == "DOI Author"
        assert saved["content"] == "doi batch text"
        assert saved["analysis_content"] == "doi batch summary"
        assert saved["safe_metadata"] == {
            "provider": "batch",
            "doi": "10.1000/xyz",
            "pdf_url": "https://example.com/doi.pdf",
        }

    paper_search_app.dependency_overrides.pop(dep_key, None)
