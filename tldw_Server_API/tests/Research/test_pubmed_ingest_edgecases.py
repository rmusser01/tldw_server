import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_pubmed_ingest_requires_pmcid(monkeypatch, paper_search_app):

    from tldw_Server_API.app.core.Third_Party import PubMed as _Pub

    def _fake_pubmed_by_id(pmid):

        return {
            "pmid": pmid,
            "pmcid": None,  # No OA PMCID available
            "title": "Test",
        }, None

    monkeypatch.setattr(_Pub, "get_pubmed_by_id", _fake_pubmed_by_id)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/pubmed/ingest",
            params={"pmid": "123456"},
        )
        assert r.status_code == 400
        assert "No PMC Open Access PMCID" in r.text


@pytest.mark.asyncio
async def test_pubmed_ingest_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    import tldw_Server_API.app.api.v1.endpoints.paper_search as paper_search
    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA
    from tldw_Server_API.app.core.Third_Party import PubMed as _Pub

    class _MediaDb:
        backend = object()

    class _FakeRepo:
        def __init__(self):
            self.calls = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 13, "pubmed-uuid", "ok"

    def _fake_pubmed_by_id(pmid):
        return {
            "pmid": pmid,
            "pmcid": "PMC123",
            "title": "Test Title",
            "authors": [{"name": "Alice"}],
            "journal": "Journal",
            "pub_date": "2020-01-01",
            "externalIds": {"DOI": "10.1000/xyz"},
            "pdf_url": None,
            "pmc_url": None,
        }, None

    def _fake_download_pmc_pdf(pmcid):
        return b"%PDF-1.5\n...", "paper.pdf", None

    async def _fake_process_pdf_task(**kwargs):
        return {"status": "Success", "content": "pubmed text", "summary": "pubmed summary"}

    media_db = _MediaDb()
    fake_repo = _FakeRepo()
    paper_search_app.dependency_overrides[get_media_db_for_user] = lambda: media_db

    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.media_db.api.get_media_repository", lambda db: fake_repo, raising=False)
    monkeypatch.setattr(_Pub, "get_pubmed_by_id", _fake_pubmed_by_id)
    monkeypatch.setattr(_OA, "download_pmc_pdf", _fake_download_pmc_pdf)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_process_pdf_task)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/pubmed/ingest",
            params={"pmid": "123456", "perform_chunking": False},
        )
        assert r.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "pmid:123456"
        assert saved["title"] == "Test Title"
        assert saved["author"] == "Alice"
        assert saved["content"] == "pubmed text"
        assert saved["analysis_content"] == "pubmed summary"
        assert '"pmcid": "123"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(get_media_db_for_user, None)
