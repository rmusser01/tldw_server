import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_pmc_oai_identify_success(monkeypatch, paper_search_app):

    def _fake_identify():

        return {"repositoryName": "PMC OAI"}, None

    from tldw_Server_API.app.core.Third_Party import PMC_OAI as _OAI
    monkeypatch.setattr(_OAI, "pmc_oai_identify", _fake_identify)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get("/api/v1/paper-search/pmc-oai/identify")
        assert r.status_code == 200
        assert r.json()["info"]["repositoryName"] == "PMC OAI"


@pytest.mark.asyncio
async def test_pmc_oai_list_records_success(monkeypatch, paper_search_app):

    def _fake_list_records(metadataPrefix, f, u, s, token):

        return [
            {"header": {"identifier": "oai:pubmedcentral.nih.gov:123"}, "metadata": {"title": "X"}},
            {"header": {"identifier": "oai:pubmedcentral.nih.gov:124"}, "metadata": {"title": "Y"}},
        ], "abc123", None

    from tldw_Server_API.app.core.Third_Party import PMC_OAI as _OAI
    monkeypatch.setattr(_OAI, "pmc_oai_list_records", _fake_list_records)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get("/api/v1/paper-search/pmc-oai/list-records", params={"metadataPrefix": "oai_dc"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["items"]) == 2
        assert data["resumption_token"] == "abc123"


@pytest.mark.asyncio
async def test_pmc_oa_identify_success(monkeypatch, paper_search_app):

    def _fake_oa_identify():

        return {"repositoryName": "PMC OA"}, None

    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA
    monkeypatch.setattr(_OA, "pmc_oa_identify", _fake_oa_identify)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get("/api/v1/paper-search/pmc-oa/identify")
        assert r.status_code == 200
        assert r.json()["info"]["repositoryName"] == "PMC OA"


@pytest.mark.asyncio
async def test_pmc_oa_query_success(monkeypatch, paper_search_app):

    def _fake_oa_query(f, u, fmt, token, idp):

        return [
            {"id": "PMC123", "links": [{"format": "pdf", "href": "http://example/pdf"}]}
        ], "nextToken", None

    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA
    monkeypatch.setattr(_OA, "pmc_oa_query", _fake_oa_query)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get("/api/v1/paper-search/pmc-oa/query", params={"from": "2024-01-01", "format": "pdf"})
        assert r.status_code == 200
        data = r.json()
        assert data["items"][0]["id"] == "PMC123"
        assert data["resumption_token"] == "nextToken"


@pytest.mark.asyncio
async def test_pmc_oa_query_filtering(monkeypatch, paper_search_app):

    def _fake_oa_query(f, u, fmt, token, idp):

        return [
            {"id": "PMC1", "license": "CC BY", "links": [{"format": "pdf", "href": "http://x/a.pdf"}]},
            {"id": "PMC2", "license": "All rights reserved", "links": [{"format": "tgz", "href": "http://x/a.tgz"}]},
        ], None, None

    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA
    monkeypatch.setattr(_OA, "pmc_oa_query", _fake_oa_query)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        # pdf_only filter
        r = await client.get("/api/v1/paper-search/pmc-oa/query", params={"pdf_only": True})
        assert r.status_code == 200
        data = r.json()
        assert len(data["items"]) == 1 and data["items"][0]["id"] == "PMC1"
        # license filter
        r2 = await client.get("/api/v1/paper-search/pmc-oa/query", params={"license_contains": "cc by"})
        assert r2.status_code == 200
        d2 = r2.json()
        assert len(d2["items"]) == 1 and d2["items"][0]["id"] == "PMC1"


@pytest.mark.asyncio
async def test_pmc_oa_ingest_pdf_success(monkeypatch, paper_search_app):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user

    class _FakeDB:
        def __init__(self):
            self.captured = []
        def add_media_with_keywords(self, **kwargs):
            self.captured.append(kwargs)
            return 101, "uuid-123", "ok"

    # Override DB dependency
    fake_db = _FakeDB()
    paper_search_app.dependency_overrides[get_media_db_for_user] = lambda: fake_db

    def _fake_download_pdf(pmcid):

        return b"%PDF-1.5\n...", "paper.pdf", None

    async def _fake_process_pdf_task(**kwargs):
        return {
            "status": "Success",
            "content": "PDF text",
            "summary": "Summary",
            "metadata": {"title": "Paper", "keywords": ["pmc"]},
        }

    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA
    monkeypatch.setattr(_OA, "download_pmc_pdf", _fake_download_pdf)

    from tldw_Server_API.app.core.Third_Party import PMC_OAI as _OAI
    monkeypatch.setattr(_OAI, "pmc_oai_get_record", lambda identifier, metadataPrefix: (None, None))

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF
    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_process_pdf_task)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/pmc-oa/ingest-pdf",
            params={
                "pmcid": "PMC1234567",
                "keywords": "k1,k2",
                "enable_ocr": True,
                "ocr_backend": "tesseract",
                "ocr_lang": "eng",
                "chunk_method": "semantic",
                "chunk_size": 400,
                "chunk_overlap": 100,
                "perform_chunking": True,
                "perform_analysis": True,
                "summarize_recursively": False,
                "summary_max_tokens": 256,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["db_id"] == 101
        assert data["media_uuid"] == "uuid-123"
        # Capture kwargs passed to DB
        assert fake_db.captured, "DB call not captured"
        kwargs = fake_db.captured[0]
        assert kwargs["chunk_options"]["method"] == "semantic"
        assert kwargs["chunk_options"]["max_size"] == 400
        assert kwargs["chunk_options"]["overlap"] == 100

    # Cleanup override
    paper_search_app.dependency_overrides.pop(get_media_db_for_user, None)


@pytest.mark.asyncio
async def test_pmc_oa_ingest_pdf_enrichment(monkeypatch, paper_search_app):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user

    class _FakeDB:
        def __init__(self):
            self.captured = []
        def add_media_with_keywords(self, **kwargs):
            self.captured.append(kwargs)
            return 101, "uuid-123", "ok"

    fake_db = _FakeDB()
    paper_search_app.dependency_overrides[get_media_db_for_user] = lambda: fake_db

    def _fake_download_pdf(pmcid):

        return b"%PDF-1.5\n...", "paper.pdf", None

    async def _fake_process_pdf_task(**kwargs):
        return {
            "status": "Success",
            "content": "PDF text",
            "summary": "Summary",
            "metadata": {"keywords": ["pmc"]},
        }

    def _fake_oai_get_record(identifier, metadataPrefix):

        return {
            "metadata": {
                "title": "OAI Title",
                "creators": ["Alice", "Bob"],
                "doi": "10.1000/abc",
                "pmid": "123456",
                "pmcid": "987654",
                "date": "2024-01-01",
                "license_urls": ["https://creativecommons.org/licenses/by/4.0/"],
                "rights": ["CC BY 4.0"],
            }
        }, None

    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA
    monkeypatch.setattr(_OA, "download_pmc_pdf", _fake_download_pdf)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF
    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_process_pdf_task)

    from tldw_Server_API.app.core.Third_Party import PMC_OAI as _OAI
    monkeypatch.setattr(_OAI, "pmc_oai_get_record", _fake_oai_get_record)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.post(
            "/api/v1/paper-search/pmc-oa/ingest-pdf",
            params={"pmcid": "PMC999", "enrich_metadata": True},
        )
        assert r.status_code == 200
        assert fake_db.captured, "DB call not captured"
        kwargs = fake_db.captured[0]
        assert kwargs["title"] == "OAI Title"
        assert kwargs["author"] == "Alice; Bob"
        # Ensure keywords merged
        assert set(kwargs["keywords"]) >= {"pmc"}
        # Safe metadata present
        assert kwargs.get("safe_metadata") is not None

    paper_search_app.dependency_overrides.pop(get_media_db_for_user, None)


@pytest.mark.asyncio
async def test_pmc_oa_fetch_pdf_success(monkeypatch, paper_search_app):

    def _fake_download_pdf(pmcid):

        return b"%PDF-1.4\n...", "PMC9999999.pdf", None

    from tldw_Server_API.app.core.Third_Party import PMC_OA as _OA
    monkeypatch.setattr(_OA, "download_pmc_pdf", _fake_download_pdf)

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        r = await client.get("/api/v1/paper-search/pmc-oa/fetch-pdf", params={"pmcid": "PMC9999999"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert "filename=\"PMC9999999.pdf\"" in r.headers.get("content-disposition", "")
