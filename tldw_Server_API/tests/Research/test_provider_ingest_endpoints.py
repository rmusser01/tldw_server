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
        return 57, self.media_uuid, "stored"


def _override_media_repo(paper_search_app, monkeypatch, media_uuid: str):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    import tldw_Server_API.app.api.v1.endpoints.paper_search as paper_search

    media_db = _MediaDb()
    fake_repo = _FakeRepo(media_uuid)
    paper_search_app.dependency_overrides[get_media_db_for_user] = lambda: media_db
    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.media_db.api.get_media_repository", lambda db: fake_repo, raising=False)
    return fake_repo, get_media_db_for_user, paper_search


def _fake_pdf_process_result(*, content: str, summary: str):
    async def _fake_process_pdf_task(**kwargs):
        return {
            "content": content,
            "summary": summary,
        }

    return _fake_process_pdf_task


@pytest.mark.asyncio
async def test_eartharxiv_ingest_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "55555555-5555-4555-8555-555555555555",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    from tldw_Server_API.app.core.Third_Party import EarthRxiv as _EarthRxiv

    monkeypatch.setattr(_EarthRxiv, "get_item_by_id", lambda osf_id: ({"title": "Earth Title", "doi": "10.1000/earth"}, None))
    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_pdf_process_result(content="earth text", summary="earth summary"))

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/paper-search/earthrxiv/ingest",
            params={"osf_id": "abc123", "perform_chunking": False},
        )
        assert response.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "eartharxiv:abc123"
        assert saved["title"] == "Earth Title"
        assert saved["content"] == "earth text"
        assert saved["analysis_content"] == "earth summary"
        assert '"source": "eartharxiv"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_osf_ingest_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "66666666-6666-4666-8666-666666666666",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    from tldw_Server_API.app.core.Third_Party import OSF as _OSF

    monkeypatch.setattr(_OSF, "get_primary_file_download_url", lambda osf_id: ("https://example.com/osf.pdf", None))
    monkeypatch.setattr(_OSF, "get_preprint_by_id", lambda osf_id: ({"title": "OSF Title", "doi": "10.1000/osf"}, None))
    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_pdf_process_result(content="osf text", summary="osf summary"))

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/paper-search/osf/ingest",
            params={"osf_id": "osf123", "perform_chunking": False},
        )
        assert response.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "osf:osf123"
        assert saved["title"] == "OSF Title"
        assert saved["content"] == "osf text"
        assert saved["analysis_content"] == "osf summary"
        assert '"source": "osf"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_zenodo_ingest_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "77777777-7777-4777-8777-777777777777",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    from tldw_Server_API.app.core.Third_Party import Zenodo as _Zenodo

    monkeypatch.setattr(
        _Zenodo,
        "get_record_by_id",
        lambda record_id: ({"title": "Zenodo Title", "doi": "10.1000/zenodo", "pdf_url": "https://example.com/zenodo.pdf"}, None),
    )
    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_pdf_process_result(content="zenodo text", summary="zenodo summary"))

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/paper-search/zenodo/ingest",
            params={"record_id": "123456", "perform_chunking": False},
        )
        assert response.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "zenodo:123456"
        assert saved["title"] == "Zenodo Title"
        assert saved["content"] == "zenodo text"
        assert saved["analysis_content"] == "zenodo summary"
        assert '"source": "zenodo"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_figshare_ingest_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "88888888-8888-4888-8888-888888888888",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    from tldw_Server_API.app.core.Third_Party import Figshare as _Figshare

    monkeypatch.setattr(_Figshare, "get_article_raw", lambda article_id: ({"title": "Figshare Title", "doi": "10.1000/fig", "id": article_id}, None))
    monkeypatch.setattr(_Figshare, "extract_pdf_download_url", lambda raw: "https://example.com/figshare.pdf")
    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_pdf_process_result(content="figshare text", summary="figshare summary"))

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/paper-search/figshare/ingest",
            params={"article_id": "42", "perform_chunking": False},
        )
        assert response.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "figshare:42"
        assert saved["title"] == "Figshare Title"
        assert saved["content"] == "figshare text"
        assert saved["analysis_content"] == "figshare summary"
        assert '"source": "figshare"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_figshare_ingest_by_doi_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "99999999-9999-4999-8999-999999999999",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    from tldw_Server_API.app.core.Third_Party import Figshare as _Figshare

    monkeypatch.setattr(_Figshare, "get_article_by_doi", lambda doi: ({"id": "84"}, None))
    monkeypatch.setattr(_Figshare, "get_article_raw", lambda article_id: ({"title": "Figshare DOI Title", "doi": "10.1000/fig-doi", "id": article_id}, None))
    monkeypatch.setattr(_Figshare, "extract_pdf_download_url", lambda raw: "https://example.com/figshare-doi.pdf")
    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_pdf_process_result(content="figshare doi text", summary="figshare doi summary"))

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/paper-search/figshare/ingest-by-doi",
            params={"doi": "10.1000/fig-doi", "perform_chunking": False},
        )
        assert response.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "figshare:84"
        assert saved["title"] == "Figshare DOI Title"
        assert saved["content"] == "figshare doi text"
        assert saved["analysis_content"] == "figshare doi summary"
        assert '"source": "figshare"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_hal_ingest_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    from tldw_Server_API.app.core.Third_Party import HAL as _HAL

    monkeypatch.setattr(
        _HAL,
        "by_docid",
        lambda docid, _unused, scope: ({"title": "HAL Title", "doi": "10.1000/hal", "pdf_url": "https://example.com/hal.pdf", "authors": "HAL Author"}, None),
    )
    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_pdf_process_result(content="hal text", summary="hal summary"))

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/paper-search/hal/ingest",
            params={"docid": "hal-123", "perform_chunking": False},
        )
        assert response.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "hal:hal-123"
        assert saved["title"] == "HAL Title"
        assert saved["author"] == "HAL Author"
        assert saved["content"] == "hal text"
        assert saved["analysis_content"] == "hal summary"
        assert '"source": "hal"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)


@pytest.mark.asyncio
async def test_vixra_ingest_uses_media_repository_for_media_db_sessions(
    monkeypatch,
    paper_search_app,
):
    fake_repo, dep_key, paper_search = _override_media_repo(
        paper_search_app,
        monkeypatch,
        "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    )

    async def _fake_download_pdf_bytes(url, **kwargs):
        return b"%PDF-1.5\n..."

    from tldw_Server_API.app.core.Third_Party import Vixra as _Vixra

    monkeypatch.setattr(_Vixra, "get_vixra_by_id", lambda vid: ({"title": "viXra Title", "pdf_url": "https://example.com/vixra.pdf"}, None))
    monkeypatch.setattr(paper_search, "_download_pdf_bytes", _fake_download_pdf_bytes)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as _PDF

    monkeypatch.setattr(_PDF, "process_pdf_task", _fake_pdf_process_result(content="vixra text", summary="vixra summary"))

    async with AsyncClient(transport=ASGITransport(app=paper_search_app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/paper-search/vixra/ingest",
            params={"vid": "1901.0001", "perform_chunking": False},
        )
        assert response.status_code == 200
        assert fake_repo.calls
        saved = fake_repo.calls[0]
        assert saved["url"] == "vixra:1901.0001"
        assert saved["title"] == "viXra Title"
        assert saved["content"] == "vixra text"
        assert saved["analysis_content"] == "vixra summary"
        assert '"source": "vixra"' in saved["safe_metadata"]

    paper_search_app.dependency_overrides.pop(dep_key, None)
