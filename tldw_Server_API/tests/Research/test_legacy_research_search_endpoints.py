from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.research import router as research_router


pytestmark = pytest.mark.integration


mini_app = FastAPI()
mini_app.include_router(research_router, prefix="/api/v1/research")


@pytest.fixture()
def legacy_research_client():
    mini_app.dependency_overrides[get_media_db_for_user] = lambda: object()
    with TestClient(mini_app) as client:
        yield client
    mini_app.dependency_overrides.clear()


def test_arxiv_search_sanitizes_generic_failure(
    legacy_research_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_search_arxiv_custom_api(query, author, year, start_index, results_per_page):
        raise RuntimeError("arxiv backend exploded")

    monkeypatch.setattr(research_module, "search_arxiv_custom_api", fake_search_arxiv_custom_api)

    resp = legacy_research_client.get(
        "/api/v1/research/arxiv-search",
        params={"query": "transformer", "page": 1, "results_per_page": 2},
    )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "An unexpected error occurred while searching arXiv"


def test_arxiv_search_includes_canonical_page_pagination(
    legacy_research_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_search_arxiv_custom_api(query, author, year, start_index, results_per_page):
        return (
            [
                {
                    "id": "arxiv-1",
                    "title": "Transformer Paper",
                    "authors": "A. Researcher",
                    "published_date": "2024-01-01",
                    "abstract": "Test abstract",
                    "pdf_url": "https://arxiv.org/pdf/1234.5678",
                }
            ],
            1,
            None,
        )

    monkeypatch.setattr(research_module, "search_arxiv_custom_api", fake_search_arxiv_custom_api)

    resp = legacy_research_client.get(
        "/api/v1/research/arxiv-search",
        params={"query": "transformer", "page": 1, "results_per_page": 2},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 2,
        "total": 1,
        "total_pages": 1,
        "has_more": False,
    }


def test_semantic_scholar_search_sanitizes_generic_failure(
    legacy_research_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_search_papers_semantic_scholar(
        query,
        offset,
        limit,
        fields_of_study_list,
        publication_types_list,
        year_range,
        venue_list,
        min_citations,
        open_access_only,
    ):
        raise RuntimeError("semantic scholar backend exploded")

    monkeypatch.setattr(
        research_module,
        "search_papers_semantic_scholar",
        fake_search_papers_semantic_scholar,
    )

    resp = legacy_research_client.get(
        "/api/v1/research/semantic-scholar-search",
        params={"query": "transformer", "page": 1, "results_per_page": 2},
    )

    assert resp.status_code == 500
    assert (
        resp.json()["detail"]
        == "An unexpected error occurred while searching Semantic Scholar"
    )


def test_semantic_scholar_search_includes_canonical_page_pagination(
    legacy_research_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_search_papers_semantic_scholar(
        query,
        offset,
        limit,
        fields_of_study_list,
        publication_types_list,
        year_range,
        venue_list,
        min_citations,
        open_access_only,
    ):
        return (
            {
                "total": 1,
                "offset": 0,
                "next": None,
                "data": [
                    {
                        "paperId": "paper-1",
                        "title": "Semantic Scholar Result",
                        "authors": [{"authorId": "1", "name": "A. Researcher"}],
                    }
                ],
            },
            None,
        )

    monkeypatch.setattr(
        research_module,
        "search_papers_semantic_scholar",
        fake_search_papers_semantic_scholar,
    )

    resp = legacy_research_client.get(
        "/api/v1/research/semantic-scholar-search",
        params={"query": "transformer", "page": 1, "results_per_page": 2},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 2,
        "total": 1,
        "total_pages": 1,
        "has_more": False,
    }
