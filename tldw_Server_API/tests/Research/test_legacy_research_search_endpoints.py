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
