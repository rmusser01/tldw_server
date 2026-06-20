from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Research.discovery.models import (
    DiscoveryMetrics,
    DiscoveryOACandidate,
    DiscoveryProvenance,
    DiscoveryResult,
    DiscoverySearchResponse,
    SourceStatus,
)


pytestmark = pytest.mark.unit


def _client_with_service(service):
    from tldw_Server_API.app.api.v1.endpoints import research_discovery

    app = FastAPI()
    app.include_router(research_discovery.router, prefix="/api/v1/research")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_discovery.get_research_discovery_service] = lambda: service
    return TestClient(app)


def _discovery_response() -> DiscoverySearchResponse:
    return DiscoverySearchResponse(
        discovery_id="rd_test",
        query="retrieval augmented generation",
        results=(
            DiscoveryResult(
                result_id="result-1",
                fingerprint="fp-1",
                primary_source_id="openalex",
                primary_provider="openalex",
                discovery_mode="api",
                title="Retrieval augmented generation overview",
                authors=("Ada Lovelace",),
                abstract="A paper about retrieval augmented generation.",
                doi="10.1000/rag",
                pmid=None,
                pmcid=None,
                arxiv_id=None,
                provider_ids={"openalex": "W123"},
                canonical_url="https://openalex.org/W123",
                published_at="2024-01-01",
                updated_at=None,
                source_category="open_research_graph",
                oa_candidates=(
                    DiscoveryOACandidate(
                        candidate_id="oa-1",
                        candidate_type="landing_page",
                        safe_url="https://example.test/paper",
                        resolver_reference=None,
                        url_redacted=False,
                        requires_reresolution=False,
                        provider="openalex",
                        access_status="open",
                        license_hint="cc-by",
                        content_type_hint="html",
                        rank=1,
                        confidence=0.9,
                        warnings=(),
                    ),
                ),
                recommended_candidate_id="oa-1",
                ingest_eligible=True,
                dedupe_confidence=1.0,
                ranking_signals={"score": 0.95},
                warnings=(),
                merged_provenance=(
                    DiscoveryProvenance(
                        source_id="openalex",
                        provider="openalex",
                        discovery_mode="api",
                        provider_ids={"openalex": "W123"},
                        url="https://openalex.org/W123",
                        source_rank=1,
                        status="ok",
                        warnings=(),
                        safe_metadata={"source": "test"},
                        adapter_version="test",
                    ),
                ),
                safe_metadata={"topic": "rag"},
                adapter_version="test",
                catalog_version="research-discovery-v1",
            ),
        ),
        source_statuses=(
            SourceStatus(
                source_id="openalex",
                provider="openalex",
                status="ok",
                message=None,
                result_count=1,
                elapsed_ms=12.5,
                warnings=(),
            ),
        ),
        warnings=(),
        effective_config={"source_ids": ["openalex"], "fallback_policy": "disabled"},
        catalog_version="research-discovery-v1",
        metrics=DiscoveryMetrics(
            selected_source_count=1,
            result_count=1,
            deduped_result_count=1,
            oa_candidate_count=1,
            elapsed_ms=20.0,
        ),
    )


def test_sources_returns_default_catalog_openalex_flags():
    from tldw_Server_API.app.api.v1.endpoints import research_discovery

    app = FastAPI()
    app.include_router(research_discovery.router, prefix="/api/v1/research")

    with TestClient(app) as client:
        response = client.get("/api/v1/research/sources")

    assert response.status_code == 200
    payload = response.json()
    assert payload["catalog_version"] == "research-discovery-v1"
    openalex = next(source for source in payload["sources"] if source["source_id"] == "openalex")
    assert openalex["configured"] is True
    assert openalex["fallback_enabled"] is False
    assert openalex["fallback_configurable"] is False
    assert openalex["capabilities"]["searchable"] is True


def test_search_passes_owner_and_request_to_service():
    captured = {}

    class StubService:
        async def search(self, **kwargs):
            captured.update(kwargs)
            return _discovery_response()

    with _client_with_service(StubService()) as client:
        response = client.post(
            "/api/v1/research/discovery/search",
            json={
                "query": " retrieval augmented generation ",
                "source_ids": ["openalex"],
                "categories": ["open_research_graph"],
                "per_source_limit": 3,
                "total_limit": 7,
                "fallback_policy": "disabled",
                "filters": {"year": 2024},
            },
        )

    assert response.status_code == 200
    assert captured == {
        "owner_user_id": "1",
        "query": " retrieval augmented generation ",
        "source_ids": ["openalex"],
        "categories": ["open_research_graph"],
        "per_source_limit": 3,
        "total_limit": 7,
        "fallback_policy": "disabled",
        "filters": {"year": 2024},
    }
    payload = response.json()
    assert payload["discovery_id"] == "rd_test"
    assert payload["results"][0]["title"] == "Retrieval augmented generation overview"
    assert payload["metrics"]["result_count"] == 1


def test_content_router_specs_include_research_discovery():
    from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs

    specs = list(iter_content_router_specs())
    spec = next(item for item in specs if item.prefix == "/api/v1/research" and item.tags == ("research-discovery",))

    assert spec.prefix == "/api/v1/research"
    assert spec.tags == ("research-discovery",)
    assert spec.router is not None


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (ValueError("source_selection_over_cap:9:8"), 422),
        (ValueError("research_discovery_fallback_disabled"), 422),
        (ValueError("research_discovery_no_runnable_sources"), 422),
        (ValueError("research_discovery_query_contains_unsafe_url"), 422),
        (ValueError("research_discovery_filters_contain_unsafe_url"), 422),
        (RuntimeError("research_discovery_all_sources_failed"), 502),
        (TimeoutError("research_discovery_total_timeout"), 504),
        (RuntimeError("research_discovery_total_timeout"), 504),
        (ValueError("unknown_source:missing"), 400),
    ],
)
def test_search_maps_service_errors(exc, expected_status):
    class StubService:
        async def search(self, **_kwargs):
            raise exc

    with _client_with_service(StubService()) as client:
        response = client.post(
            "/api/v1/research/discovery/search",
            json={"query": "open access", "source_ids": ["openalex"]},
        )

    assert response.status_code == expected_status
    assert response.json()["detail"] == str(exc)
