import asyncio

import pytest


class FakeRouter:
    def __init__(self, *, records=None, statuses=None, delay_seconds=0.0):
        self.records = list(records or [])
        self.statuses = list(statuses or [])
        self.delay_seconds = delay_seconds
        self.calls = []

    async def search_sources(self, *, query, sources, per_source_limit, filters):
        selected_sources = tuple(sources)
        self.calls.append(
            {
                "query": query,
                "source_ids": [source.source_id for source in selected_sources],
                "per_source_limit": per_source_limit,
                "filters": dict(filters),
            }
        )
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return list(self.records), list(self.statuses)


class NoopOAResolver:
    def resolve_for_result(self, **_kwargs):
        return []


class SlowOAResolver:
    def resolve_for_result(self, **_kwargs):
        import time

        time.sleep(0.1)
        return []


def source_status(source_id, status, *, provider=None, result_count=0, warnings=()):
    from tldw_Server_API.app.core.Research.discovery.models import SourceStatus

    return SourceStatus(
        source_id=source_id,
        provider=provider or source_id,
        status=status,
        message=None if status == "ok" else f"{status} message",
        result_count=result_count,
        elapsed_ms=1.0,
        warnings=tuple(warnings),
    )


def service_with_db(tmp_path, *, catalog=None, router=None, retention_hours=24):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchDiscoveryService(
        catalog=catalog,
        router=router,
        oa_resolver=NoopOAResolver(),
        db_factory=lambda _owner_user_id: db,
        snapshot_retention_hours=retention_hours,
    )
    return service, db


@pytest.mark.asyncio
async def test_successful_search_persists_sanitized_discovery_snapshot(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog()
    signed_pdf_url = "https://repo.example/files/paper.pdf?X-Amz-Signature=SECRET"
    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "source_category": "open_research_graph",
                "discovery_mode": "api",
                "adapter_version": "openalex:test",
                "source_priority": 10,
                "title": "Open research paper",
                "authors": ["A. Researcher"],
                "doi": "10.1000/Example",
                "url": "https://publisher.example/paper?token=SECRET",
                "pdf_url": signed_pdf_url,
                "raw_urls": [signed_pdf_url],
                "api_key": "SECRET",
            }
        ],
        statuses=[source_status("openalex", "ok", provider="openalex", result_count=1)],
    )
    service, db = service_with_db(tmp_path, catalog=catalog, router=router)

    response = await service.search(
        owner_user_id="user-1",
        query="open research",
        source_ids=["openalex"],
        per_source_limit=5,
        total_limit=5,
        filters={
            "api_key": "SECRET",
            "safe_filter": "open",
            "signed_url": "https://repo.example/file.pdf?token=SECRET",
        },
    )

    snapshot = db.get_discovery_snapshot(response.discovery_id, owner_user_id="user-1")
    assert response.discovery_id.startswith("rd_")
    assert response.query == "open research"
    assert len(response.results) == 1
    assert response.results[0].oa_candidates[0].safe_url == "https://repo.example/files/paper.pdf"
    assert response.effective_config["source_ids"] == ["openalex"]
    assert response.metrics.selected_source_count == 1
    assert response.metrics.result_count == 1
    assert response.metrics.deduped_result_count == 1
    assert response.metrics.oa_candidate_count >= 1
    assert response.catalog_version == catalog.catalog_version
    assert snapshot is not None
    assert snapshot.request_json["source_ids"] == ["openalex"]
    assert snapshot.request_json["filters"] == {"safe_filter": "open"}
    assert snapshot.effective_config_json["source_ids"] == ["openalex"]
    assert snapshot.response_json["metrics"]["result_count"] == 1
    assert snapshot.response_json["query"] == "open research"
    assert snapshot.response_json["results"][0]["oa_candidates"][0]["safe_url"] == (
        "https://repo.example/files/paper.pdf"
    )
    assert "SECRET" not in str(snapshot.request_json)
    assert "SECRET" not in str(snapshot.response_json)
    assert "X-Amz-Signature" not in str(snapshot.response_json)


@pytest.mark.asyncio
async def test_disabled_fallback_policy_mapping_is_sanitized_before_snapshot(tmp_path):
    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "title": "Safe fallback config",
                "doi": "10.1000/fallback",
            }
        ],
        statuses=[source_status("openalex", "ok", provider="openalex", result_count=1)],
    )
    service, db = service_with_db(tmp_path, router=router)

    response = await service.search(
        owner_user_id="user-1",
        query="fallback",
        source_ids=["openalex"],
        fallback_policy={
            "mode": "disabled",
            "resolver": "https://fallback.example/search?token=SECRET",
        },
    )

    snapshot = db.get_discovery_snapshot(response.discovery_id, owner_user_id="user-1")

    assert snapshot is not None
    assert snapshot.request_json["fallback_policy"] == "disabled"
    assert snapshot.effective_config_json["fallback_policy"] == "disabled"
    assert "SECRET" not in str(snapshot.request_json)
    assert "SECRET" not in str(snapshot.effective_config_json)


@pytest.mark.asyncio
async def test_search_attaches_additional_oa_resolver_candidates(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.oa import build_oa_candidates

    class FakeOAResolver:
        def resolve_for_result(self, *, result_fingerprint, source_id, provider, doi, provider_ids, raw_urls):
            assert raw_urls == ()
            return build_oa_candidates(
                result_fingerprint=result_fingerprint,
                source_id=source_id,
                provider=provider,
                doi=doi,
                provider_ids=provider_ids,
                raw_urls=["https://repo.example/resolver.pdf"],
            )

    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "title": "Resolver paper",
                "doi": "10.1000/resolver",
            }
        ],
        statuses=[source_status("openalex", "ok", provider="openalex", result_count=1)],
    )
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchDiscoveryService(
        router=router,
        oa_resolver=FakeOAResolver(),
        db_factory=lambda _owner_user_id: db,
    )

    response = await service.search(owner_user_id="user-1", query="resolver", source_ids=["openalex"])

    assert response.results[0].oa_candidates[0].safe_url == "https://repo.example/resolver.pdf"


@pytest.mark.asyncio
async def test_total_timeout_covers_oa_resolver_work(tmp_path):
    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "title": "Slow resolver paper",
                "doi": "10.1000/slow",
            }
        ],
        statuses=[source_status("openalex", "ok", provider="openalex", result_count=1)],
    )
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchDiscoveryService(
        router=router,
        oa_resolver=SlowOAResolver(),
        db_factory=lambda _owner_user_id: db,
    )

    with pytest.raises(TimeoutError, match="^research_discovery_total_timeout$"):
        await service.search(
            owner_user_id="user-1",
            query="slow resolver",
            source_ids=["openalex"],
            total_timeout_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_search_rejects_limit_less_than_one(tmp_path):
    service, _db = service_with_db(tmp_path, router=FakeRouter())

    with pytest.raises(ValueError, match="^research_discovery_limit_must_be_positive$"):
        await service.search(
            owner_user_id="user-1",
            query="q",
            source_ids=["openalex"],
            per_source_limit=0,
        )

    with pytest.raises(ValueError, match="^research_discovery_limit_must_be_positive$"):
        await service.search(
            owner_user_id="user-1",
            query="q",
            source_ids=["openalex"],
            total_limit=0,
        )


@pytest.mark.asyncio
async def test_search_rejects_empty_query(tmp_path):
    service, _db = service_with_db(tmp_path, router=FakeRouter())

    with pytest.raises(ValueError, match="^research_discovery_query_required$"):
        await service.search(owner_user_id="user-1", query="   ", source_ids=["openalex"])


@pytest.mark.asyncio
async def test_search_rejects_query_with_embedded_unsafe_url_before_router(tmp_path):
    router = FakeRouter()
    service, _db = service_with_db(tmp_path, router=router)

    with pytest.raises(ValueError, match="^research_discovery_query_contains_unsafe_url$"):
        await service.search(
            owner_user_id="user-1",
            query="see https://repo.example/file.pdf?token=SECRET",
            source_ids=["openalex"],
        )

    assert router.calls == []


@pytest.mark.asyncio
async def test_search_rejects_filter_text_with_embedded_unsafe_url_before_router(tmp_path):
    router = FakeRouter()
    service, _db = service_with_db(tmp_path, router=router)

    with pytest.raises(ValueError, match="^research_discovery_filters_contain_unsafe_url$"):
        await service.search(
            owner_user_id="user-1",
            query="open research",
            source_ids=["openalex"],
            filters={
                "safe_filter": "open",
                "nested": {"note": "see https://repo.example/file.pdf?token=SECRET"},
            },
        )

    assert router.calls == []


@pytest.mark.asyncio
async def test_search_rejects_over_cap_category_selection(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog(max_selected_sources=8)
    service, _db = service_with_db(tmp_path, catalog=catalog, router=FakeRouter())

    with pytest.raises(
        ValueError,
        match=r"^source_selection_over_cap:3:2$",
    ):
        await service.search(
            owner_user_id="user-1",
            query="q",
            categories=["open_research_graph"],
            max_sources=2,
        )


@pytest.mark.asyncio
async def test_partial_failure_returns_warnings_and_source_statuses(tmp_path):
    router = FakeRouter(
        records=[
            {
                "source_id": "crossref",
                "provider": "crossref",
                "title": "Recovered result",
                "doi": "10.1000/recovered",
            }
        ],
        statuses=[
            source_status("openalex", "provider_error", provider="openalex"),
            source_status("crossref", "ok", provider="crossref", result_count=1),
        ],
    )
    service, _db = service_with_db(tmp_path, router=router)

    response = await service.search(
        owner_user_id="user-1",
        query="q",
        source_ids=["openalex", "crossref"],
        limit=3,
    )

    assert len(response.results) == 1
    assert {status.source_id: status.status for status in response.source_statuses} == {
        "openalex": "provider_error",
        "crossref": "ok",
    }
    assert any("openalex" in warning and "provider_error" in warning for warning in response.warnings)
    assert response.metrics.selected_source_count == 2
    assert response.metrics.result_count == 1


@pytest.mark.asyncio
async def test_unsafe_warning_text_is_redacted_from_response_and_snapshot(tmp_path):
    unsafe_warning = "signed URL https://repo.example/paper.pdf?token=SECRET /private/key"
    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "title": "Warning paper",
                "doi": "10.1000/warning",
                "warnings": [unsafe_warning],
            }
        ],
        statuses=[
            source_status(
                "openalex",
                "ok",
                provider="openalex",
                result_count=1,
                warnings=(unsafe_warning,),
            )
        ],
    )
    service, db = service_with_db(tmp_path, router=router)

    response = await service.search(
        owner_user_id="user-1",
        query="warning",
        source_ids=["openalex"],
    )
    snapshot = db.get_discovery_snapshot(response.discovery_id, owner_user_id="user-1")

    assert snapshot is not None
    assert "warning_redacted" in response.warnings
    assert "SECRET" not in str(response)
    assert "token=SECRET" not in str(response)
    assert "/private/key" not in str(response)
    assert "SECRET" not in str(snapshot.response_json)
    assert "token=SECRET" not in str(snapshot.response_json)
    assert "/private/key" not in str(snapshot.response_json)


@pytest.mark.asyncio
async def test_status_and_error_metadata_are_redacted_from_response_and_snapshot(tmp_path):
    unsafe_text = "provider failed https://repo.example/paper.pdf?token=SECRET /private/key"
    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "title": "Status leak paper",
                "doi": "10.1000/status",
                "status": unsafe_text,
                "error_message": unsafe_text,
            }
        ],
        statuses=[source_status("openalex", "ok", provider="openalex", result_count=1)],
    )
    service, db = service_with_db(tmp_path, router=router)

    response = await service.search(
        owner_user_id="user-1",
        query="status",
        source_ids=["openalex"],
    )
    snapshot = db.get_discovery_snapshot(response.discovery_id, owner_user_id="user-1")

    assert snapshot is not None
    assert response.results[0].merged_provenance[0].status == "warning_redacted"
    assert response.results[0].merged_provenance[0].safe_metadata["error_message"] == "warning_redacted"
    assert response.results[0].safe_metadata["error_message"] == "warning_redacted"
    assert "SECRET" not in str(response)
    assert "token=SECRET" not in str(response)
    assert "/private/key" not in str(response)
    assert "SECRET" not in str(snapshot.response_json)
    assert "token=SECRET" not in str(snapshot.response_json)
    assert "/private/key" not in str(snapshot.response_json)


@pytest.mark.asyncio
async def test_total_timeout_raises_stable_timeout_error(tmp_path):
    router = FakeRouter(delay_seconds=0.05)
    service, _db = service_with_db(tmp_path, router=router)

    with pytest.raises(TimeoutError, match="^research_discovery_total_timeout$"):
        await service.search(
            owner_user_id="user-1",
            query="q",
            source_ids=["openalex"],
            total_timeout_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_all_sources_failed_raises_runtime_error(tmp_path):
    router = FakeRouter(
        statuses=[
            source_status("openalex", "provider_error", provider="openalex"),
            source_status("crossref", "timeout", provider="crossref"),
        ]
    )
    service, _db = service_with_db(tmp_path, router=router)

    with pytest.raises(RuntimeError, match="^research_discovery_all_sources_failed$"):
        await service.search(
            owner_user_id="user-1",
            query="q",
            source_ids=["openalex", "crossref"],
        )


@pytest.mark.asyncio
async def test_internal_adapter_errors_count_as_all_sources_failed(tmp_path):
    router = FakeRouter(statuses=[source_status("openalex", "internal_error", provider="openalex")])
    service, _db = service_with_db(tmp_path, router=router)

    with pytest.raises(RuntimeError, match="^research_discovery_all_sources_failed$"):
        await service.search(
            owner_user_id="user-1",
            query="q",
            source_ids=["openalex"],
        )


@pytest.mark.asyncio
async def test_no_runnable_sources_raises_value_error(tmp_path):
    router = FakeRouter(
        statuses=[
            source_status("openalex", "policy_blocked", provider="openalex"),
            source_status("crossref", "provider_not_configured", provider="crossref"),
        ]
    )
    service, _db = service_with_db(tmp_path, router=router)

    with pytest.raises(ValueError, match="^research_discovery_no_runnable_sources$"):
        await service.search(
            owner_user_id="user-1",
            query="q",
            source_ids=["openalex", "crossref"],
        )


@pytest.mark.asyncio
async def test_categories_and_source_ids_resolve_through_catalog(tmp_path):
    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "title": "Category result",
                "doi": "10.1000/category",
            }
        ],
        statuses=[
            source_status("openalex", "ok", provider="openalex", result_count=1),
            source_status("semantic_scholar", "ok", provider="semantic_scholar"),
            source_status("crossref", "ok", provider="crossref"),
        ],
    )
    service, _db = service_with_db(tmp_path, router=router)

    await service.search(
        owner_user_id="user-1",
        query="q",
        source_ids=["openalex"],
        categories=["open_research_graph"],
    )

    assert router.calls[0]["source_ids"] == ["openalex", "semantic_scholar", "crossref"]


@pytest.mark.asyncio
async def test_empty_source_selection_defaults_to_open_research_graph(tmp_path):
    router = FakeRouter(
        records=[
            {
                "source_id": "openalex",
                "provider": "openalex",
                "title": "Default result",
                "doi": "10.1000/default",
            }
        ],
        statuses=[
            source_status("openalex", "ok", provider="openalex", result_count=1),
            source_status("semantic_scholar", "ok", provider="semantic_scholar"),
            source_status("crossref", "ok", provider="crossref"),
        ],
    )
    service, db = service_with_db(tmp_path, router=router)

    response = await service.search(owner_user_id="user-1", query="q")

    assert router.calls[0]["source_ids"] == ["openalex", "semantic_scholar", "crossref"]
    assert response.effective_config["defaulted_categories"] == ["open_research_graph"]
    assert db.get_discovery_snapshot(response.discovery_id, owner_user_id="user-1") is not None


@pytest.mark.asyncio
async def test_fallback_policy_must_be_disabled_in_phase_one(tmp_path):
    service, _db = service_with_db(tmp_path, router=FakeRouter())

    with pytest.raises(ValueError, match="^research_discovery_fallback_disabled$"):
        await service.search(
            owner_user_id="user-1",
            query="q",
            source_ids=["openalex"],
            fallback_policy="enabled",
        )


@pytest.mark.asyncio
async def test_snapshot_creation_uses_configured_retention(tmp_path):
    service, db = service_with_db(tmp_path, router=FakeRouter(), retention_hours=-1)

    response = await service.search(
        owner_user_id="user-1",
        query="q",
        source_ids=["openalex"],
        limit=1,
    )

    assert response.discovery_id.startswith("rd_")
    assert db.get_discovery_snapshot(response.discovery_id, owner_user_id="user-1") is None


def test_default_service_wires_first_slice_adapter_registry():
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    service = ResearchDiscoveryService()

    assert {
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    }.issubset(service.adapter_names)
