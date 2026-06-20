import asyncio

import pytest


@pytest.mark.asyncio
async def test_router_calls_adapter_for_selected_source():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class FakeAdapter:
        async def search(self, *, query, source, limit, filters):
            return [
                {
                    "source_id": source.source_id,
                    "provider": "openalex",
                    "title": query,
                    "doi": "10.1000/example",
                }
            ]

    catalog = default_source_catalog()
    router = ResearchSourceRouter(catalog=catalog, adapters={"openalex": FakeAdapter()})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records[0]["source_id"] == "openalex"
    assert records[0]["source_category"] == "open_research_graph"
    assert records[0]["provider"] == "openalex"
    assert records[0]["discovery_mode"] == "api"
    assert records[0]["adapter_version"]
    assert records[0]["source_priority"] == catalog.get_source("openalex").priority
    assert statuses[0].source_id == "openalex"
    assert statuses[0].status == "ok"
    assert statuses[0].result_count == 1


@pytest.mark.asyncio
async def test_router_records_provider_error_without_leaking_exception_details():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import (
        DiscoveryProviderError,
        ResearchSourceRouter,
    )

    class FailingAdapter:
        async def search(self, **_kwargs):
            raise DiscoveryProviderError("secret token /private/key")

    catalog = default_source_catalog()
    router = ResearchSourceRouter(catalog=catalog, adapters={"openalex": FailingAdapter()})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "provider_error"
    assert statuses[0].message == "Provider request failed."
    assert "secret token" not in statuses[0].message
    assert "/private/key" not in statuses[0].message


@pytest.mark.asyncio
async def test_router_maps_wrapped_provider_helper_exception_to_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    def failing_provider_helper(*_args, **_kwargs):
        raise RuntimeError("secret token /private/key")

    catalog = default_source_catalog()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={"openalex": OpenAlexDiscoveryAdapter(search_fn=failing_provider_helper)},
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "provider_error"
    assert statuses[0].message == "Provider request failed."
    assert "secret token" not in statuses[0].message
    assert "/private/key" not in statuses[0].message


@pytest.mark.asyncio
async def test_router_maps_malformed_provider_payload_to_provider_error():
    from tldw_Server_API.app.core.Research.discovery.adapters import OpenAlexDiscoveryAdapter
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    def malformed_provider_helper(*_args, **_kwargs):
        return "not-json", 1, None

    catalog = default_source_catalog()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={"openalex": OpenAlexDiscoveryAdapter(search_fn=malformed_provider_helper)},
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "provider_error"
    assert statuses[0].message == "Provider request failed."
    assert "not-json" not in statuses[0].message


@pytest.mark.asyncio
async def test_router_reports_unexpected_adapter_bug_as_internal_error_without_leaking_details():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class BuggyAdapter:
        async def search(self, **_kwargs):
            raise RuntimeError("secret token /private/key")

    catalog = default_source_catalog()
    router = ResearchSourceRouter(catalog=catalog, adapters={"openalex": BuggyAdapter()})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "internal_error"
    assert statuses[0].message == "Discovery adapter failed unexpectedly."
    assert "secret token" not in statuses[0].message
    assert "/private/key" not in statuses[0].message


@pytest.mark.asyncio
async def test_router_reports_none_adapter_result_as_internal_error():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class MalformedAdapter:
        async def search(self, **_kwargs):
            return None

    catalog = default_source_catalog()
    router = ResearchSourceRouter(catalog=catalog, adapters={"openalex": MalformedAdapter()})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "internal_error"
    assert statuses[0].message == "Discovery adapter failed unexpectedly."


@pytest.mark.asyncio
async def test_router_reports_non_dict_adapter_records_as_internal_error():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class MalformedAdapter:
        async def search(self, **_kwargs):
            return ["not-a-record"]

    catalog = default_source_catalog()
    router = ResearchSourceRouter(catalog=catalog, adapters={"openalex": MalformedAdapter()})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "internal_error"
    assert statuses[0].message == "Discovery adapter failed unexpectedly."


@pytest.mark.asyncio
async def test_router_marks_source_timeout_without_blocking_other_sources():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class SlowAdapter:
        async def search(self, **_kwargs):
            await asyncio.sleep(1)
            return []

    class FastAdapter:
        async def search(self, *, source, **_kwargs):
            return [
                {
                    "source_id": source.source_id,
                    "provider": "crossref",
                    "title": "Fast",
                }
            ]

    catalog = default_source_catalog()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={"openalex": SlowAdapter(), "crossref": FastAdapter()},
        per_source_timeout_seconds=0.01,
        max_concurrency=2,
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex"), catalog.get_source("crossref")],
        per_source_limit=3,
        filters={},
    )

    assert [record["source_id"] for record in records] == ["crossref"]
    assert {status.source_id: status.status for status in statuses} == {
        "openalex": "timeout",
        "crossref": "ok",
    }
    assert statuses[0].warnings == ("provider_call_may_continue_after_timeout",)


@pytest.mark.asyncio
async def test_router_respects_rate_limiter_without_calling_adapter():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class Adapter:
        async def search(self, **_kwargs):
            raise AssertionError("rate-limited source should not call adapter")

    async def deny_openalex(source_id):
        return source_id != "openalex"

    catalog = default_source_catalog()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={"openalex": Adapter()},
        rate_limiter=deny_openalex,
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "rate_limited"


@pytest.mark.asyncio
async def test_router_reports_policy_and_configuration_blocked_sources():
    from tldw_Server_API.app.core.Research.discovery.catalog import ResearchSourceCatalog
    from tldw_Server_API.app.core.Research.discovery.models import (
        ResearchSourceCatalogEntry,
        SourceCapabilities,
    )
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    def entry(
        source_id,
        *,
        enabled=True,
        configured=True,
        requires_credentials=False,
        adapter="adapter",
        discovery_mode=None,
    ):
        return ResearchSourceCatalogEntry(
            source_id=source_id,
            display_name=source_id,
            category="test",
            subcategory=None,
            content_types=("paper",),
            access_level="credentialed_api" if requires_credentials else "public_api",
            enabled=enabled,
            configured=configured,
            default_discovery_mode=discovery_mode if discovery_mode is not None else "api",
            fallback_enabled=False,
            priority=1,
            provider_adapter=adapter,
            site_hosts=(),
            trust_notes="test",
            capabilities=SourceCapabilities(
                searchable=True,
                full_text_resolvable=False,
                ingestable=False,
                requires_credentials=requires_credentials,
                fallback_search_allowed=False,
                rate_limited=False,
            ),
            catalog_version="test-v1",
        )

    catalog = ResearchSourceCatalog(
        entries=[
            entry("disabled_repo", enabled=False),
            entry("disabled_mode", discovery_mode="disabled"),
            entry("credentialed_index", configured=False, requires_credentials=True),
            entry("no_adapter", adapter=None),
            entry("missing_adapter", adapter="missing"),
        ],
        max_selected_sources=5,
    )
    router = ResearchSourceRouter(catalog=catalog, adapters={})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[
            catalog.get_source("disabled_repo"),
            catalog.get_source("disabled_mode"),
            catalog.get_source("credentialed_index"),
            catalog.get_source("no_adapter"),
            catalog.get_source("missing_adapter"),
        ],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert {status.source_id: status.status for status in statuses} == {
        "disabled_repo": "policy_blocked",
        "disabled_mode": "policy_blocked",
        "credentialed_index": "credentials_missing",
        "no_adapter": "provider_not_configured",
        "missing_adapter": "provider_not_configured",
    }


@pytest.mark.asyncio
async def test_router_enforces_bounded_concurrency():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    active = 0
    max_seen = 0

    class CountingAdapter:
        async def search(self, *, source, **_kwargs):
            nonlocal active, max_seen
            active += 1
            max_seen = max(max_seen, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [
                {
                    "source_id": source.source_id,
                    "provider": source.source_id,
                    "title": source.source_id,
                }
            ]

    catalog = default_source_catalog(max_selected_sources=10)
    adapter = CountingAdapter()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={
            "openalex": adapter,
            "semantic_scholar": adapter,
            "crossref": adapter,
        },
        max_concurrency=1,
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[
            catalog.get_source("openalex"),
            catalog.get_source("semantic_scholar"),
            catalog.get_source("crossref"),
        ],
        per_source_limit=3,
        filters={},
    )

    assert len(records) == 3
    assert all(status.status == "ok" for status in statuses)
    assert max_seen == 1
