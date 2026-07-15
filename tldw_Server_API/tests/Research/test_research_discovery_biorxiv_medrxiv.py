"""Offline contracts for the shadow bioRxiv/medRxiv discovery family."""

from __future__ import annotations

import ast
import asyncio
import http.client
import importlib
import json
import socket
import urllib.request
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from tldw_Server_API.app.core.Research.discovery.contracts import (
    AttributionMatch,
    BoundedDecimalQueryValuePolicy,
    BoundedTextQueryValuePolicy,
    BudgetCeilings,
    CredentialRequirement,
    CredentialStatus,
    DeferredNumericCSVQueryBinding,
    DiscoveryOutcomeIdentity,
    ExactQueryValuePolicy,
    ExecutionMode,
    JSONBodyPair,
    LiteralTermsQueryValuePolicy,
    OperationKind,
    PathSlot,
    PathSlotKind,
    PredicateOperator,
    QueryMode,
    QueryPair,
    ReadinessState,
    RouteKind,
    SourceConstraint,
    SourcePredicate,
    evaluate_source_predicate,
)
from tldw_Server_API.app.core.Research.discovery.executor import (
    DiscoveryAdapterError,
    DiscoveryAdapterResult,
    DiscoveryExecutionError,
    LogicalOutcomeState,
    NumericCursor,
    PhysicalDispatchState,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
)
from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint
from tldw_Server_API.app.core.Research.discovery.planner import (
    DateIntervalQuery,
    GeneralFreeTextQuery,
    IdentifierLookupQuery,
    PlanningRequest,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    foundation_readiness,
    foundation_registry,
)
from tldw_Server_API.app.core.Security.http_hop import HTTPHopLimits

pytestmark = pytest.mark.unit

_MODULE = "tldw_Server_API.app.core.Research.discovery.biorxiv_medrxiv"
_GENERAL_ROUTE_IDS = (
    "biorxiv_europe_pmc_search_aggregator",
    "medrxiv_europe_pmc_search_aggregator",
)
_DETAILS_ROUTE_IDS = (
    "biorxiv_details_lookup_direct",
    "medrxiv_details_lookup_direct",
    "biorxiv_details_interval_direct",
    "medrxiv_details_interval_direct",
)
_FAMILY_ROUTE_IDS = _GENERAL_ROUTE_IDS + _DETAILS_ROUTE_IDS
_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "research_discovery_gateway_adapters"
_NORMALIZED_KEYS = {
    "title",
    "authors",
    "abstract",
    "snippet",
    "doi",
    "pmid",
    "pmcid",
    "arxiv_id",
    "url",
    "pdf_url",
    "provider",
    "provider_ids",
    "published_date",
    "publication_year",
    "ppr_id",
    "source_platform",
}
_DETAILS_NORMALIZED_KEYS = {
    "title",
    "authors",
    "abstract",
    "snippet",
    "doi",
    "pmid",
    "pmcid",
    "arxiv_id",
    "url",
    "pdf_url",
    "provider",
    "provider_ids",
    "published_date",
    "publication_year",
    "version",
    "license",
    "category",
    "published_doi",
    "source_platform",
}


def _module():
    return importlib.import_module(_MODULE)


def _fixture(source_id: str) -> bytes:
    return (_FIXTURE_ROOT / f"europe_pmc_{source_id}_success.json").read_bytes()


def _details_fixture(name: str) -> bytes:
    return (_FIXTURE_ROOT / f"{name}.json").read_bytes()


def _details_plan_for(
    source_id: str,
    query: IdentifierLookupQuery | DateIntervalQuery,
    *,
    result_limit: int = 120,
):
    module = _module()
    registry = module.biorxiv_medrxiv_shadow_registry()
    interval = type(query) is DateIntervalQuery
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=(source_id,),
            query=query,
            filters=(),
            result_limit=result_limit,
        ),
        registry=registry,
        readiness=module.biorxiv_medrxiv_shadow_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=BudgetCeilings(
            max_route_attempts=1,
            max_physical_dispatches=4 if interval else 1,
            max_pages_per_route=4 if interval else 1,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=80_000 if interval else 20_000,
            max_results=result_limit,
        ),
    )
    return registry, plan


def _plan_for(
    source_ids: tuple[str, ...] = ("biorxiv",),
    *,
    result_limit: int = 1,
):
    module = _module()
    registry = module.biorxiv_medrxiv_shadow_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=source_ids,
            query=GeneralFreeTextQuery("  Bounded + Discovery  "),
            filters=(),
            result_limit=result_limit,
        ),
        registry=registry,
        readiness=module.biorxiv_medrxiv_shadow_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=BudgetCeilings(
            max_route_attempts=len(source_ids),
            max_physical_dispatches=len(source_ids),
            max_pages_per_route=1,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=20_000 * len(source_ids),
            max_results=result_limit,
        ),
    )
    return registry, plan


def _response(
    route,
    intent,
    body: Any,
    *,
    status_code: Any = 200,
    content_type: str | None = "application/json",
    retry_after: Any = None,
) -> DiscoveryGatewayResponse:
    origin = route.policy.origin
    body_length = len(body) if hasattr(body, "__len__") else 0
    headers = () if content_type is None else (("content-type", content_type),)
    return DiscoveryGatewayResponse(
        status_code=status_code,
        headers=headers,
        body=body,
        trace=DiscoveryGatewayTrace(
            route_id=route.route_id,
            policy_digest=route.policy.policy_digest,
            scheme=origin.scheme,
            requested_host=origin.host,
            tls_server_name=origin.host,
            port=origin.port,
            method=intent.method,
            path=intent.path,
            query_keys=tuple(pair.name for pair in intent.query_pairs),
            timeout_ms=intent.limits.timeout_ms,
            max_response_bytes=intent.limits.max_response_bytes,
            http_limits=HTTPHopLimits(),
            status_code=status_code,
            resolved_ips=("93.184.216.34",),
            connected_ip="93.184.216.34",
            response_header_bytes=64,
            wire_bytes=body_length,
            decoded_bytes=body_length,
            elapsed_ms=1,
        ),
        redirect_location=None,
        retry_after=retry_after,
    )


class _RecordingDispatch:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, object, object]] = []

    async def __call__(self, intent, *, cursor=None, bindings=()):
        self.calls.append((intent, cursor, bindings))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


async def _invoke_body(
    source_id: str,
    body: Any,
    *,
    result_limit: int = 1,
    monotonic_clock=None,
    status_code: Any = 200,
    content_type: str | None = "application/json",
    retry_after: Any = None,
):
    registry, plan = _plan_for((source_id,), result_limit=result_limit)
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch(
        [
            _response(
                route,
                group.intents[0],
                body,
                status_code=status_code,
                content_type=content_type,
                retry_after=retry_after,
            )
        ]
    )
    factory_kwargs = {} if monotonic_clock is None else {"monotonic_clock": monotonic_clock}
    adapter = _module().biorxiv_medrxiv_gateway_adapters(**factory_kwargs)[_module().EUROPE_PMC_ADAPTER_ID]
    result = await adapter(group, dispatch)
    return result, dispatch, group


def _normalized(candidate) -> dict[str, Any]:
    record = dict(candidate.record)
    record["provider_ids"] = dict(record["provider_ids"])
    return record


def _payload(source_id: str = "biorxiv") -> dict[str, Any]:
    return json.loads(_fixture(source_id))


def _payload_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _first_record(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["resultList"]["result"][0]


def _assert_typed_error(error: BaseException, code: str) -> None:
    assert type(error) is DiscoveryAdapterError
    assert error.code == code
    assert str(error) == code


class _CountingClock:
    def __init__(self, *, step: float = 0.0) -> None:
        self.value = 0.0
        self.step = step
        self.calls = 0

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        self.calls += 1
        return current


async def _execute_payload(source_id: str, payload: object):
    module = _module()
    registry, plan = _plan_for((source_id,))

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        return _response(route, intent, _payload_bytes(payload))

    return await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "europe-pmc-dispatch-1",
    )


async def _invoke_details_payloads(
    source_id: str,
    query: IdentifierLookupQuery | DateIntervalQuery,
    payloads: list[object],
    *,
    status_code: Any = 200,
    content_type: str | None = "application/json",
    retry_after: Any = None,
    monotonic_clock=None,
):
    module = _module()
    registry, plan = _details_plan_for(source_id, query)
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    route = registry.get_route(group.route_id)
    responses: list[object] = []
    for payload in payloads:
        if isinstance(payload, BaseException):
            responses.append(payload)
        else:
            body = payload if type(payload) is bytes else _payload_bytes(payload)
            responses.append(
                _response(
                    route,
                    intent,
                    body,
                    status_code=status_code,
                    content_type=content_type,
                    retry_after=retry_after,
                )
            )
    dispatch = _RecordingDispatch(responses)
    kwargs = {} if monotonic_clock is None else {"monotonic_clock": monotonic_clock}
    adapter = module.biorxiv_medrxiv_gateway_adapters(**kwargs)[module.DETAILS_ADAPTER_ID]
    result = await adapter(group, dispatch)
    return result, dispatch, group


def test_shadow_public_versions_and_adapter_identities_are_exact() -> None:
    module = _module()

    assert module.SHADOW_CATALOG_VERSION == "research-discovery-v2-biorxiv-medrxiv-shadow"
    assert module.SHADOW_REGISTRY_VERSION == "research-discovery-v2-biorxiv-medrxiv-shadow-2026-07-15"
    assert module.SHADOW_READINESS_VERSION == "research-discovery-readiness-v2-biorxiv-medrxiv-shadow"
    assert module.ROUTE_POLICY_VERSION == "research-discovery-route-policy-v2-biorxiv-medrxiv"
    assert module.EUROPE_PMC_ADAPTER_ID == "europe_pmc_preprint_v2"
    assert module.EUROPE_PMC_ADAPTER_VERSION == "europe-pmc-preprint-v2"
    assert module.DETAILS_ADAPTER_ID == "biorxiv_details_v2"
    assert module.DETAILS_ADAPTER_VERSION == "biorxiv-details-v2"


def test_shadow_registry_adds_exact_sources_without_mutating_foundation() -> None:
    module = _module()
    foundation_before = foundation_registry()
    foundation_snapshot = asdict(foundation_before)

    shadow = module.biorxiv_medrxiv_shadow_registry()

    assert asdict(foundation_registry()) == foundation_snapshot
    assert shadow.catalog_version == module.SHADOW_CATALOG_VERSION
    assert shadow.registry_version == module.SHADOW_REGISTRY_VERSION
    assert shadow.catalog_version != foundation_before.catalog_version
    assert shadow.registry_version != foundation_before.registry_version
    assert shadow.sources[: len(foundation_before.sources)] == tuple(
        replace(source, catalog_version=module.SHADOW_CATALOG_VERSION) for source in foundation_before.sources
    )
    assert shadow.routes[: len(foundation_before.routes)] == foundation_before.routes
    assert shadow.backends[: len(foundation_before.backends)] == foundation_before.backends
    assert tuple(source.catalog_version for source in shadow.sources) == (module.SHADOW_CATALOG_VERSION,) * 10

    expected = {
        "biorxiv": {
            "display_name": "bioRxiv",
            "aliases": ("bio_rxiv",),
            "site_hosts": ("biorxiv.org",),
            "priority": 90,
            "routes": (
                "biorxiv_europe_pmc_search_aggregator",
                "biorxiv_details_lookup_direct",
                "biorxiv_details_interval_direct",
            ),
        },
        "medrxiv": {
            "display_name": "medRxiv",
            "aliases": ("med_rxiv",),
            "site_hosts": ("medrxiv.org",),
            "priority": 100,
            "routes": (
                "medrxiv_europe_pmc_search_aggregator",
                "medrxiv_details_lookup_direct",
                "medrxiv_details_interval_direct",
            ),
        },
    }
    for source_id, values in expected.items():
        source = shadow.get_source(source_id)
        assert source.display_name == values["display_name"]
        assert source.aliases == values["aliases"]
        assert source.site_hosts == values["site_hosts"]
        assert source.priority == values["priority"]
        assert source.categories == ("preprints",)
        assert source.content_types == ("preprints", "papers", "abstracts")
        assert source.surfaces == ("standalone_search", "deep_research")
        assert tuple(reference.route_id for reference in source.route_references) == values["routes"]
        assert shadow.resolve_source_id(values["aliases"][0]) == source_id


def test_shadow_registry_adds_exact_backend_and_route_identities() -> None:
    module = _module()
    foundation = foundation_registry()
    shadow = module.biorxiv_medrxiv_shadow_registry()

    assert len(shadow.sources) == len(foundation.sources) + 2
    assert len(shadow.routes) == len(foundation.routes) + 6
    assert len(shadow.backends) == len(foundation.backends) + 2
    assert tuple(route.route_id for route in shadow.routes[-6:]) == _FAMILY_ROUTE_IDS
    assert tuple((backend.backend_id, backend.display_name) for backend in shadow.backends[-2:]) == (
        ("europe_pmc_rest_api", "Europe PMC REST API"),
        ("biorxiv_details_api", "bioRxiv/medRxiv Details API"),
    )
    assert min(source.priority for source in shadow.sources[-2:]) > max(
        source.priority for source in foundation.sources
    )


@pytest.mark.parametrize(
    ("source_id", "publisher", "route_id"),
    (
        ("biorxiv", "bioRxiv", "biorxiv_europe_pmc_search_aggregator"),
        ("medrxiv", "medRxiv", "medrxiv_europe_pmc_search_aggregator"),
    ),
)
def test_europe_pmc_routes_freeze_exact_policy_and_attribution(
    source_id: str,
    publisher: str,
    route_id: str,
) -> None:
    module = _module()
    shadow = module.biorxiv_medrxiv_shadow_registry()
    source = shadow.get_source(source_id)
    route = shadow.get_route(route_id)
    reference = next(reference for reference in source.route_references if reference.route_id == route_id)

    assert route.route_kind is RouteKind.AGGREGATOR
    assert route.query_modes == (QueryMode.GENERAL_FREE_TEXT,)
    assert route.backend_id == "europe_pmc_rest_api"
    assert route.adapter_id == module.EUROPE_PMC_ADAPTER_ID
    assert route.adapter_version == module.EUROPE_PMC_ADAPTER_VERSION
    assert route.source_constraint is SourceConstraint.PROVIDER_SOURCE_FILTER
    assert route.attribution_basis == "provider_publisher"
    assert route.credential_requirement is CredentialRequirement.NONE
    assert route.fallback_order == 0
    assert route.max_physical_dispatches == 1
    assert route.policy.policy_version == module.ROUTE_POLICY_VERSION
    assert route.policy.origin.scheme == "https"
    assert route.policy.origin.host == "www.ebi.ac.uk"
    assert route.policy.origin.port == 443
    assert route.policy.methods == ("GET",)
    assert route.policy.paths == ("/europepmc/webservices/rest/search",)
    assert route.policy.allowed_query_keys == ("query", "format", "resultType", "pageSize")
    assert route.policy.pagination_query_key is None
    assert route.policy.query_value_policies == (
        LiteralTermsQueryValuePolicy(
            "query",
            f' AND SRC:PPR AND PUBLISHER:"{publisher}"',
            16,
            64,
        ),
        ExactQueryValuePolicy("format", "json"),
        ExactQueryValuePolicy("resultType", "core"),
        BoundedDecimalQueryValuePolicy("pageSize", 100),
    )
    assert route.policy.limits.max_pages == 1
    assert route.policy.limits.max_redirects == 0
    assert route.policy.limits.max_retries == 0
    assert route.policy.limits.timeout_ms == 20_000
    assert route.policy.limits.max_response_bytes == 2_097_152
    assert route.policy.limits.max_results == 100
    assert reference.source_predicate == SourcePredicate(
        ("source_platform",),
        PredicateOperator.EQUALS_ANY,
        (source_id,),
        case_sensitive=False,
    )


@pytest.mark.parametrize("source_id", ("biorxiv", "medrxiv"))
def test_details_routes_are_exact_declared_direct_capabilities(source_id: str) -> None:
    module = _module()
    shadow = module.biorxiv_medrxiv_shadow_registry()
    source = shadow.get_source(source_id)
    lookup = shadow.get_route(f"{source_id}_details_lookup_direct")
    interval = shadow.get_route(f"{source_id}_details_interval_direct")
    source_references = {reference.route_id: reference for reference in source.route_references}

    for route in (lookup, interval):
        assert route.route_kind is RouteKind.DIRECT
        assert route.backend_id == "biorxiv_details_api"
        assert route.adapter_id == module.DETAILS_ADAPTER_ID
        assert route.adapter_version == module.DETAILS_ADAPTER_VERSION
        assert route.source_constraint is SourceConstraint.NATIVE_CORPUS
        assert route.attribution_basis == "native_response"
        assert route.credential_requirement is CredentialRequirement.NONE
        assert route.policy.policy_version == module.ROUTE_POLICY_VERSION
        assert route.policy.origin.scheme == "https"
        assert route.policy.origin.host == "api.biorxiv.org"
        assert route.policy.origin.port == 443
        assert route.policy.methods == ("GET",)
        assert source_references[route.route_id].source_predicate is None

    assert lookup.query_modes == (QueryMode.IDENTIFIER_LOOKUP,)
    assert lookup.fallback_order == 0
    assert lookup.max_physical_dispatches == 1
    assert lookup.policy.paths == ()
    assert lookup.policy.allowed_query_keys == ()
    assert lookup.policy.query_value_policies == ()
    assert lookup.policy.path_template is not None
    assert lookup.policy.path_template.segments == (
        "details",
        source_id,
        PathSlot(PathSlotKind.DOI_REGISTRANT, 12),
        PathSlot(PathSlotKind.DOI_SUFFIX, 128),
        "na",
        "json",
    )
    assert lookup.policy.path_template.pagination_segment_index is None
    assert (
        lookup.policy.limits.max_pages,
        lookup.policy.limits.max_redirects,
        lookup.policy.limits.max_retries,
        lookup.policy.limits.timeout_ms,
        lookup.policy.limits.max_response_bytes,
        lookup.policy.limits.max_results,
    ) == (1, 0, 0, 20_000, 2_097_152, 30)

    assert interval.query_modes == (QueryMode.DATE_INTERVAL, QueryMode.CATEGORY_BROWSE)
    assert interval.fallback_order == 0
    assert interval.max_physical_dispatches == 4
    assert interval.policy.paths == ()
    assert interval.policy.allowed_query_keys == ("category",)
    assert interval.policy.query_value_policies == (BoundedTextQueryValuePolicy("category", 128),)
    assert interval.policy.path_template is not None
    assert interval.policy.path_template.segments == (
        "details",
        source_id,
        PathSlot(PathSlotKind.DATE, 10),
        PathSlot(PathSlotKind.DATE, 10),
        PathSlot(PathSlotKind.UINT, 10),
        "json",
    )
    assert interval.policy.path_template.pagination_segment_index == 4
    assert (
        interval.policy.limits.max_pages,
        interval.policy.limits.max_redirects,
        interval.policy.limits.max_retries,
        interval.policy.limits.timeout_ms,
        interval.policy.limits.max_response_bytes,
        interval.policy.limits.max_results,
    ) == (4, 0, 0, 20_000, 2_097_152, 120)


@pytest.mark.parametrize("mode", tuple(ExecutionMode))
def test_shadow_readiness_preserves_foundation_and_enables_fixture_proven_details(mode: ExecutionMode) -> None:
    module = _module()
    foundation = foundation_readiness(mode)
    shadow = module.biorxiv_medrxiv_shadow_readiness(mode)
    by_route = {entry.route_id: entry for entry in shadow.routes}

    assert shadow.overlay_version == module.SHADOW_READINESS_VERSION
    assert shadow.overlay_version != foundation.overlay_version
    assert shadow.execution_mode is mode
    assert shadow.routes[: len(foundation.routes)] == foundation.routes
    assert set(by_route) == {route.route_id for route in module.biorxiv_medrxiv_shadow_registry().routes}
    for route_id in _GENERAL_ROUTE_IDS:
        assert by_route[route_id].state is ReadinessState.READY
        assert by_route[route_id].credential_status is CredentialStatus.NOT_REQUIRED
        assert by_route[route_id].reason == f"{mode.value}_ready"
    for route_id in _DETAILS_ROUTE_IDS:
        assert by_route[route_id].state is ReadinessState.READY
        assert by_route[route_id].credential_status is CredentialStatus.NOT_REQUIRED
        assert by_route[route_id].reason == f"{mode.value}_ready"


def test_family_adapter_factory_is_immutable_and_matches_ready_family_identity() -> None:
    module = _module()
    adapters = module.biorxiv_medrxiv_gateway_adapters()
    registry = module.biorxiv_medrxiv_shadow_registry()
    readiness = module.biorxiv_medrxiv_shadow_readiness(ExecutionMode.OFFLINE_FIXTURE)
    ready_family_adapter_ids = {
        route.adapter_id
        for route in registry.routes[-6:]
        if readiness.get(route.route_id).state is ReadinessState.READY
    }

    assert type(adapters) is MappingProxyType
    assert (
        set(adapters)
        == {
            module.EUROPE_PMC_ADAPTER_ID,
            module.DETAILS_ADAPTER_ID,
        }
        == ready_family_adapter_ids
    )
    assert callable(adapters[module.EUROPE_PMC_ADAPTER_ID])
    assert callable(adapters[module.DETAILS_ADAPTER_ID])
    with pytest.raises(TypeError):
        adapters["forged"] = adapters[module.EUROPE_PMC_ADAPTER_ID]


def test_shadow_composition_rejects_duplicate_registry_and_adapter_ids() -> None:
    module = _module()
    shadow = module.biorxiv_medrxiv_shadow_registry()

    for field in ("sources", "routes", "backends"):
        values = getattr(shadow, field)
        with pytest.raises(ValueError, match="duplicate_"):
            replace(shadow, **{field: values + (values[-1],)})

    adapter = module.biorxiv_medrxiv_gateway_adapters()[module.EUROPE_PMC_ADAPTER_ID]
    with pytest.raises(ValueError, match="duplicate_adapter_id:europe_pmc_preprint_v2"):
        module._compose_adapter_maps(
            {module.EUROPE_PMC_ADAPTER_ID: adapter},
            {module.EUROPE_PMC_ADAPTER_ID: adapter},
        )


def test_shadow_registry_values_are_frozen() -> None:
    shadow = _module().biorxiv_medrxiv_shadow_registry()

    with pytest.raises(FrozenInstanceError):
        shadow.get_source("biorxiv").priority = 1
    with pytest.raises(FrozenInstanceError):
        shadow.get_route(_GENERAL_ROUTE_IDS[0]).fallback_order = 99


def test_family_parse_profiles_are_local_exact_and_frozen() -> None:
    module = _module()
    profiles = module._FAMILY_PARSING_PROFILES

    assert type(profiles) is MappingProxyType
    assert set(profiles) == {
        (module.EUROPE_PMC_ADAPTER_ID, module.EUROPE_PMC_ADAPTER_VERSION),
        (module.DETAILS_ADAPTER_ID, module.DETAILS_ADAPTER_VERSION),
    }
    for profile in profiles.values():
        assert (
            profile.max_input_bytes,
            profile.max_records,
            profile.max_depth,
            profile.max_nodes,
            profile.max_string_chars,
            profile.max_numeric_token_chars,
            profile.parse_deadline_ms,
        ) == (2_097_152, 120, 16, 50_000, 65_536, 32, 500)
        with pytest.raises(FrozenInstanceError):
            profile.max_records = 121


@pytest.mark.parametrize(
    ("source_id", "publisher"),
    (("biorxiv", "bioRxiv"), ("medrxiv", "medRxiv")),
)
def test_general_query_compiles_exact_one_page_europe_pmc_intent(
    source_id: str,
    publisher: str,
) -> None:
    _registry, plan = _plan_for((source_id,), result_limit=7)
    group = plan.dispatch_groups[0]
    intent = group.intents[0]

    assert len(plan.dispatch_groups) == 1
    assert len(plan.skipped) == 2
    assert group.adapter_id == "europe_pmc_preprint_v2"
    assert group.adapter_version == "europe-pmc-preprint-v2"
    assert group.fallback_order == 0
    assert group.allowance.physical_dispatches == 1
    assert intent.operation_kind is OperationKind.SEARCH
    assert intent.method == "GET"
    assert intent.path == "/europepmc/webservices/rest/search"
    assert tuple((pair.name, pair.value) for pair in intent.query_pairs) == (
        (
            "query",
            f'"Bounded" AND "Discovery" AND SRC:PPR AND PUBLISHER:"{publisher}"',
        ),
        ("format", "json"),
        ("resultType", "core"),
        ("pageSize", "7"),
    )
    assert intent.json_body_pairs == ()
    assert intent.query_bindings == ()
    assert "cursorMark" not in {pair.name for pair in intent.query_pairs}


@pytest.mark.parametrize(
    ("source_id", "expected"),
    (
        (
            "biorxiv",
            {
                "title": "Bounded bioRxiv discovery",
                "authors": ("Ada Example", "Grace Sample"),
                "abstract": "A sanitized abstract & test. Linked text",
                "snippet": "A sanitized abstract & test. Linked text",
                "doi": "10.5555/biorxiv.synthetic.2026",
                "pmid": None,
                "pmcid": None,
                "arxiv_id": None,
                "url": "https://doi.org/10.5555/biorxiv.synthetic.2026",
                "pdf_url": None,
                "provider": "europe_pmc",
                "provider_ids": {
                    "europe_pmc_id": "PPR900001",
                    "doi": "10.5555/biorxiv.synthetic.2026",
                },
                "published_date": "2026-06-10",
                "publication_year": "2026",
                "ppr_id": "PPR900001",
                "source_platform": "biorxiv",
            },
        ),
        (
            "medrxiv",
            {
                "title": "Bounded medRxiv discovery",
                "authors": ("Lin Example",),
                "abstract": "Invented clinical fixture metadata.",
                "snippet": "Invented clinical fixture metadata.",
                "doi": "10.5555/medrxiv.synthetic.2026",
                "pmid": None,
                "pmcid": None,
                "arxiv_id": None,
                "url": "https://doi.org/10.5555/medrxiv.synthetic.2026",
                "pdf_url": None,
                "provider": "europe_pmc",
                "provider_ids": {
                    "europe_pmc_id": "PPR900002",
                    "doi": "10.5555/medrxiv.synthetic.2026",
                },
                "published_date": "2026-06-11",
                "publication_year": "2026",
                "ppr_id": "PPR900002",
                "source_platform": "medrxiv",
            },
        ),
    ),
)
@pytest.mark.asyncio
async def test_sanitized_success_fixture_normalizes_only_bounded_retained_metadata(
    source_id: str,
    expected: dict[str, Any],
) -> None:
    body = _fixture(source_id)
    assert b"example.invalid" in body
    assert b"nextCursorMark" in body
    assert b"unknownFixtureField" in body

    result, dispatch, group = await _invoke_body(source_id, body)

    assert type(result) is DiscoveryAdapterResult
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    record = _normalized(candidate)
    assert record == expected
    assert set(record) == _NORMALIZED_KEYS
    assert candidate.candidate_id == DiscoveryOutcomeIdentity.from_fingerprint(build_fingerprint(expected)).document_id
    assert len(dispatch.calls) == 1
    assert dispatch.calls == [(group.intents[0], None, ())]
    assert "example.invalid" not in repr(record)
    assert "ignored-script-content" not in repr(record)
    assert "must-not-survive" not in repr(record)

    reference = next(
        reference
        for reference in _module().biorxiv_medrxiv_shadow_registry().get_source(source_id).route_references
        if reference.route_id == group.route_id
    )
    assert reference.source_predicate is not None
    assert evaluate_source_predicate(reference.source_predicate, record) is AttributionMatch.MATCH


@pytest.mark.asyncio
async def test_valid_empty_fixture_returns_one_dispatch_and_no_candidates() -> None:
    body = (_FIXTURE_ROOT / "europe_pmc_empty.json").read_bytes()

    result, dispatch, group = await _invoke_body("biorxiv", body)

    assert result == DiscoveryAdapterResult(candidates=())
    assert dispatch.calls == [(group.intents[0], None, ())]


@pytest.mark.asyncio
async def test_missing_optional_doi_uses_inert_europe_pmc_identifier_link() -> None:
    payload = json.loads(_fixture("medrxiv"))
    payload["resultList"]["result"][0].pop("doi")

    result, _dispatch, _group = await _invoke_body(
        "medrxiv",
        json.dumps(payload).encode(),
    )

    record = _normalized(result.candidates[0])
    assert record["doi"] is None
    assert record["url"] == "https://europepmc.org/article/PPR/PPR900002"
    assert record["provider_ids"] == {"europe_pmc_id": "PPR900002"}


@pytest.mark.asyncio
async def test_executor_attributes_both_sources_and_accounts_exactly_one_page_each() -> None:
    module = _module()
    registry, plan = _plan_for(("medrxiv", "biorxiv"), result_limit=2)
    calls: list[tuple[str, object]] = []

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        source_id = route.route_id.split("_", 1)[0]
        calls.append((source_id, intent))
        return _response(route, intent, _fixture(source_id))

    dispatch_ids = iter(("europe-pmc-dispatch-1", "europe-pmc-dispatch-2"))
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
    )

    assert tuple(source_id for source_id, _intent in calls) == ("biorxiv", "medrxiv")
    assert tuple(candidate.catalog_source_ids for candidate in result.candidates) == (
        ("biorxiv",),
        ("medrxiv",),
    )
    assert tuple(outcome.catalog_source_id for outcome in result.logical_outcomes) == (
        "biorxiv",
        "medrxiv",
    )
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.SUCCEEDED,
        LogicalOutcomeState.SUCCEEDED,
    )
    assert result.usage.pages == 2
    assert result.usage.route_attempts == 2
    assert result.usage.accounting.created == 2
    assert result.usage.accounting.debited == 2
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )


def test_three_fixtures_are_synthetic_sanitized_and_live_shape_only() -> None:
    biorxiv = _payload("biorxiv")
    medrxiv = _payload("medrxiv")
    empty = json.loads((_FIXTURE_ROOT / "europe_pmc_empty.json").read_bytes())
    serialized = json.dumps((biorxiv, medrxiv, empty), sort_keys=True)

    assert tuple(payload["version"] for payload in (biorxiv, medrxiv, empty)) == ("6.9",) * 3
    assert tuple(_first_record(payload)["id"] for payload in (biorxiv, medrxiv)) == (
        "PPR900001",
        "PPR900002",
    )
    assert all(_first_record(payload)["doi"].startswith("10.5555/") for payload in (biorxiv, medrxiv))
    assert all(
        author["fullName"].split()[-1] in {"Example", "Sample"}
        for payload in (biorxiv, medrxiv)
        for author in _first_record(payload)["authorList"]["author"]
    )
    assert "example.invalid" in serialized
    assert empty["hitCount"] == 0
    assert empty["resultList"]["result"] == []
    assert not any(marker in serialized.casefold() for marker in ("password", "api_key", "authtoken"))


def test_result_limit_above_route_cap_compiles_page_size_one_hundred_without_cursor() -> None:
    _registry, plan = _plan_for(("biorxiv",), result_limit=150)
    intent = plan.dispatch_groups[0].intents[0]

    assert tuple((pair.name, pair.value) for pair in intent.query_pairs)[-1] == ("pageSize", "100")
    assert "cursorMark" not in {pair.name for pair in intent.query_pairs}
    assert plan.dispatch_groups[0].limits.max_results == 100


@pytest.mark.parametrize(
    "case",
    (
        "wrong_group_type",
        "unknown_adapter_version",
        "wrong_path",
        "missing_page_size",
        "changed_format",
        "changed_result_type",
        "changed_suffix",
        "noncanonical_page_size",
        "oversized_page_size",
        "cursor_mark",
        "changed_pages",
        "changed_redirects",
        "changed_retries",
        "changed_timeout",
        "changed_response_bytes",
        "changed_route_result_limit",
        "wrong_adapter_id",
        "wrong_route_id",
        "wrong_operation",
        "wrong_method",
        "json_body",
        "query_binding",
        "extra_query_key",
        "duplicate_query_key",
        "reordered_query_keys",
        "changed_physical_allowance",
    ),
)
@pytest.mark.asyncio
async def test_forged_group_or_route_contract_rejects_before_dispatch(case: str) -> None:
    module = _module()
    _registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    candidate: object = group
    if case == "wrong_group_type":
        candidate = object()
    elif case == "unknown_adapter_version":
        candidate = replace(group, adapter_version="unknown-v2")
    elif case == "wrong_adapter_id":
        candidate = replace(group, adapter_id="forged_europe_pmc_v2")
    elif case == "wrong_route_id":
        forged_route_id = "forged_europe_pmc_search_aggregator"
        candidate = replace(
            group,
            route_id=forged_route_id,
            intents=(replace(intent, route_id=forged_route_id),),
        )
    elif case == "wrong_operation":
        candidate = replace(
            group,
            intents=(replace(intent, operation_kind=OperationKind.CONDITIONAL_SUMMARY),),
        )
    elif case == "wrong_method":
        candidate = replace(group, intents=(replace(intent, method="POST"),))
    elif case == "json_body":
        candidate = replace(
            group,
            intents=(replace(intent, json_body_pairs=(JSONBodyPair("forged", "value"),)),),
        )
    elif case == "query_binding":
        candidate = replace(
            group,
            intents=(
                replace(
                    intent,
                    query_bindings=(DeferredNumericCSVQueryBinding("forged_binding", "ids", 1, 16),),
                ),
            ),
        )
    elif case in {"extra_query_key", "duplicate_query_key", "reordered_query_keys"}:
        pairs = {
            "extra_query_key": intent.query_pairs + (QueryPair("extra", "1"),),
            "duplicate_query_key": intent.query_pairs + (QueryPair("pageSize", "1"),),
            "reordered_query_keys": tuple(reversed(intent.query_pairs)),
        }[case]
        candidate = replace(group, intents=(replace(intent, query_pairs=pairs),))
    elif case == "changed_physical_allowance":
        candidate = replace(
            group,
            allowance=replace(group.allowance, physical_dispatches=2),
        )
    elif case == "wrong_path":
        candidate = replace(group, intents=(replace(intent, path="/wrong"),))
    elif case == "missing_page_size":
        candidate = replace(group, intents=(replace(intent, query_pairs=intent.query_pairs[:-1]),))
    elif case in {"changed_format", "changed_result_type", "changed_suffix"}:
        pairs = list(intent.query_pairs)
        index, value = {
            "changed_format": (1, "xml"),
            "changed_result_type": (2, "lite"),
            "changed_suffix": (0, '"Bounded" AND "Discovery" AND SRC:PPR'),
        }[case]
        pairs[index] = replace(pairs[index], value=value)
        candidate = replace(group, intents=(replace(intent, query_pairs=tuple(pairs)),))
    elif case in {"noncanonical_page_size", "oversized_page_size"}:
        pairs = list(intent.query_pairs)
        pairs[3] = replace(pairs[3], value="01" if case == "noncanonical_page_size" else "101")
        candidate = replace(group, intents=(replace(intent, query_pairs=tuple(pairs)),))
    elif case == "cursor_mark":
        candidate = replace(
            group,
            intents=(replace(intent, query_pairs=intent.query_pairs + (QueryPair("cursorMark", "*"),)),),
        )
    else:
        limit_field, value = {
            "changed_pages": ("max_pages", 2),
            "changed_redirects": ("max_redirects", 1),
            "changed_retries": ("max_retries", 1),
            "changed_timeout": ("timeout_ms", 20_001),
            "changed_response_bytes": ("max_response_bytes", 2_097_151),
            "changed_route_result_limit": ("max_results", 99),
        }[case]
        limits = replace(group.limits, **{limit_field: value})
        physical = 1 + max(limits.max_pages - 1, 0) + limits.max_redirects + limits.max_retries
        allowance = replace(
            group.allowance,
            physical_dispatches=physical,
            pages=limits.max_pages,
            redirects=limits.max_redirects,
            retries=limits.max_retries,
        )
        candidate = replace(
            group,
            limits=limits,
            intents=(replace(intent, limits=limits),),
            allowance=allowance,
        )
    dispatch = _RecordingDispatch([])
    adapter = module.biorxiv_medrxiv_gateway_adapters()[module.EUROPE_PMC_ADAPTER_ID]

    with pytest.raises(Exception) as caught:
        await adapter(candidate, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert dispatch.calls == []


@pytest.mark.parametrize(
    "body",
    (
        b"",
        b"{",
        b"\xff",
        b'{"hitCount":0,"hitCount":0,"resultList":{"result":[]}}',
    ),
)
@pytest.mark.asyncio
async def test_malformed_utf8_json_or_duplicate_keys_are_rejected(body: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", body)

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"hitCount": 0},
        {"hitCount": 0, "resultList": []},
        {"hitCount": 0, "resultList": {}},
        {"hitCount": 0, "resultList": {"result": {}}},
        {"hitCount": True, "resultList": {"result": []}},
        {"hitCount": -1, "resultList": {"result": []}},
    ),
)
@pytest.mark.asyncio
async def test_malformed_envelope_shapes_and_hit_counts_are_rejected(payload: object) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_hit_count_smaller_than_results_is_rejected() -> None:
    payload = _payload()
    payload["hitCount"] = 0

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_positive_hit_count_with_empty_first_page_is_rejected() -> None:
    payload = json.loads((_FIXTURE_ROOT / "europe_pmc_empty.json").read_bytes())
    payload["hitCount"] = 1

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_result_count_above_requested_page_size_is_rejected_atomically() -> None:
    payload = _payload()
    payload["hitCount"] = 2
    payload["resultList"]["result"].append(dict(_first_record(payload)))

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload), result_limit=1)

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(
    ("body", "code"),
    (
        (_payload_bytes({"x": "x" * 65_537}), "provider_parse_limit_exceeded"),
        (b'{"hitCount":' + (b"1" * 33) + b',"resultList":{"result":[]}}', "provider_parse_limit_exceeded"),
        (b"x" * 2_097_153, "provider_parse_limit_exceeded"),
    ),
)
@pytest.mark.asyncio
async def test_string_numeric_and_input_byte_parse_limits_are_enforced(body: bytes, code: str) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", body)

    _assert_typed_error(caught.value, code)


@pytest.mark.asyncio
async def test_json_depth_and_node_limits_are_enforced() -> None:
    nested: object = None
    for _index in range(17):
        nested = {"x": nested}
    bodies = (
        _payload_bytes({"x": nested}),
        _payload_bytes({"x": [None] * 50_001}),
    )

    for body in bodies:
        with pytest.raises(Exception) as caught:
            await _invoke_body("biorxiv", body)
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_cooperative_parse_deadline_is_enforced() -> None:
    clock = _CountingClock(step=0.2)

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _fixture("biorxiv"), monotonic_clock=clock)

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")
    assert clock.calls >= 4


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("title", "x" * 4_097),
        ("abstractText", "x" * 65_537),
    ),
    ids=("title_over_limit", "abstract_over_limit"),
)
@pytest.mark.asyncio
async def test_title_and_abstract_bounds_are_enforced(field: str, value: str) -> None:
    payload = _payload()
    _first_record(payload)[field] = value

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize("case", ("too_many", "too_long"))
@pytest.mark.asyncio
async def test_author_count_and_text_bounds_are_enforced(case: str) -> None:
    payload = _payload()
    record = _first_record(payload)
    if case == "too_many":
        record["authorList"] = {"author": [{"fullName": "A Example"}] * 1_025}
    else:
        record["authorList"] = {"author": [{"fullName": "x" * 513}]}

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(
    "author_list",
    (
        [],
        {},
        {"author": {}},
        {"author": [None]},
        {"author": [{}]},
        {"author": [{"fullName": None}]},
        {"author": [{"fullName": "   "}]},
    ),
)
@pytest.mark.asyncio
async def test_malformed_author_shapes_are_rejected(author_list: object) -> None:
    payload = _payload()
    _first_record(payload)["authorList"] = author_list

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("field", ("title", "abstractText", "author"))
@pytest.mark.parametrize("control", ("\x00", "\x1f", "\x7f"))
@pytest.mark.asyncio
async def test_retained_text_rejects_ascii_controls(field: str, control: str) -> None:
    payload = _payload()
    record = _first_record(payload)
    if field == "author":
        record["authorList"]["author"][0]["fullName"] = f"Ada{control}Example"
    else:
        record[field] = f"safe{control}text"

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_retained_text_normalizes_safe_ascii_whitespace() -> None:
    payload = _payload()
    record = _first_record(payload)
    record["title"] = "Safe\t title\nline\rtext"
    record["abstractText"] = "Safe\n abstract\ttext"
    record["authorList"] = {"author": [{"fullName": "Ada\r\n Example"}]}

    result, _dispatch, _group = await _invoke_body("biorxiv", _payload_bytes(payload))
    normalized = result.candidates[0].record

    assert normalized["title"] == "Safe title line text"
    assert normalized["abstract"] == "Safe abstract text"
    assert normalized["authors"] == ("Ada Example",)


@pytest.mark.parametrize(
    "ppr_id",
    (
        "PPR0",
        "PPR01",
        "ppr1",
        "PPR1x",
        "PPR" + ("1" * 126),
        " PPR1",
    ),
)
@pytest.mark.asyncio
async def test_noncanonical_or_oversized_ppr_ids_are_rejected(ppr_id: str) -> None:
    payload = _payload()
    _first_record(payload)["id"] = ppr_id

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "doi",
    (
        '10.5555/bad"suffix',
        "10.5555/bad<suffix",
        "10.5555/bad>suffix",
        "10.5555/bad[suffix",
        "10.5555/bad]suffix",
        "10.5555/bad\\suffix",
        "10.5555/bad^suffix",
        "10.5555/bad`suffix",
        "10.5555/bad{suffix",
        "10.5555/bad|suffix",
        "10.5555/bad}suffix",
        "10.5555/bad%20suffix",
        "10.5555/bad?suffix",
        "10.5555/bad#suffix",
        "10.5555/bad/extra",
        "https://doi.org/10.5555/prefixed",
    ),
)
@pytest.mark.asyncio
async def test_doi_accepts_only_unescaped_single_segment_rfc3986_pchar(doi: str) -> None:
    payload = _payload()
    _first_record(payload)["doi"] = doi

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    ("first_publication_date", "publication_year"),
    (
        ("2026-02-30", "2026"),
        ("2026-06-10", "2025"),
        (None, "0000"),
        (None, "26"),
    ),
)
@pytest.mark.asyncio
async def test_invalid_dates_and_years_are_rejected(
    first_publication_date: str | None,
    publication_year: str,
) -> None:
    payload = _payload()
    record = _first_record(payload)
    if first_publication_date is None:
        record.pop("firstPublicationDate", None)
    else:
        record["firstPublicationDate"] = first_publication_date
    record["pubYear"] = publication_year

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload))

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_valid_date_derives_missing_publication_year() -> None:
    payload = _payload()
    _first_record(payload).pop("pubYear")

    result, _dispatch, _group = await _invoke_body("biorxiv", _payload_bytes(payload))

    assert result.candidates[0].record["published_date"] == "2026-06-10"
    assert result.candidates[0].record["publication_year"] == "2026"


@pytest.mark.parametrize(
    ("source", "publisher", "expected_platform", "expected_match"),
    (
        (None, "bioRxiv", None, AttributionMatch.AMBIGUOUS),
        ("ppr", "bioRxiv", None, AttributionMatch.AMBIGUOUS),
        ("PMC", "bioRxiv", None, AttributionMatch.AMBIGUOUS),
        ("PPR", None, None, AttributionMatch.AMBIGUOUS),
        ("PPR", "Unknown Preprints", None, AttributionMatch.AMBIGUOUS),
        ("PPR", "medRxiv", "medrxiv", AttributionMatch.NON_MATCH),
        ("PPR", "  BIOrXiv  ", "biorxiv", AttributionMatch.MATCH),
    ),
)
@pytest.mark.asyncio
async def test_provider_attribution_is_exact_and_fail_closed(
    source: str | None,
    publisher: str | None,
    expected_platform: str | None,
    expected_match: AttributionMatch,
) -> None:
    payload = _payload()
    record = _first_record(payload)
    if source is None:
        record.pop("source", None)
    else:
        record["source"] = source
    if publisher is None:
        record.pop("bookOrReportDetails", None)
    else:
        record["bookOrReportDetails"] = {"publisher": publisher}

    result, _dispatch, group = await _invoke_body("biorxiv", _payload_bytes(payload))
    normalized = dict(result.candidates[0].record)
    reference = next(
        reference
        for reference in _module().biorxiv_medrxiv_shadow_registry().get_source("biorxiv").route_references
        if reference.route_id == group.route_id
    )

    assert normalized.get("source_platform") == expected_platform
    assert reference.source_predicate is not None
    assert evaluate_source_predicate(reference.source_predicate, normalized) is expected_match


@pytest.mark.parametrize(
    ("source", "publisher"),
    (
        (None, "bioRxiv"),
        ("PPR", "Unknown Preprints"),
        ("PPR", "medRxiv"),
    ),
)
@pytest.mark.asyncio
async def test_ambiguous_or_nonmatching_attribution_projects_valid_empty(
    source: str | None,
    publisher: str,
) -> None:
    payload = _payload()
    record = _first_record(payload)
    if source is None:
        record.pop("source", None)
    else:
        record["source"] = source
    record["bookOrReportDetails"] = {"publisher": publisher}

    execution = await _execute_payload("biorxiv", payload)

    assert execution.candidates == ()
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.VALID_EMPTY
    assert execution.logical_outcomes[0].code is None
    assert execution.usage.pages == 1
    assert execution.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED


@pytest.mark.asyncio
async def test_exact_duplicate_records_collapse_to_one_candidate() -> None:
    payload = _payload()
    payload["hitCount"] = 2
    payload["resultList"]["result"].append(dict(_first_record(payload)))

    result, _dispatch, _group = await _invoke_body(
        "biorxiv",
        _payload_bytes(payload),
        result_limit=2,
    )

    assert len(result.candidates) == 1


@pytest.mark.parametrize("case", ("same_doi", "same_ppr", "changed_title"))
@pytest.mark.asyncio
async def test_conflicting_duplicate_identities_fail_atomically(case: str) -> None:
    payload = _payload()
    original = _first_record(payload)
    conflicting = json.loads(json.dumps(original))
    if case == "same_doi":
        conflicting["id"] = "PPR900099"
    elif case == "same_ppr":
        conflicting["doi"] = "10.5555/different.synthetic.2026"
    else:
        conflicting["title"] = "Conflicting title"
    payload["hitCount"] = 2
    payload["resultList"]["result"].append(conflicting)

    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _payload_bytes(payload), result_limit=2)

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_provider_continuation_and_fulltext_urls_are_inert_and_never_dispatched() -> None:
    payload = _payload()
    payload["nextCursorMark"] = "https://attacker.invalid/cursor?token=fixture-secret"
    payload["nextPageUrl"] = "https://attacker.invalid/next?token=fixture-secret"
    record = _first_record(payload)
    record["fullTextUrlList"] = {"fullTextUrl": [{"url": "https://attacker.invalid/file?token=fixture-secret"}]}
    registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch(
        [
            _response(route, group.intents[0], _payload_bytes(payload)),
            AssertionError("provider continuation was dispatched"),
        ]
    )
    adapter = _module().biorxiv_medrxiv_gateway_adapters()[_module().EUROPE_PMC_ADAPTER_ID]

    result = await adapter(group, dispatch)

    assert len(result.candidates) == 1
    assert dispatch.calls == [(group.intents[0], None, ())]
    assert "attacker.invalid" not in repr(result)
    assert "fixture-secret" not in repr(result)


@pytest.mark.parametrize("status_code", (201, 204, 400, 503, True, 200.0, "200", 429.0))
@pytest.mark.asyncio
async def test_non_200_status_rejects_before_mime_body_or_clock(status_code: object) -> None:
    def forbidden_clock() -> float:
        raise AssertionError("parse clock must not run for rejected status")

    with pytest.raises(Exception) as caught:
        await _invoke_body(
            "biorxiv",
            b"fixture-secret-not-json",
            status_code=status_code,
            content_type=None,
            retry_after="120",
            monotonic_clock=forbidden_clock,
        )

    _assert_typed_error(caught.value, "provider_response_rejected")
    assert caught.value.retry_after is None
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.parametrize(
    "content_type",
    (None, "text/plain", "text/html", "application/xml", "application/json; charset"),
)
@pytest.mark.asyncio
async def test_non_json_or_malformed_mime_is_rejected(content_type: str | None) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_body("biorxiv", _fixture("biorxiv"), content_type=content_type)

    _assert_typed_error(caught.value, "provider_response_rejected")


@pytest.mark.parametrize(
    "retry_after",
    ("0", "001", "120", "Wed, 21 Oct 2015 07:28:00 GMT"),
)
@pytest.mark.asyncio
async def test_429_preserves_only_valid_retry_after_without_parsing_body(retry_after: str) -> None:
    def forbidden_clock() -> float:
        raise AssertionError("parse clock must not run for 429")

    with pytest.raises(Exception) as caught:
        await _invoke_body(
            "biorxiv",
            b"fixture-secret-not-json",
            status_code=429,
            content_type=None,
            retry_after=retry_after,
            monotonic_clock=forbidden_clock,
        )

    _assert_typed_error(caught.value, "provider_rate_limited")
    assert caught.value.retry_after == retry_after
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.parametrize(
    "retry_after",
    ("-1", "+1", "1.5", "tomorrow", "120\nfixture-secret", " 120"),
)
@pytest.mark.asyncio
async def test_429_drops_invalid_retry_after(retry_after: str) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_body(
            "biorxiv",
            b"fixture-secret-not-json",
            status_code=429,
            content_type=None,
            retry_after=retry_after,
        )

    _assert_typed_error(caught.value, "provider_rate_limited")
    assert caught.value.retry_after is None
    assert retry_after not in repr(caught.value)


@pytest.mark.asyncio
async def test_adapter_propagates_gateway_timeout_object_identity() -> None:
    _registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    timed_out = DiscoveryExecutionError("gateway_timed_out")
    dispatch = _RecordingDispatch([timed_out])
    adapter = _module().biorxiv_medrxiv_gateway_adapters()[_module().EUROPE_PMC_ADAPTER_ID]

    with pytest.raises(DiscoveryExecutionError) as caught:
        await adapter(group, dispatch)

    assert caught.value is timed_out


@pytest.mark.asyncio
async def test_adapter_propagates_cancellation_object_identity() -> None:
    _registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    cancelled = asyncio.CancelledError("fixture-cancelled")
    dispatch = _RecordingDispatch([cancelled])
    adapter = _module().biorxiv_medrxiv_gateway_adapters()[_module().EUROPE_PMC_ADAPTER_ID]

    with pytest.raises(asyncio.CancelledError) as caught:
        await adapter(group, dispatch)

    assert caught.value is cancelled


@pytest.mark.parametrize("failing_source", ("biorxiv", "medrxiv"))
@pytest.mark.asyncio
async def test_one_source_parse_failure_preserves_other_source_and_exact_accounting(
    failing_source: str,
) -> None:
    module = _module()
    registry, plan = _plan_for(("medrxiv", "biorxiv"), result_limit=2)
    gateway_route_ids: list[str] = []

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        gateway_route_ids.append(route.route_id)
        source_id = route.route_id.split("_", 1)[0]
        body = b"{" if source_id == failing_source else _fixture(source_id)
        return _response(route, intent, body)

    dispatch_ids = iter(("europe-pmc-partial-1", "europe-pmc-partial-2"))
    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
    )
    successful_source = "medrxiv" if failing_source == "biorxiv" else "biorxiv"
    outcomes = {outcome.catalog_source_id: outcome for outcome in execution.logical_outcomes}

    assert gateway_route_ids == list(_GENERAL_ROUTE_IDS)
    assert len(execution.candidates) == 1
    assert execution.candidates[0].catalog_source_ids == (successful_source,)
    assert outcomes[failing_source].state is LogicalOutcomeState.FAILED
    assert outcomes[failing_source].code == "provider_payload_invalid"
    assert outcomes[successful_source].state is LogicalOutcomeState.SUCCEEDED
    assert outcomes[successful_source].code is None
    assert execution.usage.pages == 2
    assert execution.usage.route_attempts == 2
    assert execution.usage.accounting.created == 2
    assert execution.usage.accounting.debited == 2
    assert execution.usage.accounting.released == 0
    assert execution.usage.accounting.outstanding == 0
    assert tuple(record.dispatch_id for record in execution.usage.physical_records) == (
        "europe-pmc-partial-1",
        "europe-pmc-partial-2",
    )
    assert tuple(record.state for record in execution.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )


def test_five_details_fixtures_are_synthetic_sanitized_and_distinct_valid_shapes() -> None:
    names = (
        "biorxiv_details_doi_success",
        "medrxiv_details_doi_success",
        "biorxiv_details_interval_page_1",
        "biorxiv_details_interval_page_2",
        "biorxiv_details_interval_empty",
    )
    payloads = tuple(json.loads(_details_fixture(name)) for name in names)
    serialized = json.dumps(payloads, sort_keys=True)

    assert tuple(len(payload["messages"]) for payload in payloads) == (1,) * 5
    assert tuple(payload["messages"][0]["status"] for payload in payloads) == (
        "ok",
        "ok",
        "ok",
        "ok",
        "no posts found",
    )
    assert tuple(len(payload["collection"]) for payload in payloads) == (1, 2, 1, 1, 0)
    assert all(item["doi"].startswith("10.5555/") for payload in payloads for item in payload["collection"])
    assert "example.invalid" in serialized
    assert not any(marker in serialized.casefold() for marker in ("password", "api_key", "authtoken"))


@pytest.mark.parametrize(
    ("source_id", "doi", "fixture_name", "expected_title", "expected_version", "published_doi"),
    (
        (
            "biorxiv",
            "10.5555/biorxiv.details.synthetic",
            "biorxiv_details_doi_success",
            "Bounded bioRxiv details lookup",
            1,
            None,
        ),
        (
            "medrxiv",
            "10.5555/medrxiv.details.synthetic",
            "medrxiv_details_doi_success",
            "Bounded medRxiv details lookup version two",
            2,
            "10.5555/published.medrxiv.synthetic",
        ),
    ),
)
@pytest.mark.asyncio
async def test_details_doi_success_binds_exact_path_and_keeps_highest_version(
    source_id: str,
    doi: str,
    fixture_name: str,
    expected_title: str,
    expected_version: int,
    published_doi: str | None,
) -> None:
    module = _module()
    registry, plan = _details_plan_for(source_id, IdentifierLookupQuery(doi), result_limit=30)
    assert len(plan.dispatch_groups) == 1
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch([_response(route, intent, _details_fixture(fixture_name))])

    result = await module.biorxiv_medrxiv_gateway_adapters()[module.DETAILS_ADAPTER_ID](group, dispatch)

    assert group.route_id == f"{source_id}_details_lookup_direct"
    registrant, suffix = doi.split("/", 1)
    assert intent.path == f"/details/{source_id}/{registrant}/{suffix}/na/json"
    assert intent.query_pairs == ()
    assert dispatch.calls == [(intent, None, ())]
    assert len(result.candidates) == 1
    record = _normalized(result.candidates[0])
    assert set(record) == _DETAILS_NORMALIZED_KEYS
    assert record["title"] == expected_title
    assert record["doi"] == doi
    assert record["url"] == f"https://doi.org/{doi}"
    assert record["pdf_url"] is None
    assert record["provider"] == "biorxiv_details"
    assert record["provider_ids"] == {"doi": doi, "version": str(expected_version)}
    assert record["version"] == expected_version
    assert record["published_doi"] == published_doi
    assert record["source_platform"] == source_id
    assert "example.invalid" not in repr(record)
    assert "fixture-secret" not in repr(record)


@pytest.mark.parametrize(
    ("doi", "encoded_suffix"),
    (
        ("10.5555/a[b", "a%5Bb"),
        (f"10.5555/{'x' * 128}", "x" * 128),
    ),
    ids=("visible_ascii_suffix", "maximum_suffix"),
)
@pytest.mark.asyncio
async def test_details_doi_planner_boundaries_execute_bind_and_encode_landing_url(
    doi: str,
    encoded_suffix: str,
) -> None:
    module = _module()
    payload = json.loads(_details_fixture("biorxiv_details_doi_success"))
    payload["collection"][0]["doi"] = doi.upper()
    registry, plan = _details_plan_for("biorxiv", IdentifierLookupQuery(doi), result_limit=30)
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    expected_path = f"/details/biorxiv/10.5555/{encoded_suffix}/na/json"
    paths: list[str] = []

    async def gateway(route, dispatched_intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        paths.append(dispatched_intent.path)
        return _response(route, dispatched_intent, _payload_bytes(payload))

    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "details-doi-boundary-1",
    )

    assert intent.path == expected_path
    assert paths == [expected_path]
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert len(execution.candidates) == 1
    record = execution.candidates[0].record
    assert record["doi"] == doi
    assert record["provider_ids"] == {"doi": doi, "version": "1"}
    assert record["url"] == f"https://doi.org/10.5555/{encoded_suffix}"


@pytest.mark.parametrize("doi", ("10.5555/a", "10.5555/a."))
@pytest.mark.asyncio
async def test_details_exact_lookup_identity_preserves_terminal_punctuation(doi: str) -> None:
    payload = json.loads(_details_fixture("biorxiv_details_doi_success"))
    payload["collection"][0]["doi"] = doi

    result, _dispatch, _group = await _invoke_details_payloads(
        "biorxiv",
        IdentifierLookupQuery(doi),
        [payload],
    )

    candidate = result.candidates[0]
    assert candidate.record["doi"] == doi
    assert candidate.candidate_id == DiscoveryOutcomeIdentity.from_fingerprint(f"doi:{doi}").document_id


@pytest.mark.asyncio
async def test_details_publication_linkage_accepts_bounded_multi_segment_doi() -> None:
    payload = json.loads(_details_fixture("biorxiv_details_doi_success"))
    payload["collection"][0]["published"] = "10.5555/published/extra"

    result, _dispatch, _group = await _invoke_details_payloads(
        "biorxiv",
        IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
        [payload],
    )

    record = result.candidates[0].record
    assert record["published_doi"] == "10.5555/published/extra"
    assert record["url"] == "https://doi.org/10.5555/biorxiv.details.synthetic"


@pytest.mark.asyncio
async def test_details_interval_preserves_distinct_exact_doi_identities_through_executor() -> None:
    module = _module()
    query = DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience")
    payload = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    first = payload["collection"][0]
    first["doi"] = "10.5555/a"
    second = dict(first)
    second["doi"] = "10.5555/a."
    payload["collection"].append(second)
    payload["messages"][0].update({"count": 2, "total": "2"})
    expected_dois = ("10.5555/a", "10.5555/a.")
    expected_ids = tuple(DiscoveryOutcomeIdentity.from_fingerprint(f"doi:{doi}").document_id for doi in expected_dois)

    adapter_result, adapter_dispatch, adapter_group = await _invoke_details_payloads(
        "biorxiv",
        query,
        [payload],
    )

    assert tuple(candidate.record["doi"] for candidate in adapter_result.candidates) == expected_dois
    assert tuple(candidate.candidate_id for candidate in adapter_result.candidates) == expected_ids
    assert len(set(expected_ids)) == 2
    assert len(adapter_dispatch.calls) == 1

    registry, plan = _details_plan_for("biorxiv", query)
    paths: list[str] = []

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        paths.append(intent.path)
        return _response(route, intent, _payload_bytes(payload))

    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "details-exact-identity-1",
    )

    assert paths == [adapter_group.intents[0].path]
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert tuple(candidate.record["doi"] for candidate in execution.candidates) == expected_dois
    assert tuple(candidate.candidate_id for candidate in execution.candidates) == expected_ids


@pytest.mark.asyncio
async def test_details_interval_category_uses_response_derived_path_cursor() -> None:
    module = _module()
    query = DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience")
    registry, plan = _details_plan_for("biorxiv", query)
    assert len(plan.dispatch_groups) == 1
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch(
        [
            _response(route, intent, _details_fixture("biorxiv_details_interval_page_1")),
            _response(route, intent, _details_fixture("biorxiv_details_interval_page_2")),
        ]
    )

    result = await module.biorxiv_medrxiv_gateway_adapters()[module.DETAILS_ADAPTER_ID](group, dispatch)

    assert group.route_id == "biorxiv_details_interval_direct"
    assert intent.path == "/details/biorxiv/2026-06-01/2026-06-02/0/json"
    assert tuple((pair.name, pair.value) for pair in intent.query_pairs) == (("category", "neuroscience"),)
    assert dispatch.calls == [
        (intent, None, ()),
        (intent, NumericCursor(1), ()),
    ]
    assert tuple(candidate.record["doi"] for candidate in result.candidates) == (
        "10.5555/biorxiv.interval.one",
        "10.5555/biorxiv.interval.two",
    )
    assert tuple(candidate.record["version"] for candidate in result.candidates) == (1, 2)
    assert all(set(candidate.record) == _DETAILS_NORMALIZED_KEYS for candidate in result.candidates)
    assert "example.invalid" not in repr(result)
    assert "fixture-secret" not in repr(result)


@pytest.mark.asyncio
async def test_details_interval_valid_empty_is_one_page_without_continuation() -> None:
    module = _module()
    registry, plan = _details_plan_for(
        "biorxiv",
        DateIntervalQuery("2026-06-03", "2026-06-03"),
    )
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch([_response(route, intent, _details_fixture("biorxiv_details_interval_empty"))])

    result = await module.biorxiv_medrxiv_gateway_adapters()[module.DETAILS_ADAPTER_ID](group, dispatch)

    assert result == DiscoveryAdapterResult(candidates=())
    assert dispatch.calls == [(intent, None, ())]


@pytest.mark.parametrize(
    "case",
    (
        "extra_root_field",
        "extra_doi_message_field",
        "ok_with_empty_collection",
    ),
)
@pytest.mark.asyncio
async def test_details_exact_envelope_rejects_ambiguous_success_shapes(case: str) -> None:
    module = _module()
    payload = json.loads(_details_fixture("biorxiv_details_doi_success"))
    if case == "extra_root_field":
        payload["provider_echo"] = "must-not-be-trusted"
    elif case == "extra_doi_message_field":
        payload["messages"][0]["count"] = 1
    else:
        payload["collection"] = []
    registry, plan = _details_plan_for(
        "biorxiv",
        IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
        result_limit=30,
    )
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch([_response(route, intent, _payload_bytes(payload))])

    with pytest.raises(Exception) as caught:
        await module.biorxiv_medrxiv_gateway_adapters()[module.DETAILS_ADAPTER_ID](group, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert dispatch.calls == [(intent, None, ())]


@pytest.mark.asyncio
async def test_details_interval_drops_irrelevant_message_metadata() -> None:
    module = _module()
    payloads = [
        json.loads(_details_fixture("biorxiv_details_interval_page_1")),
        json.loads(_details_fixture("biorxiv_details_interval_page_2")),
    ]
    for payload in payloads:
        payload["messages"][0].pop("funder")
        payload["messages"][0].pop("count_new_papers")
        payload["messages"][0]["provider_cursor_url"] = "https://attacker.invalid/next"
    registry, plan = _details_plan_for(
        "biorxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
    )
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch([_response(route, intent, _payload_bytes(payload)) for payload in payloads])

    result = await module.biorxiv_medrxiv_gateway_adapters()[module.DETAILS_ADAPTER_ID](group, dispatch)

    assert len(result.candidates) == 2
    assert "attacker.invalid" not in repr(result)
    assert dispatch.calls == [(intent, None, ()), (intent, NumericCursor(1), ())]


@pytest.mark.asyncio
async def test_details_oversized_retained_category_is_a_parse_limit_failure() -> None:
    payload = json.loads(_details_fixture("biorxiv_details_doi_success"))
    payload["collection"][0]["category"] = "x" * 129

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            [payload],
        )

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(
    "category",
    ("ẞ" * 65, "ﬃ" * 43),
    ids=("casefold_expansion", "nfkc_expansion"),
)
@pytest.mark.asyncio
async def test_details_normalized_category_expansion_over_limit_is_parse_limit(
    category: str,
) -> None:
    payload = json.loads(_details_fixture("biorxiv_details_doi_success"))
    payload["collection"][0]["category"] = category

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            [payload],
        )

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_details_planner_approved_dotted_i_category_matches_after_casefold() -> None:
    payload = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    payload["messages"][0].update({"category": "İ", "total": 1})
    payload["collection"][0]["category"] = "İ"

    result, _dispatch, _group = await _invoke_details_payloads(
        "biorxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "İ"),
        [payload],
    )

    assert result.candidates[0].record["category"] == "i\N{COMBINING DOT ABOVE}"


@pytest.mark.parametrize(
    "case",
    (
        "application_error",
        "no_posts_with_collection",
        "missing_messages",
        "messages_not_list",
        "multiple_messages",
        "collection_not_list",
        "malformed_json",
    ),
)
@pytest.mark.asyncio
async def test_details_malformed_or_application_error_envelopes_fail_closed(case: str) -> None:
    payload: object = json.loads(_details_fixture("biorxiv_details_doi_success"))
    if case == "application_error":
        payload["messages"] = [{"status": "provider unavailable"}]
    elif case == "no_posts_with_collection":
        payload["messages"] = [{"status": "no posts found"}]
    elif case == "missing_messages":
        payload.pop("messages")
    elif case == "messages_not_list":
        payload["messages"] = {}
    elif case == "multiple_messages":
        payload["messages"].append({"status": "ok", "category": "all"})
    elif case == "collection_not_list":
        payload["collection"] = {}
    else:
        payload = b"{"

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            [payload],
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    ("case", "value"),
    (
        ("doi", "10.5555/different.details.synthetic"),
        ("server", "medRxiv"),
        ("server", "biorxiv"),
        ("version", "0"),
        ("version", "01"),
        ("version", True),
        ("version", -1),
        ("version", "not-a-number"),
        ("date", "2026-02-30"),
        ("published", "https://attacker.invalid/article"),
        ("published", "10.5555/published%2Fextra"),
        ("published", "10.5555/published?next"),
        ("published", "10.5555/published#fragment"),
        ("published", "10.5555/published\\extra"),
        ("category", "bad\tcategory"),
        ("authors", ["Ada Example"]),
    ),
)
@pytest.mark.asyncio
async def test_details_doi_record_binding_and_retained_fields_fail_closed(case: str, value: object) -> None:
    payload = json.loads(_details_fixture("biorxiv_details_doi_success"))
    payload["collection"][0][case] = value

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            [payload],
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_details_doi_requires_every_version_valid_before_highest_version_selection() -> None:
    payload = json.loads(_details_fixture("medrxiv_details_doi_success"))
    payload["collection"][0]["server"] = "bioRxiv"

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "medrxiv",
            IdentifierLookupQuery("10.5555/medrxiv.details.synthetic"),
            [payload],
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("conflict", (False, True), ids=("exact_duplicate", "same_version_conflict"))
@pytest.mark.asyncio
async def test_details_same_doi_version_duplicate_is_exact_or_fails_atomically(conflict: bool) -> None:
    payload = json.loads(_details_fixture("medrxiv_details_doi_success"))
    duplicate = json.loads(json.dumps(payload["collection"][1]))
    if conflict:
        duplicate["title"] = "Conflicting retained title"
    payload["collection"].append(duplicate)

    if conflict:
        with pytest.raises(Exception) as caught:
            await _invoke_details_payloads(
                "medrxiv",
                IdentifierLookupQuery("10.5555/medrxiv.details.synthetic"),
                [payload],
            )
        _assert_typed_error(caught.value, "provider_payload_invalid")
    else:
        result, _dispatch, _group = await _invoke_details_payloads(
            "medrxiv",
            IdentifierLookupQuery("10.5555/medrxiv.details.synthetic"),
            [payload],
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].record["version"] == 2


@pytest.mark.parametrize(
    "case",
    (
        "interval",
        "message_category",
        "cursor",
        "count",
        "total_too_small",
        "result_date",
        "result_category",
        "server",
        "later_total",
        "later_cursor",
        "later_no_posts",
        "zero_count_with_remaining",
    ),
)
@pytest.mark.asyncio
async def test_details_interval_response_binding_is_atomic(case: str) -> None:
    first = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    second = json.loads(_details_fixture("biorxiv_details_interval_page_2"))
    if case == "interval":
        first["messages"][0]["interval"] = "2026-06-01:2026-06-03"
    elif case == "message_category":
        first["messages"][0]["category"] = "genetics"
    elif case == "cursor":
        first["messages"][0]["cursor"] = 1
    elif case == "count":
        first["messages"][0]["count"] = 2
    elif case == "total_too_small":
        first["messages"][0]["total"] = 0
    elif case == "result_date":
        first["collection"][0]["date"] = "2026-06-03"
    elif case == "result_category":
        first["collection"][0]["category"] = "genetics"
    elif case == "server":
        first["collection"][0]["server"] = "medRxiv"
    elif case == "later_total":
        second["messages"][0]["total"] = 3
    elif case == "later_cursor":
        second["messages"][0]["cursor"] = 0
    elif case == "later_no_posts":
        second = {"messages": [{"status": "no posts found"}], "collection": []}
    else:
        first["messages"][0]["count"] = 0
        first["collection"] = []

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
            [first, second],
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_details_interval_accepts_canonical_string_and_integer_numeric_variants() -> None:
    first = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    second = json.loads(_details_fixture("biorxiv_details_interval_page_2"))
    first["messages"][0].update({"cursor": "0", "count": "1", "total": 2})
    second["messages"][0].update({"cursor": 1, "count": 1, "total": "2"})

    result, dispatch, intent_group = await _invoke_details_payloads(
        "biorxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
        [first, second],
    )

    assert len(result.candidates) == 2
    assert dispatch.calls == [
        (intent_group.intents[0], None, ()),
        (intent_group.intents[0], NumericCursor(1), ()),
    ]


@pytest.mark.parametrize("field", ("cursor", "count", "total"))
@pytest.mark.parametrize("value", (True, -1, 1.0, "01", "+1", "1.0", "x" * 33))
@pytest.mark.asyncio
async def test_details_interval_rejects_noncanonical_numeric_metadata(field: str, value: object) -> None:
    payload = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    payload["messages"][0][field] = value

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
            [payload],
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_details_category_binding_canonicalizes_case_whitespace_and_underscores() -> None:
    first = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    second = json.loads(_details_fixture("biorxiv_details_interval_page_2"))
    for payload in (first, second):
        payload["messages"][0]["category"] = "cell_biology"
        for item in payload["collection"]:
            item["category"] = "  Cell   Biology  "

    result, _dispatch, _group = await _invoke_details_payloads(
        "biorxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "Cell Biology"),
        [first, second],
    )

    assert tuple(candidate.record["category"] for candidate in result.candidates) == (
        "cell biology",
        "cell biology",
    )


@pytest.mark.asyncio
async def test_details_all_category_wildcard_retains_diverse_item_categories() -> None:
    first = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    second = json.loads(_details_fixture("biorxiv_details_interval_page_2"))
    first["messages"][0]["category"] = "all"
    second["messages"][0]["category"] = "All"
    first["collection"][0]["category"] = "neuroscience"
    second["collection"][0]["category"] = "cell biology"

    result, _dispatch, _group = await _invoke_details_payloads(
        "biorxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "All"),
        [first, second],
    )

    assert tuple(candidate.record["category"] for candidate in result.candidates) == (
        "neuroscience",
        "cell biology",
    )


@pytest.mark.asyncio
async def test_details_all_category_wildcard_still_binds_message_echo() -> None:
    payload = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    payload["messages"][0].update({"category": "neuroscience", "total": 1})

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            DateIntervalQuery("2026-06-01", "2026-06-02", "All"),
            [payload],
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "case",
    (
        "encoded_doi_slash",
        "multi_slash_doi",
        "oversized_doi_suffix",
        "wrong_source_path",
        "wrong_adapter_version",
        "wrong_backend",
        "wrong_operation",
        "wrong_method",
        "wrong_allowance",
        "nonzero_initial_cursor",
        "interval_query_drift",
        "oversized_category_query",
    ),
)
@pytest.mark.asyncio
async def test_details_forged_group_or_path_rejects_before_dispatch(case: str) -> None:
    module = _module()
    if case in {"nonzero_initial_cursor", "interval_query_drift", "oversized_category_query"}:
        _registry, plan = _details_plan_for(
            "biorxiv",
            DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
        )
    else:
        _registry, plan = _details_plan_for(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            result_limit=30,
        )
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    candidate = group
    if case == "encoded_doi_slash":
        candidate = replace(
            group,
            intents=(replace(intent, path="/details/biorxiv/10.5555%2Fbiorxiv.details.synthetic/na/json"),),
        )
    elif case == "multi_slash_doi":
        candidate = replace(
            group,
            normalized_query="10.5555/biorxiv/details.synthetic",
            intents=(replace(intent, path="/details/biorxiv/10.5555/biorxiv/details.synthetic/na/json"),),
        )
    elif case == "oversized_doi_suffix":
        suffix = "x" * 129
        candidate = replace(
            group,
            normalized_query=f"10.5555/{suffix}",
            intents=(replace(intent, path=f"/details/biorxiv/10.5555/{suffix}/na/json"),),
        )
    elif case == "wrong_source_path":
        candidate = replace(
            group,
            intents=(replace(intent, path=intent.path.replace("/biorxiv/", "/medrxiv/")),),
        )
    elif case == "wrong_adapter_version":
        candidate = replace(group, adapter_version="unknown-details-v2")
    elif case == "wrong_backend":
        candidate = replace(group, backend_id="forged_details_backend")
    elif case == "wrong_operation":
        candidate = replace(
            group,
            intents=(replace(intent, operation_kind=OperationKind.CONDITIONAL_SUMMARY),),
        )
    elif case == "wrong_method":
        candidate = replace(group, intents=(replace(intent, method="POST"),))
    elif case == "wrong_allowance":
        candidate = replace(group, allowance=replace(group.allowance, physical_dispatches=2))
    elif case == "nonzero_initial_cursor":
        candidate = replace(group, intents=(replace(intent, path=intent.path.replace("/0/json", "/1/json")),))
    elif case == "interval_query_drift":
        candidate = replace(group, normalized_query="2026-06-01/2026-06-03/neuroscience")
    else:
        category = "x" * 129
        candidate = replace(
            group,
            normalized_query=f"2026-06-01/2026-06-02/{category}",
            intents=(replace(intent, query_pairs=(QueryPair("category", category),)),),
        )
    dispatch = _RecordingDispatch([])

    with pytest.raises(Exception) as caught:
        await module.biorxiv_medrxiv_gateway_adapters()[module.DETAILS_ADAPTER_ID](candidate, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert dispatch.calls == []


@pytest.mark.asyncio
async def test_details_interval_stops_at_four_pages_when_total_exceeds_route_cap() -> None:
    template = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    pages: list[dict[str, Any]] = []
    for page_index in range(4):
        cursor = page_index * 30
        collection = []
        for item_index in range(30):
            sequence = cursor + item_index + 1
            item = json.loads(json.dumps(template["collection"][0]))
            item["doi"] = f"10.5555/biorxiv.cap.{sequence}"
            item["title"] = f"Bounded cap result {sequence}"
            collection.append(item)
        pages.append(
            {
                "messages": [
                    {
                        "status": "ok",
                        "category": "neuroscience",
                        "interval": "2026-06-01:2026-06-02",
                        "cursor": str(cursor) if cursor else 0,
                        "count": 30,
                        "total": "999",
                    }
                ],
                "collection": collection,
            }
        )

    result, dispatch, group = await _invoke_details_payloads(
        "biorxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
        pages,
    )

    assert len(result.candidates) == 120
    assert [call[1] for call in dispatch.calls] == [
        None,
        NumericCursor(30),
        NumericCursor(60),
        NumericCursor(90),
    ]
    assert len(dispatch.calls) == group.limits.max_pages == 4


@pytest.mark.parametrize(
    ("body", "code"),
    (
        (b'{"messages":[],"messages":[],"collection":[]}', "provider_payload_invalid"),
        (b"x" * 2_097_153, "provider_parse_limit_exceeded"),
    ),
)
@pytest.mark.asyncio
async def test_details_strict_json_and_input_bounds_are_reused(body: bytes, code: str) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            [body],
        )

    _assert_typed_error(caught.value, code)


@pytest.mark.asyncio
async def test_details_cooperative_parse_deadline_is_reused() -> None:
    clock = _CountingClock(step=0.2)

    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            [json.loads(_details_fixture("biorxiv_details_doi_success"))],
            monotonic_clock=clock,
        )

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")
    assert clock.calls >= 4


@pytest.mark.parametrize(
    ("status_code", "content_type", "retry_after", "code"),
    (
        (503, "application/json", None, "provider_response_rejected"),
        (200, "text/html", None, "provider_response_rejected"),
        (429, None, "120", "provider_rate_limited"),
    ),
)
@pytest.mark.asyncio
async def test_details_http_mime_and_rate_limit_are_checked_before_body(
    status_code: int,
    content_type: str | None,
    retry_after: str | None,
    code: str,
) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
            [b"fixture-secret-not-json"],
            status_code=status_code,
            content_type=content_type,
            retry_after=retry_after,
        )

    _assert_typed_error(caught.value, code)
    assert caught.value.retry_after == ("120" if status_code == 429 else None)
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.parametrize("failure_kind", ("malformed", "timeout", "cancelled"))
@pytest.mark.asyncio
async def test_details_later_page_failure_commits_no_earlier_candidates(failure_kind: str) -> None:
    first = json.loads(_details_fixture("biorxiv_details_interval_page_1"))
    if failure_kind == "malformed":
        failure: object = b"{"
        expected_type = DiscoveryAdapterError
        expected_code = "provider_payload_invalid"
    elif failure_kind == "timeout":
        failure = DiscoveryExecutionError("gateway_timed_out")
        expected_type = DiscoveryExecutionError
        expected_code = "gateway_timed_out"
    else:
        failure = asyncio.CancelledError("details-page-two-cancelled")
        expected_type = asyncio.CancelledError
        expected_code = None

    with pytest.raises(BaseException) as caught:
        await _invoke_details_payloads(
            "biorxiv",
            DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
            [first, failure],
        )

    assert type(caught.value) is expected_type
    if expected_code is not None:
        assert caught.value.code == expected_code
    if failure_kind in {"timeout", "cancelled"}:
        assert caught.value is failure


def _derived_medrxiv_interval_payload(fixture_name: str) -> dict[str, Any]:
    payload = json.loads(_details_fixture(fixture_name))
    for item in payload["collection"]:
        item["server"] = "medRxiv"
        item["doi"] = item["doi"].replace("biorxiv", "medrxiv")
        item["title"] = item["title"].replace("interval", "medRxiv interval")
        item["abstract"] = item["abstract"].replace("interval", "medRxiv interval")
        item["authors"] = item["authors"].replace("Example", "MedExample")
        if item["published"] != "NA":
            item["published"] = item["published"].replace("interval", "medrxiv.interval")
    return payload


@pytest.mark.asyncio
async def test_medrxiv_interval_is_derived_and_executes_real_plan_end_to_end() -> None:
    module = _module()
    fixture_names = (
        "biorxiv_details_interval_page_1",
        "biorxiv_details_interval_page_2",
    )
    originals = tuple(json.loads(_details_fixture(name)) for name in fixture_names)
    payloads = tuple(_derived_medrxiv_interval_payload(name) for name in fixture_names)
    for original, payload in zip(originals, payloads, strict=True):
        assert payload["messages"] == original["messages"]
        assert len(payload["collection"]) == len(original["collection"])
        for before, after in zip(original["collection"], payload["collection"], strict=True):
            assert {key for key in before if before[key] != after[key]} <= {
                "server",
                "doi",
                "title",
                "abstract",
                "authors",
                "published",
            }

    registry, plan = _details_plan_for(
        "medrxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
    )
    paths: list[str] = []

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        paths.append(intent.path)
        return _response(route, intent, _payload_bytes(payloads[len(paths) - 1]))

    dispatch_ids = iter(("medrxiv-details-1", "medrxiv-details-2"))
    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
    )

    assert paths == [
        "/details/medrxiv/2026-06-01/2026-06-02/0/json",
        "/details/medrxiv/2026-06-01/2026-06-02/1/json",
    ]
    assert tuple(candidate.catalog_source_ids for candidate in execution.candidates) == (
        ("medrxiv",),
        ("medrxiv",),
    )
    assert all(candidate.record["source_platform"] == "medrxiv" for candidate in execution.candidates)
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert execution.usage.pages == 2
    assert execution.usage.accounting.created == execution.usage.accounting.debited == 2
    assert tuple(record.dispatch_id for record in execution.usage.physical_records) == (
        "medrxiv-details-1",
        "medrxiv-details-2",
    )


@pytest.mark.asyncio
async def test_details_doi_executes_real_plan_with_exact_accounting() -> None:
    module = _module()
    registry, plan = _details_plan_for(
        "biorxiv",
        IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"),
        result_limit=30,
    )
    paths: list[str] = []

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        paths.append(intent.path)
        return _response(route, intent, _details_fixture("biorxiv_details_doi_success"))

    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "biorxiv-doi-dispatch-1",
    )

    assert paths == ["/details/biorxiv/10.5555/biorxiv.details.synthetic/na/json"]
    assert len(execution.candidates) == 1
    assert execution.candidates[0].catalog_source_ids == ("biorxiv",)
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert execution.usage.pages == execution.usage.route_attempts == 1
    assert execution.usage.accounting.created == execution.usage.accounting.debited == 1
    assert execution.usage.physical_records[0].dispatch_id == "biorxiv-doi-dispatch-1"


@pytest.mark.asyncio
async def test_details_valid_empty_executes_as_valid_empty_with_one_page() -> None:
    module = _module()
    registry, plan = _details_plan_for(
        "biorxiv",
        DateIntervalQuery("2026-06-03", "2026-06-03"),
    )

    async def gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        return _response(route, intent, _details_fixture("biorxiv_details_interval_empty"))

    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "biorxiv-empty-dispatch-1",
    )

    assert execution.candidates == ()
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.VALID_EMPTY
    assert execution.logical_outcomes[0].code is None
    assert execution.usage.pages == 1
    assert execution.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED


@pytest.mark.asyncio
async def test_details_expanding_request_category_executes_as_valid_empty() -> None:
    module = _module()
    category = "ẞ" * 65
    registry, plan = _details_plan_for(
        "biorxiv",
        DateIntervalQuery("2026-06-03", "2026-06-03", category),
    )
    group = plan.dispatch_groups[0]
    intent = group.intents[0]
    dispatches: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    async def gateway(route, dispatched_intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        dispatches.append(
            (
                dispatched_intent.path,
                tuple((pair.name, pair.value) for pair in dispatched_intent.query_pairs),
            )
        )
        return _response(route, dispatched_intent, _details_fixture("biorxiv_details_interval_empty"))

    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "biorxiv-expanding-empty-1",
    )

    assert dispatches == [(intent.path, (("category", category),))]
    assert execution.candidates == ()
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.VALID_EMPTY
    assert execution.logical_outcomes[0].code is None
    assert execution.usage.pages == 1
    assert execution.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED


@pytest.mark.asyncio
async def test_details_one_source_failure_preserves_other_source_and_accounting() -> None:
    module = _module()
    registry = module.biorxiv_medrxiv_shadow_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=("medrxiv", "biorxiv"),
            query=DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
            filters=(),
            result_limit=4,
        ),
        registry=registry,
        readiness=module.biorxiv_medrxiv_shadow_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=BudgetCeilings(
            max_route_attempts=2,
            max_physical_dispatches=8,
            max_pages_per_route=4,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=160_000,
            max_results=4,
        ),
    )
    med_payloads = (
        _derived_medrxiv_interval_payload("biorxiv_details_interval_page_1"),
        _derived_medrxiv_interval_payload("biorxiv_details_interval_page_2"),
    )
    med_index = 0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal med_index
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        if route.route_id.startswith("biorxiv_"):
            return _response(route, intent, b"{")
        payload = med_payloads[med_index]
        med_index += 1
        return _response(route, intent, _payload_bytes(payload))

    dispatch_ids = iter(("partial-bio-1", "partial-med-1", "partial-med-2"))
    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=module.biorxiv_medrxiv_gateway_adapters(),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
    )
    outcomes = {outcome.catalog_source_id: outcome for outcome in execution.logical_outcomes}

    assert tuple(candidate.catalog_source_ids for candidate in execution.candidates) == (
        ("medrxiv",),
        ("medrxiv",),
    )
    assert outcomes["biorxiv"].state is LogicalOutcomeState.FAILED
    assert outcomes["biorxiv"].code == "provider_payload_invalid"
    assert outcomes["medrxiv"].state is LogicalOutcomeState.SUCCEEDED
    assert execution.usage.pages == execution.usage.accounting.created == 3
    assert execution.usage.accounting.debited == 3
    assert tuple(record.dispatch_id for record in execution.usage.physical_records) == (
        "partial-bio-1",
        "partial-med-1",
        "partial-med-2",
    )


@pytest.mark.asyncio
async def test_details_executor_cancellation_stops_before_third_page() -> None:
    module = _module()
    registry, plan = _details_plan_for(
        "biorxiv",
        DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"),
    )
    calls = 0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(route, intent, _details_fixture("biorxiv_details_interval_page_1"))
        raise asyncio.CancelledError("cancel-details-pagination")

    dispatch_ids = iter(("cancel-details-1", "cancel-details-2", "must-not-be-used"))
    with pytest.raises(asyncio.CancelledError, match="cancel-details-pagination"):
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters=module.biorxiv_medrxiv_gateway_adapters(),
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: next(dispatch_ids),
        )

    assert calls == 2


def test_family_adapter_ast_has_no_transport_legacy_wrapper_or_effect_seam() -> None:
    module_path = Path(__file__).parents[2] / "app" / "core" / "Research" / "discovery" / "biorxiv_medrxiv.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    imported_from: set[tuple[str, str]] = set()
    called_names: set[str] = set()
    accessed_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            imported_modules.add(module_name)
            imported_names.update(alias.name for alias in node.names)
            imported_from.update((module_name, alias.name) for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
        elif isinstance(node, ast.Attribute):
            accessed_names.add(node.attr)

    banned_prefixes = (
        "httpx",
        "requests",
        "aiohttp",
        "http.client",
        "socket",
        "urllib3",
        "urllib.request",
        "os",
        "subprocess",
        "keyring",
        "webbrowser",
        "browser",
        "playwright",
        "selenium",
        "tldw_Server_API.app.core.http_client",
        "tldw_Server_API.app.core.AuthNZ",
        "tldw_Server_API.app.core.config",
        "tldw_Server_API.app.core.DB_Management",
        "tldw_Server_API.app.core.Ingestion_Media_Processing",
        "tldw_Server_API.app.core.Media",
        "tldw_Server_API.app.core.Security.http_hop",
        "tldw_Server_API.app.core.Third_Party",
    )
    assert not {module for module in imported_modules if module.startswith(banned_prefixes)}
    assert ("urllib", "request") not in imported_from
    assert not imported_names.intersection(
        {
            "ClientSession",
            "CookieJar",
            "Popen",
            "Session",
            "afetch",
            "browser",
            "chromium",
            "cookiejar",
            "cookies",
            "environ",
            "fetch",
            "fetch_json",
            "firefox",
            "getenv",
            "http_hop",
            "keyring",
            "request_http_hop",
            "subprocess",
            "urlopen",
            "webkit",
        }
    )
    assert not called_names.intersection(
        {
            "HTTPConnection",
            "HTTPSConnection",
            "PoolManager",
            "ClientSession",
            "CookieJar",
            "Popen",
            "Session",
            "afetch",
            "browser",
            "chromium",
            "cookiejar",
            "cookies",
            "create_connection",
            "exec",
            "fetch",
            "fetch_json",
            "firefox",
            "get_cookie",
            "getenv",
            "goto",
            "launch",
            "new_context",
            "new_page",
            "open",
            "popen",
            "putenv",
            "request_http_hop",
            "run",
            "set_cookie",
            "sleep",
            "socket",
            "system",
            "unsetenv",
            "urlopen",
            "webkit",
        }
    )
    assert not accessed_names.intersection(
        {
            "ClientSession",
            "CookieJar",
            "Session",
            "browser",
            "chromium",
            "cookiejar",
            "cookies",
            "environ",
            "firefox",
            "keyring",
            "webkit",
        }
    )


@pytest.mark.asyncio
async def test_runtime_egress_tripwires_receive_zero_requests(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("adapter attempted direct egress")

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import http_hop
    from tldw_Server_API.app.core.Third_Party import BioRxiv

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(http_client, "fetch", forbidden)
    monkeypatch.setattr(http_client, "fetch_json", forbidden)
    monkeypatch.setattr(http_hop, "request_http_hop", forbidden)
    monkeypatch.setattr(BioRxiv, "search_biorxiv", forbidden)
    monkeypatch.setattr(BioRxiv, "get_biorxiv_by_doi", forbidden)

    result, dispatch, _group = await _invoke_body("biorxiv", _fixture("biorxiv"))
    details_result, details_dispatch, details_group = await _invoke_details_payloads(
        "medrxiv",
        IdentifierLookupQuery("10.5555/medrxiv.details.synthetic"),
        [json.loads(_details_fixture("medrxiv_details_doi_success"))],
    )

    assert len(result.candidates) == 1
    assert len(dispatch.calls) == 1
    assert len(details_result.candidates) == 1
    assert details_dispatch.calls == [(details_group.intents[0], None, ())]
    assert "example.invalid" not in repr(details_result)
