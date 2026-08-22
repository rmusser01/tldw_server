"""Offline contracts for the ClinicalTrials.gov/PMC shadow family."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from tldw_Server_API.app.core.Research.discovery.clinicaltrials_pubmed_central import (
    _FAMILY_PARSING_PROFILES,
    CLINICALTRIALS_FIELDS,
    CLINICALTRIALS_GOV_ADAPTER_ID,
    CLINICALTRIALS_GOV_ADAPTER_VERSION,
    NCBI_EMAIL,
    NCBI_TOOL,
    PUBMED_CENTRAL_ADAPTER_ID,
    PUBMED_CENTRAL_ADAPTER_VERSION,
    PUBMED_IDENTITY_ADAPTER_VERSION,
    PUBMED_IDENTITY_POLICY_VERSION,
    ROUTE_POLICY_VERSION,
    SHADOW_CATALOG_VERSION,
    SHADOW_READINESS_VERSION,
    SHADOW_REGISTRY_VERSION,
    clinicaltrials_pubmed_central_shadow_registry,
)
from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    BackendDefinition,
    BoundedDecimalQueryValuePolicy,
    BudgetCeilings,
    CredentialRequirement,
    CredentialStatus,
    DeferredNumericCSVQueryBinding,
    DiscoveryOutcomeIdentity,
    ExactOrigin,
    ExactQueryValuePolicy,
    ExecutionMode,
    JSONBodyPair,
    LiteralTermsQueryValuePolicy,
    OpaqueCursorQueryValuePolicy,
    OperationKind,
    PlannedDispatchGroup,
    QueryMode,
    QueryPair,
    ReadinessOverlay,
    ReadinessState,
    RouteKind,
    RouteLimits,
    RouteReadiness,
    SourceConstraint,
)
from tldw_Server_API.app.core.Research.discovery.executor import (
    AttemptJournal,
    BoundDispatch,
    DiscoveryAdapter,
    DiscoveryAdapterError,
    DiscoveryAdapterResult,
    DiscoveryExecutionError,
    LogicalOutcomeState,
    OpaqueCursor,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
)
from tldw_Server_API.app.core.Research.discovery.gateway_adapters import (
    MonotonicClock,
    _PayloadInvalid,
)
from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint
from tldw_Server_API.app.core.Research.discovery.planner import (
    GeneralFreeTextQuery,
    PlanningError,
    PlanningRequest,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)
from tldw_Server_API.app.core.Security.http_hop import HTTPHopLimits

pytestmark = pytest.mark.unit

_MODULE = "tldw_Server_API.app.core.Research.discovery.clinicaltrials_pubmed_central"
_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "research_discovery_gateway_adapters"


def _module():
    return importlib.import_module(_MODULE)


def _budget() -> BudgetCeilings:
    return BudgetCeilings(1, 2, 1, 0, 0, 40_000, 10)


def test_shadow_registry_replaces_only_pubmed_with_the_identity_overlay() -> None:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    route = registry.get_route("pubmed_ncbi_eutils_pubmed_direct")

    assert (
        SHADOW_CATALOG_VERSION,
        SHADOW_REGISTRY_VERSION,
        SHADOW_READINESS_VERSION,
        ROUTE_POLICY_VERSION,
        CLINICALTRIALS_GOV_ADAPTER_ID,
        CLINICALTRIALS_GOV_ADAPTER_VERSION,
        PUBMED_CENTRAL_ADAPTER_ID,
        PUBMED_CENTRAL_ADAPTER_VERSION,
        PUBMED_IDENTITY_POLICY_VERSION,
        PUBMED_IDENTITY_ADAPTER_VERSION,
        NCBI_TOOL,
        NCBI_EMAIL,
    ) == (
        "research-discovery-v2-clinicaltrials-pmc-shadow",
        "research-discovery-v2-clinicaltrials-pmc-shadow-2026-08-21",
        "research-discovery-readiness-v2-clinicaltrials-pmc-shadow",
        "research-discovery-route-policy-v2-clinicaltrials-pmc",
        "clinicaltrials_gov_v2",
        "clinicaltrials-gov-v2",
        "pubmed_central_v2",
        "pubmed-central-v2",
        "research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21",
        "pubmed-v2-ncbi-identity",
        "tldw_server",
        "contact@tldwproject.com",
    )
    assert CLINICALTRIALS_FIELDS == (
        "NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,Condition,"
        "InterventionName,LeadSponsorName,StudyType,StartDate,CompletionDate,HasResults"
    )
    assert registry.catalog_version == SHADOW_CATALOG_VERSION
    assert all(source.catalog_version == SHADOW_CATALOG_VERSION for source in registry.sources)
    assert route.adapter_version == PUBMED_IDENTITY_ADAPTER_VERSION
    assert route.policy.policy_version == PUBMED_IDENTITY_POLICY_VERSION
    assert route.policy.allowed_query_keys == (
        "db",
        "term",
        "retstart",
        "retmax",
        "retmode",
        "sort",
        "datetype",
        "mindate",
        "maxdate",
        "tool",
        "email",
        "id",
    )
    assert route.policy.pagination_query_key == "retstart"
    assert route.policy.query_value_policies == ()
    assert set(_FAMILY_PARSING_PROFILES) == {
        (CLINICALTRIALS_GOV_ADAPTER_ID, CLINICALTRIALS_GOV_ADAPTER_VERSION),
        (PUBMED_CENTRAL_ADAPTER_ID, PUBMED_CENTRAL_ADAPTER_VERSION),
    }
    assert {
        identity: (
            profile.max_input_bytes,
            profile.max_records,
            profile.max_depth,
            profile.max_nodes,
            profile.max_string_chars,
            profile.max_numeric_token_chars,
            profile.parse_deadline_ms,
        )
        for identity, profile in _FAMILY_PARSING_PROFILES.items()
    } == {
        ("clinicaltrials_gov_v2", "clinicaltrials-gov-v2"): (2_097_152, 50, 16, 50_000, 65_536, 32, 500),
        ("pubmed_central_v2", "pubmed-central-v2"): (2_097_152, 100, 16, 50_000, 65_536, 32, 500),
    }


def test_shadow_registry_preserves_every_non_pubmed_route_exactly() -> None:
    foundation = foundation_registry()
    shadow = clinicaltrials_pubmed_central_shadow_registry()

    assert tuple(
        route
        for route in shadow.routes
        if route.route_id not in {"pubmed_ncbi_eutils_pubmed_direct", "clinicaltrials_gov_studies_search_direct"}
    ) == tuple(route for route in foundation.routes if route.route_id != "pubmed_ncbi_eutils_pubmed_direct")


def test_identity_overlay_plans_exact_identity_pairs_on_both_hops() -> None:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    plan = compile_discovery_plan(
        PlanningRequest(("pubmed",), "bounded discovery", (), 7),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )
    search, summary = plan.dispatch_groups[0].intents

    assert tuple((pair.name, pair.value) for pair in search.query_pairs) == (
        ("db", "pubmed"),
        ("term", "bounded discovery"),
        ("retstart", "0"),
        ("retmax", "7"),
        ("retmode", "json"),
        ("sort", "relevance"),
        ("tool", "tldw_server"),
        ("email", "contact@tldwproject.com"),
    )
    assert tuple((pair.name, pair.value) for pair in summary.query_pairs) == (
        ("db", "pubmed"),
        ("retmode", "json"),
        ("tool", "tldw_server"),
        ("email", "contact@tldwproject.com"),
    )
    assert summary.query_bindings[0].query_name == "id"
    assert QueryPair("tool", "tldw_server") in search.query_pairs


def _registry_with_pubmed_route(route: AccessRoute) -> DiscoveryRegistry:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    original_route_id = "pubmed_ncbi_eutils_pubmed_direct"
    return DiscoveryRegistry(
        catalog_version=registry.catalog_version,
        registry_version="identity-mutation-registry-v1",
        sources=tuple(
            replace(
                source,
                route_references=tuple(
                    (
                        replace(reference, route_id=route.route_id)
                        if reference.route_id == original_route_id
                        else reference
                    )
                    for reference in source.route_references
                ),
            )
            for source in registry.sources
        ),
        routes=tuple(route if item.route_id == original_route_id else item for item in registry.routes),
        backends=registry.backends,
    )


def _generic_registry_with_identity_component(component: str) -> tuple[DiscoveryRegistry, AccessRoute]:
    foundation = foundation_registry()
    source = foundation.get_source("arxiv")
    original = foundation.get_route(source.route_references[0].route_id)
    if component == "adapter_version":
        mutated = replace(original, adapter_version=PUBMED_IDENTITY_ADAPTER_VERSION)
    elif component == "policy_version":
        mutated = replace(
            original,
            policy=replace(
                original.policy,
                policy_version=PUBMED_IDENTITY_POLICY_VERSION,
                policy_digest="",
            ),
        )
    else:
        raise ValueError("unknown_identity_component")
    registry = DiscoveryRegistry(
        catalog_version=foundation.catalog_version,
        registry_version="generic-identity-component-registry-v1",
        sources=foundation.sources,
        routes=tuple(mutated if route.route_id == original.route_id else route for route in foundation.routes),
        backends=foundation.backends,
    )
    return registry, mutated


def test_identity_adapter_version_on_generic_route_fails_closed_before_plan_emission() -> None:
    registry, route = _generic_registry_with_identity_component("adapter_version")

    assert route.route_id != "pubmed_ncbi_eutils_pubmed_direct"
    assert route.backend_id != "ncbi_eutils_pubmed"
    assert route.adapter_id != "pubmed_v2"
    with pytest.raises(PlanningError, match="invalid_pubmed_route_identity"):
        compile_discovery_plan(
            PlanningRequest(("arxiv",), "bounded discovery", (), 7),
            registry=registry,
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=_budget(),
        )


def test_identity_policy_version_on_generic_route_fails_closed_before_plan_emission() -> None:
    registry, route = _generic_registry_with_identity_component("policy_version")

    assert route.route_id != "pubmed_ncbi_eutils_pubmed_direct"
    assert route.backend_id != "ncbi_eutils_pubmed"
    assert route.adapter_id != "pubmed_v2"
    with pytest.raises(PlanningError, match="invalid_pubmed_route_identity"):
        compile_discovery_plan(
            PlanningRequest(("arxiv",), "bounded discovery", (), 7),
            registry=registry,
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=_budget(),
        )


@pytest.mark.parametrize(
    "route_change",
    (
        {"route_id": "pubmed_ncbi_eutils_pubmed_partial"},
        {"backend_id": "crossref_api"},
        {"adapter_id": "pubmed_v3"},
        {"adapter_version": "foundation-v2"},
        {"policy": "foundation"},
    ),
)
def test_partial_or_swapped_pubmed_overlay_identity_fails_closed_before_plan_emission(
    route_change: dict[str, str],
) -> None:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    route = registry.get_route("pubmed_ncbi_eutils_pubmed_direct")
    policy = route.policy
    if route_change.get("policy") == "foundation":
        policy = replace(policy, policy_version="research-discovery-route-policy-v2-foundation", policy_digest="")
    mutated = replace(
        route,
        route_id=route_change.get("route_id", route.route_id),
        backend_id=route_change.get("backend_id", route.backend_id),
        adapter_id=route_change.get("adapter_id", route.adapter_id),
        adapter_version=route_change.get("adapter_version", route.adapter_version),
        policy=policy,
    )
    readiness = foundation_readiness(ExecutionMode.SYNTHETIC)
    readiness = replace(
        readiness,
        routes=tuple(
            replace(entry, route_id=mutated.route_id) if entry.route_id == route.route_id else entry
            for entry in readiness.routes
        ),
    )

    with pytest.raises(PlanningError, match="invalid_pubmed_route_identity"):
        compile_discovery_plan(
            PlanningRequest(("pubmed",), "bounded discovery", (), 7),
            registry=_registry_with_pubmed_route(mutated),
            readiness=readiness,
            budget=_budget(),
        )


@pytest.mark.parametrize(
    "filters",
    (
        (QueryPair("tool", "attacker"),),
        (QueryPair("email", "attacker@example.test"),),
        (QueryPair("tool", "attacker"), QueryPair("email", "attacker@example.test")),
    ),
)
def test_identity_overlay_rejects_user_supplied_identity_filters(filters: tuple[QueryPair, ...]) -> None:
    with pytest.raises(PlanningError, match="identity_query_filter_not_allowed"):
        compile_discovery_plan(
            PlanningRequest(("pubmed",), "bounded discovery", filters, 7),
            registry=clinicaltrials_pubmed_central_shadow_registry(),
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=_budget(),
        )


def _clinicaltrials_test_readiness(mode: ExecutionMode) -> ReadinessOverlay:
    foundation = foundation_readiness(mode)
    return ReadinessOverlay(
        overlay_version=SHADOW_READINESS_VERSION,
        execution_mode=mode,
        routes=foundation.routes
        + (
            RouteReadiness(
                route_id="clinicaltrials_gov_studies_search_direct",
                state=ReadinessState.READY,
                credential_status=CredentialStatus.NOT_REQUIRED,
                reason=f"{mode.value}_ready",
            ),
        ),
    )


def _clinicaltrials_test_adapters(clock: MonotonicClock) -> Mapping[str, DiscoveryAdapter]:
    async def adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _module()._execute_clinicaltrials_adapter(group, dispatch, clock)

    return MappingProxyType({CLINICALTRIALS_GOV_ADAPTER_ID: adapter})


def _clinical_budget(*, result_limit: int = 100) -> BudgetCeilings:
    return BudgetCeilings(1, 2, 2, 0, 0, 40_000, result_limit)


def _clinical_plan(*, result_limit: int = 100):
    registry = _module().clinicaltrials_pubmed_central_shadow_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            ("clinicaltrials_gov",),
            GeneralFreeTextQuery("  Synthetic bounded discovery  "),
            (),
            result_limit,
        ),
        registry=registry,
        readiness=_clinicaltrials_test_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=_clinical_budget(result_limit=result_limit),
    )
    return registry, plan


def _fixture_bytes(name: str) -> bytes:
    return (_FIXTURE_ROOT / f"clinicaltrials_{name}.json").read_bytes()


def _fixture_payload(name: str) -> dict[str, Any]:
    return json.loads(_fixture_bytes(name))


def _payload_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _clinical_response(
    route,
    intent,
    body: Any,
    *,
    status_code: Any = 200,
    content_type: str | None = "application/json",
    retry_after: Any = None,
) -> DiscoveryGatewayResponse:
    body_length = len(body) if hasattr(body, "__len__") else 0
    headers = () if content_type is None else (("content-type", content_type),)
    origin = route.policy.origin
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


async def _invoke_clinical_bodies(
    bodies: list[Any],
    *,
    result_limit: int = 100,
    clock: MonotonicClock | None = None,
    status_code: Any = 200,
    content_type: str | None = "application/json",
    retry_after: Any = None,
):
    registry, plan = _clinical_plan(result_limit=result_limit)
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    responses = [
        (
            body
            if isinstance(body, BaseException)
            else _clinical_response(
                route,
                group.intents[0],
                body,
                status_code=status_code,
                content_type=content_type,
                retry_after=retry_after,
            )
        )
        for body in bodies
    ]
    dispatch = _RecordingDispatch(responses)
    parser_clock = clock or _CountingClock()
    result = await _module()._execute_clinicaltrials_adapter(group, dispatch, parser_clock)
    return result, dispatch, group


def _single_record_payload() -> dict[str, Any]:
    payload = _fixture_payload("success_page_1")
    payload["totalCount"] = 1
    payload.pop("nextPageToken")
    return payload


def _minimal_study(index: int, *, title: str | None = None) -> dict[str, Any]:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": f"NCT{90_000_000 + index:08d}",
                "briefTitle": title or f"Synthetic trial {index}",
            }
        }
    }


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    current: Any = root
    for segment in path[:-1]:
        current = current.setdefault(segment, {})
    current[path[-1]] = value


def _assert_adapter_error(error: BaseException, code: str = "provider_payload_invalid") -> None:
    assert type(error) is DiscoveryAdapterError
    assert error.code == code
    assert str(error) == code


def _cloned_clinical_group() -> PlannedDispatchGroup:
    """Return a structurally independent valid group for frozen-field mutation tests."""
    _registry, plan = _clinical_plan()
    original = plan.dispatch_groups[0]
    limits = replace(original.limits)
    intent = replace(original.intents[0], limits=replace(limits))
    return replace(original, limits=limits, intents=(intent,), allowance=replace(original.allowance))


def test_clinicaltrials_constructor_and_first_page_plan_are_exact() -> None:
    registry, plan = _clinical_plan(result_limit=100)
    source = registry.get_source("clinicaltrials_gov")
    route = registry.get_route("clinicaltrials_gov_studies_search_direct")
    backend = registry.get_backend("clinicaltrials_gov_api_v2")
    group = plan.dispatch_groups[0]
    intent = group.intents[0]

    assert source.catalog_source_id == "clinicaltrials_gov"
    assert source.display_name == "ClinicalTrials.gov"
    assert source.site_hosts == ("clinicaltrials.gov",)
    assert source.aliases == ("clinical_trials_gov", "clinical_trials")
    assert source.categories == ("biomedical", "clinical_trials")
    assert source.content_types == ("clinical_trials", "study_records", "summaries")
    assert source.surfaces == ("standalone_search", "deep_research")
    assert source.route_references[0].route_id == route.route_id
    assert source.route_references[0].source_predicate is None
    assert source.priority == 110
    assert source.catalog_version == SHADOW_CATALOG_VERSION
    assert backend == BackendDefinition("clinicaltrials_gov_api_v2", "ClinicalTrials.gov API v2")

    assert route.route_kind is RouteKind.DIRECT
    assert route.query_modes == (QueryMode.GENERAL_FREE_TEXT,)
    assert route.source_constraint is SourceConstraint.NATIVE_CORPUS
    assert route.attribution_basis == "native_nct_record"
    assert route.credential_requirement is CredentialRequirement.NONE
    assert route.fallback_order == 0
    assert route.adapter_id == CLINICALTRIALS_GOV_ADAPTER_ID
    assert route.adapter_version == CLINICALTRIALS_GOV_ADAPTER_VERSION
    assert route.max_physical_dispatches == 2
    assert route.policy.policy_version == ROUTE_POLICY_VERSION
    assert route.policy.origin == ExactOrigin("https", "clinicaltrials.gov", 443)
    assert route.policy.methods == ("GET",)
    assert route.policy.paths == ("/api/v2/studies",)
    assert route.policy.allowed_query_keys == (
        "query.term",
        "format",
        "markupFormat",
        "fields",
        "pageSize",
        "countTotal",
        "pageToken",
    )
    assert route.policy.pagination_query_key == "pageToken"
    assert route.policy.allowed_json_body_keys == ()
    assert route.policy.integer_json_body_keys == ()
    assert route.policy.limits == RouteLimits(2, 0, 0, 20_000, 2_097_152, 100, 16_384)
    assert route.policy.query_value_policies == (
        LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
        ExactQueryValuePolicy("format", "json"),
        ExactQueryValuePolicy("markupFormat", "legacy"),
        ExactQueryValuePolicy("fields", CLINICALTRIALS_FIELDS),
        BoundedDecimalQueryValuePolicy("pageSize", 50),
        ExactQueryValuePolicy("countTotal", "true"),
        OpaqueCursorQueryValuePolicy("pageToken", 1_024, required=False),
    )
    assert group.allowance.pages == group.allowance.physical_dispatches == 2
    assert intent.operation_kind is OperationKind.SEARCH
    assert intent.method == "GET"
    assert intent.path == "/api/v2/studies"
    assert tuple((pair.name, pair.value) for pair in intent.query_pairs) == (
        ("query.term", '"Synthetic" AND "bounded" AND "discovery"'),
        ("format", "json"),
        ("markupFormat", "legacy"),
        ("fields", CLINICALTRIALS_FIELDS),
        ("pageSize", "50"),
        ("countTotal", "true"),
    )
    assert intent.json_body_pairs == ()
    assert intent.query_bindings == ()
    assert plan.allowance.aggregate_wall_time_ms == 40_000
    assert plan.ceilings.max_wall_time_ms == 40_000


def test_clinicaltrials_profile_is_exact_and_local() -> None:
    profile = _FAMILY_PARSING_PROFILES[(CLINICALTRIALS_GOV_ADAPTER_ID, CLINICALTRIALS_GOV_ADAPTER_VERSION)]

    assert (
        profile.max_input_bytes,
        profile.max_records,
        profile.max_depth,
        profile.max_nodes,
        profile.max_string_chars,
        profile.max_numeric_token_chars,
        profile.parse_deadline_ms,
    ) == (2_097_152, 50, 16, 50_000, 65_536, 32, 500)


@pytest.mark.parametrize(
    ("field", "value", "mirror_intent"),
    (
        ("route_id", "clinicaltrials_gov_studies_search_swapped", False),
        ("backend_id", "openalex_api", False),
        ("adapter_id", "clinicaltrials_gov_v3", False),
        ("adapter_version", "clinicaltrials-gov-v3", False),
        ("policy_digest", "0" * 64, True),
        ("fallback_order", 1, False),
        ("filters", (QueryPair("status", "attacker"),), False),
    ),
)
def test_clinicaltrials_trusted_inputs_reject_group_identity_policy_and_filter_drift(
    field: str, value: object, mirror_intent: bool
) -> None:
    group = _cloned_clinical_group()
    object.__setattr__(group, field, value)
    if mirror_intent:
        object.__setattr__(group.intents[0], field, value)

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


def test_clinicaltrials_trusted_inputs_reject_intent_route_identity_drift() -> None:
    group = _cloned_clinical_group()
    intent = group.intents[0]
    object.__setattr__(intent, "route_id", "clinicaltrials_gov_studies_search_alternate")

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize("intents", ((), "duplicate", "list"))
def test_clinicaltrials_trusted_inputs_require_one_intent_in_an_exact_tuple(intents: object) -> None:
    group = _cloned_clinical_group()
    intent = group.intents[0]
    mutated = (intent, intent) if intents == "duplicate" else [intent] if intents == "list" else intents
    object.__setattr__(group, "intents", mutated)

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("physical_dispatches", 1),
        ("pages", 1),
        ("redirects", 1),
        ("retries", 1),
    ),
)
def test_clinicaltrials_trusted_inputs_reject_every_allowance_drift(field: str, value: int) -> None:
    group = _cloned_clinical_group()
    object.__setattr__(group.allowance, field, value)

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_pages", 3),
        ("max_redirects", 1),
        ("max_retries", 1),
        ("timeout_ms", 19_999),
        ("max_response_bytes", 2_097_151),
        ("max_results", 99),
        ("max_request_body_bytes", 16_383),
    ),
)
def test_clinicaltrials_trusted_inputs_reject_every_group_route_limit_drift(field: str, value: int) -> None:
    group = _cloned_clinical_group()
    object.__setattr__(group.limits, field, value)
    object.__setattr__(group.intents[0].limits, field, value)

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_pages", 3),
        ("max_redirects", 1),
        ("max_retries", 1),
        ("timeout_ms", 19_999),
        ("max_response_bytes", 2_097_151),
        ("max_results", 99),
        ("max_request_body_bytes", 16_383),
    ),
)
def test_clinicaltrials_trusted_inputs_reject_independent_intent_limit_drift(field: str, value: int) -> None:
    group = _cloned_clinical_group()
    object.__setattr__(group.intents[0].limits, field, value)

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("policy_digest", "0" * 64),
        ("operation_kind", OperationKind.CONDITIONAL_SUMMARY),
        ("method", "POST"),
        ("path", "/api/v2/other"),
        ("json_body_pairs", (JSONBodyPair("page", 1),)),
        (
            "query_bindings",
            (DeferredNumericCSVQueryBinding("synthetic_ids", "id", 1, 8),),
        ),
    ),
)
def test_clinicaltrials_trusted_inputs_reject_independent_intent_material_drift(field: str, value: object) -> None:
    group = _cloned_clinical_group()
    object.__setattr__(group.intents[0], field, value)

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize(
    ("_case", "index", "replacement"),
    (
        ("term_key", 0, QueryPair("query.other", '"Synthetic" AND "bounded" AND "discovery"')),
        ("term_unquoted", 0, QueryPair("query.term", "Synthetic")),
        ("term_too_many", 0, QueryPair("query.term", " AND ".join(f'"term{index}"' for index in range(9)))),
        ("term_too_long", 0, QueryPair("query.term", f'"{"x" * 33}"')),
        ("term_non_alnum", 0, QueryPair("query.term", '"synthetic-trial"')),
        ("format", 1, QueryPair("format", "xml")),
        ("markup_format", 2, QueryPair("markupFormat", "markdown")),
        ("fields", 3, QueryPair("fields", f"{CLINICALTRIALS_FIELDS},Location")),
        ("page_size_key", 4, QueryPair("limit", "50")),
        ("page_size_zero", 4, QueryPair("pageSize", "0")),
        ("page_size_over", 4, QueryPair("pageSize", "51")),
        ("page_size_noncanonical", 4, QueryPair("pageSize", "050")),
        ("page_size_non_ascii", 4, QueryPair("pageSize", "５")),
        ("page_size_not_decimal", 4, QueryPair("pageSize", "fifty")),
        ("page_size_token_too_long", 4, QueryPair("pageSize", "1" * 33)),
        ("count_total", 5, QueryPair("countTotal", "false")),
    ),
)
def test_clinicaltrials_trusted_inputs_reject_first_page_query_key_or_value_drift(
    _case: str, index: int, replacement: QueryPair
) -> None:
    group = _cloned_clinical_group()
    pairs = list(group.intents[0].query_pairs)
    pairs[index] = replacement
    object.__setattr__(group.intents[0], "query_pairs", tuple(pairs))

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize("case", ("wrong_order", "first_page_token"))
def test_clinicaltrials_trusted_inputs_reject_first_page_query_shape_drift(case: str) -> None:
    group = _cloned_clinical_group()
    pairs = list(group.intents[0].query_pairs)
    if case == "wrong_order":
        pairs[1], pairs[2] = pairs[2], pairs[1]
    else:
        pairs.append(QueryPair("pageToken", "synthetic-page-two"))
    object.__setattr__(group.intents[0], "query_pairs", tuple(pairs))

    with pytest.raises(DiscoveryAdapterError, match="provider_payload_invalid"):
        _module()._trusted_clinicaltrials_inputs(group)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("2026", "2026"),
        ("2026-02", "2026-02"),
        ("2026-02-28", "2026-02-28"),
        ("0000", None),
        ("2026-13", None),
        ("2026-02-30", None),
        ("2026-2", None),
        (" 2026", None),
    ),
)
def test_clinicaltrials_partial_date_accepts_only_calendar_valid_exact_forms(value: str, expected: str | None) -> None:
    assert _module()._partial_date(value) == expected


@pytest.mark.parametrize("value", (None, 2026, True, ["2026"]))
def test_clinicaltrials_partial_date_rejects_present_non_string_values(value: object) -> None:
    with pytest.raises(_PayloadInvalid):
        _module()._partial_date(value)


def test_clinicaltrials_legacy_summary_converts_only_safe_data_nodes() -> None:
    value = "<!DOCTYPE fixture><p>Synthetic &amp; safe<!-- hidden --></p><script>ignored</script><style>hidden</style>"

    assert _module()._legacy_summary_text(value) == "Synthetic & safe"


@pytest.mark.parametrize(
    "value",
    (
        "[label](https://example.org/article)",
        "![image](https://example.org/image.png)",
        "<https://example.org/article>",
        '<a href="https://example.org/article">label</a>',
        "plain https://example.org/article",
        "www.example.org/article",
        "mailto:synthetic@example.org",
        "javascript:alert(1)",
        "data:text/plain,synthetic",
        "<unterminated synthetic markup",
        "&syntheticunknownentity;",
        "encoded unsafe control &#0;",
        "encoded unsafe surrogate &#xD800;",
        "safe\x00unsafe",
        "safe\x1funsafe",
        "safe\ud800unsafe",
        "x" * 65_537,
        "x" * 16_385,
    ),
)
def test_clinicaltrials_legacy_summary_drops_unsafe_or_overbound_optional_content(value: str) -> None:
    assert _module()._legacy_summary_text(value) is None


@pytest.mark.parametrize(
    ("value", "required", "expected"),
    (
        ("  Safe\u2003synthetic  title ", True, "Safe synthetic title"),
        ("https://example.org/article", False, None),
        ("<b>markup</b>", False, None),
        ("safe\x00unsafe", False, None),
        ("", False, None),
        ("x" * 257, False, None),
    ),
)
def test_clinicaltrials_plain_text_normalizes_or_drops_per_requiredness(
    value: str, required: bool, expected: str | None
) -> None:
    assert _module()._plain_clinical_text(value, max_chars=256, required=required) == expected


@pytest.mark.parametrize(
    "value",
    ("https://example.org/article", "<b>markup</b>", "<unterminated markup", "safe\x00unsafe", ""),
)
def test_clinicaltrials_required_plain_text_rejects_unsafe_or_empty(value: str) -> None:
    with pytest.raises(_PayloadInvalid):
        _module()._plain_clinical_text(value, max_chars=1_024, required=True)


@pytest.mark.asyncio
async def test_clinicaltrials_exact_two_page_fixture_normalizes_only_approved_projection() -> None:
    result, dispatch, group = await _invoke_clinical_bodies(
        [_fixture_bytes("success_page_1"), _fixture_bytes("success_page_2")]
    )

    assert len(result.candidates) == 2
    first = dict(result.candidates[0].record)
    second = dict(result.candidates[1].record)
    assert first == {
        "title": "Synthetic bounded trial one",
        "authors": (),
        "abstract": "Synthetic summary & bounded details.",
        "snippet": "Synthetic summary & bounded details.",
        "doi": None,
        "pmid": None,
        "pmcid": None,
        "arxiv_id": None,
        "url": "https://clinicaltrials.gov/study/NCT90000001",
        "pdf_url": None,
        "provider": "clinicaltrials_gov",
        "provider_ids": {"nct_id": "NCT90000001"},
        "source_metadata": {
            "brief_title": "Synthetic bounded trial one",
            "official_title": "Synthetic official trial title one",
            "overall_status": "RECRUITING",
            "conditions": ("Synthetic condition",),
            "interventions": ("Synthetic intervention",),
            "lead_sponsor": "Synthetic sponsor",
            "study_type": "INTERVENTIONAL",
            "start_date": "2026-01",
            "completion_date": "2027",
            "has_results": False,
        },
    }
    assert second["title"] == "Synthetic official-only trial title two"
    assert second["authors"] == ()
    assert second["abstract"] is None
    assert second["snippet"] is None
    assert second["doi"] is second["pmid"] is second["pmcid"] is second["arxiv_id"] is None
    assert second["pdf_url"] is None
    assert second["provider_ids"] == {"nct_id": "NCT90000002"}
    assert second["source_metadata"] == {
        "official_title": "Synthetic official-only trial title two",
        "overall_status": "COMPLETED",
        "conditions": (),
        "interventions": (),
        "lead_sponsor": "Synthetic sponsor two",
        "study_type": "OBSERVATIONAL",
        "start_date": "2025-02-28",
        "completion_date": "2026-02-28",
        "has_results": True,
    }
    assert tuple(candidate.candidate_id for candidate in result.candidates) == tuple(
        DiscoveryOutcomeIdentity.from_fingerprint(build_fingerprint(dict(candidate.record))).document_id
        for candidate in result.candidates
    )
    assert dispatch.calls == [
        (group.intents[0], None, ()),
        (group.intents[0], OpaqueCursor("synthetic+token/one=="), ()),
    ]
    assert "synthetic+token/one==" not in repr(result)


@pytest.mark.asyncio
async def test_clinicaltrials_exact_empty_fixture_is_terminal_after_one_call() -> None:
    result, dispatch, group = await _invoke_clinical_bodies([_fixture_bytes("empty")])

    assert result == DiscoveryAdapterResult(candidates=())
    assert dispatch.calls == [(group.intents[0], None, ())]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (lambda payload: payload.pop("totalCount"), "provider_payload_invalid"),
        (lambda payload: payload.__setitem__("totalCount", True), "provider_payload_invalid"),
        (lambda payload: payload.__setitem__("totalCount", -1), "provider_payload_invalid"),
        (lambda payload: payload.__setitem__("studies", {}), "provider_payload_invalid"),
        (lambda payload: payload.__setitem__("studies", []), "provider_payload_invalid"),
        (lambda payload: payload.pop("nextPageToken"), "provider_payload_invalid"),
        (lambda payload: payload.__setitem__("totalCount", 1), "provider_payload_invalid"),
    ),
)
@pytest.mark.asyncio
async def test_clinicaltrials_page_count_cardinality_and_token_biconditional_fail_closed(
    mutation, expected_code: str
) -> None:
    payload = _fixture_payload("success_page_1")
    mutation(payload)

    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([_payload_bytes(payload)])

    _assert_adapter_error(caught.value, expected_code)


@pytest.mark.parametrize("token", ("", "contains space", "nonascii-\u00e9", "control\x7f", "x" * 1_025))
@pytest.mark.asyncio
async def test_clinicaltrials_invalid_tokens_fail_before_continuation(token: str) -> None:
    payload = _fixture_payload("success_page_1")
    payload["nextPageToken"] = token

    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([_payload_bytes(payload)])

    _assert_adapter_error(caught.value)


@pytest.mark.asyncio
async def test_clinicaltrials_frozen_count_change_fails_atomically() -> None:
    second = _fixture_payload("success_page_2")
    second["totalCount"] = 3
    second["nextPageToken"] = "synthetic-page-three"

    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([_fixture_bytes("success_page_1"), _payload_bytes(second)])

    _assert_adapter_error(caught.value)


@pytest.mark.asyncio
async def test_clinicaltrials_page_size_ceiling_fails_before_deduplication() -> None:
    oversized_page = {"totalCount": 51, "studies": [_minimal_study(index) for index in range(51)]}
    oversized_page["nextPageToken"] = "synthetic-next"
    with pytest.raises(Exception) as page_error:
        await _invoke_clinical_bodies([_payload_bytes(oversized_page)])
    _assert_adapter_error(page_error.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_clinicaltrials_cumulative_raw_ceiling_rejects_two_individually_widened_pages_totaling_101(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    strict_page = module._clinicaltrials_page
    parsed_record_counts: list[int] = []

    def widened_page(payload, *, guard, page_size):
        assert page_size == 50
        page = strict_page(payload, guard=guard, page_size=51)
        parsed_record_counts.append(len(page.records))
        return page

    monkeypatch.setattr(module, "_clinicaltrials_page", widened_page)
    page_one = {"totalCount": 101, "studies": [_minimal_study(index) for index in range(50)]}
    page_one["nextPageToken"] = "synthetic-page-two"
    page_two = {"totalCount": 101, "studies": [_minimal_study(index + 50) for index in range(51)]}
    with pytest.raises(Exception) as cumulative_error:
        await _invoke_clinical_bodies([_payload_bytes(page_one), _payload_bytes(page_two)])
    _assert_adapter_error(cumulative_error.value)
    assert parsed_record_counts == [50, 51]


@pytest.mark.asyncio
async def test_clinicaltrials_hundred_raw_ceiling_discards_valid_repeated_ceiling_token_without_third_call() -> None:
    page_one = {"totalCount": 101, "studies": [_minimal_study(index) for index in range(50)]}
    page_one["nextPageToken"] = "synthetic-ceiling-token"
    page_two = {"totalCount": 101, "studies": [_minimal_study(index + 50) for index in range(50)]}
    page_two["nextPageToken"] = "synthetic-ceiling-token"

    result, dispatch, _group = await _invoke_clinical_bodies([_payload_bytes(page_one), _payload_bytes(page_two)])

    assert len(result.candidates) == 100
    assert len(dispatch.calls) == 2
    assert dispatch.responses == []


@pytest.mark.asyncio
async def test_clinicaltrials_small_requested_limit_still_makes_bounded_second_call() -> None:
    result, dispatch, _group = await _invoke_clinical_bodies(
        [_fixture_bytes("success_page_1"), _fixture_bytes("success_page_2")], result_limit=1
    )

    assert len(result.candidates) == 2
    assert len(dispatch.calls) == 2


@pytest.mark.parametrize("nct_id", (None, "", "NCT123", "nct90000001", "NCT9000000X", 90000001))
@pytest.mark.asyncio
async def test_clinicaltrials_missing_or_invalid_nct_id_fails_route(nct_id: object) -> None:
    payload = _single_record_payload()
    identification = payload["studies"][0]["protocolSection"]["identificationModule"]
    if nct_id is None:
        identification.pop("nctId")
    else:
        identification["nctId"] = nct_id

    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([_payload_bytes(payload)])

    _assert_adapter_error(caught.value)


@pytest.mark.asyncio
async def test_clinicaltrials_identical_duplicate_collapses_only_after_raw_accounting() -> None:
    first = _fixture_payload("success_page_1")
    second = {"totalCount": 2, "studies": [first["studies"][0]]}

    result, dispatch, _group = await _invoke_clinical_bodies([_payload_bytes(first), _payload_bytes(second)])

    assert len(result.candidates) == 1
    assert len(dispatch.calls) == 2


@pytest.mark.asyncio
async def test_clinicaltrials_conflicting_duplicate_fails_atomically() -> None:
    first = _fixture_payload("success_page_1")
    conflict = json.loads(json.dumps(first["studies"][0]))
    conflict["protocolSection"]["identificationModule"]["briefTitle"] = "Conflicting synthetic title"
    second = {"totalCount": 2, "studies": [conflict]}

    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([_payload_bytes(first), _payload_bytes(second)])

    _assert_adapter_error(caught.value)


@pytest.mark.parametrize(
    ("brief", "official", "expected_title", "metadata_keys"),
    (
        (None, "Synthetic official", "Synthetic official", {"official_title"}),
        ("Synthetic brief", None, "Synthetic brief", {"brief_title"}),
        ("https://example.org/article", "Synthetic official", "Synthetic official", {"official_title"}),
        ("Synthetic brief", "https://example.org/article", "Synthetic brief", {"brief_title"}),
        ("x" * 1_025, "Synthetic official", "Synthetic official", {"official_title"}),
        ("Synthetic brief", "x" * 4_097, "Synthetic brief", {"brief_title"}),
    ),
)
@pytest.mark.asyncio
async def test_clinicaltrials_title_candidates_drop_independently(
    brief: str | None, official: str | None, expected_title: str, metadata_keys: set[str]
) -> None:
    payload = _single_record_payload()
    identification = payload["studies"][0]["protocolSection"]["identificationModule"]
    identification.pop("briefTitle", None)
    identification.pop("officialTitle", None)
    if brief is not None:
        identification["briefTitle"] = brief
    if official is not None:
        identification["officialTitle"] = official

    result, _dispatch, _group = await _invoke_clinical_bodies([_payload_bytes(payload)])
    record = result.candidates[0].record

    assert record["title"] == expected_title
    assert set(record["source_metadata"]) & {"brief_title", "official_title"} == metadata_keys


@pytest.mark.parametrize(
    ("brief", "official"),
    (
        (None, None),
        ("https://example.org/article", None),
        ("x" * 1_025, None),
        ("<b>unsafe</b>", "javascript:unsafe"),
    ),
)
@pytest.mark.asyncio
async def test_clinicaltrials_requires_one_safe_bounded_title(brief: str | None, official: str | None) -> None:
    payload = _single_record_payload()
    identification = payload["studies"][0]["protocolSection"]["identificationModule"]
    identification.pop("briefTitle", None)
    identification.pop("officialTitle", None)
    if brief is not None:
        identification["briefTitle"] = brief
    if official is not None:
        identification["officialTitle"] = official

    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([_payload_bytes(payload)])

    _assert_adapter_error(caught.value)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("protocolSection", "descriptionModule"), []),
        (("protocolSection", "descriptionModule", "briefSummary"), 1),
        (("protocolSection", "statusModule"), []),
        (("protocolSection", "statusModule", "overallStatus"), 1),
        (("protocolSection", "statusModule", "startDateStruct"), []),
        (("protocolSection", "statusModule", "startDateStruct", "date"), 2026),
        (("protocolSection", "conditionsModule"), []),
        (("protocolSection", "conditionsModule", "conditions"), "Synthetic"),
        (("protocolSection", "conditionsModule", "conditions"), [1]),
        (("protocolSection", "armsInterventionsModule"), []),
        (("protocolSection", "armsInterventionsModule", "interventions"), {}),
        (("protocolSection", "armsInterventionsModule", "interventions"), [1]),
        (("protocolSection", "armsInterventionsModule", "interventions"), [{"name": 1}]),
        (("protocolSection", "sponsorCollaboratorsModule"), []),
        (("protocolSection", "sponsorCollaboratorsModule", "leadSponsor"), []),
        (("protocolSection", "sponsorCollaboratorsModule", "leadSponsor", "name"), 1),
        (("protocolSection", "designModule"), []),
        (("protocolSection", "designModule", "studyType"), 1),
        (("hasResults",), "false"),
    ),
)
@pytest.mark.asyncio
async def test_clinicaltrials_wrong_optional_containers_scalars_and_members_fail(
    path: tuple[str, ...], value: object
) -> None:
    payload = _single_record_payload()
    _set_path(payload["studies"][0], path, value)

    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([_payload_bytes(payload)])

    _assert_adapter_error(caught.value)


@pytest.mark.parametrize(
    ("path", "value", "dropped_key"),
    (
        (("protocolSection", "descriptionModule", "briefSummary"), "x" * 16_385, "abstract"),
        (("protocolSection", "descriptionModule", "briefSummary"), "plain https://example.org/article", "abstract"),
        (("protocolSection", "statusModule", "overallStatus"), "x" * 257, "overall_status"),
        (("protocolSection", "conditionsModule", "conditions"), ["x"] * 65, "conditions"),
        (("protocolSection", "conditionsModule", "conditions"), ["x" * 513], "conditions"),
        (("protocolSection", "armsInterventionsModule", "interventions"), [{"name": "x"}] * 65, "interventions"),
        (("protocolSection", "armsInterventionsModule", "interventions"), [{"name": "x" * 513}], "interventions"),
        (("protocolSection", "sponsorCollaboratorsModule", "leadSponsor", "name"), "x" * 1_025, "lead_sponsor"),
        (("protocolSection", "designModule", "studyType"), "x" * 257, "study_type"),
        (("protocolSection", "statusModule", "startDateStruct", "date"), "2026-13", "start_date"),
        (("protocolSection", "statusModule", "completionDateStruct", "date"), "2026-02-30", "completion_date"),
    ),
)
@pytest.mark.asyncio
async def test_clinicaltrials_overbound_or_unsafe_optional_fields_drop_whole_field(
    path: tuple[str, ...], value: object, dropped_key: str
) -> None:
    payload = _single_record_payload()
    _set_path(payload["studies"][0], path, value)

    result, _dispatch, _group = await _invoke_clinical_bodies([_payload_bytes(payload)])
    record = result.candidates[0].record

    if dropped_key == "abstract":
        assert record["abstract"] is None
        assert record["snippet"] is None
    else:
        assert dropped_key not in record["source_metadata"]


@pytest.mark.parametrize(
    ("body", "content_type", "expected_code"),
    (
        (b"{", "application/json", "provider_payload_invalid"),
        (b"not-json", "application/json", "provider_payload_invalid"),
        (b"{}", "text/plain", "provider_response_rejected"),
        (b"{}", None, "provider_response_rejected"),
    ),
)
@pytest.mark.asyncio
async def test_clinicaltrials_malformed_or_non_json_response_fails_typed(
    body: bytes, content_type: str | None, expected_code: str
) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([body], content_type=content_type)

    _assert_adapter_error(caught.value, expected_code)


@pytest.mark.asyncio
async def test_clinicaltrials_http_429_timeout_and_redirect_are_typed_and_not_retried() -> None:
    with pytest.raises(Exception) as rate_limited:
        await _invoke_clinical_bodies([b"not-json"], status_code=429, content_type=None, retry_after="120")
    _assert_adapter_error(rate_limited.value, "provider_rate_limited")
    assert rate_limited.value.retry_after == "120"

    timeout = DiscoveryExecutionError("gateway_timed_out")
    with pytest.raises(DiscoveryExecutionError) as timed_out:
        await _invoke_clinical_bodies([timeout])
    assert timed_out.value is timeout

    with pytest.raises(Exception) as redirected:
        await _invoke_clinical_bodies([b"not-json"], status_code=302, content_type=None)
    _assert_adapter_error(redirected.value, "provider_response_rejected")


@pytest.mark.parametrize("exception_type", (KeyError, TypeError, ValueError, OverflowError))
@pytest.mark.asyncio
async def test_clinicaltrials_unexpected_builtin_parser_errors_collapse_to_payload_invalid(
    monkeypatch: pytest.MonkeyPatch, exception_type: type[Exception]
) -> None:
    error = exception_type("synthetic parser failure")

    def failing_page(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(_module(), "_clinicaltrials_page", failing_page)
    with pytest.raises(DiscoveryAdapterError) as caught:
        await _invoke_clinical_bodies([_fixture_bytes("empty")])

    _assert_adapter_error(caught.value)


@pytest.mark.parametrize(
    "error",
    (
        DiscoveryExecutionError("gateway_timed_out"),
        DiscoveryAdapterError("provider_response_rejected"),
    ),
    ids=("execution", "adapter"),
)
@pytest.mark.asyncio
async def test_clinicaltrials_framework_errors_from_parser_path_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def failing_page(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(_module(), "_clinicaltrials_page", failing_page)
    with pytest.raises(type(error)) as caught:
        await _invoke_clinical_bodies([_fixture_bytes("empty")])

    assert caught.value is error


@pytest.mark.parametrize(
    ("body", "clock", "expected_code"),
    (
        (b"x" * 2_097_153, _CountingClock(), "provider_parse_limit_exceeded"),
        (
            _payload_bytes({"totalCount": 0, "studies": [], "unknown": [[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]}),
            _CountingClock(),
            "provider_parse_limit_exceeded",
        ),
        (
            _payload_bytes({"totalCount": 0, "studies": [], "unknown": [None] * 50_001}),
            _CountingClock(),
            "provider_parse_limit_exceeded",
        ),
        (
            _payload_bytes({"totalCount": 0, "studies": [], "unknown": "x" * 65_537}),
            _CountingClock(),
            "provider_parse_limit_exceeded",
        ),
        (
            b'{"totalCount":111111111111111111111111111111111,"studies":[]}',
            _CountingClock(),
            "provider_parse_limit_exceeded",
        ),
        (_fixture_bytes("empty"), _CountingClock(step=1.0), "provider_parse_deadline_exceeded"),
    ),
)
@pytest.mark.asyncio
async def test_clinicaltrials_byte_depth_node_string_numeric_and_deadline_limits(
    body: bytes, clock: MonotonicClock, expected_code: str
) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke_clinical_bodies([body], clock=clock)

    _assert_adapter_error(caught.value, expected_code)


@pytest.mark.asyncio
async def test_clinicaltrials_page_two_failure_publishes_no_candidate_through_executor() -> None:
    registry, plan = _clinical_plan(result_limit=100)
    calls = 0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal calls
        assert is_policy_active(route.route_id, route.policy.policy_digest)
        calls += 1
        body = _fixture_bytes("success_page_1") if calls == 1 else b"{"
        return _clinical_response(route, intent, body)

    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=_clinicaltrials_test_adapters(_CountingClock()),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("clinical-atomic-1", "clinical-atomic-2")).__next__,
    )

    assert calls == 2
    assert execution.candidates == ()
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert execution.logical_outcomes[0].code == "provider_payload_invalid"
    assert execution.usage.pages == 2
    assert execution.usage.accounting.created == execution.usage.accounting.debited == 2


@pytest.mark.asyncio
async def test_clinicaltrials_hundred_raw_ceiling_is_exactly_two_debits_and_no_third_reservation() -> None:
    registry, plan = _clinical_plan(result_limit=100)
    page_one = {"totalCount": 101, "studies": [_minimal_study(index) for index in range(50)]}
    page_one["nextPageToken"] = "synthetic-ceiling-one"
    page_two = {"totalCount": 101, "studies": [_minimal_study(index + 50) for index in range(50)]}
    page_two["nextPageToken"] = "synthetic-ceiling-two"
    bodies = iter((_payload_bytes(page_one), _payload_bytes(page_two)))
    calls = 0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal calls
        calls += 1
        return _clinical_response(route, intent, next(bodies))

    journal = AttemptJournal(physical_ceiling=2)
    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=_clinicaltrials_test_adapters(_CountingClock()),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("clinical-ceiling-1", "clinical-ceiling-2", "must-not-be-used")).__next__,
        journal=journal,
    )

    assert calls == 2
    assert len(execution.candidates) == 100
    assert execution.usage.pages == 2
    assert journal.accounting.created == 2
    assert journal.accounting.debited == 2
    assert journal.accounting.released == journal.accounting.outstanding == 0


@pytest.mark.asyncio
async def test_clinicaltrials_cancellation_after_page_parse_has_one_debit_and_no_continuation_call() -> None:
    registry, plan = _clinical_plan(result_limit=100)
    parser_clock = _CountingClock()
    calls = 0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal calls
        calls += 1
        return _clinical_response(route, intent, _fixture_bytes("success_page_1"))

    journal = AttemptJournal(physical_ceiling=2)
    execution = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=_clinicaltrials_test_adapters(parser_clock),
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("clinical-cancel-1", "must-not-be-used")).__next__,
        journal=journal,
        cancellation_check=lambda: parser_clock.calls >= 1,
    )

    assert calls == 1
    assert execution.candidates == ()
    assert execution.logical_outcomes[0].state is LogicalOutcomeState.CANCELLED
    assert execution.usage.pages == 1
    assert journal.accounting.created == 1
    assert journal.accounting.debited == 1
    assert journal.accounting.released == 0
    assert journal.accounting.outstanding == 0


def test_clinicaltrials_fixtures_are_exact_wholly_synthetic_shapes() -> None:
    first = _fixture_payload("success_page_1")
    second = _fixture_payload("success_page_2")
    empty_bytes = _fixture_bytes("empty")
    serialized = json.dumps((first, second), sort_keys=True)

    assert empty_bytes == b'{"totalCount":0,"studies":[]}\n'
    assert (first["totalCount"], second["totalCount"]) == (2, 2)
    assert tuple(len(payload["studies"]) for payload in (first, second)) == (1, 1)
    assert first["nextPageToken"] == "synthetic+token/one=="
    assert "nextPageToken" not in second
    assert tuple(
        payload["studies"][0]["protocolSection"]["identificationModule"]["nctId"] for payload in (first, second)
    ) == ("NCT90000001", "NCT90000002")
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "mailto:" not in serialized
    assert "contact@" not in serialized
    assert "location" not in serialized.casefold()
    assert "document" not in serialized.casefold()
