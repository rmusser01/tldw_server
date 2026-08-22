"""Offline contract tests for the explicit two-dispatch PubMed adapter."""

from __future__ import annotations

import asyncio
import http.client
import importlib
import json
import socket
import urllib.request
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from tldw_Server_API.app.core.Research.discovery import executor as executor_module
from tldw_Server_API.app.core.Research.discovery.clinicaltrials_pubmed_central import (
    clinicaltrials_pubmed_central_shadow_registry,
)
from tldw_Server_API.app.core.Research.discovery.contracts import (
    BudgetCeilings,
    DiscoveryOutcomeIdentity,
    DispatchAllowance,
    ExactOrigin,
    ExecutionMode,
    OperationKind,
    QueryPair,
)
from tldw_Server_API.app.core.Research.discovery.executor import (
    DiscoveryAdapterResult,
    LogicalOutcomeState,
    NumericCSVBindingValues,
    PhysicalDispatchState,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
    dispatch_once,
)
from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint
from tldw_Server_API.app.core.Research.discovery.planner import (
    PlanningRequest,
    compile_discovery_plan,
    expected_dispatch_group_id,
    expected_logical_attempt_id,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)
from tldw_Server_API.app.core.Security.http_hop import (
    HTTPHopLimits,
    HTTPHopResponse,
    NormalizedHTTPHopRequest,
)

pytestmark = pytest.mark.unit

_ADAPTER_ID = "pubmed_v2"
_ADAPTER_MODULE = "tldw_Server_API.app.core.Research.discovery.gateway_adapters"
_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "research_discovery_gateway_adapters"
_PMIDS = ("31415926", "27182818")
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
}


class _StringSubclass(str):
    pass


def _module():
    return importlib.import_module(_ADAPTER_MODULE)


def _fixture(kind: str) -> bytes:
    return (_FIXTURE_ROOT / f"pubmed_{kind}.json").read_bytes()


def _registry_with_response_limit(max_response_bytes: int | None) -> DiscoveryRegistry:
    base = foundation_registry()
    if max_response_bytes is None:
        return base
    route_id = base.get_source("pubmed").route_references[0].route_id
    routes = []
    for route in base.routes:
        if route.route_id != route_id:
            routes.append(route)
            continue
        limits = replace(route.policy.limits, max_response_bytes=max_response_bytes)
        routes.append(
            replace(
                route,
                policy=replace(route.policy, limits=limits, policy_digest=""),
            )
        )
    return DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version=base.registry_version,
        sources=base.sources,
        routes=tuple(routes),
        backends=base.backends,
    )


def _plan_for(
    *,
    result_limit: int = 2,
    max_response_bytes: int | None = None,
    filters: tuple[QueryPair, ...] = (),
):
    registry = _registry_with_response_limit(max_response_bytes)
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=("pubmed",),
            query="  BOUNDED   Discovery  ",
            filters=filters,
            result_limit=result_limit,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(
            max_route_attempts=1,
            max_physical_dispatches=2,
            max_pages_per_route=1,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=40_000,
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
    content_type: str | None = "application/json; charset=UTF-8",
    retry_after: Any = None,
    headers: tuple[tuple[Any, Any], ...] | None = None,
) -> DiscoveryGatewayResponse:
    origin = route.policy.origin
    if headers is None:
        headers = () if content_type is None else (("content-type", content_type),)
    body_length = len(body) if hasattr(body, "__len__") else 0
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
        self.calls += 1
        self.value += self.step
        return current


class _DelayedStepClock:
    def __init__(self, *, delay_calls: int, step: float) -> None:
        self.delay_calls = delay_calls
        self.step = step
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls <= self.delay_calls:
            return 0.0
        return (self.calls - self.delay_calls) * self.step


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _esearch(
    ids: tuple[str, ...],
    *,
    count: object | None = None,
    retmax: object | None = None,
    retstart: object = "0",
    additions: dict[str, object] | None = None,
) -> bytes:
    result: dict[str, object] = {
        "count": str(len(ids)) if count is None else count,
        "retmax": str(len(ids)) if retmax is None else retmax,
        "retstart": retstart,
        "idlist": list(ids),
    }
    if additions:
        result.update(additions)
    return _json(
        {
            "header": {"type": "esearch", "version": "0.3"},
            "esearchresult": result,
        }
    )


def _summary_record(
    pmid: str,
    *,
    title: str | None = None,
    authors: object | None = None,
    articleids: object | None = None,
) -> dict[str, object]:
    return {
        "uid": pmid,
        "title": title or f"Bounded record {pmid}",
        "authors": [] if authors is None else authors,
        "articleids": ([{"idtype": "pubmed", "value": pmid}] if articleids is None else articleids),
    }


def _esummary(
    ids: tuple[str, ...],
    *,
    uids: object | None = None,
    records: dict[str, object] | None = None,
    additions: dict[str, object] | None = None,
) -> bytes:
    result: dict[str, object] = {"uids": list(ids) if uids is None else uids}
    result.update(records or {pmid: _summary_record(pmid) for pmid in ids})
    if additions:
        result.update(additions)
    return _json(
        {
            "header": {"type": "esummary", "version": "0.3"},
            "result": result,
        }
    )


async def _invoke(
    bodies: list[Any],
    *,
    result_limit: int = 2,
    max_response_bytes: int | None = None,
    filters: tuple[QueryPair, ...] = (),
    statuses: list[object] | None = None,
    content_types: list[str | None] | None = None,
    retry_afters: list[object] | None = None,
    monotonic_clock=None,
):
    registry, plan = _plan_for(
        result_limit=result_limit,
        max_response_bytes=max_response_bytes,
        filters=filters,
    )
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    statuses = statuses or [200] * len(bodies)
    content_types = content_types or ["application/json; charset=UTF-8"] * len(bodies)
    retry_afters = retry_afters or [None] * len(bodies)
    responses = [
        _response(
            route,
            group.intents[min(index, len(group.intents) - 1)],
            body,
            status_code=statuses[index],
            content_type=content_types[index],
            retry_after=retry_afters[index],
        )
        for index, body in enumerate(bodies)
    ]
    dispatch = _RecordingDispatch(responses)
    clock = _CountingClock() if monotonic_clock is None else monotonic_clock
    adapter = _module().foundation_gateway_adapters(monotonic_clock=clock)[_ADAPTER_ID]
    result = await adapter(group, dispatch)
    return result, dispatch, group


async def _execute(responses: list[object], *, result_limit: int = 2, monotonic_clock=None):
    registry, plan = _plan_for(result_limit=result_limit)
    queued = list(responses)
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append((route, intent))
        response = queued.pop(0)
        if isinstance(response, BaseException):
            raise response
        body, status_code, content_type, retry_after = response
        return _response(
            route,
            intent,
            body,
            status_code=status_code,
            content_type=content_type,
            retry_after=retry_after,
        )

    clock = _CountingClock() if monotonic_clock is None else monotonic_clock
    adapter = _module().foundation_gateway_adapters(monotonic_clock=clock)[_ADAPTER_ID]
    dispatch_ids = iter(("pubmed-esearch-dispatch", "pubmed-esummary-dispatch"))
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={_ADAPTER_ID: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
    )
    return result, gateway_calls


def _assert_typed_error(error: BaseException, code: str) -> None:
    error_type = executor_module.DiscoveryAdapterError
    assert type(error) is error_type
    assert error.code == code
    assert str(error) == code


def _normalized(candidate) -> dict[str, Any]:
    record = dict(candidate.record)
    record["provider_ids"] = dict(record["provider_ids"])
    return record


def test_pubmed_profile_and_factory_registration_are_exact_and_immutable() -> None:
    module = _module()
    profiles = module._PARSING_PROFILES

    assert type(profiles) is MappingProxyType
    assert ("pubmed_v2", "foundation-v2") in profiles
    profile = profiles[("pubmed_v2", "foundation-v2")]
    assert (
        profile.max_input_bytes,
        profile.max_records,
        profile.max_depth,
        profile.max_nodes,
        profile.max_string_chars,
        profile.max_numeric_token_chars,
        profile.parse_deadline_ms,
    ) == (2_097_152, 100, 16, 50_000, 65_536, 32, 500)
    with pytest.raises(FrozenInstanceError):
        profile.max_records = 101

    adapters = module.foundation_gateway_adapters()
    assert tuple(adapters)[-1] == _ADAPTER_ID
    assert callable(adapters[_ADAPTER_ID])
    assert module._MAX_PUBMED_AUTHORS_PER_RECORD == 1_024
    assert module._MAX_PUBMED_ARTICLE_IDS_PER_RECORD == 64


def test_planner_freezes_two_exact_requests_and_explicit_relevance_sort() -> None:
    registry, plan = _plan_for(result_limit=7)
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    search, summary = group.intents

    assert group.adapter_id == _ADAPTER_ID
    assert group.adapter_version == "foundation-v2"
    assert route.policy.origin.host == "eutils.ncbi.nlm.nih.gov"
    assert search.operation_kind is OperationKind.SEARCH
    assert search.path == "/entrez/eutils/esearch.fcgi"
    assert tuple((pair.name, pair.value) for pair in search.query_pairs) == (
        ("db", "pubmed"),
        ("term", "bounded discovery"),
        ("retstart", "0"),
        ("retmax", "7"),
        ("retmode", "json"),
        ("sort", "relevance"),
    )
    assert summary.operation_kind is OperationKind.CONDITIONAL_SUMMARY
    assert summary.path == "/entrez/eutils/esummary.fcgi"
    assert tuple((pair.name, pair.value) for pair in summary.query_pairs) == (
        ("db", "pubmed"),
        ("retmode", "json"),
    )
    assert len(summary.query_bindings) == 1
    binding = summary.query_bindings[0]
    assert (binding.binding_id, binding.query_name, binding.max_items, binding.max_item_chars) == (
        "pubmed_esearch_ids",
        "id",
        7,
        16,
    )


@pytest.mark.asyncio
async def test_sanitized_success_fixtures_normalize_exact_records_and_bind_ids() -> None:
    result, dispatch, group = await _invoke([_fixture("esearch_success"), _fixture("esummary_success")])

    assert type(result) is DiscoveryAdapterResult
    assert len(result.candidates) == 2
    first = _normalized(result.candidates[0])
    expected = {
        "title": "Shared Discovery Record",
        "authors": ("Ada Researcher", "Grace Scientist"),
        "abstract": None,
        "snippet": None,
        "doi": "10.5555/shared.discovery.2026",
        "pmid": "31415926",
        "pmcid": "PMC3141592",
        "arxiv_id": None,
        "url": "https://pubmed.ncbi.nlm.nih.gov/31415926/",
        "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3141592/pdf/",
        "provider": "pubmed",
        "provider_ids": {
            "pubmed_id": "31415926",
            "pmid": "31415926",
            "doi": "10.5555/shared.discovery.2026",
            "pmcid": "PMC3141592",
        },
    }
    assert first == expected
    assert set(first) == _NORMALIZED_KEYS
    assert (
        result.candidates[0].candidate_id
        == DiscoveryOutcomeIdentity.from_fingerprint(build_fingerprint(expected)).document_id
    )
    assert _normalized(result.candidates[1]) == {
        "title": "Second Bounded Record",
        "authors": ("Lin Researcher",),
        "abstract": None,
        "snippet": None,
        "doi": None,
        "pmid": "27182818",
        "pmcid": None,
        "arxiv_id": None,
        "url": "https://pubmed.ncbi.nlm.nih.gov/27182818/",
        "pdf_url": None,
        "provider": "pubmed",
        "provider_ids": {"pubmed_id": "27182818", "pmid": "27182818"},
    }
    assert [call[0] for call in dispatch.calls] == list(group.intents)
    assert dispatch.calls[0][1:] == (None, ())
    assert dispatch.calls[1][1] is None
    assert dispatch.calls[1][2] == (NumericCSVBindingValues("pubmed_esearch_ids", (31415926, 27182818)),)


@pytest.mark.asyncio
async def test_empty_esearch_returns_without_summary_dispatch_or_reservation() -> None:
    result, dispatch, _group = await _invoke([_fixture("esearch_empty")])

    assert result == DiscoveryAdapterResult(candidates=())
    assert len(dispatch.calls) == 1


@pytest.mark.asyncio
async def test_executor_accounts_search_and_conditional_summary_as_distinct_hops() -> None:
    result, gateway_calls = await _execute(
        [
            (_fixture("esearch_success"), 200, "application/json", None),
            (_fixture("esummary_success"), 200, "application/json", None),
        ]
    )

    assert len(result.candidates) == 2
    assert result.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert [intent.path for _route, intent in gateway_calls] == [
        "/entrez/eutils/esearch.fcgi",
        "/entrez/eutils/esummary.fcgi",
    ]
    summary_pairs = tuple((pair.name, pair.value) for pair in gateway_calls[1][1].query_pairs)
    assert summary_pairs == (
        ("db", "pubmed"),
        ("retmode", "json"),
        ("id", "31415926,27182818"),
    )
    assert gateway_calls[1][1].query_bindings == ()
    assert tuple(record.dispatch_id for record in result.usage.physical_records) == (
        "pubmed-esearch-dispatch",
        "pubmed-esummary-dispatch",
    )
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.pages == 1
    assert result.usage.accounting.created == result.usage.accounting.debited == 2


@pytest.mark.asyncio
async def test_executor_empty_search_creates_only_one_dispatch_id() -> None:
    result, gateway_calls = await _execute([(_fixture("esearch_empty"), 200, "application/json", None)])

    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.VALID_EMPTY
    assert len(gateway_calls) == 1
    assert tuple(record.dispatch_id for record in result.usage.physical_records) == ("pubmed-esearch-dispatch",)
    assert result.usage.accounting.created == result.usage.accounting.debited == 1


def _overlay_plan_for(
    *,
    result_limit: int = 2,
    filters: tuple[QueryPair, ...] = (),
):
    registry = clinicaltrials_pubmed_central_shadow_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=("pubmed",),
            query="  BOUNDED   Discovery  ",
            filters=filters,
            result_limit=result_limit,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(1, 2, 1, 0, 0, 40_000, result_limit),
    )
    return registry, plan


def _aligned_overlay_group(group, **changes):
    """Apply coherent group fields to every intent carrying that field."""
    intent_changes = {name: changes[name] for name in ("route_id", "policy_digest", "limits") if name in changes}
    intents = tuple(replace(intent, **intent_changes) for intent in group.intents)
    return replace(group, intents=intents, **changes)


def _mutated_overlay_group(mutation: str):
    _registry, plan = _overlay_plan_for()
    group = plan.dispatch_groups[0]
    if mutation == "route_id":
        return _aligned_overlay_group(group, route_id="forged_pubmed_route")
    if mutation == "backend_id":
        return replace(group, backend_id="arxiv_api")
    if mutation == "policy_digest":
        return _aligned_overlay_group(group, policy_digest="f" * 64)
    if mutation == "limits":
        limits = replace(group.limits, timeout_ms=19_999)
        return _aligned_overlay_group(group, limits=limits)
    if mutation == "allowance":
        return replace(group, allowance=DispatchAllowance(3, 1, 0, 0))
    if mutation == "logical_source":
        attempts = tuple(replace(attempt, catalog_source_id="pubmed_forged") for attempt in group.logical_attempts)
        return replace(group, logical_attempts=attempts)
    if mutation == "filters":
        object.__setattr__(group, "filters", (object(),))
        return group
    if mutation == "filters_container":
        object.__setattr__(group, "filters", [QueryPair("year", "2025")])
        return group
    if mutation in {"filters_tool", "filters_email"}:
        name = "tool" if mutation == "filters_tool" else "email"
        object.__setattr__(group, "filters", (QueryPair(name, "attacker"),))
        return group
    if mutation in {"filters_name_type", "filters_value_type"}:
        pair = QueryPair("year", "2025")
        field = "name" if mutation == "filters_name_type" else "value"
        object.__setattr__(pair, field, _StringSubclass(getattr(pair, field)))
        object.__setattr__(group, "filters", (pair,))
        return group
    if mutation == "fallback_order":
        return replace(group, fallback_order=1)
    if mutation == "intent_route_alignment":
        object.__setattr__(group.intents[0], "route_id", "forged_pubmed_route")
        return group
    if mutation == "intent_policy_alignment":
        object.__setattr__(group.intents[0], "policy_digest", "f" * 64)
        return group
    if mutation == "intent_limits_alignment":
        object.__setattr__(group.intents[0], "limits", replace(group.limits, timeout_ms=19_999))
        return group
    if mutation == "intent_limits_scalar_type":
        forged_limits = replace(group.limits)
        object.__setattr__(forged_limits, "max_pages", True)
        object.__setattr__(group.intents[0], "limits", forged_limits)
        return group
    if mutation == "normalized_query_alignment":
        search = group.intents[0]
        query_pairs = list(search.query_pairs)
        query_pairs[1] = QueryPair("term", "forged query")
        object.__setattr__(search, "query_pairs", tuple(query_pairs))
        return group
    raise AssertionError(f"unknown test mutation: {mutation}")


def _forge_plan_group_for_policy(plan, policy_digest: str):
    """Rebuild deterministic IDs for one otherwise coherent hostile plan."""
    group = _aligned_overlay_group(plan.dispatch_groups[0], policy_digest=policy_digest)
    group_id = expected_dispatch_group_id(group)
    logical_attempts = tuple(
        replace(
            attempt,
            logical_attempt_id=expected_logical_attempt_id(attempt, group_id),
        )
        for attempt in group.logical_attempts
    )
    return replace(group, dispatch_group_id=group_id, logical_attempts=logical_attempts)


@pytest.mark.parametrize(
    "mutation",
    (
        "route_id",
        "backend_id",
        "policy_digest",
        "limits",
        "allowance",
        "logical_source",
        "filters",
        "filters_container",
        "filters_tool",
        "filters_email",
        "filters_name_type",
        "filters_value_type",
        "fallback_order",
        "intent_route_alignment",
        "intent_policy_alignment",
        "intent_limits_alignment",
        "intent_limits_scalar_type",
        "normalized_query_alignment",
    ),
)
@pytest.mark.asyncio
async def test_identity_overlay_adapter_rejects_group_trust_drift_before_dispatch(mutation: str) -> None:
    group = _mutated_overlay_group(mutation)
    dispatch = _RecordingDispatch([])

    with pytest.raises(executor_module.DiscoveryAdapterError) as caught:
        await _module().foundation_gateway_adapters()[_ADAPTER_ID](group, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert dispatch.calls == []


@pytest.mark.asyncio
async def test_identity_overlay_preserves_legacy_non_identity_filters() -> None:
    filters = (QueryPair("year", "2025"), QueryPair("language", "eng"))
    canonical_filters = tuple(sorted(filters, key=lambda pair: (pair.name, pair.value)))
    foundation_result, _foundation_dispatch, foundation_group = await _invoke(
        [_fixture("esearch_success"), _fixture("esummary_success")],
        filters=filters,
    )
    registry, plan = _overlay_plan_for(filters=filters)
    overlay_group = plan.dispatch_groups[0]
    route = registry.get_route(overlay_group.route_id)
    overlay_dispatch = _RecordingDispatch(
        [
            _response(route, overlay_group.intents[0], _fixture("esearch_success")),
            _response(route, overlay_group.intents[1], _fixture("esummary_success")),
        ]
    )

    overlay_result = await _module().foundation_gateway_adapters()[_ADAPTER_ID](
        overlay_group,
        overlay_dispatch,
    )

    assert foundation_group.filters == overlay_group.filters == canonical_filters
    assert tuple(candidate.record for candidate in overlay_result.candidates) == tuple(
        candidate.record for candidate in foundation_result.candidates
    )
    assert len(overlay_dispatch.calls) == 2


@pytest.mark.asyncio
async def test_forged_public_origin_registry_and_plan_stop_before_gateway_or_one_hop() -> None:
    registry, plan = _overlay_plan_for()
    route = registry.get_route("pubmed_ncbi_eutils_pubmed_direct")
    forged_policy = replace(
        route.policy,
        origin=ExactOrigin("https", "attacker.example", 443),
        policy_digest="",
    )
    forged_route = replace(route, policy=forged_policy)
    forged_registry = replace(
        registry,
        routes=tuple(
            forged_route if candidate.route_id == route.route_id else candidate for candidate in registry.routes
        ),
    )
    forged_group = _forge_plan_group_for_policy(plan, forged_policy.policy_digest)
    forged_plan = replace(plan, dispatch_groups=(forged_group,), plan_digest="")
    gateway_calls: list[object] = []
    one_hop_calls: list[NormalizedHTTPHopRequest] = []

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        one_hop_calls.append(request)
        body = _fixture("esearch_empty")
        return HTTPHopResponse(
            status_code=200,
            headers=(("content-type", "application/json"),),
            body=body,
            resolved_ips=("93.184.216.34",),
            connected_ip="93.184.216.34",
            response_header_bytes=64,
            wire_bytes=len(body),
        )

    async def gateway(candidate_route, intent, *, is_policy_active):
        gateway_calls.append(intent)
        return await dispatch_once(
            candidate_route,
            intent,
            is_policy_active=is_policy_active,
            one_hop=one_hop,
        )

    result = await execute_discovery_plan(
        forged_plan,
        registry=forged_registry,
        adapters={_ADAPTER_ID: _module().foundation_gateway_adapters()[_ADAPTER_ID]},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("forged-public-origin-dispatch",)).__next__,
    )

    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "adapter_failed"
    assert gateway_calls == []
    assert one_hop_calls == []
    assert result.usage.accounting.created == result.usage.accounting.debited == 0


async def _execute_overlay_via_one_hop(bodies: list[bytes]):
    registry, plan = _overlay_plan_for()
    adapter = _module().foundation_gateway_adapters()[_ADAPTER_ID]
    raw_bodies = list(bodies)
    requests: list[NormalizedHTTPHopRequest] = []
    responses: list[DiscoveryGatewayResponse] = []

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        requests.append(request)
        body = raw_bodies.pop(0)
        return HTTPHopResponse(
            status_code=200,
            headers=(("content-type", "application/json"),),
            body=body,
            resolved_ips=("93.184.216.34",),
            connected_ip="93.184.216.34",
            response_header_bytes=64,
            wire_bytes=len(body),
        )

    async def gateway(route, intent, *, is_policy_active):
        response = await dispatch_once(route, intent, is_policy_active=is_policy_active, one_hop=one_hop)
        responses.append(response)
        return response

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={_ADAPTER_ID: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("overlay-search", "overlay-summary")).__next__,
    )
    return result, requests, responses


@pytest.mark.asyncio
async def test_identity_overlay_executes_exact_two_hop_identity_shape_without_repr_leaks() -> None:
    registry, plan = _overlay_plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    dispatch = _RecordingDispatch(
        [
            _response(route, group.intents[0], _fixture("esearch_success")),
            _response(route, group.intents[1], _fixture("esummary_success")),
        ]
    )

    result = await _module().foundation_gateway_adapters()[_ADAPTER_ID](group, dispatch)

    assert len(result.candidates) == 2
    assert [tuple((pair.name, pair.value) for pair in call[0].query_pairs) for call in dispatch.calls] == [
        (
            ("db", "pubmed"),
            ("term", "bounded discovery"),
            ("retstart", "0"),
            ("retmax", "2"),
            ("retmode", "json"),
            ("sort", "relevance"),
            ("tool", "tldw_server"),
            ("email", "contact@tldwproject.com"),
        ),
        (
            ("db", "pubmed"),
            ("retmode", "json"),
            ("tool", "tldw_server"),
            ("email", "contact@tldwproject.com"),
        ),
    ]
    assert "tldw_server" not in repr(group)
    assert "contact@tldwproject.com" not in repr(group)
    assert "31415926" not in repr(dispatch.calls[1][2])


@pytest.mark.parametrize("summary_stage", (False, True))
@pytest.mark.asyncio
async def test_identity_overlay_accepts_only_the_documented_json_rate_envelope(summary_stage: bool) -> None:
    registry, plan = _overlay_plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    rate_limited = _response(
        route,
        group.intents[1 if summary_stage else 0],
        _json({"error": "API rate limit exceeded", "count": "11"}),
    )
    responses: list[object] = [rate_limited]
    if summary_stage:
        responses = [_response(route, group.intents[0], _fixture("esearch_success")), rate_limited]
    dispatch = _RecordingDispatch(responses)

    with pytest.raises(executor_module.DiscoveryAdapterError, match="provider_rate_limited"):
        await _module().foundation_gateway_adapters()[_ADAPTER_ID](group, dispatch)


@pytest.mark.asyncio
async def test_identity_overlay_executes_through_executor_gateway_and_one_hop_with_foundation_output() -> None:
    foundation_result, _dispatch, _group = await _invoke([_fixture("esearch_success"), _fixture("esummary_success")])
    result, requests, responses = await _execute_overlay_via_one_hop(
        [_fixture("esearch_success"), _fixture("esummary_success")]
    )

    assert tuple(candidate.record for candidate in result.candidates) == tuple(
        candidate.record for candidate in foundation_result.candidates
    )
    assert result.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert result.usage.pages == 1
    assert result.usage.accounting.created == result.usage.accounting.debited == 2
    assert [response.trace.query_keys for response in responses] == [
        ("db", "term", "retstart", "retmax", "retmode", "sort", "tool", "email"),
        ("db", "retmode", "tool", "email", "id"),
    ]
    assert [request.target.split("?", 1)[0] for request in requests] == [
        "/entrez/eutils/esearch.fcgi",
        "/entrez/eutils/esummary.fcgi",
    ]
    assert all("tldw_server" not in repr(item) and "contact@tldwproject.com" not in repr(item) for item in requests)


@pytest.mark.asyncio
async def test_identity_overlay_empty_esearch_stops_after_one_executor_dispatch() -> None:
    result, requests, responses = await _execute_overlay_via_one_hop([_fixture("esearch_empty")])

    assert result.logical_outcomes[0].state is LogicalOutcomeState.VALID_EMPTY
    assert len(requests) == len(responses) == 1
    assert result.usage.accounting.created == result.usage.accounting.debited == 1


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.asyncio
async def test_identity_overlay_dispatch_cancellation_at_either_stage_propagates_unchanged(stage: int) -> None:
    registry, plan = _overlay_plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    cancelled = asyncio.CancelledError(f"identity-overlay-stage-{stage}-cancelled")
    responses: list[object] = [cancelled]
    if stage == 1:
        responses = [_response(route, group.intents[0], _fixture("esearch_success")), cancelled]
    dispatch = _RecordingDispatch(responses)

    with pytest.raises(asyncio.CancelledError) as caught:
        await _module().foundation_gateway_adapters()[_ADAPTER_ID](group, dispatch)

    assert caught.value is cancelled


@pytest.mark.asyncio
async def test_identity_overlay_rejects_conflicting_records_with_one_doi_fingerprint() -> None:
    registry, plan = _overlay_plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    ids = ("31415926", "27182818")
    same_doi = "10.5555/Shared.Discovery.2026"
    summary = _esummary(
        ids,
        records={
            ids[0]: _summary_record(
                ids[0],
                title="First conflicting DOI record",
                articleids=(
                    {"idtype": "pubmed", "value": ids[0]},
                    {"idtype": "doi", "value": same_doi},
                ),
            ),
            ids[1]: _summary_record(
                ids[1],
                title="Second conflicting DOI record",
                articleids=(
                    {"idtype": "pubmed", "value": ids[1]},
                    {"idtype": "doi", "value": same_doi},
                ),
            ),
        },
    )
    dispatch = _RecordingDispatch(
        [
            _response(route, group.intents[0], _esearch(ids)),
            _response(route, group.intents[1], summary),
        ]
    )

    with pytest.raises(executor_module.DiscoveryAdapterError) as caught:
        await _module().foundation_gateway_adapters()[_ADAPTER_ID](group, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert group.adapter_version == "pubmed-v2-ncbi-identity"
    assert [call[0] for call in dispatch.calls] == list(group.intents)
    assert dispatch.calls[1][2] == (NumericCSVBindingValues("pubmed_esearch_ids", (31415926, 27182818)),)


@pytest.mark.asyncio
async def test_shared_ncbi_helper_collapses_byte_identical_same_fingerprint_records() -> None:
    registry, plan = _overlay_plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    binding = group.intents[1].query_bindings[0]
    profile = _module()._PARSING_PROFILES[(_ADAPTER_ID, group.adapter_version)]
    normalized = {
        "title": "Synthetic byte-identical normalized record",
        "authors": ("Synthetic Author",),
        "abstract": None,
        "snippet": None,
        "doi": "10.5555/synthetic-identical-record",
        "pmid": "31415926",
        "pmcid": None,
        "arxiv_id": None,
        "url": "https://pubmed.ncbi.nlm.nih.gov/31415926/",
        "pdf_url": None,
        "provider": "pubmed",
        "provider_ids": {"pmid": "31415926", "doi": "10.5555/synthetic-identical-record"},
    }
    duplicate = dict(normalized)
    assert duplicate is not normalized
    assert duplicate == normalized
    assert _json(duplicate) == _json(normalized)

    def trusted_inputs(candidate: object):
        assert candidate is group
        return group, profile, profile.max_input_bytes, 0, 2, binding

    def parse_esearch_ids(_payload: object, **kwargs: object):
        assert kwargs == {
            "profile": profile,
            "guard": kwargs["guard"],
            "retstart": 0,
            "retmax": 2,
            "binding": binding,
        }
        return (("31415926", 31_415_926), ("27182818", 27_182_818))

    def parse_summary_records(_payload: object, **kwargs: object):
        assert kwargs["expected_ids"] == ("31415926", "27182818")
        return normalized, duplicate

    dispatch = _RecordingDispatch(
        [
            _response(route, group.intents[0], _json({"synthetic": "esearch"})),
            _response(route, group.intents[1], _json({"synthetic": "esummary"})),
        ]
    )

    result = await _module()._execute_ncbi_esearch_summary(
        group,
        dispatch,
        _CountingClock(),
        trusted_inputs=trusted_inputs,
        parse_esearch_ids=parse_esearch_ids,
        parse_summary_records=parse_summary_records,
        strict_rate_envelope=False,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].record == normalized
    assert len(dispatch.calls) == 2
    assert [call[0] for call in dispatch.calls] == list(group.intents)
    assert dispatch.calls[1][2] == (NumericCSVBindingValues("pubmed_esearch_ids", (31415926, 27182818)),)


@pytest.mark.parametrize(
    "payload",
    (
        {"error": "API rate limit exceeded", "count": "11", "extra": "x"},
        {"error": "API rate limit exceeded"},
        {"error": "API rate limit exceeded", "count": 11},
        {"error": "API rate limit exceeded!", "count": "11"},
    ),
)
@pytest.mark.parametrize("summary_stage", (False, True))
@pytest.mark.asyncio
async def test_identity_overlay_rejects_near_match_json_rate_envelopes_as_payload_invalid(
    payload: dict[str, object], summary_stage: bool
) -> None:
    registry, plan = _overlay_plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    bodies = [_json(payload)]
    if summary_stage:
        bodies = [_fixture("esearch_success"), _json(payload)]
    dispatch = _RecordingDispatch(
        [_response(route, group.intents[min(index, 1)], body) for index, body in enumerate(bodies)]
    )

    with pytest.raises(executor_module.DiscoveryAdapterError) as caught:
        await _module().foundation_gateway_adapters()[_ADAPTER_ID](group, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_short_final_esearch_page_uses_reported_retmax_not_requested_capacity() -> None:
    result, dispatch, _group = await _invoke(
        [
            _esearch(("31415926",), count="1", retmax="1"),
            _esummary(("31415926",)),
        ],
        result_limit=2,
    )

    assert len(result.candidates) == 1
    assert len(dispatch.calls) == 2


@pytest.mark.asyncio
async def test_esearch_total_above_ten_thousand_is_valid_metadata() -> None:
    result, _dispatch, _group = await _invoke(
        [
            _esearch(("31415926",), count="586404", retmax="1"),
            _esummary(("31415926",)),
        ]
    )

    assert len(result.candidates) == 1


@pytest.mark.parametrize(
    "body",
    (
        _esearch(("31415926",), count=1),
        _esearch(("31415926",), retmax=1),
        _esearch(("31415926",), retstart=0),
        _esearch(("31415926",), retmax="2"),
        _esearch(("31415926",), count="0"),
        _esearch((), count="1", retmax="0"),
        _esearch(("31415926",), retstart="1"),
        _json({"header": {"type": "esummary", "version": "0.3"}, "esearchresult": {}}),
        _json({"header": {"type": "esearch", "version": "0.3"}}),
    ),
)
@pytest.mark.asyncio
async def test_esearch_envelope_and_cursor_mismatches_reject_before_summary(body: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "ids",
    (
        ("0",),
        ("01",),
        ("-1",),
        ("+1",),
        ("１",),
        ("1,2",),
        ("12345678901234567",),
        ("31415926", "31415926"),
    ),
)
@pytest.mark.asyncio
async def test_malformed_or_duplicate_pmids_reject_before_summary(ids: tuple[str, ...]) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([_esearch(ids)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_esearch_id_cardinality_cannot_exceed_planned_binding_limit() -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([_esearch(_PMIDS)], result_limit=1)

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_esearch_fatal_error_envelope_is_sanitized() -> None:
    body = _esearch((), additions={"ERROR": "fixture-secret"})

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_response_rejected")
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.asyncio
async def test_nonempty_esearch_diagnostics_are_bounded_dropped_and_still_searchable() -> None:
    result, dispatch, _group = await _invoke([_fixture("esearch_diagnostic_success"), _fixture("esummary_success")])

    assert len(result.candidates) == 2
    assert len(dispatch.calls) == 2
    assert "sanitized absent alternative" not in repr(result)


@pytest.mark.parametrize("key", ("errorlist", "warninglist"))
@pytest.mark.asyncio
async def test_malformed_esearch_diagnostic_shape_is_rejected(key: str) -> None:
    body = _esearch((), additions={key: {"messages": "not-a-list"}})

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_esearch_warning_list_is_bounded_but_dropped_from_valid_results() -> None:
    search = _esearch(
        ("31415926",),
        additions={"warninglist": {"phrasesignored": ["fixture-secret"]}},
    )

    result, _dispatch, _group = await _invoke(
        [search, _esummary(("31415926",))],
        result_limit=1,
    )

    assert len(result.candidates) == 1
    assert "fixture-secret" not in repr(result)


@pytest.mark.parametrize(
    "summary",
    (
        _esummary(_PMIDS, uids=[_PMIDS[0]]),
        _esummary(_PMIDS, uids=[*_PMIDS, "16180339"]),
        _esummary(_PMIDS, records={_PMIDS[0]: _summary_record(_PMIDS[0])}),
        _esummary(
            _PMIDS,
            additions={"16180339": _summary_record("16180339")},
        ),
        _esummary(
            _PMIDS,
            records={
                _PMIDS[0]: _summary_record(_PMIDS[1]),
                _PMIDS[1]: _summary_record(_PMIDS[1]),
            },
        ),
    ),
)
@pytest.mark.asyncio
async def test_partial_foreign_or_mismatched_summaries_fail_atomically(summary: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([_fixture("esearch_success"), summary])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_reordered_summary_uids_preserve_esearch_result_order() -> None:
    summary = _esummary(_PMIDS, uids=list(reversed(_PMIDS)))

    result, _dispatch, _group = await _invoke([_fixture("esearch_success"), summary])

    assert [candidate.record["pmid"] for candidate in result.candidates] == list(_PMIDS)


@pytest.mark.asyncio
async def test_per_uid_http_200_error_is_provider_rejection_and_commits_zero() -> None:
    result, gateway_calls = await _execute(
        [
            (_fixture("esearch_success"), 200, "application/json", None),
            (_fixture("esummary_partial_error"), 200, "application/json", None),
        ]
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_response_rejected"
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.accounting.created == result.usage.accounting.debited == 2
    assert "sanitized missing" not in repr(result)


@pytest.mark.asyncio
async def test_summary_rate_limit_accounts_both_hops_and_commits_zero() -> None:
    result, gateway_calls = await _execute(
        [
            (_fixture("esearch_success"), 200, "application/json", None),
            (b"fixture-secret", 429, "text/plain", "120"),
        ]
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_rate_limited"
    assert result.logical_outcomes[0].retry_after == "120"
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.accounting.created == result.usage.accounting.debited == 2
    assert "fixture-secret" not in repr(result)


@pytest.mark.parametrize(
    "record",
    (
        {"uid": "31415926", "authors": [], "articleids": []},
        _summary_record("31415926", authors={}),
        _summary_record("31415926", articleids={}),
        _summary_record("31415926", authors=[{"authtype": "Author"}]),
        _summary_record(
            "31415926",
            articleids=[
                {"idtype": "pubmed", "value": "31415926"},
                {"idtype": "pubmed", "value": "31415926"},
            ],
        ),
        _summary_record(
            "31415926",
            articleids=[{"idtype": "pubmed", "value": "27182818"}],
        ),
        _summary_record(
            "31415926",
            articleids=[
                {"idtype": "pubmed", "value": "31415926"},
                {"idtype": "doi", "value": "not-a-doi"},
            ],
        ),
        _summary_record(
            "31415926",
            articleids=[
                {"idtype": "pubmed", "value": "31415926"},
                {"idtype": "pmc", "value": "3141592"},
            ],
        ),
    ),
)
@pytest.mark.asyncio
async def test_malformed_present_summary_record_is_rejected(record: dict[str, object]) -> None:
    summary = _esummary(("31415926",), records={"31415926": record})

    with pytest.raises(Exception) as caught:
        await _invoke([_esearch(("31415926",)), summary], result_limit=1)

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("field", ("authors", "articleids"))
@pytest.mark.asyncio
async def test_pubmed_record_collection_limits_are_enforced(field: str) -> None:
    if field == "authors":
        record = _summary_record(
            "31415926",
            authors=[{"name": f"Researcher {index}"} for index in range(1_025)],
        )
    else:
        article_ids = [{"idtype": "pubmed", "value": "31415926"}]
        article_ids.extend({"idtype": "pii", "value": f"sanitized-{index}"} for index in range(64))
        record = _summary_record("31415926", articleids=article_ids)
    summary = _esummary(("31415926",), records={"31415926": record})

    with pytest.raises(Exception) as caught:
        await _invoke([_esearch(("31415926",)), summary], result_limit=1)

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.asyncio
async def test_non_json_content_type_rejects_at_either_stage(stage: int) -> None:
    bodies = [_fixture("esearch_success"), _fixture("esummary_success")][: stage + 1]
    content_types = ["application/json"] * len(bodies)
    content_types[stage] = "text/html"

    with pytest.raises(Exception) as caught:
        await _invoke(bodies, content_types=content_types)

    _assert_typed_error(caught.value, "provider_response_rejected")


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.asyncio
async def test_429_at_either_stage_preserves_only_validated_retry_after(stage: int) -> None:
    bodies = [_fixture("esearch_success"), b"fixture-secret"][: stage + 1]
    statuses = [200] * len(bodies)
    statuses[stage] = 429
    retry_afters = [None] * len(bodies)
    retry_afters[stage] = "120"

    with pytest.raises(Exception) as caught:
        await _invoke(
            bodies,
            statuses=statuses,
            content_types=["application/json"] * len(bodies),
            retry_afters=retry_afters,
        )

    _assert_typed_error(caught.value, "provider_rate_limited")
    assert caught.value.retry_after == "120"
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.asyncio
async def test_http_200_rate_limit_envelope_is_typed_and_sanitized() -> None:
    body = _json({"error": "API rate limit exceeded", "count": "11"})

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_rate_limited")
    assert caught.value.retry_after is None


@pytest.mark.asyncio
async def test_unknown_http_200_error_envelope_is_sanitized_provider_rejection() -> None:
    body = _json({"error": "fixture-secret provider detail"})

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_response_rejected")
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.parametrize(
    "body",
    (
        b"",
        b"{",
        b"[]",
        b'{"header":{},"header":{}}',
        b"\xff\xfe{\x00}\x00",
        b"\xef\xbb\xbf{}",
    ),
)
@pytest.mark.asyncio
async def test_malformed_duplicate_or_non_utf8_json_is_rejected(body: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_route_response_byte_limit_is_enforced_before_parse() -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([_fixture("esearch_empty")], max_response_bytes=16)

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_summary_response_byte_limit_fails_after_only_two_accounted_calls() -> None:
    with pytest.raises(Exception) as caught:
        await _invoke(
            [_fixture("esearch_success"), _fixture("esummary_success")],
            max_response_bytes=512,
        )

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize("body", (b"{", b"[]", b"\xff\xfe{\x00}\x00"))
@pytest.mark.asyncio
async def test_malformed_or_non_utf8_summary_json_is_rejected(body: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([_fixture("esearch_success"), body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_parse_deadline_is_enforced_on_esearch() -> None:
    clock = _CountingClock(step=0.2)

    with pytest.raises(Exception) as caught:
        await _invoke([_fixture("esearch_empty")], monotonic_clock=clock)

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")
    assert clock.calls >= 4


@pytest.mark.asyncio
async def test_parse_deadline_is_enforced_on_summary() -> None:
    clock = _DelayedStepClock(delay_calls=5, step=0.2)

    with pytest.raises(Exception) as caught:
        await _invoke(
            [_fixture("esearch_success"), _fixture("esummary_success")],
            monotonic_clock=clock,
        )

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.asyncio
async def test_dispatch_cancellation_at_either_stage_propagates_unchanged(stage: int) -> None:
    registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    cancelled = asyncio.CancelledError(f"fixture-stage-{stage}-cancelled")
    responses: list[object] = [cancelled]
    if stage == 1:
        responses = [_response(route, group.intents[0], _fixture("esearch_success")), cancelled]
    dispatch = _RecordingDispatch(responses)
    adapter = _module().foundation_gateway_adapters()[_ADAPTER_ID]

    with pytest.raises(asyncio.CancelledError) as caught:
        await adapter(group, dispatch)

    assert caught.value is cancelled


@pytest.mark.asyncio
async def test_existing_gateway_timeout_classification_propagates_unchanged() -> None:
    _registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    timed_out = executor_module.DiscoveryExecutionError("aggregate_deadline_exceeded")
    dispatch = _RecordingDispatch([timed_out])
    adapter = _module().foundation_gateway_adapters()[_ADAPTER_ID]

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await adapter(group, dispatch)

    assert caught.value is timed_out


@pytest.mark.asyncio
async def test_unknown_adapter_version_rejects_before_dispatch() -> None:
    _registry, plan = _plan_for()
    group = replace(plan.dispatch_groups[0], adapter_version="unknown-v2")
    dispatch = _RecordingDispatch([])
    adapter = _module().foundation_gateway_adapters()[_ADAPTER_ID]

    with pytest.raises(Exception) as caught:
        await adapter(group, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert dispatch.calls == []


@pytest.mark.asyncio
async def test_runtime_egress_legacy_and_result_link_tripwires_receive_zero_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("PubMed adapter attempted direct egress or legacy execution")

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import http_hop
    from tldw_Server_API.app.core.Third_Party import PubMed

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(http_client, "fetch", forbidden)
    monkeypatch.setattr(http_client, "fetch_json", forbidden)
    monkeypatch.setattr(http_hop, "request_http_hop", forbidden)
    monkeypatch.setattr(PubMed, "search_pubmed", forbidden)
    monkeypatch.setattr(PubMed, "get_pubmed_by_id", forbidden)

    result, dispatch, _group = await _invoke([_fixture("esearch_success"), _fixture("esummary_success")])

    assert len(result.candidates) == 2
    assert result.candidates[0].record["url"] == "https://pubmed.ncbi.nlm.nih.gov/31415926/"
    assert result.candidates[0].record["pdf_url"] == ("https://pmc.ncbi.nlm.nih.gov/articles/PMC3141592/pdf/")
    assert len(dispatch.calls) == 2
