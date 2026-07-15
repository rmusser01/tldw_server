"""Offline compatibility proofs for the research discovery V2 foundation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess  # nosec B404 - fixed interpreter and inline read-only probe.
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.schemas.research_discovery_schemas import (
    ResearchDiscoverySearchResponse,
)
from tldw_Server_API.app.core.Research.discovery import registry as registry_module
from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
from tldw_Server_API.app.core.Research.discovery.contracts import (
    BudgetCeilings,
    DiscoveryProvenanceV2,
    ExecutionMode,
)
from tldw_Server_API.app.core.Research.discovery.executor import (
    DiscoveryAdapterResult,
    DiscoveryCandidate,
    LogicalOutcomeState,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
)
from tldw_Server_API.app.core.Research.discovery.gateway_adapters import foundation_gateway_adapters
from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint
from tldw_Server_API.app.core.Research.discovery.planner import (
    PlanningRequest,
    canonical_plan_bytes,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import foundation_readiness, foundation_registry
from tldw_Server_API.app.core.Security.http_hop import HTTPHopLimits

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "research_discovery_gateway_adapters"
_COMPATIBILITY_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Docs"
    / "Design"
    / "research_source_inventory"
    / "research-discovery-v2-foundation-v1-projection-v1.json"
)
_RECORDED_FIXTURES = {
    "semantic_scholar_v2": ("semantic_scholar_success.json",),
    "crossref_v2": ("crossref_success.json",),
    "arxiv_v2": ("arxiv_success.xml",),
    "pubmed_v2": ("pubmed_esearch_success.json", "pubmed_esummary_success.json"),
    "zenodo_v2": ("zenodo_success.json",),
    "figshare_v2": ("figshare_success.json",),
    "osf_v2": ("osf_success.json",),
}
_FOUNDATION_SOURCE_IDS = (
    "openalex",
    "semantic_scholar",
    "crossref",
    "arxiv",
    "pubmed",
    "zenodo",
    "figshare",
    "osf",
)
_V2_PRODUCTION_MODULES = tuple(
    f"tldw_Server_API.app.core.Research.discovery.{module_name}"
    for module_name in ("contracts", "registry", "planner", "executor", "gateway", "gateway_adapters")
)


class _RecordingV1Adapter:
    def __init__(self, source_id: str, records: list[dict[str, object]] | None = None) -> None:
        self.source_id = source_id
        self.records = records
        self.calls: list[dict[str, object]] = []

    async def search(self, *, query, source, limit, filters):
        self.calls.append(
            {
                "filters": dict(filters),
                "limit": limit,
                "query": query,
                "source_id": source.source_id,
            }
        )
        records = self.records
        if records is None:
            records = [
                {
                    "doi": f"10.4242/{self.source_id}",
                    "title": f"{self.source_id} legacy result",
                    "url": f"https://records.example/{self.source_id}",
                }
            ]
        return [dict(record) for record in records]


class _NoIOOAResolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def resolve_for_result(self, **kwargs):
        self.calls.append(dict(kwargs))
        return []


def _foundation_plan(source_ids: tuple[str, ...], *, result_limit: int):
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=source_ids,
            query="offline compatibility",
            filters=(),
            result_limit=result_limit,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(
            max_route_attempts=len(source_ids),
            max_physical_dispatches=8,
            max_pages_per_route=1,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=160_000,
            max_results=result_limit,
        ),
    )
    return registry, plan


def test_additive_request_policy_fields_preserve_exact_foundation_plan_bytes() -> None:
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=_FOUNDATION_SOURCE_IDS,
            query="  Causal   Inference  ",
            filters=(),
            result_limit=25,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=BudgetCeilings(16, 20, 1, 0, 0, 500_000, 100),
    )
    encoded = canonical_plan_bytes(plan)

    assert tuple((route.route_id, route.policy.policy_digest) for route in registry.routes) == (
        (
            "openalex_openalex_api_direct",
            "a4a26b31b2c424c472d93307f0f3091e821bea490a290a63076e4020bde48606",
        ),
        (
            "semantic_scholar_semantic_scholar_graph_api_direct",
            "caafcd1e81e8c62b9d14282ff7c04e8d7fc67d6dc172a570d47fae90df09421a",
        ),
        (
            "crossref_metadata_search_crossref_api_direct",
            "83a16ef3dd1ab57321062746907adf638a17ab19129433b9488eb8ba29c6ac70",
        ),
        (
            "arxiv_arxiv_api_direct",
            "d4697303d03a2aeb989b811ef0c371f871b5727cfc38bdbc8f1272340479d384",
        ),
        (
            "pubmed_ncbi_eutils_pubmed_direct",
            "0d121b0af2720904ba0aceb50ae18d57a2ee81b2d634d804fbcfc07c324476ab",
        ),
        (
            "zenodo_zenodo_records_api_direct",
            "ccfc7defbd128a7d8b86619e35772876f2c5378d8eba62f3401a97baa3c2ff5b",
        ),
        (
            "figshare_figshare_public_api_direct",
            "a6300381a69db2962a30eeb61f19482b57caf6602f28f01151e31851742a1c0d",
        ),
        (
            "open_science_framework_osf_api_direct",
            "e0c8a099197ae79f7ba1084d05c576d1f04a39fa24d1785e7d5fd8f0924bad25",
        ),
    )
    assert plan.plan_digest == "2e9869bc7ed6b51fe8ffe823dff2e392933e275b54bc2997e8542ba89829f403"
    assert hashlib.sha256(encoded).hexdigest() == "991d3a67132058625bdfef00836240cacc6b91370510c9b9d0762181310c9d46"
    assert len(encoded) == 10_936


def _gateway_response(
    route,
    intent,
    *,
    body: bytes = b'{"data":[]}',
) -> DiscoveryGatewayResponse:
    origin = route.policy.origin
    default_port = 443 if origin.scheme == "https" else 80
    requested_host = origin.host if origin.port == default_port else f"{origin.host}:{origin.port}"
    content_type = "application/atom+xml; charset=utf-8" if route.adapter_id == "arxiv_v2" else "application/json"
    return DiscoveryGatewayResponse(
        status_code=200,
        headers=(("content-type", content_type),),
        body=body,
        trace=DiscoveryGatewayTrace(
            route_id=route.route_id,
            policy_digest=route.policy.policy_digest,
            scheme=origin.scheme,
            requested_host=requested_host,
            tls_server_name=origin.host if origin.scheme == "https" else None,
            port=origin.port,
            method=intent.method,
            path=intent.path,
            query_keys=tuple(pair.name for pair in intent.query_pairs),
            timeout_ms=intent.limits.timeout_ms,
            max_response_bytes=intent.limits.max_response_bytes,
            http_limits=HTTPHopLimits(),
            status_code=200,
            resolved_ips=("93.184.216.34",),
            connected_ip="93.184.216.34",
            response_header_bytes=64,
            wire_bytes=len(body),
            decoded_bytes=len(body),
            elapsed_ms=1,
        ),
        redirect_location=None,
        retry_after=None,
    )


def _adapter(*records: tuple[str, dict[str, object]], revoke: dict[str, bool] | None = None):
    async def run(group, dispatch):
        await dispatch(group.intents[0])
        if revoke is not None:
            revoke["active"] = False
        return DiscoveryAdapterResult(
            candidates=tuple(DiscoveryCandidate(candidate_id, record) for candidate_id, record in records)
        )

    return run


async def _gateway(route, intent, *, is_policy_active):
    assert is_policy_active(route.route_id, route.policy.policy_digest) is True
    return _gateway_response(route, intent)


async def _execute_recorded_foundation(source_ids: tuple[str, ...] = _FOUNDATION_SOURCE_IDS):
    registry, plan = _foundation_plan(source_ids, result_limit=100)
    fixture_queues = {
        adapter_id: [(_FIXTURE_ROOT / filename).read_bytes() for filename in filenames]
        for adapter_id, filenames in _RECORDED_FIXTURES.items()
    }
    gateway_calls: list[tuple[str, str]] = []

    async def fixture_gateway(route, intent, *, is_policy_active):
        assert is_policy_active(route.route_id, route.policy.policy_digest) is True
        gateway_calls.append((route.adapter_id, intent.path))
        return _gateway_response(
            route,
            intent,
            body=fixture_queues[route.adapter_id].pop(0),
        )

    dispatch_count = sum(len(filenames) for filenames in _RECORDED_FIXTURES.values())
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=foundation_gateway_adapters(monotonic_clock=lambda: 0.0),
        gateway=fixture_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(f"dispatch-{index}" for index in range(dispatch_count)).__next__,
        monotonic_clock=lambda: 0.0,
    )
    assert dict.fromkeys(fixture_queues, 0) == {
        adapter_id: len(remaining) for adapter_id, remaining in fixture_queues.items()
    }
    return registry, plan, result, gateway_calls


async def _legacy_projection(plan, result, *, tmp_path: Path):
    """Feed V2 contributions through the real frozen V1 service contract."""
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    catalog = default_source_catalog(max_selected_sources=100)
    records_by_source: dict[str, list[dict[str, object]]] = {source_id: [] for source_id in _FOUNDATION_SOURCE_IDS}
    for candidate in result.candidates:
        for contribution in candidate.contributions:
            for source_id in contribution.provenance.requested_catalog_source_ids:
                records_by_source[source_id].append(dict(contribution.record))

    adapters = {
        source_id: _RecordingV1Adapter(source_id, records_by_source[source_id]) for source_id in _FOUNDATION_SOURCE_IDS
    }
    resolver = _NoIOOAResolver()
    service = ResearchDiscoveryService(
        catalog=catalog,
        router=ResearchSourceRouter(catalog=catalog, adapters=adapters),
        snapshot_db=ResearchSessionsDB(tmp_path / "v2-compatibility-v1.db"),
        oa_resolver=resolver,
    )
    response = await service.search(
        owner_user_id="offline-v2-compatibility",
        query=plan.normalized_query,
        source_ids=tuple(reversed(_FOUNDATION_SOURCE_IDS)),
        categories=(),
        per_source_limit=2,
        total_limit=16,
        filters={},
    )
    return ResearchDiscoverySearchResponse.model_validate(response), adapters, resolver


def _stable_v1_response_projection(response: ResearchDiscoverySearchResponse) -> dict[str, Any]:
    """Serialize every V1 public field except generated identifiers and timings."""
    payload = response.model_dump(mode="json")
    payload.pop("discovery_id")
    payload["metrics"].pop("elapsed_ms")
    for status in payload["source_statuses"]:
        status.pop("elapsed_ms")
    return payload


def _compatibility_contract() -> dict[str, Any]:
    assert (
        _COMPATIBILITY_CONTRACT_PATH.is_file()
    ), f"missing V2-to-V1 compatibility contract: {_COMPATIBILITY_CONTRACT_PATH}"
    return json.loads(_COMPATIBILITY_CONTRACT_PATH.read_text(encoding="utf-8"))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@pytest.mark.asyncio
async def test_duplicate_documents_do_not_consume_the_unique_result_cap() -> None:
    registry, plan = _foundation_plan(
        ("semantic_scholar", "crossref", "zenodo"),
        result_limit=2,
    )
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    shared = ("shared-document", {"title": "Shared paper", "doi": "10.1000/shared"})
    unique = ("unique-document", {"title": "Unique paper", "doi": "10.1000/unique"})

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: _adapter(shared),
            groups["crossref"].adapter_id: _adapter(shared),
            groups["zenodo"].adapter_id: _adapter(unique),
        },
        gateway=_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-1", "dispatch-2", "dispatch-3")).__next__,
    )

    assert tuple(candidate.candidate_id for candidate in result.candidates) == (
        "shared-document",
        "unique-document",
    )
    assert result.candidates[0].catalog_source_ids == ("semantic_scholar", "crossref")
    assert tuple(contribution.provenance.route_id for contribution in result.candidates[0].contributions) == (
        groups["semantic_scholar"].route_id,
        groups["crossref"].route_id,
    )
    assert result.truncated_candidates == 0


@pytest.mark.asyncio
async def test_duplicate_after_full_cap_augments_provenance_without_inflating_truncation() -> None:
    registry, plan = _foundation_plan(
        ("semantic_scholar", "crossref", "zenodo"),
        result_limit=1,
    )
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    shared = ("shared-document", {"title": "Shared paper", "doi": "10.1000/shared"})
    unique = ("unique-document", {"title": "Unique paper", "doi": "10.1000/unique"})

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: _adapter(shared),
            groups["crossref"].adapter_id: _adapter(shared),
            groups["zenodo"].adapter_id: _adapter(unique),
        },
        gateway=_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-1", "dispatch-2", "dispatch-3")).__next__,
    )

    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("shared-document",)
    assert result.candidates[0].catalog_source_ids == ("semantic_scholar", "crossref")
    assert len(result.candidates[0].contributions) == 2
    assert result.truncated_candidates == 1


@pytest.mark.asyncio
async def test_candidate_identity_conflict_fails_the_later_group_atomically() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"), result_limit=2)
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    shared = ("shared-document", {"title": "Shared paper", "doi": "10.1000/shared"})
    conflicting = ("shared-document", {"title": "Different paper", "doi": "10.1000/different"})

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: _adapter(shared),
            groups["crossref"].adapter_id: _adapter(shared, conflicting),
        },
        gateway=_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-1", "dispatch-2")).__next__,
    )

    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("shared-document",)
    assert result.candidates[0].catalog_source_ids == ("semantic_scholar",)
    assert len(result.candidates[0].contributions) == 1
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.SUCCEEDED,
        LogicalOutcomeState.FAILED,
    )
    assert result.logical_outcomes[1].code == "candidate_identity_conflict"


@pytest.mark.asyncio
async def test_revoked_later_group_cannot_add_provenance_to_committed_document() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"), result_limit=1)
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    shared = ("shared-document", {"title": "Shared paper", "doi": "10.1000/shared"})
    policy_state = {"active": True}

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: _adapter(shared),
            groups["crossref"].adapter_id: _adapter(shared, revoke=policy_state),
        },
        gateway=_gateway,
        policy_is_active=lambda _route_id, _digest: policy_state["active"],
        dispatch_id_factory=iter(("dispatch-1", "dispatch-2")).__next__,
    )

    assert result.candidates[0].catalog_source_ids == ("semantic_scholar",)
    assert len(result.candidates[0].contributions) == 1
    assert result.logical_outcomes[1].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[1].code == "dispatch_policy_inactive"


@pytest.mark.asyncio
async def test_committed_transport_origin_does_not_alias_mutable_registry_state() -> None:
    registry, plan = _foundation_plan(("semantic_scholar",), result_limit=1)
    group = plan.dispatch_groups[0]
    route_origin = registry.get_route(group.route_id).policy.origin
    original_origin = (route_origin.scheme, route_origin.host, route_origin.port)

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: _adapter(("candidate", {"title": "Candidate", "doi": "10.1000/candidate"}))},
        gateway=_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-1",
    )

    committed_origin = result.candidates[0].contributions[0].provenance.transport_origin
    assert committed_origin is not route_origin
    object.__setattr__(route_origin, "host", "mutated.example")
    assert committed_origin is not None
    assert (committed_origin.scheme, committed_origin.host, committed_origin.port) == original_origin


@pytest.mark.asyncio
async def test_attribution_basis_mutation_after_dispatch_fails_the_group_closed() -> None:
    registry, plan = _foundation_plan(("semantic_scholar",), result_limit=1)
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)

    async def mutating_adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        object.__setattr__(route, "attribution_basis", "forged-by-adapter")
        return DiscoveryAdapterResult(
            candidates=(
                DiscoveryCandidate(
                    "candidate",
                    {"title": "Candidate", "doi": "10.1000/candidate"},
                ),
            )
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: mutating_adapter},
        gateway=_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-1",
    )

    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "registry_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_basis", ("", "   "))
async def test_invalid_attribution_basis_fails_before_runtime_effects(invalid_basis: str) -> None:
    registry, plan = _foundation_plan(("semantic_scholar",), result_limit=1)
    group = plan.dispatch_groups[0]
    object.__setattr__(registry.get_route(group.route_id), "attribution_basis", invalid_basis)
    calls: list[str] = []

    async def forbidden(*_args, **_kwargs):
        calls.append("runtime")
        raise AssertionError("runtime must not execute")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: forbidden},
        gateway=forbidden,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("dispatch-id") or "dispatch-1",
    )

    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "registry_mismatch"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trace_field", "forged_value"),
    (
        ("scheme", "http"),
        ("requested_host", "forged.example"),
        ("port", 444),
        ("tls_server_name", "forged.example"),
    ),
)
async def test_gateway_trace_origin_must_match_the_registered_route(
    trace_field: str,
    forged_value: object,
) -> None:
    registry, plan = _foundation_plan(("semantic_scholar",), result_limit=1)
    group = plan.dispatch_groups[0]

    async def forged_gateway(route, intent, *, is_policy_active):
        response = _gateway_response(route, intent)
        return replace(
            response,
            trace=replace(response.trace, **{trace_field: forged_value}),
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: _adapter(("candidate", {"title": "Candidate", "doi": "10.1000/candidate"}))},
        gateway=forged_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-1",
    )

    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "gateway_response_mismatch"


def test_provenance_contract_keeps_retrieval_origin_unclaimed() -> None:
    fields = DiscoveryProvenanceV2.__dataclass_fields__

    assert "transport_origin" in fields
    assert "reported_document_origin" in fields
    assert "retrieval_observed_origin" in fields


@pytest.mark.asyncio
async def test_recorded_foundation_merges_repeated_identity_with_complete_route_provenance() -> None:
    registry, plan, result, gateway_calls = await _execute_recorded_foundation()
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    ready_source_ids = tuple(source_id for source_id in _FOUNDATION_SOURCE_IDS if source_id != "openalex")

    assert tuple(candidate.candidate_id for candidate in result.candidates) == (
        "research_document_v2:6762d490c99b8b7c1943efbc298d597b",
        "research_document_v2:9e284a07ed61c2f1c2283e3b8a81763c",
    )
    shared, pubmed_only = result.candidates
    assert shared.catalog_source_ids == ready_source_ids
    assert pubmed_only.catalog_source_ids == ("pubmed",)
    assert shared.record["provider"] == "semantic_scholar"
    assert pubmed_only.record["provider"] == "pubmed"
    assert tuple(contribution.record["provider"] for contribution in shared.contributions) == ready_source_ids
    assert tuple(contribution.provenance.route_id for contribution in shared.contributions) == tuple(
        groups[source_id].route_id for source_id in ready_source_ids
    )
    assert tuple(
        contribution.provenance.requested_catalog_source_ids for contribution in shared.contributions
    ) == tuple((source_id,) for source_id in ready_source_ids)
    for contribution in (*shared.contributions, *pubmed_only.contributions):
        provenance = contribution.provenance
        route = registry.get_route(provenance.route_id)
        assert provenance.transport_origin == route.policy.origin
        assert provenance.reported_document_origin is None
        assert provenance.retrieval_observed_origin is None
        assert provenance.catalog_version == plan.catalog_version
        assert provenance.adapter_version == route.adapter_version
        assert provenance.policy_digest == route.policy.policy_digest
    assert Counter(adapter_id for adapter_id, _path in gateway_calls) == Counter(
        {adapter_id: len(filenames) for adapter_id, filenames in _RECORDED_FIXTURES.items()}
    )
    assert len(result.usage.physical_records) == 8
    assert result.truncated_candidates == 0


def test_v2_to_v1_compatibility_contract_is_canonical_json() -> None:
    """The reviewed complete-response golden has one canonical representation."""
    raw_contract = _COMPATIBILITY_CONTRACT_PATH.read_text(encoding="utf-8")
    assert raw_contract == _canonical_json(json.loads(raw_contract))


@pytest.mark.asyncio
async def test_recorded_v2_results_have_a_stable_additive_v1_serialization_projection(
    tmp_path: Path,
) -> None:
    _registry, plan, result, _gateway_calls = await _execute_recorded_foundation()

    projection, adapters, resolver = await _legacy_projection(plan, result, tmp_path=tmp_path)
    serialized = projection.model_dump(mode="json")
    round_tripped = ResearchDiscoverySearchResponse.model_validate_json(projection.model_dump_json()).model_dump(
        mode="json"
    )
    contract = _compatibility_contract()

    assert round_tripped == serialized
    assert contract["contract_version"] == "research-discovery-v2-foundation-v1-projection-v1"
    assert _stable_v1_response_projection(projection) == contract["stable_response"]
    assert tuple(build_fingerprint(dict(candidate.record)) for candidate in result.candidates) == tuple(
        item["fingerprint"] for item in serialized["results"]
    )
    assert all(
        candidate.candidate_id != item["result_id"] for candidate, item in zip(result.candidates, serialized["results"])
    )
    assert all(
        adapter.calls
        == [
            {
                "filters": {},
                "limit": 2,
                "query": plan.normalized_query,
                "source_id": source_id,
            }
        ]
        for source_id, adapter in adapters.items()
    )
    assert len(resolver.calls) == 2
    assert tuple(outcome.catalog_source_id for outcome in result.logical_outcomes) == tuple(
        source_id for source_id in _FOUNDATION_SOURCE_IDS if source_id != "openalex"
    )
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (LogicalOutcomeState.SUCCEEDED,) * 7
    assert tuple(
        (target.requested_source_id, target.status.value, target.code.value, target.reason) for target in result.skipped
    ) == (
        (
            "openalex",
            "unavailable",
            "credentialed_out_of_scope",
            "credentialed_route_not_authorized_for_foundation",
        ),
    )
    outcomes_by_source = {
        outcome.catalog_source_id: (outcome.state.value, outcome.code) for outcome in result.logical_outcomes
    }
    skipped_by_source = {
        target.requested_source_id: (target.status.value, target.code.value) for target in result.skipped
    }
    assert tuple(
        (source_id, *(outcomes_by_source.get(source_id) or skipped_by_source[source_id]))
        for source_id in _FOUNDATION_SOURCE_IDS
    ) == (
        ("openalex", "unavailable", "credentialed_out_of_scope"),
        ("semantic_scholar", "succeeded", None),
        ("crossref", "succeeded", None),
        ("arxiv", "succeeded", None),
        ("pubmed", "succeeded", None),
        ("zenodo", "succeeded", None),
        ("figshare", "succeeded", None),
        ("osf", "succeeded", None),
    )
    assert all(
        contribution.provenance.retrieval_observed_origin is None
        for candidate in result.candidates
        for contribution in candidate.contributions
    )


@pytest.mark.asyncio
async def test_recorded_projection_is_independent_of_requested_source_order() -> None:
    _registry, _plan, forward, _calls = await _execute_recorded_foundation()
    _registry, _plan, reversed_result, _calls = await _execute_recorded_foundation(
        tuple(reversed(_FOUNDATION_SOURCE_IDS))
    )

    def stable_projection(result):
        return tuple(
            (
                candidate.candidate_id,
                candidate.catalog_source_ids,
                tuple(
                    (
                        contribution.provenance.requested_catalog_source_ids,
                        contribution.provenance.route_id,
                        dict(contribution.record),
                    )
                    for contribution in candidate.contributions
                ),
            )
            for candidate in result.candidates
        )

    assert stable_projection(reversed_result) == stable_projection(forward)


def test_foundation_construction_requires_an_explicit_nonproduction_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        registry_module,
        "foundation_registry",
        lambda: calls.append("registry") or (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert tuple(ExecutionMode) == (ExecutionMode.OFFLINE_FIXTURE, ExecutionMode.SYNTHETIC)
    assert (
        inspect.signature(registry_module.foundation_readiness).parameters["execution_mode"].default
        is inspect.Parameter.empty
    )
    with pytest.raises(TypeError, match="execution_mode_must_be_ExecutionMode"):
        registry_module.foundation_readiness("live")  # type: ignore[arg-type]
    assert calls == []


def test_standalone_and_deep_research_entrypoints_leave_v2_unloaded(tmp_path: Path) -> None:
    script = r"""
import importlib
import json
import sys

prefix = "tldw_Server_API.app.core.Research.discovery"
endpoint = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.research_discovery")
service = endpoint.get_research_discovery_service()
before = sorted(name for name in sys.modules if name == prefix or name.startswith(prefix + "."))
broker = importlib.import_module("tldw_Server_API.app.core.Research.broker")
jobs = importlib.import_module("tldw_Server_API.app.core.Research.jobs")
after = sorted(name for name in sys.modules if name == prefix or name.startswith(prefix + "."))
forbidden = tuple(
    f"{prefix}.{name}"
    for name in ("contracts", "registry", "planner", "executor", "gateway", "gateway_adapters")
)
print(json.dumps({
    "adapter_names": service.adapter_names,
    "broker_module": broker.__name__,
    "discovery_after": after,
    "discovery_before": before,
    "forbidden_loaded": sorted(name for name in forbidden if name in sys.modules),
    "jobs_module": jobs.__name__,
    "router_module": type(service._router).__module__,
    "service_module": type(service).__module__,
}, sort_keys=True))
"""
    env = os.environ.copy()
    env.update(
        {
            "AUTH_MODE": "single_user",
            "AUTHNZ_SCHEDULER_DISABLED": "1",
            "AUTO_DOWNLOAD_MODELS": "false",
            "CIRCUIT_BREAKER_REGISTRY_MODE": "memory",
            "DATABASE_URL": f"sqlite:///{tmp_path / 'users.db'}",
            "DISABLE_AUTHNZ_SCHEDULER": "1",
            "DISABLE_NLTK_DOWNLOADS": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SINGLE_USER_API_KEY": "task-12968-test-key",
            "TEMP": str(tmp_path),
            "TEST_MODE": "true",
            "TLDW_TEST_MODE": "true",
            "TMP": str(tmp_path),
            "TMPDIR": str(tmp_path),
            "USER_DB_BASE_DIR": str(tmp_path / "user-dbs"),
            "WORKFLOWS_SCHEDULER_ENABLED": "false",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path(__file__).parents[3]), env.get("PYTHONPATH"))))
    completed = subprocess.run(  # nosec B603 - fixed interpreter and inline probe.
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_legacy_modules = [
        "tldw_Server_API.app.core.Research.discovery",
        "tldw_Server_API.app.core.Research.discovery.adapters",
        "tldw_Server_API.app.core.Research.discovery.catalog",
        "tldw_Server_API.app.core.Research.discovery.identity",
        "tldw_Server_API.app.core.Research.discovery.models",
        "tldw_Server_API.app.core.Research.discovery.oa",
        "tldw_Server_API.app.core.Research.discovery.router",
        "tldw_Server_API.app.core.Research.discovery.service",
    ]

    assert evidence == {
        "adapter_names": sorted(_FOUNDATION_SOURCE_IDS),
        "broker_module": "tldw_Server_API.app.core.Research.broker",
        "discovery_after": expected_legacy_modules,
        "discovery_before": expected_legacy_modules,
        "forbidden_loaded": [],
        "jobs_module": "tldw_Server_API.app.core.Research.jobs",
        "router_module": "tldw_Server_API.app.core.Research.discovery.router",
        "service_module": "tldw_Server_API.app.core.Research.discovery.service",
    }
    assert not set(evidence["forbidden_loaded"]) & set(_V2_PRODUCTION_MODULES)


def test_standalone_endpoint_calls_each_v1_source_once_and_never_executes_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_db_pool
    from tldw_Server_API.app.api.v1.endpoints import research_discovery as endpoint
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery import executor as executor_module
    from tldw_Server_API.app.core.Research.discovery import gateway as gateway_module
    from tldw_Server_API.app.core.Research.discovery import gateway_adapters as gateway_adapters_module
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    catalog = default_source_catalog(max_selected_sources=100)
    adapters = {source_id: _RecordingV1Adapter(source_id) for source_id in _FOUNDATION_SOURCE_IDS}
    resolver = _NoIOOAResolver()
    snapshot_db = ResearchSessionsDB(tmp_path / "standalone-v1.db")
    service = ResearchDiscoveryService(
        catalog=catalog,
        router=ResearchSourceRouter(catalog=catalog, adapters=adapters),
        snapshot_db=snapshot_db,
        oa_resolver=resolver,
    )
    v2_calls: list[str] = []

    async def forbidden_execute(*_args, **_kwargs):
        v2_calls.append("execute")
        raise AssertionError("V2 executor must remain disabled")

    async def forbidden_dispatch(*_args, **_kwargs):
        v2_calls.append("dispatch")
        raise AssertionError("V2 gateway must remain disabled")

    def forbidden_factory(*_args, **_kwargs):
        v2_calls.append("factory")
        raise AssertionError("V2 adapters must remain disabled")

    monkeypatch.setattr(executor_module, "execute_discovery_plan", forbidden_execute)
    monkeypatch.setattr(gateway_module, "dispatch_once", forbidden_dispatch)
    monkeypatch.setattr(gateway_adapters_module, "foundation_gateway_adapters", forbidden_factory)

    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/research")
    app.dependency_overrides[endpoint.get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[endpoint.get_research_discovery_service] = lambda: service
    app.dependency_overrides[endpoint.check_rate_limit] = lambda: None
    app.dependency_overrides[get_db_pool] = lambda: SimpleNamespace(pool=None)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/research/discovery/search",
                json={
                    "categories": [],
                    "fallback_policy": "disabled",
                    "filters": {"language": "en", "year_from": 2020},
                    "per_source_limit": 2,
                    "query": "legacy execution contract",
                    "source_ids": list(reversed(_FOUNDATION_SOURCE_IDS)),
                    "total_limit": 16,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    payload = response.json()
    assert tuple(status["source_id"] for status in payload["source_statuses"]) == _FOUNDATION_SOURCE_IDS
    assert tuple(item["primary_source_id"] for item in payload["results"]) == _FOUNDATION_SOURCE_IDS
    assert payload["effective_config"]["source_ids"] == list(_FOUNDATION_SOURCE_IDS)
    assert all(
        adapters[source_id].calls
        == [
            {
                "filters": {"language": "en", "year_from": 2020},
                "limit": 2,
                "query": "legacy execution contract",
                "source_id": source_id,
            }
        ]
        for source_id in _FOUNDATION_SOURCE_IDS
    )
    assert v2_calls == []
    assert len(resolver.calls) == 8
    assert Counter(call["source_id"] for call in resolver.calls) == Counter(_FOUNDATION_SOURCE_IDS)
    assert snapshot_db.get_discovery_snapshot(payload["discovery_id"], owner_user_id="1") is not None
