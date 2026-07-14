"""Pure-contract tests for the research discovery V2 foundation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    AttributionMatch,
    BackendDefinition,
    BudgetCeilings,
    CredentialRequirement,
    CredentialStatus,
    DiscoveryOutcomeIdentity,
    DiscoveryPlan,
    DiscoveryProvenanceV2,
    DispatchAllowance,
    DispatchIntent,
    ExactOrigin,
    ExecutionMode,
    OperationKind,
    PlannedAttempt,
    PlannedBudgetAllowance,
    PredicateOperator,
    QueryMode,
    QueryPair,
    ReadinessOverlay,
    ReadinessState,
    RequestedTarget,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    RouteReadiness,
    SourceConstraint,
    SourceDefinition,
    SourcePredicate,
    SourceRouteReference,
    canonical_policy_digest,
    evaluate_source_predicate,
    stable_document_id_v2,
)
from tldw_Server_API.app.core.Research.discovery.identity import (
    build_fingerprint,
    stable_result_id,
)


def _policy(*, digest: str = "") -> RoutePolicy:
    return RoutePolicy(
        policy_version="policy-v1",
        origin=ExactOrigin(scheme="https", host="api.example.test", port=443),
        methods=("GET",),
        paths=("/works",),
        allowed_query_keys=("query",),
        limits=RouteLimits(
            max_pages=1,
            max_redirects=0,
            max_retries=0,
            timeout_ms=1_000,
            max_response_bytes=4_096,
            max_results=25,
        ),
        policy_digest=digest,
    )


def _route() -> AccessRoute:
    return AccessRoute(
        route_id="example_api_direct",
        backend_id="example_api",
        adapter_id="example_v2",
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.STRUCTURED_QUERY,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native_response",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=1,
        adapter_version="example-v2",
        policy=_policy(),
    )


def test_contract_values_are_frozen_slots_dataclasses() -> None:
    predicate = SourcePredicate(
        field_path=("source", "id"),
        operator=PredicateOperator.EQUALS_ANY,
        values=("example",),
    )
    source_ref = SourceRouteReference(
        route_id="example_api_direct",
        source_predicate=None,
    )
    source = SourceDefinition(
        catalog_source_id="example",
        display_name="Example",
        aliases=("example_legacy",),
        categories=("papers",),
        content_types=("works",),
        surfaces=("standalone_search", "deep_research"),
        route_references=(source_ref,),
        site_hosts=("example.test",),
        priority=10,
        catalog_version="catalog-v2",
    )
    readiness = RouteReadiness(
        route_id="example_api_direct",
        state=ReadinessState.READY,
        credential_status=CredentialStatus.NOT_REQUIRED,
        reason="fixture_ready",
    )
    overlay = ReadinessOverlay(
        overlay_version="overlay-v1",
        execution_mode=ExecutionMode.OFFLINE_FIXTURE,
        routes=(readiness,),
    )
    allowance = DispatchAllowance(
        physical_dispatches=1,
        pages=1,
        redirects=0,
        retries=0,
    )
    intent = DispatchIntent(
        route_id="example_api_direct",
        policy_digest=_policy().policy_digest,
        operation_kind=OperationKind.SEARCH,
        method="GET",
        path="/works",
        query_pairs=(QueryPair(name="query", value="test"),),
        limits=_policy().limits,
    )
    requested_target = RequestedTarget(
        catalog_source_id="example",
        selection_reason="explicit",
        source_predicate=predicate,
    )
    provenance = DiscoveryProvenanceV2(
        requested_catalog_source_ids=("example",),
        route_id="example_api_direct",
        backend_id="example_api",
        transport_origin=ExactOrigin("https", "api.example.test", 443),
        reported_document_origin=ExactOrigin("https", "documents.example.test", 443),
        retrieval_observed_origin=None,
        attribution_basis="provider_source_field",
        catalog_version="catalog-v2",
        adapter_version="example-v2",
        policy_digest=_policy().policy_digest,
    )
    identity = DiscoveryOutcomeIdentity.from_fingerprint("doi:10.1000/example")
    values = (
        BackendDefinition("example_api", "Example API"),
        BudgetCeilings(2, 2, 1, 0, 0, 2_000, 25),
        predicate,
        source_ref,
        source,
        _policy(),
        _route(),
        readiness,
        overlay,
        allowance,
        intent,
        requested_target,
        provenance,
        identity,
    )

    for value in values:
        assert is_dataclass(value)
        assert value.__dataclass_params__.frozen is True
        assert not hasattr(value, "__dict__")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, fields(value)[0].name, "changed")


@pytest.mark.parametrize(
    ("constructor", "expected"),
    [
        (lambda: BackendDefinition("Not Normalized", "Example"), "backend_id"),
        (lambda: QueryPair("", "value"), "query_pair_name"),
        (
            lambda: SourceDefinition(
                catalog_source_id="bad id",
                display_name="Bad",
                aliases=(),
                categories=("papers",),
                content_types=("works",),
                surfaces=("standalone_search",),
                route_references=(SourceRouteReference("example_api_direct", None),),
                site_hosts=(),
                priority=1,
                catalog_version="catalog-v2",
            ),
            "catalog_source_id",
        ),
        (
            lambda: SourceRouteReference("../route", None),
            "route_id",
        ),
    ],
)
def test_catalog_and_route_identifiers_are_validated(constructor: object, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        constructor()  # type: ignore[operator]


def test_contracts_reject_mutable_collection_inputs() -> None:
    with pytest.raises(TypeError, match="tuple"):
        SourcePredicate(
            field_path=["source"],  # type: ignore[arg-type]
            operator=PredicateOperator.EQUALS_ANY,
            values=("example",),
        )


def test_nested_contract_values_must_have_the_declared_immutable_types() -> None:
    predicate = SourcePredicate(
        field_path=("source", "id"),
        operator=PredicateOperator.EQUALS_ANY,
        values=("example",),
    )
    target = RequestedTarget("example", "explicit", predicate)
    intent = DispatchIntent(
        route_id="example_api_direct",
        policy_digest=_policy().policy_digest,
        operation_kind=OperationKind.SEARCH,
        method="GET",
        path="/works",
        query_pairs=(QueryPair("query", "test"),),
        limits=_policy().limits,
    )
    allowance = DispatchAllowance(1, 1, 0, 0)
    valid_attempt = PlannedAttempt(
        attempt_id="attempt_v2_example",
        route_id="example_api_direct",
        backend_id="example_api",
        policy_digest=_policy().policy_digest,
        normalized_query="test",
        filters=(),
        requested_targets=(target,),
        fallback_order=0,
        intents=(intent,),
        allowance=allowance,
    )
    plan_values = {
        "planner_version": "planner-v2",
        "catalog_version": "catalog-v2",
        "registry_version": "registry-v2",
        "readiness_version": "readiness-v2",
        "execution_mode": ExecutionMode.SYNTHETIC,
        "normalized_query": "test",
        "filters": (),
        "attempts": (valid_attempt,),
        "skipped": (),
        "ceilings": BudgetCeilings(1, 1, 1, 0, 0, 1_000, 1),
        "allowance": PlannedBudgetAllowance(1, 1, 1, 0, 0, 1_000, 1),
    }

    constructors = (
        lambda: SourceDefinition(
            catalog_source_id="example",
            display_name="Example",
            aliases=(),
            categories=("papers",),
            content_types=("works",),
            surfaces=("standalone_search",),
            route_references=(object(),),  # type: ignore[arg-type]
            site_hosts=(),
            priority=1,
            catalog_version="catalog-v2",
        ),
        lambda: ReadinessOverlay(
            overlay_version="overlay-v1",
            execution_mode=ExecutionMode.SYNTHETIC,
            routes=(object(),),  # type: ignore[arg-type]
        ),
        lambda: PlannedAttempt(
            attempt_id="attempt_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            policy_digest=_policy().policy_digest,
            normalized_query="test",
            filters=(object(),),  # type: ignore[arg-type]
            requested_targets=(target,),
            fallback_order=0,
            intents=(intent,),
            allowance=allowance,
        ),
        lambda: PlannedAttempt(
            attempt_id="attempt_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            policy_digest=_policy().policy_digest,
            normalized_query="test",
            filters=(),
            requested_targets=(object(),),  # type: ignore[arg-type]
            fallback_order=0,
            intents=(intent,),
            allowance=allowance,
        ),
        lambda: PlannedAttempt(
            attempt_id="attempt_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            policy_digest=_policy().policy_digest,
            normalized_query="test",
            filters=(),
            requested_targets=(target,),
            fallback_order=0,
            intents=(object(),),  # type: ignore[arg-type]
            allowance=allowance,
        ),
        lambda: DiscoveryPlan(**{**plan_values, "filters": (object(),)}),  # type: ignore[arg-type]
        lambda: DiscoveryPlan(**{**plan_values, "attempts": (object(),)}),  # type: ignore[arg-type]
        lambda: DiscoveryPlan(**{**plan_values, "skipped": (object(),)}),  # type: ignore[arg-type]
    )

    for constructor in constructors:
        with pytest.raises(TypeError):
            constructor()
    with pytest.raises(TypeError, match="tuple"):
        ReadinessOverlay(
            overlay_version="overlay-v1",
            execution_mode=ExecutionMode.SYNTHETIC,
            routes=[],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "origin",
    [
        ("HTTPS", "api.example.test", 443),
        ("https", "API.EXAMPLE.TEST", 443),
        ("https", "api.example.test/path", 443),
        ("https", "user@api.example.test", 443),
        ("https", "api.example.test", 0),
    ],
)
def test_exact_origins_require_normalized_components(origin: tuple[str, str, int]) -> None:
    with pytest.raises(ValueError, match="origin"):
        ExactOrigin(*origin)


def test_route_policy_computes_and_revalidates_canonical_digest() -> None:
    policy = _policy()

    assert policy.policy_digest == canonical_policy_digest(policy)
    assert len(policy.policy_digest) == 64
    with pytest.raises(ValueError, match="policy_digest_mismatch"):
        _policy(digest="0" * 64)


def test_query_modes_predicates_and_credentials_are_typed() -> None:
    with pytest.raises(TypeError, match="query_mode"):
        AccessRoute(
            **{
                **{
                    field.name: getattr(_route(), field.name)
                    for field in fields(AccessRoute)
                    if field.name != "query_modes"
                },
                "query_modes": ("structured_query",),
            }
        )
    with pytest.raises(TypeError, match="credential_requirement"):
        AccessRoute(
            **{
                **{
                    field.name: getattr(_route(), field.name)
                    for field in fields(AccessRoute)
                    if field.name != "credential_requirement"
                },
                "credential_requirement": "none",
            }
        )


def test_source_predicate_reports_match_nonmatch_and_ambiguity() -> None:
    predicate = SourcePredicate(
        field_path=("source", "name"),
        operator=PredicateOperator.EQUALS_ANY,
        values=("Example Journal",),
    )

    assert evaluate_source_predicate(predicate, {"source": {"name": "example journal"}}) is AttributionMatch.MATCH
    assert evaluate_source_predicate(predicate, {"source": {"name": "Other"}}) is AttributionMatch.NON_MATCH
    assert evaluate_source_predicate(predicate, {"source": {}}) is AttributionMatch.AMBIGUOUS
    assert evaluate_source_predicate(predicate, {"source": {"name": {"nested": "value"}}}) is AttributionMatch.AMBIGUOUS


@pytest.mark.parametrize("mode", ["production", "", None])
def test_readiness_overlay_has_no_production_default(mode: object) -> None:
    with pytest.raises(TypeError, match="execution_mode"):
        ReadinessOverlay(
            overlay_version="overlay-v1",
            execution_mode=mode,  # type: ignore[arg-type]
            routes=(),
        )


def test_dispatch_intent_is_descriptive_and_cannot_account_or_dispatch() -> None:
    intent = DispatchIntent(
        route_id="example_api_direct",
        policy_digest=_policy().policy_digest,
        operation_kind=OperationKind.SEARCH,
        method="GET",
        path="/works",
        query_pairs=(QueryPair("query", "test"),),
        limits=_policy().limits,
    )

    assert [field.name for field in fields(DispatchIntent)] == [
        "route_id",
        "policy_digest",
        "operation_kind",
        "method",
        "path",
        "query_pairs",
        "limits",
    ]
    assert not any(hasattr(intent, name) for name in ("dispatch", "reserve", "debit", "release"))
    assert not any(callable(getattr(intent, field.name)) for field in fields(intent))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: DispatchAllowance(-1, 0, 0, 0),
        lambda: BudgetCeilings(1, -1, 1, 0, 0, 1_000, 1),
        lambda: RouteLimits(0, 0, 0, 1_000, 4_096, 1),
    ],
)
def test_budget_and_allowance_values_reject_negative_or_impossible_limits(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


def test_route_and_attempt_allowances_cover_pages_redirects_and_retries() -> None:
    expanded_policy = RoutePolicy(
        policy_version="policy-v1",
        origin=ExactOrigin("https", "api.example.test", 443),
        methods=("GET",),
        paths=("/works",),
        allowed_query_keys=("query",),
        limits=RouteLimits(2, 1, 1, 1_000, 4_096, 25),
    )

    with pytest.raises(ValueError, match="physical_dispatch"):
        AccessRoute(
            route_id="example_api_direct",
            backend_id="example_api",
            adapter_id="example_v2",
            route_kind=RouteKind.DIRECT,
            query_modes=(QueryMode.STRUCTURED_QUERY,),
            source_constraint=SourceConstraint.NATIVE_CORPUS,
            attribution_basis="native_response",
            credential_requirement=CredentialRequirement.NONE,
            fallback_order=0,
            max_physical_dispatches=1,
            adapter_version="example-v2",
            policy=expanded_policy,
        )
    with pytest.raises(ValueError, match="physical_dispatch"):
        DispatchAllowance(physical_dispatches=1, pages=2, redirects=1, retries=1)

    intent = DispatchIntent(
        route_id="example_api_direct",
        policy_digest=expanded_policy.policy_digest,
        operation_kind=OperationKind.SEARCH,
        method="GET",
        path="/works",
        query_pairs=(QueryPair("query", "test"),),
        limits=expanded_policy.limits,
    )
    with pytest.raises(ValueError, match="physical_dispatch"):
        PlannedAttempt(
            attempt_id="attempt_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            policy_digest=expanded_policy.policy_digest,
            normalized_query="test",
            filters=(),
            requested_targets=(RequestedTarget("example", "explicit", None),),
            fallback_order=0,
            intents=(
                intent,
                replace(
                    intent,
                    operation_kind=OperationKind.CONDITIONAL_SUMMARY,
                    path="/summary",
                ),
            ),
            allowance=DispatchAllowance(physical_dispatches=1, pages=1, redirects=0, retries=0),
        )
    with pytest.raises(ValueError, match="physical_dispatch"):
        PlannedAttempt(
            attempt_id="attempt_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            policy_digest=expanded_policy.policy_digest,
            normalized_query="test",
            filters=(),
            requested_targets=(RequestedTarget("example", "explicit", None),),
            fallback_order=0,
            intents=(intent,),
            allowance=DispatchAllowance(physical_dispatches=0, pages=0, redirects=0, retries=0),
        )


def test_v2_document_identity_is_route_independent_and_v1_identity_is_unchanged() -> None:
    fingerprint = build_fingerprint(
        {
            "doi": "https://doi.org/10.1000/ABC",
            "title": "Ignored",
            "provider": "crossref",
            "source_id": "crossref",
        }
    )
    v2_from_crossref = DiscoveryOutcomeIdentity.from_fingerprint(fingerprint)
    v2_from_aggregator = DiscoveryOutcomeIdentity(
        fingerprint=fingerprint,
        document_id=stable_document_id_v2(fingerprint),
    )

    assert fingerprint == "doi:10.1000/abc"
    assert stable_result_id(fingerprint, "crossref", "crossref") == "discovery_result:6b62633ded3841e5d2a405c1"
    assert v2_from_crossref == v2_from_aggregator
    assert v2_from_crossref.document_id == "research_document_v2:91b32f9a00bd805c33baf538295e5726"
    assert "crossref" not in v2_from_crossref.document_id
    assert "aggregator" not in v2_from_crossref.document_id


def test_normal_contract_import_is_side_effect_free(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment.pop("PYTEST_CURRENT_TEST", None)
    environment.pop("TESTING", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repository_root), environment.get("PYTHONPATH", "")) if value
    )
    script = """
import importlib
import json
import os
from pathlib import Path
import sys

def reject_side_effect(event, args):
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if (
            isinstance(mode, str) and any(marker in mode for marker in "wax+")
        ) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        ):
            raise RuntimeError(f"write_open:{args[0]}")
    if event in {"os.mkdir", "sqlite3.connect", "socket.__new__"}:
        raise RuntimeError(event)

sys.addaudithook(reject_side_effect)
contracts = importlib.import_module("tldw_Server_API.app.core.Research.discovery.contracts")
registry_module = importlib.import_module("tldw_Server_API.app.core.Research.discovery.registry")
planner = importlib.import_module("tldw_Server_API.app.core.Research.discovery.planner")
registry = registry_module.foundation_registry()
readiness = registry_module.foundation_readiness(contracts.ExecutionMode.SYNTHETIC)
planner.compile_discovery_plan(
    planner.PlanningRequest(("arxiv",), "test", (), 1),
    registry=registry,
    readiness=readiness,
    budget=contracts.BudgetCeilings(1, 1, 1, 0, 0, 20_000, 1),
)
discovery_prefix = "tldw_Server_API.app.core.Research.discovery."
forbidden_suffixes = {"adapters", "catalog", "models", "router", "service"}
loaded = sorted(
    name for name in sys.modules
    if name.startswith(discovery_prefix) and name.removeprefix(discovery_prefix).split(".", 1)[0] in forbidden_suffixes
)
forbidden_prefixes = (
    "tldw_Server_API.app.core.Research.artifact_store",
    "tldw_Server_API.app.core.Research.broker",
    "tldw_Server_API.app.core.Research.checkpoint_service",
    "tldw_Server_API.app.core.Research.limits",
    "tldw_Server_API.app.core.Research.planner",
    "tldw_Server_API.app.core.Research.synthesizer",
    "tldw_Server_API.app.core.AuthNZ",
    "tldw_Server_API.app.core.DB_Management",
    "tldw_Server_API.app.core.config",
    "tldw_Server_API.app.services.workflows",
)
loaded.extend(
    sorted(name for name in sys.modules if name.startswith(forbidden_prefixes))
)
created = sorted(str(path.relative_to(Path.cwd())) for path in Path.cwd().rglob("*") if path.is_file())
print(json.dumps({"loaded": loaded, "created": created}))
"""

    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence == {"loaded": [], "created": []}


def test_legacy_discovery_package_exports_keep_defining_object_identity() -> None:
    import importlib

    package = importlib.import_module("tldw_Server_API.app.core.Research.discovery")
    defining_modules = {
        "CATALOG_VERSION": ".catalog",
        "DiscoveryExecutionPolicy": ".models",
        "DiscoveryMetrics": ".models",
        "DiscoveryProviderRouter": ".service",
        "DiscoverySearchResponse": ".models",
        "DiscoverySourceStatus": ".models",
        "ResearchSourceCatalog": ".catalog",
        "ResearchSourceCatalogEntry": ".models",
        "ResearchDiscoveryService": ".service",
        "ResearchSourceRouter": ".router",
        "SourceCapabilities": ".models",
        "SourceSelectionError": ".models",
        "SourceStatus": ".models",
        "default_discovery_adapters": ".adapters",
        "default_source_catalog": ".catalog",
    }

    assert package.__all__ == list(defining_modules)
    for name, module_name in defining_modules.items():
        module = importlib.import_module(module_name, package.__name__)
        assert getattr(package, name) is getattr(module, name)
    assert set(package.__all__).issubset(dir(package))


def test_legacy_research_package_exports_keep_defining_object_identity() -> None:
    import importlib

    package = importlib.import_module("tldw_Server_API.app.core.Research")
    defining_modules = {
        "ResearchArtifactStore": ".artifact_store",
        "ResearchBroker": ".broker",
        "ResearchLimits": ".limits",
        "ResearchSynthesizer": ".synthesizer",
        "apply_checkpoint_patch": ".checkpoint_service",
        "build_initial_plan": ".planner",
        "ensure_limit_available": ".limits",
    }

    assert package.__all__ == list(defining_modules)
    for name, module_name in defining_modules.items():
        module = importlib.import_module(module_name, package.__name__)
        assert getattr(package, name) is getattr(module, name)
    assert set(package.__all__).issubset(dir(package))


def test_research_package_keeps_implicit_submodule_import_behavior() -> None:
    import importlib

    from tldw_Server_API.app.core.Research import jobs, jobs_worker

    assert jobs is importlib.import_module("tldw_Server_API.app.core.Research.jobs")
    assert jobs_worker is importlib.import_module("tldw_Server_API.app.core.Research.jobs_worker")


@pytest.mark.parametrize(
    ("package_name", "submodule_names"),
    [
        (
            "tldw_Server_API.app.core.Research",
            (
                "artifact_store",
                "broker",
                "checkpoint_service",
                "limits",
                "planner",
                "synthesizer",
            ),
        ),
        (
            "tldw_Server_API.app.core.Research.discovery",
            ("adapters", "catalog", "models", "router", "service"),
        ),
    ],
)
def test_legacy_packages_keep_lazy_submodule_attributes(
    package_name: str,
    submodule_names: tuple[str, ...],
) -> None:
    import importlib

    package = importlib.import_module(package_name)
    for submodule_name in submodule_names:
        expected = importlib.import_module(f"{package_name}.{submodule_name}")
        package.__dict__.pop(submodule_name, None)
        try:
            assert getattr(package, submodule_name) is expected
            assert submodule_name in dir(package)
            assert submodule_name not in package.__all__
        finally:
            setattr(package, submodule_name, expected)
