"""Pure-contract tests for the research discovery V2 foundation."""

from __future__ import annotations

import ast
import json
import os
import subprocess  # nosec B404
import sys
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core import exceptions as core_exceptions
from tldw_Server_API.app.core.Research.discovery import contracts as contracts_module
from tldw_Server_API.app.core.Research.discovery import executor as executor_module
from tldw_Server_API.app.core.Research.discovery import gateway as gateway_module
from tldw_Server_API.app.core.Research.discovery import gateway_adapters as gateway_adapters_module
from tldw_Server_API.app.core.Research.discovery import planner as planner_module
from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    AttributionMatch,
    BackendDefinition,
    BoundedDecimalQueryValuePolicy,
    BoundedTextQueryValuePolicy,
    BudgetCeilings,
    CredentialRequirement,
    CredentialStatus,
    DeferredNumericCSVQueryBinding,
    DiscoveryOutcomeIdentity,
    DiscoveryPlan,
    DiscoveryProvenanceV2,
    DispatchAllowance,
    DispatchIntent,
    ExactOrigin,
    ExactQueryValuePolicy,
    ExecutionMode,
    JSONBodyPair,
    LiteralTermsQueryValuePolicy,
    OpaqueCursorQueryValuePolicy,
    OperationKind,
    PathSlot,
    PathSlotKind,
    PathTemplate,
    PlannedBudgetAllowance,
    PlannedDispatchGroup,
    PlannedLogicalAttempt,
    PredicateOperator,
    QueryMode,
    QueryPair,
    ReadinessOverlay,
    ReadinessState,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    RouteReadiness,
    SourceConstraint,
    SourceDefinition,
    SourcePredicate,
    SourceRouteReference,
    canonical_plan_digest,
    canonical_policy_digest,
    evaluate_source_predicate,
    stable_document_id_v2,
)
from tldw_Server_API.app.core.Research.discovery.identity import (
    build_fingerprint,
    stable_result_id,
)

pytestmark = pytest.mark.unit


def test_discovery_exception_classes_are_centralized_with_compatible_aliases() -> None:
    compatibility_exports = {
        "DiscoveryGatewayError": (gateway_module, executor_module),
        "DiscoveryExecutionError": (executor_module,),
        "DiscoveryAdapterError": (executor_module, gateway_adapters_module),
        "PlanningError": (planner_module,),
        "_PayloadInvalid": (gateway_adapters_module,),
        "_ParseLimitExceeded": (gateway_adapters_module,),
        "_ParseDeadlineExceeded": (gateway_adapters_module,),
    }

    for name, modules in compatibility_exports.items():
        centralized = getattr(core_exceptions, name)
        assert centralized.__module__ == core_exceptions.__name__
        assert all(getattr(module, name) is centralized for module in modules)

    local_names_by_module = {
        gateway_module: {"DiscoveryGatewayError"},
        executor_module: {"DiscoveryExecutionError", "DiscoveryAdapterError"},
        planner_module: {"PlanningError"},
        gateway_adapters_module: {
            "_PayloadInvalid",
            "_ParseLimitExceeded",
            "_ParseDeadlineExceeded",
        },
    }
    for module, names in local_names_by_module.items():
        source = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        local_class_names = {node.name for node in source.body if isinstance(node, ast.ClassDef)}
        assert local_class_names.isdisjoint(names)


def test_digest_bound_request_policy_contracts_are_public() -> None:
    expected_names = (
        "PathSlotKind",
        "PathSlot",
        "PathTemplate",
        "ExactQueryValuePolicy",
        "BoundedDecimalQueryValuePolicy",
        "LiteralTermsQueryValuePolicy",
        "BoundedTextQueryValuePolicy",
        "OpaqueCursorQueryValuePolicy",
        "QueryValuePolicy",
    )

    assert all(hasattr(contracts_module, name) for name in expected_names)


def test_path_policy_contracts_are_closed_frozen_and_bounded() -> None:
    slots = (
        PathSlot(PathSlotKind.DATE, 10),
        PathSlot(PathSlotKind.UINT, 10),
        PathSlot(PathSlotKind.DOI_REGISTRANT, 12),
        PathSlot(PathSlotKind.DOI_SUFFIX, 128),
    )
    template = PathTemplate(("details", "biorxiv", slots[1], "json"), pagination_segment_index=2)

    assert template.segments[2] is slots[1]
    for value in (*slots, template):
        assert is_dataclass(value)
        assert value.__dataclass_params__.frozen is True
        assert not hasattr(value, "__dict__")
    for slot in slots:
        with pytest.raises(FrozenInstanceError):
            slot.max_chars = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        template.segments = ()  # type: ignore[misc]

    with pytest.raises(TypeError, match="path_slot_kind"):
        PathSlot("date", 10)  # type: ignore[arg-type]
    for kind, maximum in (
        (PathSlotKind.DATE, 10),
        (PathSlotKind.UINT, 10),
        (PathSlotKind.DOI_REGISTRANT, 12),
        (PathSlotKind.DOI_SUFFIX, 128),
    ):
        for invalid in (True, 0, -1, maximum + 1):
            with pytest.raises(ValueError, match="path_slot_max_chars"):
                PathSlot(kind, invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("literal", ("", " ", "has space", "has/slash", "has\\backslash", "café"))
def test_path_template_rejects_non_visible_or_non_segment_literals(literal: str) -> None:
    with pytest.raises(ValueError, match="path_template_literal"):
        PathTemplate(("details", literal, PathSlot(PathSlotKind.UINT, 10)))


@pytest.mark.parametrize("literal", (".", "..", "%2F", "%2f", "%5C", "%252F", "?", "#"))
def test_path_template_rejects_literals_with_url_path_semantics(literal: str) -> None:
    with pytest.raises(ValueError, match="path_template_literal"):
        PathTemplate(("details", literal, PathSlot(PathSlotKind.UINT, 10)))


def test_path_template_requires_exact_segments_and_uint_pagination_slot() -> None:
    class SlotSubclass(PathSlot):
        pass

    slot = PathSlot(PathSlotKind.UINT, 10)
    assert PathTemplate(("details", slot), pagination_segment_index=1).segments == ("details", slot)

    with pytest.raises(TypeError, match="path_template_segments"):
        PathTemplate(["details", slot])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="path_template_segments"):
        PathTemplate(())
    with pytest.raises(TypeError, match="path_template_segment"):
        PathTemplate(("details", SlotSubclass(PathSlotKind.UINT, 10)))
    for index in (True, -1, 2):
        with pytest.raises(ValueError, match="pagination_segment_index"):
            PathTemplate(("details", slot), pagination_segment_index=index)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pagination_segment_index"):
        PathTemplate(
            ("details", PathSlot(PathSlotKind.DATE, 10)),
            pagination_segment_index=1,
        )


def test_query_value_policy_contracts_require_exact_types_and_frozen_bounds() -> None:
    class StringSubclass(str):
        pass

    policies = (
        ExactQueryValuePolicy("format", "json"),
        BoundedDecimalQueryValuePolicy("pageSize", 100),
        LiteralTermsQueryValuePolicy(
            "query",
            ' AND SRC:PPR AND PUBLISHER:"bioRxiv"',
            16,
            64,
        ),
        BoundedTextQueryValuePolicy("category", 128),
        OpaqueCursorQueryValuePolicy("pageToken", 1_024),
    )
    assert tuple(policy.required for policy in policies) == (True, True, True, False, False)
    for policy in policies:
        assert is_dataclass(policy)
        assert policy.__dataclass_params__.frozen is True
        assert not hasattr(policy, "__dict__")

    for constructor in (
        lambda: ExactQueryValuePolicy(StringSubclass("format"), "json"),
        lambda: ExactQueryValuePolicy("format", StringSubclass("json")),
        lambda: ExactQueryValuePolicy("format", ""),
        lambda: BoundedDecimalQueryValuePolicy("pageSize", True),
        lambda: BoundedDecimalQueryValuePolicy("pageSize", 0),
        lambda: LiteralTermsQueryValuePolicy("query", " suffix", 17, 64),
        lambda: LiteralTermsQueryValuePolicy("query", " suffix", 16, 65),
        lambda: BoundedTextQueryValuePolicy("category", 129),
    ):
        with pytest.raises((TypeError, ValueError)):
            constructor()
    for policy in policies:
        with pytest.raises(TypeError, match="required"):
            replace(policy, required=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_chars", [0, 1_025, True, "1024"])
def test_opaque_query_policy_rejects_noncanonical_bounds(max_chars: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpaqueCursorQueryValuePolicy("pageToken", max_chars)  # type: ignore[arg-type]


def test_query_pair_repr_hides_value_without_changing_semantics() -> None:
    pair = QueryPair("pageToken", "opaque-token-sentinel")

    assert "opaque-token-sentinel" not in repr(pair)
    assert pair == QueryPair("pageToken", "opaque-token-sentinel")
    assert hash(pair) == hash(QueryPair("pageToken", "opaque-token-sentinel"))
    assert asdict(pair) == {"name": "pageToken", "value": "opaque-token-sentinel"}


def _template_policy(
    *,
    template: PathTemplate | None = None,
    query_value_policies: tuple[object, ...] = (),
    allowed_query_keys: tuple[str, ...] = (),
) -> RoutePolicy:
    return RoutePolicy(
        policy_version="policy-v2",
        origin=ExactOrigin("https", "api.example.test", 443),
        methods=("GET",),
        paths=(),
        path_template=template
        or PathTemplate(
            (
                "details",
                "biorxiv",
                PathSlot(PathSlotKind.DATE, 10),
                PathSlot(PathSlotKind.DATE, 10),
                PathSlot(PathSlotKind.UINT, 10),
                "json",
            ),
            pagination_segment_index=4,
        ),
        allowed_query_keys=allowed_query_keys,
        query_value_policies=query_value_policies,  # type: ignore[arg-type]
        limits=RouteLimits(2, 0, 0, 1_000, 4_096, 25),
    )


def test_route_policy_requires_exactly_one_path_channel() -> None:
    dynamic = _template_policy()

    assert dynamic.paths == ()
    assert dynamic.path_template is not None
    with pytest.raises(ValueError, match="path_channel"):
        replace(dynamic, paths=("/details",), policy_digest="")
    with pytest.raises(ValueError, match="path_channel"):
        replace(dynamic, path_template=None, policy_digest="")
    with pytest.raises(TypeError, match="path_template"):
        replace(dynamic, path_template=object(), policy_digest="")


def test_route_policy_preserves_legacy_full_positional_policy_digest_binding() -> None:
    origin = ExactOrigin("https", "api.example.test", 443)
    limits = RouteLimits(1, 0, 0, 1_000, 4_096, 25)
    expected = RoutePolicy("policy-v1", origin, ("GET",), ("/search",), (), limits)

    positional = RoutePolicy(
        "policy-v1",
        origin,
        ("GET",),
        ("/search",),
        (),
        limits,
        None,
        None,
        (),
        (),
        expected.policy_digest,
    )

    assert positional.policy_digest == expected.policy_digest
    assert positional.path_template is None
    assert positional.query_value_policies == ()


def test_template_pagination_is_exclusive_with_query_and_json_channels() -> None:
    dynamic = _template_policy()

    with pytest.raises(ValueError, match="pagination"):
        replace(
            dynamic,
            allowed_query_keys=("page",),
            pagination_query_key="page",
            policy_digest="",
        )
    with pytest.raises(ValueError, match="pagination"):
        replace(
            dynamic,
            methods=("POST",),
            allowed_json_body_keys=("page",),
            integer_json_body_keys=("page",),
            pagination_json_body_key="page",
            policy_digest="",
        )


def test_access_route_counts_one_template_as_one_initial_dispatch() -> None:
    dynamic = _template_policy()
    route = AccessRoute(
        route_id="example.details",
        backend_id="example",
        adapter_id="example.details",
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.DATE_INTERVAL,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native response",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=2,
        adapter_version="v1",
        policy=dynamic,
    )

    assert route.policy.path_template is dynamic.path_template
    with pytest.raises(ValueError, match="physical_dispatches"):
        replace(route, max_physical_dispatches=1)


def test_route_policy_query_value_rules_exactly_cover_allowed_keys() -> None:
    rules = (
        LiteralTermsQueryValuePolicy("query", " AND SRC:PPR", 16, 64),
        ExactQueryValuePolicy("format", "json"),
        BoundedDecimalQueryValuePolicy("pageSize", 100),
        BoundedTextQueryValuePolicy("category", 128),
        OpaqueCursorQueryValuePolicy("pageToken", 1_024),
    )
    policy = _template_policy(
        allowed_query_keys=("query", "format", "pageSize", "category", "pageToken"),
        query_value_policies=rules,
    )

    assert policy.query_value_policies == rules
    with pytest.raises(TypeError, match="query_value_policies"):
        replace(policy, query_value_policies=list(rules), policy_digest="")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="query_value_policy"):
        replace(policy, query_value_policies=(*rules[:-1], object()), policy_digest="")
    with pytest.raises(ValueError, match="query_value_policy"):
        replace(policy, query_value_policies=(*rules, rules[0]), policy_digest="")
    with pytest.raises(ValueError, match="query_value_policy"):
        replace(policy, allowed_query_keys=("query", "format", "pageSize", "category"), policy_digest="")


def test_empty_literal_suffix_is_contract_valid_and_digest_bound() -> None:
    policy = LiteralTermsQueryValuePolicy("query.term", "", 8, 32)
    assert policy.fixed_suffix == ""
    nonempty = LiteralTermsQueryValuePolicy("query.term", " AND FIXED", 8, 32)
    empty_route = _template_policy(
        query_value_policies=(policy,),
        allowed_query_keys=("query.term",),
    )
    nonempty_route = _template_policy(
        query_value_policies=(nonempty,),
        allowed_query_keys=("query.term",),
    )

    assert canonical_policy_digest(empty_route) != canonical_policy_digest(nonempty_route)


def test_dynamic_path_and_query_rules_are_bound_into_policy_digest() -> None:
    template = PathTemplate(
        (
            "details",
            "biorxiv",
            PathSlot(PathSlotKind.DATE, 10),
            PathSlot(PathSlotKind.DATE, 10),
            PathSlot(PathSlotKind.UINT, 10),
            "json",
        ),
        pagination_segment_index=4,
    )
    rules = (
        LiteralTermsQueryValuePolicy("query", " AND SRC:PPR", 16, 64),
        ExactQueryValuePolicy("format", "json"),
        BoundedDecimalQueryValuePolicy("pageSize", 100),
        BoundedTextQueryValuePolicy("category", 128),
    )
    base = _template_policy(
        template=template,
        allowed_query_keys=("query", "format", "pageSize", "category"),
        query_value_policies=rules,
    )
    template_mutations = (
        replace(template, segments=("details", "medrxiv", *template.segments[2:])),
        replace(
            template,
            segments=(*template.segments[:2], PathSlot(PathSlotKind.DOI_SUFFIX, 10), *template.segments[3:]),
        ),
        replace(
            template,
            segments=(*template.segments[:4], PathSlot(PathSlotKind.UINT, 9), *template.segments[5:]),
        ),
        replace(template, pagination_segment_index=None),
    )
    rule_mutations = (
        replace(rules[0], fixed_suffix=" AND SRC:MED"),
        replace(rules[0], max_terms=15),
        replace(rules[0], max_term_chars=63),
        replace(rules[0], required=False),
        replace(rules[1], value="xml"),
        replace(rules[2], maximum=99),
        replace(rules[3], max_chars=127),
        replace(rules[3], required=True),
    )

    digests = {base.policy_digest}
    for mutated_template in template_mutations:
        digests.add(replace(base, path_template=mutated_template, policy_digest="").policy_digest)
    for mutated_rule in rule_mutations:
        index = next(index for index, rule in enumerate(rules) if type(rule) is type(mutated_rule))
        mutated_rules = (*rules[:index], mutated_rule, *rules[index + 1 :])
        digests.add(replace(base, query_value_policies=mutated_rules, policy_digest="").policy_digest)

    assert len(digests) == 1 + len(template_mutations) + len(rule_mutations)


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


_INVALID_TRANSPORT_PATHS = (
    "/has space",
    "/has\tcontrol",
    "/has\r\ncontrol",
    "/has\x1fcontrol",
    "/has\x7fcontrol",
    "/café",
    "/bad%",
    "/bad%2",
    "/bad%GG",
)


@pytest.mark.parametrize("path", _INVALID_TRANSPORT_PATHS)
def test_route_policy_rejects_non_transport_safe_static_paths(path: str) -> None:
    with pytest.raises(ValueError, match="invalid_policy_paths"):
        replace(_policy(), paths=(path,), policy_digest="")


def test_route_policy_accepts_valid_percent_escapes() -> None:
    policy = replace(_policy(), paths=("/works/%20archive/%7E",), policy_digest="")

    assert policy.paths == ("/works/%20archive/%7E",)


@pytest.mark.parametrize("path", _INVALID_TRANSPORT_PATHS)
def test_dispatch_intent_rejects_non_transport_safe_paths(path: str) -> None:
    policy = _policy()

    with pytest.raises(ValueError, match="invalid_intent_path"):
        DispatchIntent(
            route_id="example_api_direct",
            policy_digest=policy.policy_digest,
            operation_kind=OperationKind.SEARCH,
            method="GET",
            path=path,
            query_pairs=(),
            limits=policy.limits,
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


def _logical_attempt(
    *,
    logical_attempt_id: str = "logical_attempt_v2_example",
    catalog_source_id: str = "example",
    source_predicate: SourcePredicate | None = None,
) -> PlannedLogicalAttempt:
    return PlannedLogicalAttempt(
        logical_attempt_id=logical_attempt_id,
        catalog_source_id=catalog_source_id,
        selection_reason="explicit",
        source_predicate=source_predicate,
    )


def _dispatch_group(
    *,
    dispatch_group_id: str = "dispatch_group_v2_example",
    logical_attempts: tuple[PlannedLogicalAttempt, ...] | None = None,
    policy: RoutePolicy | None = None,
    allowance: DispatchAllowance | None = None,
    filters: tuple[QueryPair, ...] = (),
    normalized_query: str = "test",
) -> PlannedDispatchGroup:
    policy = policy or _policy()
    return PlannedDispatchGroup(
        dispatch_group_id=dispatch_group_id,
        route_id="example_api_direct",
        backend_id="example_api",
        adapter_id="example_v2",
        adapter_version="example-v2",
        policy_digest=policy.policy_digest,
        limits=policy.limits,
        normalized_query=normalized_query,
        filters=filters,
        logical_attempts=logical_attempts if logical_attempts is not None else (_logical_attempt(),),
        fallback_order=0,
        intents=(
            DispatchIntent(
                route_id="example_api_direct",
                policy_digest=policy.policy_digest,
                operation_kind=OperationKind.SEARCH,
                method="GET",
                path="/works",
                query_pairs=(QueryPair("query", normalized_query),),
                limits=policy.limits,
            ),
        ),
        allowance=allowance
        or DispatchAllowance(
            policy.limits.max_pages + policy.limits.max_redirects + policy.limits.max_retries,
            policy.limits.max_pages,
            policy.limits.max_redirects,
            policy.limits.max_retries,
        ),
    )


def _plan(
    *,
    dispatch_groups: tuple[PlannedDispatchGroup, ...] | None = None,
    ceilings: BudgetCeilings | None = None,
    filters: tuple[QueryPair, ...] = (),
    normalized_query: str = "test",
    result_limit: int = 1,
) -> DiscoveryPlan:
    return DiscoveryPlan(
        planner_version="planner-v2",
        catalog_version="catalog-v2",
        registry_version="registry-v2",
        readiness_version="readiness-v2",
        execution_mode=ExecutionMode.SYNTHETIC,
        normalized_query=normalized_query,
        filters=filters,
        result_limit=result_limit,
        dispatch_groups=dispatch_groups if dispatch_groups is not None else (_dispatch_group(),),
        skipped=(),
        ceilings=ceilings or BudgetCeilings(1, 1, 1, 0, 0, 1_000, 1),
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
    logical_attempt = _logical_attempt(source_predicate=predicate)
    dispatch_group = _dispatch_group(logical_attempts=(logical_attempt,))
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
        logical_attempt,
        dispatch_group,
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
    logical_attempt = _logical_attempt(source_predicate=predicate)
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
    valid_dispatch_group = PlannedDispatchGroup(
        dispatch_group_id="dispatch_group_v2_example",
        route_id="example_api_direct",
        backend_id="example_api",
        adapter_id="example_v2",
        adapter_version="example-v2",
        policy_digest=_policy().policy_digest,
        limits=_policy().limits,
        normalized_query="test",
        filters=(),
        logical_attempts=(logical_attempt,),
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
        "result_limit": 1,
        "dispatch_groups": (valid_dispatch_group,),
        "skipped": (),
        "ceilings": BudgetCeilings(1, 1, 1, 0, 0, 1_000, 1),
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
        lambda: PlannedDispatchGroup(
            dispatch_group_id="dispatch_group_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            adapter_id="example_v2",
            adapter_version="example-v2",
            policy_digest=_policy().policy_digest,
            limits=_policy().limits,
            normalized_query="test",
            filters=(object(),),  # type: ignore[arg-type]
            logical_attempts=(logical_attempt,),
            fallback_order=0,
            intents=(intent,),
            allowance=allowance,
        ),
        lambda: PlannedDispatchGroup(
            dispatch_group_id="dispatch_group_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            adapter_id="example_v2",
            adapter_version="example-v2",
            policy_digest=_policy().policy_digest,
            limits=_policy().limits,
            normalized_query="test",
            filters=(),
            logical_attempts=(object(),),  # type: ignore[arg-type]
            fallback_order=0,
            intents=(intent,),
            allowance=allowance,
        ),
        lambda: PlannedDispatchGroup(
            dispatch_group_id="dispatch_group_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            adapter_id="example_v2",
            adapter_version="example-v2",
            policy_digest=_policy().policy_digest,
            limits=_policy().limits,
            normalized_query="test",
            filters=(),
            logical_attempts=(logical_attempt,),
            fallback_order=0,
            intents=(object(),),  # type: ignore[arg-type]
            allowance=allowance,
        ),
        lambda: DiscoveryPlan(**{**plan_values, "filters": (object(),)}),  # type: ignore[arg-type]
        lambda: DiscoveryPlan(**{**plan_values, "dispatch_groups": (object(),)}),  # type: ignore[arg-type]
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


def test_request_shape_rules_are_bound_into_the_policy_digest() -> None:
    base = _policy()
    pagination = RoutePolicy(
        policy_version=base.policy_version,
        origin=base.origin,
        methods=base.methods,
        paths=base.paths,
        allowed_query_keys=("query", "page"),
        limits=base.limits,
        pagination_query_key="page",
    )
    json_body = RoutePolicy(
        policy_version=base.policy_version,
        origin=base.origin,
        methods=("POST",),
        paths=base.paths,
        allowed_query_keys=base.allowed_query_keys,
        limits=base.limits,
        allowed_json_body_keys=("search_for",),
    )

    assert len({base.policy_digest, pagination.policy_digest, json_body.policy_digest}) == 3
    with pytest.raises(ValueError, match="pagination"):
        replace(base, pagination_query_key="page", policy_digest="")


def test_unrelated_policy_digest_keeps_pre_body_pagination_material() -> None:
    assert _policy().policy_digest == "8ec7b6572f32690e1425390518077742607bee40f87224c45913d7c5f54e7865"


def test_json_body_pairs_allow_only_bounded_exact_strings_or_nonnegative_integers() -> None:
    class StringSubclass(str):
        pass

    assert JSONBodyPair("search_for", "test").value == "test"
    assert JSONBodyPair("page", 0).value == 0
    assert JSONBodyPair("page", 2_147_483_647).value == 2_147_483_647

    for value in (StringSubclass("test"), True, -1, 2_147_483_648, 1.5):
        with pytest.raises(ValueError, match="json_body_pair_value"):
            JSONBodyPair("page", value)  # type: ignore[arg-type]


def test_policy_declares_exactly_one_pagination_channel() -> None:
    base = _policy()
    body_pagination = RoutePolicy(
        policy_version=base.policy_version,
        origin=base.origin,
        methods=("POST",),
        paths=base.paths,
        allowed_query_keys=(),
        limits=base.limits,
        pagination_json_body_key="page",
        allowed_json_body_keys=("search_for", "page", "page_size"),
        integer_json_body_keys=("page", "page_size"),
    )

    assert body_pagination.pagination_query_key is None
    assert body_pagination.pagination_json_body_key == "page"
    assert body_pagination.integer_json_body_keys == ("page", "page_size")
    assert body_pagination.policy_digest != base.policy_digest

    with pytest.raises(ValueError, match="pagination"):
        replace(
            body_pagination,
            allowed_query_keys=("cursor",),
            pagination_query_key="cursor",
            policy_digest="",
        )
    with pytest.raises(ValueError, match="pagination"):
        replace(body_pagination, pagination_json_body_key="cursor", policy_digest="")


def test_policy_integer_json_body_keys_are_exact_bounded_schema_material() -> None:
    base = _policy()
    body_pagination = RoutePolicy(
        policy_version=base.policy_version,
        origin=base.origin,
        methods=("POST",),
        paths=base.paths,
        allowed_query_keys=(),
        limits=base.limits,
        pagination_json_body_key="page",
        allowed_json_body_keys=("search_for", "page", "page_size"),
        integer_json_body_keys=("page", "page_size"),
    )

    with pytest.raises(TypeError, match="integer_json_body_keys"):
        replace(body_pagination, integer_json_body_keys=["page"], policy_digest="")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer_json_body"):
        replace(body_pagination, integer_json_body_keys=("page", "page"), policy_digest="")
    with pytest.raises(ValueError, match="integer_json_body"):
        replace(body_pagination, integer_json_body_keys=("page", "unknown"), policy_digest="")
    with pytest.raises(ValueError, match="pagination"):
        replace(body_pagination, integer_json_body_keys=("page_size",), policy_digest="")


@pytest.mark.parametrize(
    "changes",
    (
        {"allowed_json_body_keys": ("query",)},
        {"allowed_json_body_keys": ("search_for",)},
    ),
    ids=("query-body-key-overlap", "get-authorizes-json-body"),
)
def test_policy_rejects_ambiguous_or_non_post_json_body_channels(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="json_body"):
        replace(_policy(), policy_digest="", **changes)


def test_dispatch_intent_models_bounded_json_and_deferred_numeric_csv_input() -> None:
    policy = RoutePolicy(
        policy_version="policy-v1",
        origin=ExactOrigin("https", "api.example.test", 443),
        methods=("POST",),
        paths=("/works",),
        allowed_query_keys=("page", "id"),
        limits=RouteLimits(1, 0, 0, 1_000, 4_096, 25, max_request_body_bytes=512),
        pagination_query_key="page",
        allowed_json_body_keys=("search_for",),
    )
    body_pair = JSONBodyPair("search_for", "test")
    binding = DeferredNumericCSVQueryBinding(
        binding_id="search_result_ids",
        query_name="id",
        max_items=25,
        max_item_chars=16,
    )
    intent = DispatchIntent(
        route_id="example_api_direct",
        policy_digest=policy.policy_digest,
        operation_kind=OperationKind.CONDITIONAL_SUMMARY,
        method="POST",
        path="/works",
        query_pairs=(QueryPair("page", "1"),),
        limits=policy.limits,
        json_body_pairs=(body_pair,),
        query_bindings=(binding,),
    )

    assert intent.json_body_pairs == (body_pair,)
    assert intent.query_bindings == (binding,)
    assert intent.limits.max_request_body_bytes == 512
    for value in (body_pair, binding):
        assert is_dataclass(value)
        assert value.__dataclass_params__.frozen is True
        assert not hasattr(value, "__dict__")


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"json_body_pairs": (JSONBodyPair("search_for", "a"), JSONBodyPair("search_for", "b"))},
            "duplicate_json_body",
        ),
        (
            {
                "query_bindings": (
                    DeferredNumericCSVQueryBinding("first_ids", "id", 10, 16),
                    DeferredNumericCSVQueryBinding("second_ids", "id", 10, 16),
                )
            },
            "duplicate_query_binding",
        ),
        (
            {"query_bindings": (DeferredNumericCSVQueryBinding("result_ids", "query", 10, 16),)},
            "binding_query_conflict",
        ),
    ],
)
def test_dispatch_intent_rejects_ambiguous_body_and_binding_shapes(
    changes: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        DispatchIntent(
            route_id="example_api_direct",
            policy_digest=_policy().policy_digest,
            operation_kind=OperationKind.SEARCH,
            method="GET",
            path="/works",
            query_pairs=(QueryPair("query", "test"),),
            limits=_policy().limits,
            **changes,
        )


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


def test_source_predicate_values_are_canonical_and_cannot_match_everything() -> None:
    canonical = SourcePredicate(
        field_path=("source", "name"),
        operator=PredicateOperator.CONTAINS_ANY,
        values=("  Journal   Name ", "ARCHIVE"),
    )
    equivalent = SourcePredicate(
        field_path=("source", "name"),
        operator=PredicateOperator.CONTAINS_ANY,
        values=("archive", "journal name"),
    )

    assert canonical.values == ("archive", "journal name")
    assert canonical == equivalent
    assert hash(canonical) == hash(equivalent)
    with pytest.raises(ValueError, match="invalid_source_predicate_values"):
        SourcePredicate(
            field_path=("source", "name"),
            operator=PredicateOperator.CONTAINS_ANY,
            values=(" \t\n ",),
        )
    with pytest.raises(ValueError, match="duplicate_source_predicate_value"):
        SourcePredicate(
            field_path=("source", "name"),
            operator=PredicateOperator.EQUALS_ANY,
            values=("Journal", " journal "),
        )


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
        "json_body_pairs",
        "query_bindings",
    ]
    assert not any(hasattr(intent, name) for name in ("dispatch", "reserve", "debit", "release"))
    assert not any(callable(getattr(intent, field.name)) for field in fields(intent))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: DispatchAllowance(-1, 0, 0, 0),
        lambda: BudgetCeilings(1, -1, 1, 0, 0, 1_000, 1),
        lambda: RouteLimits(0, 0, 0, 1_000, 4_096, 1),
        lambda: RouteLimits(1, 0, 0, 1_000, 4_096, 1, max_request_body_bytes=True),
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
        PlannedDispatchGroup(
            dispatch_group_id="dispatch_group_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            adapter_id="example_v2",
            adapter_version="example-v2",
            policy_digest=expanded_policy.policy_digest,
            limits=expanded_policy.limits,
            normalized_query="test",
            filters=(),
            logical_attempts=(_logical_attempt(),),
            fallback_order=0,
            intents=(
                intent,
                replace(
                    intent,
                    operation_kind=OperationKind.CONDITIONAL_SUMMARY,
                    path="/summary",
                ),
            ),
            allowance=DispatchAllowance(physical_dispatches=4, pages=2, redirects=1, retries=1),
        )
    with pytest.raises(ValueError, match="allowance_limits_mismatch"):
        PlannedDispatchGroup(
            dispatch_group_id="dispatch_group_v2_example",
            route_id="example_api_direct",
            backend_id="example_api",
            adapter_id="example_v2",
            adapter_version="example-v2",
            policy_digest=expanded_policy.policy_digest,
            limits=expanded_policy.limits,
            normalized_query="test",
            filters=(),
            logical_attempts=(_logical_attempt(),),
            fallback_order=0,
            intents=(intent,),
            allowance=DispatchAllowance(physical_dispatches=0, pages=0, redirects=0, retries=0),
        )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda group: replace(
                group,
                intents=(replace(group.intents[0], route_id="other_route"),),
            ),
            "intent_route_mismatch",
        ),
        (
            lambda group: replace(
                group,
                intents=(replace(group.intents[0], policy_digest="0" * 64),),
            ),
            "intent_policy_mismatch",
        ),
        (
            lambda group: replace(
                group,
                intents=(
                    replace(
                        group.intents[0],
                        limits=replace(group.limits, max_results=group.limits.max_results - 1),
                    ),
                ),
            ),
            "intent_limits_mismatch",
        ),
        (
            lambda group: replace(
                group,
                allowance=replace(group.allowance, pages=0),
            ),
            "allowance_limits_mismatch",
        ),
    ],
)
def test_dispatch_group_rejects_cross_route_policy_limit_and_allowance_state(
    change: object,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        change(_dispatch_group())  # type: ignore[operator]


def test_dispatch_group_rejects_duplicate_logical_attempt_and_target_ids() -> None:
    logical_attempt = _logical_attempt()
    duplicate_attempt_id = replace(logical_attempt, catalog_source_id="other")
    duplicate_target_id = replace(
        logical_attempt,
        logical_attempt_id="logical_attempt_v2_other",
    )

    with pytest.raises(ValueError, match="duplicate_logical_attempt_id"):
        _dispatch_group(logical_attempts=(logical_attempt, duplicate_attempt_id))
    with pytest.raises(ValueError, match="duplicate_logical_target"):
        _dispatch_group(logical_attempts=(logical_attempt, duplicate_target_id))


def test_plan_rejects_duplicate_ids_and_cross_group_query_or_filter_state() -> None:
    first = _dispatch_group()
    duplicate_group_id = replace(
        first,
        logical_attempts=(
            _logical_attempt(
                logical_attempt_id="logical_attempt_v2_other",
                catalog_source_id="other",
            ),
        ),
    )
    duplicate_logical_id = replace(
        first,
        dispatch_group_id="dispatch_group_v2_other",
    )

    with pytest.raises(ValueError, match="duplicate_dispatch_group_id"):
        _plan(dispatch_groups=(first, duplicate_group_id))
    with pytest.raises(ValueError, match="duplicate_logical_attempt_id"):
        _plan(dispatch_groups=(first, duplicate_logical_id))
    with pytest.raises(ValueError, match="plan_query_mismatch"):
        _plan(dispatch_groups=(first,), normalized_query="other")
    with pytest.raises(ValueError, match="plan_filters_mismatch"):
        _plan(filters=(QueryPair("year", "2025"),))


def test_plan_derives_allowance_and_rejects_zero_budget_work_or_forged_aggregate() -> None:
    plan = _plan(result_limit=101, ceilings=BudgetCeilings(1, 1, 1, 0, 0, 1_000, 25))

    assert next(field for field in fields(DiscoveryPlan) if field.name == "allowance").init is False
    assert plan.allowance == PlannedBudgetAllowance(1, 1, 1, 0, 0, 1_000, 25)
    with pytest.raises(ValueError, match="budget_exceeded:route_attempts"):
        _plan(ceilings=BudgetCeilings(0, 0, 0, 0, 0, 0, 0))
    with pytest.raises(TypeError, match="allowance"):
        DiscoveryPlan(
            planner_version="planner-v2",
            catalog_version="catalog-v2",
            registry_version="registry-v2",
            readiness_version="readiness-v2",
            execution_mode=ExecutionMode.SYNTHETIC,
            normalized_query="test",
            filters=(),
            result_limit=1,
            dispatch_groups=(),
            skipped=(),
            ceilings=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
            allowance=PlannedBudgetAllowance(0, 0, 0, 0, 0, 0, 0),  # type: ignore[call-arg]
        )


def test_plan_digest_is_canonical_and_rejects_nonmatching_caller_value() -> None:
    plan = _plan()
    plan_digest = plan.plan_digest

    assert isinstance(plan_digest, str)
    assert len(plan_digest) == 64
    assert set(plan_digest) <= set("0123456789abcdef")
    assert canonical_plan_digest(plan) == plan_digest
    with pytest.raises(ValueError, match="plan_digest_mismatch"):
        replace(plan, plan_digest="0" * 64)


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


def _pure_module_violations(source: str, filename: str) -> list[str]:
    forbidden_import_modules = {"tldw_server_api.app.core.security.http_hop"}
    forbidden_import_parts = {
        "aiohttp",
        "config",
        "configparser",
        "configuration",
        "database",
        "databases",
        "db",
        "db_management",
        "dotenv",
        "ftplib",
        "http",
        "http_hop",
        "httpx",
        "requests",
        "settings",
        "smtplib",
        "socket",
        "sqlite3",
        "urllib",
        "urllib3",
        "workflow",
        "workflows",
    }
    forbidden_calls = {"getenv", "open", "read_bytes", "read_text", "urlopen"}
    forbidden_import_symbols = {*forbidden_calls, "environ", "request_http_hop"}
    violations: list[str] = []

    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        imported: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            imported = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = (
                *((node.module,) if node.module else ()),
                *(alias.name for alias in node.names),
            )
        for module_name in imported:
            parts = {part.casefold() for part in module_name.split(".")}
            normalized_name = module_name.casefold()
            if (
                normalized_name in forbidden_import_modules
                or parts & forbidden_import_parts
                or normalized_name in forbidden_import_symbols
            ):
                violations.append(f"{filename}:{node.lineno}:import:{module_name}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                violations.append(f"{filename}:{node.lineno}:call:{node.func.id}")
            elif isinstance(node.func, ast.Attribute) and (
                node.func.attr in forbidden_calls or node.func.attr.startswith("read_")
            ):
                violations.append(f"{filename}:{node.lineno}:call:{node.func.attr}")
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            violations.append(f"{filename}:{node.lineno}:environment_read")
        if isinstance(node, ast.Name) and node.id == "environ":
            violations.append(f"{filename}:{node.lineno}:environment_read")

    return violations


@pytest.mark.parametrize(
    "source",
    (
        "import tldw_Server_API.app.core.Security.http_hop",
        "from tldw_Server_API.app.core.Security import http_hop",
        "from tldw_Server_API.app.core.Security.http_hop import request_http_hop",
    ),
)
def test_pure_module_scanner_rejects_transport_facade_imports(source: str) -> None:
    assert _pure_module_violations(source, "synthetic_gateway_import.py")


def test_pure_foundation_modules_have_no_io_or_runtime_service_dependencies() -> None:
    source_root = Path(__file__).resolve().parents[2] / "app" / "core" / "Research" / "discovery"
    violations = [
        violation
        for filename in ("contracts.py", "registry.py", "planner.py")
        for violation in _pure_module_violations(
            (source_root / filename).read_text(encoding="utf-8"),
            filename,
        )
    ]

    assert violations == []


def test_normal_contract_import_is_side_effect_free(tmp_path: Path) -> None:
    """Run a fixed-interpreter, read-only subprocess to prove import isolation."""

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
        raw_path = args[0]
        path = os.fspath(raw_path).casefold() if isinstance(raw_path, (str, bytes, os.PathLike)) else ""
        code_read = path.endswith((".py", ".pyc", ".so", ".dylib"))
        read_open = bool(path) and not code_read
        if (
            isinstance(mode, str) and any(marker in mode for marker in "wax+")
        ) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        ) or read_open:
            raise RuntimeError(f"forbidden_open:{args[0]}")
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
forbidden_suffixes = {"adapters", "catalog", "identity", "models", "router", "service"}
loaded = sorted(
    name for name in sys.modules
    if name.startswith(discovery_prefix) and name.removeprefix(discovery_prefix).split(".", 1)[0] in forbidden_suffixes
)
forbidden_prefixes = (
    "tldw_Server_API.app.core.Research.artifact_store",
    "tldw_Server_API.app.core.Research.broker",
    "tldw_Server_API.app.core.Research.checkpoint_service",
    "tldw_Server_API.app.core.Research.limits",
    "tldw_Server_API.app.core.Research.models",
    "tldw_Server_API.app.core.Research.planner",
    "tldw_Server_API.app.core.Research.providers",
    "tldw_Server_API.app.core.Research.synthesizer",
    "tldw_Server_API.app.core.Security.http_hop",
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

    completed = subprocess.run(  # nosec B603
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
                "models",
                "planner",
                "providers",
                "synthesizer",
            ),
        ),
        (
            "tldw_Server_API.app.core.Research.discovery",
            ("adapters", "catalog", "identity", "models", "router", "service"),
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
