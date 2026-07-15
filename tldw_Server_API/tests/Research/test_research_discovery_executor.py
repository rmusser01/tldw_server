"""Accounted execution tests for the research discovery V2 foundation."""

from __future__ import annotations

import ast
import asyncio
import inspect
from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_Server_API.app.core.Research.discovery import executor as executor_module
from tldw_Server_API.app.core.Research.discovery.contracts import (
    BudgetCeilings,
    CredentialStatus,
    ExecutionMode,
    OperationKind,
    PredicateOperator,
    ReadinessState,
    SkippedCode,
    SkippedStatus,
    SkippedTarget,
    SourceConstraint,
    SourcePredicate,
    SourceRouteReference,
)
from tldw_Server_API.app.core.Research.discovery.executor import (
    AttemptJournal,
    DiscoveryAdapterResult,
    DiscoveryCandidate,
    DispatchAccounting,
    LogicalOutcomeState,
    NumericCSVBindingValues,
    NumericCursor,
    PhysicalDispatchState,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayError,
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
)
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
from tldw_Server_API.app.core.Security.http_hop import HTTPHopLimits


def _semantic_scholar_plan():
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=("semantic_scholar",),
            query="accounted execution",
            filters=(),
            result_limit=3,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(
            max_route_attempts=1,
            max_physical_dispatches=1,
            max_pages_per_route=1,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=20_000,
            max_results=3,
        ),
    )
    return registry, plan


def _gateway_response(route, intent) -> DiscoveryGatewayResponse:
    origin = route.policy.origin
    default_port = 443 if origin.scheme == "https" else 80
    requested_host = origin.host if origin.port == default_port else f"{origin.host}:{origin.port}"
    return DiscoveryGatewayResponse(
        status_code=200,
        headers=(("content-type", "application/json"),),
        body=b'{"data":[]}',
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
            wire_bytes=11,
            decoded_bytes=11,
            elapsed_ms=1,
        ),
        redirect_location=None,
        retry_after=None,
    )


async def _assert_plan_validation_precedes_effects(registry, plan) -> None:
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    calls: list[str] = []

    async def should_not_run(*args, **kwargs):
        calls.append("runtime")
        raise AssertionError("must not run")

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: should_not_run for group in plan.dispatch_groups},
            gateway=should_not_run,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
            journal=journal,
        )

    assert caught.value.code == "plan_validation_failed"
    assert calls == []
    assert journal.records == ()
    assert journal.accounting == DispatchAccounting(
        physical_ceiling=plan.ceilings.max_physical_dispatches,
        created=0,
        debited=0,
        released=0,
        outstanding=0,
    )


def _replace_plan_with_fresh_digest(plan, **changes):
    changes["plan_digest"] = ""
    return replace(plan, **changes)


def _foundation_plan(source_ids: tuple[str, ...], *, result_limit: int = 3):
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=source_ids,
            query="accounted execution",
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


def _fallback_plan():
    base = foundation_registry()
    semantic_route_id = base.get_source("semantic_scholar").route_references[0].route_id
    registry = DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version=base.registry_version,
        sources=base.sources,
        routes=tuple(
            replace(route, fallback_order=1) if route.route_id == semantic_route_id else route for route in base.routes
        ),
        backends=base.backends,
    )
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=("semantic_scholar",),
            query="accounted execution",
            filters=(),
            result_limit=1,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(1, 1, 1, 0, 0, 20_000, 1),
    )
    return registry, plan


def _paginated_semantic_scholar_plan():
    base = foundation_registry()
    route_id = base.get_source("semantic_scholar").route_references[0].route_id
    routes = []
    for route in base.routes:
        if route.route_id != route_id:
            routes.append(route)
            continue
        limits = replace(route.policy.limits, max_pages=2)
        policy = replace(route.policy, limits=limits, policy_digest="")
        routes.append(replace(route, max_physical_dispatches=2, policy=policy))
    registry = DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version=base.registry_version,
        sources=base.sources,
        routes=tuple(routes),
        backends=base.backends,
    )
    plan = compile_discovery_plan(
        PlanningRequest(("semantic_scholar",), "accounted execution", (), 3),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(1, 2, 2, 0, 0, 40_000, 3),
    )
    return registry, plan


def _coalesced_semantic_scholar_plan():
    base = foundation_registry()
    source = base.get_source("semantic_scholar")
    route_id = source.route_references[0].route_id
    first_predicate = SourcePredicate(
        ("source", "collection"),
        PredicateOperator.EQUALS_ANY,
        ("shared",),
    )
    second_predicate = SourcePredicate(
        ("source", "collection"),
        PredicateOperator.EQUALS_ANY,
        ("other",),
    )
    first_source = replace(
        source,
        route_references=(SourceRouteReference(route_id, first_predicate),),
    )
    second_source = replace(
        source,
        catalog_source_id="semantic_scholar_secondary",
        display_name="Semantic Scholar Secondary",
        aliases=(),
        priority=source.priority + 1,
        route_references=(SourceRouteReference(route_id, second_predicate),),
    )
    registry = DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version=base.registry_version,
        sources=tuple(
            first_source if item.catalog_source_id == source.catalog_source_id else item for item in base.sources
        )
        + (second_source,),
        routes=tuple(
            (
                replace(
                    route,
                    source_constraint=SourceConstraint.PROVIDER_SOURCE_FILTER,
                    attribution_basis="source.collection",
                )
                if route.route_id == route_id
                else route
            )
            for route in base.routes
        ),
        backends=base.backends,
    )
    plan = compile_discovery_plan(
        PlanningRequest(
            ("semantic_scholar", "semantic_scholar_secondary"),
            "accounted execution",
            (),
            3,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(2, 1, 1, 0, 0, 20_000, 3),
    )
    return registry, plan


def _retryable_foundation_plan(
    source_id: str,
    *,
    route_retries: int = 1,
    budget_retries: int | None = None,
):
    base = foundation_registry()
    route_id = base.get_source(source_id).route_references[0].route_id
    routes = []
    for route in base.routes:
        if route.route_id != route_id:
            routes.append(route)
            continue
        limits = replace(route.policy.limits, max_retries=route_retries)
        policy = replace(route.policy, limits=limits, policy_digest="")
        routes.append(replace(route, max_physical_dispatches=1 + route_retries, policy=policy))
    registry = DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version=base.registry_version,
        sources=base.sources,
        routes=tuple(routes),
        backends=base.backends,
    )
    plan = compile_discovery_plan(
        PlanningRequest((source_id,), "accounted execution", (), 3),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(
            1,
            1 + route_retries,
            1,
            0,
            route_retries if budget_retries is None else budget_retries,
            40_000,
            3,
        ),
    )
    return registry, plan


def _retryable_semantic_scholar_plan():
    return _retryable_foundation_plan("semantic_scholar")


def _redirectable_foundation_plan(
    source_id: str = "semantic_scholar",
    *,
    route_redirects: int = 1,
    budget_redirects: int | None = None,
):
    base = foundation_registry()
    route_id = base.get_source(source_id).route_references[0].route_id
    routes = []
    for route in base.routes:
        if route.route_id != route_id:
            routes.append(route)
            continue
        limits = replace(route.policy.limits, max_redirects=route_redirects)
        policy = replace(route.policy, limits=limits, policy_digest="")
        routes.append(replace(route, max_physical_dispatches=1 + route_redirects, policy=policy))
    registry = DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version=base.registry_version,
        sources=base.sources,
        routes=tuple(routes),
        backends=base.backends,
    )
    plan = compile_discovery_plan(
        PlanningRequest((source_id,), "accounted execution", (), 3),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(
            1,
            1 + route_redirects,
            1,
            route_redirects if budget_redirects is None else budget_redirects,
            0,
            40_000,
            3,
        ),
    )
    return registry, plan


def test_executor_public_contract_is_typed_and_frozen() -> None:
    cursor = NumericCursor(0)
    binding = NumericCSVBindingValues("result_ids", (1, 2))

    assert cursor.value == 0
    assert binding.values == (1, 2)
    assert issubclass(executor_module.DiscoveryExecutionError, ValueError)
    with pytest.raises(FrozenInstanceError):
        cursor.value = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="cursor"):
        NumericCursor(-1)
    with pytest.raises(ValueError, match="binding"):
        NumericCSVBindingValues("result_ids", (0,))


def test_attempt_journal_reserve_dispatch_success_accounting() -> None:
    journal = AttemptJournal(physical_ceiling=1)

    reserved = journal.reserve(
        dispatch_id="dispatch-1",
        dispatch_group_id="group-1",
        route_id="route-1",
        operation_kind=OperationKind.SEARCH,
    )
    dispatching = journal.mark_dispatching("dispatch-1")
    succeeded = journal.mark_succeeded("dispatch-1")

    assert reserved.state is PhysicalDispatchState.RESERVED
    assert dispatching.state is PhysicalDispatchState.DISPATCHING
    assert succeeded.state is PhysicalDispatchState.SUCCEEDED
    assert journal.records == (succeeded,)
    assert journal.accounting == DispatchAccounting(
        created=1,
        debited=1,
        released=0,
        outstanding=0,
        physical_ceiling=1,
    )


def test_attempt_journal_release_reuses_capacity() -> None:
    journal = AttemptJournal(physical_ceiling=1)
    journal.reserve(
        dispatch_id="dispatch-unused",
        dispatch_group_id="group-1",
        route_id="route-1",
        operation_kind=OperationKind.SEARCH,
    )
    released = journal.release("dispatch-unused", PhysicalDispatchState.SKIPPED)

    journal.reserve(
        dispatch_id="dispatch-reused",
        dispatch_group_id="group-1",
        route_id="route-1",
        operation_kind=OperationKind.SEARCH,
    )
    journal.mark_dispatching("dispatch-reused")

    assert released.state is PhysicalDispatchState.SKIPPED
    assert journal.accounting == DispatchAccounting(
        created=2,
        debited=1,
        released=1,
        outstanding=0,
        physical_ceiling=1,
    )


def test_attempt_journal_rejects_duplicate_dispatch_id_before_second_record() -> None:
    journal = AttemptJournal(physical_ceiling=2)
    journal.reserve(
        dispatch_id="dispatch-collision",
        dispatch_group_id="group-1",
        route_id="route-1",
        operation_kind=OperationKind.SEARCH,
    )

    with pytest.raises(ValueError, match="duplicate_dispatch_id"):
        journal.reserve(
            dispatch_id="dispatch-collision",
            dispatch_group_id="group-1",
            route_id="route-1",
            operation_kind=OperationKind.SEARCH,
        )

    assert len(journal.records) == 1
    assert journal.accounting.created == 1


@pytest.mark.asyncio
async def test_scripted_adapter_dispatches_only_through_bound_capability() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    gateway_calls = []
    adapter_observations = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append((route, intent))
        assert is_policy_active(route.route_id, route.policy.policy_digest) is True
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        adapter_observations.append(bound_group)
        response = await dispatch(bound_group.intents[0])
        assert response.status_code == 200
        return DiscoveryAdapterResult(
            candidates=(
                DiscoveryCandidate(
                    candidate_id="paper-1",
                    record={"title": "Accounted execution"},
                ),
            )
        )

    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    dispatch_ids = iter(("dispatch-happy",))
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda route_id, digest: (route_id == group.route_id and digest == group.policy_digest),
        dispatch_id_factory=lambda: next(dispatch_ids),
        journal=journal,
    )

    assert adapter_observations == [group]
    assert gateway_calls == [(registry.get_route(group.route_id), group.intents[0])]
    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("paper-1",)
    assert result.candidates[0].catalog_source_ids == ("semantic_scholar",)
    assert result.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert journal.records[0].dispatch_id == "dispatch-happy"
    assert journal.records[0].state is PhysicalDispatchState.SUCCEEDED
    assert journal.accounting.debited == 1


def test_attempt_journal_supports_all_debited_terminal_states_and_rejects_release_after_debit() -> None:
    terminal_transitions = (
        ("mark_failed", PhysicalDispatchState.FAILED),
        ("mark_timed_out", PhysicalDispatchState.TIMED_OUT),
        ("mark_indeterminate_after_dispatch", PhysicalDispatchState.INDETERMINATE_AFTER_DISPATCH),
    )
    for method_name, expected_state in terminal_transitions:
        journal = AttemptJournal(physical_ceiling=1)
        dispatch_id = f"dispatch-{expected_state.value}"
        journal.reserve(
            dispatch_id=dispatch_id,
            dispatch_group_id="group-1",
            route_id="route-1",
            operation_kind=OperationKind.SEARCH,
        )
        journal.mark_dispatching(dispatch_id)

        terminal = getattr(journal, method_name)(dispatch_id)

        assert terminal.state is expected_state
        assert journal.accounting.debited == 1
        with pytest.raises(ValueError, match="invalid_dispatch_state_transition"):
            journal.release(dispatch_id, PhysicalDispatchState.CANCELLED)


def test_attempt_journal_allows_zero_ceiling_and_rejects_collision_with_released_id() -> None:
    zero = AttemptJournal(physical_ceiling=0)
    assert zero.accounting == DispatchAccounting(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="physical_dispatch_ceiling_exhausted"):
        zero.reserve(
            dispatch_id="dispatch-impossible",
            dispatch_group_id="group-1",
            route_id="route-1",
            operation_kind=OperationKind.SEARCH,
        )

    journal = AttemptJournal(physical_ceiling=1)
    journal.reserve(
        dispatch_id="dispatch-released",
        dispatch_group_id="group-1",
        route_id="route-1",
        operation_kind=OperationKind.SEARCH,
    )
    journal.release("dispatch-released", PhysicalDispatchState.SKIPPED)
    with pytest.raises(ValueError, match="duplicate_dispatch_id"):
        journal.reserve(
            dispatch_id="dispatch-released",
            dispatch_group_id="group-1",
            route_id="route-1",
            operation_kind=OperationKind.SEARCH,
        )


@pytest.mark.asyncio
async def test_retained_capability_cannot_reopen_during_or_after_later_group() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    retained = {}
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append((route.route_id, intent.path))
        return _gateway_response(route, intent)

    async def first_adapter(group, dispatch):
        retained["first"] = dispatch
        await dispatch(group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    async def second_adapter(group, dispatch):
        retained["second_intent"] = group.intents[0]
        before = (len(journal.records), len(gateway_calls))
        with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
            await retained["first"](group.intents[0])
        assert caught.value.code == "dispatch_capability_closed"
        assert (len(journal.records), len(gateway_calls)) == before
        await dispatch(group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: first_adapter,
            groups["crossref"].adapter_id: second_adapter,
        },
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-1", "dispatch-2")).__next__,
        journal=journal,
    )

    before = (len(journal.records), len(gateway_calls))
    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await retained["first"](retained["second_intent"])
    assert caught.value.code == "dispatch_capability_closed"
    assert (len(journal.records), len(gateway_calls)) == before == (2, 2)


@pytest.mark.asyncio
async def test_adapter_facing_dispatch_exposes_only_its_callable_public_surface() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    visible = set()
    forbidden = {
        "_gateway",
        "_journal",
        "_registry",
        "_trusted_group",
        "_used",
        "close",
        "closed",
    }

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        visible.update(name for name in forbidden if hasattr(dispatch, name))
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-hidden-controller",
    )

    assert visible == set()


@pytest.mark.asyncio
async def test_retained_dispatch_lifetime_ignores_unrelated_function_attributes() -> None:
    registry, plan = _foundation_plan(("pubmed",))
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    retained = {}
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(intent.path)
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        retained["dispatch"] = dispatch
        retained["unused_intent"] = bound_group.intents[1]
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    dispatch_ids = iter(("dispatch-first", "dispatch-must-not-be-used"))
    await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
        journal=journal,
    )

    dispatch = retained["dispatch"]
    dispatch.closed = False
    dispatch.close = lambda: None
    dispatch._gateway = lambda *args, **kwargs: gateway_calls.append("bypassed")
    dispatch._journal = object()
    dispatch._registry = object()
    dispatch._trusted_group = object()
    dispatch._used = set()
    before = (len(journal.records), len(gateway_calls))

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await dispatch(retained["unused_intent"])

    assert caught.value.code == "dispatch_capability_closed"
    assert (len(journal.records), len(gateway_calls)) == before == (1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("query", "body", "group"))
async def test_adapter_mutation_of_exposed_plan_copy_rejects_before_reservation(mutation: str) -> None:
    source_id = "figshare" if mutation == "body" else "semantic_scholar"
    registry, plan = _foundation_plan((source_id,))
    trusted_group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    gateway_calls = 0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal gateway_calls
        gateway_calls += 1
        return _gateway_response(route, intent)

    async def adapter(group, dispatch):
        assert group is not trusted_group
        assert group.intents[0] is not trusted_group.intents[0]
        if mutation == "query":
            object.__setattr__(group.intents[0].query_pairs[0], "value", "secret-mutated-query")
        elif mutation == "body":
            object.__setattr__(group.intents[0].json_body_pairs[0], "value", "secret-mutated-body")
        else:
            object.__setattr__(group, "route_id", "mutated.route")
        with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
            await dispatch(group.intents[0])
        assert caught.value.code == "bound_plan_mutated"
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={trusted_group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "must-not-be-created",
        journal=journal,
    )

    assert journal.records == ()
    assert gateway_calls == 0
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "bound_plan_mutated"
    assert "secret-mutated" not in repr(result)


@pytest.mark.asyncio
async def test_registry_is_revalidated_after_adapter_before_candidate_commit() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)

    async def gateway(bound_route, intent, *, is_policy_active):
        return _gateway_response(bound_route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        object.__setattr__(route, "backend_id", "mutated_backend")
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("must-not-commit", {"title": "secret-provider-record"}),)
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-success-before-revocation",
        journal=journal,
    )

    assert journal.records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "registry_mismatch"
    assert "secret-provider-record" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    (
        "route_id",
        "policy_digest",
        "method",
        "path",
        "query_keys",
        "query_keys_type",
        "query_key_scalar",
        "status_code",
    ),
)
async def test_gateway_trace_mismatch_fails_debited_dispatch_before_success(mismatch: str) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)

    async def gateway(route, intent, *, is_policy_active):
        response = _gateway_response(route, intent)
        if mismatch == "status_code":
            return replace(response, status_code=201)
        if mismatch == "query_keys":
            return replace(response, trace=replace(response.trace, query_keys=("mismatch",)))
        if mismatch == "query_keys_type":
            return replace(
                response,
                trace=replace(response.trace, query_keys=list(response.trace.query_keys)),
            )
        if mismatch == "query_key_scalar":
            return replace(response, trace=replace(response.trace, query_keys=(1,)))
        replacement = "0" * 64 if mismatch == "policy_digest" else "mismatch"
        return replace(response, trace=replace(response.trace, **{mismatch: replacement}))

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-trace-mismatch",
        journal=journal,
    )

    assert journal.records[0].state is PhysicalDispatchState.FAILED
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "gateway_response_mismatch"


@pytest.mark.asyncio
async def test_injected_journal_ceiling_must_equal_plan_ceiling_before_work() -> None:
    registry, plan = _semantic_scholar_plan()
    calls = 0

    async def should_not_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={plan.dispatch_groups[0].adapter_id: should_not_run},
            gateway=should_not_run,
            policy_is_active=lambda _route_id, _digest: True,
            journal=AttemptJournal(physical_ceiling=2),
        )

    assert caught.value.code == "journal_ceiling_mismatch"
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "journal_case",
    (
        "duck",
        "subclass",
        "reserved",
        "released",
        "debited",
        "bool_ceiling",
        "incorrect_ceiling",
        "hostile_records",
    ),
)
async def test_injected_journal_must_be_exact_pristine_and_match_live_ceiling(
    journal_case: str,
) -> None:
    registry, plan = (
        _foundation_plan(("semantic_scholar", "crossref")) if journal_case == "duck" else _semantic_scholar_plan()
    )
    if journal_case == "duck":
        object.__setattr__(plan.ceilings, "max_physical_dispatches", 1)

    class LyingJournal:
        physical_ceiling = 1

        @property
        def records(self):
            return ()

        @property
        def accounting(self):
            return DispatchAccounting(0, 0, 0, 0, 1)

        def is_pristine(self, *, physical_ceiling):
            return True

        def reserve(self, **kwargs):
            return None

        def release(self, *args, **kwargs):
            return None

        def mark_dispatching(self, dispatch_id):
            return None

        def mark_succeeded(self, dispatch_id):
            return None

        def mark_failed(self, dispatch_id):
            return None

        def mark_timed_out(self, dispatch_id):
            return None

        def mark_indeterminate_after_dispatch(self, dispatch_id):
            return None

    class LyingJournalSubclass(AttemptJournal):
        __slots__ = ()

        def is_pristine(self, *, physical_ceiling):
            return True

    class HostileRecords(dict):
        def values(self):
            raise RuntimeError("secret_hostile_records")

    if journal_case == "duck":
        journal = LyingJournal()
    elif journal_case == "subclass":
        journal = LyingJournalSubclass(physical_ceiling=1)
    else:
        journal = AttemptJournal(physical_ceiling=1)
        if journal_case in {"reserved", "released", "debited"}:
            journal.reserve(
                dispatch_id="existing-dispatch",
                dispatch_group_id="existing-group",
                route_id="existing-route",
                operation_kind=OperationKind.SEARCH,
            )
        if journal_case == "released":
            journal.release("existing-dispatch", PhysicalDispatchState.SKIPPED)
        elif journal_case == "debited":
            journal.mark_dispatching("existing-dispatch")
        elif journal_case in {"bool_ceiling", "incorrect_ceiling"}:
            slot = "_physical_ceiling" if hasattr(journal, "_physical_ceiling") else "physical_ceiling"
            object.__setattr__(journal, slot, True if journal_case == "bool_ceiling" else 2)
        elif journal_case == "hostile_records":
            object.__setattr__(journal, "_records", HostileRecords())

    expected_code = (
        "invalid_injected_journal"
        if journal_case in {"duck", "subclass"}
        else (
            "journal_ceiling_mismatch"
            if journal_case in {"bool_ceiling", "incorrect_ceiling"}
            else "journal_not_pristine"
        )
    )
    calls: list[str] = []

    async def adapter(bound_group, dispatch):
        calls.append("adapter")
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        return _gateway_response(route, intent)

    def dispatch_id_factory():
        calls.append("id")
        return f"dispatch-{calls.count('id')}"

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter for group in plan.dispatch_groups},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=dispatch_id_factory,
            journal=journal,
        )

    assert caught.value.code == expected_code
    assert calls == []


@pytest.mark.asyncio
async def test_injected_journal_public_ceiling_cannot_be_raised_between_pages() -> None:
    registry, plan = _paginated_semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_physical_dispatches", 1)
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    gateway_calls: list[int] = []
    assignment_failed = False
    second_dispatch_code = None

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(int({pair.name: pair.value for pair in intent.query_pairs}["offset"]))
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        nonlocal assignment_failed, second_dispatch_code
        await dispatch(bound_group.intents[0])
        try:
            journal.physical_ceiling = 2  # type: ignore[misc]
        except AttributeError:
            assignment_failed = True
        try:
            await dispatch(bound_group.intents[0], cursor=NumericCursor(3))
        except executor_module.DiscoveryExecutionError as error:
            second_dispatch_code = error.code
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-page-1", "dispatch-page-2")).__next__,
        journal=journal,
    )

    assert assignment_failed is True
    assert second_dispatch_code == "physical_dispatch_ceiling_exhausted"
    assert gateway_calls == [0]
    assert result.usage.accounting.physical_ceiling == 1
    assert result.usage.accounting.debited == 1


@pytest.mark.asyncio
async def test_injected_journal_history_cannot_be_cleared_between_pages() -> None:
    registry, plan = _paginated_semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_physical_dispatches", 1)
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    gateway_offsets: list[int] = []
    dispatch_ids = iter(("dispatch-page-1", "dispatch-page-2"))

    async def gateway(route, intent, *, is_policy_active):
        gateway_offsets.append(int({pair.name: pair.value for pair in intent.query_pairs}["offset"]))
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        journal._records.clear()
        await dispatch(bound_group.intents[0], cursor=NumericCursor(3))
        return DiscoveryAdapterResult(candidates=())

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=dispatch_ids.__next__,
            journal=journal,
        )

    assert caught.value.code == "journal_lineage_mismatch"
    assert gateway_offsets == [0]
    assert next(dispatch_ids) == "dispatch-page-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", ("identity", "in_place"))
async def test_injected_journal_record_lineage_cannot_be_rewritten_between_pages(rewrite: str) -> None:
    registry, plan = _paginated_semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_physical_dispatches", 1)
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    calls: list[str] = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        return _gateway_response(route, intent)

    def dispatch_id_factory() -> str:
        calls.append("id")
        return f"dispatch-{calls.count('id')}"

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        if rewrite == "identity":
            journal._records["dispatch-1"] = replace(journal._records["dispatch-1"])
        else:
            object.__setattr__(journal._records["dispatch-1"], "route_id", "forged-route")
        try:
            await dispatch(bound_group.intents[0], cursor=NumericCursor(3))
        except executor_module.DiscoveryExecutionError:
            pass
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("must-not-commit", {"title": "forged lineage"}),))

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=dispatch_id_factory,
            journal=journal,
        )

    assert caught.value.code == "journal_lineage_mismatch"
    assert calls == ["id", "gateway"]


@pytest.mark.asyncio
async def test_injected_journal_cannot_be_mutated_by_dispatch_id_factory() -> None:
    registry, plan = _paginated_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=2)
    calls: list[str] = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        return _gateway_response(route, intent)

    def dispatch_id_factory() -> str:
        calls.append("id")
        journal.reserve(
            dispatch_id="factory-forged",
            dispatch_group_id=group.dispatch_group_id,
            route_id=group.route_id,
            operation_kind=OperationKind.SEARCH,
        )
        return "executor-dispatch"

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=dispatch_id_factory,
            journal=journal,
        )

    assert caught.value.code == "journal_lineage_mismatch"
    assert calls == ["id"]


@pytest.mark.asyncio
async def test_injected_journal_history_cannot_be_cleared_between_groups() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
    object.__setattr__(plan.ceilings, "max_physical_dispatches", 1)
    journal = AttemptJournal(physical_ceiling=1)
    first_group, second_group = plan.dispatch_groups
    calls: list[str] = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append(f"gateway:{route.route_id}")
        return _gateway_response(route, intent)

    async def first_adapter(bound_group, dispatch):
        calls.append("adapter:first")
        await dispatch(bound_group.intents[0])
        journal._records.clear()
        return DiscoveryAdapterResult(candidates=())

    async def second_adapter(bound_group, dispatch):
        calls.append("adapter:second")
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={
                first_group.adapter_id: first_adapter,
                second_group.adapter_id: second_adapter,
            },
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=iter(("dispatch-group-1", "dispatch-group-2")).__next__,
            journal=journal,
        )

    assert caught.value.code == "journal_lineage_mismatch"
    assert calls == ["adapter:first", f"gateway:{first_group.route_id}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    (
        ("ceiling", "journal_ceiling_mismatch"),
        ("records_storage", "journal_accounting_invalid"),
    ),
)
async def test_private_journal_state_corruption_after_dispatch_is_rejected_before_commit(
    corruption: str,
    expected_code: str,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    calls: list[str] = []

    class RecordsSubclass(dict):
        pass

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        calls.append("adapter")
        await dispatch(bound_group.intents[0])
        if corruption == "ceiling":
            object.__setattr__(journal, "_physical_ceiling", 2)
        else:
            object.__setattr__(journal, "_records", RecordsSubclass(journal._records))
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("must-not-commit", {"title": "private corruption"}),)
        )

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: calls.append("id") or "dispatch-private-corruption",
            journal=journal,
        )

    assert caught.value.code == expected_code
    assert calls == ["adapter", "id", "gateway"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_mode", "expected_code"),
    (("missing", "missing_adapter"), ("raising", "adapter_failed"), ("malformed", "malformed_adapter_result")),
)
async def test_adapter_failures_are_sanitized_and_do_not_abort_later_group(
    failure_mode: str,
    expected_code: str,
) -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(route.route_id)
        return _gateway_response(route, intent)

    async def broken_adapter(group, dispatch):
        if failure_mode == "raising":
            raise RuntimeError("secret-adapter-token")
        return {"secret": "malformed-adapter-result"}

    async def good_adapter(group, dispatch):
        await dispatch(group.intents[0])
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("crossref-paper", {"title": "later group survives"}),)
        )

    adapters = {groups["crossref"].adapter_id: good_adapter}
    if failure_mode != "missing":
        adapters[groups["semantic_scholar"].adapter_id] = broken_adapter
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters=adapters,
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-later-group",)).__next__,
        journal=journal,
    )

    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.FAILED,
        LogicalOutcomeState.SUCCEEDED,
    )
    assert result.logical_outcomes[0].code == expected_code
    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("crossref-paper",)
    assert len(gateway_calls) == 1
    assert "secret-adapter" not in repr(result)


@pytest.mark.asyncio
async def test_parse_failure_after_response_keeps_physical_success_and_sanitizes_logical_failure() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        raise ValueError("secret-provider-body-parse-error")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-parse-failure",
        journal=journal,
    )

    assert journal.records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "adapter_failed"
    assert "secret-provider" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gateway_error", "physical_state", "logical_state", "expected_code"),
    (
        (
            RuntimeError("secret-gateway-error"),
            PhysicalDispatchState.FAILED,
            LogicalOutcomeState.FAILED,
            "gateway_failed",
        ),
        (TimeoutError(), PhysicalDispatchState.TIMED_OUT, LogicalOutcomeState.TIMED_OUT, "gateway_timed_out"),
    ),
)
async def test_gateway_failure_after_dispatch_is_debited_and_sanitized(
    gateway_error: Exception,
    physical_state: PhysicalDispatchState,
    logical_state: LogicalOutcomeState,
    expected_code: str,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)

    async def gateway(route, intent, *, is_policy_active):
        raise gateway_error

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-gateway-failure",
        journal=journal,
    )

    assert journal.records[0].state is physical_state
    assert result.logical_outcomes[0].state is logical_state
    assert result.logical_outcomes[0].code == expected_code
    assert "secret-gateway" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_lineage", (False, True))
async def test_real_task_cancellation_during_gateway_retains_debit_and_original_cancel(
    corrupt_lineage: bool,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    gateway_entered = asyncio.Event()
    blocker = asyncio.Event()

    async def gateway(route, intent, *, is_policy_active):
        gateway_entered.set()
        await blocker.wait()
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    execution = asyncio.create_task(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: "dispatch-cancelled-in-flight",
            journal=journal,
        )
    )
    await gateway_entered.wait()
    if corrupt_lineage:
        journal._records.clear()
    execution.cancel("original-cancel")

    with pytest.raises(asyncio.CancelledError) as caught:
        await execution

    assert caught.value.args == ("original-cancel",)
    if corrupt_lineage:
        assert journal.records == ()
    else:
        assert journal.records[0].state is PhysicalDispatchState.INDETERMINATE_AFTER_DISPATCH
        assert journal.accounting.debited == 1


@pytest.mark.asyncio
async def test_real_task_cancellation_outside_dispatch_re_raises_without_record() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    adapter_entered = asyncio.Event()
    blocker = asyncio.Event()

    async def adapter(bound_group, dispatch):
        adapter_entered.set()
        await blocker.wait()
        return DiscoveryAdapterResult(candidates=())

    execution = asyncio.create_task(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
            policy_is_active=lambda _route_id, _digest: True,
            journal=journal,
        )
    )
    await adapter_entered.wait()
    execution.cancel("outer-adapter-cancel")

    with pytest.raises(asyncio.CancelledError) as caught:
        await execution

    assert caught.value.args == ("outer-adapter-cancel",)
    assert journal.records == ()


@pytest.mark.asyncio
async def test_outer_task_cancellation_wins_when_gateway_child_completes_same_turn() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    execution: asyncio.Task | None = None

    async def gateway(route, intent, *, is_policy_active):
        assert execution is not None
        execution.cancel("cancel-raced-done-child")
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    execution = asyncio.create_task(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: "dispatch-cancel-race",
            journal=journal,
        )
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await execution

    assert caught.value.args == ("cancel-raced-done-child",)
    assert journal.records[0].state is PhysicalDispatchState.INDETERMINATE_AFTER_DISPATCH
    assert journal.accounting.debited == 1


@pytest.mark.asyncio
async def test_aggregate_timeout_suppresses_child_response_returned_during_cancellation() -> None:
    registry, plan = _semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_wall_time_ms", 1)
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    blocker = asyncio.Event()

    async def gateway(route, intent, *, is_policy_active):
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            return _gateway_response(route, intent)
        raise AssertionError("blocker must stay unset")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("must-not-commit", {"title": "late child result"}),)
        )

    result = await asyncio.wait_for(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: "dispatch-aggregate-timeout",
            journal=journal,
            monotonic_clock=lambda: 0.0,
        ),
        timeout=0.25,
    )

    assert journal.records[0].state is PhysicalDispatchState.TIMED_OUT
    assert result.logical_outcomes[0].state is LogicalOutcomeState.TIMED_OUT
    assert result.logical_outcomes[0].code == "aggregate_deadline_exceeded"
    assert result.candidates == ()


@pytest.mark.asyncio
async def test_outer_cancellation_during_timeout_child_cleanup_remains_indeterminate() -> None:
    registry, plan = _semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_wall_time_ms", 1)
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    cleanup_started = asyncio.Event()
    child_finished = asyncio.Event()
    blocker = asyncio.Event()
    gateway_task = None

    async def gateway(route, intent, *, is_policy_active):
        nonlocal gateway_task
        gateway_task = asyncio.current_task()
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            try:
                await blocker.wait()
            finally:
                child_finished.set()
            raise AssertionError("cleanup blocker must stay unset") from None
        raise AssertionError("gateway blocker must stay unset")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    execution = asyncio.create_task(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: "dispatch-timeout-cleanup-cancel",
            journal=journal,
            monotonic_clock=lambda: 0.0,
        )
    )
    await cleanup_started.wait()
    execution.cancel("cancel-during-timeout-cleanup")

    with pytest.raises(asyncio.CancelledError) as caught:
        await execution

    assert caught.value.args == ("cancel-during-timeout-cleanup",)
    assert child_finished.is_set()
    assert gateway_task is not None and gateway_task.done()
    assert journal.records[0].state is PhysicalDispatchState.INDETERMINATE_AFTER_DISPATCH
    assert journal.accounting.debited == 1


@pytest.mark.asyncio
async def test_repeated_outer_cancellation_still_drains_timeout_child() -> None:
    registry, plan = _semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_wall_time_ms", 1)
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    cancellations_seen = [asyncio.Event(), asyncio.Event()]
    blocker = asyncio.Event()
    gateway_task = None

    async def gateway(route, intent, *, is_policy_active):
        nonlocal gateway_task
        gateway_task = asyncio.current_task()
        for cancellation_seen in cancellations_seen:
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
        await blocker.wait()
        raise AssertionError("gateway blocker must stay unset")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    execution = asyncio.create_task(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: "dispatch-repeated-cancel",
            journal=journal,
            monotonic_clock=lambda: 0.0,
        )
    )
    await cancellations_seen[0].wait()
    execution.cancel("first-outer-cancel")
    await cancellations_seen[1].wait()
    execution.cancel("second-outer-cancel")

    with pytest.raises(asyncio.CancelledError) as caught:
        await execution

    assert caught.value.args == ("first-outer-cancel",)
    assert gateway_task is not None and gateway_task.done()
    assert journal.records[0].state is PhysicalDispatchState.INDETERMINATE_AFTER_DISPATCH
    assert journal.accounting.debited == 1


def test_executor_avoids_python_311_only_task_cancellation_apis() -> None:
    tree = ast.parse(inspect.getsource(executor_module))

    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cancelling"
        for node in ast.walk(tree)
    )


def test_executor_catches_distinct_python_310_asyncio_timeout_error() -> None:
    tree = ast.parse(inspect.getsource(executor_module))

    def exception_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return None

    assert any(
        isinstance(handler.type, ast.Tuple)
        and {exception_name(item) for item in handler.type.elts} >= {"TimeoutError", "asyncio.TimeoutError"}
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler))
    )


@pytest.mark.asyncio
async def test_immediate_gateway_timeout_remains_gateway_timeout_not_aggregate() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)

    async def gateway(route, intent, *, is_policy_active):
        raise TimeoutError("provider timeout")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-inner-timeout",
        journal=journal,
    )

    assert journal.records[0].state is PhysicalDispatchState.TIMED_OUT
    assert result.logical_outcomes[0].code == "gateway_timed_out"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancellation_behavior", "expected_code"),
    (
        ("cancel", "execution_cancelled"),
        ("raise", "cancellation_check_failed"),
        ("non_bool", "cancellation_check_failed"),
    ),
)
async def test_execution_control_startup_cancellation_precedes_clock_and_projects_all_groups(
    cancellation_behavior: str,
    expected_code: str,
) -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref", "arxiv"))
    cancellation_calls = 0

    def cancellation_check():
        nonlocal cancellation_calls
        cancellation_calls += 1
        if cancellation_behavior == "raise":
            raise RuntimeError("secret cancellation failure")
        if cancellation_behavior == "non_bool":
            return 1
        return True

    def clock():
        raise AssertionError("cancellation must be checked before the clock")

    async def forbidden_adapter(bound_group, dispatch):
        raise AssertionError("adapters must not run after a startup stop")

    async def forbidden_gateway(route, intent, *, is_policy_active):
        raise AssertionError("gateway must not run after a startup stop")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: forbidden_adapter for group in plan.dispatch_groups},
        gateway=forbidden_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        cancellation_check=cancellation_check,
        monotonic_clock=clock,
    )

    assert cancellation_calls == 1
    assert result.usage.physical_records == ()
    assert result.usage.route_attempts == 0
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.CANCELLED,
        LogicalOutcomeState.CANCELLED,
        LogicalOutcomeState.CANCELLED,
    )
    assert {outcome.code for outcome in result.logical_outcomes} == {expected_code}


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_clock", ("bool", "nan", "infinite", "backward"))
async def test_execution_control_invalid_or_backward_clock_fails_closed_globally(
    invalid_clock: str,
) -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref", "arxiv"))
    adapter_calls = 0
    id_calls = 0
    values = iter((1.0, 0.0))

    def clock():
        if invalid_clock == "bool":
            return True
        if invalid_clock == "nan":
            return float("nan")
        if invalid_clock == "infinite":
            return float("inf")
        return next(values)

    def dispatch_id_factory():
        nonlocal id_calls
        id_calls += 1
        return "must-not-reserve"

    async def adapter(bound_group, dispatch):
        nonlocal adapter_calls
        adapter_calls += 1
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    async def forbidden_gateway(route, intent, *, is_policy_active):
        raise AssertionError("gateway must not run with an invalid execution clock")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter for group in plan.dispatch_groups},
        gateway=forbidden_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
        monotonic_clock=clock,
    )

    assert adapter_calls == (1 if invalid_clock == "backward" else 0)
    assert id_calls == 0
    assert result.usage.physical_records == ()
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.FAILED,
        LogicalOutcomeState.FAILED,
        LogicalOutcomeState.FAILED,
    )
    assert {outcome.code for outcome in result.logical_outcomes} == {"execution_clock_invalid"}


@pytest.mark.asyncio
async def test_execution_control_deadline_equality_stops_before_runtime_work() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
    object.__setattr__(plan.ceilings, "max_wall_time_ms", 0)

    async def forbidden_adapter(bound_group, dispatch):
        raise AssertionError("adapter must not run at the aggregate deadline")

    async def forbidden_gateway(route, intent, *, is_policy_active):
        raise AssertionError("gateway must not run at the aggregate deadline")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: forbidden_adapter for group in plan.dispatch_groups},
        gateway=forbidden_gateway,
        policy_is_active=lambda _route_id, _digest: True,
        cancellation_check=lambda: False,
        monotonic_clock=lambda: 0.0,
    )

    assert result.usage.physical_records == ()
    assert result.usage.route_attempts == 0
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.TIMED_OUT,
        LogicalOutcomeState.TIMED_OUT,
    )
    assert {outcome.code for outcome in result.logical_outcomes} == {"aggregate_deadline_exceeded"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_stop", "expected_record_state", "expected_outcome_state", "expected_code"),
    (
        (
            "cancellation",
            PhysicalDispatchState.CANCELLED,
            LogicalOutcomeState.CANCELLED,
            "execution_cancelled",
        ),
        (
            "deadline",
            PhysicalDispatchState.SKIPPED,
            LogicalOutcomeState.TIMED_OUT,
            "aggregate_deadline_exceeded",
        ),
        (
            "invalid_clock",
            PhysicalDispatchState.SKIPPED,
            LogicalOutcomeState.FAILED,
            "execution_clock_invalid",
        ),
    ),
)
async def test_execution_control_post_reserve_stop_releases_unused_reservation(
    runtime_stop: str,
    expected_record_state: PhysicalDispatchState,
    expected_outcome_state: LogicalOutcomeState,
    expected_code: str,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    reservation_started = False
    policy_calls = 0

    def cancellation_check():
        return runtime_stop == "cancellation" and reservation_started

    def clock():
        if not reservation_started:
            return 0.0
        if runtime_stop == "deadline":
            return plan.ceilings.max_wall_time_ms / 1000
        if runtime_stop == "invalid_clock":
            return float("nan")
        return 0.0

    def policy_is_active(_route_id, _digest):
        nonlocal policy_calls, reservation_started
        policy_calls += 1
        if policy_calls == 2:
            reservation_started = True
        return True

    async def forbidden_gateway(route, intent, *, is_policy_active):
        raise AssertionError("gateway must not run after a post-reserve stop")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=forbidden_gateway,
        policy_is_active=policy_is_active,
        dispatch_id_factory=lambda: "dispatch-unused-after-stop",
        journal=journal,
        cancellation_check=cancellation_check,
        monotonic_clock=clock,
    )

    assert policy_calls == 2
    assert journal.records[0].state is expected_record_state
    assert journal.accounting.released == 1
    assert result.logical_outcomes[0].state is expected_outcome_state
    assert result.logical_outcomes[0].code == expected_code
    assert result.candidates == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy_mutation", "expected_state", "expected_code"),
    (
        ("cancellation", LogicalOutcomeState.CANCELLED, "execution_cancelled"),
        ("invalid_clock", LogicalOutcomeState.FAILED, "execution_clock_invalid"),
    ),
)
async def test_false_policy_result_cannot_hide_latched_stop_from_later_groups(
    policy_mutation: str,
    expected_state: LogicalOutcomeState,
    expected_code: str,
) -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    stopped = False
    adapter_calls = []

    def policy_is_active(route_id, _digest):
        nonlocal stopped
        if route_id == groups["semantic_scholar"].route_id and not stopped:
            stopped = True
            return False
        return True

    def clock():
        if policy_mutation == "invalid_clock" and stopped:
            return float("nan")
        return 0.0

    async def first_adapter(bound_group, dispatch):
        adapter_calls.append("semantic_scholar")
        await dispatch(bound_group.intents[0])
        raise AssertionError("stopped dispatch must not return")

    async def forbidden_second_adapter(bound_group, dispatch):
        adapter_calls.append("crossref")
        raise AssertionError("later adapter must not start after a policy callback stop")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: first_adapter,
            groups["crossref"].adapter_id: forbidden_second_adapter,
        },
        gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
        policy_is_active=policy_is_active,
        cancellation_check=lambda: policy_mutation == "cancellation" and stopped,
        monotonic_clock=clock,
    )

    assert adapter_calls == ["semantic_scholar"]
    assert result.usage.physical_records == ()
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (expected_state, expected_state)
    assert {outcome.code for outcome in result.logical_outcomes} == {expected_code}


@pytest.mark.asyncio
async def test_post_reserve_false_policy_mutation_releases_as_cancelled_stop() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    cancelled = False
    policy_calls = 0

    def policy_is_active(_route_id, _digest):
        nonlocal cancelled, policy_calls
        policy_calls += 1
        if policy_calls == 2:
            cancelled = True
            return False
        return True

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        raise AssertionError("cancelled dispatch must not return")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
        policy_is_active=policy_is_active,
        dispatch_id_factory=lambda: "dispatch-false-policy-cancel",
        journal=journal,
        cancellation_check=lambda: cancelled,
        monotonic_clock=lambda: 0.0,
    )

    assert policy_calls == 2
    assert journal.records[0].state is PhysicalDispatchState.CANCELLED
    assert journal.accounting.released == 1
    assert result.logical_outcomes[0].state is LogicalOutcomeState.CANCELLED
    assert result.logical_outcomes[0].code == "execution_cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("bound_group", "bound_plan_mutated"),
        ("registry_route", "registry_mismatch"),
    ),
)
async def test_post_reserve_policy_mutation_releases_before_dispatch(
    mutation: str,
    expected_code: str,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    journal = AttemptJournal(physical_ceiling=1)
    policy_calls = 0
    gateway_calls = 0
    exposed_group = None

    def policy_is_active(_route_id, _digest):
        nonlocal policy_calls
        policy_calls += 1
        if policy_calls == 2:
            if mutation == "bound_group":
                object.__setattr__(exposed_group, "backend_id", "mutated_backend")
            else:
                object.__setattr__(route, "backend_id", "mutated_backend")
        return True

    async def gateway(bound_route, intent, *, is_policy_active):
        nonlocal gateway_calls
        gateway_calls += 1
        return _gateway_response(bound_route, intent)

    async def adapter(bound_group, dispatch):
        nonlocal exposed_group
        exposed_group = bound_group
        await dispatch(bound_group.intents[0])
        raise AssertionError("mutated dispatch must not return")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=policy_is_active,
        dispatch_id_factory=lambda: "dispatch-policy-mutation",
        journal=journal,
        monotonic_clock=lambda: 0.0,
    )

    assert policy_calls == 2
    assert gateway_calls == 0
    assert journal.records[0].state is PhysicalDispatchState.SKIPPED
    assert journal.accounting.released == 1
    assert journal.accounting.outstanding == 0
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_lineage", (False, True))
async def test_post_reserve_policy_cancel_releases_and_preserves_original_cancel(
    corrupt_lineage: bool,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    policy_calls = 0

    def policy_is_active(_route_id, _digest):
        nonlocal policy_calls
        policy_calls += 1
        if policy_calls == 2:
            if corrupt_lineage:
                journal._records.clear()
            raise asyncio.CancelledError("policy-cancel")
        return True

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    with pytest.raises(asyncio.CancelledError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
            policy_is_active=policy_is_active,
            dispatch_id_factory=lambda: "dispatch-policy-cancel",
            journal=journal,
            monotonic_clock=lambda: 0.0,
        )

    assert caught.value.args == ("policy-cancel",)
    if corrupt_lineage:
        assert journal.records == ()
    else:
        assert journal.records[0].state is PhysicalDispatchState.CANCELLED
        assert journal.accounting.released == 1
        assert journal.accounting.outstanding == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_stop", ("cancellation", "deadline"))
async def test_normal_adapter_exception_cannot_hide_global_stop(
    runtime_stop: str,
) -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    stopped = False
    adapter_calls = []

    def clock():
        if runtime_stop == "deadline" and stopped:
            return plan.ceilings.max_wall_time_ms / 1000
        return 0.0

    async def first_adapter(bound_group, dispatch):
        nonlocal stopped
        adapter_calls.append("semantic_scholar")
        stopped = True
        raise RuntimeError("secret adapter failure after stop")

    async def forbidden_second_adapter(bound_group, dispatch):
        adapter_calls.append("crossref")
        raise AssertionError("later adapter must not start after the global stop")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: first_adapter,
            groups["crossref"].adapter_id: forbidden_second_adapter,
        },
        gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
        policy_is_active=lambda _route_id, _digest: True,
        cancellation_check=lambda: runtime_stop == "cancellation" and stopped,
        monotonic_clock=clock,
    )

    expected_state = LogicalOutcomeState.CANCELLED if runtime_stop == "cancellation" else LogicalOutcomeState.TIMED_OUT
    expected_code = "execution_cancelled" if runtime_stop == "cancellation" else "aggregate_deadline_exceeded"
    assert adapter_calls == ["semantic_scholar"]
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (expected_state, expected_state)
    assert {outcome.code for outcome in result.logical_outcomes} == {expected_code}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory_behavior", "expected_state", "expected_code"),
    (
        ("raise_after_cancel", LogicalOutcomeState.CANCELLED, "execution_cancelled"),
        ("invalid_after_clock", LogicalOutcomeState.FAILED, "execution_clock_invalid"),
    ),
)
async def test_dispatch_id_factory_failure_cannot_hide_global_stop(
    factory_behavior: str,
    expected_state: LogicalOutcomeState,
    expected_code: str,
) -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    stopped = False
    adapter_calls = []

    def dispatch_id_factory():
        nonlocal stopped
        stopped = True
        if factory_behavior == "raise_after_cancel":
            raise RuntimeError("secret id failure after cancellation")
        return ""

    def clock():
        if factory_behavior == "invalid_after_clock" and stopped:
            return float("nan")
        return 0.0

    async def first_adapter(bound_group, dispatch):
        adapter_calls.append("semantic_scholar")
        await dispatch(bound_group.intents[0])
        raise AssertionError("stopped dispatch must not return")

    async def forbidden_second_adapter(bound_group, dispatch):
        adapter_calls.append("crossref")
        raise AssertionError("later adapter must not start after the global stop")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: first_adapter,
            groups["crossref"].adapter_id: forbidden_second_adapter,
        },
        gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
        cancellation_check=lambda: factory_behavior == "raise_after_cancel" and stopped,
        monotonic_clock=clock,
    )

    assert adapter_calls == ["semantic_scholar"]
    assert result.usage.physical_records == ()
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (expected_state, expected_state)
    assert {outcome.code for outcome in result.logical_outcomes} == {expected_code}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy_call", "mutation", "expected_code"),
    (
        (3, "bound_group", "bound_plan_mutated"),
        (3, "registry_route", "registry_mismatch"),
        (4, "bound_group", "bound_plan_mutated"),
        (4, "registry_route", "registry_mismatch"),
    ),
)
async def test_root_policy_mutation_suppresses_candidate_commit(
    policy_call: int,
    mutation: str,
    expected_code: str,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    journal = AttemptJournal(physical_ceiling=1)
    policy_calls = 0
    exposed_group = None

    def policy_is_active(_route_id, _digest):
        nonlocal policy_calls
        policy_calls += 1
        if policy_calls == policy_call:
            if mutation == "bound_group":
                object.__setattr__(exposed_group, "backend_id", "mutated_backend")
            else:
                object.__setattr__(route, "backend_id", "mutated_backend")
        return True

    async def gateway(bound_route, intent, *, is_policy_active):
        return _gateway_response(bound_route, intent)

    async def adapter(bound_group, dispatch):
        nonlocal exposed_group
        exposed_group = bound_group
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("must-not-commit", {"title": "policy mutation"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=policy_is_active,
        dispatch_id_factory=lambda: "dispatch-before-policy-mutation",
        journal=journal,
        monotonic_clock=lambda: 0.0,
    )

    assert policy_calls == policy_call
    assert journal.records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == expected_code


@pytest.mark.asyncio
async def test_final_valid_clock_callback_mutation_suppresses_candidate_commit() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    policy_calls = 0
    exposed_group = None
    clock_mutated = False

    def policy_is_active(_route_id, _digest):
        nonlocal policy_calls
        policy_calls += 1
        return True

    def clock():
        nonlocal clock_mutated
        if policy_calls == 4 and exposed_group is not None and not clock_mutated:
            object.__setattr__(exposed_group, "backend_id", "mutated_backend")
            clock_mutated = True
        return 0.0

    async def gateway(bound_route, intent, *, is_policy_active):
        return _gateway_response(bound_route, intent)

    async def adapter(bound_group, dispatch):
        nonlocal exposed_group
        exposed_group = bound_group
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("must-not-commit", {"title": "clock mutation"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=policy_is_active,
        dispatch_id_factory=lambda: "dispatch-before-clock-mutation",
        journal=journal,
        monotonic_clock=clock,
    )

    assert policy_calls == 4
    assert clock_mutated is True
    assert journal.records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "bound_plan_mutated"


@pytest.mark.asyncio
async def test_execution_control_enormous_wall_time_fails_with_typed_clock_stop() -> None:
    registry, plan = _semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_wall_time_ms", 10**1000)
    group = plan.dispatch_groups[0]

    async def forbidden_adapter(bound_group, dispatch):
        raise AssertionError("adapter must not run with an unrepresentable deadline")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: forbidden_adapter},
        gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
        policy_is_active=lambda _route_id, _digest: True,
        monotonic_clock=lambda: 0.0,
    )

    assert result.usage.physical_records == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "execution_clock_invalid"


@pytest.mark.asyncio
async def test_late_cancellation_preserves_prior_commit_and_projects_remaining_groups() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref", "arxiv"))
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    cancelled = False
    adapter_calls = []
    id_calls = []

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def first_adapter(bound_group, dispatch):
        adapter_calls.append("semantic_scholar")
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("preserved", {"title": "preserved prior result"}),)
        )

    async def second_adapter(bound_group, dispatch):
        nonlocal cancelled
        adapter_calls.append("crossref")
        cancelled = True
        await dispatch(bound_group.intents[0])
        raise AssertionError("cancelled dispatch must not return")

    async def forbidden_third_adapter(bound_group, dispatch):
        adapter_calls.append("arxiv")
        raise AssertionError("later adapters must not run after the latched stop")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: first_adapter,
            groups["crossref"].adapter_id: second_adapter,
            groups["arxiv"].adapter_id: forbidden_third_adapter,
        },
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: id_calls.append("id") or f"dispatch-{len(id_calls)}",
        cancellation_check=lambda: cancelled,
        monotonic_clock=lambda: 0.0,
    )

    assert adapter_calls == ["semantic_scholar", "crossref"]
    assert id_calls == ["id"]
    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("preserved",)
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.SUCCEEDED,
        LogicalOutcomeState.CANCELLED,
        LogicalOutcomeState.CANCELLED,
    )
    assert tuple(outcome.code for outcome in result.logical_outcomes) == (
        None,
        "execution_cancelled",
        "execution_cancelled",
    )


@pytest.mark.asyncio
async def test_real_wait_timeout_latches_with_static_clock_and_preserves_prior_commit() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref", "arxiv"))
    object.__setattr__(plan.ceilings, "max_wall_time_ms", 50)
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    blocker = asyncio.Event()
    adapter_calls = []
    id_calls = []

    async def gateway(route, intent, *, is_policy_active):
        if route.route_id == groups["semantic_scholar"].route_id:
            return _gateway_response(route, intent)
        await blocker.wait()
        raise AssertionError("blocked gateway must be cancelled")

    async def first_adapter(bound_group, dispatch):
        adapter_calls.append("semantic_scholar")
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("preserved-before-timeout", {"title": "preserved"}),)
        )

    async def timed_out_adapter(bound_group, dispatch):
        adapter_calls.append("crossref")
        with pytest.raises(executor_module.DiscoveryExecutionError) as first:
            await dispatch(bound_group.intents[0])
        assert first.value.code == "aggregate_deadline_exceeded"
        with pytest.raises(executor_module.DiscoveryExecutionError) as second:
            await dispatch(bound_group.intents[0])
        assert second.value.code == "aggregate_deadline_exceeded"
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("must-not-commit", {"title": "caught timeout"}),))

    async def forbidden_third_adapter(bound_group, dispatch):
        adapter_calls.append("arxiv")
        raise AssertionError("later adapter must not run after a real wait timeout")

    result = await asyncio.wait_for(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={
                groups["semantic_scholar"].adapter_id: first_adapter,
                groups["crossref"].adapter_id: timed_out_adapter,
                groups["arxiv"].adapter_id: forbidden_third_adapter,
            },
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: id_calls.append("id") or f"dispatch-{len(id_calls)}",
            monotonic_clock=lambda: 0.0,
        ),
        timeout=1.0,
    )

    assert adapter_calls == ["semantic_scholar", "crossref"]
    assert id_calls == ["id", "id"]
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.TIMED_OUT,
    )
    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("preserved-before-timeout",)
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.SUCCEEDED,
        LogicalOutcomeState.TIMED_OUT,
        LogicalOutcomeState.TIMED_OUT,
    )
    assert tuple(outcome.code for outcome in result.logical_outcomes) == (
        None,
        "aggregate_deadline_exceeded",
        "aggregate_deadline_exceeded",
    )


@pytest.mark.asyncio
async def test_late_cancellation_wins_at_real_wait_timeout_terminal_boundary() -> None:
    registry, plan = _semantic_scholar_plan()
    object.__setattr__(plan.ceilings, "max_wall_time_ms", 1)
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    cancelled = False
    blocker = asyncio.Event()

    async def gateway(route, intent, *, is_policy_active):
        nonlocal cancelled
        cancelled = True
        await blocker.wait()
        raise AssertionError("blocked gateway must be cancelled")

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        raise AssertionError("cancelled timeout dispatch must not return")

    result = await asyncio.wait_for(
        execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: "dispatch-timeout-cancel-precedence",
            journal=journal,
            cancellation_check=lambda: cancelled,
            monotonic_clock=lambda: 0.0,
        ),
        timeout=0.25,
    )

    assert journal.records[0].state is PhysicalDispatchState.TIMED_OUT
    assert result.logical_outcomes[0].state is LogicalOutcomeState.CANCELLED
    assert result.logical_outcomes[0].code == "execution_cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_boundary", ("post_terminal", "after_adapter", "final_commit"))
@pytest.mark.parametrize("runtime_stop", ("cancellation", "deadline"))
async def test_runtime_stop_suppresses_definitive_response_before_candidate_commit(
    stop_boundary: str,
    runtime_stop: str,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=1)
    stopped = False
    adapter_done = False
    post_adapter_policy_checks = 0

    def cancellation_check():
        return runtime_stop == "cancellation" and stopped

    def clock():
        if runtime_stop == "deadline" and stopped:
            return plan.ceilings.max_wall_time_ms / 1000
        return 0.0

    def policy_is_active(_route_id, _digest):
        nonlocal post_adapter_policy_checks, stopped
        if adapter_done:
            post_adapter_policy_checks += 1
            if stop_boundary == "final_commit" and post_adapter_policy_checks == 2:
                stopped = True
        return True

    async def gateway(route, intent, *, is_policy_active):
        nonlocal stopped
        if stop_boundary == "post_terminal":
            stopped = True
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        nonlocal adapter_done, stopped
        await dispatch(bound_group.intents[0])
        adapter_done = True
        if stop_boundary == "after_adapter":
            stopped = True
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("must-not-commit", {"title": "definitive response"}),)
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=policy_is_active,
        dispatch_id_factory=lambda: "dispatch-definitive-response",
        journal=journal,
        cancellation_check=cancellation_check,
        monotonic_clock=clock,
    )

    expected_state = LogicalOutcomeState.CANCELLED if runtime_stop == "cancellation" else LogicalOutcomeState.TIMED_OUT
    expected_code = "execution_cancelled" if runtime_stop == "cancellation" else "aggregate_deadline_exceeded"
    assert journal.records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is expected_state
    assert result.logical_outcomes[0].code == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize("continuation", ("retry", "redirect"))
@pytest.mark.parametrize("runtime_stop", ("cancellation", "deadline"))
async def test_post_terminal_stop_wins_before_retry_or_redirect_decision(
    continuation: str,
    runtime_stop: str,
) -> None:
    if continuation == "retry":
        registry, plan = _retryable_foundation_plan(
            "semantic_scholar",
            route_retries=0,
            budget_retries=0,
        )
    else:
        registry, plan = _redirectable_foundation_plan(
            route_redirects=0,
            budget_redirects=0,
        )
    group = plan.dispatch_groups[0]
    stopped = False
    calls = []

    def cancellation_check():
        return runtime_stop == "cancellation" and stopped

    def clock():
        if runtime_stop == "deadline" and stopped:
            return plan.ceilings.max_wall_time_ms / 1000
        return 0.0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal stopped
        calls.append("gateway")
        stopped = True
        if continuation == "retry":
            raise DiscoveryGatewayError("hop_failed", retryable=True)
        response = _gateway_response(route, intent)
        return replace(
            response,
            status_code=302,
            trace=replace(response.trace, status_code=302),
            redirect_location=f"{intent.path}?limit=3&offset=0&query=accounted+execution",
        )

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        raise AssertionError("stopped continuation must not return")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-terminal-stop",
        cancellation_check=cancellation_check,
        monotonic_clock=clock,
    )

    expected_state = LogicalOutcomeState.CANCELLED if runtime_stop == "cancellation" else LogicalOutcomeState.TIMED_OUT
    expected_code = "execution_cancelled" if runtime_stop == "cancellation" else "aggregate_deadline_exceeded"
    assert calls == ["id", "gateway"]
    assert result.usage.physical_records[0].state is (
        PhysicalDispatchState.FAILED if continuation == "retry" else PhysicalDispatchState.SUCCEEDED
    )
    assert result.logical_outcomes[0].state is expected_state
    assert result.logical_outcomes[0].code == expected_code
    assert result.usage.retries == result.usage.redirects == 0


@pytest.mark.asyncio
async def test_skipped_only_plan_performs_zero_runtime_work_with_zero_ceiling() -> None:
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(("openalex",), "accounted execution", (), 1),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
    )
    journal = AttemptJournal(physical_ceiling=0)
    calls = []

    async def malicious(*args, **kwargs):
        calls.append("called")
        raise AssertionError("must not run")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={"openalex_v2": malicious},
        gateway=malicious,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-impossible",
        journal=journal,
    )

    assert result.skipped == plan.skipped
    assert result.logical_outcomes == ()
    assert result.candidates == ()
    assert journal.records == ()
    assert calls == []


@pytest.mark.asyncio
async def test_fallback_group_is_projected_skipped_without_runtime_work() -> None:
    registry, plan = _fallback_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    calls = []

    async def malicious(*args, **kwargs):
        calls.append("called")
        raise AssertionError("must not run")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: malicious},
        gateway=malicious,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-impossible",
        journal=journal,
    )

    assert result.logical_outcomes[0].state is LogicalOutcomeState.SKIPPED
    assert result.logical_outcomes[0].code == "fallback_not_executed"
    assert journal.records == ()
    assert calls == []


@pytest.mark.asyncio
async def test_attributed_match_truncated_by_global_cap_is_not_valid_empty() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"), result_limit=1)
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    def adapter_for(candidate_id):
        async def adapter(group, dispatch):
            await dispatch(group.intents[0])
            return DiscoveryAdapterResult(candidates=(DiscoveryCandidate(candidate_id, {"title": candidate_id}),))

        return adapter

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: adapter_for("first"),
            groups["crossref"].adapter_id: adapter_for("truncated-but-matched"),
        },
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-first", "dispatch-second")).__next__,
    )

    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("first",)
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.SUCCEEDED,
        LogicalOutcomeState.SUCCEEDED,
    )
    assert result.truncated_candidates == 1


@pytest.mark.asyncio
async def test_initial_search_and_one_typed_page_are_independently_accounted() -> None:
    registry, plan = _paginated_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    gateway_intents = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_intents.append(intent)
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        await dispatch(bound_group.intents[0], cursor=NumericCursor(10))
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-initial", "dispatch-page-2")).__next__,
    )

    assert len(gateway_intents) == 2
    assert gateway_intents[0] == group.intents[0]
    assert gateway_intents[0] is not group.intents[0]
    assert gateway_intents[1] is not gateway_intents[0]
    assert tuple(pair.value for pair in gateway_intents[0].query_pairs if pair.name == "offset") == ("0",)
    assert tuple(pair.value for pair in gateway_intents[1].query_pairs if pair.name == "offset") == ("10",)
    assert isinstance(result.usage, executor_module.DiscoveryExecutionUsage)
    assert tuple(record.dispatch_id for record in result.usage.physical_records) == (
        "dispatch-initial",
        "dispatch-page-2",
    )
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.accounting == DispatchAccounting(2, 2, 0, 0, 2)
    assert (
        result.usage.route_attempts,
        result.usage.pages,
        result.usage.redirects,
        result.usage.retries,
    ) == (1, 2, 0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_code", "expected_work"),
    (
        ("before-initial", "search_not_ready", 0),
        ("repeat-initial", "pagination_cursor_repeated", 1),
        ("repeat-page", "pagination_cursor_repeated", 2),
        ("overflow", "page_ceiling_exhausted", 2),
        ("string", "invalid_pagination_cursor", 1),
    ),
)
async def test_invalid_page_requests_reject_before_id_reservation(
    case: str,
    expected_code: str,
    expected_work: int,
) -> None:
    registry, plan = _paginated_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    gateway_calls = []
    dispatch_ids = []
    observed_errors = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(intent)
        return _gateway_response(route, intent)

    def dispatch_id_factory():
        dispatch_id = f"dispatch-{len(dispatch_ids) + 1}"
        dispatch_ids.append(dispatch_id)
        return dispatch_id

    async def adapter(bound_group, dispatch):
        search = bound_group.intents[0]
        try:
            if case == "before-initial":
                await dispatch(search, cursor=NumericCursor(10))
            else:
                await dispatch(search)
                if case == "repeat-initial":
                    await dispatch(search, cursor=NumericCursor(0))
                elif case == "string":
                    await dispatch(search, cursor="https://evil.example/")
                else:
                    await dispatch(search, cursor=NumericCursor(10))
                    await dispatch(search, cursor=NumericCursor(10 if case == "repeat-page" else 20))
        except Exception as error:  # noqa: BLE001 - assert the sanitized boundary outside the adapter.
            observed_errors.append(error)
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
    )

    assert len(observed_errors) == 1
    assert isinstance(observed_errors[0], executor_module.DiscoveryExecutionError)
    assert observed_errors[0].code == expected_code
    assert len(dispatch_ids) == len(gateway_calls) == expected_work
    assert len(result.usage.physical_records) == result.usage.pages == expected_work
    assert result.usage.accounting.created == result.usage.accounting.debited == expected_work
    assert result.usage.route_attempts == 1
    assert result.usage.redirects == result.usage.retries == 0


@pytest.mark.asyncio
async def test_pubmed_search_then_bound_summary_uses_fresh_grounded_intent() -> None:
    registry, plan = _foundation_plan(("pubmed",))
    group = plan.dispatch_groups[0]
    gateway_intents = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_intents.append(intent)
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        await dispatch(
            bound_group.intents[1],
            bindings=(NumericCSVBindingValues("pubmed_esearch_ids", (1, 2)),),
        )
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-search", "dispatch-summary")).__next__,
    )

    assert len(gateway_intents) == 2
    summary = gateway_intents[1]
    assert summary is not group.intents[1]
    assert summary.operation_kind is OperationKind.CONDITIONAL_SUMMARY
    assert tuple((pair.name, pair.value) for pair in summary.query_pairs) == (
        ("db", "pubmed"),
        ("retmode", "json"),
        ("id", "1,2"),
    )
    assert summary.query_bindings == ()
    assert tuple(record.dispatch_id for record in result.usage.physical_records) == (
        "dispatch-search",
        "dispatch-summary",
    )
    assert result.usage.pages == 1


@pytest.mark.asyncio
async def test_pubmed_summary_missing_binding_rejects_before_second_id() -> None:
    registry, plan = _foundation_plan(("pubmed",))
    group = plan.dispatch_groups[0]
    dispatch_ids = []
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(intent)
        return _gateway_response(route, intent)

    def dispatch_id_factory():
        dispatch_id = f"dispatch-{len(dispatch_ids) + 1}"
        dispatch_ids.append(dispatch_id)
        return dispatch_id

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        await dispatch(bound_group.intents[1])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
    )

    assert dispatch_ids == ["dispatch-1"]
    assert len(gateway_calls) == 1
    assert result.logical_outcomes[0].code == "binding_values_required"
    assert len(result.usage.physical_records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_code", "expected_work"),
    (
        ("wrong_id", "binding_values_mismatch", 1),
        ("duplicate_id", "binding_values_mismatch", 1),
        ("extra_id", "binding_values_mismatch", 1),
        ("too_many", "binding_values_limit_exceeded", 1),
        ("too_long", "binding_values_limit_exceeded", 1),
        ("empty_values", "binding_values_mismatch", 1),
        ("non_integer", "binding_values_mismatch", 1),
        ("nonpositive", "binding_values_mismatch", 1),
        ("bindings_on_search", "bindings_not_allowed", 0),
        ("summary_before_search", "search_not_ready", 0),
        ("cursor_and_binding", "cursor_and_bindings_conflict", 1),
    ),
)
async def test_pubmed_binding_adversaries_reject_before_extra_reservation(
    case: str,
    expected_code: str,
    expected_work: int,
) -> None:
    registry, plan = _foundation_plan(("pubmed",))
    group = plan.dispatch_groups[0]
    declaration = group.intents[1].query_bindings[0]
    dispatch_ids = []
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(intent)
        return _gateway_response(route, intent)

    def dispatch_id_factory():
        dispatch_id = f"dispatch-{len(dispatch_ids) + 1}"
        dispatch_ids.append(dispatch_id)
        return dispatch_id

    valid = NumericCSVBindingValues(declaration.binding_id, (1, 2))
    if case == "wrong_id":
        bindings = (NumericCSVBindingValues("wrong_binding", (1, 2)),)
    elif case == "duplicate_id":
        bindings = (valid, valid)
    elif case == "extra_id":
        bindings = (valid, NumericCSVBindingValues("extra_binding", (3,)))
    elif case == "too_many":
        bindings = (NumericCSVBindingValues(declaration.binding_id, tuple(range(1, declaration.max_items + 2))),)
    elif case == "too_long":
        bindings = (NumericCSVBindingValues(declaration.binding_id, (10**declaration.max_item_chars,)),)
    elif case in {"empty_values", "non_integer", "nonpositive"}:
        hostile = NumericCSVBindingValues(declaration.binding_id, (1,))
        hostile_values = {
            "empty_values": (),
            "non_integer": ("1",),
            "nonpositive": (0,),
        }[case]
        object.__setattr__(hostile, "values", hostile_values)
        bindings = (hostile,)
    else:
        bindings = (valid,)

    async def adapter(bound_group, dispatch):
        search, summary = bound_group.intents
        if case == "bindings_on_search":
            await dispatch(search, bindings=bindings)
        elif case == "summary_before_search":
            await dispatch(summary, bindings=bindings)
        else:
            await dispatch(search)
            await dispatch(
                summary,
                cursor=NumericCursor(1) if case == "cursor_and_binding" else None,
                bindings=bindings,
            )
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
    )

    assert len(dispatch_ids) == len(gateway_calls) == expected_work
    assert len(result.usage.physical_records) == expected_work
    assert result.logical_outcomes[0].code == expected_code


@pytest.mark.asyncio
async def test_pubmed_empty_search_result_may_omit_summary() -> None:
    registry, plan = _foundation_plan(("pubmed",))
    group = plan.dispatch_groups[0]

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-search-only",
    )

    assert len(result.usage.physical_records) == 1
    assert result.usage.pages == 1
    assert result.logical_outcomes[0].state is LogicalOutcomeState.VALID_EMPTY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    ((300, "gateway_redirect_invalid"), (400, "search_not_ready"), (500, "search_not_ready")),
)
async def test_pubmed_non_2xx_search_does_not_authorize_summary(
    status_code: int,
    expected_code: str,
) -> None:
    registry, plan = _foundation_plan(("pubmed",))
    group = plan.dispatch_groups[0]
    dispatch_ids = []

    async def gateway(route, intent, *, is_policy_active):
        response = _gateway_response(route, intent)
        return replace(response, status_code=status_code, trace=replace(response.trace, status_code=status_code))

    def dispatch_id_factory():
        dispatch_id = f"dispatch-{len(dispatch_ids) + 1}"
        dispatch_ids.append(dispatch_id)
        return dispatch_id

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        await dispatch(
            bound_group.intents[1],
            bindings=(NumericCSVBindingValues("pubmed_esearch_ids", (1, 2)),),
        )
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
    )

    assert dispatch_ids == ["dispatch-1"]
    assert result.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.logical_outcomes[0].code == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize("with_candidate", (False, True))
async def test_adapter_result_without_successful_search_is_not_committed(
    with_candidate: bool,
) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    calls = []

    async def should_not_run(*args, **kwargs):
        calls.append("gateway")
        raise AssertionError("must not run")

    async def adapter(bound_group, dispatch):
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("ungrounded", {"title": "must not commit"}),) if with_candidate else ()
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=should_not_run,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
    )

    assert calls == []
    assert result.candidates == ()
    assert result.usage.physical_records == ()
    assert result.logical_outcomes[0].code == "missing_search_dispatch"


@pytest.mark.asyncio
async def test_coalesced_group_counts_logical_attempts_and_one_physical_search() -> None:
    registry, plan = _coalesced_semantic_scholar_plan()
    group = plan.dispatch_groups[0]

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        retained = {
            "source": {"collection": "shared"},
            "items": [{"rank": 1}],
        }
        candidate = DiscoveryCandidate("shared", retained)
        retained["source"]["collection"] = "mutated-after-candidate"
        retained["items"][0]["rank"] = 99
        return DiscoveryAdapterResult(candidates=(candidate,))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-shared",
    )

    assert result.candidates[0].catalog_source_ids == ("semantic_scholar",)
    assert tuple(outcome.state for outcome in result.logical_outcomes) == (
        LogicalOutcomeState.SUCCEEDED,
        LogicalOutcomeState.VALID_EMPTY,
    )
    assert result.candidates[0].record["source"]["collection"] == "shared"
    assert result.candidates[0].record["items"][0]["rank"] == 1
    assert result.usage.route_attempts == 2
    assert len(result.usage.physical_records) == 1
    assert result.usage.accounting == DispatchAccounting(1, 1, 0, 0, 1)


@pytest.mark.asyncio
async def test_runtime_route_attempt_ceiling_rejects_group_before_adapter_or_id() -> None:
    registry, plan = _coalesced_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    object.__setattr__(plan.ceilings, "max_route_attempts", 1)
    calls = []

    async def should_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("must not run")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: should_not_run},
        gateway=should_not_run,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
    )

    assert calls == []
    assert result.usage.route_attempts == 0
    assert result.usage.physical_records == ()
    assert tuple(outcome.code for outcome in result.logical_outcomes) == (
        "route_attempt_ceiling_exhausted",
        "route_attempt_ceiling_exhausted",
    )


@pytest.mark.asyncio
async def test_duplicate_dispatch_id_on_page_rejects_before_second_gateway() -> None:
    registry, plan = _paginated_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(intent)
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        await dispatch(bound_group.intents[0], cursor=NumericCursor(10))
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-collision",
    )

    assert len(gateway_calls) == 1
    assert len(result.usage.physical_records) == 1
    assert result.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.logical_outcomes[0].code == "duplicate_dispatch_id"


@pytest.mark.asyncio
@pytest.mark.parametrize("timed_out", (False, True))
async def test_typed_retryable_gateway_failure_retries_same_get_intent(
    timed_out: bool,
) -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    gateway_intents = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_intents.append(intent)
        if len(gateway_intents) == 1:
            raise DiscoveryGatewayError("hop_failed", retryable=True, timed_out=timed_out)
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        response = await dispatch(bound_group.intents[0])
        assert response.status_code == 200
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("retry-grounded", {"title": "retry grounded"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-first", "dispatch-retry")).__next__,
    )

    assert len(gateway_intents) == 2
    assert gateway_intents[0] is gateway_intents[1]
    assert tuple(record.dispatch_id for record in result.usage.physical_records) == (
        "dispatch-first",
        "dispatch-retry",
    )
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.TIMED_OUT if timed_out else PhysicalDispatchState.FAILED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.pages == 1
    assert result.usage.retries == 1
    assert result.usage.redirects == 0
    assert result.usage.possible_duplicate_work is True
    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("retry-grounded",)


@pytest.mark.asyncio
@pytest.mark.parametrize("timed_out", (False, True))
async def test_typed_nonretryable_gateway_failure_never_retries(timed_out: bool) -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        raise DiscoveryGatewayError("hop_failed", retryable=False, timed_out=timed_out)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-terminal",
    )

    assert calls == ["id", "gateway"]
    assert result.usage.physical_records[0].state is (
        PhysicalDispatchState.TIMED_OUT if timed_out else PhysicalDispatchState.FAILED
    )
    assert result.logical_outcomes[0].code == ("gateway_timed_out" if timed_out else "gateway_hop_failed")
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", ("group", "global"))
async def test_retry_allowance_exhaustion_rejects_before_second_id(exhausted: str) -> None:
    if exhausted == "group":
        registry, plan = _retryable_foundation_plan(
            "semantic_scholar",
            route_retries=0,
            budget_retries=1,
        )
    else:
        registry, plan = _retryable_semantic_scholar_plan()
        object.__setattr__(plan.ceilings, "max_retries", 0)
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        raise DiscoveryGatewayError("hop_failed", retryable=True)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-only",
    )

    assert calls == ["id", "gateway"]
    assert len(result.usage.physical_records) == 1
    assert result.logical_outcomes[0].code == "gateway_retry_exhausted"
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False


@pytest.mark.asyncio
async def test_retryable_post_gateway_failure_never_retries() -> None:
    registry, plan = _retryable_foundation_plan("figshare")
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        raise DiscoveryGatewayError("hop_failed", retryable=True)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-post",
    )

    assert calls == ["id", "gateway"]
    assert len(result.usage.physical_records) == 1
    assert result.logical_outcomes[0].code == "gateway_retry_not_allowed"
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gateway_error", "expected_state", "expected_code"),
    (
        (RuntimeError("secret-runtime"), PhysicalDispatchState.FAILED, "gateway_failed"),
        (TimeoutError(), PhysicalDispatchState.TIMED_OUT, "gateway_timed_out"),
    ),
)
async def test_generic_gateway_errors_are_never_inferred_retryable(
    gateway_error: Exception,
    expected_state: PhysicalDispatchState,
    expected_code: str,
) -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        raise gateway_error

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-generic",
    )

    assert calls == ["id", "gateway"]
    assert result.usage.physical_records[0].state is expected_state
    assert result.logical_outcomes[0].code == expected_code
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False


@pytest.mark.asyncio
@pytest.mark.parametrize("hostile_field", ("code_subclass", "unknown_code", "retryable", "timed_out"))
async def test_hostile_gateway_error_metadata_fails_closed(hostile_field: str) -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    calls = []
    error = DiscoveryGatewayError("hop_failed", retryable=True)

    class StringSubclass(str):
        pass

    if hostile_field == "code_subclass":
        error.code = StringSubclass("hop_failed")
    elif hostile_field == "unknown_code":
        error.code = "secret_provider_code"
    elif hostile_field == "retryable":
        error.retryable = 1
    else:
        error.timed_out = 1

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        raise error

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-hostile",
    )

    assert calls == ["id", "gateway"]
    assert result.usage.physical_records[0].state is PhysicalDispatchState.FAILED
    assert result.logical_outcomes[0].code == "gateway_error_invalid"
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False
    assert "secret_provider" not in repr(result)


@pytest.mark.asyncio
async def test_semantically_identical_relative_redirect_reaches_final_2xx() -> None:
    registry, plan = _redirectable_foundation_plan()
    group = plan.dispatch_groups[0]
    gateway_intents = []
    adapter_statuses = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_intents.append(intent)
        response = _gateway_response(route, intent)
        if len(gateway_intents) == 1:
            location = f"{intent.path}?limit=3&offset=0&query=accounted+execution"
            return replace(
                response,
                status_code=302,
                trace=replace(response.trace, status_code=302),
                redirect_location=location,
            )
        return response

    async def adapter(bound_group, dispatch):
        response = await dispatch(bound_group.intents[0])
        adapter_statuses.append(response.status_code)
        return DiscoveryAdapterResult(
            candidates=(DiscoveryCandidate("redirect-grounded", {"title": "redirect grounded"}),)
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-redirect", "dispatch-final")).__next__,
    )

    assert len(gateway_intents) == 2
    assert gateway_intents[1] is not gateway_intents[0]
    assert gateway_intents[1].path == gateway_intents[0].path
    assert frozenset(gateway_intents[1].query_pairs) == frozenset(gateway_intents[0].query_pairs)
    assert adapter_statuses == [200]
    assert tuple(record.dispatch_id for record in result.usage.physical_records) == (
        "dispatch-redirect",
        "dispatch-final",
    )
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.pages == 1
    assert result.usage.redirects == 1
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False
    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("redirect-grounded",)


def test_executor_delegates_redirect_parsing_without_urllib() -> None:
    source = inspect.getsource(executor_module)

    assert "urllib" not in source
    assert "urlsplit(" not in source
    assert "parse_qsl(" not in source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ("missing", "changed_query", "changed_limit", "changed_path", "cross_origin", "malformed", "hostile"),
)
async def test_invalid_redirect_location_rejects_before_second_id(case: str) -> None:
    registry, plan = _redirectable_foundation_plan()
    group = plan.dispatch_groups[0]
    calls = []

    class StringSubclass(str):
        pass

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        valid = f"{intent.path}?query=accounted+execution&offset=0&limit=3"
        locations = {
            "missing": None,
            "changed_query": f"{intent.path}?query=changed&offset=0&limit=3",
            "changed_limit": f"{intent.path}?query=accounted+execution&offset=0&limit=99",
            "changed_path": "/undeclared/path?query=accounted+execution&offset=0&limit=3",
            "cross_origin": f"https://attacker.example{valid}",
            "malformed": 7,
            "hostile": "https://secret-location-token@attacker.example/hidden",
        }
        location = locations[case]
        if case == "malformed":
            location = StringSubclass(valid)
        response = _gateway_response(route, intent)
        return replace(
            response,
            status_code=302,
            trace=replace(response.trace, status_code=302),
            redirect_location=location,
        )

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("must-not-ground", {"title": "must not ground"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-invalid-redirect",
    )

    assert calls == ["id", "gateway"]
    assert len(result.usage.physical_records) == 1
    assert result.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.logical_outcomes[0].code == "gateway_redirect_invalid"
    assert result.candidates == ()
    assert result.usage.redirects == 0
    assert "secret-location-token" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", ("group", "global"))
async def test_redirect_allowance_exhaustion_rejects_before_second_id(exhausted: str) -> None:
    if exhausted == "group":
        registry, plan = _redirectable_foundation_plan(route_redirects=0, budget_redirects=1)
    else:
        registry, plan = _redirectable_foundation_plan()
        object.__setattr__(plan.ceilings, "max_redirects", 0)
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        response = _gateway_response(route, intent)
        return replace(
            response,
            status_code=302,
            trace=replace(response.trace, status_code=302),
            redirect_location=f"{intent.path}?query=accounted+execution&offset=0&limit=3",
        )

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-exhausted",
    )

    assert calls == ["id", "gateway"]
    assert result.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.logical_outcomes[0].code == "gateway_redirect_exhausted"
    assert result.usage.redirects == 0


@pytest.mark.asyncio
async def test_second_redirect_after_allowance_rejects_before_third_id() -> None:
    registry, plan = _redirectable_foundation_plan()
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        response = _gateway_response(route, intent)
        return replace(
            response,
            status_code=302,
            trace=replace(response.trace, status_code=302),
            redirect_location=f"{intent.path}?query=accounted+execution&offset=0&limit=3",
        )

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or f"dispatch-{calls.count('id')}",
    )

    assert calls == ["id", "gateway", "id", "gateway"]
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.redirects == 1
    assert result.logical_outcomes[0].code == "gateway_redirect_exhausted"


@pytest.mark.asyncio
async def test_post_redirect_is_never_followed() -> None:
    registry, plan = _redirectable_foundation_plan("figshare")
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        response = _gateway_response(route, intent)
        return replace(
            response,
            status_code=302,
            trace=replace(response.trace, status_code=302),
            redirect_location=f"{intent.path}?page=1&page_size=3",
        )

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-post-redirect",
    )

    assert calls == ["id", "gateway"]
    assert result.usage.physical_records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.logical_outcomes[0].code == "gateway_redirect_invalid"
    assert result.usage.redirects == 0


@pytest.mark.asyncio
async def test_2xx_redirect_metadata_is_ignored() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]

    async def gateway(route, intent, *, is_policy_active):
        return replace(
            _gateway_response(route, intent),
            redirect_location="https://secret-ignored.example/",
        )

    async def adapter(bound_group, dispatch):
        response = await dispatch(bound_group.intents[0])
        assert response.status_code == 200
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("grounded", {"title": "grounded"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-2xx",
    )

    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("grounded",)
    assert result.usage.redirects == 0
    assert "secret-ignored" not in repr(result)


@pytest.mark.asyncio
async def test_redirect_followed_by_4xx_does_not_ground_candidates() -> None:
    registry, plan = _redirectable_foundation_plan()
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        response = _gateway_response(route, intent)
        if len(calls) == 1:
            return replace(
                response,
                status_code=302,
                trace=replace(response.trace, status_code=302),
                redirect_location=f"{intent.path}?query=accounted+execution&offset=0&limit=3",
            )
        return replace(response, status_code=400, trace=replace(response.trace, status_code=400))

    async def adapter(bound_group, dispatch):
        response = await dispatch(bound_group.intents[0])
        assert response.status_code == 400
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("not-grounded", {"title": "not grounded"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-redirect", "dispatch-4xx")).__next__,
    )

    assert calls == ["gateway", "gateway"]
    assert result.candidates == ()
    assert result.logical_outcomes[0].code == "missing_search_dispatch"
    assert result.usage.redirects == 1


@pytest.mark.asyncio
async def test_adapter_execution_error_without_controller_failure_is_sanitized() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]

    async def adapter(bound_group, dispatch):
        raise executor_module.DiscoveryExecutionError("secret_adapter_selected_code")

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=lambda *args, **kwargs: pytest.fail("gateway must not run"),
        policy_is_active=lambda _route_id, _digest: True,
    )

    assert result.logical_outcomes[0].code == "adapter_failed"
    assert "secret_adapter_selected_code" not in repr(result)


@pytest.mark.asyncio
async def test_dispatch_rejects_adapter_spawned_task_before_reservation() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    gateway_calls = []
    detached_tasks = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(intent)
        return _gateway_response(route, intent)

    async def detached_dispatch(dispatch, intent):
        try:
            await dispatch(intent)
        except executor_module.DiscoveryExecutionError as error:
            return error.code
        return "unexpected_success"

    async def adapter(bound_group, dispatch):
        detached_tasks.append(asyncio.create_task(detached_dispatch(dispatch, bound_group.intents[0])))
        await asyncio.sleep(0)
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "must-not-reserve",
        journal=journal,
    )
    task_results = await asyncio.gather(*detached_tasks)
    after_result = (journal.records, tuple(gateway_calls))
    await asyncio.sleep(0)

    assert task_results == ["dispatch_task_mismatch"]
    assert result.logical_outcomes[0].code == "dispatch_task_mismatch"
    assert (journal.records, tuple(gateway_calls)) == after_result == ((), ())


def test_candidate_records_are_deeply_copied_and_frozen() -> None:
    retained = {
        "source": {"collection": "shared"},
        "items": [{"rank": 1}, True, None, 1.5],
    }
    candidate = DiscoveryCandidate("candidate", retained)

    retained["source"]["collection"] = "mutated"
    retained["items"][0]["rank"] = 99
    retained["items"].append("late")

    assert candidate.record["source"]["collection"] == "shared"
    assert candidate.record["items"][0]["rank"] == 1
    assert len(candidate.record["items"]) == 4
    with pytest.raises(TypeError):
        candidate.record["source"]["collection"] = "blocked"  # type: ignore[index]


def test_candidate_records_reject_unsupported_and_hostile_nested_values_cleanly() -> None:
    class StringSubclass(str):
        pass

    class HostileMapping(dict):
        def items(self):
            raise RuntimeError("secret_hostile_mapping")

    for record in (
        {"unsupported": object()},
        {"scalar_subclass": StringSubclass("value")},
        {"nested": HostileMapping(value=1)},
    ):
        with pytest.raises(ValueError, match="candidate_record_invalid") as caught:
            DiscoveryCandidate("candidate", record)
        assert "secret_hostile_mapping" not in repr(caught.value)


@pytest.mark.asyncio
async def test_policy_revoked_at_final_commit_suppresses_candidates_but_keeps_debit() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    journal = AttemptJournal(physical_ceiling=plan.ceilings.max_physical_dispatches)
    adapter_done = False
    post_adapter_checks = 0

    def policy_is_active(_route_id, _digest):
        nonlocal post_adapter_checks
        if not adapter_done:
            return True
        post_adapter_checks += 1
        return post_adapter_checks == 1

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        nonlocal adapter_done
        await dispatch(bound_group.intents[0])
        adapter_done = True
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("must-not-commit", {"title": "revoked"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=policy_is_active,
        dispatch_id_factory=lambda: "dispatch-before-final-revocation",
        journal=journal,
    )

    assert journal.records[0].state is PhysicalDispatchState.SUCCEEDED
    assert result.candidates == ()
    assert result.logical_outcomes[0].code == "dispatch_policy_inactive"
    assert post_adapter_checks == 2


@pytest.mark.asyncio
async def test_retry_selected_but_duplicate_id_does_not_count_as_continuation() -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        raise DiscoveryGatewayError("hop_failed", retryable=True)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or "dispatch-collision",
    )

    assert calls == ["id", "gateway", "id"]
    assert result.logical_outcomes[0].code == "duplicate_dispatch_id"
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_stop", "expected_state", "expected_code"),
    (
        ("cancellation", LogicalOutcomeState.CANCELLED, "execution_cancelled"),
        ("invalid_clock", LogicalOutcomeState.FAILED, "execution_clock_invalid"),
        ("backward_clock", LogicalOutcomeState.FAILED, "execution_clock_invalid"),
    ),
)
async def test_retry_duplicate_id_factory_cannot_hide_global_stop(
    runtime_stop: str,
    expected_state: LogicalOutcomeState,
    expected_code: str,
) -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    factory_calls = 0
    stopped = False
    gateway_calls = 0

    def dispatch_id_factory():
        nonlocal factory_calls, stopped
        factory_calls += 1
        if factory_calls == 2:
            stopped = True
        return "dispatch-collision"

    def clock():
        if runtime_stop == "invalid_clock" and stopped:
            return float("nan")
        if runtime_stop == "backward_clock" and stopped:
            return 9.0
        return 10.0

    async def gateway(route, intent, *, is_policy_active):
        nonlocal gateway_calls
        gateway_calls += 1
        raise DiscoveryGatewayError("hop_failed", retryable=True)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
        cancellation_check=lambda: runtime_stop == "cancellation" and stopped,
        monotonic_clock=clock,
    )

    assert factory_calls == 2
    assert gateway_calls == 1
    assert len(result.usage.physical_records) == 1
    assert result.usage.physical_records[0].state is PhysicalDispatchState.FAILED
    assert result.logical_outcomes[0].state is expected_state
    assert result.logical_outcomes[0].code == expected_code
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False


@pytest.mark.asyncio
async def test_redirect_selected_but_id_factory_failure_does_not_count_as_continuation() -> None:
    registry, plan = _redirectable_foundation_plan()
    group = plan.dispatch_groups[0]
    gateway_calls = 0
    dispatch_ids = iter(("dispatch-first",))

    async def gateway(route, intent, *, is_policy_active):
        nonlocal gateway_calls
        gateway_calls += 1
        response = _gateway_response(route, intent)
        return replace(
            response,
            status_code=302,
            trace=replace(response.trace, status_code=302),
            redirect_location=f"{intent.path}?query=accounted+execution&offset=0&limit=3",
        )

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_ids.__next__,
    )

    assert gateway_calls == 1
    assert result.logical_outcomes[0].code == "dispatch_id_factory_failed"
    assert result.usage.redirects == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("revocation_point", ("before_reservation", "after_reservation"))
async def test_retry_rechecks_policy_around_continuation_reservation(revocation_point: str) -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    active = True
    id_calls = 0
    gateway_calls = 0

    def policy_is_active(_route_id, _digest):
        return active

    def dispatch_id_factory():
        nonlocal active, id_calls
        id_calls += 1
        if id_calls == 2 and revocation_point == "after_reservation":
            active = False
        return f"dispatch-{id_calls}"

    async def gateway(route, intent, *, is_policy_active):
        nonlocal active, gateway_calls
        gateway_calls += 1
        if revocation_point == "before_reservation":
            active = False
        raise DiscoveryGatewayError("hop_failed", retryable=True)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=policy_is_active,
        dispatch_id_factory=dispatch_id_factory,
    )

    expected_states = (
        (PhysicalDispatchState.FAILED,)
        if revocation_point == "before_reservation"
        else (PhysicalDispatchState.FAILED, PhysicalDispatchState.SKIPPED)
    )
    assert tuple(record.state for record in result.usage.physical_records) == expected_states
    assert id_calls == (1 if revocation_point == "before_reservation" else 2)
    assert gateway_calls == 1
    assert result.logical_outcomes[0].code == "dispatch_policy_inactive"
    assert result.usage.retries == 0
    assert result.usage.possible_duplicate_work is False


@pytest.mark.asyncio
async def test_retry_rechecks_exposed_group_before_continuation_id() -> None:
    registry, plan = _retryable_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    exposed_group = None
    calls = []

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        object.__setattr__(exposed_group, "backend_id", "mutated_backend")
        raise DiscoveryGatewayError("hop_failed", retryable=True)

    async def adapter(bound_group, dispatch):
        nonlocal exposed_group
        exposed_group = bound_group
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: calls.append("id") or f"dispatch-{calls.count('id')}",
    )

    assert calls == ["id", "gateway"]
    assert result.logical_outcomes[0].code == "bound_plan_mutated"
    assert result.usage.retries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_point", ("before_reservation", "after_reservation"))
async def test_redirect_rechecks_registry_around_continuation_reservation(replacement_point: str) -> None:
    registry, plan = _redirectable_foundation_plan()
    group = plan.dispatch_groups[0]
    original_route = registry.get_route(group.route_id)
    replacement_route = replace(original_route, adapter_version="replacement_version")
    id_calls = 0
    gateway_calls = 0

    def replace_registry_route():
        object.__setattr__(
            registry,
            "routes",
            tuple(replacement_route if route.route_id == group.route_id else route for route in registry.routes),
        )

    def dispatch_id_factory():
        nonlocal id_calls
        id_calls += 1
        if id_calls == 2 and replacement_point == "after_reservation":
            replace_registry_route()
        return f"dispatch-{id_calls}"

    async def gateway(route, intent, *, is_policy_active):
        nonlocal gateway_calls
        gateway_calls += 1
        response = _gateway_response(route, intent)
        if replacement_point == "before_reservation":
            replace_registry_route()
        return replace(
            response,
            status_code=302,
            trace=replace(response.trace, status_code=302),
            redirect_location=f"{intent.path}?query=accounted+execution&offset=0&limit=3",
        )

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=dispatch_id_factory,
    )

    expected_states = (
        (PhysicalDispatchState.SUCCEEDED,)
        if replacement_point == "before_reservation"
        else (PhysicalDispatchState.SUCCEEDED, PhysicalDispatchState.SKIPPED)
    )
    assert tuple(record.state for record in result.usage.physical_records) == expected_states
    assert id_calls == (1 if replacement_point == "before_reservation" else 2)
    assert gateway_calls == 1
    assert result.logical_outcomes[0].code == "registry_mismatch"
    assert result.usage.redirects == 0


@pytest.mark.asyncio
async def test_executor_uses_pre_adapter_plan_snapshot_for_caps_counters_and_result_metadata() -> None:
    registry, plan = _foundation_plan(("semantic_scholar", "crossref"), result_limit=2)
    groups = {group.logical_attempts[0].catalog_source_id: group for group in plan.dispatch_groups}
    original_skipped = plan.skipped
    original_physical_ceiling = plan.ceilings.max_physical_dispatches
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append(route.route_id)
        return _gateway_response(route, intent)

    async def first_adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        object.__setattr__(plan.allowance, "returned_results", 0)
        object.__setattr__(plan.ceilings, "max_route_attempts", 0)
        object.__setattr__(plan.ceilings, "max_physical_dispatches", 0)
        object.__setattr__(plan.ceilings, "max_pages_per_route", 0)
        object.__setattr__(plan, "catalog_version", "secret_mutated_catalog_version")
        object.__setattr__(plan, "registry_version", "secret_mutated_registry_version")
        object.__setattr__(plan, "skipped", ("secret_mutated_skipped",))
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("first", {"title": "first"}),))

    async def second_adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=(DiscoveryCandidate("second", {"title": "second"}),))

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={
            groups["semantic_scholar"].adapter_id: first_adapter,
            groups["crossref"].adapter_id: second_adapter,
        },
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=iter(("dispatch-first", "dispatch-second")).__next__,
    )

    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("first", "second")
    assert result.usage.route_attempts == 2
    assert result.usage.accounting.physical_ceiling == original_physical_ceiling
    assert len(gateway_calls) == 2
    assert result.skipped == original_skipped
    assert "secret_mutated" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("derived_allowance", "top_level_invariant"))
async def test_pre_entry_plan_corruption_is_rejected_before_adapter_or_id(corruption: str) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    calls = []

    if corruption == "derived_allowance":
        object.__setattr__(plan.allowance, "returned_results", 999_999)
    else:
        object.__setattr__(plan, "result_limit", 0)

    async def adapter(bound_group, dispatch):
        calls.append("adapter")
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(candidates=())

    async def gateway(route, intent, *, is_policy_active):
        calls.append("gateway")
        return _gateway_response(route, intent)

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: adapter},
            gateway=gateway,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
        )

    assert caught.value.code == "plan_validation_failed"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_mutation", ("semantic_query", "figshare_body"))
async def test_pre_entry_valid_payload_mutation_is_rejected_before_any_effect(payload_mutation: str) -> None:
    if payload_mutation == "semantic_query":
        registry, plan = _semantic_scholar_plan()
        pair = plan.dispatch_groups[0].intents[0].query_pairs[0]
    else:
        registry, plan = _foundation_plan(("figshare",))
        pair = plan.dispatch_groups[0].intents[0].json_body_pairs[0]
    object.__setattr__(pair, "value", "valid but mutated request material")
    plan = _replace_plan_with_fresh_digest(plan)
    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_mutation",
    ("dispatch_group_id", "logical_attempt_id", "selection_reason"),
)
async def test_pre_entry_valid_identity_mutation_is_rejected_before_any_effect(identity_mutation: str) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    attempt = group.logical_attempts[0]
    if identity_mutation == "dispatch_group_id":
        object.__setattr__(group, "dispatch_group_id", "dispatch_group_v2_000000000000000000000000")
    elif identity_mutation == "logical_attempt_id":
        object.__setattr__(attempt, "logical_attempt_id", "logical_attempt_v2_000000000000000000000000")
    else:
        object.__setattr__(attempt, "selection_reason", "valid but mutated selection reason")
    plan = _replace_plan_with_fresh_digest(plan)
    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan_mutation",
    (
        "result_limit_cap_expansion",
        "result_limit_cap_expansion_cleared_digest",
        "planner_version",
        "readiness_version",
        "execution_mode",
        "skipped_status",
        "skipped_code",
        "skipped_reason",
    ),
)
async def test_pre_entry_compiler_owned_plan_mutation_is_rejected_before_any_effect(plan_mutation: str) -> None:
    if plan_mutation.startswith("skipped_"):
        registry, plan = _foundation_plan(("openalex", "semantic_scholar"), result_limit=1)
        skipped = plan.skipped[0]
        if plan_mutation == "skipped_status":
            object.__setattr__(skipped, "status", SkippedStatus.SKIPPED)
        elif plan_mutation == "skipped_code":
            object.__setattr__(skipped, "code", SkippedCode.ROUTE_NOT_READY)
        else:
            object.__setattr__(skipped, "reason", "forged but valid reason")
    else:
        registry, plan = _foundation_plan(("semantic_scholar",), result_limit=1)
        if plan_mutation.startswith("result_limit_cap_expansion"):
            object.__setattr__(plan, "result_limit", 3)
            object.__setattr__(plan.allowance, "returned_results", 3)
            object.__setattr__(plan.ceilings, "max_results", 3)
            if plan_mutation.endswith("cleared_digest"):
                object.__setattr__(plan, "plan_digest", "")
        elif plan_mutation == "planner_version":
            object.__setattr__(plan, "planner_version", "forged-planner-v2")
        elif plan_mutation == "readiness_version":
            object.__setattr__(plan, "readiness_version", "forged-readiness-v2")
        else:
            assert plan_mutation == "execution_mode"
            object.__setattr__(plan, "execution_mode", ExecutionMode.OFFLINE_FIXTURE)

    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "semantic_forgery",
    (
        "credentialed_status",
        "credentialed_code",
        "credentialed_reason",
        "credentialless_status",
        "credentialless_code",
    ),
)
async def test_pre_entry_skipped_route_semantics_are_rejected_before_any_effect(semantic_forgery: str) -> None:
    registry = foundation_registry()
    if semantic_forgery.startswith("credentialed_"):
        plan = compile_discovery_plan(
            PlanningRequest(("openalex",), "accounted execution", (), 1),
            registry=registry,
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
        )
    else:
        route_id = registry.get_source("semantic_scholar").route_references[0].route_id
        readiness = foundation_readiness(ExecutionMode.SYNTHETIC)
        readiness = replace(
            readiness,
            routes=tuple(
                (
                    replace(
                        entry,
                        state=ReadinessState.DISABLED,
                        credential_status=CredentialStatus.NOT_REQUIRED,
                        reason="disabled for skipped semantics test",
                    )
                    if entry.route_id == route_id
                    else entry
                )
                for entry in readiness.routes
            ),
        )
        plan = compile_discovery_plan(
            PlanningRequest(("semantic_scholar",), "accounted execution", (), 1),
            registry=registry,
            readiness=readiness,
            budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
        )
    skipped = plan.skipped[0]
    if semantic_forgery.endswith("status"):
        forged = replace(
            skipped,
            status=SkippedStatus.SKIPPED if semantic_forgery.startswith("credentialed_") else SkippedStatus.UNAVAILABLE,
        )
    elif semantic_forgery.endswith("code"):
        forged = replace(
            skipped,
            code=(
                SkippedCode.ROUTE_NOT_READY
                if semantic_forgery.startswith("credentialed_")
                else SkippedCode.CREDENTIALED_OUT_OF_SCOPE
            ),
        )
    else:
        forged = replace(skipped, reason="forged credentialed reason")
    plan = _replace_plan_with_fresh_digest(plan, skipped=(forged,))

    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering_mutation", ("dispatch_groups", "logical_attempts", "skipped"))
async def test_pre_entry_noncanonical_order_is_rejected_before_any_effect(ordering_mutation: str) -> None:
    if ordering_mutation == "dispatch_groups":
        registry, plan = _foundation_plan(("semantic_scholar", "crossref"))
        object.__setattr__(plan, "dispatch_groups", tuple(reversed(plan.dispatch_groups)))
    elif ordering_mutation == "logical_attempts":
        registry, plan = _coalesced_semantic_scholar_plan()
        group = plan.dispatch_groups[0]
        object.__setattr__(group, "logical_attempts", tuple(reversed(group.logical_attempts)))
    else:
        registry = foundation_registry()
        disabled_route_ids = {
            registry.get_source(source_id).route_references[0].route_id
            for source_id in ("semantic_scholar", "crossref")
        }
        readiness = foundation_readiness(ExecutionMode.SYNTHETIC)
        readiness = replace(
            readiness,
            routes=tuple(
                (
                    replace(
                        entry,
                        state=ReadinessState.DISABLED,
                        credential_status=CredentialStatus.NOT_REQUIRED,
                        reason="disabled for deterministic ordering test",
                    )
                    if entry.route_id in disabled_route_ids
                    else entry
                )
                for entry in readiness.routes
            ),
        )
        plan = compile_discovery_plan(
            PlanningRequest(("semantic_scholar", "crossref"), "accounted execution", (), 1),
            registry=registry,
            readiness=readiness,
            budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
        )
        assert len(plan.skipped) == 2
        object.__setattr__(plan, "skipped", tuple(reversed(plan.skipped)))
    plan = _replace_plan_with_fresh_digest(plan)
    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
async def test_pre_entry_duplicate_skipped_target_is_rejected_before_any_effect() -> None:
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(("openalex",), "accounted execution", (), 1),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
    )
    object.__setattr__(plan, "skipped", (plan.skipped[0], plan.skipped[0]))
    plan = _replace_plan_with_fresh_digest(plan)
    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repeated_logical_index", "selection_reason"),
    (
        (0, "repeated minimum coalesced target"),
        (1, "repeated nonminimum coalesced target"),
    ),
    ids=("minimum", "nonminimum"),
)
async def test_pre_entry_repeated_coalesced_target_is_rejected_before_any_effect(
    repeated_logical_index: int,
    selection_reason: str,
) -> None:
    registry, plan = _coalesced_semantic_scholar_plan()
    first_group = plan.dispatch_groups[0]
    repeated_attempt = first_group.logical_attempts[repeated_logical_index]
    first_intent = first_group.intents[0]
    second_intent = replace(
        first_intent,
        query_pairs=(
            replace(first_intent.query_pairs[0], value="alternate accounted execution"),
            *first_intent.query_pairs[1:],
        ),
    )
    second_attempt = replace(
        repeated_attempt,
        logical_attempt_id="logical_attempt_v2_000000000000000000000000",
        selection_reason=selection_reason,
    )
    second_group = replace(
        first_group,
        dispatch_group_id="dispatch_group_v2_000000000000000000000000",
        logical_attempts=(second_attempt,),
        intents=(second_intent,),
    )
    second_group_id = expected_dispatch_group_id(second_group)
    second_attempt = replace(
        second_attempt,
        logical_attempt_id=expected_logical_attempt_id(second_attempt, second_group_id),
    )
    second_group = replace(
        second_group,
        dispatch_group_id=second_group_id,
        logical_attempts=(second_attempt,),
    )
    plan = _replace_plan_with_fresh_digest(
        plan,
        dispatch_groups=(first_group, second_group),
        ceilings=replace(
            plan.ceilings,
            max_route_attempts=3,
            max_physical_dispatches=2,
            max_wall_time_ms=40_000,
        ),
    )
    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
async def test_pre_entry_executable_and_skipped_target_overlap_is_rejected_before_any_effect() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    attempt = group.logical_attempts[0]
    plan = _replace_plan_with_fresh_digest(
        plan,
        skipped=(
            SkippedTarget(
                requested_source_id=attempt.catalog_source_id,
                route_id=group.route_id,
                status=SkippedStatus.SKIPPED,
                code=SkippedCode.ROUTE_NOT_READY,
                reason="duplicate executable target",
            ),
        ),
    )
    await _assert_plan_validation_precedes_effects(registry, plan)


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("unknown_source", "predicate_mismatch"))
async def test_logical_attempt_must_match_canonical_registry_source_reference(corruption: str) -> None:
    registry, plan = _coalesced_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    first_attempt, second_attempt = group.logical_attempts
    calls = []

    if corruption == "unknown_source":
        object.__setattr__(first_attempt, "catalog_source_id", "nonexistent_source")
    else:
        object.__setattr__(first_attempt, "source_predicate", second_attempt.source_predicate)

    async def should_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("must not run")

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: should_not_run},
            gateway=should_not_run,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
        )

    assert caught.value.code == "plan_validation_failed"
    assert calls == []


@pytest.mark.asyncio
async def test_valid_plan_predicate_mutation_is_rejected_before_adapter_or_id() -> None:
    registry, plan = _coalesced_semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    attempt = group.logical_attempts[0]
    predicate = attempt.source_predicate
    registry_predicate = registry.get_source(attempt.catalog_source_id).route_references[0].source_predicate
    assert predicate is not None
    assert registry_predicate is not None
    original_registry_values = registry_predicate.values
    object.__setattr__(predicate, "values", ("attacker",))
    calls: list[str] = []

    async def should_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("must not run")

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={group.adapter_id: should_not_run},
            gateway=should_not_run,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
        )

    assert caught.value.code == "plan_validation_failed"
    assert registry_predicate.values == original_registry_values
    assert calls == []


@pytest.mark.asyncio
async def test_skipped_target_must_match_canonical_registry_source_route_reference() -> None:
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(("openalex",), "accounted execution", (), 1),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
    )
    object.__setattr__(plan.skipped[0], "requested_source_id", "semantic_scholar")
    calls = []

    async def should_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("must not run")

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters={},
            gateway=should_not_run,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
        )

    assert caught.value.code == "plan_validation_failed"
    assert calls == []


@pytest.mark.asyncio
async def test_adapter_result_boundary_refreezes_replaced_valid_candidate_record() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    candidate = DiscoveryCandidate("candidate", {"title": "constructed"})
    adapter_result = DiscoveryAdapterResult(candidates=(candidate,))
    replacement = {"title": "snapshot", "nested": [{"rank": 1}]}

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        object.__setattr__(candidate, "record", replacement)
        return adapter_result

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-result-snapshot",
    )
    replacement["title"] = "mutated-after-return"
    replacement["nested"][0]["rank"] = 99

    assert result.candidates[0].record["title"] == "snapshot"
    assert result.candidates[0].record["nested"][0]["rank"] == 1
    with pytest.raises(TypeError):
        result.candidates[0].record["title"] = "blocked"  # type: ignore[index]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    ("candidates_list", "candidate_instance", "candidate_id", "candidate_record"),
)
async def test_post_construction_adapter_result_corruption_is_sanitized(corruption: str) -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    candidate = DiscoveryCandidate("candidate", {"title": "constructed"})
    adapter_result = DiscoveryAdapterResult(candidates=(candidate,))

    class SecretValue:
        def __repr__(self) -> str:
            return "secret_hostile_adapter_value"

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        if corruption == "candidates_list":
            object.__setattr__(adapter_result, "candidates", [candidate])
        elif corruption == "candidate_instance":
            object.__setattr__(adapter_result, "candidates", (object(),))
        elif corruption == "candidate_id":
            object.__setattr__(candidate, "candidate_id", "")
        else:
            object.__setattr__(candidate, "record", {"value": SecretValue()})
        return adapter_result

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-malformed-adapter-result",
    )

    assert result.candidates == ()
    assert result.logical_outcomes[0].code == "malformed_adapter_result"
    assert "secret_hostile_adapter_value" not in repr(result)


@pytest.mark.asyncio
async def test_live_result_ceiling_truncates_candidates_without_erasing_logical_success() -> None:
    registry, plan = _semantic_scholar_plan()
    group = plan.dispatch_groups[0]
    object.__setattr__(plan.ceilings, "max_results", 1)

    async def gateway(route, intent, *, is_policy_active):
        return _gateway_response(route, intent)

    async def adapter(bound_group, dispatch):
        await dispatch(bound_group.intents[0])
        return DiscoveryAdapterResult(
            candidates=tuple(
                DiscoveryCandidate(f"candidate-{index}", {"title": f"result {index}"}) for index in range(3)
            )
        )

    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: "dispatch-live-result-ceiling",
    )

    assert tuple(candidate.candidate_id for candidate in result.candidates) == ("candidate-0",)
    assert result.truncated_candidates == 2
    assert result.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    (
        "empty_logical_attempt_id",
        "typed_logical_attempt_id",
        "skipped_status_string",
        "skipped_code_string",
        "query_pair",
        "json_body_pair",
        "query_binding",
        "route_limits_type",
        "dispatch_allowance_type",
        "source_predicate_type",
    ),
)
async def test_nested_plan_contract_corruption_is_rejected_before_adapter_or_id(corruption: str) -> None:
    if corruption in {"skipped_status_string", "skipped_code_string"}:
        registry = foundation_registry()
        plan = compile_discovery_plan(
            PlanningRequest(("openalex",), "accounted execution", (), 1),
            registry=registry,
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
        )
        skipped = plan.skipped[0]
        if corruption == "skipped_status_string":
            object.__setattr__(skipped, "status", skipped.status.value)
        else:
            object.__setattr__(skipped, "code", skipped.code.value)
    elif corruption == "json_body_pair":
        registry, plan = _foundation_plan(("figshare",))
        object.__setattr__(plan.dispatch_groups[0].intents[0].json_body_pairs[0], "name", "")
    elif corruption == "query_binding":
        registry, plan = _foundation_plan(("pubmed",))
        object.__setattr__(plan.dispatch_groups[0].intents[1].query_bindings[0], "max_items", True)
    elif corruption == "source_predicate_type":
        registry, plan = _coalesced_semantic_scholar_plan()
        predicate = plan.dispatch_groups[0].logical_attempts[0].source_predicate
        assert predicate is not None
        object.__setattr__(predicate, "operator", predicate.operator.value)
    else:
        registry, plan = _semantic_scholar_plan()
        group = plan.dispatch_groups[0]
        if corruption == "empty_logical_attempt_id":
            object.__setattr__(group.logical_attempts[0], "logical_attempt_id", "")
        elif corruption == "typed_logical_attempt_id":
            object.__setattr__(group.logical_attempts[0], "logical_attempt_id", 7)
        elif corruption == "query_pair":
            object.__setattr__(group.intents[0].query_pairs[0], "name", "")
        elif corruption == "route_limits_type":
            object.__setattr__(group.limits, "max_pages", True)
        else:
            assert corruption == "dispatch_allowance_type"
            object.__setattr__(group.allowance, "physical_dispatches", True)

    calls: list[str] = []

    async def should_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("must not run")

    adapters = {group.adapter_id: should_not_run for group in plan.dispatch_groups}
    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await execute_discovery_plan(
            plan,
            registry=registry,
            adapters=adapters,
            gateway=should_not_run,
            policy_is_active=lambda _route_id, _digest: True,
            dispatch_id_factory=lambda: calls.append("id") or "must-not-reserve",
        )

    assert caught.value.code == "plan_validation_failed"
    assert calls == []
