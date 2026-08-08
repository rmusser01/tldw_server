from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    MembershipAuthority,
    MembershipLockBackend,
    MembershipLockPhase,
    MembershipLockPlan,
    MembershipLockSet,
    MembershipMutation,
    MembershipMutationKind,
    MembershipPlanningPreflight,
    MembershipRowLock,
    MembershipScopeType,
    MembershipWriterContractError,
    OrganizationOwnerPreflight,
    TeamParentOrganization,
    plan_membership_lock_statements,
    plan_membership_write,
)


class _IntSubclass(int):
    pass


def _mutation(
    scope_type: MembershipScopeType,
    scope_id: int,
    user_id: int,
    kind: MembershipMutationKind,
    role: str | None = None,
) -> MembershipMutation:
    return MembershipMutation(
        scope_type=scope_type,
        scope_id=scope_id,
        user_id=user_id,
        kind=kind,
        role=role,
    )


def _row(
    scope_type: MembershipScopeType,
    scope_id: int,
    user_id: int,
) -> MembershipRowLock:
    return MembershipRowLock(
        scope_type=scope_type,
        scope_id=scope_id,
        user_id=user_id,
    )


def _owner_coverage(
    organization_id: int,
    owner_user_ids: tuple[int, ...],
) -> OrganizationOwnerPreflight:
    return OrganizationOwnerPreflight(
        organization_id=organization_id,
        owner_user_ids=owner_user_ids,
    )


def _complex_inputs():
    actor = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    team_add = _mutation(
        MembershipScopeType.TEAM,
        20,
        9,
        MembershipMutationKind.ADD,
        "member",
    )
    org_role = _mutation(
        MembershipScopeType.ORGANIZATION,
        5,
        8,
        MembershipMutationKind.UPDATE_ROLE,
        "member",
    )
    org_remove = _mutation(
        MembershipScopeType.ORGANIZATION,
        3,
        9,
        MembershipMutationKind.REMOVE,
    )
    team_remove = _mutation(
        MembershipScopeType.TEAM,
        10,
        8,
        MembershipMutationKind.REMOVE,
    )
    mutations = (
        team_add,
        org_role,
        org_remove,
        team_remove,
        team_add,
    )
    preflight = MembershipPlanningPreflight(
        team_parents=(
            TeamParentOrganization(team_id=10, organization_id=3),
            TeamParentOrganization(team_id=20, organization_id=5),
        ),
        organization_owners=(
            _owner_coverage(3, (6, 9)),
            _owner_coverage(5, (7, 8)),
        ),
    )
    return actor, mutations, preflight


def test_complete_lock_set_is_unique_sorted_and_fully_scoped() -> None:
    actor, mutations, preflight = _complex_inputs()

    plan = plan_membership_write(
        context=actor,
        mutations=mutations,
        preflight=preflight,
    )

    assert plan.lock_set.user_ids == (4, 8, 9)
    assert plan.lock_set.org_ids == (3, 5)
    assert plan.lock_set.team_ids == (10, 20)
    assert plan.lock_set.membership_rows == (
        _row(MembershipScopeType.ORGANIZATION, 3, 4),
        _row(MembershipScopeType.ORGANIZATION, 5, 4),
        _row(MembershipScopeType.TEAM, 10, 8),
        _row(MembershipScopeType.TEAM, 20, 9),
    )
    assert plan.lock_set.owner_rows == (
        _row(MembershipScopeType.ORGANIZATION, 3, 6),
        _row(MembershipScopeType.ORGANIZATION, 3, 9),
        _row(MembershipScopeType.ORGANIZATION, 5, 7),
        _row(MembershipScopeType.ORGANIZATION, 5, 8),
    )


def test_opposite_request_orders_have_identical_locks_and_preserve_mutations() -> None:
    actor, mutations, preflight = _complex_inputs()
    opposite = tuple(reversed(mutations))

    forward_plan = plan_membership_write(
        context=actor,
        mutations=mutations,
        preflight=preflight,
    )
    reverse_plan = plan_membership_write(
        context=actor,
        mutations=opposite,
        preflight=preflight,
    )

    assert forward_plan.lock_set == reverse_plan.lock_set
    assert forward_plan.mutations == mutations
    assert reverse_plan.mutations == opposite
    assert forward_plan.mutations.count(mutations[0]) == 2


def test_team_parent_organizations_drive_scoped_actor_authorization() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    mutations = (
        _mutation(
            MembershipScopeType.TEAM,
            20,
            9,
            MembershipMutationKind.ADD,
            "member",
        ),
    )

    plan = plan_membership_write(
        context=actor,
        mutations=mutations,
        preflight=MembershipPlanningPreflight(
            team_parents=(
                TeamParentOrganization(team_id=20, organization_id=5),
            ),
        ),
    )

    assert plan.lock_set.org_ids == (5,)
    assert plan.lock_set.team_ids == (20,)
    assert plan.lock_set.membership_rows == (
        _row(MembershipScopeType.ORGANIZATION, 5, 4),
        _row(MembershipScopeType.TEAM, 20, 9),
    )


def test_missing_or_conflicting_team_parent_preflight_fails_closed() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    mutations = (
        _mutation(
            MembershipScopeType.TEAM,
            20,
            9,
            MembershipMutationKind.REMOVE,
        ),
    )

    with pytest.raises(
        MembershipWriterContractError,
        match="Invalid membership writer contract\\.",
    ):
        plan_membership_write(
            context=actor,
            mutations=mutations,
            preflight=MembershipPlanningPreflight(),
        )
    with pytest.raises(MembershipWriterContractError):
        plan_membership_write(
            context=actor,
            mutations=mutations,
            preflight=MembershipPlanningPreflight(
                team_parents=(
                    TeamParentOrganization(team_id=20, organization_id=5),
                    TeamParentOrganization(team_id=20, organization_id=6),
                ),
            ),
        )


def test_platform_admin_is_actor_attributed_without_synthetic_membership() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=12,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutation = _mutation(
        MembershipScopeType.ORGANIZATION,
        3,
        9,
        MembershipMutationKind.ADD,
        "member",
    )

    plan = plan_membership_write(
        context=actor,
        mutations=(mutation,),
        preflight=MembershipPlanningPreflight(),
    )

    assert plan.lock_set.user_ids == (9, 12)
    assert plan.lock_set.membership_rows == (
        _row(MembershipScopeType.ORGANIZATION, 3, 9),
    )


def test_rows_shared_by_actor_target_and_owner_exist_only_in_owner_phase() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    shared = _row(MembershipScopeType.ORGANIZATION, 3, 4)
    mutation = _mutation(
        MembershipScopeType.ORGANIZATION,
        3,
        4,
        MembershipMutationKind.REMOVE,
    )

    plan = plan_membership_write(
        context=actor,
        mutations=(mutation,),
        preflight=MembershipPlanningPreflight(
            organization_owners=(_owner_coverage(3, (4,)),),
        ),
    )

    assert plan.lock_set.membership_rows == ()
    assert plan.lock_set.owner_rows == (shared,)


def test_owner_preflight_rejects_coverage_for_non_sensitive_organizations() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutation = _mutation(
        MembershipScopeType.ORGANIZATION,
        3,
        9,
        MembershipMutationKind.ADD,
        "member",
    )
    with pytest.raises(MembershipWriterContractError):
        plan_membership_write(
            context=actor,
            mutations=(mutation,),
            preflight=MembershipPlanningPreflight(
                organization_owners=(_owner_coverage(3, (7,)),),
            ),
        )


@pytest.mark.parametrize(
    ("user_ids", "org_ids", "team_ids", "membership_rows", "owner_rows"),
    [
        (
            (7,),
            (),
            (),
            (
                _row(MembershipScopeType.ORGANIZATION, 3, 7),
            ),
            (),
        ),
        (
            (7,),
            (),
            (),
            (_row(MembershipScopeType.TEAM, 5, 7),),
            (),
        ),
        (
            (),
            (3,),
            (),
            (
                _row(MembershipScopeType.ORGANIZATION, 3, 7),
            ),
            (),
        ),
        (
            (),
            (),
            (5,),
            (),
            (_row(MembershipScopeType.TEAM, 5, 7),),
        ),
        (
            (),
            (3,),
            (),
            (),
            (
                _row(MembershipScopeType.ORGANIZATION, 4, 7),
            ),
        ),
    ],
)
def test_lock_set_rejects_cross_field_scope_or_target_gaps(
    user_ids,
    org_ids,
    team_ids,
    membership_rows,
    owner_rows,
) -> None:
    with pytest.raises(MembershipWriterContractError):
        MembershipLockSet(
            user_ids=user_ids,
            org_ids=org_ids,
            team_ids=team_ids,
            membership_rows=membership_rows,
            owner_rows=owner_rows,
        )


def test_owner_rows_do_not_require_unaffected_owner_user_locks() -> None:
    owner = _row(MembershipScopeType.ORGANIZATION, 3, 99)

    lock_set = MembershipLockSet(
        user_ids=(7,),
        org_ids=(3,),
        team_ids=(),
        membership_rows=(
            _row(MembershipScopeType.ORGANIZATION, 3, 7),
        ),
        owner_rows=(owner,),
    )

    assert lock_set.owner_rows == (owner,)
    assert 99 not in lock_set.user_ids


def test_lock_plan_retains_exact_factory_inputs() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutations = (
        _mutation(
            MembershipScopeType.ORGANIZATION,
            3,
            9,
            MembershipMutationKind.ADD,
            "member",
        ),
    )
    preflight = MembershipPlanningPreflight()

    plan = plan_membership_write(
        context=context,
        mutations=mutations,
        preflight=preflight,
    )

    assert plan.context is context
    assert plan.mutations is mutations
    assert plan.preflight is preflight


def test_lock_plan_rejects_a_lock_set_not_derived_from_its_inputs() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutations = (
        _mutation(
            MembershipScopeType.ORGANIZATION,
            3,
            9,
            MembershipMutationKind.ADD,
            "member",
        ),
    )
    preflight = MembershipPlanningPreflight()
    incomplete_lock_set = MembershipLockSet(
        user_ids=(4, 9),
        org_ids=(3,),
        team_ids=(),
        membership_rows=(),
        owner_rows=(),
    )

    with pytest.raises(MembershipWriterContractError):
        MembershipLockPlan(
            context=context,
            mutations=mutations,
            preflight=preflight,
            lock_set=incomplete_lock_set,
        )


def test_owner_sensitive_mutation_requires_explicit_owner_preflight() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutation = _mutation(
        MembershipScopeType.ORGANIZATION,
        3,
        9,
        MembershipMutationKind.REMOVE,
    )

    with pytest.raises(MembershipWriterContractError):
        plan_membership_write(
            context=context,
            mutations=(mutation,),
            preflight=MembershipPlanningPreflight(),
        )


def test_owner_preflight_requires_complete_multi_org_coverage() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutations = (
        _mutation(
            MembershipScopeType.ORGANIZATION,
            3,
            9,
            MembershipMutationKind.REMOVE,
        ),
        _mutation(
            MembershipScopeType.ORGANIZATION,
            5,
            8,
            MembershipMutationKind.UPDATE_ROLE,
            "member",
        ),
    )
    preflight = MembershipPlanningPreflight(
        organization_owners=(_owner_coverage(3, (6, 9)),),
    )

    with pytest.raises(MembershipWriterContractError):
        plan_membership_write(
            context=context,
            mutations=mutations,
            preflight=preflight,
        )


def test_owner_preflight_rejects_duplicate_org_coverage() -> None:
    coverage = _owner_coverage(3, (6, 9))

    with pytest.raises(MembershipWriterContractError):
        MembershipPlanningPreflight(
            organization_owners=(coverage, coverage),
        )


def test_owner_preflight_rejects_wrong_org_coverage() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutation = _mutation(
        MembershipScopeType.ORGANIZATION,
        3,
        9,
        MembershipMutationKind.REMOVE,
    )
    preflight = MembershipPlanningPreflight(
        organization_owners=(_owner_coverage(4, (7,)),),
    )

    with pytest.raises(MembershipWriterContractError):
        plan_membership_write(
            context=context,
            mutations=(mutation,),
            preflight=preflight,
        )


def test_explicit_zero_owner_preflight_is_retained() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutation = _mutation(
        MembershipScopeType.ORGANIZATION,
        3,
        9,
        MembershipMutationKind.REMOVE,
    )
    coverage = _owner_coverage(3, ())
    preflight = MembershipPlanningPreflight(organization_owners=(coverage,))

    plan = plan_membership_write(
        context=context,
        mutations=(mutation,),
        preflight=preflight,
    )

    assert plan.preflight is preflight
    assert plan.preflight.organization_owners == (coverage,)
    assert plan.lock_set.owner_rows == ()


def test_owner_preflight_retains_deterministic_fully_scoped_identities() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    mutations = (
        _mutation(
            MembershipScopeType.ORGANIZATION,
            5,
            8,
            MembershipMutationKind.UPDATE_ROLE,
            "member",
        ),
        _mutation(
            MembershipScopeType.ORGANIZATION,
            3,
            9,
            MembershipMutationKind.REMOVE,
        ),
    )
    coverages = (
        _owner_coverage(3, (6, 9)),
        _owner_coverage(5, (7, 8)),
    )
    preflight = MembershipPlanningPreflight(organization_owners=coverages)

    plan = plan_membership_write(
        context=context,
        mutations=mutations,
        preflight=preflight,
    )

    assert plan.preflight is preflight
    assert plan.preflight.organization_owners == coverages
    assert plan.lock_set.owner_rows == (
        _row(MembershipScopeType.ORGANIZATION, 3, 6),
        _row(MembershipScopeType.ORGANIZATION, 3, 9),
        _row(MembershipScopeType.ORGANIZATION, 5, 7),
        _row(MembershipScopeType.ORGANIZATION, 5, 8),
    )
    assert not set(plan.lock_set.membership_rows) & set(plan.lock_set.owner_rows)


@pytest.mark.parametrize(
    ("organization_id", "owner_user_ids"),
    [
        (0, ()),
        (True, ()),
        (_IntSubclass(3), ()),
        (3, [7]),
        (3, (8, 7)),
        (3, (7, 7)),
        (3, (True,)),
        (3, (_IntSubclass(7),)),
    ],
)
def test_owner_preflight_requires_exact_canonical_ids(
    organization_id,
    owner_user_ids,
) -> None:
    with pytest.raises(MembershipWriterContractError):
        _owner_coverage(organization_id, owner_user_ids)


def test_team_parent_preflight_requires_exact_team_coverage() -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    mutation = _mutation(
        MembershipScopeType.TEAM,
        20,
        9,
        MembershipMutationKind.REMOVE,
    )
    parent = TeamParentOrganization(team_id=20, organization_id=5)

    with pytest.raises(MembershipWriterContractError):
        plan_membership_write(
            context=context,
            mutations=(mutation,),
            preflight=MembershipPlanningPreflight(
                team_parents=(
                    parent,
                    TeamParentOrganization(team_id=30, organization_id=5),
                ),
            ),
        )
    with pytest.raises(MembershipWriterContractError):
        MembershipPlanningPreflight(team_parents=(parent, parent))


def test_statement_planning_rejects_a_bare_incomplete_team_lock_set() -> None:
    incomplete_lock_set = MembershipLockSet(
        user_ids=(9,),
        org_ids=(),
        team_ids=(20,),
        membership_rows=(
            _row(MembershipScopeType.TEAM, 20, 9),
        ),
        owner_rows=(),
    )

    with pytest.raises(MembershipWriterContractError):
        plan_membership_lock_statements(
            incomplete_lock_set,
            backend=MembershipLockBackend.POSTGRESQL,
        )


@pytest.mark.parametrize(
    ("preflight", "org_ids"),
    [
        (MembershipPlanningPreflight(), ()),
        (
            MembershipPlanningPreflight(
                team_parents=(
                    TeamParentOrganization(team_id=20, organization_id=6),
                ),
            ),
            (5,),
        ),
    ],
)
def test_invalid_team_parent_plan_cannot_reach_statement_planning(
    preflight,
    org_ids,
) -> None:
    context = ActorMembershipWriteContext(
        actor_user_id=4,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    mutation = _mutation(
        MembershipScopeType.TEAM,
        20,
        9,
        MembershipMutationKind.ADD,
        "member",
    )
    incomplete_lock_set = MembershipLockSet(
        user_ids=(4, 9),
        org_ids=org_ids,
        team_ids=(20,),
        membership_rows=(
            _row(MembershipScopeType.TEAM, 20, 9),
        ),
        owner_rows=(),
    )

    with pytest.raises(MembershipWriterContractError):
        invalid_plan = MembershipLockPlan(
            context=context,
            mutations=(mutation,),
            preflight=preflight,
            lock_set=incomplete_lock_set,
        )
        plan_membership_lock_statements(
            invalid_plan,
            backend=MembershipLockBackend.POSTGRESQL,
        )


def test_postgresql_statements_follow_the_total_lock_phase_order() -> None:
    actor, mutations, preflight = _complex_inputs()
    plan = plan_membership_write(
        context=actor,
        mutations=mutations,
        preflight=preflight,
    )

    statements = plan_membership_lock_statements(
        plan,
        backend=MembershipLockBackend.POSTGRESQL,
    )

    assert tuple((item.phase, item.parameters) for item in statements) == (
        (MembershipLockPhase.USER_ROWS, (4,)),
        (MembershipLockPhase.USER_ROWS, (8,)),
        (MembershipLockPhase.USER_ROWS, (9,)),
        (MembershipLockPhase.ORGANIZATION_ROWS, (3,)),
        (MembershipLockPhase.ORGANIZATION_ROWS, (5,)),
        (MembershipLockPhase.TEAM_ROWS, (10,)),
        (MembershipLockPhase.TEAM_ROWS, (20,)),
        (MembershipLockPhase.MEMBERSHIP_ROWS, (3, 4)),
        (MembershipLockPhase.MEMBERSHIP_ROWS, (5, 4)),
        (MembershipLockPhase.MEMBERSHIP_ROWS, (10, 8)),
        (MembershipLockPhase.MEMBERSHIP_ROWS, (20, 9)),
        (MembershipLockPhase.OWNER_ROWS, (3, 6)),
        (MembershipLockPhase.OWNER_ROWS, (3, 9)),
        (MembershipLockPhase.OWNER_ROWS, (5, 7)),
        (MembershipLockPhase.OWNER_ROWS, (5, 8)),
    )
    assert all("FOR UPDATE" in item.sql for item in statements)
    assert all("$1" in item.sql for item in statements)
    assert all(
        "$2" in item.sql
        for item in statements
        if item.phase
        in {MembershipLockPhase.MEMBERSHIP_ROWS, MembershipLockPhase.OWNER_ROWS}
    )
    assert "org_members" in statements[7].sql
    assert "team_members" in statements[9].sql


def test_postgresql_statements_use_canonical_public_relations() -> None:
    actor, mutations, preflight = _complex_inputs()
    plan = plan_membership_write(
        context=actor,
        mutations=mutations,
        preflight=preflight,
    )

    statements = plan_membership_lock_statements(
        plan,
        backend=MembershipLockBackend.POSTGRESQL,
    )

    user_sql = "SELECT id FROM public.users WHERE id = $1 FOR UPDATE"
    org_sql = "SELECT id FROM public.organizations WHERE id = $1 FOR UPDATE"
    team_sql = "SELECT id FROM public.teams WHERE id = $1 FOR UPDATE"
    org_member_sql = (
        "SELECT user_id FROM public.org_members "
        "WHERE org_id = $1 AND user_id = $2 FOR UPDATE"
    )
    team_member_sql = (
        "SELECT user_id FROM public.team_members "
        "WHERE team_id = $1 AND user_id = $2 FOR UPDATE"
    )
    assert tuple(item.sql for item in statements) == (
        user_sql,
        user_sql,
        user_sql,
        org_sql,
        org_sql,
        team_sql,
        team_sql,
        org_member_sql,
        org_member_sql,
        team_member_sql,
        team_member_sql,
        org_member_sql,
        org_member_sql,
        org_member_sql,
        org_member_sql,
    )


def test_sqlite_computes_the_same_lock_set_without_row_lock_sql() -> None:
    actor, mutations, preflight = _complex_inputs()
    plan = plan_membership_write(
        context=actor,
        mutations=mutations,
        preflight=preflight,
    )

    assert (
        plan_membership_lock_statements(
            plan,
            backend=MembershipLockBackend.SQLITE,
        )
        == ()
    )


def test_statement_planning_rejects_unknown_backend_without_echoing_input() -> None:
    actor, mutations, preflight = _complex_inputs()
    plan = plan_membership_write(
        context=actor,
        mutations=mutations,
        preflight=preflight,
    )

    with pytest.raises(MembershipWriterContractError) as exc_info:
        plan_membership_lock_statements(plan, backend="submitted-secret")

    assert str(exc_info.value) == "Invalid membership writer contract."
    assert "submitted-secret" not in repr(exc_info.value)
