from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipLockBackend,
    MembershipLockPhase,
    MembershipLockSet,
    MembershipLockStatement,
    MembershipMutation,
    MembershipMutationKind,
    MembershipPlanningPreflight,
    MembershipRowLock,
    MembershipScopeType,
    MembershipWriteContext,
    MembershipWriterContractError,
    OfflineMigrationContextRejected,
    OrganizationOwnerPreflight,
    TeamParentOrganization,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
    plan_membership_write,
    validate_membership_write_context,
)


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


class _SecretValue:
    def __repr__(self) -> str:
        return "submitted-secret"


def _assert_sanitized_contract_error(callable_) -> None:
    with pytest.raises(MembershipWriterContractError) as exc_info:
        callable_()

    assert str(exc_info.value) == "Invalid membership writer contract."
    assert "submitted-secret" not in repr(exc_info.value)


def test_context_union_is_closed_and_discriminated_by_dataclass_type() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=11,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    trusted = TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.REGISTRATION,
    )

    assert get_args(MembershipWriteContext) == (
        ActorMembershipWriteContext,
        TrustedMembershipWriteContext,
    )
    assert type(actor) is ActorMembershipWriteContext
    assert type(trusted) is TrustedMembershipWriteContext


def test_closed_enums_expose_only_the_approved_values() -> None:
    assert {item.value for item in AnchorOwnership} == {
        "caller_owns_anchor",
        "writer_owns_anchor",
    }
    assert {item.value for item in MembershipAuthority} == {
        "scoped_membership",
        "platform_admin",
    }
    assert {item.value for item in TrustedMembershipReason} == {
        "registration",
        "bootstrap",
        "offline_migration",
    }


@pytest.mark.parametrize(
    "enum_type",
    [
        AnchorOwnership,
        MembershipAuthority,
        TrustedMembershipReason,
        MembershipScopeType,
        MembershipMutationKind,
        MembershipLockBackend,
        MembershipLockPhase,
    ],
)
def test_closed_enums_reject_unknown_values_without_echoing_them(enum_type) -> None:
    _assert_sanitized_contract_error(lambda: enum_type("submitted-secret"))


@pytest.mark.parametrize(
    "actor_user_id",
    [True, False, 0, -1, 1.0, "1", _IntSubclass(1)],
)
def test_actor_context_requires_an_exact_positive_integer(actor_user_id) -> None:
    _assert_sanitized_contract_error(
        lambda: ActorMembershipWriteContext(
            actor_user_id=actor_user_id,
            required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
        )
    )


def test_contexts_require_exact_enum_members() -> None:
    _assert_sanitized_contract_error(
        lambda: ActorMembershipWriteContext(
            actor_user_id=1,
            required_authority="submitted-secret",
        )
    )
    _assert_sanitized_contract_error(
        lambda: TrustedMembershipWriteContext(trusted_reason="submitted-secret")
    )


def test_serving_validation_is_explicit_and_rejects_offline_migration() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=7,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    registration = TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.REGISTRATION,
    )
    bootstrap = TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.BOOTSTRAP,
    )
    offline = TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.OFFLINE_MIGRATION,
    )

    assert validate_membership_write_context(actor, serving=True) is actor
    assert validate_membership_write_context(registration, serving=True) is registration
    assert validate_membership_write_context(bootstrap, serving=True) is bootstrap
    assert validate_membership_write_context(offline, serving=False) is offline
    with pytest.raises(
        OfflineMigrationContextRejected,
        match="Offline migration membership context is unavailable while serving\\.",
    ):
        validate_membership_write_context(offline, serving=True)


@pytest.mark.parametrize("serving", [0, 1, None, "true"])
def test_serving_validation_requires_an_exact_boolean(serving) -> None:
    context = TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.BOOTSTRAP,
    )

    _assert_sanitized_contract_error(
        lambda: validate_membership_write_context(context, serving=serving)
    )


def test_context_validation_rejects_objects_outside_the_union() -> None:
    _assert_sanitized_contract_error(
        lambda: validate_membership_write_context(_SecretValue(), serving=True)
    )


@pytest.mark.parametrize("invalid_id", [True, 0, -1, "1", _IntSubclass(1)])
def test_scope_team_organization_and_user_ids_are_exact_positive_ints(
    invalid_id,
) -> None:
    constructors = (
        lambda: MembershipMutation(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=invalid_id,
            user_id=1,
            kind=MembershipMutationKind.REMOVE,
        ),
        lambda: MembershipMutation(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=1,
            user_id=invalid_id,
            kind=MembershipMutationKind.REMOVE,
        ),
        lambda: MembershipRowLock(
            scope_type=MembershipScopeType.TEAM,
            scope_id=invalid_id,
            user_id=1,
        ),
        lambda: TeamParentOrganization(team_id=invalid_id, organization_id=1),
        lambda: TeamParentOrganization(team_id=1, organization_id=invalid_id),
    )

    for constructor in constructors:
        _assert_sanitized_contract_error(constructor)


def test_mutations_validate_closed_kind_and_role_shape() -> None:
    _assert_sanitized_contract_error(
        lambda: MembershipMutation(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=1,
            user_id=2,
            kind="submitted-secret",
        )
    )
    _assert_sanitized_contract_error(
        lambda: MembershipMutation(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=1,
            user_id=2,
            kind=MembershipMutationKind.ADD,
        )
    )
    _assert_sanitized_contract_error(
        lambda: MembershipMutation(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=1,
            user_id=2,
            kind=MembershipMutationKind.REMOVE,
            role="submitted-secret",
        )
    )


@pytest.mark.parametrize(
    "scope_type",
    [MembershipScopeType.ORGANIZATION, MembershipScopeType.TEAM],
)
@pytest.mark.parametrize(
    "kind",
    [MembershipMutationKind.ADD, MembershipMutationKind.UPDATE_ROLE],
)
@pytest.mark.parametrize(
    "role",
    [
        pytest.param("submitted-secret", id="unknown"),
        pytest.param(" \t\n", id="whitespace-only"),
        pytest.param("Owner", id="owner-case-variant"),
        pytest.param("ADMIN", id="admin-case-variant"),
        pytest.param("Lead", id="lead-case-variant"),
        pytest.param("MEMBER", id="member-case-variant"),
        pytest.param("member" * 100, id="overlength"),
        pytest.param(_StrSubclass("member"), id="exact-str-required"),
    ],
)
def test_role_bearing_mutations_reject_noncanonical_roles(
    scope_type,
    kind,
    role,
) -> None:
    _assert_sanitized_contract_error(
        lambda: MembershipMutation(
            scope_type=scope_type,
            scope_id=1,
            user_id=2,
            kind=kind,
            role=role,
        )
    )


@pytest.mark.parametrize(
    "scope_type",
    [MembershipScopeType.ORGANIZATION, MembershipScopeType.TEAM],
)
@pytest.mark.parametrize(
    "kind",
    [MembershipMutationKind.ADD, MembershipMutationKind.UPDATE_ROLE],
)
@pytest.mark.parametrize("role", ["owner", "admin", "lead", "member"])
def test_role_bearing_mutations_accept_each_canonical_role(
    scope_type,
    kind,
    role,
) -> None:
    mutation = MembershipMutation(
        scope_type=scope_type,
        scope_id=1,
        user_id=2,
        kind=kind,
        role=role,
    )

    assert mutation.role == role


def test_mutation_role_is_hidden_from_direct_and_plan_repr() -> None:
    role = "owner"
    mutation = MembershipMutation(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=23,
        user_id=47,
        kind=MembershipMutationKind.UPDATE_ROLE,
        role=role,
    )
    context = ActorMembershipWriteContext(
        actor_user_id=53,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    preflight = MembershipPlanningPreflight(
        organization_owners=(
            OrganizationOwnerPreflight(organization_id=23, owner_user_ids=()),
        ),
    )
    plan = plan_membership_write(
        context=context,
        mutations=(mutation,),
        preflight=preflight,
    )

    for rendered in (repr(mutation), repr(plan)):
        assert f"role={role!r}" not in rendered
        assert "scope_id=23" in rendered
        assert "user_id=47" in rendered
        assert "MembershipMutationKind.UPDATE_ROLE" in rendered
    assert "lock_set=MembershipLockSet" in repr(plan)


def test_contract_dataclasses_are_immutable_and_hashable() -> None:
    actor = ActorMembershipWriteContext(
        actor_user_id=1,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )
    trusted = TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.BOOTSTRAP,
    )
    mutation = MembershipMutation(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=2,
        user_id=3,
        kind=MembershipMutationKind.ADD,
        role="member",
    )
    row = MembershipRowLock(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=2,
        user_id=3,
    )
    parent = TeamParentOrganization(team_id=4, organization_id=2)
    owner_preflight = OrganizationOwnerPreflight(
        organization_id=2,
        owner_user_ids=(3,),
    )
    preflight = MembershipPlanningPreflight(
        team_parents=(parent,),
        organization_owners=(owner_preflight,),
    )
    plan_preflight = MembershipPlanningPreflight()
    plan = plan_membership_write(
        context=actor,
        mutations=(mutation,),
        preflight=plan_preflight,
    )
    lock_set = plan.lock_set
    statement = MembershipLockStatement(
        phase=MembershipLockPhase.USER_ROWS,
        sql="SELECT id FROM public.users WHERE id = $1 FOR UPDATE",
        parameters=(1,),
    )
    contracts = {
        actor,
        trusted,
        mutation,
        row,
        parent,
        owner_preflight,
        preflight,
        plan_preflight,
        lock_set,
        plan,
        statement,
    }

    assert len(contracts) == 11
    with pytest.raises(FrozenInstanceError):
        mutation.user_id = 99


def test_tuple_contracts_reject_mutable_or_malformed_collections() -> None:
    row = MembershipRowLock(
        scope_type=MembershipScopeType.ORGANIZATION,
        scope_id=1,
        user_id=2,
    )
    owner_preflight = OrganizationOwnerPreflight(
        organization_id=1,
        owner_user_ids=(2,),
    )

    _assert_sanitized_contract_error(
        lambda: MembershipPlanningPreflight(
            team_parents=[],
            organization_owners=(owner_preflight,),
        )
    )
    _assert_sanitized_contract_error(
        lambda: MembershipLockSet(
            user_ids=(2, 2),
            org_ids=(1,),
            team_ids=(),
            membership_rows=(),
            owner_rows=(row,),
        )
    )
