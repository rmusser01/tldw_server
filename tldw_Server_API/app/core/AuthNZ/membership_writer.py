"""Immutable membership-write contracts and deterministic lock planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

_INVALID_CONTRACT_MESSAGE = "Invalid membership writer contract."
_OFFLINE_MIGRATION_SERVING_MESSAGE = (
    "Offline migration membership context is unavailable while serving."
)


class MembershipWriterContractError(ValueError):
    """Raised when membership planning input violates the closed contract."""

    def __init__(self) -> None:
        super().__init__(_INVALID_CONTRACT_MESSAGE)


class OfflineMigrationContextRejected(MembershipWriterContractError):
    """Raised when an offline-only context reaches a serving boundary."""

    def __init__(self) -> None:
        ValueError.__init__(self, _OFFLINE_MIGRATION_SERVING_MESSAGE)


class _ClosedMembershipEnum(str, Enum):
    @classmethod
    def _missing_(cls, value: object) -> None:
        del value
        raise MembershipWriterContractError()


class AnchorOwnership(_ClosedMembershipEnum):
    """Selects which transaction owner advances affected profile anchors."""

    CALLER_OWNS_ANCHOR = "caller_owns_anchor"
    WRITER_OWNS_ANCHOR = "writer_owns_anchor"


class MembershipAuthority(_ClosedMembershipEnum):
    """Persisted authority that an actor must prove after locks are held."""

    SCOPED_MEMBERSHIP = "scoped_membership"
    PLATFORM_ADMIN = "platform_admin"


class TrustedMembershipReason(_ClosedMembershipEnum):
    """Audited non-actor reasons allowed to invoke membership writes."""

    REGISTRATION = "registration"
    BOOTSTRAP = "bootstrap"
    OFFLINE_MIGRATION = "offline_migration"


class MembershipScopeType(_ClosedMembershipEnum):
    """Membership table scope."""

    ORGANIZATION = "organization"
    TEAM = "team"


class MembershipMutationKind(_ClosedMembershipEnum):
    """Closed membership operations supported by the shared writer."""

    ADD = "add"
    REMOVE = "remove"
    UPDATE_ROLE = "update_role"


class MembershipLockBackend(_ClosedMembershipEnum):
    """Backends with distinct row-lock execution requirements."""

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class MembershipLockPhase(_ClosedMembershipEnum):
    """Total PostgreSQL membership lock order."""

    USER_ROWS = "user_rows"
    ORGANIZATION_ROWS = "organization_rows"
    TEAM_ROWS = "team_rows"
    MEMBERSHIP_ROWS = "membership_rows"
    OWNER_ROWS = "owner_rows"


def _require_positive_id(value: object) -> None:
    if type(value) is not int or value <= 0:
        raise MembershipWriterContractError()


def _require_exact_tuple(value: object) -> None:
    if type(value) is not tuple:
        raise MembershipWriterContractError()


def _validate_sorted_unique_ids(values: tuple[int, ...]) -> None:
    _require_exact_tuple(values)
    if any(type(value) is not int or value <= 0 for value in values):
        raise MembershipWriterContractError()
    if values != tuple(sorted(set(values))):
        raise MembershipWriterContractError()


def _row_sort_key(row: MembershipRowLock) -> tuple[str, int, int]:
    return row.scope_type.value, row.scope_id, row.user_id


def _owner_row_sort_key(row: MembershipRowLock) -> tuple[int, int]:
    return row.user_id, row.scope_id


@dataclass(frozen=True, slots=True)
class ActorMembershipWriteContext:
    """Actor-attributed membership authorization requirements."""

    actor_user_id: int
    required_authority: MembershipAuthority

    def __post_init__(self) -> None:
        _require_positive_id(self.actor_user_id)
        if type(self.required_authority) is not MembershipAuthority:
            raise MembershipWriterContractError()


@dataclass(frozen=True, slots=True)
class TrustedMembershipWriteContext:
    """Audited system-attributed membership authorization requirements."""

    trusted_reason: TrustedMembershipReason

    def __post_init__(self) -> None:
        if type(self.trusted_reason) is not TrustedMembershipReason:
            raise MembershipWriterContractError()


MembershipWriteContext = ActorMembershipWriteContext | TrustedMembershipWriteContext


@dataclass(frozen=True, slots=True)
class MembershipMutation:
    """One request-ordered membership mutation."""

    scope_type: MembershipScopeType
    scope_id: int
    user_id: int
    kind: MembershipMutationKind
    role: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.scope_type) is not MembershipScopeType:
            raise MembershipWriterContractError()
        _require_positive_id(self.scope_id)
        _require_positive_id(self.user_id)
        if type(self.kind) is not MembershipMutationKind:
            raise MembershipWriterContractError()
        role_required = self.kind in {
            MembershipMutationKind.ADD,
            MembershipMutationKind.UPDATE_ROLE,
        }
        if role_required:
            if type(self.role) is not str or not self.role:
                raise MembershipWriterContractError()
        elif self.role is not None:
            raise MembershipWriterContractError()


@dataclass(frozen=True, slots=True)
class MembershipRowLock:
    """Fully scoped identity for one organization or team membership row."""

    scope_type: MembershipScopeType
    scope_id: int
    user_id: int

    def __post_init__(self) -> None:
        if type(self.scope_type) is not MembershipScopeType:
            raise MembershipWriterContractError()
        _require_positive_id(self.scope_id)
        _require_positive_id(self.user_id)


@dataclass(frozen=True, slots=True)
class TeamParentOrganization:
    """Immutable parent-organization data captured before lock planning."""

    team_id: int
    organization_id: int

    def __post_init__(self) -> None:
        _require_positive_id(self.team_id)
        _require_positive_id(self.organization_id)


@dataclass(frozen=True, slots=True)
class OrganizationOwnerPreflight:
    """Complete owner identities captured for one organization."""

    organization_id: int
    owner_user_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_positive_id(self.organization_id)
        _validate_sorted_unique_ids(self.owner_user_ids)


@dataclass(frozen=True, slots=True)
class MembershipPlanningPreflight:
    """Immutable database-derived facts supplied to the pure planner."""

    team_parents: tuple[TeamParentOrganization, ...] = ()
    organization_owners: tuple[OrganizationOwnerPreflight, ...] = ()

    def __post_init__(self) -> None:
        _require_exact_tuple(self.team_parents)
        _require_exact_tuple(self.organization_owners)
        if any(type(item) is not TeamParentOrganization for item in self.team_parents):
            raise MembershipWriterContractError()
        if any(
            type(item) is not OrganizationOwnerPreflight
            for item in self.organization_owners
        ):
            raise MembershipWriterContractError()
        team_ids = tuple(item.team_id for item in self.team_parents)
        if team_ids != tuple(sorted(set(team_ids))):
            raise MembershipWriterContractError()
        owner_org_ids = tuple(
            item.organization_id for item in self.organization_owners
        )
        if owner_org_ids != tuple(sorted(set(owner_org_ids))):
            raise MembershipWriterContractError()


def _validate_sorted_unique_rows(values: tuple[MembershipRowLock, ...]) -> None:
    _require_exact_tuple(values)
    if any(type(row) is not MembershipRowLock for row in values):
        raise MembershipWriterContractError()
    if values != tuple(sorted(set(values), key=_row_sort_key)):
        raise MembershipWriterContractError()


def _validate_sorted_unique_owner_rows(
    values: tuple[MembershipRowLock, ...],
) -> None:
    _require_exact_tuple(values)
    if any(
        type(row) is not MembershipRowLock
        or row.scope_type is not MembershipScopeType.ORGANIZATION
        for row in values
    ):
        raise MembershipWriterContractError()
    if values != tuple(sorted(set(values), key=_owner_row_sort_key)):
        raise MembershipWriterContractError()


@dataclass(frozen=True, slots=True)
class MembershipLockSet:
    """Complete canonical lock identities for one membership write."""

    user_ids: tuple[int, ...]
    org_ids: tuple[int, ...]
    team_ids: tuple[int, ...]
    membership_rows: tuple[MembershipRowLock, ...]
    owner_rows: tuple[MembershipRowLock, ...]

    def __post_init__(self) -> None:
        _validate_sorted_unique_ids(self.user_ids)
        _validate_sorted_unique_ids(self.org_ids)
        _validate_sorted_unique_ids(self.team_ids)
        _validate_sorted_unique_rows(self.membership_rows)
        _validate_sorted_unique_owner_rows(self.owner_rows)
        if set(self.membership_rows) & set(self.owner_rows):
            raise MembershipWriterContractError()
        user_ids = set(self.user_ids)
        org_ids = set(self.org_ids)
        team_ids = set(self.team_ids)
        for row in self.membership_rows:
            if row.user_id not in user_ids:
                raise MembershipWriterContractError()
            if (
                row.scope_type is MembershipScopeType.ORGANIZATION
                and row.scope_id not in org_ids
            ):
                raise MembershipWriterContractError()
            if (
                row.scope_type is MembershipScopeType.TEAM
                and row.scope_id not in team_ids
            ):
                raise MembershipWriterContractError()
        if any(
            row.scope_type is not MembershipScopeType.ORGANIZATION
            or row.scope_id not in org_ids
            for row in self.owner_rows
        ):
            raise MembershipWriterContractError()


@dataclass(frozen=True, slots=True)
class MembershipLockPlan:
    """Canonical locks paired with untouched request-ordered mutations."""

    context: MembershipWriteContext
    mutations: tuple[MembershipMutation, ...]
    preflight: MembershipPlanningPreflight
    lock_set: MembershipLockSet

    def __post_init__(self) -> None:
        validate_membership_write_context(self.context, serving=False)
        _require_exact_tuple(self.mutations)
        if any(type(item) is not MembershipMutation for item in self.mutations):
            raise MembershipWriterContractError()
        if type(self.preflight) is not MembershipPlanningPreflight:
            raise MembershipWriterContractError()
        if type(self.lock_set) is not MembershipLockSet:
            raise MembershipWriterContractError()
        if self.lock_set != _build_membership_lock_set(
            context=self.context,
            mutations=self.mutations,
            preflight=self.preflight,
        ):
            raise MembershipWriterContractError()


@dataclass(frozen=True, slots=True)
class MembershipLockStatement:
    """Static parameterized PostgreSQL lock statement description."""

    phase: MembershipLockPhase
    sql: str
    parameters: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.phase) is not MembershipLockPhase:
            raise MembershipWriterContractError()
        if type(self.sql) is not str or not self.sql:
            raise MembershipWriterContractError()
        _require_exact_tuple(self.parameters)
        if any(type(value) is not int or value <= 0 for value in self.parameters):
            raise MembershipWriterContractError()


def validate_membership_write_context(
    context: MembershipWriteContext,
    *,
    serving: bool,
) -> MembershipWriteContext:
    """Validate a context for an explicitly supplied serving state."""

    if type(serving) is not bool:
        raise MembershipWriterContractError()
    if type(context) is ActorMembershipWriteContext:
        return context
    if type(context) is not TrustedMembershipWriteContext:
        raise MembershipWriterContractError()
    if (
        serving
        and context.trusted_reason is TrustedMembershipReason.OFFLINE_MIGRATION
    ):
        raise OfflineMigrationContextRejected()
    return context


def _build_membership_lock_set(
    *,
    context: MembershipWriteContext,
    mutations: tuple[MembershipMutation, ...],
    preflight: MembershipPlanningPreflight,
) -> MembershipLockSet:
    validate_membership_write_context(context, serving=False)
    _require_exact_tuple(mutations)
    if any(type(item) is not MembershipMutation for item in mutations):
        raise MembershipWriterContractError()
    if type(preflight) is not MembershipPlanningPreflight:
        raise MembershipWriterContractError()

    team_to_org = {
        parent.team_id: parent.organization_id for parent in preflight.team_parents
    }
    user_ids = {mutation.user_id for mutation in mutations}
    org_ids = {
        mutation.scope_id
        for mutation in mutations
        if mutation.scope_type is MembershipScopeType.ORGANIZATION
    }
    team_ids = {
        mutation.scope_id
        for mutation in mutations
        if mutation.scope_type is MembershipScopeType.TEAM
    }
    if tuple(team_to_org) != tuple(sorted(team_ids)):
        raise MembershipWriterContractError()
    org_ids.update(team_to_org[team_id] for team_id in team_ids)

    membership_rows = {
        MembershipRowLock(
            scope_type=mutation.scope_type,
            scope_id=mutation.scope_id,
            user_id=mutation.user_id,
        )
        for mutation in mutations
    }
    if type(context) is ActorMembershipWriteContext:
        user_ids.add(context.actor_user_id)
        if context.required_authority is MembershipAuthority.SCOPED_MEMBERSHIP:
            membership_rows.update(
                MembershipRowLock(
                    scope_type=MembershipScopeType.ORGANIZATION,
                    scope_id=org_id,
                    user_id=context.actor_user_id,
                )
                for org_id in org_ids
            )

    owner_sensitive_org_ids = {
        mutation.scope_id
        for mutation in mutations
        if mutation.scope_type is MembershipScopeType.ORGANIZATION
        and mutation.kind
        in {MembershipMutationKind.REMOVE, MembershipMutationKind.UPDATE_ROLE}
    }
    owner_preflight_org_ids = tuple(
        item.organization_id for item in preflight.organization_owners
    )
    if owner_preflight_org_ids != tuple(sorted(owner_sensitive_org_ids)):
        raise MembershipWriterContractError()
    owner_rows = {
        MembershipRowLock(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=item.organization_id,
            user_id=owner_user_id,
        )
        for item in preflight.organization_owners
        for owner_user_id in item.owner_user_ids
    }
    membership_rows.difference_update(owner_rows)

    return MembershipLockSet(
        user_ids=tuple(sorted(user_ids)),
        org_ids=tuple(sorted(org_ids)),
        team_ids=tuple(sorted(team_ids)),
        membership_rows=tuple(sorted(membership_rows, key=_row_sort_key)),
        owner_rows=tuple(sorted(owner_rows, key=_owner_row_sort_key)),
    )


def plan_membership_write(
    *,
    context: MembershipWriteContext,
    mutations: tuple[MembershipMutation, ...],
    preflight: MembershipPlanningPreflight,
) -> MembershipLockPlan:
    """Derive canonical locks without changing request mutation order."""

    lock_set = _build_membership_lock_set(
        context=context,
        mutations=mutations,
        preflight=preflight,
    )
    return MembershipLockPlan(
        context=context,
        mutations=mutations,
        preflight=preflight,
        lock_set=lock_set,
    )


_USER_LOCK_SQL = "SELECT id FROM public.users WHERE id = $1 FOR UPDATE"
_ORGANIZATION_LOCK_SQL = (
    "SELECT id FROM public.organizations WHERE id = $1 FOR UPDATE"
)
_TEAM_LOCK_SQL = "SELECT id FROM public.teams WHERE id = $1 FOR UPDATE"
_ORGANIZATION_MEMBERSHIP_LOCK_SQL = (
    "SELECT user_id FROM public.org_members "
    "WHERE org_id = $1 AND user_id = $2 FOR UPDATE"
)
_TEAM_MEMBERSHIP_LOCK_SQL = (
    "SELECT user_id FROM public.team_members "
    "WHERE team_id = $1 AND user_id = $2 FOR UPDATE"
)


def _membership_statement(
    row: MembershipRowLock,
    phase: MembershipLockPhase,
) -> MembershipLockStatement:
    sql = (
        _ORGANIZATION_MEMBERSHIP_LOCK_SQL
        if row.scope_type is MembershipScopeType.ORGANIZATION
        else _TEAM_MEMBERSHIP_LOCK_SQL
    )
    return MembershipLockStatement(
        phase=phase,
        sql=sql,
        parameters=(row.scope_id, row.user_id),
    )


def plan_membership_lock_statements(
    plan: MembershipLockPlan,
    *,
    backend: MembershipLockBackend,
) -> tuple[MembershipLockStatement, ...]:
    """Describe row locks in total phase order without performing I/O."""

    if type(plan) is not MembershipLockPlan:
        raise MembershipWriterContractError()
    if type(backend) is not MembershipLockBackend:
        raise MembershipWriterContractError()
    if backend is MembershipLockBackend.SQLITE:
        return ()

    lock_set = plan.lock_set
    statements: list[MembershipLockStatement] = []
    statements.extend(
        MembershipLockStatement(
            phase=MembershipLockPhase.USER_ROWS,
            sql=_USER_LOCK_SQL,
            parameters=(user_id,),
        )
        for user_id in lock_set.user_ids
    )
    statements.extend(
        MembershipLockStatement(
            phase=MembershipLockPhase.ORGANIZATION_ROWS,
            sql=_ORGANIZATION_LOCK_SQL,
            parameters=(org_id,),
        )
        for org_id in lock_set.org_ids
    )
    statements.extend(
        MembershipLockStatement(
            phase=MembershipLockPhase.TEAM_ROWS,
            sql=_TEAM_LOCK_SQL,
            parameters=(team_id,),
        )
        for team_id in lock_set.team_ids
    )
    statements.extend(
        _membership_statement(row, MembershipLockPhase.MEMBERSHIP_ROWS)
        for row in lock_set.membership_rows
    )
    statements.extend(
        _membership_statement(row, MembershipLockPhase.OWNER_ROWS)
        for row in lock_set.owner_rows
    )
    return tuple(statements)
