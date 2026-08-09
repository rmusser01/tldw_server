"""Immutable membership-write contracts and deterministic lock planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from tldw_Server_API.app.core.AuthNZ.exceptions import (
    RollbackSignal,
    UserRegistrationException,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _execute_membership_scope_sql,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    ProfileVersionError,
    VersionedUserWriteGateway,
    normalize_profile_version,
)

_INVALID_CONTRACT_MESSAGE = "Invalid membership writer contract."
_OFFLINE_MIGRATION_SERVING_MESSAGE = (
    "Offline migration membership context is unavailable while serving."
)
_MEMBERSHIP_ROLES = frozenset({"owner", "admin", "lead", "member"})


class MembershipWriterContractError(ValueError):
    """Raised when membership planning input violates the closed contract."""

    def __init__(self) -> None:
        super().__init__(_INVALID_CONTRACT_MESSAGE)


class OfflineMigrationContextRejected(MembershipWriterContractError):
    """Raised when an offline-only context reaches a serving boundary."""

    def __init__(self) -> None:
        ValueError.__init__(self, _OFFLINE_MIGRATION_SERVING_MESSAGE)


class MembershipWriteError(UserRegistrationException):
    """Base class for sanitized runtime membership-write failures."""


class MembershipReadError(UserRegistrationException):
    """A membership-state read failed without exposing backend details."""

    def __init__(self) -> None:
        super().__init__("Membership state could not be read.")


class MembershipAuthorizationError(MembershipWriteError):
    """The persisted actor authority is insufficient for the locked scopes."""

    def __init__(self) -> None:
        super().__init__("Membership write is not authorized.")


class MembershipScopeNotFound(MembershipWriteError):
    """A requested organization or team is absent or inactive."""

    def __init__(self) -> None:
        super().__init__("Membership scope was not found.")


class MembershipTargetNotFound(MembershipWriteError):
    """A requested target user is absent."""

    def __init__(self) -> None:
        super().__init__("Membership target was not found.")


class MembershipParentRequired(MembershipWriteError):
    """A team-add target lacks an active parent-organization membership."""

    def __init__(self) -> None:
        super().__init__("Active parent organization membership is required.")


class MembershipPreflightChanged(MembershipWriteError):
    """Database-derived lock inputs changed before all locks were held."""

    def __init__(self) -> None:
        super().__init__("Membership write preconditions changed.")


class _MembershipScopeDeletionRetry(RollbackSignal):
    """Private signal forcing scope-deletion retry after transaction rollback."""


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


class MembershipMutationRelationship(_ClosedMembershipEnum):
    """Closed relationship between request-ordered membership mutations."""

    DEFAULT_TEAM_COMPANION = "default_team_companion"
    ORGANIZATION_COMPANION = "organization_companion"


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
    AUTHORITY_ROWS = "authority_rows"


def _require_positive_id(value: object) -> None:
    if type(value) is not int or value <= 0:
        raise MembershipWriterContractError()


def _require_membership_role(value: object) -> None:
    if type(value) is not str or value not in _MEMBERSHIP_ROLES:
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
    relationship: MembershipMutationRelationship | None = None

    def __post_init__(self) -> None:
        if type(self.scope_type) is not MembershipScopeType:
            raise MembershipWriterContractError()
        _require_positive_id(self.scope_id)
        _require_positive_id(self.user_id)
        if type(self.kind) is not MembershipMutationKind:
            raise MembershipWriterContractError()
        if self.relationship is not None:
            if type(self.relationship) is not MembershipMutationRelationship:
                raise MembershipWriterContractError()
            if (
                self.scope_type is not MembershipScopeType.TEAM
                or self.kind
                not in {MembershipMutationKind.ADD, MembershipMutationKind.REMOVE}
                or (
                    self.relationship
                    is MembershipMutationRelationship.ORGANIZATION_COMPANION
                    and self.kind
                    not in {
                        MembershipMutationKind.ADD,
                        MembershipMutationKind.REMOVE,
                    }
                )
            ):
                raise MembershipWriterContractError()
        role_required = self.kind in {
            MembershipMutationKind.ADD,
            MembershipMutationKind.UPDATE_ROLE,
        }
        if role_required:
            _require_membership_role(self.role)
        elif self.role is not None:
            raise MembershipWriterContractError()


@dataclass(frozen=True, slots=True)
class MembershipMutationResult:
    """Immutable outcome for one request-ordered membership mutation."""

    mutation: MembershipMutation
    changed: bool
    found: bool
    role: str | None = field(default=None, repr=False)
    organization_id: int | None = None
    error: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.mutation) is not MembershipMutation:
            raise MembershipWriterContractError()
        if type(self.changed) is not bool or type(self.found) is not bool:
            raise MembershipWriterContractError()
        if self.role is not None and type(self.role) is not str:
            raise MembershipWriterContractError()
        if self.organization_id is not None:
            _require_positive_id(self.organization_id)
        is_team = self.mutation.scope_type is MembershipScopeType.TEAM
        if is_team != (self.organization_id is not None):
            raise MembershipWriterContractError()
        if self.error is not None and self.error not in {
            "owner_required",
            "org_membership_required",
        }:
            raise MembershipWriterContractError()
        if self.changed and not self.found:
            raise MembershipWriterContractError()
        if (
            self.mutation.kind is MembershipMutationKind.ADD
            and not self.found
            and self.error != "org_membership_required"
        ):
            raise MembershipWriterContractError()
        if self.error == "owner_required" and (self.changed or not self.found):
            raise MembershipWriterContractError()
        if self.error == "owner_required" and not (
            self.mutation.scope_type is MembershipScopeType.ORGANIZATION
            and self.mutation.kind
            in {MembershipMutationKind.REMOVE, MembershipMutationKind.UPDATE_ROLE}
            and self.found
            and not self.changed
        ):
            raise MembershipWriterContractError()
        if self.error == "org_membership_required" and not (
            self.mutation.scope_type is MembershipScopeType.TEAM
            and self.mutation.kind is MembershipMutationKind.ADD
            and not self.changed
            and not self.found
            and self.organization_id is not None
        ):
            raise MembershipWriterContractError()

    def to_legacy_result(self) -> dict[str, Any] | None:
        """Return the historical repository result for this mutation."""

        mutation = self.mutation
        scope_key = (
            "org_id"
            if mutation.scope_type is MembershipScopeType.ORGANIZATION
            else "team_id"
        )
        if mutation.kind is MembershipMutationKind.UPDATE_ROLE and not self.found:
            return None
        if mutation.kind is MembershipMutationKind.REMOVE:
            result: dict[str, Any] = {
                scope_key: mutation.scope_id,
                "user_id": mutation.user_id,
                "removed": self.changed,
            }
            if self.error is not None:
                result["error"] = self.error
            return result
        result = {
            scope_key: mutation.scope_id,
            "user_id": mutation.user_id,
            "role": self.role if self.role is not None else mutation.role,
        }
        if (
            mutation.kind is MembershipMutationKind.ADD
            and self.organization_id is not None
        ):
            result["org_id"] = self.organization_id
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass(frozen=True, slots=True)
class MembershipUserVersionFloor:
    """Complete pre/post composite-version inputs for one changed user."""

    user_id: int
    pre_mutation_floor: datetime
    post_mutation_floor: datetime

    def __post_init__(self) -> None:
        _require_positive_id(self.user_id)
        pre = _normalize_contract_time(self.pre_mutation_floor)
        post = _normalize_contract_time(self.post_mutation_floor)
        object.__setattr__(self, "pre_mutation_floor", pre)
        object.__setattr__(self, "post_mutation_floor", post)

    @property
    def version_floor(self) -> datetime:
        return max(self.pre_mutation_floor, self.post_mutation_floor)


@dataclass(frozen=True, slots=True)
class MembershipWriteResult:
    """Request-ordered outcomes and complete anchor inputs for changed users."""

    mutation_results: tuple[MembershipMutationResult, ...]
    affected_user_ids: tuple[int, ...]
    version_floors: tuple[MembershipUserVersionFloor, ...]

    def __post_init__(self) -> None:
        _require_exact_tuple(self.mutation_results)
        _require_exact_tuple(self.version_floors)
        if any(
            type(item) is not MembershipMutationResult
            for item in self.mutation_results
        ):
            raise MembershipWriterContractError()
        _validate_sorted_unique_ids(self.affected_user_ids)
        if any(
            type(item) is not MembershipUserVersionFloor
            for item in self.version_floors
        ):
            raise MembershipWriterContractError()
        floor_ids = tuple(item.user_id for item in self.version_floors)
        if floor_ids != self.affected_user_ids:
            raise MembershipWriterContractError()
        changed_user_ids = tuple(
            sorted(
                {
                    item.mutation.user_id
                    for item in self.mutation_results
                    if item.changed
                }
            )
        )
        if changed_user_ids != self.affected_user_ids:
            raise MembershipWriterContractError()

    def floor_for(self, user_id: int) -> datetime:
        _require_positive_id(user_id)
        for item in self.version_floors:
            if item.user_id == user_id:
                return item.version_floor
        raise MembershipWriterContractError()


def _normalize_contract_time(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is None:
        raise MembershipWriterContractError()
    try:
        normalized = normalize_profile_version(value)
    except ProfileVersionError:
        raise MembershipWriterContractError() from None
    return normalized.astimezone(timezone.utc)


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
class MembershipRowSnapshot:
    """Complete persisted membership row captured during deletion discovery."""

    scope_type: MembershipScopeType
    scope_id: int
    user_id: int
    role: str
    status: str

    def __post_init__(self) -> None:
        if type(self.scope_type) is not MembershipScopeType:
            raise MembershipWriterContractError()
        _require_positive_id(self.scope_id)
        _require_positive_id(self.user_id)
        if type(self.role) is not str or type(self.status) is not str:
            raise MembershipWriterContractError()

    @property
    def identity(self) -> MembershipRowLock:
        return MembershipRowLock(
            scope_type=self.scope_type,
            scope_id=self.scope_id,
            user_id=self.user_id,
        )


@dataclass(frozen=True, slots=True)
class TeamParentOrganization:
    """Immutable parent-organization data captured before lock planning."""

    team_id: int
    organization_id: int

    def __post_init__(self) -> None:
        _require_positive_id(self.team_id)
        _require_positive_id(self.organization_id)


@dataclass(frozen=True, slots=True)
class MembershipScopeDeletionSnapshot:
    """Frozen complete scope and membership set used by a deletion attempt."""

    scope_type: MembershipScopeType
    scope_id: int
    organization_ids: tuple[int, ...]
    team_parents: tuple[TeamParentOrganization, ...]
    membership_rows: tuple[MembershipRowSnapshot, ...]

    def __post_init__(self) -> None:
        if type(self.scope_type) is not MembershipScopeType:
            raise MembershipWriterContractError()
        _require_positive_id(self.scope_id)
        _validate_sorted_unique_ids(self.organization_ids)
        _require_exact_tuple(self.team_parents)
        _require_exact_tuple(self.membership_rows)
        if any(
            type(item) is not TeamParentOrganization
            for item in self.team_parents
        ):
            raise MembershipWriterContractError()
        if any(
            type(item) is not MembershipRowSnapshot
            for item in self.membership_rows
        ):
            raise MembershipWriterContractError()

        team_ids = tuple(item.team_id for item in self.team_parents)
        if team_ids != tuple(sorted(set(team_ids))):
            raise MembershipWriterContractError()
        if any(
            item.organization_id not in self.organization_ids
            for item in self.team_parents
        ):
            raise MembershipWriterContractError()
        row_identities = tuple(item.identity for item in self.membership_rows)
        if row_identities != tuple(sorted(set(row_identities), key=_row_sort_key)):
            raise MembershipWriterContractError()

        if self.scope_type is MembershipScopeType.ORGANIZATION:
            if self.organization_ids != (self.scope_id,):
                raise MembershipWriterContractError()
            if any(
                row.scope_type is MembershipScopeType.ORGANIZATION
                and row.scope_id != self.scope_id
                for row in self.membership_rows
            ):
                raise MembershipWriterContractError()
        else:
            if len(self.organization_ids) != 1 or team_ids != (self.scope_id,):
                raise MembershipWriterContractError()
            if any(
                row.scope_type is not MembershipScopeType.TEAM
                or row.scope_id != self.scope_id
                for row in self.membership_rows
            ):
                raise MembershipWriterContractError()

        known_team_ids = set(team_ids)
        if any(
            row.scope_type is MembershipScopeType.TEAM
            and row.scope_id not in known_team_ids
            for row in self.membership_rows
        ):
            raise MembershipWriterContractError()

    @property
    def team_ids(self) -> tuple[int, ...]:
        return tuple(item.team_id for item in self.team_parents)

    @property
    def affected_user_ids(self) -> tuple[int, ...]:
        return tuple(sorted({row.user_id for row in self.membership_rows}))


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

    for index, mutation in enumerate(mutations):
        if mutation.relationship not in {
            MembershipMutationRelationship.DEFAULT_TEAM_COMPANION,
            MembershipMutationRelationship.ORGANIZATION_COMPANION,
        }:
            continue
        parent_org_id = team_to_org[mutation.scope_id]
        if not any(
            candidate.scope_type is MembershipScopeType.ORGANIZATION
            and candidate.scope_id == parent_org_id
            and candidate.user_id == mutation.user_id
            and candidate.kind is mutation.kind
            for candidate in mutations[:index]
        ):
            raise MembershipWriterContractError()

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
            membership_rows.update(
                MembershipRowLock(
                    scope_type=MembershipScopeType.TEAM,
                    scope_id=team_id,
                    user_id=context.actor_user_id,
                )
                for team_id in team_ids
            )
    membership_rows.update(
        MembershipRowLock(
            scope_type=MembershipScopeType.ORGANIZATION,
            scope_id=team_to_org[mutation.scope_id],
            user_id=mutation.user_id,
        )
        for mutation in mutations
        if mutation.scope_type is MembershipScopeType.TEAM
        and mutation.kind is MembershipMutationKind.ADD
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
_ROLE_AUTHORITY_LOCK_SQL = (
    "SELECT r.id FROM public.roles r WHERE EXISTS ("
    "SELECT 1 FROM public.user_roles ur "
    "WHERE ur.user_id = $1 AND ur.role_id = r.id) "
    "ORDER BY r.id FOR UPDATE OF r"
)
_PERMISSION_AUTHORITY_LOCK_SQL = (
    "SELECT p.id FROM public.permissions p WHERE EXISTS ("
    "SELECT 1 FROM public.role_permissions rp "
    "JOIN public.user_roles ur ON ur.role_id = rp.role_id "
    "WHERE ur.user_id = $1 AND rp.permission_id = p.id) OR EXISTS ("
    "SELECT 1 FROM public.user_permissions up "
    "WHERE up.user_id = $1 AND up.permission_id = p.id) "
    "ORDER BY p.id FOR UPDATE OF p"
)
_USER_ROLE_AUTHORITY_LOCK_SQL = (
    "SELECT ur.role_id FROM public.user_roles ur WHERE ur.user_id = $1 "
    "ORDER BY ur.role_id FOR UPDATE OF ur"
)
_ROLE_PERMISSION_AUTHORITY_LOCK_SQL = (
    "SELECT rp.role_id, rp.permission_id FROM public.role_permissions rp "
    "WHERE EXISTS (SELECT 1 FROM public.user_roles ur "
    "WHERE ur.user_id = $1 AND ur.role_id = rp.role_id) "
    "ORDER BY rp.role_id, rp.permission_id FOR UPDATE OF rp"
)
_USER_PERMISSION_AUTHORITY_LOCK_SQL = (
    "SELECT up.permission_id FROM public.user_permissions up "
    "WHERE up.user_id = $1 ORDER BY up.permission_id FOR UPDATE OF up"
)
_ACTIVE_MEMBERSHIP_SQL = "LOWER(COALESCE(status, '')) = 'active'"


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
    if (
        type(plan.context) is ActorMembershipWriteContext
        and plan.context.required_authority is MembershipAuthority.PLATFORM_ADMIN
    ):
        statements.extend(
            MembershipLockStatement(
                phase=MembershipLockPhase.AUTHORITY_ROWS,
                sql=sql,
                parameters=(plan.context.actor_user_id,),
            )
            for sql in (
                _ROLE_AUTHORITY_LOCK_SQL,
                _PERMISSION_AUTHORITY_LOCK_SQL,
                _USER_ROLE_AUTHORITY_LOCK_SQL,
                _ROLE_PERMISSION_AUTHORITY_LOCK_SQL,
                _USER_PERMISSION_AUTHORITY_LOCK_SQL,
            )
        )
    return tuple(statements)


class MembershipWriter:
    """Apply ordered membership mutations on one caller-owned transaction."""

    def __init__(self, db_pool: Any) -> None:
        self._db_pool = db_pool
        self._backend = (
            MembershipLockBackend.POSTGRESQL
            if getattr(db_pool, "pool", None) is not None
            else MembershipLockBackend.SQLITE
        )

    async def authorize_organization_creation(
        self,
        *,
        conn: Any,
        context: MembershipWriteContext,
        owner_user_id: int | None,
    ) -> None:
        """Lock and authorize user rows before inserting an organization."""

        validate_membership_write_context(context, serving=True)
        if owner_user_id is not None:
            _require_positive_id(owner_user_id)
        user_ids = {owner_user_id} if owner_user_id is not None else set()
        if type(context) is ActorMembershipWriteContext:
            user_ids.add(context.actor_user_id)

        missing_user_ids: set[int] = set()
        for user_id in sorted(user_ids):
            if self._backend is MembershipLockBackend.POSTGRESQL:
                row = await conn.fetchrow(_USER_LOCK_SQL, user_id)
            else:
                row = await self._read_user(conn, user_id)
            if row is None:
                missing_user_ids.add(user_id)

        if type(context) is ActorMembershipWriteContext:
            if context.actor_user_id in missing_user_ids:
                raise MembershipAuthorizationError()
            if context.required_authority is MembershipAuthority.PLATFORM_ADMIN:
                await self._authorize_platform_admin_actor(
                    conn,
                    context.actor_user_id,
                )
            elif owner_user_id == context.actor_user_id:
                actor = await self._read_user(conn, context.actor_user_id)
                if actor is None or not _is_active(actor.get("is_active")):
                    raise MembershipAuthorizationError()
            else:
                raise MembershipAuthorizationError()
        if owner_user_id in missing_user_ids:
            raise MembershipTargetNotFound()

    async def apply_membership_mutations(
        self,
        *,
        conn: Any,
        context: MembershipWriteContext,
        mutations: tuple[MembershipMutation, ...],
        anchor_ownership: AnchorOwnership,
        operation_time: datetime,
    ) -> MembershipWriteResult:
        """Apply mutations without acquiring or ending a database transaction."""

        validate_membership_write_context(context, serving=True)
        if type(anchor_ownership) is not AnchorOwnership:
            raise MembershipWriterContractError()
        _require_exact_tuple(mutations)
        if any(type(item) is not MembershipMutation for item in mutations):
            raise MembershipWriterContractError()
        operation_time = _normalize_contract_time(operation_time)
        if not mutations:
            return MembershipWriteResult((), (), ())

        preflight = await self._read_preflight(conn, mutations)
        plan = plan_membership_write(
            context=context,
            mutations=mutations,
            preflight=preflight,
        )
        await self._execute_lock_plan(conn, plan)
        await self._recheck_preflight(conn, plan)
        scopes = await self._read_locked_scopes(conn, plan)
        await self._authorize_context(conn, plan, scopes)
        await self._require_targets_exist(conn, mutations)
        blocked_parent_membership = await self._team_add_parent_preconditions(
            conn,
            plan,
            scopes,
        )

        target_user_ids = tuple(sorted({item.user_id for item in mutations}))
        version_gateway = VersionedUserWriteGateway(
            "postgres"
            if self._backend is MembershipLockBackend.POSTGRESQL
            else "sqlite",
            clock=lambda: operation_time,
        )
        pre_floors = {
            user_id: await version_gateway.capture_floor(
                conn,
                user_id=user_id,
                lock_user=False,
            )
            for user_id in target_user_ids
        }

        results: list[MembershipMutationResult] = []
        blocked_org_removals: set[tuple[int, int]] = set()
        for index, mutation in enumerate(mutations):
            if index in blocked_parent_membership:
                results.append(
                    MembershipMutationResult(
                        mutation=mutation,
                        changed=False,
                        found=False,
                        organization_id=int(
                            scopes["teams"][mutation.scope_id]["org_id"]
                        ),
                        error="org_membership_required",
                    )
                )
                continue
            if self._is_blocked_team_companion(
                mutation,
                scopes,
                blocked_org_removals,
            ):
                results.append(await self._skipped_remove_result(conn, mutation, scopes))
                continue
            result = await self._apply_mutation(conn, mutation, scopes)
            results.append(result)
            if (
                mutation.scope_type is MembershipScopeType.ORGANIZATION
                and mutation.kind is MembershipMutationKind.REMOVE
                and result.error == "owner_required"
            ):
                blocked_org_removals.add((mutation.scope_id, mutation.user_id))

        affected_user_ids = tuple(
            sorted(
                {
                    result.mutation.user_id
                    for result in results
                    if result.changed
                }
            )
        )
        floors: list[MembershipUserVersionFloor] = []
        for user_id in affected_user_ids:
            post_floor = await version_gateway.capture_floor(
                conn,
                user_id=user_id,
                lock_user=False,
            )
            floor = MembershipUserVersionFloor(
                user_id=user_id,
                pre_mutation_floor=pre_floors[user_id],
                post_mutation_floor=post_floor,
            )
            floors.append(floor)
            if anchor_ownership is AnchorOwnership.WRITER_OWNS_ANCHOR:
                await version_gateway.final_touch(
                    conn,
                    user_id=user_id,
                    version_floor=floor.version_floor,
                )

        return MembershipWriteResult(
            mutation_results=tuple(results),
            affected_user_ids=affected_user_ids,
            version_floors=tuple(floors),
        )

    async def lock_provisioning_parent(
        self,
        *,
        conn: Any,
        context: MembershipWriteContext,
        organization_id: int,
        target_user_id: int,
    ) -> None:
        """Lock lower-order rows before creating a default team in this transaction."""

        validate_membership_write_context(context, serving=True)
        _require_positive_id(organization_id)
        _require_positive_id(target_user_id)
        if self._backend is MembershipLockBackend.SQLITE:
            organization = await self._read_organization(conn, organization_id)
            if organization is None or not _is_active(organization.get("is_active")):
                raise MembershipScopeNotFound()
            if await self._read_user(conn, target_user_id) is None:
                raise MembershipTargetNotFound()
            return

        user_ids = {target_user_id}
        if type(context) is ActorMembershipWriteContext:
            user_ids.add(context.actor_user_id)
        for user_id in sorted(user_ids):
            row = await conn.fetchrow(_USER_LOCK_SQL, user_id)
            if row is None:
                raise MembershipTargetNotFound()
        organization = await conn.fetchrow(_ORGANIZATION_LOCK_SQL, organization_id)
        if organization is None:
            raise MembershipScopeNotFound()
        locked = await self._read_organization(conn, organization_id)
        if locked is None or not _is_active(locked.get("is_active")):
            raise MembershipScopeNotFound()

    async def transfer_organization_ownership(
        self,
        *,
        conn: Any,
        context: ActorMembershipWriteContext,
        organization_id: int,
        current_owner_user_id: int,
        new_owner_user_id: int,
        anchor_ownership: AnchorOwnership,
        operation_time: datetime,
    ) -> tuple[MembershipMutationResult, MembershipMutationResult]:
        """Transfer owner pointer and roles under the membership lock order."""

        if type(context) is not ActorMembershipWriteContext:
            raise MembershipWriterContractError()
        validate_membership_write_context(context, serving=True)
        _require_positive_id(organization_id)
        _require_positive_id(current_owner_user_id)
        _require_positive_id(new_owner_user_id)
        if current_owner_user_id == new_owner_user_id:
            raise MembershipWriterContractError()
        if type(anchor_ownership) is not AnchorOwnership:
            raise MembershipWriterContractError()
        operation_time = _normalize_contract_time(operation_time)
        mutations = (
            MembershipMutation(
                scope_type=MembershipScopeType.ORGANIZATION,
                scope_id=organization_id,
                user_id=new_owner_user_id,
                kind=MembershipMutationKind.UPDATE_ROLE,
                role="owner",
            ),
            MembershipMutation(
                scope_type=MembershipScopeType.ORGANIZATION,
                scope_id=organization_id,
                user_id=current_owner_user_id,
                kind=MembershipMutationKind.UPDATE_ROLE,
                role="admin",
            ),
        )
        before_lock = await self._read_organization(conn, organization_id)
        if before_lock is None:
            raise MembershipScopeNotFound()
        if int(before_lock.get("owner_user_id") or 0) != current_owner_user_id:
            raise MembershipPreflightChanged()

        preflight = await self._read_preflight(conn, mutations)
        plan = plan_membership_write(
            context=context,
            mutations=mutations,
            preflight=preflight,
        )
        await self._execute_lock_plan(conn, plan)
        await self._recheck_preflight(conn, plan)
        scopes = await self._read_locked_scopes(conn, plan)
        await self._authorize_context(conn, plan, scopes)
        await self._require_targets_exist(conn, mutations)

        locked_organization = scopes["organizations"][organization_id]
        if (
            int(locked_organization.get("owner_user_id") or 0)
            != current_owner_user_id
        ):
            raise MembershipPreflightChanged()
        scoped_actor = (
            context
            if type(context) is ActorMembershipWriteContext
            and context.required_authority is MembershipAuthority.SCOPED_MEMBERSHIP
            else None
        )
        if (
            scoped_actor is not None
            and scoped_actor.actor_user_id != current_owner_user_id
        ):
            raise MembershipAuthorizationError()
        for user_id in (current_owner_user_id, new_owner_user_id):
            user = await self._read_user(conn, user_id)
            membership = await self._read_membership(
                conn,
                MembershipScopeType.ORGANIZATION,
                organization_id,
                user_id,
            )
            if user is None or not _is_active(user.get("is_active")):
                raise MembershipPreflightChanged()
            if not _is_active_membership(membership):
                raise MembershipPreflightChanged()
            if (
                user_id == current_owner_user_id
                and str(membership.get("role") or "").strip().lower() != "owner"
            ):
                if scoped_actor is not None:
                    raise MembershipAuthorizationError()
                raise MembershipPreflightChanged()

        user_ids = tuple(sorted((current_owner_user_id, new_owner_user_id)))
        version_gateway = VersionedUserWriteGateway(
            "postgres"
            if self._backend is MembershipLockBackend.POSTGRESQL
            else "sqlite",
            clock=lambda: operation_time,
        )
        pre_floors = {
            user_id: await version_gateway.capture_floor(
                conn,
                user_id=user_id,
                lock_user=False,
            )
            for user_id in user_ids
        }
        results = (
            await self._apply_mutation(conn, mutations[0], scopes),
            await self._apply_mutation(conn, mutations[1], scopes),
        )
        if not all(result.found for result in results):
            raise MembershipPreflightChanged()
        await self._update_organization_owner_pointer(
            conn,
            organization_id=organization_id,
            new_owner_user_id=new_owner_user_id,
        )
        for user_id in user_ids:
            post_floor = await version_gateway.capture_floor(
                conn,
                user_id=user_id,
                lock_user=False,
            )
            if anchor_ownership is AnchorOwnership.WRITER_OWNS_ANCHOR:
                await version_gateway.final_touch(
                    conn,
                    user_id=user_id,
                    version_floor=max(pre_floors[user_id], post_floor),
                )
        return results

    async def discover_scope_deletion(
        self,
        *,
        conn: Any,
        scope_type: MembershipScopeType,
        scope_id: int,
    ) -> MembershipScopeDeletionSnapshot | None:
        """Read the complete affected set without opening a transaction."""

        if type(scope_type) is not MembershipScopeType:
            raise MembershipWriterContractError()
        _require_positive_id(scope_id)

        if scope_type is MembershipScopeType.ORGANIZATION:
            organization = await self._read_organization(conn, scope_id)
            if organization is None:
                return None
            team_parents = await self._read_child_team_parents(conn, scope_id)
            membership_rows = await self._read_scope_membership_snapshots(
                conn,
                scope_type=scope_type,
                scope_id=scope_id,
                team_ids=tuple(item.team_id for item in team_parents),
            )
            return MembershipScopeDeletionSnapshot(
                scope_type=scope_type,
                scope_id=scope_id,
                organization_ids=(scope_id,),
                team_parents=team_parents,
                membership_rows=membership_rows,
            )

        team = await self._read_team(conn, scope_id)
        if team is None:
            return None
        organization_id = int(team["org_id"])
        return MembershipScopeDeletionSnapshot(
            scope_type=scope_type,
            scope_id=scope_id,
            organization_ids=(organization_id,),
            team_parents=(
                TeamParentOrganization(
                    team_id=scope_id,
                    organization_id=organization_id,
                ),
            ),
            membership_rows=await self._read_scope_membership_snapshots(
                conn,
                scope_type=scope_type,
                scope_id=scope_id,
                team_ids=(scope_id,),
            ),
        )

    async def apply_scope_deletion(
        self,
        *,
        conn: Any,
        context: MembershipWriteContext,
        snapshot: MembershipScopeDeletionSnapshot,
        anchor_ownership: AnchorOwnership,
        operation_time: datetime,
    ) -> MembershipWriteResult:
        """Remove a frozen scope's memberships on one caller transaction."""

        validate_membership_write_context(context, serving=True)
        if type(snapshot) is not MembershipScopeDeletionSnapshot:
            raise MembershipWriterContractError()
        if type(anchor_ownership) is not AnchorOwnership:
            raise MembershipWriterContractError()
        operation_time = _normalize_contract_time(operation_time)

        lock_set = self._build_scope_deletion_lock_set(
            context=context,
            snapshot=snapshot,
        )
        await self._execute_scope_deletion_locks(
            conn,
            context=context,
            snapshot=snapshot,
            lock_set=lock_set,
        )
        await self._require_deletion_scopes_exist(conn, snapshot)
        await self._authorize_scope_deletion(conn, context, snapshot)

        version_gateway = VersionedUserWriteGateway(
            "postgres"
            if self._backend is MembershipLockBackend.POSTGRESQL
            else "sqlite",
            clock=lambda: operation_time,
        )
        pre_floors = {
            user_id: await version_gateway.capture_floor(
                conn,
                user_id=user_id,
                lock_user=False,
            )
            for user_id in snapshot.affected_user_ids
        }
        team_to_org = {
            item.team_id: item.organization_id
            for item in snapshot.team_parents
        }
        results: list[MembershipMutationResult] = []
        for row in snapshot.membership_rows:
            mutation = MembershipMutation(
                scope_type=row.scope_type,
                scope_id=row.scope_id,
                user_id=row.user_id,
                kind=MembershipMutationKind.REMOVE,
            )
            await self._delete_membership(conn, mutation)
            results.append(
                MembershipMutationResult(
                    mutation=mutation,
                    changed=True,
                    found=True,
                    organization_id=(
                        team_to_org[row.scope_id]
                        if row.scope_type is MembershipScopeType.TEAM
                        else None
                    ),
                )
            )

        floors: list[MembershipUserVersionFloor] = []
        for user_id in snapshot.affected_user_ids:
            floor = MembershipUserVersionFloor(
                user_id=user_id,
                pre_mutation_floor=pre_floors[user_id],
                post_mutation_floor=await version_gateway.capture_floor(
                    conn,
                    user_id=user_id,
                    lock_user=False,
                ),
            )
            floors.append(floor)
            if anchor_ownership is AnchorOwnership.WRITER_OWNS_ANCHOR:
                await version_gateway.final_touch(
                    conn,
                    user_id=user_id,
                    version_floor=floor.version_floor,
                )

        return MembershipWriteResult(
            mutation_results=tuple(results),
            affected_user_ids=snapshot.affected_user_ids,
            version_floors=tuple(floors),
        )

    @staticmethod
    def is_scope_deletion_retry(exc: BaseException) -> bool:
        """Identify only the writer's private transaction-control signal."""

        return type(exc) is _MembershipScopeDeletionRetry

    def _build_scope_deletion_lock_set(
        self,
        *,
        context: MembershipWriteContext,
        snapshot: MembershipScopeDeletionSnapshot,
    ) -> MembershipLockSet:
        user_ids = set(snapshot.affected_user_ids)
        membership_rows = {row.identity for row in snapshot.membership_rows}
        owner_rows = {
            row.identity
            for row in snapshot.membership_rows
            if snapshot.scope_type is MembershipScopeType.ORGANIZATION
            and row.scope_type is MembershipScopeType.ORGANIZATION
            and row.role.strip().lower() == "owner"
            and row.status.strip().lower() == "active"
        }
        if type(context) is ActorMembershipWriteContext:
            user_ids.add(context.actor_user_id)
            if context.required_authority is MembershipAuthority.SCOPED_MEMBERSHIP:
                membership_rows.add(
                    MembershipRowLock(
                        scope_type=MembershipScopeType.ORGANIZATION,
                        scope_id=snapshot.organization_ids[0],
                        user_id=context.actor_user_id,
                    )
                )
        membership_rows.difference_update(owner_rows)
        return MembershipLockSet(
            user_ids=tuple(sorted(user_ids)),
            org_ids=snapshot.organization_ids,
            team_ids=snapshot.team_ids,
            membership_rows=tuple(sorted(membership_rows, key=_row_sort_key)),
            owner_rows=tuple(sorted(owner_rows, key=_row_sort_key)),
        )

    async def _execute_scope_deletion_locks(
        self,
        conn: Any,
        *,
        context: MembershipWriteContext,
        snapshot: MembershipScopeDeletionSnapshot,
        lock_set: MembershipLockSet,
    ) -> None:
        del context
        if self._backend is MembershipLockBackend.SQLITE:
            await self._recheck_scope_deletion_snapshot(conn, snapshot)
            return

        for user_id in lock_set.user_ids:
            await conn.fetchrow(_USER_LOCK_SQL, user_id)
        for org_id in lock_set.org_ids:
            await conn.fetchrow(_ORGANIZATION_LOCK_SQL, org_id)
        if snapshot.scope_type is MembershipScopeType.ORGANIZATION:
            await self._recheck_scope_deletion_snapshot(conn, snapshot)
        for team_id in lock_set.team_ids:
            await conn.fetchrow(_TEAM_LOCK_SQL, team_id)
        if snapshot.scope_type is MembershipScopeType.TEAM:
            await self._recheck_scope_deletion_snapshot(conn, snapshot)
        for row in lock_set.membership_rows:
            statement = _membership_statement(
                row,
                MembershipLockPhase.MEMBERSHIP_ROWS,
            )
            await conn.fetchrow(statement.sql, *statement.parameters)
        for row in lock_set.owner_rows:
            statement = _membership_statement(row, MembershipLockPhase.OWNER_ROWS)
            await conn.fetchrow(statement.sql, *statement.parameters)

    async def _recheck_scope_deletion_snapshot(
        self,
        conn: Any,
        snapshot: MembershipScopeDeletionSnapshot,
    ) -> None:
        current = await self.discover_scope_deletion(
            conn=conn,
            scope_type=snapshot.scope_type,
            scope_id=snapshot.scope_id,
        )
        if current != snapshot:
            raise _MembershipScopeDeletionRetry()

    async def _require_deletion_scopes_exist(
        self,
        conn: Any,
        snapshot: MembershipScopeDeletionSnapshot,
    ) -> None:
        for organization_id in snapshot.organization_ids:
            organization = await self._read_organization(conn, organization_id)
            if organization is None:
                raise MembershipScopeNotFound()
        team_to_org = {
            item.team_id: item.organization_id
            for item in snapshot.team_parents
        }
        for team_id in snapshot.team_ids:
            team = await self._read_team(conn, team_id)
            if (
                team is None
                or int(team.get("org_id") or 0) != team_to_org[team_id]
            ):
                raise MembershipScopeNotFound()

    async def _authorize_scope_deletion(
        self,
        conn: Any,
        context: MembershipWriteContext,
        snapshot: MembershipScopeDeletionSnapshot,
    ) -> None:
        if type(context) is TrustedMembershipWriteContext:
            return
        actor = await self._read_user(conn, context.actor_user_id)
        if actor is None or not _is_active(actor.get("is_active")):
            raise MembershipAuthorizationError()
        if context.required_authority is MembershipAuthority.PLATFORM_ADMIN:
            role = str(actor.get("role") or "").strip().lower()
            if bool(actor.get("is_superuser")) or role in {
                "owner",
                "super_admin",
                "admin",
            }:
                return
            if await self._has_persisted_platform_admin(conn, context.actor_user_id):
                return
            raise MembershipAuthorizationError()

        organization_id = snapshot.organization_ids[0]
        membership = await self._read_membership(
            conn,
            MembershipScopeType.ORGANIZATION,
            organization_id,
            context.actor_user_id,
        )
        if snapshot.scope_type is MembershipScopeType.ORGANIZATION:
            organization = await self._read_organization(conn, organization_id)
            if (
                organization is not None
                and int(organization.get("owner_user_id") or 0)
                == context.actor_user_id
                and _is_active_membership(membership)
                and str(membership.get("role") or "").strip().lower() == "owner"
            ):
                return
            raise MembershipAuthorizationError()
        if _is_active_membership_admin(membership):
            return
        raise MembershipAuthorizationError()

    async def _team_add_parent_preconditions(
        self,
        conn: Any,
        plan: MembershipLockPlan,
        scopes: dict[str, dict[int, dict[str, Any]]],
    ) -> set[int]:
        blocked: set[int] = set()
        for index, mutation in enumerate(plan.mutations):
            if (
                mutation.scope_type is not MembershipScopeType.TEAM
                or mutation.kind is not MembershipMutationKind.ADD
            ):
                continue
            organization_id = int(scopes["teams"][mutation.scope_id]["org_id"])
            membership = await self._read_membership(
                conn,
                MembershipScopeType.ORGANIZATION,
                organization_id,
                mutation.user_id,
            )
            if _is_active_membership(membership):
                continue
            if membership is None and mutation.relationship in {
                MembershipMutationRelationship.DEFAULT_TEAM_COMPANION,
                MembershipMutationRelationship.ORGANIZATION_COMPANION,
            }:
                continue
            blocked.add(index)
        return blocked

    async def _read_preflight(
        self,
        conn: Any,
        mutations: tuple[MembershipMutation, ...],
    ) -> MembershipPlanningPreflight:
        team_parents: list[TeamParentOrganization] = []
        for team_id in sorted(
            {
                mutation.scope_id
                for mutation in mutations
                if mutation.scope_type is MembershipScopeType.TEAM
            }
        ):
            row = await self._read_team(conn, team_id)
            if row is None:
                raise MembershipScopeNotFound()
            team_parents.append(
                TeamParentOrganization(
                    team_id=team_id,
                    organization_id=int(row["org_id"]),
                )
            )

        owner_org_ids = sorted(
            {
                mutation.scope_id
                for mutation in mutations
                if mutation.scope_type is MembershipScopeType.ORGANIZATION
                and mutation.kind
                in {MembershipMutationKind.REMOVE, MembershipMutationKind.UPDATE_ROLE}
            }
        )
        organization_owners: list[OrganizationOwnerPreflight] = []
        for org_id in owner_org_ids:
            organization_owners.append(
                OrganizationOwnerPreflight(
                    organization_id=org_id,
                    owner_user_ids=await self._read_owner_user_ids(conn, org_id),
                )
            )
        return MembershipPlanningPreflight(
            team_parents=tuple(team_parents),
            organization_owners=tuple(organization_owners),
        )

    async def _execute_lock_plan(
        self,
        conn: Any,
        plan: MembershipLockPlan,
    ) -> None:
        for statement in plan_membership_lock_statements(
            plan,
            backend=self._backend,
        ):
            if statement.phase is MembershipLockPhase.AUTHORITY_ROWS:
                await conn.fetch(statement.sql, *statement.parameters)
            else:
                await conn.fetchrow(statement.sql, *statement.parameters)

    async def _recheck_preflight(
        self,
        conn: Any,
        plan: MembershipLockPlan,
    ) -> None:
        current = await self._read_preflight(conn, plan.mutations)
        if current != plan.preflight:
            raise MembershipPreflightChanged()

    async def _read_locked_scopes(
        self,
        conn: Any,
        plan: MembershipLockPlan,
    ) -> dict[str, dict[int, dict[str, Any]]]:
        organizations: dict[int, dict[str, Any]] = {}
        for org_id in plan.lock_set.org_ids:
            row = await self._read_organization(conn, org_id)
            if row is None or not _is_active(row.get("is_active")):
                raise MembershipScopeNotFound()
            organizations[org_id] = row

        teams: dict[int, dict[str, Any]] = {}
        for team_id in plan.lock_set.team_ids:
            row = await self._read_team(conn, team_id)
            if row is None or not _is_active(row.get("is_active")):
                raise MembershipScopeNotFound()
            parent_org_id = int(row["org_id"])
            if parent_org_id not in organizations:
                raise MembershipPreflightChanged()
            teams[team_id] = row
        return {"organizations": organizations, "teams": teams}

    async def _authorize_context(
        self,
        conn: Any,
        plan: MembershipLockPlan,
        scopes: dict[str, dict[int, dict[str, Any]]],
    ) -> None:
        context = plan.context
        if type(context) is TrustedMembershipWriteContext:
            return

        actor = await self._read_user(conn, context.actor_user_id)
        if actor is None or not _is_active(actor.get("is_active")):
            raise MembershipAuthorizationError()
        if context.required_authority is MembershipAuthority.PLATFORM_ADMIN:
            await self._authorize_platform_admin_actor(
                conn,
                context.actor_user_id,
                actor=actor,
            )
            return

        organization_mutation_ids = sorted(
            {
                mutation.scope_id
                for mutation in plan.mutations
                if mutation.scope_type is MembershipScopeType.ORGANIZATION
            }
        )
        for org_id in organization_mutation_ids:
            organization = scopes["organizations"][org_id]
            membership = await self._read_membership(
                conn,
                MembershipScopeType.ORGANIZATION,
                org_id,
                context.actor_user_id,
            )
            if _is_active_membership_admin(membership):
                continue
            if self._is_self_owner_bootstrap(plan, org_id, organization):
                continue
            raise MembershipAuthorizationError()

        for team_id in sorted(
            {
                mutation.scope_id
                for mutation in plan.mutations
                if mutation.scope_type is MembershipScopeType.TEAM
            }
        ):
            organization_id = int(scopes["teams"][team_id]["org_id"])
            organization_membership = await self._read_membership(
                conn,
                MembershipScopeType.ORGANIZATION,
                organization_id,
                context.actor_user_id,
            )
            if _is_active_membership_admin(organization_membership):
                continue
            team_membership = await self._read_membership(
                conn,
                MembershipScopeType.TEAM,
                team_id,
                context.actor_user_id,
            )
            if _is_active_team_membership_admin(team_membership):
                continue
            if self._is_self_owner_bootstrap(
                plan,
                organization_id,
                scopes["organizations"][organization_id],
            ):
                continue
            raise MembershipAuthorizationError()

    async def _authorize_platform_admin_actor(
        self,
        conn: Any,
        actor_user_id: int,
        *,
        actor: dict[str, Any] | None = None,
    ) -> None:
        actor = actor or await self._read_user(conn, actor_user_id)
        if actor is None or not _is_active(actor.get("is_active")):
            raise MembershipAuthorizationError()
        role = str(actor.get("role") or "").strip().lower()
        legacy_admin = bool(actor.get("is_superuser")) or role in {
            "owner",
            "super_admin",
            "admin",
        }
        if not legacy_admin and not await self._has_persisted_platform_admin(
            conn,
            actor_user_id,
        ):
            raise MembershipAuthorizationError()

    async def _has_persisted_platform_admin(
        self,
        conn: Any,
        user_id: int,
    ) -> bool:
        """Resolve active global RBAC authority on the supplied connection."""

        if self._backend is MembershipLockBackend.POSTGRESQL:
            user_role_rows = await conn.fetch(
                "SELECT ur.role_id, "
                "(ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP) AS active "
                "FROM public.user_roles ur WHERE ur.user_id = $1 "
                "ORDER BY ur.role_id",
                user_id,
            )
            role_rows = await conn.fetch(
                "SELECT r.id, r.name FROM public.roles r "
                "WHERE EXISTS (SELECT 1 FROM public.user_roles ur "
                "WHERE ur.user_id = $1 AND ur.role_id = r.id) "
                "ORDER BY r.id",
                user_id,
            )
            role_permission_rows = await conn.fetch(
                "SELECT rp.role_id, rp.permission_id "
                "FROM public.role_permissions rp "
                "WHERE EXISTS (SELECT 1 FROM public.user_roles ur "
                "WHERE ur.user_id = $1 AND ur.role_id = rp.role_id) "
                "ORDER BY rp.role_id, rp.permission_id",
                user_id,
            )
            permission_rows = await conn.fetch(
                "SELECT p.id, p.name FROM public.permissions p WHERE "
                "EXISTS (SELECT 1 FROM public.role_permissions rp "
                "JOIN public.user_roles ur ON ur.role_id = rp.role_id "
                "WHERE ur.user_id = $1 AND rp.permission_id = p.id) OR "
                "EXISTS (SELECT 1 FROM public.user_permissions up "
                "WHERE up.user_id = $1 AND up.permission_id = p.id) "
                "ORDER BY p.id",
                user_id,
            )
            direct_rows = await conn.fetch(
                "SELECT up.permission_id, up.granted, "
                "(up.expires_at IS NULL OR up.expires_at > CURRENT_TIMESTAMP) AS active "
                "FROM public.user_permissions up WHERE up.user_id = $1 "
                "ORDER BY up.permission_id",
                user_id,
            )

            active_role_ids = {
                int(_row_value(row, "role_id", 0))
                for row in user_role_rows
                if bool(_row_value(row, "active", 1))
            }
            role_names = {
                str(_row_value(row, "name", 1)).strip().lower()
                for row in role_rows
                if int(_row_value(row, "id", 0)) in active_role_ids
            }
            permission_names = {
                int(_row_value(row, "id", 0)): str(
                    _row_value(row, "name", 1)
                ).strip().lower()
                for row in permission_rows
            }
            permissions = {
                permission_names[permission_id]
                for row in role_permission_rows
                if int(_row_value(row, "role_id", 0)) in active_role_ids
                and (permission_id := int(_row_value(row, "permission_id", 1)))
                in permission_names
            }
            for row in direct_rows:
                if not bool(_row_value(row, "active", 2)):
                    continue
                permission = permission_names.get(
                    int(_row_value(row, "permission_id", 0))
                )
                if permission is None:
                    continue
                if bool(_row_value(row, "granted", 1)):
                    permissions.add(permission)
                else:
                    permissions.discard(permission)
            return bool(role_names & {"owner", "super_admin", "admin"}) or bool(
                permissions & {"*", "system.configure"}
            )
        else:
            role_rows = await _sqlite_fetchall(
                conn,
                "SELECT r.name FROM main.user_roles ur "
                "JOIN main.roles r ON r.id = ur.role_id "
                "WHERE ur.user_id = ? AND "
                "(ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)",
                (user_id,),
            )
            role_permission_rows = await _sqlite_fetchall(
                conn,
                "SELECT p.name FROM main.permissions p "
                "JOIN main.role_permissions rp ON rp.permission_id = p.id "
                "JOIN main.user_roles ur ON ur.role_id = rp.role_id "
                "WHERE ur.user_id = ? AND "
                "(ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP)",
                (user_id,),
            )
            direct_rows = await _sqlite_fetchall(
                conn,
                "SELECT p.name, up.granted FROM main.permissions p "
                "JOIN main.user_permissions up ON up.permission_id = p.id "
                "WHERE up.user_id = ? AND "
                "(up.expires_at IS NULL OR up.expires_at > CURRENT_TIMESTAMP)",
                (user_id,),
            )

        role_names = {
            str(_row_value(row, "name", 0)).strip().lower() for row in role_rows
        }
        permissions = {
            str(_row_value(row, "name", 0)).strip().lower()
            for row in role_permission_rows
        }
        for row in direct_rows:
            permission = str(_row_value(row, "name", 0)).strip().lower()
            if bool(_row_value(row, "granted", 1)):
                permissions.add(permission)
            else:
                permissions.discard(permission)
        return bool(role_names & {"owner", "super_admin", "admin"}) or bool(
            permissions & {"*", "system.configure"}
        )

    async def has_persisted_platform_admin(
        self,
        conn: Any,
        user_id: int,
    ) -> bool:
        """Return persisted global authority after the caller locks the user."""

        _require_positive_id(user_id)
        return await self._has_persisted_platform_admin(conn, user_id)

    @staticmethod
    def _is_self_owner_bootstrap(
        plan: MembershipLockPlan,
        org_id: int,
        organization: dict[str, Any],
    ) -> bool:
        context = plan.context
        if type(context) is not ActorMembershipWriteContext:
            return False
        if organization.get("owner_user_id") != context.actor_user_id:
            return False
        org_mutations = []
        team_parents = {
            item.team_id: item.organization_id for item in plan.preflight.team_parents
        }
        for mutation in plan.mutations:
            mutation_org_id = (
                mutation.scope_id
                if mutation.scope_type is MembershipScopeType.ORGANIZATION
                else team_parents[mutation.scope_id]
            )
            if mutation_org_id == org_id:
                org_mutations.append(mutation)
        owner_adds = [
            mutation
            for mutation in org_mutations
            if mutation.scope_type is MembershipScopeType.ORGANIZATION
            and mutation.kind is MembershipMutationKind.ADD
            and mutation.user_id == context.actor_user_id
            and str(mutation.role).strip().lower() == "owner"
        ]
        return len(owner_adds) == 1 and all(
            mutation.user_id == context.actor_user_id
            and (
                mutation is owner_adds[0]
                or (
                    mutation.scope_type is MembershipScopeType.TEAM
                    and mutation.kind is MembershipMutationKind.ADD
                )
            )
            for mutation in org_mutations
        )

    async def _require_targets_exist(
        self,
        conn: Any,
        mutations: tuple[MembershipMutation, ...],
    ) -> None:
        for user_id in sorted({item.user_id for item in mutations}):
            if await self._read_user(conn, user_id) is None:
                raise MembershipTargetNotFound()

    async def _apply_mutation(
        self,
        conn: Any,
        mutation: MembershipMutation,
        scopes: dict[str, dict[int, dict[str, Any]]],
    ) -> MembershipMutationResult:
        current = await self._read_membership(
            conn,
            mutation.scope_type,
            mutation.scope_id,
            mutation.user_id,
        )
        organization_id = (
            int(scopes["teams"][mutation.scope_id]["org_id"])
            if mutation.scope_type is MembershipScopeType.TEAM
            else None
        )
        if mutation.kind is MembershipMutationKind.ADD:
            if current is not None:
                return MembershipMutationResult(
                    mutation=mutation,
                    changed=False,
                    found=True,
                    role=str(current.get("role") or mutation.role),
                    organization_id=organization_id,
                )
            await self._insert_membership(conn, mutation)
            return MembershipMutationResult(
                mutation=mutation,
                changed=True,
                found=True,
                role=mutation.role,
                organization_id=organization_id,
            )

        if current is None:
            return MembershipMutationResult(
                mutation=mutation,
                changed=False,
                found=False,
                organization_id=organization_id,
            )
        current_role = str(current.get("role") or "member")
        if (
            mutation.scope_type is MembershipScopeType.ORGANIZATION
            and _is_active_membership(current)
            and current_role.strip().lower() == "owner"
            and (
                mutation.kind is MembershipMutationKind.REMOVE
                or str(mutation.role).strip().lower() != "owner"
            )
            and await self._read_owner_count(conn, mutation.scope_id) <= 1
        ):
            return MembershipMutationResult(
                mutation=mutation,
                changed=False,
                found=True,
                role=current_role,
                error="owner_required",
            )

        if mutation.kind is MembershipMutationKind.UPDATE_ROLE:
            if current_role == mutation.role:
                return MembershipMutationResult(
                    mutation=mutation,
                    changed=False,
                    found=True,
                    role=current_role,
                    organization_id=organization_id,
                )
            await self._update_membership_role(conn, mutation)
            return MembershipMutationResult(
                mutation=mutation,
                changed=True,
                found=True,
                role=mutation.role,
                organization_id=organization_id,
            )

        await self._delete_membership(conn, mutation)
        return MembershipMutationResult(
            mutation=mutation,
            changed=True,
            found=True,
            organization_id=organization_id,
        )

    @staticmethod
    def _is_blocked_team_companion(
        mutation: MembershipMutation,
        scopes: dict[str, dict[int, dict[str, Any]]],
        blocked_org_removals: set[tuple[int, int]],
    ) -> bool:
        if (
            mutation.scope_type is not MembershipScopeType.TEAM
            or mutation.kind is not MembershipMutationKind.REMOVE
            or mutation.relationship
            not in {
                MembershipMutationRelationship.DEFAULT_TEAM_COMPANION,
                MembershipMutationRelationship.ORGANIZATION_COMPANION,
            }
        ):
            return False
        team = scopes["teams"][mutation.scope_id]
        return (int(team["org_id"]), mutation.user_id) in blocked_org_removals

    async def _skipped_remove_result(
        self,
        conn: Any,
        mutation: MembershipMutation,
        scopes: dict[str, dict[int, dict[str, Any]]],
    ) -> MembershipMutationResult:
        current = await self._read_membership(
            conn,
            mutation.scope_type,
            mutation.scope_id,
            mutation.user_id,
        )
        return MembershipMutationResult(
            mutation=mutation,
            changed=False,
            found=current is not None,
            organization_id=int(scopes["teams"][mutation.scope_id]["org_id"]),
        )

    async def _read_organization(
        self,
        conn: Any,
        org_id: int,
    ) -> dict[str, Any] | None:
        if self._backend is MembershipLockBackend.POSTGRESQL:
            row = await conn.fetchrow(
                "SELECT id, owner_user_id, is_active FROM public.organizations "
                "WHERE id = $1",
                org_id,
            )
        else:
            row = await _sqlite_fetchone(
                conn,
                "SELECT id, owner_user_id, is_active FROM main.organizations "
                "WHERE id = ?",
                (org_id,),
            )
        if row is None:
            return None
        return {
            "id": _row_value(row, "id", 0),
            "owner_user_id": _row_value(row, "owner_user_id", 1),
            "is_active": _row_value(row, "is_active", 2),
        }

    async def _read_team(self, conn: Any, team_id: int) -> dict[str, Any] | None:
        if self._backend is MembershipLockBackend.POSTGRESQL:
            row = await conn.fetchrow(
                "SELECT id, org_id, is_active FROM public.teams WHERE id = $1",
                team_id,
            )
        else:
            row = await _sqlite_fetchone(
                conn,
                "SELECT id, org_id, is_active FROM main.teams WHERE id = ?",
                (team_id,),
            )
        if row is None:
            return None
        return {
            "id": _row_value(row, "id", 0),
            "org_id": _row_value(row, "org_id", 1),
            "is_active": _row_value(row, "is_active", 2),
        }

    async def _read_child_team_parents(
        self,
        conn: Any,
        organization_id: int,
    ) -> tuple[TeamParentOrganization, ...]:
        if self._backend is MembershipLockBackend.POSTGRESQL:
            rows = await conn.fetch(
                "SELECT id, org_id FROM public.teams WHERE org_id = $1 ORDER BY id",
                organization_id,
            )
        else:
            rows = await _sqlite_fetchall(
                conn,
                "SELECT id, org_id FROM main.teams WHERE org_id = ? ORDER BY id",
                (organization_id,),
            )
        return tuple(
            TeamParentOrganization(
                team_id=int(_row_value(row, "id", 0)),
                organization_id=int(_row_value(row, "org_id", 1)),
            )
            for row in rows
        )

    async def _read_scope_membership_snapshots(
        self,
        conn: Any,
        *,
        scope_type: MembershipScopeType,
        scope_id: int,
        team_ids: tuple[int, ...],
    ) -> tuple[MembershipRowSnapshot, ...]:
        rows: list[MembershipRowSnapshot] = []
        if scope_type is MembershipScopeType.ORGANIZATION:
            if self._backend is MembershipLockBackend.POSTGRESQL:
                organization_rows = await conn.fetch(
                    "SELECT user_id, role, status FROM public.org_members "
                    "WHERE org_id = $1 ORDER BY user_id",
                    scope_id,
                )
            else:
                organization_rows = await _sqlite_fetchall(
                    conn,
                    "SELECT user_id, role, status FROM main.org_members "
                    "WHERE org_id = ? ORDER BY user_id",
                    (scope_id,),
                )
            rows.extend(
                MembershipRowSnapshot(
                    scope_type=MembershipScopeType.ORGANIZATION,
                    scope_id=scope_id,
                    user_id=int(_row_value(row, "user_id", 0)),
                    role=str(_row_value(row, "role", 1) or ""),
                    status=str(_row_value(row, "status", 2) or ""),
                )
                for row in organization_rows
            )

        for team_id in team_ids:
            if self._backend is MembershipLockBackend.POSTGRESQL:
                team_rows = await conn.fetch(
                    "SELECT user_id, role, status FROM public.team_members "
                    "WHERE team_id = $1 ORDER BY user_id",
                    team_id,
                )
            else:
                team_rows = await _sqlite_fetchall(
                    conn,
                    "SELECT user_id, role, status FROM main.team_members "
                    "WHERE team_id = ? ORDER BY user_id",
                    (team_id,),
                )
            rows.extend(
                MembershipRowSnapshot(
                    scope_type=MembershipScopeType.TEAM,
                    scope_id=team_id,
                    user_id=int(_row_value(row, "user_id", 0)),
                    role=str(_row_value(row, "role", 1) or ""),
                    status=str(_row_value(row, "status", 2) or ""),
                )
                for row in team_rows
            )
        return tuple(sorted(rows, key=lambda row: _row_sort_key(row.identity)))

    async def _read_user(self, conn: Any, user_id: int) -> dict[str, Any] | None:
        if self._backend is MembershipLockBackend.POSTGRESQL:
            row = await conn.fetchrow(
                "SELECT id, is_active, is_superuser, role FROM public.users "
                "WHERE id = $1",
                user_id,
            )
        else:
            row = await _sqlite_fetchone(
                conn,
                "SELECT id, is_active, is_superuser, role FROM main.users WHERE id = ?",
                (user_id,),
            )
        if row is None:
            return None
        return {
            "id": _row_value(row, "id", 0),
            "is_active": _row_value(row, "is_active", 1),
            "is_superuser": _row_value(row, "is_superuser", 2),
            "role": _row_value(row, "role", 3),
        }

    async def _read_membership(
        self,
        conn: Any,
        scope_type: MembershipScopeType,
        scope_id: int,
        user_id: int,
    ) -> dict[str, Any] | None:
        is_org = scope_type is MembershipScopeType.ORGANIZATION
        table = "org_members" if is_org else "team_members"
        scope_column = "org_id" if is_org else "team_id"
        if self._backend is MembershipLockBackend.POSTGRESQL:
            row = await conn.fetchrow(
                f"SELECT role, status FROM public.{table} "  # nosec B608
                f"WHERE {scope_column} = $1 AND user_id = $2",
                scope_id,
                user_id,
            )
        else:
            row = await _sqlite_fetchone(
                conn,
                f"SELECT role, status FROM main.{table} "  # nosec B608
                f"WHERE {scope_column} = ? AND user_id = ?",
                (scope_id, user_id),
            )
        if row is None:
            return None
        return {
            "role": _row_value(row, "role", 0),
            "status": _row_value(row, "status", 1),
        }

    async def _read_owner_user_ids(
        self,
        conn: Any,
        org_id: int,
    ) -> tuple[int, ...]:
        if self._backend is MembershipLockBackend.POSTGRESQL:
            rows = await conn.fetch(
                "SELECT user_id FROM public.org_members WHERE org_id = $1 "  # nosec B608
                "AND LOWER(role) = 'owner' "
                f"AND {_ACTIVE_MEMBERSHIP_SQL} ORDER BY user_id",
                org_id,
            )
        else:
            cursor = await conn.execute(
                "SELECT user_id FROM main.org_members WHERE org_id = ? "  # nosec B608
                "AND LOWER(role) = 'owner' "
                f"AND {_ACTIVE_MEMBERSHIP_SQL} ORDER BY user_id",
                (org_id,),
            )
            rows = await cursor.fetchall()
        return tuple(int(_row_value(row, "user_id", 0)) for row in rows)

    async def _read_owner_count(self, conn: Any, org_id: int) -> int:
        return len(await self._read_owner_user_ids(conn, org_id))

    async def _insert_membership(
        self,
        conn: Any,
        mutation: MembershipMutation,
    ) -> None:
        is_org = mutation.scope_type is MembershipScopeType.ORGANIZATION
        if self._backend is MembershipLockBackend.POSTGRESQL:
            if is_org:
                await _execute_membership_scope_sql(
                    conn,
                    "INSERT INTO public.org_members (org_id, user_id, role) "
                    "VALUES ($1, $2, $3)",
                    mutation.scope_id,
                    mutation.user_id,
                    mutation.role,
                    backend="postgres",
                )
            else:
                await _execute_membership_scope_sql(
                    conn,
                    "INSERT INTO public.team_members (team_id, user_id, role) "
                    "VALUES ($1, $2, $3)",
                    mutation.scope_id,
                    mutation.user_id,
                    mutation.role,
                    backend="postgres",
                )
        else:
            if is_org:
                await _execute_membership_scope_sql(
                    conn,
                    "INSERT INTO main.org_members (org_id, user_id, role) "
                    "VALUES (?, ?, ?)",
                    (mutation.scope_id, mutation.user_id, mutation.role),
                    backend="sqlite",
                )
            else:
                await _execute_membership_scope_sql(
                    conn,
                    "INSERT INTO main.team_members (team_id, user_id, role) "
                    "VALUES (?, ?, ?)",
                    (mutation.scope_id, mutation.user_id, mutation.role),
                    backend="sqlite",
                )

    async def _update_membership_role(
        self,
        conn: Any,
        mutation: MembershipMutation,
    ) -> None:
        is_org = mutation.scope_type is MembershipScopeType.ORGANIZATION
        if self._backend is MembershipLockBackend.POSTGRESQL:
            if is_org:
                await _execute_membership_scope_sql(
                    conn,
                    "UPDATE public.org_members SET role = $3 "
                    "WHERE org_id = $1 AND user_id = $2",
                    mutation.scope_id,
                    mutation.user_id,
                    mutation.role,
                    backend="postgres",
                )
            else:
                await _execute_membership_scope_sql(
                    conn,
                    "UPDATE public.team_members SET role = $3 "
                    "WHERE team_id = $1 AND user_id = $2",
                    mutation.scope_id,
                    mutation.user_id,
                    mutation.role,
                    backend="postgres",
                )
        else:
            if is_org:
                await _execute_membership_scope_sql(
                    conn,
                    "UPDATE main.org_members SET role = ? "
                    "WHERE org_id = ? AND user_id = ?",
                    (mutation.role, mutation.scope_id, mutation.user_id),
                    backend="sqlite",
                )
            else:
                await _execute_membership_scope_sql(
                    conn,
                    "UPDATE main.team_members SET role = ? "
                    "WHERE team_id = ? AND user_id = ?",
                    (mutation.role, mutation.scope_id, mutation.user_id),
                    backend="sqlite",
                )

    async def _update_organization_owner_pointer(
        self,
        conn: Any,
        *,
        organization_id: int,
        new_owner_user_id: int,
    ) -> None:
        if self._backend is MembershipLockBackend.POSTGRESQL:
            await conn.execute(
                "UPDATE public.organizations SET owner_user_id = $1, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                new_owner_user_id,
                organization_id,
            )
            return
        await conn.execute(
            "UPDATE main.organizations SET owner_user_id = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_owner_user_id, organization_id),
        )

    async def _delete_membership(
        self,
        conn: Any,
        mutation: MembershipMutation,
    ) -> None:
        is_org = mutation.scope_type is MembershipScopeType.ORGANIZATION
        if self._backend is MembershipLockBackend.POSTGRESQL:
            if is_org:
                await _execute_membership_scope_sql(
                    conn,
                    "DELETE FROM public.org_members "
                    "WHERE org_id = $1 AND user_id = $2",
                    mutation.scope_id,
                    mutation.user_id,
                    backend="postgres",
                )
            else:
                await _execute_membership_scope_sql(
                    conn,
                    "DELETE FROM public.team_members "
                    "WHERE team_id = $1 AND user_id = $2",
                    mutation.scope_id,
                    mutation.user_id,
                    backend="postgres",
                )
        else:
            if is_org:
                await _execute_membership_scope_sql(
                    conn,
                    "DELETE FROM main.org_members "
                    "WHERE org_id = ? AND user_id = ?",
                    (mutation.scope_id, mutation.user_id),
                    backend="sqlite",
                )
            else:
                await _execute_membership_scope_sql(
                    conn,
                    "DELETE FROM main.team_members "
                    "WHERE team_id = ? AND user_id = ?",
                    (mutation.scope_id, mutation.user_id),
                    backend="sqlite",
                )


async def _sqlite_fetchone(
    conn: Any,
    sql: str,
    parameters: tuple[Any, ...],
) -> Any:
    cursor = await conn.execute(sql, parameters)
    return await cursor.fetchone()


async def _sqlite_fetchall(
    conn: Any,
    sql: str,
    parameters: tuple[Any, ...],
) -> Any:
    cursor = await conn.execute(sql, parameters)
    return await cursor.fetchall()


def _row_value(row: Any, name: str, index: int) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return row[index]


def _is_active(value: Any) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int:
        return value == 1
    if type(value) is str:
        return value.strip().lower() in {"1", "true", "active"}
    return False


def _is_active_membership_admin(membership: dict[str, Any] | None) -> bool:
    return _is_active_membership(membership) and str(
        membership.get("role") or ""
    ).strip().lower() in {"owner", "admin"}


def _is_active_team_membership_admin(
    membership: dict[str, Any] | None,
) -> bool:
    return _is_active_membership(membership) and str(
        membership.get("role") or ""
    ).strip().lower() in {"owner", "admin", "lead"}


def _is_active_membership(membership: dict[str, Any] | None) -> bool:
    return membership is not None and (
        str(membership.get("status") or "").strip().lower() == "active"
    )
