"""Runtime firewall for profile-visible writes on managed AuthNZ connections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from inspect import getattr_static
from threading import Lock
from typing import Any

import sqlglot
from asyncpg.pool import PoolConnectionProxy
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from tldw_Server_API.app.core.AuthNZ.profile_user_fields import (
    PROFILE_VISIBLE_USER_FIELDS,
    RAW_SAFE_USER_UPDATE_FIELDS,
)

_MAX_SQL_BYTES = 16 * 1024
_GUARD_IDENTITY_ATTRIBUTE = "_authnz_profile_user_guard_identity"
_BACKEND_ATTRIBUTE = "_authnz_profile_user_backend"
_SUPPORTED_BACKENDS = frozenset({"sqlite", "postgres"})
_MEMBERSHIP_TABLES = frozenset({"org_members", "team_members"})
_PARENT_SCOPE_TABLES = frozenset({"organizations", "teams"})
_POSTGRES_MEMBERSHIP_TIMESTAMP_REPAIR_SQL = {
    "org_members": (
        "UPDATE public.org_members SET added_at = "
        "COALESCE(added_at, CURRENT_TIMESTAMP) WHERE added_at IS NULL"
    ),
    "team_members": (
        "UPDATE public.team_members SET added_at = "
        "COALESCE(added_at, CURRENT_TIMESTAMP) WHERE added_at IS NULL"
    ),
}
_POSTGRES_ASYNCPG_SAVEPOINT_COMMAND_RE = re.compile(
    r"(?:SAVEPOINT|RELEASE SAVEPOINT|ROLLBACK TO) "
    r"__asyncpg_savepoint_[0-9a-f]+__;",
    re.ASCII,
)
_PROFILE_USER_DOMAIN = "profile_users"
_MEMBERSHIP_SCOPE_DOMAIN = "membership_scope"
_REJECTED_CREATE_KINDS = frozenset(
    {"FUNCTION", "PROCEDURE", "RULE", "TRIGGER", "VIEW"}
)
_REQUIRED_USERS_BOOTSTRAP_COLUMNS = frozenset(
    {
        "id",
        "uuid",
        "username",
        "email",
        "password_hash",
        "metadata",
        "role",
        "is_active",
        "is_verified",
        "is_superuser",
        "email_verified",
        "two_factor_enabled",
        "failed_login_attempts",
        "locked_until",
        "storage_quota_mb",
        "storage_used_mb",
        "created_at",
        "updated_at",
        "profile_version",
        "last_login",
        "email_verified_at",
        "two_factor_secret",
        "totp_secret",
        "backup_codes",
        "created_by",
        "password_changed_at",
    }
)
_CANONICAL_USERS_BOOTSTRAP_CREATE_ARGUMENTS = frozenset(
    {
        "this",
        "kind",
        "replace",
        "refresh",
        "unique",
        "expression",
        "exists",
        "properties",
        "indexes",
        "no_schema_binding",
        "begin",
        "clone",
        "concurrently",
        "clustered",
    }
)
_POSTGRES_USERS_BOOTSTRAP_TYPE_SIGNATURES = frozenset(
    {
        frozenset(
            {
                "id": "SERIAL",
                "uuid": "UUID",
                "username": username_type,
                "email": "VARCHAR(255)",
                "password_hash": "TEXT",
                "metadata": "JSONB",
                "role": "VARCHAR(50)",
                "is_active": "BOOLEAN",
                "is_verified": "BOOLEAN",
                "is_superuser": "BOOLEAN",
                "email_verified": "BOOLEAN",
                "two_factor_enabled": "BOOLEAN",
                "failed_login_attempts": "INT",
                "locked_until": "TIMESTAMPTZ",
                "storage_quota_mb": "INT",
                "storage_used_mb": storage_used_type,
                "created_at": "TIMESTAMPTZ",
                "updated_at": "TIMESTAMPTZ",
                "profile_version": "TIMESTAMPTZ",
                "last_login": "TIMESTAMPTZ",
                "email_verified_at": "TIMESTAMPTZ",
                "two_factor_secret": "TEXT",
                "totp_secret": "TEXT",
                "backup_codes": "TEXT",
                "created_by": "INT",
                "password_changed_at": "TIMESTAMPTZ",
            }.items()
        )
        for username_type, storage_used_type in (
            ("VARCHAR(255)", "DOUBLE PRECISION"),
            ("VARCHAR(50)", "INT"),
        )
    }
)
_SQLITE_USERS_BOOTSTRAP_TYPE_SIGNATURE = frozenset(
    {
        "id": "INTEGER",
        "uuid": "TEXT",
        "username": "TEXT",
        "email": "TEXT",
        "password_hash": "TEXT",
        "metadata": "TEXT",
        "role": "TEXT",
        "is_active": "INTEGER",
        "is_verified": "INTEGER",
        "is_superuser": "INTEGER",
        "email_verified": "INTEGER",
        "two_factor_enabled": "INTEGER",
        "failed_login_attempts": "INTEGER",
        "locked_until": "TIMESTAMP",
        "storage_quota_mb": "INTEGER",
        "storage_used_mb": "INTEGER",
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
        "profile_version": "TEXT",
        "last_login": "TIMESTAMP",
        "email_verified_at": "TIMESTAMP",
        "two_factor_secret": "TEXT",
        "totp_secret": "TEXT",
        "backup_codes": "TEXT",
        "created_by": "INTEGER",
        "password_changed_at": "TIMESTAMP",
    }.items()
)
_SQLITE_SCHEMA_CATALOGS = frozenset(
    {
        "sqlite_master",
        "sqlite_schema",
        "sqlite_temp_master",
        "sqlite_temp_schema",
    }
)
_SUPPORTED_STATEMENT_ROOTS = (
    exp.Query,
    exp.DML,
    exp.DDL,
    exp.Alter,
    exp.Create,
    exp.Drop,
    exp.TruncateTable,
    exp.Pragma,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Set,
    exp.Use,
    exp.Grant,
    exp.Revoke,
    exp.Analyze,
)


class ProfileUserWriteRejected(RuntimeError):
    """A managed AuthNZ boundary rejected an unversioned users write."""

    def __init__(self) -> None:
        super().__init__("Profile-visible AuthNZ users write rejected")


@dataclass(frozen=True, slots=True)
class _SqlClassification:
    protected: bool
    operation: str
    columns: tuple[str, ...]
    domain: str | None = None


_CAPABILITY_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True, eq=False, init=False)
class _ProfileUserSql:
    text: str
    backend: str
    operation: str
    columns: tuple[str, ...]
    execution_mode: str
    _nonce: object

    def __new__(cls, construction_token: object = None) -> _ProfileUserSql:
        if cls is not _ProfileUserSql or construction_token is not _CAPABILITY_CONSTRUCTION_TOKEN:
            raise TypeError("Profile user SQL capabilities are gateway-owned")
        return object.__new__(cls)

    def __init__(self, construction_token: object = None) -> None:
        del construction_token

    def __copy__(self) -> _ProfileUserSql:
        return _construct_profile_user_sql(
            text=self.text,
            backend=self.backend,
            operation=self.operation,
            columns=self.columns,
            execution_mode=self.execution_mode,
            nonce=self._nonce,
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class _MembershipScopeSql:
    text: str
    backend: str
    operation: str
    columns: tuple[str, ...]
    execution_mode: str
    _nonce: object

    def __new__(cls, construction_token: object = None) -> _MembershipScopeSql:
        if (
            cls is not _MembershipScopeSql
            or construction_token is not _CAPABILITY_CONSTRUCTION_TOKEN
        ):
            raise TypeError("Membership scope SQL capabilities are writer-owned")
        return object.__new__(cls)

    def __init__(self, construction_token: object = None) -> None:
        del construction_token


@dataclass(frozen=True, slots=True)
class _CapabilityRecord:
    capability: _ProfileUserSql | _MembershipScopeSql
    connection_identity: object
    text: str
    backend: str
    operation: str
    columns: tuple[str, ...]
    execution_mode: str
    nonce: object


_capability_lock = Lock()
_active_capabilities: dict[int, _CapabilityRecord] = {}


def _guard_sql(
    query: Any,
    *,
    backend: str,
    connection_identity: object,
    operation: str,
) -> str:
    """Return concrete unprotected SQL or reject before managed DB I/O."""
    if type(query) is _ProfileUserSql:
        return _consume_profile_user_sql(
            query,
            backend=backend,
            connection_identity=connection_identity,
            execution_mode=operation,
        )
    if type(query) is _MembershipScopeSql:
        return _consume_membership_scope_sql(
            query,
            backend=backend,
            connection_identity=connection_identity,
            execution_mode=operation,
        )
    if (
        type(query) is not str
        or type(backend) is not str
        or backend not in _SUPPORTED_BACKENDS
        or connection_identity is None
        or type(operation) is not str
        or not operation
    ):
        raise ProfileUserWriteRejected()
    if not query or len(query) > _MAX_SQL_BYTES:
        raise ProfileUserWriteRejected()
    try:
        encoded_size = len(query.encode("utf-8"))
    except (UnicodeError, AttributeError):
        raise ProfileUserWriteRejected() from None
    if not query.strip() or encoded_size > _MAX_SQL_BYTES:
        raise ProfileUserWriteRejected()
    classification = _classify_sql(query, backend)
    if classification.protected:
        raise ProfileUserWriteRejected()
    return query


def _profile_user_connection_identity(connection: object) -> object:
    """Resolve the stable identity shared by managed connection adapters."""
    connection = _unwrap_trusted_pool_proxy(connection)
    try:
        identity = getattr(connection, _GUARD_IDENTITY_ATTRIBUTE)
    except AttributeError:
        identity = connection
    except BaseException:  # noqa: BLE001 - hostile adapter access is sanitized
        raise ProfileUserWriteRejected() from None
    if identity is None:
        raise ProfileUserWriteRejected()
    return identity


def _profile_user_backend(connection: object) -> str | None:
    """Return a validated managed backend marker, or None for legacy adapters."""
    connection = _unwrap_trusted_pool_proxy(connection)
    try:
        getattr_static(connection, _BACKEND_ATTRIBUTE)
    except AttributeError:
        return None
    except BaseException:  # noqa: BLE001 - hostile adapter access is sanitized
        raise ProfileUserWriteRejected() from None
    try:
        backend = getattr(connection, _BACKEND_ATTRIBUTE)
    except BaseException:  # noqa: BLE001 - hostile adapter access is sanitized
        raise ProfileUserWriteRejected() from None
    if type(backend) is not str or backend not in _SUPPORTED_BACKENDS:
        raise ProfileUserWriteRejected()
    return backend


def _unwrap_trusted_pool_proxy(connection: object) -> object:
    """Return the guarded connection behind asyncpg's exact pool proxy type."""
    if type(connection) is not PoolConnectionProxy:
        return connection
    try:
        wrapped = connection._con
    except BaseException:  # noqa: BLE001 - proxy state is fail-closed
        raise ProfileUserWriteRejected() from None
    if wrapped is None:
        raise ProfileUserWriteRejected()
    return wrapped


def _canonical_users_table(backend: str) -> str:
    """Return the serving schema-qualified users relation for one backend."""
    if backend == "sqlite":
        return "main.users"
    if backend == "postgres":
        return "public.users"
    raise ProfileUserWriteRejected()


def _mint_profile_user_sql(
    query: str,
    *,
    backend: str,
    connection_identity: object,
    operation: str,
    columns: tuple[str, ...],
    execution_mode: str = "execute",
) -> _ProfileUserSql:
    """Mint a single-use authorization for one exact managed connection."""
    if (
        type(query) is not str
        or type(backend) is not str
        or backend not in _SUPPORTED_BACKENDS
        or connection_identity is None
        or type(operation) is not str
        or type(execution_mode) is not str
        or not execution_mode
        or type(columns) is not tuple
        or (not columns and operation not in {"alter", "create"})
        or any(type(column) is not str or not column for column in columns)
    ):
        raise ProfileUserWriteRejected()
    if not query or len(query) > _MAX_SQL_BYTES:
        raise ProfileUserWriteRejected()
    try:
        encoded_size = len(query.encode("utf-8"))
    except (UnicodeError, AttributeError):
        raise ProfileUserWriteRejected() from None
    if not query.strip() or encoded_size > _MAX_SQL_BYTES:
        raise ProfileUserWriteRejected()

    lowered_columns = tuple(column.lower() for column in columns)
    normalized_columns = (
        lowered_columns
        if operation == "insert"
        else tuple(sorted(set(lowered_columns)))
    )
    if columns != normalized_columns:
        raise ProfileUserWriteRejected()
    classification = _classify_sql(query, backend)
    if operation == "create" and not _is_canonical_users_bootstrap_sql(
        query,
        backend=backend,
    ):
        raise ProfileUserWriteRejected()
    if (
        not classification.protected
        or classification.domain != _PROFILE_USER_DOMAIN
        or classification.operation != operation
        or classification.columns != normalized_columns
    ):
        raise ProfileUserWriteRejected()

    nonce = object()
    capability = _construct_profile_user_sql(
        text=query,
        backend=backend,
        operation=operation,
        columns=normalized_columns,
        execution_mode=execution_mode,
        nonce=nonce,
    )
    record = _CapabilityRecord(
        capability=capability,
        connection_identity=connection_identity,
        text=query,
        backend=backend,
        operation=operation,
        columns=normalized_columns,
        execution_mode=execution_mode,
        nonce=nonce,
    )
    with _capability_lock:
        _active_capabilities[id(capability)] = record
    return capability


def _mint_membership_scope_sql(
    query: str,
    *,
    backend: str,
    connection_identity: object,
    execution_mode: str,
) -> _MembershipScopeSql:
    """Mint one exact membership/scope-deletion statement execution."""
    if (
        type(query) is not str
        or type(backend) is not str
        or backend not in _SUPPORTED_BACKENDS
        or connection_identity is None
        or type(execution_mode) is not str
        or not execution_mode
    ):
        raise ProfileUserWriteRejected()
    if not query or len(query) > _MAX_SQL_BYTES:
        raise ProfileUserWriteRejected()
    try:
        encoded_size = len(query.encode("utf-8"))
    except (UnicodeError, AttributeError):
        raise ProfileUserWriteRejected() from None
    if not query.strip() or encoded_size > _MAX_SQL_BYTES:
        raise ProfileUserWriteRejected()

    classification = _classify_sql(query, backend)
    if (
        not classification.protected
        or classification.domain != _MEMBERSHIP_SCOPE_DOMAIN
    ):
        raise ProfileUserWriteRejected()
    nonce = object()
    capability = _construct_membership_scope_sql(
        text=query,
        backend=backend,
        operation=classification.operation,
        columns=classification.columns,
        execution_mode=execution_mode,
        nonce=nonce,
    )
    record = _CapabilityRecord(
        capability=capability,
        connection_identity=connection_identity,
        text=query,
        backend=backend,
        operation=classification.operation,
        columns=classification.columns,
        execution_mode=execution_mode,
        nonce=nonce,
    )
    with _capability_lock:
        _active_capabilities[id(capability)] = record
    return capability


def _construct_profile_user_sql(
    *,
    text: str,
    backend: str,
    operation: str,
    columns: tuple[str, ...],
    execution_mode: str,
    nonce: object,
) -> _ProfileUserSql:
    capability = _ProfileUserSql(_CAPABILITY_CONSTRUCTION_TOKEN)
    object.__setattr__(capability, "text", text)
    object.__setattr__(capability, "backend", backend)
    object.__setattr__(capability, "operation", operation)
    object.__setattr__(capability, "columns", columns)
    object.__setattr__(capability, "execution_mode", execution_mode)
    object.__setattr__(capability, "_nonce", nonce)
    return capability


def _construct_membership_scope_sql(
    *,
    text: str,
    backend: str,
    operation: str,
    columns: tuple[str, ...],
    execution_mode: str,
    nonce: object,
) -> _MembershipScopeSql:
    capability = _MembershipScopeSql(_CAPABILITY_CONSTRUCTION_TOKEN)
    object.__setattr__(capability, "text", text)
    object.__setattr__(capability, "backend", backend)
    object.__setattr__(capability, "operation", operation)
    object.__setattr__(capability, "columns", columns)
    object.__setattr__(capability, "execution_mode", execution_mode)
    object.__setattr__(capability, "_nonce", nonce)
    return capability


def _consume_profile_user_sql(
    capability: _ProfileUserSql,
    *,
    backend: str,
    connection_identity: object,
    execution_mode: str,
) -> str:
    with _capability_lock:
        record = _active_capabilities.pop(id(capability), None)
    if (
        record is None
        or record.capability is not capability
        or capability.text != record.text
        or capability.backend != record.backend
        or capability.operation != record.operation
        or capability.columns != record.columns
        or capability.execution_mode != record.execution_mode
        or capability._nonce is not record.nonce
        or backend != record.backend
        or connection_identity is not record.connection_identity
        or execution_mode != record.execution_mode
    ):
        raise ProfileUserWriteRejected()
    return record.text


def _consume_membership_scope_sql(
    capability: _MembershipScopeSql,
    *,
    backend: str,
    connection_identity: object,
    execution_mode: str,
) -> str:
    with _capability_lock:
        record = _active_capabilities.pop(id(capability), None)
    if (
        record is None
        or record.capability is not capability
        or capability.text != record.text
        or capability.backend != record.backend
        or capability.operation != record.operation
        or capability.columns != record.columns
        or capability.execution_mode != record.execution_mode
        or capability._nonce is not record.nonce
        or backend != record.backend
        or connection_identity is not record.connection_identity
        or execution_mode != record.execution_mode
    ):
        raise ProfileUserWriteRejected()
    return record.text


def _revoke_profile_user_sql(capability: object) -> None:
    if type(capability) is not _ProfileUserSql:
        return
    with _capability_lock:
        record = _active_capabilities.get(id(capability))
        if record is not None and record.capability is capability:
            del _active_capabilities[id(capability)]


def _revoke_membership_scope_sql(capability: object) -> None:
    if type(capability) is not _MembershipScopeSql:
        return
    with _capability_lock:
        record = _active_capabilities.get(id(capability))
        if record is not None and record.capability is capability:
            del _active_capabilities[id(capability)]


def _active_capability_count() -> int:
    with _capability_lock:
        return len(_active_capabilities)


@lru_cache(maxsize=4096)
def _classify_sql(query: str, backend: str) -> _SqlClassification:
    normalized_command = " ".join(query.strip().rstrip(";").split()).casefold()
    if backend == "postgres" and query.isascii() and normalized_command in {
        "create extension if not exists pgcrypto",
        'create extension if not exists "uuid-ossp"',
    }:
        return _SqlClassification(False, "trusted_extension_bootstrap", ())
    if (
        backend == "postgres"
        and _POSTGRES_ASYNCPG_SAVEPOINT_COMMAND_RE.fullmatch(query)
    ):
        return _SqlClassification(False, "asyncpg_savepoint", ())

    # sqlglot treats PostgreSQL VALIDATE CONSTRAINT as an opaque Command.
    if (
        backend == "postgres"
        and query.isascii()
        and normalized_command
        == "alter table share_tokens validate constraint "
        "ck_share_tokens_resource_type"
    ):
        return _SqlClassification(False, "trusted_constraint_validation", ())
    try:
        statements = tuple(
            statement
            for statement in sqlglot.parse(
                query,
                read=backend,
                error_level="RAISE",
                error_message_context=0,
            )
            if statement is not None
        )
    except (ParseError, TokenError, ValueError, TypeError, RecursionError):
        raise ProfileUserWriteRejected() from None
    if len(statements) != 1:
        raise ProfileUserWriteRejected()

    statement = statements[0]
    if isinstance(statement, exp.Command) or not isinstance(
        statement,
        _SUPPORTED_STATEMENT_ROOTS,
    ):
        raise ProfileUserWriteRejected()
    if (
        isinstance(statement, exp.Create)
        and statement.args.get("kind") in _REJECTED_CREATE_KINDS
    ):
        raise ProfileUserWriteRejected()
    if isinstance(statement, exp.Create) and _create_derives_from_users(statement):
        return _SqlClassification(True, "create", (), _PROFILE_USER_DOMAIN)
    if isinstance(statement, (exp.Drop, exp.TruncateTable)):
        return _SqlClassification(True, statement.key.lower(), (), "global_ddl")
    if isinstance(statement, exp.Create) and _targets_users(statement.this):
        return _SqlClassification(True, "create", (), _PROFILE_USER_DOMAIN)
    if backend == "sqlite" and isinstance(statement, exp.Pragma):
        if _pragma_name(statement) == "writable_schema":
            raise ProfileUserWriteRejected()

    unprotected_classification: _SqlClassification | None = None
    protected_classification: _SqlClassification | None = None
    for mutation in statement.walk():
        classification = _classify_mutation(mutation, backend=backend)
        if classification is None:
            continue
        if classification.protected:
            if protected_classification is not None:
                return _SqlClassification(True, "compound", (), "compound")
            protected_classification = classification
        else:
            unprotected_classification = classification
    return protected_classification or unprotected_classification or _SqlClassification(
        False,
        "read_or_unprotected_write",
        (),
    )


def _classify_mutation(
    node: exp.Expression,
    *,
    backend: str,
) -> _SqlClassification | None:
    target = node.args.get("this")
    if backend == "sqlite" and isinstance(
        node,
        (exp.Update, exp.Insert, exp.Delete, exp.Merge),
    ):
        if _target_table_name(target) in _SQLITE_SCHEMA_CATALOGS:
            return _SqlClassification(True, "schema_catalog_write", (), "schema_catalog")

    target_table = _target_table_name(target)
    if isinstance(node, (exp.Update, exp.Insert, exp.Delete, exp.Merge)):
        if target_table in _MEMBERSHIP_TABLES:
            if isinstance(node, exp.Update):
                columns = tuple(sorted(_update_columns(node) or {"<ambiguous>"}))
            elif isinstance(node, exp.Insert):
                columns = _insert_columns(node) or ("<ambiguous>",)
            else:
                columns = ()
            return _SqlClassification(
                True,
                node.key.lower(),
                columns,
                _MEMBERSHIP_SCOPE_DOMAIN,
            )
        if isinstance(node, exp.Delete) and target_table in _PARENT_SCOPE_TABLES:
            return _SqlClassification(
                True,
                "delete",
                (),
                _MEMBERSHIP_SCOPE_DOMAIN,
            )

    if isinstance(node, exp.Alter):
        actions = tuple(node.args.get("actions") or ())
        source_is_users = _targets_users(node.this)
        renames_to_users = any(
            isinstance(action, exp.AlterRename) and _targets_users(action.this)
            for action in actions
        )
        if renames_to_users and not source_is_users:
            return _SqlClassification(True, "alter", (), _PROFILE_USER_DOMAIN)
        if not source_is_users:
            return None
        protected = not actions or not all(
            _is_safe_users_alter_action(action) for action in actions
        )
        return _SqlClassification(
            protected,
            "alter",
            (),
            _PROFILE_USER_DOMAIN,
        )

    if isinstance(node, exp.Update):
        if not _targets_users(node.this):
            return None
        columns = _update_columns(node)
        protected = (
            columns is None
            or not columns <= RAW_SAFE_USER_UPDATE_FIELDS
            or bool(PROFILE_VISIBLE_USER_FIELDS.intersection(columns))
        )
        return _SqlClassification(
            protected,
            "update",
            tuple(sorted(columns or {"<ambiguous>"})),
            _PROFILE_USER_DOMAIN,
        )

    if isinstance(node, exp.Insert):
        if _targets_users(node.this):
            columns = _insert_columns(node)
            return _SqlClassification(
                True,
                "insert",
                columns or ("<ambiguous>",),
                _PROFILE_USER_DOMAIN,
            )
        return None

    if isinstance(node, exp.Merge):
        if _targets_users(node.this):
            return _SqlClassification(
                True,
                node.key.lower(),
                (),
                _PROFILE_USER_DOMAIN,
            )
        return None

    if isinstance(node, exp.Copy) and node.args.get("kind"):
        copy_target = _target_table_name(node.this)
        if copy_target == "users":
            return _SqlClassification(True, "copy", (), _PROFILE_USER_DOMAIN)
        if copy_target in _MEMBERSHIP_TABLES:
            return _SqlClassification(
                True,
                "copy",
                (),
                _MEMBERSHIP_SCOPE_DOMAIN,
            )
    if isinstance(node, exp.Delete) and _targets_users(node.this):
        return _SqlClassification(True, "delete", (), _PROFILE_USER_DOMAIN)
    if isinstance(node, exp.TruncateTable) and any(
        _targets_users(target) for target in node.expressions
    ):
        return _SqlClassification(True, "truncate", (), _PROFILE_USER_DOMAIN)
    if (
        isinstance(node, exp.Drop)
        and str(node.args.get("kind") or "").upper() == "TABLE"
        and _targets_users(node.this)
    ):
        return _SqlClassification(True, "drop", (), _PROFILE_USER_DOMAIN)
    return None


def _is_safe_users_alter_action(action: exp.Expression) -> bool:
    if getattr(action, "name", "").lower() == "profile_version":
        return False
    if isinstance(action, exp.AddConstraint):
        return not any(
            identifier.name.lower() == "profile_version"
            for identifier in action.find_all(exp.Identifier)
            if identifier.name
        )
    if isinstance(action, exp.ColumnDef):
        return True
    if not isinstance(action, exp.AlterColumn):
        return False
    if (
        action.args.get("drop")
        or action.args.get("dtype") is not None
        or action.args.get("collate")
        or action.args.get("using")
    ):
        return False
    return (
        action.args.get("allow_null") is False
        or action.args.get("default") is not None
    )


def _is_canonical_users_bootstrap(
    statement: exp.Create,
    *,
    backend: str,
) -> bool:
    if (
        set(statement.args) != _CANONICAL_USERS_BOOTSTRAP_CREATE_ARGUMENTS
        or str(statement.args.get("kind") or "").upper() != "TABLE"
        or statement.args.get("exists") is not True
        or statement.args.get("replace") is not False
        or statement.args.get("refresh") is not False
        or statement.args.get("unique") is not False
        or statement.args.get("expression") is not None
        or statement.args.get("properties") is not None
        or statement.args.get("indexes") != []
        or statement.args.get("no_schema_binding") is not None
        or statement.args.get("begin") is not None
        or statement.args.get("clone") is not None
        or statement.args.get("concurrently") is not False
        or statement.args.get("clustered") is not None
    ):
        return False
    target = statement.this
    if not isinstance(target, exp.Schema) or not target.expressions:
        return False
    columns = tuple(target.expressions)
    if not all(isinstance(column, exp.ColumnDef) for column in columns):
        return False
    if len(columns) != len(_REQUIRED_USERS_BOOTSTRAP_COLUMNS):
        return False
    column_names = {
        column.name.lower()
        for column in columns
        if isinstance(column, exp.ColumnDef) and column.name
    }
    if column_names != _REQUIRED_USERS_BOOTSTRAP_COLUMNS:
        return False
    definitions = {
        column.name.lower(): column
        for column in columns
        if isinstance(column, exp.ColumnDef) and column.name
    }
    if not _bootstrap_column_types_are_canonical(definitions, backend=backend):
        return False
    if not _bootstrap_column_constraints_are_canonical(
        definitions,
        backend=backend,
    ):
        return False
    expected_defaults = {
        "uuid": (
            {"GEN_RANDOM_UUID()", "UUID_GENERATE_V4()"}
            if backend == "postgres"
            else {"LOWER(HEX(RANDOMBLOB(16)))"}
        ),
        "role": {"'USER'"},
        "is_active": {"TRUE", "1"},
        "is_verified": {"FALSE", "0"},
        "is_superuser": {"FALSE", "0"},
        "email_verified": {"FALSE", "0"},
        "two_factor_enabled": {"FALSE", "0"},
        "failed_login_attempts": {"0"},
        "storage_quota_mb": {"5120"},
        "storage_used_mb": {"0", "0.0"},
        "created_at": {"CURRENT_TIMESTAMP"},
        "updated_at": {"CURRENT_TIMESTAMP"},
        "profile_version": (
            {"CURRENT_TIMESTAMP"}
            if backend == "postgres"
            else {"STRFTIME('%Y-%M-%DT%H:%M:%F000Z', 'NOW')"}
        ),
    }
    if any(
        _bootstrap_column_default(definitions[name], backend=backend)
        not in allowed_defaults
        for name, allowed_defaults in expected_defaults.items()
    ):
        return False
    metadata_default = _bootstrap_column_default(
        definitions["metadata"],
        backend=backend,
    )
    allowed_metadata_defaults = (
        {None, "CAST('{}' AS JSON)", "CAST('{}' AS JSONB)"}
        if backend == "postgres"
        else {None, "'{}'"}
    )
    if metadata_default not in allowed_metadata_defaults:
        return False
    target = target.this
    if not isinstance(target, exp.Table):
        return False
    expected_schema = "main" if backend == "sqlite" else "public"
    return bool(
        target.name
        and target.name.lower() == "users"
        and target.db
        and target.db.lower() == expected_schema
        and not target.catalog
    )


def _is_canonical_users_bootstrap_sql(query: str, *, backend: str) -> bool:
    try:
        statements = tuple(
            statement
            for statement in sqlglot.parse(
                query,
                read=backend,
                error_level="RAISE",
                error_message_context=0,
            )
            if statement is not None
        )
    except (ParseError, TokenError, ValueError, TypeError, RecursionError):
        return False
    return bool(
        len(statements) == 1
        and isinstance(statements[0], exp.Create)
        and _is_canonical_users_bootstrap(statements[0], backend=backend)
    )


def _bootstrap_column_constraints_are_canonical(
    definitions: dict[str, exp.ColumnDef],
    *,
    backend: str,
) -> bool:
    not_null_default = (
        exp.NotNullColumnConstraint,
        exp.DefaultColumnConstraint,
    )
    expected: dict[str, tuple[type[exp.Expression], ...]] = {
        "id": (exp.PrimaryKeyColumnConstraint,),
        "uuid": (
            exp.UniqueColumnConstraint,
            exp.NotNullColumnConstraint,
            exp.DefaultColumnConstraint,
        ),
        "username": (
            exp.UniqueColumnConstraint,
            exp.NotNullColumnConstraint,
        ),
        "email": (
            exp.UniqueColumnConstraint,
            exp.NotNullColumnConstraint,
        ),
        "password_hash": (exp.NotNullColumnConstraint,),
        "metadata": (),
        "role": not_null_default,
        "is_active": not_null_default,
        "is_verified": not_null_default,
        "is_superuser": not_null_default,
        "email_verified": not_null_default,
        "two_factor_enabled": not_null_default,
        "failed_login_attempts": not_null_default,
        "locked_until": (),
        "storage_quota_mb": not_null_default,
        "storage_used_mb": not_null_default,
        "created_at": not_null_default,
        "updated_at": not_null_default,
        "profile_version": not_null_default,
        "last_login": (),
        "email_verified_at": (),
        "two_factor_secret": (),
        "totp_secret": (),
        "backup_codes": (),
        "created_by": (exp.Reference,),
        "password_changed_at": (),
    }
    if backend == "sqlite":
        expected["id"] += (exp.AutoIncrementColumnConstraint,)

    for column_name, definition in definitions.items():
        constraints = tuple(definition.args.get("constraints") or ())
        kinds = tuple(constraint.args.get("kind") for constraint in constraints)
        if any(not isinstance(kind, exp.Expression) for kind in kinds):
            return False

        expected_kinds = expected[column_name]
        check_kinds = tuple(
            kind for kind in kinds if isinstance(kind, exp.CheckColumnConstraint)
        )
        ordinary_kinds = tuple(
            kind for kind in kinds if not isinstance(kind, exp.CheckColumnConstraint)
        )
        if not _constraint_type_multiset_matches(ordinary_kinds, expected_kinds):
            if column_name == "metadata" and _constraint_type_multiset_matches(
                ordinary_kinds,
                (exp.DefaultColumnConstraint,),
            ):
                pass
            else:
                return False
        if check_kinds:
            if (
                backend != "sqlite"
                or column_name not in {"username", "email"}
                or len(check_kinds) != 1
                or _normalized_constraint_sql(check_kinds[0], backend=backend)
                != f"CHECK (LENGTH({column_name.upper()}) <= 255)"
            ):
                return False
        if len(kinds) != len(ordinary_kinds) + len(check_kinds):
            return False
        if any(
            not _bootstrap_simple_constraint_is_canonical(kind, backend=backend)
            for kind in ordinary_kinds
            if not isinstance(
                kind,
                (exp.DefaultColumnConstraint, exp.Reference),
            )
        ):
            return False

    return _bootstrap_reference_is_canonical(
        definitions["created_by"],
        backend=backend,
    )


def _constraint_type_multiset_matches(
    actual: tuple[exp.Expression, ...],
    expected: tuple[type[exp.Expression], ...],
) -> bool:
    return len(actual) == len(expected) and all(
        sum(isinstance(kind, expected_type) for kind in actual)
        == expected.count(expected_type)
        for expected_type in set(expected)
    )


def _normalized_constraint_sql(kind: exp.Expression, *, backend: str) -> str:
    return " ".join(kind.sql(dialect=backend).strip().upper().split())


def _bootstrap_simple_constraint_is_canonical(
    kind: exp.Expression,
    *,
    backend: str,
) -> bool:
    expected_sql: dict[type[exp.Expression], str] = {
        exp.PrimaryKeyColumnConstraint: "PRIMARY KEY",
        exp.AutoIncrementColumnConstraint: "AUTOINCREMENT",
        exp.UniqueColumnConstraint: "UNIQUE",
        exp.NotNullColumnConstraint: "NOT NULL",
    }
    return _normalized_constraint_sql(kind, backend=backend) == expected_sql.get(
        type(kind),
    )


def _bootstrap_reference_is_canonical(
    definition: exp.ColumnDef,
    *,
    backend: str,
) -> bool:
    references = tuple(
        constraint.args.get("kind")
        for constraint in definition.args.get("constraints") or ()
        if isinstance(constraint.args.get("kind"), exp.Reference)
    )
    if len(references) != 1:
        return False
    reference = references[0]
    target = reference.args.get("this")
    if not isinstance(target, exp.Schema) or len(target.expressions) != 1:
        return False
    table = target.this
    referenced_column = target.expressions[0]
    if (
        not isinstance(table, exp.Table)
        or table.name.lower() != "users"
        or not isinstance(referenced_column, exp.Identifier)
        or referenced_column.name.lower() != "id"
    ):
        return False
    expected_schema = "public" if backend == "postgres" else None
    actual_schema = table.db.lower() if table.db else None
    if actual_schema != expected_schema:
        return False
    options = tuple(
        " ".join(str(option).strip().upper().split())
        for option in reference.args.get("options") or ()
    )
    return options == ("ON DELETE SET NULL",)


def _bootstrap_column_default(
    definition: exp.ColumnDef,
    *,
    backend: str,
) -> str | None:
    for constraint in definition.args.get("constraints") or ():
        kind = constraint.args.get("kind")
        if isinstance(kind, exp.DefaultColumnConstraint):
            expression = kind.args.get("this")
            if not isinstance(expression, exp.Expression):
                return None
            normalized = expression.sql(dialect=backend).strip().upper()
            while normalized.startswith("(") and normalized.endswith(")"):
                normalized = normalized[1:-1].strip()
            return normalized
    return None


def _bootstrap_column_types_are_canonical(
    definitions: dict[str, exp.ColumnDef],
    *,
    backend: str,
) -> bool:
    types = {
        name: definition.args["kind"].sql(dialect=backend).strip().upper()
        for name, definition in definitions.items()
    }
    if backend == "postgres":
        return (
            frozenset(types.items())
            in _POSTGRES_USERS_BOOTSTRAP_TYPE_SIGNATURES
        )
    return frozenset(types.items()) == _SQLITE_USERS_BOOTSTRAP_TYPE_SIGNATURE


def _create_derives_from_users(statement: exp.Create) -> bool:
    properties = statement.args.get("properties")
    if not isinstance(properties, exp.Properties):
        return False
    for property_expression in properties.expressions:
        if not isinstance(
            property_expression,
            (exp.InheritsProperty, exp.PartitionedOfProperty),
        ):
            continue
        if any(_targets_users(table) for table in property_expression.find_all(exp.Table)):
            return True
    return False


async def _execute_profile_users_bootstrap(
    connection: Any,
    statement: str,
    *,
    backend: str,
) -> Any:
    """Execute one canonical users bootstrap through a managed boundary."""
    managed_backend = _profile_user_backend(connection)
    if managed_backend is None:
        if not _is_canonical_users_bootstrap_sql(statement, backend=backend):
            raise ProfileUserWriteRejected()
        return await connection.execute(statement)
    if managed_backend != backend:
        raise ProfileUserWriteRejected()
    capability = _mint_profile_user_sql(
        statement,
        backend=backend,
        connection_identity=_profile_user_connection_identity(connection),
        operation="create",
        columns=(),
    )
    try:
        return await connection.execute(capability)
    finally:
        _revoke_profile_user_sql(capability)


async def _execute_membership_scope_sql(
    connection: Any,
    query: str,
    *parameters: Any,
    backend: str,
) -> Any:
    """Execute one protected writer-owned statement on a managed connection."""
    managed_backend = _profile_user_backend(connection)
    if managed_backend is None:
        classification = _classify_sql(query, backend)
        if (
            not classification.protected
            or classification.domain != _MEMBERSHIP_SCOPE_DOMAIN
        ):
            raise ProfileUserWriteRejected()
        return await connection.execute(query, *parameters)
    if managed_backend != backend:
        raise ProfileUserWriteRejected()
    capability = _mint_membership_scope_sql(
        query,
        backend=backend,
        connection_identity=_profile_user_connection_identity(connection),
        execution_mode="execute",
    )
    try:
        return await connection.execute(capability, *parameters)
    finally:
        _revoke_membership_scope_sql(capability)


async def _execute_postgres_membership_timestamp_repair(
    connection: Any,
    *,
    table_name: str,
) -> Any:
    """Backfill one canonical membership timestamp during schema readiness."""
    if type(table_name) is not str:
        raise ProfileUserWriteRejected()
    query = _POSTGRES_MEMBERSHIP_TIMESTAMP_REPAIR_SQL.get(table_name)
    if query is None:
        raise ProfileUserWriteRejected()
    return await _execute_membership_scope_sql(
        connection,
        query,
        backend="postgres",
    )


def _pragma_name(statement: exp.Pragma) -> str | None:
    target = statement.this
    if isinstance(target, exp.EQ):
        target = target.this
    if isinstance(target, (exp.Column, exp.Identifier, exp.Anonymous, exp.Var)):
        name = target.name
        return name.lower() if name else None
    return None


def _target_table_name(target: Any) -> str | None:
    if isinstance(target, exp.Schema):
        target = target.this
    if not isinstance(target, exp.Table) or not target.name:
        return None
    return target.name.lower()


def _targets_users(target: Any) -> bool:
    return _target_table_name(target) == "users"


def _update_columns(update: exp.Update) -> frozenset[str] | None:
    columns: set[str] = set()
    for assignment in update.expressions:
        if not isinstance(assignment, exp.EQ):
            return None
        target = assignment.this
        if not isinstance(target, exp.Column) or not target.name:
            return None
        columns.add(target.name.lower())
    return frozenset(columns) if columns else None


def _insert_columns(insert: exp.Insert) -> tuple[str, ...] | None:
    target = insert.this
    if not isinstance(target, exp.Schema) or not target.expressions:
        return None
    columns: list[str] = []
    for column in target.expressions:
        if not isinstance(column, exp.Identifier) or not column.name:
            return None
        columns.append(column.name.lower())
    if len(columns) != len(set(columns)):
        return None
    return tuple(columns)


def _classification_cache_clear() -> None:
    _classify_sql.cache_clear()


def _classification_cache_info() -> Any:
    return _classify_sql.cache_info()
