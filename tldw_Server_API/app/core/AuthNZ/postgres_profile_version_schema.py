"""Canonical PostgreSQL ownership for ``public.users.profile_version``."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _mint_profile_user_sql,
    _profile_user_backend,
    _profile_user_connection_identity,
    _revoke_profile_user_sql,
)

_USERS_EXISTS_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'users'
) AS exists
""".strip()

_PROFILE_METADATA_SQL = """
SELECT data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'users'
  AND column_name = 'profile_version'
""".strip()

_UPDATED_AT_METADATA_SQL = """
SELECT data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'users'
  AND column_name = 'updated_at'
""".strip()

_PROFILE_VERSION_MIGRATION_LOCK_KEY = 0x544C44575F505631
_PROFILE_VERSION_MIGRATION_LOCK_SQL = "SELECT pg_advisory_xact_lock($1)"
_PROFILE_VERSION_MIGRATION_LOCK_SYNC_SQL = (
    "SELECT pg_advisory_xact_lock(6074305139467900465)"
)
_USER_TIMESTAMP_COLUMNS = (
    "created_at",
    "updated_at",
    "last_login",
    "locked_until",
    "email_verified_at",
    "password_changed_at",
)
_USER_TIMESTAMP_METADATA_SQL = """
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'users'
  AND column_name = ANY($1::text[])
""".strip()

_INDIRECT_WRITE_AUDIT_SQL = r"""
SELECT (
    SELECT COUNT(*)
    FROM pg_trigger AS trigger
    JOIN pg_class AS relation ON relation.oid = trigger.tgrelid
    JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
    WHERE NOT trigger.tgisinternal
      AND namespace.nspname = 'public'
      AND relation.relname = 'users'
) + (
    SELECT COUNT(*)
    FROM pg_proc AS routine
    JOIN pg_namespace AS namespace ON namespace.oid = routine.pronamespace
    WHERE routine.prokind IN ('f', 'p')
      AND namespace.nspname NOT IN ('pg_catalog', 'information_schema')
      AND NOT EXISTS (
          SELECT 1
          FROM pg_depend AS dependency
          WHERE dependency.classid = 'pg_proc'::regclass
            AND dependency.objid = routine.oid
            AND dependency.deptype = 'e'
      )
      AND (
          pg_get_functiondef(routine.oid) ~* '\mEXECUTE\M'
          OR (
              pg_get_functiondef(routine.oid) ~* '\musers\M'
              AND pg_get_functiondef(routine.oid)
                  ~* '\m(UPDATE|INSERT|DELETE|MERGE|TRUNCATE|COPY)\M'
          )
      )
) + (
    SELECT COUNT(*)
    FROM pg_rewrite AS rewrite
    JOIN pg_class AS relation ON relation.oid = rewrite.ev_class
    JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
    WHERE rewrite.rulename <> '_RETURN'
      AND NOT EXISTS (
          SELECT 1
          FROM pg_depend AS dependency
          WHERE dependency.classid = 'pg_rewrite'::regclass
            AND dependency.objid = rewrite.oid
            AND dependency.deptype = 'e'
      )
      AND pg_get_ruledef(rewrite.oid) ~* '\musers\M'
      AND pg_get_ruledef(rewrite.oid)
          ~* '\m(UPDATE|INSERT|DELETE|MERGE|TRUNCATE|COPY)\M'
) + (
    SELECT COUNT(*)
    FROM information_schema.views AS view_info
    JOIN pg_namespace AS view_namespace
      ON view_namespace.nspname = view_info.table_schema
    JOIN pg_class AS view_relation
      ON view_relation.relnamespace = view_namespace.oid
     AND view_relation.relname = view_info.table_name
    JOIN pg_rewrite AS view_rewrite
      ON view_rewrite.ev_class = view_relation.oid
     AND view_rewrite.rulename = '_RETURN'
    JOIN pg_depend AS view_dependency
      ON view_dependency.classid = 'pg_rewrite'::regclass
     AND view_dependency.objid = view_rewrite.oid
     AND view_dependency.refclassid = 'pg_class'::regclass
    JOIN pg_class AS base_relation
      ON base_relation.oid = view_dependency.refobjid
    JOIN pg_namespace AS base_namespace
      ON base_namespace.oid = base_relation.relnamespace
    WHERE view_info.is_updatable = 'YES'
      AND base_namespace.nspname = 'public'
      AND base_relation.relname = 'users'
) + (
    SELECT COUNT(*)
    FROM pg_inherits AS inheritance
    JOIN pg_class AS parent_relation ON parent_relation.oid = inheritance.inhparent
    JOIN pg_namespace AS parent_namespace
      ON parent_namespace.oid = parent_relation.relnamespace
    WHERE parent_namespace.nspname = 'public'
      AND parent_relation.relname = 'users'
) + (
    SELECT COUNT(*)
    FROM pg_event_trigger AS event_trigger
    WHERE event_trigger.evtenabled <> 'D'
      AND NOT EXISTS (
          SELECT 1
          FROM pg_depend AS dependency
          WHERE dependency.classid = 'pg_event_trigger'::regclass
            AND dependency.objid = event_trigger.oid
            AND dependency.deptype = 'e'
      )
) AS unsafe_count
""".strip()

_ADD_PROFILE_SQL = (
    "ALTER TABLE public.users ADD COLUMN profile_version TIMESTAMPTZ"
)
_NORMALIZE_UPDATED_AT_SQL = """
ALTER TABLE public.users
ALTER COLUMN updated_at TYPE TIMESTAMPTZ
USING updated_at AT TIME ZONE 'UTC'
""".strip()
_NORMALIZE_PROFILE_SQL = """
ALTER TABLE public.users
ALTER COLUMN profile_version TYPE TIMESTAMPTZ
USING profile_version AT TIME ZONE 'UTC'
""".strip()
_BACKFILL_SQL = "UPDATE public.users SET profile_version = updated_at"
_SET_DEFAULT_SQL = """
ALTER TABLE public.users
ALTER COLUMN profile_version SET DEFAULT CURRENT_TIMESTAMP
""".strip()
_SET_NOT_NULL_SQL = """
ALTER TABLE public.users
ALTER COLUMN profile_version SET NOT NULL
""".strip()
_NULL_PROFILE_COUNT_SQL = (
    "SELECT COUNT(*) AS invalid_count FROM public.users "
    "WHERE profile_version IS NULL"
)
_NULL_UPDATED_AT_COUNT_SQL = (
    "SELECT COUNT(*) AS invalid_count FROM public.users WHERE updated_at IS NULL"
)


async def ensure_postgres_profile_version_on_connection(conn: Any) -> None:
    """Repair and validate one caller-owned PostgreSQL transaction connection."""
    await conn.fetchval(
        _PROFILE_VERSION_MIGRATION_LOCK_SQL,
        _PROFILE_VERSION_MIGRATION_LOCK_KEY,
    )
    users_exists = bool(await conn.fetchval(_USERS_EXISTS_SQL))
    if not users_exists:
        raise RuntimeError(
            "AuthNZ profile_version migration requires public.users"
        )
    await _audit_indirect_writes_async(conn)
    profile_metadata = await conn.fetchrow(_PROFILE_METADATA_SQL)
    profile_existed = profile_metadata is not None
    updated_at_type = await conn.fetchval(_UPDATED_AT_METADATA_SQL)
    if updated_at_type == "timestamp without time zone":
        await _execute_async_protected(conn, _NORMALIZE_UPDATED_AT_SQL, "alter", ())
    elif updated_at_type != "timestamp with time zone":
        raise RuntimeError(
            "AuthNZ profile_version migration found an invalid updated_at type"
        )

    if not profile_existed:
        await _execute_async_protected(conn, _ADD_PROFILE_SQL, "alter", ())
        if int(await conn.fetchval(_NULL_UPDATED_AT_COUNT_SQL)):
            raise RuntimeError(
                "AuthNZ profile_version migration found null updated_at values"
            )
        await _execute_async_protected(
            conn,
            _BACKFILL_SQL,
            "update",
            ("profile_version",),
        )
    else:
        profile_type = str(profile_metadata["data_type"])
        if profile_type == "timestamp without time zone":
            await _execute_async_protected(conn, _NORMALIZE_PROFILE_SQL, "alter", ())
        elif profile_type != "timestamp with time zone":
            raise RuntimeError(
                "AuthNZ profile_version migration found an invalid column type"
            )
        if int(await conn.fetchval(_NULL_PROFILE_COUNT_SQL)):
            raise RuntimeError(
                "AuthNZ profile_version readiness validation failed"
            )

    if int(await conn.fetchval(_NULL_PROFILE_COUNT_SQL)):
        raise RuntimeError("AuthNZ profile_version readiness validation failed")
    await _execute_async_protected(conn, _SET_DEFAULT_SQL, "alter", ())
    await _execute_async_protected(conn, _SET_NOT_NULL_SQL, "alter", ())
    _validate_ready_metadata(await conn.fetchrow(_PROFILE_METADATA_SQL))
    await _audit_indirect_writes_async(conn)


async def ensure_postgres_user_timestamp_timezones_on_connection(conn: Any) -> None:
    """Normalize known legacy user timestamps inside the caller transaction."""
    await conn.fetchval(
        _PROFILE_VERSION_MIGRATION_LOCK_SQL,
        _PROFILE_VERSION_MIGRATION_LOCK_KEY,
    )
    rows = await conn.fetch(_USER_TIMESTAMP_METADATA_SQL, list(_USER_TIMESTAMP_COLUMNS))
    metadata = {str(row["column_name"]): str(row["data_type"]) for row in rows}
    for column_name in _USER_TIMESTAMP_COLUMNS:
        data_type = metadata.get(column_name)
        if data_type is None or data_type == "timestamp with time zone":
            continue
        if data_type != "timestamp without time zone":
            raise RuntimeError(
                "AuthNZ user timestamp migration found an invalid column type"
            )
        statement = (
            f"ALTER TABLE public.users ALTER COLUMN {column_name} "
            f"TYPE TIMESTAMPTZ USING {column_name} AT TIME ZONE 'UTC'"
        )
        await _execute_async_protected(conn, statement, "alter", ())


def ensure_postgres_profile_version_sync(
    executor: Any,
    *,
    connection: Any,
) -> None:
    """Synchronous counterpart over a caller-owned transaction connection."""
    _sync_first(
        executor,
        _PROFILE_VERSION_MIGRATION_LOCK_SYNC_SQL,
        connection=connection,
    )
    users_row = _sync_first(executor, _USERS_EXISTS_SQL, connection=connection)
    if not users_row or not bool(users_row.get("exists")):
        raise RuntimeError(
            "AuthNZ profile_version migration requires public.users"
        )
    _audit_indirect_writes_sync(executor, connection=connection)

    profile_metadata = _sync_first(
        executor,
        _PROFILE_METADATA_SQL,
        connection=connection,
    )
    profile_existed = profile_metadata is not None
    updated_metadata = _sync_first(
        executor,
        _UPDATED_AT_METADATA_SQL,
        connection=connection,
    )
    updated_at_type = updated_metadata.get("data_type") if updated_metadata else None
    if updated_at_type == "timestamp without time zone":
        _sync_execute(executor, _NORMALIZE_UPDATED_AT_SQL, connection=connection)
    elif updated_at_type != "timestamp with time zone":
        raise RuntimeError(
            "AuthNZ profile_version migration found an invalid updated_at type"
        )

    if not profile_existed:
        _sync_execute(executor, _ADD_PROFILE_SQL, connection=connection)
        if _sync_count(
            executor,
            _NULL_UPDATED_AT_COUNT_SQL,
            connection=connection,
        ):
            raise RuntimeError(
                "AuthNZ profile_version migration found null updated_at values"
            )
        _sync_execute(executor, _BACKFILL_SQL, connection=connection)
    else:
        profile_type = str(profile_metadata.get("data_type"))
        if profile_type == "timestamp without time zone":
            _sync_execute(executor, _NORMALIZE_PROFILE_SQL, connection=connection)
        elif profile_type != "timestamp with time zone":
            raise RuntimeError(
                "AuthNZ profile_version migration found an invalid column type"
            )
        if _sync_count(
            executor,
            _NULL_PROFILE_COUNT_SQL,
            connection=connection,
        ):
            raise RuntimeError(
                "AuthNZ profile_version readiness validation failed"
            )

    if _sync_count(executor, _NULL_PROFILE_COUNT_SQL, connection=connection):
        raise RuntimeError("AuthNZ profile_version readiness validation failed")
    _sync_execute(executor, _SET_DEFAULT_SQL, connection=connection)
    _sync_execute(executor, _SET_NOT_NULL_SQL, connection=connection)
    _validate_ready_metadata(
        _sync_first(executor, _PROFILE_METADATA_SQL, connection=connection)
    )
    _audit_indirect_writes_sync(executor, connection=connection)


async def _execute_async_protected(
    conn: Any,
    statement: str,
    operation: str,
    columns: tuple[str, ...],
) -> None:
    managed_backend = _profile_user_backend(conn)
    if managed_backend is None:
        await conn.execute(statement)
        return
    if managed_backend != "postgres":
        raise RuntimeError("AuthNZ profile_version backend mismatch")
    capability = _mint_profile_user_sql(
        statement,
        backend="postgres",
        connection_identity=_profile_user_connection_identity(conn),
        operation=operation,
        columns=columns,
    )
    try:
        await conn.execute(capability)
    finally:
        _revoke_profile_user_sql(capability)


async def _audit_indirect_writes_async(conn: Any) -> None:
    if int(await conn.fetchval(_INDIRECT_WRITE_AUDIT_SQL)):
        raise RuntimeError(
            "AuthNZ profile_version found an unsafe indirect users write"
        )


def _audit_indirect_writes_sync(executor: Any, *, connection: Any) -> None:
    row = _sync_first(executor, _INDIRECT_WRITE_AUDIT_SQL, connection=connection)
    if not row or type(row.get("unsafe_count")) is not int:
        raise RuntimeError("AuthNZ profile_version indirect-write audit failed")
    if row["unsafe_count"]:
        raise RuntimeError(
            "AuthNZ profile_version found an unsafe indirect users write"
        )


def _validate_ready_metadata(metadata: Any) -> None:
    if metadata is None:
        raise RuntimeError("AuthNZ profile_version readiness validation failed")
    try:
        data_type = metadata["data_type"]
        nullable = metadata["is_nullable"]
        default = metadata["column_default"]
    except (KeyError, TypeError):
        raise RuntimeError(
            "AuthNZ profile_version readiness validation failed"
        ) from None
    normalized_default = default.strip().lower() if isinstance(default, str) else None
    if (
        data_type != "timestamp with time zone"
        or nullable != "NO"
        or normalized_default not in {"current_timestamp", "now()"}
    ):
        raise RuntimeError("AuthNZ profile_version readiness validation failed")


def _sync_execute(executor: Any, statement: str, *, connection: Any) -> Any:
    return executor.execute(statement, (), connection=connection)


def _sync_first(executor: Any, statement: str, *, connection: Any) -> Any:
    result = _sync_execute(executor, statement, connection=connection)
    try:
        rows = result.rows
    except (AttributeError, TypeError):
        raise RuntimeError(
            "AuthNZ profile_version readiness validation failed"
        ) from None
    if not rows:
        return None
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError(
            "AuthNZ profile_version readiness validation failed"
        )
    return rows[0]


def _sync_count(executor: Any, statement: str, *, connection: Any) -> int:
    row = _sync_first(executor, statement, connection=connection)
    if not row or type(row.get("invalid_count")) is not int:
        raise RuntimeError("AuthNZ profile_version readiness validation failed")
    return row["invalid_count"]


__all__ = [
    "ensure_postgres_profile_version_on_connection",
    "ensure_postgres_profile_version_sync",
]
