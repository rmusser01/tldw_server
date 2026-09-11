from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema import (
    _validate_ready_metadata,
    ensure_postgres_profile_version_on_connection,
    ensure_postgres_profile_version_sync,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql


class _Result:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []


class _LegacyExecutor:
    def __init__(self, *, unsafe_indirect_writes: int = 0) -> None:
        self.profile_exists = False
        self.profile_ready = False
        self.unsafe_indirect_writes = unsafe_indirect_writes
        self.statements: list[str] = []

    def execute(
        self,
        statement: str,
        parameters: tuple[Any, ...] = (),
        *,
        connection: object | None = None,
    ) -> _Result:
        del parameters, connection
        normalized = " ".join(statement.split())
        self.statements.append(normalized)
        if "information_schema.tables" in normalized:
            return _Result([{"exists": True}])
        if "column_name = 'updated_at'" in normalized:
            return _Result([{"data_type": "timestamp without time zone"}])
        if "column_name = 'profile_version'" in normalized:
            if not self.profile_exists:
                return _Result()
            return _Result(
                [
                    {
                        "data_type": "timestamp with time zone",
                        "is_nullable": "NO" if self.profile_ready else "YES",
                        "column_default": (
                            "CURRENT_TIMESTAMP" if self.profile_ready else None
                        ),
                    }
                ]
            )
        if "pg_trigger" in normalized:
            return _Result([{"unsafe_count": self.unsafe_indirect_writes}])
        if "COUNT(*) AS invalid_count" in normalized:
            return _Result([{"invalid_count": 0}])
        if "ADD COLUMN profile_version" in normalized:
            self.profile_exists = True
        if "ALTER COLUMN profile_version SET DEFAULT" in normalized:
            self.profile_ready = True
        return _Result()


def test_sync_helper_backfills_from_updated_at_without_wall_clock_jump() -> None:
    executor = _LegacyExecutor()

    ensure_postgres_profile_version_sync(executor, connection=object())

    assert any(
        statement
        == "UPDATE public.users SET profile_version = updated_at"
        for statement in executor.statements
    )
    assert not any(
        "SET profile_version = CURRENT_TIMESTAMP" in statement
        for statement in executor.statements
    )
    assert any(
        "updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC'"
        in statement
        for statement in executor.statements
    )


def test_sync_helper_rejects_indirect_user_write_objects() -> None:
    executor = _LegacyExecutor(unsafe_indirect_writes=1)

    with pytest.raises(RuntimeError, match="indirect.*users write"):
        ensure_postgres_profile_version_sync(executor, connection=object())


@pytest.mark.parametrize(
    "invalid_default",
    [
        "'notnow'::text",
        "now() + interval '1 day'",
        "CURRENT_TIMESTAMP + interval '1 second'",
    ],
)
def test_ready_metadata_rejects_noncanonical_current_time_defaults(
    invalid_default: str,
) -> None:
    with pytest.raises(RuntimeError, match="readiness validation failed"):
        _validate_ready_metadata(
            {
                "data_type": "timestamp with time zone",
                "is_nullable": "NO",
                "column_default": invalid_default,
            }
        )


@pytest.mark.parametrize("canonical_default", ["CURRENT_TIMESTAMP", "now()"])
def test_ready_metadata_accepts_canonical_current_time_defaults(
    canonical_default: str,
) -> None:
    _validate_ready_metadata(
        {
            "data_type": "timestamp with time zone",
            "is_nullable": "NO",
            "column_default": canonical_default,
        }
    )


def test_indirect_write_audit_covers_dynamic_and_non_dml_paths() -> None:
    executor = _LegacyExecutor()

    ensure_postgres_profile_version_sync(executor, connection=object())

    audit_sql = next(
        statement for statement in executor.statements if "pg_trigger" in statement
    )
    assert "pg_event_trigger" in audit_sql
    assert "TRUNCATE" in audit_sql
    assert "COPY" in audit_sql
    assert r"\mEXECUTE\M" in audit_sql
    assert "pg_depend" in audit_sql
    assert "information_schema.views" in audit_sql
    assert "is_updatable" in audit_sql
    assert "pg_inherits" in audit_sql


def test_sync_helper_acquires_advisory_lock_before_schema_inspection() -> None:
    executor = _LegacyExecutor()

    ensure_postgres_profile_version_sync(executor, connection=object())

    assert "pg_advisory_xact_lock" in executor.statements[0]
    assert "information_schema.tables" in executor.statements[1]


@pytest.mark.asyncio
async def test_async_helper_crosses_managed_boundary_for_readiness_writes() -> None:
    class _ManagedConnection:
        _authnz_profile_user_backend = "postgres"

        def __init__(self) -> None:
            self._authnz_profile_user_guard_identity = object()
            self.profile_exists = False
            self.profile_ready = False
            self.executed: list[str] = []

        async def fetchval(self, statement: str, *args: Any) -> Any:
            normalized = " ".join(statement.split())
            if "pg_advisory_xact_lock" in normalized:
                assert args == (0x544C44575F505631,)
                return None
            if "information_schema.tables" in normalized:
                return True
            if "column_name = 'updated_at'" in normalized:
                return "timestamp with time zone"
            if "pg_trigger" in normalized:
                return 0
            if "COUNT(*)" in normalized:
                return 0
            raise AssertionError(f"unexpected fetchval: {normalized}")

        async def fetchrow(self, statement: str) -> Any:
            normalized = " ".join(statement.split())
            if "column_name = 'profile_version'" not in normalized:
                raise AssertionError(f"unexpected fetchrow: {normalized}")
            if not self.profile_exists:
                return None
            return {
                "data_type": "timestamp with time zone",
                "is_nullable": "NO" if self.profile_ready else "YES",
                "column_default": (
                    "CURRENT_TIMESTAMP" if self.profile_ready else None
                ),
            }

        async def execute(self, statement: object) -> None:
            concrete = _guard_sql(
                statement,
                backend="postgres",
                connection_identity=self._authnz_profile_user_guard_identity,
                operation="execute",
            )
            normalized = " ".join(concrete.split())
            self.executed.append(normalized)
            if "ADD COLUMN profile_version" in normalized:
                self.profile_exists = True
            if "ALTER COLUMN profile_version SET DEFAULT" in normalized:
                self.profile_ready = True

    connection = _ManagedConnection()

    await ensure_postgres_profile_version_on_connection(connection)

    assert connection.profile_ready is True
    assert any("ADD COLUMN profile_version" in sql for sql in connection.executed)
    assert any("SET profile_version = updated_at" in sql for sql in connection.executed)
