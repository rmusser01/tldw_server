"""Migration helpers extracted from Media DB schema bootstrap."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from tldw_Server_API.app.core.DB_Management.media_db.errors import SchemaError
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.policies import (
    ensure_postgres_policies,
)

PostgresMigrationMap = dict[int, Callable[[Any], None]]


class SupportsPostgresMigrations(Protocol):
    """Protocol for DB objects that expose PostgreSQL migration body methods."""

    def _postgres_migrate_to_v5(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v6(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v7(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v8(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v9(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v10(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v11(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v12(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v13(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v14(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v15(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v16(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v17(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v18(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v19(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v20(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v21(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v22(self, conn: Any) -> None: ...
    def _postgres_migrate_to_v23(self, conn: Any) -> None: ...
    def _update_schema_version_postgres(self, conn: Any, version: int) -> None: ...
    def _ensure_postgres_rls(self, conn: Any) -> None: ...


def build_postgres_migration_map(db: SupportsPostgresMigrations) -> PostgresMigrationMap:
    """Return the bound PostgreSQL migration map for the supplied DB object."""

    return {
        5: db._postgres_migrate_to_v5,
        6: db._postgres_migrate_to_v6,
        7: db._postgres_migrate_to_v7,
        8: db._postgres_migrate_to_v8,
        9: db._postgres_migrate_to_v9,
        10: db._postgres_migrate_to_v10,
        11: db._postgres_migrate_to_v11,
        12: db._postgres_migrate_to_v12,
        13: db._postgres_migrate_to_v13,
        14: db._postgres_migrate_to_v14,
        15: db._postgres_migrate_to_v15,
        16: db._postgres_migrate_to_v16,
        17: db._postgres_migrate_to_v17,
        18: db._postgres_migrate_to_v18,
        19: db._postgres_migrate_to_v19,
        20: db._postgres_migrate_to_v20,
        21: db._postgres_migrate_to_v21,
        22: db._postgres_migrate_to_v22,
        23: db._postgres_migrate_to_v23,
    }


def get_postgres_migrations(db: SupportsPostgresMigrations) -> PostgresMigrationMap:
    """Return the PostgreSQL migration mapping owned by the package helper."""

    return build_postgres_migration_map(db)


def run_postgres_migrations(
    db: SupportsPostgresMigrations,
    conn: Any,
    current_version: int,
    target_version: int,
) -> None:
    """Run PostgreSQL migrations via the package-owned registry and loop."""

    migrations = get_postgres_migrations(db)
    applied_version = current_version

    for version in sorted(migrations):
        if applied_version < version <= target_version:
            migrations[version](conn)
            db._update_schema_version_postgres(conn, version)
            applied_version = version

    ensure_postgres_policies(db, conn)

    if applied_version < target_version:
        raise SchemaError(
            f"PostgreSQL migration path incomplete for MediaDatabase: reached {applied_version}, expected {target_version}."
        )


__all__ = [
    "PostgresMigrationMap",
    "SupportsPostgresMigrations",
    "build_postgres_migration_map",
    "get_postgres_migrations",
    "run_postgres_migrations",
]
