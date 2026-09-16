"""Runtime-backed factory helpers for Media DB sessions."""

from __future__ import annotations

import configparser
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseBackend
from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_class import (
    load_media_database_cls,
)


BackendLoader = Callable[[], DatabaseBackend | None]
ContentBackendResolver = Callable[[], DatabaseBackend | None]
RUNTIME_FACTORY_EXCEPTIONS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    configparser.Error,
)


@dataclass(frozen=True, slots=True)
class MediaDbRuntimeConfig:
    default_db_path: str
    default_config: configparser.ConfigParser
    postgres_content_mode: bool
    backend_loader: BackendLoader


def _load_media_database_cls() -> type[Any]:
    return load_media_database_cls()


def create_media_database(
    client_id: str,
    *,
    db_path: str | Path | None = None,
    backend: DatabaseBackend | None = None,
    config: configparser.ConfigParser | None = None,
    runtime: MediaDbRuntimeConfig,
) -> Any:
    """Instantiate a MediaDatabase using runtime-scoped defaults."""

    media_database_cls = _load_media_database_cls()
    cfg = config or runtime.default_config
    target_path = Path(db_path) if db_path else Path(runtime.default_db_path)
    backend_to_use = backend if backend is not None else runtime.backend_loader()

    if runtime.postgres_content_mode:
        if (
            backend_to_use is None
            or backend_to_use.backend_type != BackendType.POSTGRESQL
        ):
            raise RuntimeError(
                "PostgreSQL content backend configured but backend could not be created. "
                "Ensure psycopg is installed and TLDW_CONTENT_PG_* settings are set."
            )
        return media_database_cls(
            db_path=str(target_path),
            client_id=client_id,
            backend=backend_to_use,
            config=cfg,
        )

    return media_database_cls(
        db_path=str(target_path),
        client_id=client_id,
        backend=backend_to_use,
        config=cfg,
    )


def get_current_media_schema_version() -> int:
    """Return the current Media DB schema version from the canonical runtime class."""

    media_database_cls = _load_media_database_cls()
    return int(media_database_cls._CURRENT_SCHEMA_VERSION)


def validate_postgres_content_backend(
    *,
    get_content_backend_instance: ContentBackendResolver,
    runtime: MediaDbRuntimeConfig,
) -> None:
    """Ensure the configured PostgreSQL content backend is usable."""

    backend: DatabaseBackend | None
    try:
        backend = get_content_backend_instance()
    except RuntimeError:
        raise

    if backend is None:
        if runtime.postgres_content_mode:
            raise RuntimeError(
                "PostgreSQL content backend configured but not initialized. "
                "Ensure psycopg is installed and TLDW_CONTENT_PG_* (or DATABASE_URL) is set."
            )
        return

    if backend.backend_type != BackendType.POSTGRESQL:
        if runtime.postgres_content_mode:
            raise RuntimeError(
                "PostgreSQL content backend required but a different backend was provided."
            )
        return

    validator = create_media_database(
        "content_backend_validator",
        db_path=":memory:",
        backend=backend,
        runtime=runtime,
    )
    try:
        with backend.transaction() as conn:  # type: ignore[arg-type]
            try:
                version_result = backend.execute(
                    "SELECT version FROM schema_version LIMIT 1",
                    connection=conn,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "PostgreSQL content schema validation failed while reading schema_version. "
                    "The content schema may be missing, inaccessible, or not yet migrated. "
                    "Inspect the database connection and, if needed, run migrations: "
                    "python -m tldw_Server_API.app.core.DB_Management.migration_tools "
                    "or apply the SQL under app/core/DB_Management/migrations/."
                ) from exc
            current_version_row = version_result.first if version_result else None
            if isinstance(current_version_row, dict):
                current_version_raw = current_version_row.get("version")
            else:
                current_version_raw = current_version_row
            try:
                current_version = int(current_version_raw or 0)
            except (TypeError, ValueError):
                current_version = 0

            expected_version = validator._CURRENT_SCHEMA_VERSION
            if current_version != expected_version:
                raise RuntimeError(
                    "PostgreSQL content schema validation failed: schema is outdated. "
                    f"Current version={current_version}, expected={expected_version}. "
                    "Run migrations: python -m tldw_Server_API.app.core.DB_Management.migration_tools "
                    "or apply the SQL under app/core/DB_Management/migrations/."
                )

            required_policies = {
                "media": [
                    "media_visibility_access",
                ],
                "sync_log": [
                    "sync_scope_admin",
                    "sync_scope_personal",
                    "sync_scope_org",
                    "sync_scope_team",
                ],
                "operationownedclonekeywords": [
                    "owned_clone_pending_keyword_access",
                ],
            }

            for table, policies in required_policies.items():
                for policy in policies:
                    policy_exists = validator._postgres_policy_exists(conn, table, policy)
                    if not policy_exists:
                        raise RuntimeError(
                            "PostgreSQL content RLS validation failed: policy "
                            f"'{policy}' on table '{table}' is missing or could not be inspected. "
                            "Apply policies via pg_rls_policies.ensure_* helpers or run: "
                            "python -m tldw_Server_API.app.core.DB_Management.migration_tools --apply-rls"
                        )
    finally:
        with contextlib.suppress(RUNTIME_FACTORY_EXCEPTIONS):
            validator.close_connection()


__all__ = [
    "MediaDbRuntimeConfig",
    "RUNTIME_FACTORY_EXCEPTIONS",
    "create_media_database",
    "get_current_media_schema_version",
    "validate_postgres_content_backend",
]
