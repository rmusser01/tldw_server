"""Request-scoped session helpers for the Media DB package."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import suppress
from typing import Callable

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseBackend,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
    is_factory_managed_backend,
    release_managed_backend,
)

DatabaseFactory = Callable[..., object]


def _load_default_media_database_factory() -> DatabaseFactory:
    """Load the default Media DB constructor on demand."""
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
        _load_media_database_cls,
    )

    return _load_media_database_cls()


def _create_sqlite_backend(db_path: str, client_id: str) -> DatabaseBackend:
    """Create the shared SQLite backend used by request-scoped Media DB wrappers."""
    return DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=db_path,
            client_id=client_id,
        )
    )


@dataclass(slots=True)
class MediaDbSession:
    """Request-scoped wrapper around a Media DB handle.

    The wrapped ``database`` object owns the actual backend resources. Callers
    use this object as a thin proxy and normally do not manage backend cleanup
    directly; request teardown should call ``release_context_connection()`` so
    pooled or context-bound connections can be released when supported.
    """

    db_path: str
    client_id: str
    database: object
    org_id: int | None = None
    team_id: int | None = None
    owns_backend_resources: bool = False

    def __getattr__(self, item: str) -> object:
        """Delegate attribute access to the wrapped database implementation."""
        return getattr(self.database, item)

    def release_context_connection(self) -> None:
        """Release any request-bound connection held by the wrapped database."""
        release = getattr(self.database, "release_context_connection", None)
        if callable(release):
            release()
        close_connection = getattr(self.database, "close_connection", None)
        if callable(close_connection):
            with suppress(AttributeError, OSError, RuntimeError, TypeError, ValueError):
                close_connection()


@dataclass(slots=True)
class MediaDbFactory:
    """Factory for request-scoped ``MediaDbSession`` handles.

    The factory caches constructor inputs for a specific content database path
    and optionally owns a shared backend (for example a SQLite backend created
    once per process). Each ``for_request()`` call returns a fresh session that
    applies request-local ``org_id`` and ``team_id`` overrides to the wrapped
    database handle.
    """

    db_path: str
    client_id: str
    backend: DatabaseBackend | None = None
    database_factory: DatabaseFactory | None = None

    @classmethod
    def for_sqlite_path(cls, db_path: str, client_id: str) -> MediaDbFactory:
        """Create a factory that owns a shared SQLite backend for one DB path."""
        return cls(
            db_path=db_path,
            client_id=client_id,
            backend=_create_sqlite_backend(db_path, client_id),
        )

    def for_request(
        self,
        *,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> MediaDbSession:
        """Create a fresh request-scoped session for this content database.

        The returned session wraps a newly constructed Media DB object. When the
        wrapped object exposes ``default_org_id`` or ``default_team_id``, those
        request-scoped overrides are applied before the session is returned.
        Callers should release request-bound connections via the session during
        request teardown rather than closing the shared factory backend.
        """

        database_factory = self.database_factory
        if database_factory is None:
            database_factory = _load_default_media_database_factory()

        if self.backend is not None:
            database = database_factory(
                db_path=self.db_path,
                client_id=self.client_id,
                backend=self.backend,
            )
        else:
            database = database_factory(
                db_path=self.db_path,
                client_id=self.client_id,
            )

        if hasattr(database, "default_org_id"):
            database.default_org_id = org_id
        if hasattr(database, "default_team_id"):
            database.default_team_id = team_id

        return MediaDbSession(
            db_path=self.db_path,
            client_id=self.client_id,
            database=database,
            org_id=org_id,
            team_id=team_id,
            owns_backend_resources=self.backend is None,
        )

    def close(self) -> None:
        """Close any pooled backend resources owned by this factory."""
        backend = self.backend
        if backend is None:
            return
        self.backend = None
        if (
            getattr(backend, "backend_type", None) == BackendType.SQLITE
            and is_factory_managed_backend(backend)
        ):
            release_managed_backend(backend)
            return
        try:
            backend.get_pool().close_all()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return
