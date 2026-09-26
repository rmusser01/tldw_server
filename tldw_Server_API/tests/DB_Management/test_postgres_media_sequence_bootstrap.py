"""Media handles must preserve allocated IDs despite forced-RLS visibility."""

from dataclasses import replace
from secrets import token_hex

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context


@pytest.mark.integration
def test_unscoped_media_handle_bootstrap_preserves_ids_under_forced_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    """A new unscoped handle cannot reset IDs hidden by the tenant policy."""
    admin = DatabaseBackendFactory.create_backend(pg_database_config)
    role = f"email_sequence_{token_hex(6)}"
    password = token_hex(24)
    ident = admin.escape_identifier
    database = ident(pg_database_config.pg_database)
    original_owner = admin.execute("SELECT current_user").scalar
    backend = None
    handles = []
    admin.execute(
        f"CREATE ROLE {ident(role)} LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS"  # nosec B608 - generated role/password, escaped identifier
    )
    try:
        admin.execute(f"ALTER DATABASE {database} OWNER TO {ident(role)}")  # nosec B608 - escaped fixture identifiers
        config = replace(pg_database_config, connection_string=None, pg_user=role, pg_password=password)
        backend = DatabaseBackendFactory.create_backend(config)
        ids = []
        for index, user_id in enumerate((42, 42, 43)):
            # Match the ingestion worker: construct first, then enter user scope.
            db = MediaDatabase(":memory:", client_id=str(user_id), backend=backend)
            handles.append(db)
            with scoped_context(user_id=user_id):
                media_id, _, _ = db.add_media_with_keywords(
                    url=f"synthetic://sequence/{index}",
                    title=f"Synthetic {index}",
                    media_type="document",
                    content=f"Unique synthetic body {index}",
                    keywords=[],
                )
                ids.append(media_id)
        assert len(set(ids)) == 3
        with scoped_context(user_id=42):
            assert backend.execute("SELECT COUNT(*) FROM media").scalar == 2
        with scoped_context(user_id=43):
            assert backend.execute("SELECT COUNT(*) FROM media").scalar == 1
    finally:
        for db in handles:
            db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()
        admin.execute(f"REASSIGN OWNED BY {ident(role)} TO {ident(original_owner)}")  # nosec B608 - escaped fixture identifiers
        admin.execute(f"ALTER DATABASE {database} OWNER TO {ident(original_owner)}")  # nosec B608 - escaped fixture identifiers
        admin.execute(f"DROP OWNED BY {ident(role)}")  # nosec B608 - isolated fixture role
        admin.execute(f"DROP ROLE {ident(role)}")  # nosec B608 - isolated fixture role
        admin.get_pool().close_all()
