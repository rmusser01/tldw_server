from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    get_authnz_migrations,
    migration_097_seed_notes_graph_semantic_manage_permission,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    NOTES_GRAPH_READ,
    NOTES_GRAPH_SEMANTIC_MANAGE,
    NOTES_GRAPH_WRITE,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.migrations import (
    Migration,
    MigrationManager,
)

pytestmark = pytest.mark.unit


def _create_rbac_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE roles (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
        CREATE TABLE permissions (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL, category TEXT NOT NULL
        );
        CREATE TABLE role_permissions (
            role_id INTEGER NOT NULL, permission_id INTEGER NOT NULL,
            PRIMARY KEY (role_id, permission_id)
        );
        INSERT INTO roles(id, name) VALUES
            (1, 'admin'), (2, 'user'), (3, 'moderator'), (4, 'reviewer'), (5, 'viewer');
        """
    )


def _semantic_grants(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            """
            SELECT r.name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE p.name = ?
            """,
            (NOTES_GRAPH_SEMANTIC_MANAGE,),
        )
    }


def test_semantic_manage_permission_is_catalogued_for_single_user_defaults() -> None:
    assert NOTES_GRAPH_SEMANTIC_MANAGE == "notes.graph.semantic.manage"
    assert NOTES_GRAPH_SEMANTIC_MANAGE in Settings().SINGLE_USER_DEFAULT_PERMISSIONS


def test_migration_097_grants_semantic_management_to_approved_roles() -> None:
    conn = sqlite3.connect(":memory:")
    _create_rbac_schema(conn)

    migration_097_seed_notes_graph_semantic_manage_permission(conn)
    migration_097_seed_notes_graph_semantic_manage_permission(conn)

    assert _semantic_grants(conn) == {"admin", "user", "moderator"}
    assert get_authnz_migrations()[-1].version == 97


def test_migration_097_rollback_reapply_preserves_revoked_mappings(tmp_path) -> None:
    db_path = tmp_path / "authnz.db"
    manager = MigrationManager(db_path)
    with sqlite3.connect(db_path) as conn:
        _create_rbac_schema(conn)
    manager.add_migration(
        Migration(
            97,
            "Seed Notes graph semantic management permission",
            migration_097_seed_notes_graph_semantic_manage_permission,
        )
    )

    manager.migrate(97)
    with sqlite3.connect(db_path) as conn:
        assert _semantic_grants(conn) == {"admin", "user", "moderator"}
        conn.execute(
            """
            DELETE FROM role_permissions
            WHERE role_id = (SELECT id FROM roles WHERE name = 'user')
              AND permission_id = (
                  SELECT id FROM permissions WHERE name = ?
              )
            """,
            (NOTES_GRAPH_SEMANTIC_MANAGE,),
        )

    manager.rollback(96)
    manager.migrate(97)
    with sqlite3.connect(db_path) as conn:
        assert _semantic_grants(conn) == {"admin", "moderator"}
        conn.execute(
            """
            DELETE FROM role_permissions
            WHERE permission_id = (
                SELECT id FROM permissions WHERE name = ?
            )
            """,
            (NOTES_GRAPH_SEMANTIC_MANAGE,),
        )

    manager.rollback(96)
    manager.migrate(97)
    with sqlite3.connect(db_path) as conn:
        assert _semantic_grants(conn) == set()


def test_semantic_manage_permission_does_not_replace_graph_read_or_write() -> None:
    semantic_only = {NOTES_GRAPH_SEMANTIC_MANAGE}

    assert NOTES_GRAPH_READ not in semantic_only
    assert NOTES_GRAPH_WRITE not in semantic_only
