from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    get_authnz_migrations,
    migration_096_seed_notes_graph_semantic_manage_permission,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    NOTES_GRAPH_READ,
    NOTES_GRAPH_SEMANTIC_MANAGE,
    NOTES_GRAPH_WRITE,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = pytest.mark.unit


def test_semantic_manage_permission_is_catalogued_for_single_user_defaults() -> None:
    assert NOTES_GRAPH_SEMANTIC_MANAGE == "notes.graph.semantic.manage"
    assert NOTES_GRAPH_SEMANTIC_MANAGE in Settings().SINGLE_USER_DEFAULT_PERMISSIONS


def test_migration_096_grants_semantic_management_to_approved_roles() -> None:
    conn = sqlite3.connect(":memory:")
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

    migration_096_seed_notes_graph_semantic_manage_permission(conn)
    migration_096_seed_notes_graph_semantic_manage_permission(conn)

    grants = set(
        conn.execute(
            """
            SELECT r.name, p.name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            """
        )
    )
    assert grants == {
        (role, NOTES_GRAPH_SEMANTIC_MANAGE)
        for role in ("admin", "user", "moderator")
    }
    assert get_authnz_migrations()[-1].version == 96


def test_semantic_manage_permission_does_not_replace_graph_read_or_write() -> None:
    semantic_only = {NOTES_GRAPH_SEMANTIC_MANAGE}

    assert NOTES_GRAPH_READ not in semantic_only
    assert NOTES_GRAPH_WRITE not in semantic_only
