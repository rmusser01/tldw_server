from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    get_authnz_migrations,
    migration_095_seed_notes_graph_suggestion_permissions,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    KEYWORDS_CREATE,
    NOTES_GRAPH_SUGGEST,
    NOTES_LINK_KEYWORD,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = pytest.mark.unit


def test_notes_graph_suggestion_permission_catalog_constants_and_single_user_defaults() -> None:
    assert NOTES_GRAPH_SUGGEST == "notes.graph.suggest"
    assert NOTES_LINK_KEYWORD == "notes.link_keyword"
    assert KEYWORDS_CREATE == "keywords.create"

    defaults = set(Settings().SINGLE_USER_DEFAULT_PERMISSIONS)
    assert {NOTES_GRAPH_SUGGEST, NOTES_LINK_KEYWORD, KEYWORDS_CREATE} <= defaults


def test_migration_095_seeds_catalog_and_only_notes_writing_roles() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE roles (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE permissions (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            category TEXT NOT NULL
        );
        CREATE TABLE role_permissions (
            role_id INTEGER NOT NULL,
            permission_id INTEGER NOT NULL,
            PRIMARY KEY (role_id, permission_id)
        );
        INSERT INTO roles(id,name) VALUES
            (1,'admin'),(2,'user'),(3,'moderator'),(4,'reviewer'),(5,'viewer');
        """
    )

    migration_095_seed_notes_graph_suggestion_permissions(conn)
    migration_095_seed_notes_graph_suggestion_permissions(conn)

    permissions = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM permissions WHERE category IN ('notes', 'keywords')"
        ).fetchall()
    }
    assert permissions == {NOTES_GRAPH_SUGGEST, NOTES_LINK_KEYWORD, KEYWORDS_CREATE}
    grants = set(
        conn.execute(
            """
            SELECT r.name,p.name
            FROM role_permissions rp
            JOIN roles r ON r.id=rp.role_id
            JOIN permissions p ON p.id=rp.permission_id
            """
        ).fetchall()
    )
    expected = {
        (role, permission)
        for role in ("admin", "user", "moderator")
        for permission in (NOTES_GRAPH_SUGGEST, NOTES_LINK_KEYWORD, KEYWORDS_CREATE)
    }
    assert grants == expected
    assert not {role for role, _permission in grants} & {"reviewer", "viewer"}
    assert get_authnz_migrations()[-1].version == 95
