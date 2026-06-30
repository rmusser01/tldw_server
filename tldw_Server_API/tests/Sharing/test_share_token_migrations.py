"""SQLite migration coverage for workspace share tokens."""
from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    migration_001_create_users_table,
    migration_077_create_sharing_tables,
    migration_087_expand_share_tokens_resource_type_for_prototypes,
)

pytestmark = pytest.mark.unit


def test_migration_087_retries_from_clean_rebuild_table() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_077_create_sharing_tables(conn)
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES (1, 'owner', 'owner@test.com', 'hash')"
        )
        conn.execute(
            """
            INSERT INTO share_tokens (
                id, token_hash, token_prefix, resource_type, resource_id,
                owner_user_id, access_level, allow_clone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "hash_retry",
                "pref",
                "workspace",
                "prototype_workspace::pws_retry",
                1,
                "view_chat",
                1,
            ),
        )
        conn.execute(
            """
            CREATE TABLE share_tokens_new (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                token_hash      TEXT UNIQUE NOT NULL,
                token_prefix    TEXT NOT NULL,
                resource_type   TEXT NOT NULL
                    CHECK (resource_type IN ('chatbook', 'workspace', 'prototype_workspace')),
                resource_id     TEXT NOT NULL,
                owner_user_id   INTEGER NOT NULL,
                access_level    TEXT NOT NULL DEFAULT 'view_chat',
                allow_clone     INTEGER NOT NULL DEFAULT 1,
                password_hash   TEXT,
                max_uses        INTEGER,
                use_count       INTEGER NOT NULL DEFAULT 0,
                expires_at      TIMESTAMP,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                revoked_at      TIMESTAMP,
                FOREIGN KEY (owner_user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO share_tokens_new (
                id, token_hash, token_prefix, resource_type, resource_id,
                owner_user_id, access_level, allow_clone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "hash_retry",
                "pref",
                "prototype_workspace",
                "pws_retry",
                1,
                "view_chat",
                1,
            ),
        )
        conn.commit()

        migration_087_expand_share_tokens_resource_type_for_prototypes(conn)

        row = conn.execute(
            "SELECT resource_type, resource_id FROM share_tokens WHERE token_hash = ?",
            ("hash_retry",),
        ).fetchone()
        scratch = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'share_tokens_new'"
        ).fetchone()

        assert row == ("prototype_workspace", "pws_retry")
        assert scratch is None
    finally:
        conn.close()
