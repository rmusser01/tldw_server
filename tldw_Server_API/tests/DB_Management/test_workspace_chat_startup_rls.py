"""Non-bypass PostgreSQL owners retain isolated receipt tombstones."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def test_forced_owner_rls_all_operations_and_orphans(pg_restricted_backend) -> None:
    """Exercise actual upgraded policies as the table owner without RLS bypass."""
    db = CharactersRAGDB(":memory:", client_id="alice", backend=pg_restricted_backend)
    try:
        assert db.backend.table_exists("workspace_chat_startup_receipts")
        db.upsert_workspace("live-workspace", "Workspace")
        cid = db.add_conversation({"scope_type": "workspace", "workspace_id": "live-workspace"})
        with db.transaction() as conn:
            flags = conn.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = 'workspace_chat_startup_receipts'::regclass"
            ).fetchone()
            assert flags["relrowsecurity"] and flags["relforcerowsecurity"]
            for owner in ("alice", "bob"):
                conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner,))
                conn.execute(
                    "INSERT INTO workspace_chat_startup_receipts (owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id) VALUES (?, ?, ?, ?, ?)",
                    (owner, "a" * 64, "b" * 64, "c" * 64, "live-workspace"),
                )
            conn.execute("SELECT set_config('app.current_user_id', 'alice', true)")
            conn.execute(
                "UPDATE workspace_chat_startup_receipts SET conversation_id = ? WHERE owner_user_id = 'alice'", (cid,)
            )
            conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
            conn.execute("DELETE FROM workspaces WHERE id = 'live-workspace'")
            assert (
                conn.execute(
                    "SELECT conversation_id FROM workspace_chat_startup_receipts WHERE owner_user_id = 'alice'"
                ).fetchone()["conversation_id"]
                is None
            )
        for owner in ("alice", "bob", "", None):
            visible = 1 if owner else 0
            with db.transaction() as conn:
                conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner or "",))
                rows = conn.execute("SELECT owner_user_id FROM workspace_chat_startup_receipts").fetchall()
                assert [row["owner_user_id"] for row in rows] == ([owner] if owner else [])
                assert (
                    conn.execute("SELECT COUNT(*) AS n FROM workspace_chat_startup_receipts").fetchone()["n"] == visible
                )
                other = "bob" if owner == "alice" else "alice"
                assert (
                    conn.execute(
                        "UPDATE workspace_chat_startup_receipts SET invalidated_at = CURRENT_TIMESTAMP WHERE owner_user_id = ?",
                        (other,),
                    ).rowcount
                    == 0
                )
                assert (
                    conn.execute(
                        "DELETE FROM workspace_chat_startup_receipts WHERE owner_user_id = ?", (other,)
                    ).rowcount
                    == 0
                )
                assert (
                    conn.execute(
                        "SELECT conversation_id FROM workspace_chat_startup_receipts WHERE owner_user_id = ? AND key_digest = ?",
                        (other, "a" * 64),
                    ).fetchone()
                    is None
                )
            with pytest.raises(DatabaseError):
                with db.transaction() as conn:
                    conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner or "",))
                    conn.execute(
                        "INSERT INTO workspace_chat_startup_receipts (owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id) VALUES (?, ?, ?, ?, ?)",
                        (other, "d" * 64, "b" * 64, "c" * 64, "gone"),
                    )
            if owner:
                with pytest.raises(DatabaseError):
                    with db.transaction() as conn:
                        conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner,))
                        conn.execute(
                            "UPDATE workspace_chat_startup_receipts SET owner_user_id = ? WHERE owner_user_id = ?",
                            (other, owner),
                        )
        with db.transaction() as conn:
            conn.execute("SELECT set_config('app.current_user_id', 'alice', true)")
            assert (
                conn.execute(
                    "UPDATE workspace_chat_startup_receipts SET invalidated_at = CURRENT_TIMESTAMP WHERE owner_user_id = 'alice'"
                ).rowcount
                == 1
            )
            assert (
                conn.execute("DELETE FROM workspace_chat_startup_receipts WHERE owner_user_id = 'alice'").rowcount == 1
            )
    finally:
        db.close_all_connections()
