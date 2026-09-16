"""Native history snapshots and immutable projection transaction contracts."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError, snapshot_to_wire
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

OWNER_KEY = "server:test/account:alice"


@pytest.fixture(params=["sqlite", "postgres"])
def history_db(request, tmp_path):
    """Use real independent test storage on each supported backend."""
    kwargs = {"db_path": str(tmp_path / "history.sqlite"), "client_id": "alice"}
    if request.param == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(**kwargs)
    conversation_id = db.add_conversation({"character_id": 1, "title": "History"})
    yield db, conversation_id
    db.close_connection()
    if request.param == "postgres":
        db.backend.get_pool().close_all()


def snapshot(db, conversation_id, **kwargs):
    return db.get_conversation_history_snapshot(conversation_id, owner_client_id="alice", owner_key=OWNER_KEY, **kwargs)


def confirm_body(snap, projection_id="first", path=None):
    wire = snapshot_to_wire(snap)
    ids = [row["id"] for row in wire["nodes"]] if path is None else path
    return {
        "version": 1,
        "projection_id": projection_id,
        "owner_key": snap.owner_key,
        "conversation_id": snap.conversation_id,
        "source_digest": snap.source_digest,
        "fences": wire["fences"],
        "source_members": [{"id": row["id"], "revision": row["revision"]} for row in wire["nodes"]],
        "ordered_path_ids": ids,
        "cursor": {"kind": "after_message", "message_id": ids[-1]} if ids else {"kind": "empty"},
        "selection_revision": 1,
    }


def confirm(db, cid, body, **kwargs):
    return db.confirm_legacy_history_projection(body, owner_client_id="alice", owner_key=OWNER_KEY, **kwargs)


def add(db, cid, text="repeat", **kwargs):
    return db.add_message({"conversation_id": cid, "sender": "user", "content": text, **kwargs})


def test_legacy_two_views_replay_after_append_and_no_source_rewrite(history_db):
    db, cid = history_db
    a, b = add(db, cid), add(db, cid)
    original = snapshot(db, cid)
    assert original.interpretation_status["kind"] == "legacy_review_required"
    first_body, second_body = confirm_body(original, "first", [a]), confirm_body(original, "second", [b, a])
    first, second = confirm(db, cid, first_body), confirm(db, cid, second_body)
    assert first["projection_digest"] != second["projection_digest"]
    add(db, cid, "later")
    assert confirm(db, cid, first_body) == first
    assert snapshot(db, cid, projection_id="first").interpretation_status["ordered_path_ids"] == (a,)
    assert snapshot(db, cid, projection_id="second").interpretation_status["ordered_path_ids"] == (b, a)
    assert snapshot(db, cid).interpretation_status["kind"] == "legacy_review_required"
    assert all(row["parent_id"] is None for row in snapshot(db, cid).nodes)
    with pytest.raises(HistorySelectionError, match="projection_id_conflict"):
        confirm(db, cid, {**first_body, "selection_revision": 2})


def test_source_cas_owner_deleted_and_caller_rollback(history_db):
    db, cid = history_db
    add(db, cid)
    body = confirm_body(snapshot(db, cid))
    with pytest.raises(NotFoundError):
        db.get_conversation_history_snapshot(cid, owner_client_id="bob", owner_key=OWNER_KEY)
    with pytest.raises(NotFoundError):
        db.confirm_legacy_history_projection(body, owner_client_id="bob", owner_key=OWNER_KEY)
    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as conn:
            confirm(db, cid, body, conn=conn)
            raise RuntimeError("rollback")
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM conversation_history_projections").fetchone()["n"] == 0
    add(db, cid, "changed")
    with pytest.raises(HistorySelectionError, match="stale_source"):
        confirm(db, cid, body)
    current = confirm_body(snapshot(db, cid))
    confirm(db, cid, current)
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET deleted = TRUE WHERE id = ?", (cid,))
    with pytest.raises(NotFoundError):
        confirm(db, cid, current)


def test_snapshot_revisions_cover_metadata_assets_and_selected_content(history_db):
    db, cid = history_db
    mid = add(db, cid, images=[{"data": b"first", "mime": "image/png"}, {"data": b"other", "mime": "image/png"}])
    original = snapshot(db, cid)
    content = db.get_conversation_history_selected_content(
        cid, [mid], snapshot=original, owner_client_id="alice", owner_key=OWNER_KEY
    )
    assert content[0]["images"] == ("data:image/png;base64,Zmlyc3Q=", "data:image/png;base64,b3RoZXI=")
    with db.transaction() as conn:
        conn.execute("UPDATE message_images SET image_data = ? WHERE message_id = ? AND position = 1", (b"equal", mid))
    changed = snapshot(db, cid)
    assert original.nodes[0]["revision"] != changed.nodes[0]["revision"]
    with pytest.raises(HistorySelectionError, match="stale_source"):
        db.get_conversation_history_selected_content(
            cid, [mid], snapshot=original, owner_client_id="alice", owner_key=OWNER_KEY
        )
    db.set_message_metadata_extra(mid, {"tool_result": "changed"})
    assert snapshot(db, cid).nodes[0]["revision"] != changed.nodes[0]["revision"]


def test_complete_snapshot_equal_timestamp_ties_and_constant_query_count(history_db):
    db, cid = history_db
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO messages(id, conversation_id, sender, content, timestamp, last_modified, client_id) "
            "VALUES (?, ?, 'user', 'same', '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z', 'alice')",
            [(f"m{i:05d}", cid) for i in range(20001)],
        )
        conn.execute("UPDATE messages SET last_modified = '2026-01-01T00:00:00Z' WHERE id = 'm20000'")
        conn.execute("UPDATE messages SET last_modified = '2026-01-03T00:00:00Z' WHERE id = 'm00000'")
    db.set_message_metadata_extra("m20000", {"review_note": "first by modification time"})
    queries = []

    class ObservedConnection:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, key):
            return getattr(self.conn, key)

        def execute(self, sql, params=()):
            queries.append(sql)
            return self.conn.execute(sql, params)

    snap = snapshot(db, cid, conn=ObservedConnection(db.get_connection()))
    assert len(queries) == 1
    assert len(snap.nodes) == 20001
    assert [row["id"] for row in snap.nodes] == ["m20000", *[f"m{i:05d}" for i in range(1, 20000)], "m00000"]
    assert snap.interpretation_status["kind"] == "legacy_review_required"
    db.upsert_conversation_settings(cid, {"temperature": 0.25})
    configured = snapshot(db, cid)
    assert configured.storage_context_digest != snap.storage_context_digest
    confirm(db, cid, confirm_body(configured, path=["m20000", "m00000"]))
    queries.clear()
    accepted = snapshot(db, cid, projection_id="first", conn=ObservedConnection(db.get_connection()))
    assert len(queries) == 1
    assert len(accepted.nodes) == 20001
    assert accepted.interpretation_status["ordered_path_ids"] == ("m20000", "m00000")
    content = db.get_conversation_history_selected_content(
        cid, ["m20000"], snapshot=accepted, owner_client_id="alice", owner_key=OWNER_KEY
    )
    assert content[0]["extra_metadata"] == {"review_note": "first by modification time"}


def test_unversioned_chain_is_automatic_but_imported_claim_cannot_authorize_branch(history_db):
    db, cid = history_db
    a = add(db, cid)
    b = add(db, cid, parent_message_id=a)
    assert snapshot(db, cid).interpretation_status["kind"] == "parent_graph_v1"
    c = add(db, cid, parent_message_id=a)
    db.set_message_metadata_extra(
        c, {"history_admission": {"version": 1, "interpretation": {"kind": "parent_graph_v1"}}}
    )
    assert snapshot(db, cid).interpretation_status["kind"] == "legacy_review_required"
    body = confirm_body(snapshot(db, cid), path=[a, b])
    with pytest.raises(HistorySelectionError, match="invalid_projection"):
        confirm(db, cid, {**body, "ordered_path_ids": [a, a]})
    with pytest.raises(HistorySelectionError, match="invalid_projection"):
        confirm(db, cid, {**body, "cursor": {"kind": "after_message", "message_id": c}})


def test_statement_snapshot_remains_coherent_when_other_connection_commits_edit(history_db):
    db, cid = history_db
    mid = add(db, cid, "before")
    original = snapshot(db, cid)
    statement_ready, edit_done = threading.Event(), threading.Event()

    def edit():
        try:
            assert statement_ready.wait(10)
            db.update_message(mid, {"content": "after"}, expected_version=1)
        finally:
            edit_done.set()
            db.close_connection()

    class PausedConnection:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, key):
            return getattr(self.conn, key)

        def execute(self, sql, params=()):
            result = self.conn.execute(sql, params)
            if "WITH requested" in sql:
                statement_ready.set()
                assert edit_done.wait(10)
            return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(edit)
        captured = snapshot(db, cid, conn=PausedConnection(db.get_connection()))
        worker.result(timeout=15)
    assert captured.source_digest == original.source_digest
    assert captured.fences == original.fences
    changed = snapshot(db, cid)
    assert changed.source_digest != original.source_digest
    with pytest.raises(HistorySelectionError, match="stale_source"):
        confirm(db, cid, confirm_body(original))


@pytest.mark.parametrize("mutation", ["message", "metadata"])
def test_confirm_serializes_with_row_first_edit_without_lock_inversion(history_db, monkeypatch, mutation):
    db, cid = history_db
    mid = add(db, cid, "before")
    body = confirm_body(snapshot(db, cid))
    row_locked, release_edit = threading.Event(), threading.Event()
    is_postgres = db.backend_type == BackendType.POSTGRESQL

    advance = db.message_store._advance_history_version

    def pause_before_fence(conn, conversation_id):
        # The real edit has already changed/locked its message or metadata row.
        row_locked.set()
        assert release_edit.wait(10)
        advance(conn, conversation_id)

    monkeypatch.setattr(db.message_store, "_advance_history_version", pause_before_fence)

    def edit():
        try:
            if mutation == "message":
                db.update_message(mid, {"content": "after"}, expected_version=1)
            else:
                db.set_message_metadata_extra(mid, {"changed": True})
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        editing = pool.submit(edit)
        assert row_locked.wait(10)
        if is_postgres:
            # Uncommitted row edits are invisible; confirmation may accept the old
            # coherent snapshot without waiting for that row while holding its fence.
            accepted = confirm(db, cid, body)
            release_edit.set()
            editing.result(timeout=15)
            assert confirm(db, cid, body) == accepted
            with pytest.raises(HistorySelectionError, match="stale_projection"):
                snapshot(db, cid, projection_id="first")
        else:
            # SQLite's write transaction serializes the entire edit before CAS.
            release_edit.set()
            editing.result(timeout=15)
            with pytest.raises(HistorySelectionError, match="stale_source"):
                confirm(db, cid, body)


def test_accepted_projection_exposes_only_protected_descendant_base(history_db):
    from tldw_Server_API.app.core.Chat.history_selection import resolve_legacy_projection

    db, cid = history_db
    a, b = add(db, cid), add(db, cid)
    source = snapshot(db, cid)
    confirm(db, cid, confirm_body(source, "first", [a]))
    confirm(db, cid, confirm_body(source, "second", [b, a]))
    x = add(db, cid, parent_message_id=a)
    y = add(db, cid, parent_message_id=a)
    with db.transaction() as conn:
        for mid, pid in [(x, "first"), (y, "second")]:
            conn.execute(
                "UPDATE messages SET history_admission_json = ? WHERE id = ?",
                (json.dumps({"version": 1, "interpretation": {"kind": "legacy_linear_v1", "projection_id": pid}}), mid),
            )
    first = snapshot(db, cid, projection_id="first")
    second = snapshot(db, cid, projection_id="second")
    assert first.interpretation_status["ordered_path_ids"] == (a,)
    assert [
        row["id"]
        for row in resolve_legacy_projection(
            first.nodes, [a], {"kind": "after_message", "message_id": x}, projection_id="first"
        )
    ] == [a, x]
    assert [
        row["id"]
        for row in resolve_legacy_projection(
            second.nodes, [b, a], {"kind": "after_message", "message_id": y}, projection_id="second"
        )
    ] == [b, a, y]
    with pytest.raises(HistorySelectionError, match="interpretation_mismatch"):
        resolve_legacy_projection(first.nodes, [a], {"kind": "after_message", "message_id": y}, projection_id="first")


def test_independent_connections_confirm_two_immutable_paths(history_db):
    db, cid = history_db
    a, b = add(db, cid), add(db, cid)
    source = snapshot(db, cid)
    start = threading.Barrier(2)

    def accept(pid, path):
        try:
            start.wait(timeout=10)
            return confirm(db, cid, confirm_body(source, pid, path))
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(accept, "first", [a])
        second = pool.submit(accept, "second", [b, a])
        assert first.result(timeout=20)["ordered_path_ids"] == [a]
        assert second.result(timeout=20)["ordered_path_ids"] == [b, a]
    assert snapshot(db, cid, projection_id="first").interpretation_status["ordered_path_ids"] == (a,)
    assert snapshot(db, cid, projection_id="second").interpretation_status["ordered_path_ids"] == (b, a)


def test_confirmation_rejects_malformed_cursor_and_path_shapes(history_db):
    db, cid = history_db
    mid = add(db, cid, id="x")
    body = confirm_body(snapshot(db, cid), path=[mid])
    for changed in [
        {"ordered_path_ids": "x"},
        {"cursor": {"kind": "empty", "message_id": "x"}},
        {"cursor": {"kind": "after_message", "message_id": "x", "extra": True}},
    ]:
        with pytest.raises(HistorySelectionError, match="invalid_projection"):
            confirm(db, cid, {**body, **changed})


def test_selected_content_retains_tools_and_metadata_without_manifest_payload(history_db):
    db, cid = history_db
    mid = add(db, cid)
    db.add_message_metadata(mid, tool_calls=[{"id": "tool1", "name": "lookup"}])
    db.set_message_metadata_extra(mid, {"rag_context": {"source": "kept"}})
    snap = snapshot(db, cid)
    content = db.get_conversation_history_selected_content(
        cid, [mid], snapshot=snap, owner_client_id="alice", owner_key=OWNER_KEY
    )
    assert content[0]["tool_calls"] == [{"id": "tool1", "name": "lookup"}]
    assert content[0]["extra_metadata"] == {"rag_context": {"source": "kept"}}
    assert "tool_calls" not in snap.nodes[0]
    assert "extra_metadata" not in snap.nodes[0]


def test_projection_is_immutable_at_storage_boundary(history_db):
    db, cid = history_db
    add(db, cid)
    accepted = confirm(db, cid, confirm_body(snapshot(db, cid)))
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    with pytest.raises((DatabaseError, sqlite3.DatabaseError)):
        with db.transaction() as conn:
            conn.execute("UPDATE conversation_history_projections SET ordered_path_ids_json = '[]'")
    assert (
        confirm(
            db, cid, {key: value for key, value in accepted.items() if key not in {"created_at", "projection_digest"}}
        )
        == accepted
    )


def test_selected_attachment_budget_fails_explicitly_without_truncation(history_db):
    db, cid = history_db
    mid = add(db, cid, images=[{"data": b"x" * (5 * 1024 * 1024), "mime": "image/png"} for _ in range(7)])
    snap = snapshot(db, cid)
    assert len(snap.nodes[0]["assets"]) == 7
    with pytest.raises(HistorySelectionError, match="selected_content_too_large"):
        db.get_conversation_history_selected_content(
            cid, [mid], snapshot=snap, owner_client_id="alice", owner_key=OWNER_KEY
        )


def test_conversation_settings_fences_and_namespace_are_independent(history_db):
    db, cid = history_db
    add(db, cid)
    original = snapshot(db, cid)
    db.upsert_conversation_settings(cid, {"temperature": 0.3})
    settings_changed = snapshot(db, cid)
    assert settings_changed.fences.settings != original.fences.settings
    assert settings_changed.fences.history == original.fences.history
    assert settings_changed.source_digest == original.source_digest
    assert settings_changed.storage_context_digest != original.storage_context_digest
    with pytest.raises(HistorySelectionError, match="stale_source"):
        confirm(db, cid, confirm_body(original))
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET version = version + 1 WHERE id = ?", (cid,))
    changed = snapshot(db, cid)
    assert changed.fences.conversation != settings_changed.fences.conversation
    assert changed.fences.settings == settings_changed.fences.settings
    assert changed.fences.history == settings_changed.fences.history
    with pytest.raises(HistorySelectionError, match="stale_source"):
        confirm(db, cid, confirm_body(settings_changed))
    result = confirm(db, cid, confirm_body(changed))
    from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import LegacyHistoryProjectionV1

    assert LegacyHistoryProjectionV1.model_validate(result).projection_id == "first"
    with pytest.raises(HistorySelectionError, match="missing_projection"):
        db.get_conversation_history_snapshot(
            cid, owner_client_id="alice", owner_key="server:other/account:alice", projection_id="first"
        )


def test_storage_context_binds_behavior_without_requiring_character_readiness(history_db):
    from tldw_Server_API.tests.DB_Management.test_character_behavior_snapshot_migration import _snapshot

    db, cid = history_db
    mid = add(db, cid)
    absent = snapshot(db, cid)
    assert absent.interpretation_status["kind"] == "parent_graph_v1"
    with db.transaction() as conn:
        db.conversation_resume_store.put_behavior_snapshot(cid, _snapshot(), conn=conn)
    captured = snapshot(db, cid)
    assert captured.fences == absent.fences
    assert captured.storage_context_digest != absent.storage_context_digest
    with db.transaction() as conn:
        conn.execute("DELETE FROM conversation_behavior_snapshots WHERE conversation_id = ?", (cid,))
        from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import build_behavior_snapshot

        payload = json.loads(_snapshot().canonical_bytes)
        payload["participants"][0]["identity"]["name"] = "Changed accepted identity"
        db.conversation_resume_store.put_behavior_snapshot(cid, build_behavior_snapshot(payload), conn=conn)
    changed = snapshot(db, cid)
    assert changed.fences == captured.fences
    assert changed.storage_context_digest != captured.storage_context_digest
    with pytest.raises(HistorySelectionError, match="stale_source"):
        db.get_conversation_history_selected_content(
            cid, [mid], snapshot=captured, owner_client_id="alice", owner_key=OWNER_KEY
        )
