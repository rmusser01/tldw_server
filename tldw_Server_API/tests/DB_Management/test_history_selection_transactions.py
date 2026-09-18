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


def selected(db, cid, cursor=None):
    from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection

    snap = snapshot_to_wire(snapshot(db, cid))
    result = resolve_history_selection(snap, {
        "owner_key": OWNER_KEY, "conversation_id": cid,
        "interpretation": {"kind": "parent_graph_v1"},
        "cursor": cursor or {"kind": "empty"}, "selection_revision": 3,
    }, "send", "client-provenance")
    assert result["status"] == "ready"
    return result["selection"]


def admit(db, cid, selection, mid="accepted-user", **kwargs):
    return db.append_selected_history_input(
        cid, selection, {"id": mid, "sender": "user", "content": "new input", **kwargs},
        owner_client_id="alice", owner_key=OWNER_KEY,
    )


def settle(db, cid, admission, mid="accepted-assistant", **kwargs):
    reference = {key: admission[key] for key in (
        "version", "owner_key", "conversation_id", "input_message_id",
        "input_message_revision", "selection_digest",
    )}
    return db.settle_history_admission(
        cid, reference, {"id": mid, "sender": "assistant", "content": "response", **kwargs},
        owner_client_id="alice", owner_key=OWNER_KEY,
    )


def test_selected_admission_empty_parent_replay_and_late_settlement(history_db):
    db, cid = history_db
    selection = selected(db, cid)
    accepted = admit(db, cid, selection)
    assert db.get_message_by_id("accepted-user")["parent_message_id"] is None
    assert admit(db, cid, selection) == accepted
    another = admit(db, cid, selected(db, cid), "other-user")
    assert another["input_message_id"] != accepted["input_message_id"]
    settle(db, cid, accepted)
    assert db.get_message_by_id("accepted-assistant")["parent_message_id"] == "accepted-user"
    assert settle(db, cid, accepted) == "accepted-assistant"
    with pytest.raises(HistorySelectionError, match="message_id_conflict"):
        settle(db, cid, another)


def test_selected_admission_conflicting_parent_and_stale_source_write_nothing(history_db):
    db, cid = history_db
    initial = selected(db, cid)
    with pytest.raises(HistorySelectionError, match="parent_mismatch"):
        admit(db, cid, initial, parent_message_id="missing")
    assert not snapshot(db, cid).nodes
    root = add(db, cid)
    selection = selected(db, cid, {"kind": "after_message", "message_id": root})
    db.update_message(root, {"content": "edited"}, expected_version=1)
    with pytest.raises(HistorySelectionError, match="stale_selection"):
        admit(db, cid, selection)
    assert len(snapshot(db, cid).nodes) == 1


def test_selected_admission_rollback_protected_write_and_settlement_parent_edit(history_db, monkeypatch):
    db, cid = history_db
    selection = selected(db, cid)
    store = db.message_store
    original = store._write_history_authority
    def fail(*args, **kwargs):
        raise RuntimeError("protected write failed")
    monkeypatch.setattr(store, "_write_history_authority", fail)
    with pytest.raises(RuntimeError, match="protected write failed"):
        admit(db, cid, selection)
    assert not snapshot(db, cid).nodes
    monkeypatch.setattr(store, "_write_history_authority", original)
    accepted = admit(db, cid, selection)
    db.update_message("accepted-user", {"content": "edited"}, expected_version=1)
    with pytest.raises(HistorySelectionError, match="stale_parent"):
        settle(db, cid, accepted)
    assert len(snapshot(db, cid).nodes) == 1


def test_settlement_survives_settings_but_rejects_workspace_scope_change(history_db):
    db, cid = history_db
    accepted = admit(db, cid, selected(db, cid))
    db.upsert_conversation_settings(cid, {"model": "changed"})
    settle(db, cid, accepted)
    db.upsert_workspace("other", "Other")
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET scope_type = 'workspace', workspace_id = 'other' WHERE id = ?", (cid,))
    with pytest.raises(HistorySelectionError, match="stale_scope"):
        settle(db, cid, accepted, "late")


def test_admitted_tool_metadata_and_assistant_retry_drift(history_db):
    db, cid = history_db
    accepted = admit(db, cid, selected(db, cid), extra_metadata={"sender_role": "user", "name": "Alice"})
    assert db.get_message_metadata("accepted-user")["extra"]["name"] == "Alice"
    settle(db, cid, accepted)
    db.update_message("accepted-assistant", {"content": "changed"}, expected_version=1)
    with pytest.raises(HistorySelectionError, match="message_id_conflict"):
        settle(db, cid, accepted)


def test_history_preview_unicode_and_image_only(history_db):
    db, cid = history_db
    text = "😸界" * 150
    mid = add(db, cid, text)
    image = add(db, cid, "", images=[{"data": b"one", "mime": "image/png"}])
    snap = snapshot_to_wire(snapshot(db, cid))
    assert snap["nodes"][0]["preview"] == text[:200]
    assert snap["nodes"][1]["preview"] == ""
    content = db.get_conversation_history_selected_content(cid, [mid, image], snapshot=snapshot(db, cid),
        owner_client_id="alice", owner_key=OWNER_KEY)
    assert content[0]["message"] == text


def test_history_sqlite_function_follows_real_connection_lifetime(history_db, tmp_path):
    db, cid = history_db
    if db.backend_type != BackendType.SQLITE:
        pytest.skip("SQLite connection lifetime")
    for _ in range(3):
        add(db, cid)
    other = CharactersRAGDB(db_path=str(tmp_path / "second.sqlite"), client_id="alice")
    other_cid = other.add_conversation({"character_id": 1, "title": "Second"})
    first_conn, second_conn = db.get_connection(), other.get_connection()
    snapshot(db, cid, conn=first_conn)
    cursor = first_conn.execute("SELECT h1_sha256(content) FROM messages")
    cursor.fetchone()
    try:
        snapshot(other, other_cid, conn=second_conn)
        assert len(snapshot(db, cid, conn=first_conn).nodes) == 3
        from tldw_Server_API.app.core.DB_Management.chacha.message_store import MessageStore
        assert len(MessageStore(db).get_conversation_history_snapshot(
            cid, owner_client_id="alice", owner_key=OWNER_KEY, conn=first_conn).nodes) == 3
    finally:
        cursor.close()
        other.close_all_connections()


def test_server_input_chain_atomic_acceptance_and_replay(history_db):
    db, cid = history_db
    selection = selected(db, cid)
    accepted = db.append_selected_history_inputs(cid, selection,
        [{"sender": "user", "content": "question"}, {"sender": "tool", "content": "evidence"}],
        owner_client_id="alice", owner_key=OWNER_KEY)
    final = db.get_message_by_id(accepted["input_message_id"])
    assert final["sender"] == "tool"
    assert final["parent_message_id"] is not None
    assert accepted["selection_digest"] == selection["selection_digest"]
    with pytest.raises(HistorySelectionError, match="selection_already_consumed"):
        db.append_selected_history_inputs(cid, selection, [{"sender": "tool", "content": "again"}],
            owner_client_id="alice", owner_key=OWNER_KEY)
    db.update_message(final["parent_message_id"], {"content": "earlier changed"}, expected_version=1)
    with pytest.raises(HistorySelectionError, match="stale_parent"):
        settle(db, cid, accepted)
    assert db.count_messages_for_conversation(cid) == 2


def test_server_input_chain_rolls_back_middle_failure(history_db, monkeypatch):
    db, cid = history_db
    selection = selected(db, cid)
    original = db.message_store.add_message
    calls = 0
    def fail_second(data, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("middle input")
        return original(data, **kwargs)
    monkeypatch.setattr(db.message_store, "add_message", fail_second)
    with pytest.raises(RuntimeError, match="middle input"):
        db.append_selected_history_inputs(cid, selection,
            [{"sender": "user", "content": "first"}, {"sender": "tool", "content": "second"}],
            owner_client_id="alice", owner_key=OWNER_KEY)
    assert not snapshot(db, cid).nodes


def test_simultaneous_server_completion_selection_consumed_once(history_db):
    db, cid = history_db
    selection = selected(db, cid)
    barrier = threading.Barrier(2)
    def run():
        barrier.wait(timeout=10)
        try:
            return db.append_selected_history_inputs(cid, selection, [{"sender": "tool", "content": "same turn"}],
                owner_client_id="alice", owner_key=OWNER_KEY)
        except HistorySelectionError as exc:
            return exc.code
        finally:
            db.close_connection()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert "selection_already_consumed" in results
    assert db.count_messages_for_conversation(cid) == 1


def test_admission_waits_for_actual_edit_then_rejects_retained_drift(history_db):
    db, cid = history_db
    mid = add(db, cid)
    selection = selected(db, cid, {"kind": "after_message", "message_id": mid})
    editing, release = threading.Event(), threading.Event()
    def edit():
        try:
            with db.transaction() as conn:
                db.update_message(mid, {"content": "racing edit"}, expected_version=1, conn=conn)
                editing.set()
                assert release.wait(timeout=10)
        finally:
            db.close_connection()
    with ThreadPoolExecutor(max_workers=2) as pool:
        edit_future = pool.submit(edit)
        assert editing.wait(timeout=10)
        admission_future = pool.submit(admit, db, cid, selection)
        release.set()
        edit_future.result(timeout=10)
        with pytest.raises(HistorySelectionError, match="stale_selection"):
            admission_future.result(timeout=10)
    assert db.count_messages_for_conversation(cid) == 1


def test_settlement_retry_rejects_metadata_drift_without_row_version(history_db):
    db, cid = history_db
    accepted = admit(db, cid, selected(db, cid))
    settle(db, cid, accepted, extra_metadata={"sender_role": "assistant"})
    db.add_message_metadata("accepted-assistant", extra={"sender_role": "assistant", "changed": True})
    with pytest.raises(HistorySelectionError, match="message_id_conflict"):
        settle(db, cid, accepted, extra_metadata={"sender_role": "assistant"})


def test_settlement_rejects_input_metadata_drift_without_row_version(history_db):
    db, cid = history_db
    accepted = admit(db, cid, selected(db, cid), extra_metadata={"sender_role": "user"})
    db.add_message_metadata("accepted-user", extra={"sender_role": "user", "changed": True})
    with pytest.raises(HistorySelectionError, match="stale_parent"):
        settle(db, cid, accepted)
    assert db.count_messages_for_conversation(cid) == 1


def test_skill_visibility_read_does_not_create_registry_and_detects_stale_eligible_rows(history_db):
    db, _ = history_db
    with db.transaction() as conn:
        existed = db.backend.table_exists("skill_registry", connection=conn)
    assert db.history_skills_may_be_visible() is False
    with db.transaction() as conn:
        assert db.backend.table_exists("skill_registry", connection=conn) is existed
    db._ensure_skill_registry_table()
    with db.transaction() as conn:
        conn.execute("INSERT INTO skill_registry (name, directory_path, uuid, user_invocable, disable_model_invocation) "
                     "VALUES (?, ?, ?, TRUE, TRUE)", ("disabled", "/missing", "disabled-id"))
    assert db.history_skills_may_be_visible() is False
    with db.transaction() as conn:
        conn.execute("UPDATE skill_registry SET disable_model_invocation = FALSE WHERE name = ?", ("disabled",))
    assert db.history_skills_may_be_visible() is True


def legacy_selection(db, cid, *, before_first=False):
    from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection

    for mid in ("legacy-a", "legacy-b"):
        add(db, cid, mid, id=mid, parent_message_id=None)
    original = snapshot(db, cid)
    confirm(db, cid, confirm_body(original, "accepted-legacy", ["legacy-a", "legacy-b"]))
    current = snapshot_to_wire(snapshot(db, cid, projection_id="accepted-legacy"))
    cursor = (
        {"kind": "before_message", "message_id": "legacy-a"}
        if before_first
        else {"kind": "after_message", "message_id": "legacy-b"}
    )
    return resolve_history_selection(
        current,
        {
            "owner_key": OWNER_KEY,
            "conversation_id": cid,
            "interpretation": {"kind": "legacy_linear_v1", "projection_id": "accepted-legacy"},
            "cursor": cursor,
            "selection_revision": 1,
        },
        "send",
        "legacy-retry",
    )["selection"]


@pytest.mark.parametrize("before_first", [False, True])
def test_legacy_result_retry_has_stable_substantive_state(history_db, before_first):
    db, cid = history_db
    accepted = admit(db, cid, legacy_selection(db, cid, before_first=before_first))
    assert db.get_message_by_id("accepted-user")["parent_message_id"] == (None if before_first else "legacy-b")
    assert settle(db, cid, accepted) == settle(db, cid, accepted)


@pytest.mark.parametrize("drift", ["edit", "delete"])
def test_legacy_settlement_ignores_old_source_drift_but_checks_accepted_chain(history_db, drift):
    db, cid = history_db
    accepted = db.append_selected_history_inputs(
        cid,
        legacy_selection(db, cid),
        [{"sender": "user", "content": "first"}, {"sender": "tool", "content": "last"}],
        owner_client_id="alice",
        owner_key=OWNER_KEY,
    )
    first = db.get_message_by_id(accepted["input_message_id"])["parent_message_id"]
    with db.transaction() as conn:
        if drift == "edit":
            conn.execute(
                "UPDATE messages SET content = ?, version = version + 1 WHERE id = ?",
                ("edited old source", "legacy-a"),
            )
        else:
            conn.execute("UPDATE messages SET deleted = TRUE WHERE id = ?", ("legacy-a",))
    assert settle(db, cid, accepted) == settle(db, cid, accepted)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?", ("edited accepted input without version bump", first)
        )
    with pytest.raises(HistorySelectionError, match="stale_parent"):
        settle(db, cid, accepted, mid="must-not-write")
    assert db.get_message_by_id("must-not-write") is None


def test_native_plain_fork_context_is_bound_to_same_statement(history_db):
    db, _ = history_db
    cid = db.add_conversation({"title": "Plain", "character_id": None})
    plain = snapshot_to_wire(snapshot(db, cid))
    assert plain["native_fork_context"] == {
        "policy": "plain_v1", "supported": True,
        "storage_context_digest": plain["storage_context_digest"],
    }
    db.upsert_conversation_settings(cid, {})
    empty = snapshot_to_wire(snapshot(db, cid))
    assert empty["native_fork_context"]["supported"] is True
    assert empty["storage_context_digest"] != plain["storage_context_digest"]
    with db.transaction() as conn:
        conn.execute("UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?", ('{"system_prompt":"required"}', cid))
    required = snapshot_to_wire(snapshot(db, cid))
    assert required["native_fork_context"]["supported"] is False
    assert required["storage_context_digest"] != empty["storage_context_digest"]


@pytest.mark.parametrize("stored", ["null", "[]", "", "broken", '{"unknown":null}'])
def test_native_plain_fork_rejects_unreadable_or_required_settings(history_db, stored):
    db, _ = history_db
    cid = db.add_conversation({"title": "Plain", "character_id": None})
    db.upsert_conversation_settings(cid, {})
    with db.transaction() as conn:
        conn.execute("UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?", (stored, cid))
    assert snapshot_to_wire(snapshot(db, cid))["native_fork_context"]["supported"] is False


def test_native_plain_fork_rejects_required_identity_and_behavior(history_db):
    db, character = history_db
    assert snapshot_to_wire(snapshot(db, character))["native_fork_context"]["supported"] is False
    cid = db.add_conversation({"title": "Plain", "character_id": None})
    before = snapshot(db, cid)
    from tldw_Server_API.tests.DB_Management.test_character_behavior_snapshot_migration import _snapshot
    with db.transaction() as conn:
        db.conversation_resume_store.put_behavior_snapshot(cid, _snapshot(), conn=conn)
    after = snapshot_to_wire(snapshot(db, cid))
    assert after["native_fork_context"]["supported"] is False
    assert after["storage_context_digest"] != before.storage_context_digest
