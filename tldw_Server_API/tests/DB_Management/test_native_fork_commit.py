"""Lossless private native fork commit and durable original-key recovery."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeForkCaptureRequestV1, NativeForkRequestV1
from tldw_Server_API.app.core.Chat.native_fork_projection import capture_native_fork, native_fork_request_digest
from tldw_Server_API.app.core.DB_Management.chacha.native_fork_store import NativeAttempt, NativeForkStoreError
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import native_fork_db as native_fork_db
from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import owner

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)


def prepare(db, *, tools=False, empty=False):
    source = db.add_conversation({"title": "Original", "character_id": None})
    mids = []
    parent = None
    if not empty:
        for role, text in [("user", "Question"), ("assistant", "Stopped answer")]:
            mid = db.add_message(
                {"conversation_id": source, "sender": role, "content": text, "parent_message_id": parent}
            )
            mids.append(mid)
            parent = mid
        assert db.add_message_metadata(
            mids[1],
            extra={
                "sender_role": "assistant",
                "generation_status": "interrupted",
                "model_id": "local-model",
                "provider": "custom",
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        )
        if tools:
            assert db.add_message_metadata(
                mids[1],
                tool_calls=[
                    {
                        "id": "original-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"term":"retained"}'},
                    }
                ],
                extra={"sender_role": "assistant"},
            )
            mid = db.add_message(
                {
                    "conversation_id": source,
                    "sender": "tool",
                    "content": "[empty tool result]",
                    "parent_message_id": parent,
                }
            )
            mids.append(mid)
            assert db.add_message_metadata(
                mid,
                extra={
                    "sender_role": "tool",
                    "tool_call_id": "original-call",
                    "content_placeholder_reason": "empty_tool_result",
                },
            )
    assert db.upsert_conversation_settings(
        source,
        {
            "provider": "custom",
            "model": "local-model",
            "pinnedMessageIds": mids[:1],
            "summary": {"enabled": True, "thresholdMessages": 20, "windowMessages": 5, "content": "excluded summary"},
        },
    )
    view = {
        "view_session_id": "native-test",
        "owner_key": owner().owner_key,
        "conversation_id": source,
        "interpretation": {"kind": "parent_graph_v1"},
        "cursor": {"kind": "after_message", "message_id": mids[-1]} if mids else {"kind": "empty"},
        "selection_revision": 0,
    }
    captured = capture_native_fork(db, owner(), NativeForkCaptureRequestV1(view=view))
    request, attempt = accept_review(db, captured, owner())
    return source, mids, request, attempt


def accept_review(db, captured, authorized):
    data = captured.model_dump(mode="json", exclude={"required_effects", "reasons"})
    data.update(operation_id=str(uuid4()), child_title="Independent child", request_digest="0" * 64)
    request = NativeForkRequestV1.model_validate(data)
    request = NativeForkRequestV1.model_validate({**data, "request_digest": native_fork_request_digest(request)})
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            authorized,
            "native_fork_v1",
            request.operation_id,
            request.request_digest,
            request.model_dump(mode="json"),
            conn=conn,
        )
        attempt = db.native_forks.claim_attempt(
            authorized, "native_fork_v1", request.operation_id, request.request_digest, now=NOW, conn=conn
        )
    assert isinstance(attempt, NativeAttempt)
    return request, attempt


def commit(db, request, attempt, *, now=NOW):
    with db.transaction() as conn:
        return db.native_forks.commit_fork(owner(), request, attempt, now=now, conn=conn)


def children(db, source):
    return db.execute_query("SELECT id FROM conversations WHERE id <> ?", (source,), read_only=True).fetchall()


def test_neutral_commit_retains_generation_settings_and_independent_graph(native_fork_db):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    result = commit(db, request, attempt)
    receipt = result.receipt
    assert result.state == "committed" and set(receipt.message_map) == set(mids)
    assert not set(receipt.message_map.values()) & set(mids)
    child = db.get_conversation_by_id(receipt.child_id)
    assert child["root_id"] == receipt.child_id and child["parent_conversation_id"] is None
    assert child["character_id"] is None and child["assistant_kind"] is None
    assert child["required_projection_version"] == "native-fork-v1"
    settings = db.get_conversation_settings(receipt.child_id)["settings"]
    assert settings["pinnedMessageIds"] == [receipt.message_map[mids[0]]]
    assert settings["summary"] == {"enabled": True, "thresholdMessages": 20, "windowMessages": 5}
    rows = db.get_messages_for_conversation(receipt.child_id)
    assert [x["content"] for x in rows] == ["Question", "Stopped answer"]
    assert [x["parent_message_id"] for x in rows] == [None, receipt.message_map[mids[0]]]
    extra = db.get_message_metadata(receipt.message_map[mids[1]])["extra"]
    assert extra["generation_status"] == "interrupted" and extra["usage"]["total_tokens"] == 7
    snapshot = db.get_conversation_history_snapshot(
        receipt.child_id, owner_client_id="alice", owner_key=owner().owner_key
    )
    assert snapshot.interpretation_status["kind"] == "parent_graph_v1"
    assert all(row["settled"] for row in snapshot.nodes)


def test_same_key_recovers_exact_immutable_map_after_source_deletion(native_fork_db, monkeypatch):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    original = commit(db, request, attempt)
    assert db.hard_delete_conversation(source)
    monkeypatch.setattr(db.message_store, "read_native_fork_source", lambda *a, **k: pytest.fail("replay read source"))
    replay = commit(db, request, attempt, now=NOW + timedelta(days=1))
    assert replay == original
    with pytest.raises(TypeError):
        replay.receipt.message_map[mids[0]] = "other"
    assert len(children(db, source)) == 1


def test_closed_tools_remap_ids_and_preserve_empty_result(native_fork_db):
    db = native_fork_db
    source, mids, request, attempt = prepare(db, tools=True)
    result = commit(db, request, attempt)
    call = db.get_message_metadata(result.receipt.message_map[mids[1]])
    answer = db.get_message_metadata(result.receipt.message_map[mids[2]])
    call_id = call["tool_calls"][0]["id"]
    assert call_id != "original-call" and answer["extra"]["tool_call_id"] == call_id
    assert call["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"term":"retained"}'}
    assert answer["extra"]["content_placeholder_reason"] == "empty_tool_result"
    rows = db.get_messages_for_conversation(result.receipt.child_id)
    assert rows[-1]["content"] == "[empty tool result]"


def test_empty_selection_creates_ready_neutral_child_without_greeting(native_fork_db):
    db = native_fork_db
    source, mids, request, attempt = prepare(db, empty=True)
    result = commit(db, request, attempt)
    assert dict(result.receipt.message_map) == {} and db.get_messages_for_conversation(result.receipt.child_id) == []
    row = db.execute_query(
        "SELECT native_bundle_json FROM conversations WHERE id = ?", (result.receipt.child_id,), read_only=True
    ).fetchone()
    assert json.loads(row["native_bundle_json"])["state"] == "ready"


@pytest.mark.parametrize(
    "writer", ["add_conversation", "add_message", "add_message_metadata", "upsert_conversation_settings"]
)
def test_failed_writer_rolls_back_every_child_row_and_receipt(native_fork_db, monkeypatch, writer):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    original = getattr(db, writer)

    def fail_after_write(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(db, writer, fail_after_write)
    with pytest.raises(RuntimeError, match="injected write failure"):
        commit(db, request, attempt)
    assert children(db, source) == []
    with db.transaction() as conn:
        result = db.native_forks.resolve_operation(
            owner(), "native_fork_v1", request.operation_id, request.request_digest, conn=conn
        )
    assert result.state == "preparing" and result.receipt is None


@pytest.mark.parametrize("corruption", ["empty", "owner", "operation", "child", "map", "request"])
def test_corrupt_committed_receipt_refuses_without_allocating(native_fork_db, corruption):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    child = db.add_conversation({"title": "Committed child", "character_id": None})
    receipt = {
        "protocol": "native_atomic_v1",
        "projection_version": "native-fork-v1",
        "operation_kind": "native_fork_v1",
        "state": "committed",
        "operation_id": request.operation_id,
        "owner_key": owner().owner_key,
        "child_id": child,
        "message_map": {mid: str(uuid4()) for mid in mids},
    }
    if corruption == "empty":
        receipt = {}
    elif corruption == "owner":
        receipt["owner_key"] = "native:other"
    elif corruption == "operation":
        receipt["operation_id"] = str(uuid4())
    elif corruption == "child":
        receipt["child_id"] = source
    elif corruption == "map":
        receipt["message_map"] = {"omitted": str(uuid4())}
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET required_projection_version = 'native-fork-v1', native_creation_operation_kind = 'native_fork_v1', native_creation_operation_id = ? WHERE id = ?",
            (request.operation_id, child),
        )
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ?, result_json = ? WHERE operation_id = ? AND client_id = 'alice'",
            (child, json.dumps(receipt), request.operation_id),
        )
        if corruption == "request":
            conn.execute(
                "UPDATE native_chat_operations SET canonical_request_json = '{}' WHERE operation_id = ? AND client_id = 'alice'",
                (request.operation_id,),
            )
    with db.transaction() as conn:
        with pytest.raises(NativeForkStoreError, match="operation_receipt_invalid"):
            db.native_forks.resolve_operation(
                owner(), "native_fork_v1", request.operation_id, request.request_digest, conn=conn
            )
    assert len(children(db, source)) == 1


@pytest.mark.parametrize("change", ["message", "settings"])
def test_retained_source_change_refuses_before_child_allocation(native_fork_db, change):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    if change == "message":
        assert db.update_message(mids[1], {"content": "Changed answer"}, expected_version=1)
    else:
        assert db.upsert_conversation_settings(source, {"provider": "custom", "model": "changed-model"})
    with pytest.raises(NativeForkStoreError, match="stale_retained_source"):
        commit(db, request, attempt)
    assert children(db, source) == []


def test_excluded_summary_and_same_text_storage_revision_do_not_invalidate_retained_request(native_fork_db):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    settings = db.get_conversation_settings(source)["settings"]
    settings["summary"]["content"] = "Another excluded summary"
    assert db.upsert_conversation_settings(source, settings)
    assert db.update_message(mids[0], {"content": "Question"}, expected_version=1)
    result = commit(db, request, attempt)
    assert result.state == "committed" and len(result.receipt.message_map) == 2


@pytest.mark.parametrize("carrier", ["claim", "reference", "bundle", "projection", "image", "candidate"])
def test_unsupported_carriers_cannot_be_mistaken_for_empty_manifest(native_fork_db, carrier):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    with db.transaction() as conn:
        if carrier in {"claim", "reference"}:
            conn.execute(
                "INSERT INTO native_chat_asset_claims (client_id, claim_id, owner_key, conversation_id, asset_id, asset_revision, content_hash, size_bytes, mime_type, representation, state, created_at, updated_at) VALUES ('alice', 'claim-one', 'native:alice', ?, 'asset-one', 1, ?, 1, 'image/png', 'file_v1', 'live', ?, ?)",
                (source, "a" * 64, NOW.isoformat(), NOW.isoformat()),
            )
            if carrier == "reference":
                conn.execute(
                    "UPDATE native_chat_asset_claims SET conversation_id = 'other' WHERE claim_id = 'claim-one'"
                )
                conn.execute(
                    "INSERT INTO native_chat_asset_references (client_id, reference_id, owner_key, conversation_id, claim_id, position, revision, context_enabled, created_at, updated_at) VALUES ('alice', 'ref-one', 'native:alice', ?, 'claim-one', 0, 1, ?, ?, ?)",
                    (source, True, NOW.isoformat(), NOW.isoformat()),
                )
        elif carrier in {"bundle", "projection"}:
            column = "native_bundle_json" if carrier == "bundle" else "required_projection_version"
            conn.execute(
                f"UPDATE conversations SET {column} = ? WHERE id = ?",
                ("{}" if carrier == "bundle" else "native-fork-v1", source),
            )
        elif carrier == "image":
            conn.execute(
                "UPDATE messages SET image_data = ?, image_mime_type = 'image/png' WHERE id = ?",
                (b"owned image", mids[0]),
            )
        else:
            db.native_assets.reserve_candidate(
                owner(),
                "native_fork_v1",
                request.operation_id,
                request.request_digest,
                attempt.generation,
                "candidate-one",
                "namespace",
                "upload",
                "a" * 64,
                1,
                "image/png",
                "file_v1",
                now=NOW,
                conn=conn,
            )
    expected = "reconciliation_required" if carrier == "candidate" else "unsupported_native_fork"
    with pytest.raises(NativeForkStoreError, match=expected):
        commit(db, request, attempt)
    assert children(db, source) == []


@pytest.mark.parametrize("mode", ["expired", "replaced", "wrong_lease", "boolean_generation"])
def test_stale_attempt_cannot_publish_a_child(native_fork_db, mode):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    now = NOW + timedelta(seconds=301) if mode in {"expired", "replaced"} else NOW
    if mode == "replaced":
        with db.transaction() as conn:
            db.native_forks.claim_attempt(
                owner(), "native_fork_v1", request.operation_id, request.request_digest, now=now, conn=conn
            )
    elif mode == "wrong_lease":
        attempt = NativeAttempt(attempt.generation, (NOW + timedelta(seconds=100)).isoformat())
    elif mode == "boolean_generation":
        attempt = NativeAttempt(True, attempt.lease_expires_at)
    with pytest.raises(NativeForkStoreError, match="stale_attempt"):
        commit(db, request, attempt, now=now)
    assert children(db, source) == []


def test_commit_requires_active_caller_transaction(native_fork_db):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    with pytest.raises(NativeForkStoreError, match="active_transaction_required"):
        db.native_forks.commit_fork(owner(), request, attempt, now=NOW, conn=db.get_connection())
    assert children(db, source) == []


@pytest.mark.parametrize("boundary", ["provenance", "protection", "receipt"])
def test_failed_final_sql_boundary_rolls_back_graph_and_operation(native_fork_db, monkeypatch, boundary):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    writer = "add_message_metadata" if boundary == "provenance" else "upsert_conversation_settings"
    original = getattr(db, writer)

    def interfere(identity, *args, **kwargs):
        result = original(identity, *args, **kwargs)
        conn = kwargs["conn"]
        if boundary == "provenance":
            conn.execute("DELETE FROM messages WHERE id = ?", (identity,))
        elif boundary == "protection":
            conn.execute("UPDATE conversations SET client_id = 'other' WHERE id = ?", (identity,))
        else:
            conn.execute(
                "UPDATE native_chat_operations SET state = 'expired' WHERE operation_id = ?", (request.operation_id,)
            )
        return result

    monkeypatch.setattr(db, writer, interfere)
    expected = {
        "provenance": "provenance_write_failed",
        "protection": "child_protection_write_failed",
        "receipt": "operation_commit_conflict",
    }[boundary]
    with pytest.raises(NativeForkStoreError, match=expected):
        commit(db, request, attempt)
    assert children(db, source) == []
    with db.transaction() as conn:
        assert (
            db.native_forks.resolve_operation(
                owner(), "native_fork_v1", request.operation_id, request.request_digest, conn=conn
            ).state
            == "preparing"
        )


@pytest.mark.parametrize("hard", [False, True])
def test_child_deletion_burns_original_key_even_after_restore(native_fork_db, hard):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    result = commit(db, request, attempt)
    child = result.receipt.child_id
    if hard:
        assert db.hard_delete_conversation(child)
    else:
        version = db.get_conversation_by_id(child)["version"]
        assert db.soft_delete_conversation(child, version)
        assert db.restore_conversation(child, version + 1)
    replay = commit(db, request, attempt, now=NOW + timedelta(days=1))
    assert replay.state == "gone" and replay.receipt is None
    assert len(children(db, source)) == (0 if hard else 1)


def test_fresh_child_provenance_contains_no_source_admission(native_fork_db):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    with db.transaction() as conn:
        for mid in mids:
            conn.execute(
                "UPDATE messages SET history_admission_json = ? WHERE id = ?",
                (
                    json.dumps(
                        {
                            "version": 1,
                            "settled": True,
                            "interpretation": {"kind": "parent_graph_v1"},
                            "source_execution_authority": "must not copy",
                        }
                    ),
                    mid,
                ),
            )
    result = commit(db, request, attempt)
    rows = db.execute_query(
        "SELECT history_admission_json FROM messages WHERE conversation_id = ?",
        (result.receipt.child_id,),
        read_only=True,
    ).fetchall()
    assert all(
        json.loads(row["history_admission_json"])["native_fork"]["conversation_id"] == result.receipt.child_id
        for row in rows
    )
    assert all("source_execution_authority" not in row["history_admission_json"] for row in rows)


@pytest.mark.parametrize("scoped", [False, True])
def test_postgres_guard_accepts_current_legacy_and_scoped_caller_transactions(monkeypatch, scoped):
    """Guard contract only; this does not replace official physical PG qualification."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.chacha.native_fork_store import NativeForkStore

    vector = Path(__file__).parents[1] / "fixtures" / "native_fork_v1.json"
    request = NativeForkRequestV1.model_validate(json.loads(vector.read_text())["vectors"][0]["request"])
    raw = object()
    state = SimpleNamespace(conn=raw, tx_depth=1)
    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL, client_id="alice", _connection_state=lambda: state)
    conn = SimpleNamespace(_db=db, _connection=raw, _operation_state=state if scoped else None)
    store = NativeForkStore(db)
    monkeypatch.setattr(store, "_row", lambda *args, **kwargs: None)
    result = store.commit_fork(owner(), request, NativeAttempt(1, NOW.isoformat()), now=NOW, conn=conn)
    assert result.state == "not_recorded"


@pytest.mark.parametrize(
    "column,value",
    [("native_creation_operation_kind", "native_fork_v1"), ("native_creation_operation_id", "original-operation")],
)
def test_each_native_creation_marker_refuses_before_allocation(native_fork_db, column, value):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    with db.transaction() as conn:
        conn.execute(f"UPDATE conversations SET {column} = ? WHERE id = ?", (value, source))
    with pytest.raises(NativeForkStoreError, match="unsupported_native_fork_source_bundle"):
        commit(db, request, attempt)
    assert children(db, source) == []
    with db.transaction() as conn:
        assert (
            db.native_forks.resolve_operation(
                owner(), "native_fork_v1", request.operation_id, request.request_digest, conn=conn
            ).state
            == "preparing"
        )


def test_closed_empty_assistant_keeps_literal_empty_content_and_typed_calls(native_fork_db):
    db = native_fork_db
    source, mids, _, _ = prepare(db, tools=True)
    with db.transaction() as conn:
        conn.execute("UPDATE messages SET content = '' WHERE id = ?", (mids[1],))
    captured = capture_native_fork(
        db,
        owner(),
        NativeForkCaptureRequestV1(
            view={
                "view_session_id": "empty-assistant",
                "conversation_id": source,
                "owner_key": owner().owner_key,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "after_message", "message_id": mids[-1]},
                "selection_revision": 0,
            }
        ),
    )
    request, attempt = accept_review(db, captured, owner())
    result = commit(db, request, attempt)
    rows = db.get_messages_for_conversation(result.receipt.child_id)
    assert (
        rows[1]["content"] == ""
        and db.get_message_metadata(rows[1]["id"])["tool_calls"][0]["function"]["name"] == "lookup"
    )


def test_accepted_character_binding_refuses_before_child_allocation(native_fork_db):
    from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import build_behavior_snapshot
    from tldw_Server_API.tests.Chat.unit.test_native_fork_projection import AcceptedState

    db = native_fork_db
    accepted = AcceptedState.build().state
    source = db.add_conversation({"title": "Accepted character", "character_id": 1})
    with db.transaction() as conn:
        db.conversation_resume_store.put_behavior_snapshot(
            source, build_behavior_snapshot(accepted["behavior_snapshot"]["payload"]), conn=conn
        )
        assert db.upsert_conversation_settings(source, accepted["settings"], conn=conn)
    captured = capture_native_fork(
        db,
        owner(),
        NativeForkCaptureRequestV1(
            view={
                "view_session_id": "character",
                "owner_key": owner().owner_key,
                "conversation_id": source,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "empty"},
                "selection_revision": 0,
            }
        ),
    )
    assert "accepted_character_v1" in captured.required_effects
    request, attempt = accept_review(db, captured, owner())
    with pytest.raises(NativeForkStoreError, match="unsupported_native_fork_character_binding"):
        commit(db, request, attempt)
    assert children(db, source) == []


def test_reviewed_legacy_order_becomes_independent_parent_graph(native_fork_db):
    from tldw_Server_API.tests.DB_Management.test_history_selection_transactions import confirm_body

    db = native_fork_db
    source, mids, _, _ = prepare(db)
    snapshot = db.get_conversation_history_snapshot(source, owner_client_id="alice", owner_key=owner().owner_key)
    body = confirm_body(snapshot, projection_id="reviewed-copy", path=list(reversed(mids)))
    db.confirm_legacy_history_projection(body, owner_client_id="alice", owner_key=owner().owner_key)
    captured = capture_native_fork(
        db,
        owner(),
        NativeForkCaptureRequestV1(
            view={
                "view_session_id": "reviewed",
                "owner_key": owner().owner_key,
                "conversation_id": source,
                "interpretation": {"kind": "legacy_linear_v1", "projection_id": "reviewed-copy"},
                "cursor": {"kind": "after_message", "message_id": mids[0]},
                "selection_revision": 1,
            }
        ),
    )
    request, attempt = accept_review(db, captured, owner())
    result = commit(db, request, attempt)
    rows = db.get_messages_for_conversation(result.receipt.child_id)
    assert [row["content"] for row in rows] == ["Stopped answer", "Question"]
    assert [row["parent_message_id"] for row in rows] == [None, rows[0]["id"]]
    snapshot = db.get_conversation_history_snapshot(
        result.receipt.child_id, owner_client_id="alice", owner_key=owner().owner_key
    )
    assert snapshot.interpretation_status["kind"] == "parent_graph_v1"


@pytest.mark.parametrize("committed", [False, True])
def test_workspace_closure_blocks_commit_or_receipt_disclosure(native_fork_db, committed):
    from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import workspace_owner

    db = native_fork_db
    authorized = workspace_owner()
    db.upsert_workspace(authorized.scope.workspace_id, "Owned workspace")
    source = db.add_conversation(
        {
            "title": "Workspace source",
            "character_id": None,
            "scope_type": "workspace",
            "workspace_id": authorized.scope.workspace_id,
        }
    )
    captured = capture_native_fork(
        db,
        authorized,
        NativeForkCaptureRequestV1(
            view={
                "view_session_id": "workspace",
                "owner_key": authorized.owner_key,
                "conversation_id": source,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "empty"},
                "selection_revision": 0,
            }
        ),
    )
    request, attempt = accept_review(db, captured, authorized)
    if committed:
        with db.transaction() as conn:
            assert db.native_forks.commit_fork(authorized, request, attempt, now=NOW, conn=conn).state == "committed"
    with db.transaction() as conn:
        conn.execute(
            "UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, authorized.scope.workspace_id)
        )
    with db.transaction() as conn:
        with pytest.raises(NativeForkStoreError, match="workspace_native_unavailable"):
            db.native_forks.commit_fork(authorized, request, attempt, now=NOW, conn=conn)
    state = db.execute_query(
        "SELECT state FROM native_chat_operations WHERE operation_id = ?", (request.operation_id,), read_only=True
    ).fetchone()["state"]
    assert state == ("committed" if committed else "preparing")
    assert len(children(db, source)) == (1 if committed else 0)


def test_concurrent_same_key_commits_one_child_and_original_map(native_fork_db):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    barrier = Barrier(2)

    def worker():
        barrier.wait(timeout=20)
        try:
            return commit(db, request, attempt)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert results[0] == results[1] and results[0].state == "committed"
    assert len(children(db, source)) == 1


def test_selected_history_is_not_truncated_to_list_page_size(native_fork_db):
    db = native_fork_db
    source = db.add_conversation({"title": "Long reviewed path", "character_id": None})
    mids = []
    with db.transaction() as conn:
        parent = None
        for index in range(205):
            mid = db.add_message(
                {
                    "conversation_id": source,
                    "sender": "user",
                    "content": f"Retained {index}",
                    "parent_message_id": parent,
                },
                conn=conn,
            )
            mids.append(mid)
            parent = mid
    captured = capture_native_fork(
        db,
        owner(),
        NativeForkCaptureRequestV1(
            view={
                "view_session_id": "long",
                "owner_key": owner().owner_key,
                "conversation_id": source,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "after_message", "message_id": mids[-1]},
                "selection_revision": 0,
            }
        ),
    )
    request, attempt = accept_review(db, captured, owner())
    result = commit(db, request, attempt)
    rows = db.get_messages_for_conversation(result.receipt.child_id, limit=300)
    assert list(result.receipt.message_map) == mids and len(rows) == 205
    assert rows[-1]["content"] == "Retained 204"


def test_explicit_asset_manifest_refuses_without_source_lookup(native_fork_db, monkeypatch):
    db = native_fork_db
    source = db.add_conversation({"title": "Image source", "character_id": None})
    mid = db.add_message(
        {
            "conversation_id": source,
            "sender": "user",
            "content": "Owned image",
            "images": [{"data": b"image bytes", "mime": "image/png"}],
        }
    )
    captured = capture_native_fork(
        db,
        owner(),
        NativeForkCaptureRequestV1(
            view={
                "view_session_id": "image",
                "owner_key": owner().owner_key,
                "conversation_id": source,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "after_message", "message_id": mid},
                "selection_revision": 0,
            }
        ),
    )
    request, attempt = accept_review(db, captured, owner())
    monkeypatch.setattr(
        db.message_store, "read_native_fork_source", lambda *a, **k: pytest.fail("unsupported asset read source")
    )
    with pytest.raises(NativeForkStoreError, match="unsupported_native_fork_assets"):
        commit(db, request, attempt)
    assert children(db, source) == []


@pytest.mark.parametrize("change", ["digest", "owner", "unrecorded", "expired_key"])
def test_original_operation_identity_and_terminal_keys_do_not_create_children(native_fork_db, change, monkeypatch):
    db = native_fork_db
    source, mids, request, attempt = prepare(db)
    authorized = owner("native:other") if change == "owner" else owner()
    if change == "digest":
        request = NativeForkRequestV1.model_validate(
            {**request.model_dump(mode="json"), "child_title": "Changed title"}
        )
    elif change == "unrecorded":
        data = {**request.model_dump(mode="json"), "operation_id": str(uuid4())}
        request = NativeForkRequestV1.model_validate(data)
        request = NativeForkRequestV1.model_validate({**data, "request_digest": native_fork_request_digest(request)})
    elif change == "expired_key":
        with db.transaction() as conn:
            conn.execute(
                "UPDATE native_chat_operations SET state = 'expired' WHERE operation_id = ?", (request.operation_id,)
            )
    monkeypatch.setattr(
        db.message_store, "read_native_fork_source", lambda *a, **k: pytest.fail("operation boundary read source")
    )
    with db.transaction() as conn:
        if change in {"digest", "owner"}:
            expected = "operation_id_conflict" if change == "digest" else "operation_owner_mismatch"
            with pytest.raises(NativeForkStoreError, match=expected):
                db.native_forks.commit_fork(authorized, request, attempt, now=NOW, conn=conn)
        else:
            result = db.native_forks.commit_fork(authorized, request, attempt, now=NOW, conn=conn)
            assert result.state == ("not_recorded" if change == "unrecorded" else "expired") and result.receipt is None
    assert children(db, source) == []


def test_edited_empty_tool_placeholder_is_retained_as_literal_text(native_fork_db):
    db = native_fork_db
    source, mids, _, _ = prepare(db, tools=True)
    assert db.update_message(mids[-1], {"content": "[empty tool result]"}, expected_version=1)
    captured = capture_native_fork(
        db,
        owner(),
        NativeForkCaptureRequestV1(
            view={
                "view_session_id": "literal-tool",
                "owner_key": owner().owner_key,
                "conversation_id": source,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "after_message", "message_id": mids[-1]},
                "selection_revision": 0,
            }
        ),
    )
    request, attempt = accept_review(db, captured, owner())
    result = commit(db, request, attempt)
    meta = db.get_message_metadata(result.receipt.message_map[mids[-1]])
    assert "content_placeholder_reason" not in meta["extra"]
    assert db.get_messages_for_conversation(result.receipt.child_id)[-1]["content"] == "[empty tool result]"


def test_actual_empty_committed_fork_has_original_empty_context(native_fork_db):
    from tldw_Server_API.tests.DB_Management.test_native_asset_context_read import read

    db = native_fork_db
    _, _, request, attempt = prepare(db, empty=True)
    result = commit(db, request, attempt)
    context = read(db, result.child_conversation_id, authorized=owner())
    assert context.asset_manifest_revision == "1" and context.asset_manifest == ()


def test_actual_deleted_then_restored_fork_reads_without_reviving_original_key(native_fork_db):
    from tldw_Server_API.tests.DB_Management.test_native_asset_context_read import read

    db = native_fork_db
    _, _, request, attempt = prepare(db, empty=True)
    result = commit(db, request, attempt)
    cid = result.child_conversation_id
    version = db.get_conversation_by_id(cid)["version"]
    assert db.soft_delete_conversation(cid, version)
    with pytest.raises(ValueError):
        read(db, cid, authorized=owner())
    assert db.restore_conversation(cid, version + 1)
    before = dict(
        db.execute_query(
            "SELECT * FROM native_chat_operations WHERE operation_id = ?", (request.operation_id,)
        ).fetchone()
    )
    assert before["state"] == "gone" and before["result_json"] is None
    context = read(db, cid, authorized=owner())
    assert context.asset_manifest == () and context.asset_manifest_revision == "1"
    assert (
        dict(
            db.execute_query(
                "SELECT * FROM native_chat_operations WHERE operation_id = ?", (request.operation_id,)
            ).fetchone()
        )
        == before
    )
