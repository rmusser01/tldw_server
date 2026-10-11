"""Durable user-turn contract tests against real conversation storage."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Barrier
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.chat_service import build_context_and_messages
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


def turn_request(cid, uid, **overrides):
    """Build the public opt-in request, not an internal test-only representation."""
    return ChatCompletionRequest.model_validate(
        {
            "model": "local-model",
            "conversation_id": cid,
            "save_to_db": True,
            "tldw_turn": {"user_message_id": uid},
            "messages": [{"role": "user", "content": "Original question"}],
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"conversation_id": None},
        {"save_to_db": False},
        {"save_to_db": None},
        {"history_message_limit": 0},
        {"tldw_turn": {"user_message_id": "not-a-uuid"}},
        {"tldw_turn": {"user_message_id": str(uuid4()), "unknown": True}},
        {"tldw_continuation": {"from_message_id": str(uuid4()), "mode": "branch"}},
        {"messages": [{"role": "system", "content": "context only"}]},
        {"messages": [{"role": "user", "content": "old"}, {"role": "user", "content": "new"}]},
        {"messages": [{"role": "assistant", "content": "old"}, {"role": "user", "content": "new"}]},
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}
            ]
        },
    ],
)
def test_durable_request_rejects_unsupported_shapes(overrides):
    with pytest.raises(ValidationError):
        turn_request(str(uuid4()), str(uuid4()), **overrides)


def test_durable_request_normalizes_text_parts_without_trimming():
    request = turn_request(
        str(uuid4()),
        str(uuid4()),
        messages=[
            {"role": "system", "content": "request-local retrieval"},
            {"role": "user", "content": [{"type": "text", "text": " first "}, {"type": "text", "text": "second"}]},
        ],
    )
    assert request.messages[-1].content == " first \nsecond"




@pytest.fixture
def durable_conversation(populated_chacha_db):
    db = populated_chacha_db
    db.upsert_workspace("durable-workspace", "Durable workspace")
    return db.add_conversation(
        {
            "title": "Durable turn",
            "scope_type": "workspace",
            "workspace_id": "durable-workspace",
        }
    )


async def build_turn(db, conversation_id, user_message_id, **overrides):
    """Exercise the production context/persistence boundary before inference."""
    runtime = {}
    result = await build_context_and_messages(
        chat_db=db,
        request_data=turn_request(conversation_id, user_message_id, **overrides),
        loop=asyncio.get_running_loop(),
        metrics=get_chat_metrics(),
        default_save_to_db=False,
        final_conversation_id=conversation_id,
        save_message_fn=_save_message_turn_to_db,
        runtime_state=runtime,
    )
    return result, runtime


@pytest.mark.asyncio
async def test_durable_retry_rejects_independently_appended_image(populated_chacha_db, durable_conversation):
    db, cid, uid = populated_chacha_db, durable_conversation, str(uuid4())
    await build_turn(db, cid, uid)
    db.append_message_image(uid, b"independently appended image", "image/png")
    assert db.get_message_by_id(uid)["image_data"] is None
    before_rows = db.get_messages_for_conversation(cid)
    before_images = db.get_message_images(uid, strict=True)
    before_conversation = db.get_conversation_by_id(cid)
    with pytest.raises(HTTPException) as rejected:
        await build_turn(db, cid, uid)
    assert rejected.value.status_code == 409
    assert db.get_messages_for_conversation(cid) == before_rows
    assert db.get_message_images(uid, strict=True) == before_images
    assert db.get_conversation_by_id(cid) == before_conversation


def payload_text(message):
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(part["text"] for part in content)


@pytest.mark.asyncio
async def test_commit_before_lost_receipt_and_retries_preserve_one_user_and_legacy_history(
    populated_chacha_db,
    durable_conversation,
):
    db, cid, uid = populated_chacha_db, durable_conversation, str(uuid4())
    for sender, content in [("user", "Legacy question"), ("assistant", "Legacy answer")]:
        db.add_message({"conversation_id": cid, "sender": sender, "content": content})
    # Discard the entire first result, simulating post-commit/pre-ack disconnect.
    await build_turn(db, cid, uid)
    partial_id = db.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": "Partial attempt", "parent_message_id": uid}
    )
    partial_before = db.get_message_by_id(partial_id)
    for _ in range(3):
        result, runtime = await build_turn(
            db,
            cid,
            uid,
            messages=[
                {"role": "system", "content": "request-local retrieval"},
                {"role": "user", "content": [{"type": "text", "text": "Original question"}]},
            ],
        )
        assert [(msg["role"], payload_text(msg)) for msg in result[4]] == [
            ("system", "request-local retrieval"),
            ("user", "Legacy question"),
            ("assistant", "Legacy answer"),
            ("user", "Original question"),
        ]
        assert runtime["user_message_id"] == uid
        assert runtime["assistant_parent_message_id"] == uid
        assert "tldw_message_id" not in runtime
    assert db.count_messages_for_conversation(cid) == 4
    assert db.get_message_by_id(partial_id) == partial_before


@pytest.mark.asyncio
async def test_distinct_ids_allow_same_text_but_later_user_blocks_old_retry(populated_chacha_db, durable_conversation):
    db, cid = populated_chacha_db, durable_conversation
    old_id, new_id = str(uuid4()), str(uuid4())
    await build_turn(db, cid, old_id)
    await build_turn(db, cid, new_id)
    with pytest.raises(HTTPException) as error:
        await build_turn(db, cid, old_id)
    assert error.value.status_code == 409
    assert db.count_messages_for_conversation(cid) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["content", "role", "deleted", "foreign"])
async def test_identity_conflicts_do_not_mutate_rows(populated_chacha_db, durable_conversation, conflict):
    db, cid, uid = populated_chacha_db, durable_conversation, str(uuid4())
    target = db.add_conversation({"title": "Other"}) if conflict == "foreign" else cid
    db.add_message(
        {
            "id": uid,
            "conversation_id": target,
            "sender": "assistant" if conflict == "role" else "user",
            "content": "changed" if conflict == "content" else "Original question",
        }
    )
    if conflict == "deleted":
        db.soft_delete_message(uid, 1)
    before = db.get_messages_for_conversation(target, include_deleted=True)
    with pytest.raises(HTTPException) as error:
        await build_turn(db, cid, uid)
    assert error.value.status_code == 409
    assert db.get_messages_for_conversation(target, include_deleted=True) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("unauthorized", ["owner", "character", "missing", "workspace"])
async def test_scope_is_authorized_before_identity_lookup(populated_chacha_db, unauthorized):
    db, uid = populated_chacha_db, str(uuid4())
    cid = db.add_conversation({"title": "Scope", "client_id": "foreign" if unauthorized == "owner" else db.client_id})
    overrides = {}
    if unauthorized == "character":
        overrides["character_id"] = str(db.add_character_card({"name": "Other", "client_id": db.client_id}))
    if unauthorized == "missing":
        cid = str(uuid4())
    if unauthorized == "workspace":
        overrides.update(scope_type="workspace", workspace_id="foreign-workspace")
    with pytest.raises(HTTPException) as error:
        await build_turn(db, cid, uid, **overrides)
    assert error.value.status_code in (404, 409)
    assert db.get_message_by_id(uid) is None


@pytest.mark.asyncio
async def test_foreign_workspace_is_rejected_before_reusing_user(populated_chacha_db):
    db = populated_chacha_db
    foreign = CharactersRAGDB(db.db_path_str, client_id="other-owner")
    try:
        foreign.upsert_workspace("foreign-workspace", "Foreign")
    finally:
        foreign.close_all_connections()
    cid = db.add_conversation(
        {"title": "Wrong workspace", "scope_type": "workspace", "workspace_id": "foreign-workspace"}
    )
    uid = db.add_message({"conversation_id": cid, "sender": "user", "content": "Original question"})
    with pytest.raises(HTTPException) as error:
        await build_turn(db, cid, uid)
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_missing_stored_character_cannot_silently_fall_back(populated_chacha_db):
    db = populated_chacha_db
    character_id = db.add_character_card({"name": "Removed", "client_id": db.client_id})
    cid = db.add_conversation({"title": "Removed assistant", "character_id": character_id})
    db.soft_delete_character_card(character_id, 1)
    with pytest.raises(HTTPException) as error:
        await build_turn(db, cid, str(uuid4()))
    assert error.value.status_code in (404, 409)
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.asyncio
async def test_owned_persona_turn_reuses_identity_without_character_binding(populated_chacha_db):
    db = populated_chacha_db
    persona_id = str(uuid4())
    db.create_persona_profile(
        {
            "id": persona_id,
            "user_id": db.client_id,
            "name": "Durable persona",
            "mode": "session_scoped",
            "system_prompt": "Persona context",
            "is_active": True,
        }
    )
    cid = db.add_conversation({"title": "Persona", "assistant_kind": "persona", "assistant_id": persona_id})
    uid = str(uuid4())
    for _ in range(2):
        result, runtime = await build_turn(db, cid, uid)
        assert result[0]["system_prompt"] == "Persona context"
        assert runtime["assistant_context"]["assistant_id"] == persona_id
        assert runtime["assistant_parent_message_id"] == uid
    assert db.count_messages_for_conversation(cid) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (1, ["recent", "Original question"]),
        (2, ["older reply", "recent", "Original question"]),
        (20, ["older", "older reply", "recent", "Original question"]),
    ],
)
async def test_history_limit_counts_previous_messages_plus_anchor_in_chronological_order(
    populated_chacha_db, durable_conversation, limit, expected
):
    db, cid, uid = populated_chacha_db, durable_conversation, str(uuid4())
    db.add_message({"conversation_id": cid, "sender": "user", "content": "older"})
    db.add_message({"conversation_id": cid, "sender": "assistant", "content": "older reply"})
    db.add_message({"conversation_id": cid, "sender": "user", "content": "recent"})
    for _ in range(2):
        result, _ = await build_turn(db, cid, uid, history_message_limit=limit, history_message_order="desc")
        assert [payload_text(message) for message in result[4]] == expected


@pytest.mark.asyncio
async def test_later_user_role_metadata_blocks_retry(populated_chacha_db, durable_conversation):
    db, cid, uid = populated_chacha_db, durable_conversation, str(uuid4())
    await build_turn(db, cid, uid)
    later = db.add_message({"conversation_id": cid, "sender": "Legacy display name", "content": "Later question"})
    db.add_message_metadata(later, extra={"sender_role": "user"})
    with pytest.raises(HTTPException) as error:
        await build_turn(db, cid, uid)
    assert error.value.status_code == 409
    assert db.count_messages_for_conversation(cid) == 2


def test_concurrent_same_id_across_conversations_returns_conflict(populated_chacha_db, temp_db_path):
    db, uid = populated_chacha_db, str(uuid4())
    cids = [db.add_conversation({"title": f"Race {index}"}) for index in range(2)]
    peers = [CharactersRAGDB(str(temp_db_path), client_id=db.client_id) for _ in cids]
    contexts = [db.get_conversation_by_id(cid) for cid in cids]
    barrier = Barrier(2)

    def insert(index):
        peer = peers[index]
        barrier.wait(timeout=10)
        try:
            peer.insert_or_validate_user_turn(
                cids[index], uid, "Question", owner_client_id=db.client_id, conversation_context=contexts[index]
            )
            return "inserted"
        except ConflictError:
            return "conflict"
        finally:
            peer.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert sorted(executor.map(insert, range(2))) == ["conflict", "inserted"]
        assert sum(db.count_messages_for_conversation(cid) for cid in cids) == 1
    finally:
        for peer in peers:
            peer.close_all_connections()


def test_durable_timestamp_is_not_bumped_by_future_import(populated_chacha_db, durable_conversation):
    db, cid = populated_chacha_db, durable_conversation
    latest = datetime.fromisoformat("2099-12-31T23:59:59+00:00")
    db.add_message(
        {
            "conversation_id": cid,
            "sender": "user",
            "content": "Legacy timestamp",
            "timestamp": latest.isoformat(sep=" "),
        }
    )
    uid = str(uuid4())
    db.insert_or_validate_user_turn(
        cid, uid, "Question", owner_client_id=db.client_id, conversation_context=db.get_conversation_by_id(cid)
    )
    saved = db.get_message_by_id(uid)
    assert datetime.fromisoformat(saved["timestamp"].replace("Z", "+00:00")) < latest


@pytest.mark.parametrize("writer", ["ordinary", "sync", "sql"])
@pytest.mark.parametrize("timestamp_kind", ["natural", "equal", "backdated"])
@pytest.mark.parametrize("sender", ["assistant", "user"])
def test_retry_boundary_uses_insertion_order_for_every_writer(
    populated_chacha_db, durable_conversation, writer, timestamp_kind, sender
):
    db, cid = populated_chacha_db, durable_conversation
    prior = db.add_message(
        {"conversation_id": cid, "sender": "assistant", "content": "Future import", "timestamp": "2099-01-01T00:00:00.000Z"}
    )
    uid = "ffffffff-ffff-4fff-bfff-ffffffffffff"
    context = db.get_conversation_by_id(cid)
    history = db.insert_or_validate_user_turn(
        cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context
    )
    assert [row["id"] for row in history] == [prior, uid]
    timestamp = {
        "natural": None,
        "equal": db.get_message_by_id(uid)["timestamp"],
        "backdated": "1999-01-01T00:00:00Z",
    }[timestamp_kind]
    later_id = "00000000-0000-4000-8000-000000000001"
    if writer == "ordinary":
        db.add_message(
            {"id": later_id, "conversation_id": cid, "sender": sender, "content": "Later", "timestamp": timestamp}
        )
    elif writer == "sync":
        db.append_message_from_sync(
            stable_message_id=later_id, conversation_id=cid, sender=sender, content="Later",
            timestamp=timestamp, sync_client_id=db.client_id, object_revision=1, payload_hash="later",
        )
    else:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO messages(id, conversation_id, sender, content, timestamp, client_id) VALUES (?, ?, ?, ?, ?, ?)",
                (later_id, cid, sender, "Later", timestamp or db._get_current_utc_timestamp_iso(), db.client_id),
            )
    before = db.get_message_by_id(later_id)
    if sender == "user":
        with pytest.raises(ConflictError, match="later user"):
            db.insert_or_validate_user_turn(
                cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context
            )
    else:
        history = db.insert_or_validate_user_turn(
            cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context
        )
        assert [row["id"] for row in history] == [prior, uid]
        assert all("sequence" not in row and "message_id" not in row for row in history)
    assert db.get_message_by_id(later_id) == before


def exercise_concurrent_identity(db, conversation_id, open_peer):
    """Competing independent DB instances must rely on the database, not a local lock."""
    uid = str(uuid4())
    context = db.get_conversation_by_id(conversation_id)
    peers = [open_peer() for _ in range(4)]
    barrier = Barrier(len(peers))

    def insert(peer):
        barrier.wait(timeout=10)
        try:
            return peer.insert_or_validate_user_turn(
                conversation_id,
                uid,
                "Original question",
                owner_client_id=db.client_id,
                conversation_context=context,
                history_limit=20,
            )
        finally:
            peer.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=len(peers)) as executor:
            results = list(executor.map(insert, peers))
        assert all(result[-1]["id"] == uid for result in results)
        assert db.count_messages_for_conversation(conversation_id) == 1
        with pytest.raises(ConflictError):
            db.add_message(
                {"id": uid, "conversation_id": conversation_id, "sender": "user", "content": "Original question"}
            )
    finally:
        for peer in peers:
            peer.close_all_connections()


def test_sqlite_concurrent_identity_uses_independent_instances(populated_chacha_db, durable_conversation, temp_db_path):
    exercise_concurrent_identity(
        populated_chacha_db,
        durable_conversation,
        lambda: CharactersRAGDB(str(temp_db_path), client_id=populated_chacha_db.client_id),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_stream_failure_receipt_does_not_ack_assistant(populated_chacha_db, durable_conversation, partial):
    from tldw_Server_API.app.core.Chat.streaming_utils import create_streaming_response_with_timeout

    db, cid, uid = populated_chacha_db, durable_conversation, str(uuid4())
    _, runtime = await build_turn(db, cid, uid)

    def provider():
        if partial:
            yield 'data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n'
        yield 'data: {"error":{"message":"provider interrupted"}}\n\n'
        yield "data: [DONE]\n\n"

    chunks = [
        chunk
        async for chunk in create_streaming_response_with_timeout(
            provider(),
            cid,
            "unit-provider",
            user_message_id=runtime["user_message_id"],
        )
    ]
    payloads = [
        json.loads(line[6:])
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    receipts = [payload for payload in payloads if payload.get("tldw_user_message_id") == uid]
    assert receipts
    assert all("tldw_message_id" not in payload for payload in receipts)
    await build_turn(db, cid, uid)
    assert db.count_messages_for_conversation(cid) == 1
