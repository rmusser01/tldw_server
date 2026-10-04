"""Selected-durable orchestration checks with real SQLite, not native UAT."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import HistoryAdmissionReferenceV1, HistorySelectionV1
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection, snapshot_to_wire

pytestmark = pytest.mark.unit
DIGEST = "a" * 64


def selected_request(db, cid, uid, cursor=None, **fields):
    snapshot = snapshot_to_wire(
        db.get_conversation_history_snapshot(cid, owner_client_id=db.client_id, owner_key="owner")
    )
    selection = resolve_history_selection(
        snapshot,
        {
            "owner_key": "owner",
            "conversation_id": cid,
            "interpretation": {"kind": "parent_graph_v1"},
            "cursor": cursor or {"kind": "empty"},
            "selection_revision": 1,
        },
        "send",
        DIGEST,
    )["selection"]
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        stream=False,
        conversation_id=cid,
        save_to_db=True,
        messages=[
            {"role": "system", "content": " frozen evidence "},
            {"role": "user", "content": " original {{char}} "},
        ],
        **fields,
    )
    # Internal service tests do not wait for the independently owned public schema.
    request.tldw_turn = SimpleNamespace(
        user_message_id=uid,
        history_v1=SimpleNamespace(kind="selection", selection=HistorySelectionV1.model_validate(selection)),
        result_v1=SimpleNamespace(sources=()),
    )
    return request


async def build(db, request):
    conversation = db.get_conversation_by_id(request.conversation_id)
    runtime = {
        "history_owner_key": "owner",
        "history_owner_client_id": db.client_id,
        "history_scope": {key: conversation[key] for key in ("scope_type", "workspace_id")},
    }
    result = await chat_service.build_context_and_messages(
        db, request, asyncio.get_running_loop(), MagicMock(), True, request.conversation_id, AsyncMock(), runtime
    )
    return result, runtime


@pytest.mark.asyncio
async def test_single_input_admission_keeps_selected_sibling_and_literal_text(populated_chacha_db, monkeypatch):
    db = populated_chacha_db
    cid = db.add_conversation({"title": "Selected durable"})
    root = str(uuid4())
    empty = selected_request(db, cid, root).tldw_turn.history_v1.selection.model_dump(mode="json")
    accepted = db.append_selected_history_input(
        cid,
        empty,
        {"id": root, "sender": "user", "content": "earlier {{char}}"},
        owner_client_id=db.client_id,
        owner_key="owner",
    )
    reference = {
        key: accepted[key]
        for key in (
            "version",
            "owner_key",
            "conversation_id",
            "input_message_id",
            "input_message_revision",
            "selection_digest",
        )
    }

    def answer(text):
        return db.settle_history_admission(
            cid,
            reference,
            {"id": str(uuid4()), "sender": "assistant", "content": text, "parent_message_id": root},
            owner_client_id=db.client_id,
            owner_key="owner",
        )

    selected = answer("chosen {{user}}")
    answer("unselected")
    uid = str(uuid4())
    request = selected_request(db, cid, uid, {"kind": "after_message", "message_id": selected})

    def forbidden(*args, **kwargs):
        raise AssertionError("selected durable must not use legacy or plural admission")

    monkeypatch.setattr(db, "insert_or_validate_user_turn", forbidden)
    monkeypatch.setattr(db, "append_selected_history_inputs", forbidden)
    result, runtime = await build(db, request)
    assert result[4] == [
        {"role": "system", "content": " frozen evidence "},
        {"role": "user", "content": "earlier {{char}}"},
        {"role": "assistant", "content": "chosen {{user}}"},
        {"role": "user", "content": " original {{char}} "},
    ]
    assert db.get_message_by_id(uid)["parent_message_id"] == selected
    assert runtime["tldw_history_admission_v1"]["input_message_id"] == uid
    rows, _ = db.read_history_recovery_messages(
        cid,
        owner_client_id=db.client_id,
        owner_key="owner",
        scope={"scope_type": "global", "workspace_id": None},
        message_id=uid,
    )
    assert rows[0]["tldw_history_recovery_v1"]["status"] == "input_verified"


@pytest.mark.asyncio
async def test_empty_selection_replay_keeps_one_user(populated_chacha_db):
    db = populated_chacha_db
    cid = db.add_conversation({"title": "Replay"})
    uid = str(uuid4())
    request = selected_request(db, cid, uid)
    first, first_runtime = await build(db, request)
    second, second_runtime = await build(db, request)
    assert first[4] == second[4]
    assert db.count_messages_for_conversation(cid) == 1
    assert db.get_message_by_id(uid)["parent_message_id"] is None
    assert first_runtime["tldw_history_admission_v1"] == second_runtime["tldw_history_admission_v1"]


@pytest.mark.asyncio
async def test_before_first_selection_appends_a_null_parent(populated_chacha_db):
    db = populated_chacha_db
    cid = db.add_conversation({"title": "Before first"})
    root = str(uuid4())
    await build(db, selected_request(db, cid, root))
    uid = str(uuid4())
    request = selected_request(db, cid, uid, {"kind": "before_message", "message_id": root})
    result, _ = await build(db, request)
    assert db.get_message_by_id(uid)["parent_message_id"] is None
    assert result[4] == [
        {"role": "system", "content": " frozen evidence "},
        {"role": "user", "content": " original {{char}} "},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", [None, "text", "metadata"])
async def test_accepted_retry_revalidates_admitted_path_not_latest_leaf(populated_chacha_db, drift):
    db = populated_chacha_db
    cid = db.add_conversation({"title": "Accepted retry"})
    root = str(uuid4())
    first = selected_request(db, cid, root)
    await build(db, first)
    uid = str(uuid4())
    request = selected_request(db, cid, uid, {"kind": "after_message", "message_id": root})
    original, runtime = await build(db, request)
    admitted = runtime["tldw_history_admission_v1"]
    reference = {
        key: admitted[key]
        for key in (
            "version",
            "owner_key",
            "conversation_id",
            "input_message_id",
            "input_message_revision",
            "selection_digest",
        )
    }
    request.tldw_turn.history_v1 = SimpleNamespace(
        kind="admission",
        request_context_digest="b" * 64,
        admission=HistoryAdmissionReferenceV1.model_validate(reference),
    )
    request.model = "explicit-retry-model"
    await build(db, selected_request(db, cid, str(uuid4()), {"kind": "after_message", "message_id": root}))
    if drift == "text":
        db.update_message(root, {"content": "edited retained input"}, 1)
    elif drift == "metadata":
        db.add_message_metadata(root, extra={"sender_role": "user", "changed": True})
    count = db.count_messages_for_conversation(cid)
    if drift:
        with pytest.raises(HTTPException) as rejected:
            await build(db, request)
        assert rejected.value.status_code == 409
    else:
        retried, retried_runtime = await build(db, request)
        assert retried[4] == original[4]
        assert retried_runtime["tldw_history_admission_v1"] == admitted
    assert db.count_messages_for_conversation(cid) == count


@pytest.mark.asyncio
async def test_inherited_persona_default_cannot_become_neutral_context(populated_chacha_db):
    db = populated_chacha_db
    db.upsert_workspace("persona-default", "Persona default")
    workspace = db.get_workspace("persona-default")
    db.update_workspace(
        "persona-default",
        {
            "assistant_defaults_json": {
                "assistant_kind": "persona",
                "assistant_id": str(uuid4()),
                "persona_memory_mode": "none",
            }
        },
        workspace["version"],
    )
    cid = db.add_conversation(
        {"title": "Unbound inherited persona", "scope_type": "workspace", "workspace_id": "persona-default"}
    )
    with pytest.raises(HTTPException) as rejected:
        await build(db, selected_request(db, cid, str(uuid4())))
    assert rejected.value.status_code == 409
    assert rejected.value.detail["code"] == "unsupported_history_context_inherited_assistant"
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("sampling", [None, 0.7])
async def test_bound_sampling_must_be_explicit_and_equal_before_append(populated_chacha_db, sampling):
    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation

    db = populated_chacha_db
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": 1, "title": "Frozen sampling"},
        prompt_preset_id="st_default",
        provider="openai",
        model="gpt-4o-mini",
        sampling={"temperature": 0.23},
    )
    fields = {} if sampling is None else {"temperature": sampling}
    request = selected_request(db, cid, str(uuid4()), **fields)
    before = request.temperature
    with pytest.raises(HTTPException) as rejected:
        await build(db, request)
    assert rejected.value.status_code == 409
    assert request.temperature == before
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.asyncio
async def test_wire_verifier_reads_body_not_model_defaults():
    verify = getattr(endpoint, "_verify_selected_durable_wire", None)
    assert callable(verify), "raw body digest admission fence is missing"
    body = {
        "model": "local",
        "api_provider": "openai",
        "stream": False,
        "optional": None,
        "tldw_turn": {
            "user_message_id": str(uuid4()),
            "result_v1": {"version": 1, "sources": []},
            "history_v1": {"kind": "admission", "request_context_digest": DIGEST},
        },
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []}, receive)
    data = SimpleNamespace(
        tldw_turn=SimpleNamespace(history_v1=SimpleNamespace(kind="admission", request_context_digest=DIGEST))
    )
    with pytest.raises(HTTPException) as rejected:
        await verify(request, data)
    assert rejected.value.status_code == 409
    assert rejected.value.detail["code"] == "request_context_digest_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "metadata", "unverified"])
async def test_normalized_settlement_requires_committed_verified_result(populated_chacha_db, monkeypatch, fault):
    settle = getattr(endpoint, "_settle_selected_durable_result", None)
    assert callable(settle), "normalized bounded result settlement is missing"
    db = populated_chacha_db
    cid = db.add_conversation({"title": "Settlement"})
    uid = str(uuid4())
    _, runtime = await build(db, selected_request(db, cid, uid))
    admission = runtime["tldw_history_admission_v1"]
    reference = {
        key: admission[key]
        for key in (
            "version",
            "owner_key",
            "conversation_id",
            "input_message_id",
            "input_message_revision",
            "selection_digest",
        )
    }
    sources = [{"name": "Paper", "type": "pdf", "mode": "rag", "url": "", "pageContent": "evidence", "metadata": {}}]
    metadata = {"version": 1, "request_context_digest": DIGEST, "sources": sources}
    if fault == "metadata":

        def fail(*args, **kwargs):
            raise RuntimeError("metadata fault")

        monkeypatch.setattr(db.message_store, "_add_message_metadata_with_conn", fail)
    elif fault == "unverified":
        original = db.read_history_recovery_messages

        def unverified(*args, **kwargs):
            rows, total = original(*args, **kwargs)
            rows[0]["tldw_history_recovery_v1"] = {"version": 1, "status": "unverified", "code": "live_state_mismatch"}
            return rows, total

        monkeypatch.setattr(db, "read_history_recovery_messages", unverified)
    call = settle(
        db,
        cid,
        {"role": "assistant", "name": "ignored display name", "content": " exact answer "},
        reference=reference,
        owner_client_id=db.client_id,
        owner_key="owner",
        scope={"scope_type": "global", "workspace_id": None},
        result_metadata=metadata,
        runtime=runtime,
    )
    if fault:
        with pytest.raises((HTTPException, RuntimeError)):
            await call
        assert "tldw_history_result_v1" not in runtime
        assert db.count_messages_for_conversation(cid) == (1 if fault == "metadata" else 2)
    else:
        result_id = await call
        receipt = runtime["tldw_history_result_v1"]
        assert receipt["result_message_id"] == result_id
        assert receipt["admission"] == reference
        assert receipt["sources"] == sources
        assert db.get_message_by_id(result_id)["content"] == " exact answer "
        assert db.get_message_metadata(result_id)["extra"] == {
            "sender_role": "assistant",
            "history_result_v1": metadata,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
async def test_raw_stream_qualification_preserves_text_and_closes_source(asynchronous, invalid):
    chunks = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        "data: [DONE]\n\n",
    ]
    if invalid:
        chunks[1] = 'data: {"choices":[{"delta":{"content":[{"type":"text","text":"world"}]}}]}\n\n'
    closed = []

    async def async_source():
        try:
            for chunk in chunks:
                yield chunk
        finally:
            closed.append(True)

    def sync_source():
        try:
            yield from chunks
        finally:
            closed.append(True)

    qualified = chat_service._validated_selected_durable_stream(async_source() if asynchronous else sync_source())

    async def collect():
        return [chunk async for chunk in qualified] if asynchronous else list(qualified)

    if invalid:
        with pytest.raises(HTTPException) as rejected:
            await collect()
        assert rejected.value.detail["code"] == "unsupported_history_result_projection"
    else:
        assert await collect() == chunks
    assert closed == [True]
