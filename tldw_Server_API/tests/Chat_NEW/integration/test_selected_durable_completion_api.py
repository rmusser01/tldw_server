"""HTTP/SQLite regression tests with a provider double; these are not native UAT."""

import asyncio
import json
import re
import threading
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.Chat.history_wire import selected_durable_request_digest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture
def selected_api(credentialed_test_client, populated_chacha_db, auth_headers, monkeypatch):
    from tldw_Server_API.app.core.Chat import rate_limiter

    # Capacity is not this suite's subject; retain an isolated, enabled limiter.
    monkeypatch.setenv("TEST_CHAT_TOKENS_PER_MINUTE", "1000000")
    monkeypatch.setattr(rate_limiter, "_rate_limiter", rate_limiter.get_rate_limiter())
    rate_limiter.initialize_rate_limiter()
    client = credentialed_test_client
    db = CharactersRAGDB(populated_chacha_db.db_path_str, client_id="1", owner_user_id="1")
    cid = db.add_conversation({"title": "Selected durable HTTP"})
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        yield client, db, cid, auth_headers
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
        db.close_all_connections()


@pytest.fixture
def isolated_usage_api(selected_api, tmp_path, monkeypatch, test_openai_server_credential_factory):
    """Keep real AuthNZ usage isolated from other xdist workers' token records."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    client = selected_api[0]
    with monkeypatch.context() as usage_env:
        usage_env.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'usage.db'}")
        usage_env.setenv("OPENAI_API_KEY", "")
        usage_env.setitem(endpoint.API_KEYS, "openai", None)
        reset_settings()
        client.portal.call(reset_db_pool)

        async def seed_owner():
            pool = await get_db_pool()
            return await ensure_test_user(pool, "usage-quota-owner")

        try:
            assert client.portal.call(seed_owner) == 1
            with test_openai_server_credential_factory():
                yield selected_api
        finally:
            client.portal.call(reset_db_pool)
    reset_settings()


def body_for(client, cid, headers, stream=False, sources=None, max_tokens=None, user_content=" original {{char}} "):
    captured = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection",
        headers=headers,
        json={
            "purpose": "send",
            "view": {
                "view_session_id": "completion-test",
                "conversation_id": cid,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "empty"},
                "selection_revision": 1,
            },
        },
    )
    assert captured.status_code == 200, captured.text
    body = {
        "api_provider": "openai",
        "model": "gpt-4o-mini",
        "stream": stream,
        "conversation_id": cid,
        "save_to_db": True,
        "messages": [
            {"role": "system", "content": " frozen <doc id='0'>evidence</doc> "},
            {"role": "user", "content": user_content},
        ],
        "tldw_turn": {"user_message_id": str(uuid4()), "result_v1": {"version": 1, "sources": sources or []}},
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    capture = captured.json()
    selection = resolve_history_selection(
        capture["snapshot"], capture["view"], "send", selected_durable_request_digest(body)
    )["selection"]
    body["tldw_turn"]["history_v1"] = {"version": 1, "kind": "selection", "selection": selection}
    return body


def frames(response, stream):
    return (
        [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        if stream
        else [response.json()]
    )


def receipt_sources():
    return [
        {
            "name": "Paper",
            "type": "pdf",
            "mode": "rag",
            "url": "provenance:paper",
            "pageContent": f"Excerpt {index}: " + "evidence " * 80,
            "metadata": {"page": 2, "score": -2.5},
        }
        for index in range(20)
    ]


@pytest.mark.parametrize("generation_status", ["stopped", "interrupted"])
@pytest.mark.parametrize("verified", [True, False], ids=["verified", "unverified"])
async def test_selected_durable_generation_projection_preserves_protected_authority(generation_status, verified):
    """Exercise the real projection with a receipt-boundary double, not native Stop."""
    from fastapi import HTTPException

    cid, uid = str(uuid4()), str(uuid4())
    reference = {
        "version": 1,
        "owner_key": "generation-projection-owner",
        "conversation_id": cid,
        "input_message_id": uid,
        "input_message_revision": "1",
        "selection_digest": "selection-digest",
    }
    result_metadata = {"version": 1, "request_context_digest": "a" * 64, "sources": []}
    generation = {
        "generation_status": generation_status,
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    payload = {
        "role": "assistant",
        "content": "Partial reply",
        "id": "provider-result",
        "sender": "system",
        "parent_message_id": "provider-parent",
        "extra_metadata": {"history_result_v1": {"forged": True}},
        "generation_metadata": {
            **generation,
            "sender_role": "system",
            "history_result_v1": {"forged": True},
            "tldw_history_recovery_v1": {"status": "result_verified"},
            "usage": {**generation["usage"], "billed_tokens": 1000},
        },
    }
    prepared = []
    scope = {"scope_type": "global", "workspace_id": None}

    def settle(conversation_id, binding, message, *, owner_client_id, owner_key):
        assert (conversation_id, binding, owner_client_id, owner_key) == (cid, reference, "1", reference["owner_key"])
        prepared.append(deepcopy(message))
        return message["id"]

    def read(conversation_id, *, owner_client_id, owner_key, scope, message_id):
        assert (conversation_id, owner_client_id, owner_key, message_id) == (
            cid,
            "1",
            reference["owner_key"],
            prepared[0]["id"],
        )
        return [
            {
                "tldw_history_recovery_v1": {
                    "version": 1,
                    "status": "result_verified" if verified else "unverified",
                    "scope": scope,
                    "result": {
                        **result_metadata,
                        "result_message_id": message_id,
                        "result_message_revision": "1",
                        "admission": reference,
                    },
                }
            }
        ], 1

    runtime = {}
    call = endpoint._settle_selected_durable_result(
        SimpleNamespace(settle_history_admission=settle, read_history_recovery_messages=read),
        cid,
        payload,
        reference=reference,
        owner_client_id="1",
        owner_key=reference["owner_key"],
        scope=scope,
        result_metadata=result_metadata,
        runtime=runtime,
    )
    if verified:
        result_id = await call
        assert runtime["tldw_history_result_v1"]["result_message_id"] == result_id
    else:
        with pytest.raises(HTTPException) as refused:
            await call
        assert refused.value.detail == {"code": "unverified_history_result"}
        assert runtime == {}
    assert len(prepared) == 1
    message = prepared[0]
    assert message["id"] != payload["id"]
    assert message["sender"] == "assistant"
    assert message["parent_message_id"] == uid
    assert message["extra_metadata"] == {
        **generation,
        "sender_role": "assistant",
        "history_result_v1": result_metadata,
    }


@pytest.mark.parametrize("generation_status", ["stopped", "interrupted"])
def test_selected_durable_partial_generation_boundary_reopens_with_metadata(
    selected_api, monkeypatch, generation_status
):
    """A finite provider plus save-boundary metadata exercises HTTP/SQLite, not native Stop."""
    from tldw_Server_API.app.core.Chat import chat_service

    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers, stream=True)
    generation = {
        "generation_status": generation_status,
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    save = chat_service.save_assistant_message
    seen_payloads = []

    async def partial_generation_boundary(**kwargs):
        payload = deepcopy(kwargs["payload"])
        assert payload["role"] == "assistant"
        payload["generation_metadata"] = {
            **generation,
            "sender_role": "system",
            "history_result_v1": {"forged": True},
            "usage": {**generation["usage"], "billed_tokens": 1000},
        }
        seen_payloads.append(payload)
        return await save(**{**kwargs, "payload": payload})

    monkeypatch.setattr(chat_service, "save_assistant_message", partial_generation_boundary)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert len(seen_payloads) == 1
    rows = db.get_messages_for_conversation(cid)
    assert len(rows) == 2
    result = next(row for row in rows if row["sender"] == "assistant")
    extra = db.get_message_metadata(result["id"])["extra"]
    expected_result = {"version": 1, "request_context_digest": selected_durable_request_digest(body), "sources": []}
    assert extra == {**generation, "sender_role": "assistant", "history_result_v1": expected_result}
    events = frames(response, True)
    assert not any("error" in event for event in events), response.text
    receipt = next(event["tldw_history_result_v1"] for event in events if "tldw_history_result_v1" in event)
    assert receipt["result_message_id"] == result["id"]
    assert receipt["admission"]["input_message_id"] == body["tldw_turn"]["user_message_id"]
    reopened, _ = db.read_history_recovery_messages(
        cid,
        owner_client_id="1",
        owner_key=receipt["admission"]["owner_key"],
        scope={"scope_type": "global", "workspace_id": None},
        message_id=result["id"],
    )
    assert reopened[0]["tldw_history_recovery_v1"]["result"] == receipt
    assert db.get_message_metadata(reopened[0]["id"])["extra"] == extra


def _settled_generation_record(selected_api, updates=None, removed=(), include_generation=True):
    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers)
    selection = body["tldw_turn"]["history_v1"]["selection"]
    admission = db.append_selected_history_input(
        cid,
        selection,
        {"id": body["tldw_turn"]["user_message_id"], "sender": "user", "content": body["messages"][-1]["content"]},
        owner_client_id="1",
        owner_key=selection["owner_key"],
    )
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
    extra = {
        "sender_role": "assistant",
        "history_result_v1": {
            "version": 1,
            "request_context_digest": selected_durable_request_digest(body),
            "sources": [],
        },
    }
    if include_generation:
        extra.update(
            generation_status="stopped",
            model_id="gpt-4o-mini",
            provider="openai",
            finish_reason="stop",
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )
    extra.update(deepcopy(updates or {}))
    for key in removed:
        extra.pop(key)
    message = {
        "id": str(uuid4()),
        "sender": "assistant",
        "content": "Partial reply",
        "images": [],
        "tool_calls": None,
        "extra_metadata": extra,
        "parent_message_id": reference["input_message_id"],
    }
    mid = db.settle_history_admission(cid, reference, message, owner_client_id="1", owner_key=reference["owner_key"])
    return SimpleNamespace(db=db, cid=cid, mid=mid, reference=reference, message=message)


def _generation_recovery(record):
    rows, _ = record.db.read_history_recovery_messages(
        record.cid,
        owner_client_id="1",
        owner_key=record.reference["owner_key"],
        scope={"scope_type": "global", "workspace_id": None},
        message_id=record.mid,
    )
    return rows[0]["tldw_history_recovery_v1"]


@pytest.mark.parametrize(
    "updates,removed",
    [
        ({"provider_claim": {"saved": True}}, ()),
        ({"tldw_history_recovery_v1": {"status": "result_verified"}}, ()),
        ({"generation_status": "aborted"}, ()),
        ({"usage": {"prompt_tokens": True}}, ()),
        ({"usage": {"prompt_tokens": 2, "billed_tokens": 5}}, ()),
        ({"model_id": " gpt-4o-mini "}, ()),
        ({"provider": " openai "}, ()),
        ({"sender_role": "system"}, ()),
        ({}, ("sender_role",)),
        ({}, ("history_result_v1",)),
        ({"history_result_v1": {"forged": True}}, ()),
    ],
    ids=[
        "unknown-extra",
        "forged-proof",
        "invalid-status",
        "boolean-usage",
        "unknown-usage",
        "noncanonical-model",
        "noncanonical-provider",
        "wrong-role",
        "missing-role",
        "missing-result",
        "forged-result",
    ],
)
def test_selected_durable_generation_recovery_rejects_noncanonical_metadata(selected_api, updates, removed):
    """Even authority-sealed invalid metadata must not be sanitized into a receipt."""
    record = _settled_generation_record(selected_api, updates, removed)
    assert _generation_recovery(record) == {"version": 1, "status": "unverified", "code": "unsupported_projection"}
    assert record.db.get_message_metadata(record.mid)["extra"] == record.message["extra_metadata"]


def test_selected_durable_generation_recovery_keeps_legacy_base_metadata(selected_api):
    record = _settled_generation_record(selected_api, include_generation=False)
    proof = _generation_recovery(record)
    assert proof["status"] == "result_verified"
    assert proof["result"]["result_message_id"] == record.mid
    assert proof["result"]["admission"] == record.reference


@pytest.mark.parametrize("fence", ["live-state", "intent"])
def test_selected_durable_generation_recovery_rejects_live_metadata_mutation(selected_api, monkeypatch, fence):
    """Both full-metadata hashes fence edits; the intent case holds only the state boundary fixed."""
    record = _settled_generation_record(selected_api)
    assert _generation_recovery(record)["status"] == "result_verified"
    state = record.db.message_store._history_message_state
    with record.db.transaction() as conn:
        original_state = state(
            record.cid, record.mid, owner_client_id="1", owner_key=record.reference["owner_key"], conn=conn
        )
    assert record.db.set_message_metadata_extra(record.mid, {"generation_status": "complete"})
    if fence == "intent":

        def original_result_state(conversation_id, message_id, **kwargs):
            return (
                original_state
                if (conversation_id, message_id) == (record.cid, record.mid)
                else state(conversation_id, message_id, **kwargs)
            )

        monkeypatch.setattr(record.db.message_store, "_history_message_state", original_result_state)
    assert _generation_recovery(record) == {
        "version": 1,
        "status": "unverified",
        "code": "live_state_mismatch" if fence == "live-state" else "unsupported_projection",
    }


@pytest.mark.parametrize("cited", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_monthly_token_admission_excludes_persistence_only_sources(selected_api, monkeypatch, cited, stream):
    """The same inference prompt fits its allowance regardless of citation receipts."""
    client, db, cid, headers = selected_api
    sources = (
        [
            {
                "name": "Paper",
                "type": "pdf",
                "mode": "rag",
                "url": "provenance:paper",
                "pageContent": f"Excerpt {index}: " + "evidence " * 80,
                "metadata": {"page": 2, "score": -2.5},
            }
            for index in range(20)
        ]
        if cited
        else []
    )
    body = body_for(client, cid, headers, stream, sources)

    async def allowance(_uid, key):
        return 3000 if key == "limits.llm_tokens_per_month" else None

    async def unused_month(_uid):
        return 0

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", allowance)
    monkeypatch.setattr(endpoint, "llm_tokens_this_month", unused_month)
    provider = endpoint.perform_chat_api_call
    calls = []

    def record(*args, **kwargs):
        messages = kwargs.get("messages_payload") if "messages_payload" in kwargs else args[1]
        assert messages[-1]["content"] == " original {{char}} "
        assert " frozen <doc id='0'>evidence</doc> " in kwargs["system_message"]
        assert "tldw_turn" not in kwargs
        calls.append(True)
        return provider(*args, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", record)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert calls == [True]
    receipt = next(
        event["tldw_history_result_v1"] for event in frames(response, stream) if "tldw_history_result_v1" in event
    )
    assert receipt["sources"] == sources
    assert db.count_messages_for_conversation(cid) == 2


@pytest.mark.parametrize("gate", ["legacy", "rg"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("payload", ["plain", "citations", "large-prompt"])
def test_finite_token_admission_charges_inference_not_receipts(selected_api, monkeypatch, gate, stream, payload):
    from tldw_Server_API.app.core.Chat.rate_limiter import ConversationRateLimiter, RateLimitConfig
    from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RGRequest

    client, db, cid, headers = selected_api
    sources = receipt_sources() if payload == "citations" else []
    body = body_for(
        client,
        cid,
        headers,
        stream,
        sources,
        max_tokens=32,
        user_content="inference " * 1600 if payload == "large-prompt" else " original {{char}} ",
    )
    monkeypatch.setenv("RG_ENABLED", "1" if gate == "rg" else "0")
    policy = {"tokens": {"per_min": 3000, "burst": 1.0}, "scopes": ["global", "user"]}
    governor = MemoryResourceGovernor(policies={"chat.default": policy}, time_source=lambda: 1000.0)
    loader = SimpleNamespace(
        get_policy=lambda _: policy,
        get_snapshot=lambda: SimpleNamespace(route_map={"default": "chat.default"}),
    )
    monkeypatch.setattr(client.app.state, "rg_governor", governor, raising=False)
    monkeypatch.setattr(client.app.state, "rg_policy_loader", loader, raising=False)
    reservations = []
    if gate == "rg":
        # A partially consumed real bucket distinguishes prompt admission from capacity clamping.
        decision, _ = client.portal.call(
            governor.reserve,
            RGRequest(entity="user:1", categories={"tokens": {"units": 1000}}, tags={"policy_id": "chat.default"}),
        )
        assert decision.allowed
        reserve = governor.reserve

        async def record_reserve(req, op_id=None):
            reservations.append(req.categories["tokens"]["units"])
            return await reserve(req, op_id=op_id)

        monkeypatch.setattr(governor, "reserve", record_reserve)
    limiter = ConversationRateLimiter(RateLimitConfig(per_user_tokens_per_minute=3000, burst_multiplier=1.0))
    monkeypatch.setattr(endpoint, "get_rate_limiter", lambda: limiter)
    provider = endpoint.perform_chat_api_call
    calls = []

    def record(*args, **kwargs):
        calls.append(True)
        return provider(*args, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", record)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    if gate == "rg":
        assert len(reservations) == 1
    if payload == "large-prompt":
        assert response.status_code == 429, response.text
        assert calls == []
        assert db.count_messages_for_conversation(cid) == 0
    else:
        assert response.status_code == 200, response.text
        assert calls == [True]
        receipt = next(
            event["tldw_history_result_v1"] for event in frames(response, stream) if "tldw_history_result_v1" in event
        )
        assert receipt["sources"] == sources
        assert db.count_messages_for_conversation(cid) == 2


@pytest.mark.parametrize("execution", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cited", [False, True])
def test_owned_queue_estimates_exclude_persistence_only_sources(selected_api, monkeypatch, execution, stream, cited):
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Chat.request_queue import RequestQueue

    client, db, cid, headers = selected_api
    sources = receipt_sources() if cited else []
    body = body_for(client, cid, headers, stream, sources, max_tokens=32)
    queue = RequestQueue(max_queue_size=2, max_concurrent=1, timeout=5)
    enqueue = queue.enqueue
    estimates = []

    async def record_enqueue(*args, **kwargs):
        estimates.append(kwargs["estimated_tokens"])
        return await enqueue(*args, **kwargs)

    monkeypatch.setattr(queue, "enqueue", record_enqueue)
    monkeypatch.setenv("RG_ENABLED", "0")
    monkeypatch.setenv("FORCE_CHAT_QUEUE_IN_TESTS", "1")
    monkeypatch.setattr(endpoint, "QUEUED_EXECUTION", execution)
    monkeypatch.setattr(endpoint, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    client.portal.call(queue.start)
    try:
        response = client.post("/api/v1/chat/completions", headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert len(estimates) == 1
        assert 0 < estimates[0] < 3000
        receipt = next(
            event["tldw_history_result_v1"] for event in frames(response, stream) if "tldw_history_result_v1" in event
        )
        assert receipt["sources"] == sources
        assert db.count_messages_for_conversation(cid) == 2
    finally:
        client.portal.call(queue.stop)


@pytest.mark.parametrize("stream", [False, True])
def test_full_persistence_envelope_still_obeys_request_size_limit(selected_api, monkeypatch, stream):
    from tldw_Server_API.app.api.v1.schemas import chat_validators

    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers, stream, receipt_sources(), max_tokens=32)
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 3000)
    calls = []

    def unexpected_provider(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Size refusal must precede provider dispatch")

    monkeypatch.setattr(endpoint, "perform_chat_api_call", unexpected_provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 413, response.text
    assert calls == []
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.parametrize("invalid", ["text-parts", "image", "tool"])
def test_rejected_projection_records_consumed_tokens_before_next_quota_check(isolated_usage_api, monkeypatch, invalid):
    """A rejected result cannot hide provider consumption from the monthly gate."""
    from tldw_Server_API.app.core.Chat import chat_service

    client, db, cid, headers = isolated_usage_api
    monkeypatch.setenv("LLM_USAGE_ENABLED", "true")
    used_before = client.portal.call(endpoint.llm_tokens_this_month, 1)

    async def allowance(_uid, key):
        return used_before + 5000 if key == "limits.llm_tokens_per_month" else None

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", allowance)
    body = body_for(client, cid, headers)
    message = {"role": "assistant", "content": "Answer"}
    if invalid == "text-parts":
        message["content"] = [{"type": "text", "text": "Answer"}]
    elif invalid == "image":
        message["images"] = [{"url": "data:image/png;base64,AA=="}]
    else:
        message["tool_calls"] = [
            {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]
    calls = []

    def provider(*args, **kwargs):
        calls.append(True)
        return {
            "id": "consumed-invalid-projection",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 750000, "completion_tokens": 250000, "total_tokens": 1000000},
        }

    recorded_usage = []
    log_usage = chat_service.log_llm_usage

    async def record_usage(**kwargs):
        recorded_usage.append(kwargs)
        await log_usage(**kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)
    monkeypatch.setattr(chat_service, "log_llm_usage", record_usage)
    rejected = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert rejected.status_code == 409, rejected.text
    assert rejected.json()["detail"]["code"] == "unsupported_history_result_projection"
    assert db.count_messages_for_conversation(cid) == 1
    assert len(recorded_usage) == 1
    assert recorded_usage[0]["total_tokens"] == 1000000
    assert client.portal.call(endpoint.llm_tokens_this_month, 1) == used_before + 1000000

    next_cid = db.add_conversation({"title": "Quota after rejected projection"})
    next_body = body_for(client, next_cid, headers)
    denied = client.post("/api/v1/chat/completions", headers=headers, json=next_body)
    assert denied.status_code == 402, denied.text
    assert denied.json()["detail"]["category"] == "llm_tokens_month"
    assert calls == [True]
    assert len(recorded_usage) == 1
    assert db.count_messages_for_conversation(next_cid) == 0


@pytest.mark.parametrize(
    ("stream", "unified", "settled_before_admission"),
    [(False, False, False), (True, False, False), (True, True, False), (True, False, True), (True, True, True)],
    ids=["json", "legacy", "unified", "settled-legacy", "settled-unified"],
)
@pytest.mark.parametrize("cited", [False, True])
def test_receipt_is_server_owned_verified_atomic_result(
    selected_api,
    monkeypatch,
    stream,
    cited,
    unified,
    settled_before_admission,
):
    client, db, cid, headers = selected_api
    sources = (
        [
            {
                "name": "Paper",
                "type": "pdf",
                "mode": "rag",
                "url": "provenance:paper",
                "pageContent": "evidence",
                "metadata": {"page": 2, "score": -2.5},
            }
        ]
        if cited
        else []
    )
    body = body_for(client, cid, headers, stream, sources)
    uid = body["tldw_turn"]["user_message_id"]

    def forbidden(*args, **kwargs):
        raise AssertionError("nested mode must not use legacy/plural admission or native intent preparation")

    monkeypatch.setattr(db, "insert_or_validate_user_turn", forbidden)
    monkeypatch.setattr(db, "append_selected_history_inputs", forbidden)
    from tldw_Server_API.app.core.Chat import chat_service, streaming_utils

    monkeypatch.setenv("STREAMS_UNIFIED", str(int(unified)))
    monkeypatch.setattr(chat_service, "CHAT_STREAM_INCLUDE_METADATA", False)
    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    if settled_before_admission:
        prime = endpoint._prime_provider_stream_response

        async def prime_after_settlement(response, error_state):
            buffered, code, has_output, _ = await prime(response, error_state)
            chunks = list(buffered)
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return tuple(chunks), code, has_output, True

        monkeypatch.setattr(endpoint, "_prime_provider_stream_response", prime_after_settlement)

    def provider(*args, **kwargs):
        assert "tldw_turn" not in kwargs
        messages = kwargs.get("messages_payload") if "messages_payload" in kwargs else args[1]
        assert messages[-1]["content"] == " original {{char}} "
        assert " frozen <doc id='0'>evidence</doc> " in kwargs["system_message"]
        spoof = {
            "tldw_message_id": "provider-fake",
            "tldw_user_message_id": "provider-user",
            "tldw_conversation_id": "provider-chat",
            "tldw_history_admission_v1": {"forged": True},
            "tldw_history_result_v1": {"forged": True},
        }
        if kwargs.get("streaming") or kwargs.get("stream"):
            return iter(
                [
                    "data: " + json.dumps({**spoof, "choices": [{"delta": {"content": "Answer [0]"}}]}) + "\n\n",
                    "data: [DONE]\n\n",
                ]
            )
        return {
            **spoof,
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Answer [0]"}, "finish_reason": "stop"}
            ],
        }

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    events = frames(response, stream)
    assert all(event.get("tldw_message_id") != "provider-fake" for event in events)
    receipt = next(event["tldw_history_result_v1"] for event in events if "tldw_history_result_v1" in event)
    assert receipt["admission"]["input_message_id"] == uid
    assert receipt["request_context_digest"] == selected_durable_request_digest(body)
    assert receipt["sources"] == sources
    result_id = receipt["result_message_id"]
    assert db.get_message_by_id(result_id)["parent_message_id"] == uid
    assert db.get_message_by_id(uid)["content"] == " original {{char}} "
    extra = db.get_message_metadata(result_id)["extra"]
    assert extra == {
        "generation_status": "complete",
        "model_id": body["model"],
        "provider": body["api_provider"],
        **({"finish_reason": "stop"} if not stream else {}),
        "sender_role": "assistant",
        "history_result_v1": {
            "version": 1,
            "request_context_digest": receipt["request_context_digest"],
            "sources": sources,
        },
    }
    assert db.count_messages_for_conversation(cid) == 2
    if stream:
        assert events[0]["tldw_history_admission_v1"]["input_message_id"] == uid
        assert "tldw_history_result_v1" not in events[0]


@pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
@pytest.mark.parametrize(
    ("raw_text", "redacted_text"),
    [
        (" Raw answer [0]\n ", None),
        ('{"tldw_message_id":"provider-fake","tldw_history_result_v1":{"forged":true}}', None),
        (" secret answer [0]\n ", " [REDACTED] answer [0]\n "),
    ],
    ids=["text", "json-looking-text", "redaction"],
)
def test_raw_string_result_returns_verified_receipt_without_resettling(
    selected_api,
    monkeypatch,
    normalize: bool,
    raw_text: str,
    redacted_text: str | None,
) -> None:
    """Return the exact output-safe text after one protected settlement."""
    client, db, cid, headers = selected_api
    sources = [
        {
            "name": "Paper",
            "type": "pdf",
            "mode": "rag",
            "url": "provenance:paper",
            "pageContent": "evidence",
            "metadata": {"page": 2},
        },
    ]
    body = body_for(client, cid, headers, sources=sources)
    uid = body["tldw_turn"]["user_message_id"]
    expected_text = raw_text if redacted_text is None else redacted_text
    if redacted_text is not None:
        from tldw_Server_API.app.core.Moderation.moderation_service import (
            ModerationPolicy,
            ModerationService,
            PatternRule,
        )
        from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator

        moderation = ModerationService.__new__(ModerationService)
        moderation._lock = threading.RLock()
        moderation._policy_evaluator = PolicyEvaluator()
        moderation._max_scan_chars = 200_000
        moderation._match_window_chars = 4_096
        moderation._max_fallback_scan_chars = 800_000
        moderation._max_replacements_per_pattern = 1_000
        moderation._user_overrides = {}
        moderation._global_policy = ModerationPolicy(
            enabled=True,
            input_action="warn",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=[PatternRule(regex=re.compile("secret"), action="redact", phase="output")],
        )
        monkeypatch.setattr(endpoint, "get_moderation_service", lambda: moderation)
    monkeypatch.setenv("CHAT_FORCE_NORMALIZE_STRING_RESPONSES", str(normalize).lower())
    monkeypatch.setattr(endpoint, "perform_chat_api_call", lambda *args, **kwargs: raw_text)
    settle = db.settle_history_admission
    settled_ids = []

    def record_settlement(
        conversation_id: str,
        reference: Mapping[str, Any],
        message: Mapping[str, Any],
        *,
        owner_client_id: str,
        owner_key: str,
        conn: Any | None = None,
    ) -> str:
        """Record real settlement IDs while preserving protected persistence."""
        result_id = settle(
            conversation_id,
            reference,
            message,
            owner_client_id=owner_client_id,
            owner_key=owner_key,
            conn=conn,
        )
        settled_ids.append(result_id)
        return result_id

    monkeypatch.setattr(db, "settle_history_admission", record_settlement)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert db.count_messages_for_conversation(cid) == 2
    assert len(settled_ids) == 1
    result_id = settled_ids[0]
    persisted_text = db.get_message_by_id(result_id)["content"]
    assert persisted_text == expected_text
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == body["model"]
    assert len(payload["choices"]) == 1
    choice = payload["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == persisted_text
    assert payload["tldw_conversation_id"] == cid
    assert payload["tldw_user_message_id"] == uid
    assert payload["tldw_message_id"] == result_id
    receipt = payload["tldw_history_result_v1"]
    assert receipt["result_message_id"] == result_id
    assert receipt["admission"]["input_message_id"] == uid
    assert receipt["request_context_digest"] == selected_durable_request_digest(body)
    assert receipt["sources"] == sources
    assert db.get_message_by_id(result_id)["parent_message_id"] == uid
    assert db.get_message_metadata(result_id)["extra"] == {
        "generation_status": "complete",
        "model_id": body["model"],
        "provider": body["api_provider"],
        "finish_reason": "stop",
        "sender_role": "assistant",
        "history_result_v1": {
            "version": 1,
            "request_context_digest": receipt["request_context_digest"],
            "sources": sources,
        },
    }
    rows, _ = db.read_history_recovery_messages(
        cid,
        owner_client_id="1",
        owner_key=receipt["admission"]["owner_key"],
        scope={"scope_type": "global", "workspace_id": None},
        message_id=result_id,
    )
    assert rows[0]["tldw_history_recovery_v1"] == {
        "version": 1,
        "status": "result_verified",
        "scope": {"scope_type": "global", "workspace_id": None},
        "result": receipt,
    }


def test_nonselected_raw_string_response_keeps_normalization_opt_in(selected_api, monkeypatch) -> None:
    """Nonselected turns retain their configured raw-string response shape."""
    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers)
    body.pop("tldw_turn")
    body["save_to_db"] = False
    monkeypatch.setenv("CHAT_FORCE_NORMALIZE_STRING_RESPONSES", "false")
    monkeypatch.setattr(endpoint, "perform_chat_api_call", lambda *args, **kwargs: "legacy raw answer")
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert response.json() == "legacy raw answer"
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.parametrize("change", ["stream", "system", "null", "digest", "sources"])
def test_raw_digest_mismatch_rejects_before_any_write(selected_api, monkeypatch, change):
    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers)
    if change == "stream":
        body["stream"] = True
    elif change == "system":
        body["messages"][0]["content"] += " changed"
    elif change == "null":
        body["temperature"] = None
    elif change == "sources":
        body["tldw_turn"]["result_v1"]["sources"] = [
            {"name": "Paper", "type": "pdf", "mode": "rag", "url": "", "pageContent": "evidence", "metadata": {}}
        ]
    else:
        body["tldw_turn"]["history_v1"]["selection"]["request_context_digest"] = "b" * 64

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched digest must not reach admission/provider")

    monkeypatch.setattr(db, "append_selected_history_input", forbidden)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "request_context_digest_mismatch"
    assert db.count_messages_for_conversation(cid) == 0


def test_resolved_inference_cannot_rewrite_finalized_wire(selected_api, monkeypatch):
    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers)
    original = endpoint.resolve_provider_and_model

    def rewrite(**kwargs):
        resolved = list(original(**kwargs))
        resolved[3] = "silently-changed-model"
        return tuple(resolved)

    monkeypatch.setattr(endpoint, "resolve_provider_and_model", rewrite)

    def forbidden(*args, **kwargs):
        raise AssertionError("rewritten inference must not reach admission/provider")

    monkeypatch.setattr(db, "append_selected_history_input", forbidden)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "resolved_inference_mismatch"
    assert db.count_messages_for_conversation(cid) == 0
    assert db.get_conversation_settings(cid) is None


@pytest.mark.parametrize("guard", ["tools", "legacy", "top-level", "sync", "skills"])
def test_unsupported_combinations_stay_closed_before_admission(selected_api, monkeypatch, guard):
    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers)
    if guard == "tools":
        body["tools"] = []
    elif guard == "legacy":
        body["metadata"] = {"tldw_retry_failed_turn": True}
    elif guard == "top-level":
        body["tldw_history_selection_v1"] = body["tldw_turn"]["history_v1"]["selection"]
    elif guard == "skills":
        monkeypatch.setattr(db, "history_skills_may_be_visible", lambda: True)
    else:
        from tldw_Server_API.app.api.v1.endpoints import character_messages

        monkeypatch.setattr(character_messages, "_active_message_sync_service", lambda *args: object())

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported combinations cannot dispatch or append")

    monkeypatch.setattr(db, "append_selected_history_input", forbidden)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == (409 if guard in {"skills", "sync"} else 422), response.text
    assert db.count_messages_for_conversation(cid) == 0


@pytest.mark.parametrize("stream", [False, True])
def test_accepted_reference_retry_uses_fresh_attempt_digest_without_second_user(selected_api, monkeypatch, stream):
    client, db, cid, headers = selected_api
    initial = body_for(client, cid, headers, stream)
    provider = endpoint.perform_chat_api_call
    observed_messages = []

    def record(*args, **kwargs):
        messages = kwargs.get("messages_payload") if "messages_payload" in kwargs else args[1]
        observed_messages.append(deepcopy(messages))
        return provider(*args, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", record)
    first = client.post("/api/v1/chat/completions", headers=headers, json=initial)
    assert first.status_code == 200, first.text
    first_result = next(
        event["tldw_history_result_v1"] for event in frames(first, stream) if "tldw_history_result_v1" in event
    )
    retry = deepcopy(initial)
    retry["messages"][0]["content"] = " freshly prepared request-local context "
    retry["tldw_turn"]["history_v1"] = {
        "version": 1,
        "kind": "admission",
        "admission": first_result["admission"],
        "request_context_digest": selected_durable_request_digest(retry),
    }
    second = client.post("/api/v1/chat/completions", headers=headers, json=retry)
    assert second.status_code == 200, second.text
    second_result = next(
        event["tldw_history_result_v1"] for event in frames(second, stream) if "tldw_history_result_v1" in event
    )
    assert second_result["admission"] == first_result["admission"]
    assert second_result["request_context_digest"] != first_result["request_context_digest"]
    assert second_result["result_message_id"] != first_result["result_message_id"]
    assert len(observed_messages) == 2
    assert all(messages == [{"role": "user", "content": " original {{char}} "}] for messages in observed_messages)
    assert db.count_messages_for_conversation(cid) == 3


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_transformative_input_moderation_rejects_before_selected_admission(selected_api, monkeypatch, stream, retry):
    from tldw_Server_API.app.core.Moderation.models import ModerationPolicy, PatternRule
    from tldw_Server_API.app.core.Moderation.moderation_service import ModerationService
    from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator

    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers, stream)
    uid = body["tldw_turn"]["user_message_id"]
    if retry:
        first = client.post("/api/v1/chat/completions", headers=headers, json=body)
        assert first.status_code == 200, first.text
        receipt = next(
            event["tldw_history_result_v1"] for event in frames(first, stream) if "tldw_history_result_v1" in event
        )
        body["messages"][0]["content"] = " fresh frozen systems "
        body["tldw_turn"]["history_v1"] = {
            "version": 1,
            "kind": "admission",
            "admission": receipt["admission"],
            "request_context_digest": selected_durable_request_digest(body),
        }
    # Use the real evaluator/redactor without loading mutable runtime config.
    moderation = ModerationService.__new__(ModerationService)
    moderation._lock = threading.RLock()
    moderation._policy_evaluator = PolicyEvaluator()
    moderation._max_scan_chars = 200_000
    moderation._match_window_chars = 4_096
    moderation._max_fallback_scan_chars = 800_000
    moderation._max_replacements_per_pattern = 1_000
    moderation._user_overrides = {}
    moderation._global_policy = ModerationPolicy(
        enabled=True,
        input_action="redact",
        output_enabled=False,
        per_user_overrides=False,
        block_patterns=[PatternRule(regex=re.compile("original"), action="redact", phase="input")],
    )
    assert (
        moderation.redact_text(body["messages"][-1]["content"], moderation._global_policy)
        != body["messages"][-1]["content"]
    )
    monkeypatch.setattr(endpoint, "get_moderation_service", lambda: moderation)
    calls = []
    provider = endpoint.perform_chat_api_call

    def record(*args, **kwargs):
        calls.append(True)
        return provider(*args, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", record)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "selected_durable_input_transformation"
    assert calls == []
    assert db.count_messages_for_conversation(cid) == (2 if retry else 0)
    if retry:
        assert db.get_message_by_id(uid)["content"] == " original {{char}} "


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("invalid", ["role", "image", "text-parts", "image-parts", "number", "tool"])
def test_raw_provider_projection_rejects_before_result_authority(selected_api, monkeypatch, stream, invalid):
    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers, stream)
    message = {"role": "assistant", "content": "Answer"}
    if invalid == "role":
        message["role"] = "user"
    elif invalid == "image":
        message["images"] = [{"url": "data:image/png;base64,AA=="}]
    elif invalid == "text-parts":
        message["content"] = [{"type": "text", "text": "Answer"}]
    elif invalid == "image-parts":
        message["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    elif invalid == "number":
        message["content"] = 7
    else:
        message["tool_calls"] = [
            {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]

    def provider(*args, **kwargs):
        if stream:
            # Force the error through SSE after a valid first output, not preflight HTTP.
            return iter(
                [
                    "data: "
                    + json.dumps({"choices": [{"delta": {"role": "assistant", "content": "prefix "}}]})
                    + "\n\n",
                    "data: " + json.dumps({"choices": [{"delta": message}]}) + "\n\n",
                    "data: [DONE]\n\n",
                ]
            )
        return {
            "id": "test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    if stream:
        assert response.status_code == 200, response.text
        events = frames(response, True)
        assert events[0]["tldw_history_admission_v1"]["input_message_id"] == body["tldw_turn"]["user_message_id"]
        assert any(
            choice.get("delta", {}).get("content") == "prefix "
            for event in events
            for choice in event.get("choices", [])
        ), response.text
        assert any("error" in event for event in events), response.text
        assert not any(event.get("success") is True for event in events)
        assert not any(choice.get("finish_reason") == "stop" for event in events for choice in event.get("choices", []))
        assert not any("tldw_history_result_v1" in event or "tldw_message_id" in event for event in events)
    else:
        # Generic usability already rejects a numeric JSON response before projection.
        assert response.status_code == (502 if invalid == "number" else 409), response.text
        if invalid != "number":
            assert response.json()["detail"]["code"] == "unsupported_history_result_projection"
    assert db.count_messages_for_conversation(cid) == 1


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "malformed", ["utf8", None, "projection"], ids=["malformed-utf8", "provider-interruption", "malformed-projection"]
)
def test_raw_byte_validation_fences_partial_result_authority(selected_api, monkeypatch, asynchronous, malformed):
    client, db, cid, headers = selected_api
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    body = body_for(client, cid, headers, True)
    prefix = (
        "data: " + json.dumps({"choices": [{"delta": {"role": "assistant", "content": "prefix "}}]}) + "\n\n"
    ).encode("utf-8")
    rejected_chunk = (
        b"\xff" if malformed == "utf8" else b'data: {"choices":[{"delta":{"role":"user","content":"bad"}}]}\n\n'
    )

    def synchronous_stream():
        yield prefix
        if malformed:
            yield rejected_chunk
        else:
            raise RuntimeError("provider connection closed")

    async def asynchronous_stream():
        yield prefix
        if malformed:
            yield rejected_chunk
        else:
            raise RuntimeError("provider connection closed")

    monkeypatch.setattr(
        endpoint,
        "perform_chat_api_call",
        lambda *args, **kwargs: asynchronous_stream() if asynchronous else synchronous_stream(),
    )
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    events = frames(response, True)
    assert any(
        choice.get("delta", {}).get("content") == "prefix " for event in events for choice in event.get("choices", [])
    ), response.text
    assert any("error" in event for event in events), response.text
    if malformed:
        assert not any(event.get("success") is True for event in events)
        assert not any(choice.get("finish_reason") == "stop" for event in events for choice in event.get("choices", []))
        assert not any("tldw_history_result_v1" in event or "tldw_message_id" in event for event in events)
        assert db.count_messages_for_conversation(cid) == 1
    else:
        assert any("tldw_history_result_v1" in event for event in events), response.text
        assert db.count_messages_for_conversation(cid) == 2


@pytest.mark.parametrize("fault", ["metadata", "unverified"])
def test_mandatory_sse_settlement_fault_is_an_explicit_unknown_terminal(selected_api, monkeypatch, fault):
    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers, True)
    uid = body["tldw_turn"]["user_message_id"]
    if fault == "metadata":

        def fail(*args, **kwargs):
            raise RuntimeError("private metadata failure details")

        monkeypatch.setattr(db.message_store, "_add_message_metadata_with_conn", fail)
    else:
        read = db.read_history_recovery_messages

        def unverified(*args, **kwargs):
            rows, total = read(*args, **kwargs)
            rows[0]["tldw_history_recovery_v1"] = {"version": 1, "status": "unverified", "code": "live_state_mismatch"}
            return rows, total

        monkeypatch.setattr(db, "read_history_recovery_messages", unverified)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    events = frames(response, True)
    assert events[0]["tldw_history_admission_v1"]["input_message_id"] == uid
    errors = [event["error"] for event in events if "error" in event]
    assert errors == [
        {
            "code": "selected_durable_result_unverified",
            "type": "history_result_error",
            "message": "Selected durable result could not be verified.",
        }
    ], response.text
    assert not any(event.get("success") is True for event in events)
    assert not any(choice.get("finish_reason") == "stop" for event in events for choice in event.get("choices", []))
    assert not any("tldw_history_result_v1" in event or "tldw_message_id" in event for event in events)
    assert "private metadata failure details" not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")
    assert db.count_messages_for_conversation(cid) == (1 if fault == "metadata" else 2)


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("fault", [None, "metadata", "unverified"])
def test_completed_sse_consumption_is_accounted_once_even_if_settlement_fails(
    isolated_usage_api, monkeypatch, unified, fault
):
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor

    client, db, cid, headers = isolated_usage_api
    monkeypatch.setenv("LLM_USAGE_ENABLED", "true")
    monkeypatch.setenv("STREAMS_UNIFIED", "1" if unified else "0")
    monkeypatch.setenv("RG_ENABLED", "1")
    used_before = client.portal.call(endpoint.llm_tokens_this_month, 1)
    limit = used_before + 5000

    async def allowance(_uid, key):
        return limit if key == "limits.llm_tokens_per_month" else None

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", allowance)
    policy = {"tokens": {"per_min": 1000000, "burst": 1.0}, "scopes": ["global", "user"]}
    governor = MemoryResourceGovernor(policies={"chat.default": policy}, time_source=lambda: 1000.0)
    loader = SimpleNamespace(
        get_policy=lambda _: policy,
        get_snapshot=lambda: SimpleNamespace(route_map={"default": "chat.default"}),
    )
    monkeypatch.setattr(client.app.state, "rg_governor", governor, raising=False)
    monkeypatch.setattr(client.app.state, "rg_policy_loader", loader, raising=False)
    commits = []
    commit = governor.commit

    async def record_commit(handle_id, actuals=None, op_id=None):
        commits.append((handle_id, actuals))
        return await commit(handle_id, actuals=actuals, op_id=op_id)

    monkeypatch.setattr(governor, "commit", record_commit)
    usage = []
    log_usage = chat_service.log_llm_usage

    async def record_usage(**kwargs):
        usage.append(kwargs)
        await log_usage(**kwargs)

    monkeypatch.setattr(chat_service, "log_llm_usage", record_usage)
    body = body_for(client, cid, headers, True)
    reply = "Answer " * 128
    calls = []

    def provider(*args, **kwargs):
        calls.append(True)
        return iter(
            [
                "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": reply}}]}) + "\n\n",
                "data: [DONE]\n\n",
            ]
        )

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)
    if fault == "metadata":

        def fail(*args, **kwargs):
            raise RuntimeError("private metadata failure details")

        monkeypatch.setattr(db.message_store, "_add_message_metadata_with_conn", fail)
    elif fault == "unverified":
        read = db.read_history_recovery_messages

        def unverified(*args, **kwargs):
            rows, total = read(*args, **kwargs)
            rows[0]["tldw_history_recovery_v1"] = {"version": 1, "status": "unverified", "code": "live_state_mismatch"}
            return rows, total

        monkeypatch.setattr(db, "read_history_recovery_messages", unverified)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    events = frames(response, True)
    if fault:
        assert [event["error"]["code"] for event in events if "error" in event] == [
            "selected_durable_result_unverified"
        ], response.text
        assert not any(event.get("success") is True for event in events)
        assert not any(choice.get("finish_reason") == "stop" for event in events for choice in event.get("choices", []))
        assert not any("tldw_history_result_v1" in event or "tldw_message_id" in event for event in events)
    else:
        assert any("tldw_history_result_v1" in event for event in events)
    assert response.text.rstrip().endswith("data: [DONE]")
    assert db.count_messages_for_conversation(cid) == (1 if fault == "metadata" else 2)
    assert len(usage) == 1
    total = usage[0]["total_tokens"]
    assert usage[0]["completion_tokens"] == len(reply) // 4
    assert total == usage[0]["prompt_tokens"] + usage[0]["completion_tokens"]
    assert total > 100
    assert client.portal.call(endpoint.llm_tokens_this_month, 1) == used_before + total
    assert len(commits) == 1
    assert commits[0][1] == {"tokens": total}

    limit = used_before + total
    next_cid = db.add_conversation({"title": "Quota after streamed settlement"})
    next_body = body_for(client, next_cid, headers, True, max_tokens=1)
    denied = client.post("/api/v1/chat/completions", headers=headers, json=next_body)
    assert denied.status_code == 402, denied.text
    assert denied.json()["detail"]["category"] == "llm_tokens_month"
    assert calls == [True]
    assert len(usage) == 1
    assert [actuals for _, actuals in commits] == [{"tokens": total}, {"tokens": 0}]
    assert db.count_messages_for_conversation(next_cid) == 0


@pytest.mark.parametrize("stream,fault", [(False, False), (True, False), (True, True)])
def test_completed_stream_billing_has_one_owner_even_before_preflight_returns(selected_api, monkeypatch, stream, fault):
    from tldw_Server_API.app.api.v1.API_Deps import billing_deps
    from tldw_Server_API.app.core.Billing.enforcement import EnforcementAction, LimitCheckResult
    from tldw_Server_API.app.core.Chat import chat_service

    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers, stream)
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.setenv("RG_ENABLED", "0")
    enforcer = SimpleNamespace(
        check_limit=AsyncMock(
            return_value=LimitCheckResult(
                category=endpoint.LimitCategory.LLM_TOKENS_MONTH.value,
                action=EnforcementAction.ALLOW,
                current=0,
                limit=1000000,
                percent_used=0.0,
            )
        ),
        apply_usage_delta=Mock(return_value=True),
        invalidate_cache=Mock(),
    )
    monkeypatch.setattr(endpoint, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(billing_deps, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(billing_deps, "billing_checks_active", AsyncMock(return_value=True))
    monkeypatch.setattr(endpoint, "get_billing_enforcer", lambda: enforcer)
    monkeypatch.setattr(billing_deps, "get_billing_enforcer", lambda: enforcer)
    monkeypatch.setitem(client.app.dependency_overrides, endpoint.get_billing_org_id, lambda: 42)
    ledger = AsyncMock()
    monkeypatch.setattr(endpoint.cost_units, "record_cost_units_for_entity", ledger)
    usage = AsyncMock()
    monkeypatch.setattr(chat_service, "log_llm_usage", usage)
    prime = endpoint._prime_provider_stream_response

    async def prime_after_completion(response, error_state):
        buffered, code, has_output, _ = await prime(response, error_state)

        async def drain():
            chunks = list(buffered)
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return tuple(chunks), code, has_output, True

        result = await asyncio.wait_for(drain(), timeout=2)
        assert ledger.await_count == 1
        return result

    monkeypatch.setattr(endpoint, "_prime_provider_stream_response", prime_after_completion)

    def provider(*args, **kwargs):
        if stream:
            return iter(['data: {"choices":[{"index":0,"delta":{"content":"Answer"}}]}\n\n', "data: [DONE]\n\n"])
        return {
            "id": "billing-control",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)
    if fault:

        def fail(*args, **kwargs):
            raise RuntimeError("private metadata failure details")

        monkeypatch.setattr(db.message_store, "_add_message_metadata_with_conn", fail)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    events = frames(response, stream)
    if fault:
        assert [event["error"]["code"] for event in events if "error" in event] == [
            "selected_durable_result_unverified"
        ]
        assert not any(event.get("success") is True for event in events)
        assert not any("tldw_history_result_v1" in event or "tldw_message_id" in event for event in events)
    else:
        assert any("tldw_history_result_v1" in event for event in events)
    assert db.count_messages_for_conversation(cid) == (1 if fault else 2)
    assert usage.await_count == 1
    total = usage.await_args.kwargs["total_tokens"]
    assert total > 0
    enforcer.apply_usage_delta.assert_called_once_with(42, endpoint.LimitCategory.LLM_TOKENS_MONTH, total)
    assert ledger.await_count == 1
    assert ledger.await_args.kwargs["entity_scope"] == "org"
    assert ledger.await_args.kwargs["entity_value"] == "42"
    assert ledger.await_args.kwargs["tokens"] == total


def test_selected_durable_receipt_wrapper_preserves_sse_control_frames(selected_api, monkeypatch):
    """Receipt decoration must not discard comment-only keepalive frames."""
    from starlette.responses import StreamingResponse

    client, db, cid, headers = selected_api
    body = body_for(client, cid, headers, True)
    controls = [": heartbeat unchanged\n\n", "event: keepalive\nid: checkpoint\nretry: 1000\n\n"]

    async def stream_response(**kwargs):
        async def accepted():
            for control in controls:
                yield control
            yield 'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(accepted(), media_type="text/event-stream")

    monkeypatch.setattr(endpoint, "execute_streaming_call", stream_response)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert all(control in response.text for control in controls), response.text
    assert (
        frames(response, True)[0]["tldw_history_admission_v1"]["input_message_id"]
        == body["tldw_turn"]["user_message_id"]
    )
    assert not any("tldw_history_result_v1" in event for event in frames(response, True))
    assert db.count_messages_for_conversation(cid) == 1
