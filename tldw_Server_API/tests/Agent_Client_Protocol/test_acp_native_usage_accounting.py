"""Reported native usage through the real SQLite persistence and store boundary."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore

pytestmark = pytest.mark.unit


@pytest.fixture
def owned_db(tmp_path):
    db = ACPSessionsDB(db_path=str(tmp_path / "native_usage.db"))
    db.register_session(session_id="owned", user_id=7, agent_type="opencode", name="Native", cwd="/tmp/owned")  # nosec B108 # isolated session metadata; no filesystem access
    yield db
    db.close()


@pytest.mark.parametrize("reported,expected", [
    ({"inputTokens": 201, "outputTokens": 27, "totalTokens": 7930, "thoughtTokens": 22, "cachedReadTokens": 7680},
     {"prompt_tokens": 201, "completion_tokens": 27, "total_tokens": 7930}),
    ({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 11},
     {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 11}),
    ({"input_tokens": 3, "output_tokens": 5},
     {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}),
    ({"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
     {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    ({"prompt_tokens": 0, "inputTokens": 201, "completion_tokens": 0, "outputTokens": 27, "total_tokens": 0, "totalTokens": 7930},
     {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    ({"inputTokens": 3, "outputTokens": 5, "totalTokens": -1},
     {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}),
    ({}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    ({"used": 7881, "size": 200000}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
])
def test_record_prompt_preserves_reported_usage_and_exact_native_answer(owned_db, reported, expected):
    result = {"stopReason": "end_turn", "usage": reported, "_meta": {}}
    usage = owned_db.record_prompt("owned", [{"role": "user", "content": "Read the fixture"}], result,
                                  assistant_text="ORBIT-742 repository fixture.")
    assert usage == expected  # nosec B101 # pytest regression oracle, not production authorization
    saved = owned_db.get_session("owned")
    assert {key: saved[key] for key in expected} == expected  # nosec B101 # pytest regression oracle, not production authorization
    messages = owned_db.get_messages("owned")
    assert [message["content"] for message in messages] == ["Read the fixture", "ORBIT-742 repository fixture."]  # nosec B101 # pytest regression oracle, not production authorization
    assert messages[-1]["raw_data"] == result  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.parametrize("reported", [
    {"inputTokens": -1, "outputTokens": "27", "totalTokens": True},
    {"inputTokens": float("inf"), "outputTokens": [], "totalTokens": -3},
    ["invalid"], None,
])
def test_invalid_usage_cannot_change_counters_or_prevent_transcript_recording(owned_db, reported):
    result = {"usage": reported}
    usage = owned_db.record_prompt("owned", [{"role": "user", "content": "Question"}], result, assistant_text="Answer")
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # nosec B101 # pytest regression oracle, not production authorization
    assert owned_db.get_session("owned")["message_count"] == 2  # nosec B101 # pytest regression oracle, not production authorization
    assert owned_db.get_messages("owned")[-1]["content"] == "Answer"  # nosec B101 # pytest regression oracle, not production authorization


@pytest.mark.asyncio
async def test_store_accumulates_native_turn_totals_without_adding_cached_or_thought_tokens(owned_db):
    store = ACPSessionStore(db=owned_db)
    first = {"inputTokens": 201, "outputTokens": 27, "totalTokens": 7930, "thoughtTokens": 22, "cachedReadTokens": 7680}
    await store.record_prompt("owned", [{"role": "user", "content": "First"}], {"usage": first}, assistant_text="First answer")
    turn = await store.record_prompt("owned", [{"role": "user", "content": "Second"}],
                                    {"usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}}, assistant_text="Second answer")
    record = await store.get_session("owned")
    assert turn.to_dict() == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}  # nosec B101 # pytest regression oracle, not production authorization
    assert record.usage.to_dict() == {"prompt_tokens": 203, "completion_tokens": 30, "total_tokens": 7935}  # nosec B101 # pytest regression oracle, not production authorization
    assert record.message_count == 4  # nosec B101 # pytest regression oracle, not production authorization
    assert [message["content"] for message in record.messages] == ["First", "First answer", "Second", "Second answer"]  # nosec B101 # pytest regression oracle, not production authorization
