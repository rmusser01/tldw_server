import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from tldw_Server_API.app.core.Persona.session_manager import PlanConfirmationError, SessionManager


pytestmark = pytest.mark.unit


def test_session_manager_plan_roundtrip_and_consume():
    manager = SessionManager()
    manager.put_plan(
        session_id="sess_1",
        user_id="user_1",
        persona_id="research_assistant",
        plan_id="plan_1",
        steps=[
            {"idx": 1, "tool": "summarize", "args": {}},
            {"idx": 0, "tool": "rag_search", "args": {"query": "hello"}},
        ],
    )

    pending = manager.get_plan(session_id="sess_1", plan_id="plan_1", user_id="user_1")
    assert pending is not None
    assert [step.idx for step in pending.steps] == [0, 1]

    consumed = manager.get_plan(
        session_id="sess_1",
        plan_id="plan_1",
        user_id="user_1",
        consume=True,
    )
    assert consumed is not None
    assert manager.get_plan(session_id="sess_1", plan_id="plan_1", user_id="user_1") is None


def test_session_manager_plan_lookup_rejects_session_or_user_mismatch():
    manager = SessionManager()
    manager.put_plan(
        session_id="sess_1",
        user_id="user_1",
        persona_id="research_assistant",
        plan_id="plan_1",
        steps=[{"idx": 0, "tool": "rag_search", "args": {"query": "hello"}}],
    )

    assert manager.get_plan(session_id="sess_2", plan_id="plan_1", user_id="user_1") is None
    assert manager.get_plan(session_id="sess_1", plan_id="plan_1", user_id="user_2") is None


def test_session_manager_rejects_resume_session_owner_mismatch():
    manager = SessionManager()
    _ = manager.create(user_id="user_1", persona_id="research_assistant", resume_session_id="sess_1")

    with pytest.raises(ValueError, match="ownership mismatch"):
        manager.create(user_id="user_2", persona_id="research_assistant", resume_session_id="sess_1")


def test_session_manager_rejects_resume_session_persona_mismatch():
    manager = SessionManager()
    _ = manager.create(user_id="user_1", persona_id="research_assistant", resume_session_id="sess_1")

    with pytest.raises(ValueError, match="persona mismatch"):
        manager.create(user_id="user_1", persona_id="writer_assistant", resume_session_id="sess_1")


def test_session_manager_clear_plans():
    manager = SessionManager()
    manager.put_plan(
        session_id="sess_clear",
        user_id="user_1",
        persona_id="research_assistant",
        plan_id="plan_1",
        steps=[{"idx": 0, "tool": "rag_search", "args": {"query": "hello"}}],
    )
    manager.put_plan(
        session_id="sess_clear",
        user_id="user_1",
        persona_id="research_assistant",
        plan_id="plan_2",
        steps=[{"idx": 0, "tool": "summarize", "args": {}}],
    )

    cleared = manager.clear_plans(session_id="sess_clear", user_id="user_1")
    assert cleared == 2
    assert manager.get_plan(session_id="sess_clear", plan_id="plan_1", user_id="user_1") is None
    assert manager.get_plan(session_id="sess_clear", plan_id="plan_2", user_id="user_1") is None


def test_session_manager_turn_append_and_list_limit():
    manager = SessionManager()
    manager.append_turn(
        session_id="sess_turns",
        user_id="user_1",
        persona_id="research_assistant",
        role="user",
        content="hello",
        turn_type="user_message",
    )
    manager.append_turn(
        session_id="sess_turns",
        user_id="user_1",
        persona_id="research_assistant",
        role="assistant",
        content="hi there",
        turn_type="assistant_delta",
    )

    turns = manager.list_turns(session_id="sess_turns", user_id="user_1")
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"

    limited = manager.list_turns(session_id="sess_turns", user_id="user_1", limit=1)
    assert len(limited) == 1
    assert limited[0]["role"] == "assistant"


def test_session_manager_list_sessions_and_snapshot():
    manager = SessionManager()
    manager.append_turn(
        session_id="sess_1",
        user_id="user_1",
        persona_id="research_assistant",
        role="user",
        content="hello",
        turn_type="user_message",
    )
    manager.append_turn(
        session_id="sess_2",
        user_id="user_1",
        persona_id="research_assistant",
        role="user",
        content="another",
        turn_type="user_message",
    )
    manager.append_turn(
        session_id="sess_3",
        user_id="user_2",
        persona_id="research_assistant",
        role="user",
        content="foreign",
        turn_type="user_message",
    )

    listed = manager.list_sessions(user_id="user_1")
    assert len(listed) == 2
    assert all(item["session_id"] in {"sess_1", "sess_2"} for item in listed)
    assert all(item["turn_count"] == 1 for item in listed)

    snapshot = manager.get_session_snapshot(session_id="sess_1", user_id="user_1", limit_turns=10)
    assert snapshot is not None
    assert snapshot["session_id"] == "sess_1"
    assert snapshot["turn_count"] == 1
    assert len(snapshot["turns"]) == 1

    assert manager.get_session_snapshot(session_id="sess_1", user_id="user_2") is None


def test_session_manager_preferences_roundtrip():
    manager = SessionManager()
    _ = manager.create(user_id="user_1", persona_id="research_assistant", resume_session_id="sess_prefs")

    updated = manager.update_preferences(
        session_id="sess_prefs",
        user_id="user_1",
        preferences={"use_memory_context": False, "memory_top_k": 2},
    )
    assert updated["use_memory_context"] is False
    assert updated["memory_top_k"] == 2

    prefs = manager.get_preferences(session_id="sess_prefs", user_id="user_1")
    assert prefs["use_memory_context"] is False
    assert prefs["memory_top_k"] == 2

    with pytest.raises(ValueError, match="ownership mismatch"):
        manager.update_preferences(
            session_id="sess_prefs",
            user_id="user_2",
            preferences={"use_memory_context": True},
        )


def test_session_manager_turn_cap_applies_oldest_trim():
    manager = SessionManager(max_turns_per_session=2)
    manager.append_turn(
        session_id="sess_turn_cap",
        user_id="user_1",
        persona_id="research_assistant",
        role="user",
        content="first",
        turn_type="user_message",
    )
    manager.append_turn(
        session_id="sess_turn_cap",
        user_id="user_1",
        persona_id="research_assistant",
        role="assistant",
        content="second",
        turn_type="assistant_delta",
    )
    manager.append_turn(
        session_id="sess_turn_cap",
        user_id="user_1",
        persona_id="research_assistant",
        role="assistant",
        content="third",
        turn_type="assistant_delta",
    )

    turns = manager.list_turns(session_id="sess_turn_cap", user_id="user_1")
    assert len(turns) == 2
    assert [turn["content"] for turn in turns] == ["second", "third"]


def test_session_manager_pending_plan_cap_evicts_oldest():
    manager = SessionManager(max_pending_plans_per_session=1)
    manager.put_plan(
        session_id="sess_plan_cap",
        user_id="user_1",
        persona_id="research_assistant",
        plan_id="plan_1",
        steps=[{"idx": 0, "tool": "rag_search", "args": {"query": "one"}}],
    )
    manager.put_plan(
        session_id="sess_plan_cap",
        user_id="user_1",
        persona_id="research_assistant",
        plan_id="plan_2",
        steps=[{"idx": 0, "tool": "summarize", "args": {}}],
    )

    assert manager.get_plan(session_id="sess_plan_cap", plan_id="plan_1", user_id="user_1") is None
    assert manager.get_plan(session_id="sess_plan_cap", plan_id="plan_2", user_id="user_1") is not None


def test_session_manager_prunes_expired_sessions_on_access():
    manager = SessionManager(session_ttl_seconds=1)
    manager.create(user_id="user_1", persona_id="research_assistant", resume_session_id="sess_ttl")
    manager._sessions["sess_ttl"].updated_at = datetime.now(timezone.utc) - timedelta(seconds=5)  # noqa: SLF001

    assert manager.get("sess_ttl") is None


def test_session_manager_applies_payload_caps_with_truncation_markers():
    manager = SessionManager(
        max_turn_content_chars=32,
        max_turn_metadata_chars=48,
    )
    manager.append_turn(
        session_id="sess_retention_caps",
        user_id="user_1",
        persona_id="research_assistant",
        role="tool",
        content="secret-value-" + ("x" * 200),
        turn_type="tool_result",
        metadata={
            "api_key": "super-secret-token-value",
            "blob": "y" * 400,
            "nested": {"child": "z" * 100},
        },
    )

    turns = manager.list_turns(session_id="sess_retention_caps", user_id="user_1")
    assert len(turns) == 1
    turn = turns[0]
    assert len(turn["content"]) <= 32
    assert turn["content"].endswith("[truncated]")

    metadata = turn["metadata"]
    assert metadata.get("_truncated") is True
    assert metadata.get("original_char_count", 0) > 48
    serialized_metadata = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
    assert "super-secret-token-value" not in serialized_metadata

    retention = metadata.get("_retention") or {}
    assert retention.get("content_truncated") is True
    assert retention.get("metadata_truncated") is True


def test_latest_pending_plan_snapshot_is_owned_detached_and_non_consuming():
    manager = SessionManager()
    for plan_id in ("old", "latest"):
        manager.put_plan(
            session_id="s",
            user_id="u",
            persona_id="p",
            plan_id=plan_id,
            steps=[{"idx": 0, "tool": "rag_search", "args": {"query": plan_id}}],
        )
    snapshot = manager.get_latest_plan_snapshot(session_id="s", user_id="u", persona_id="p")
    assert snapshot["plan_id"] == "latest"
    assert snapshot["steps"][0]["args"] == {"query": "latest"}
    assert "policy" not in snapshot["steps"][0]
    snapshot["steps"][0]["args"]["query"] = "mutated"
    assert manager.get_plan(session_id="s", user_id="u", plan_id="latest").steps[0].args == {"query": "latest"}
    assert manager.get_latest_plan_snapshot(session_id="s", user_id="other", persona_id="p") is None
    assert manager.get_latest_plan_snapshot(session_id="s", user_id="u", persona_id="other") is None
    manager.get_plan(session_id="s", user_id="u", plan_id="latest", consume=True)
    assert manager.get_latest_plan_snapshot(session_id="s", user_id="u", persona_id="p")["plan_id"] == "old"


def test_latest_pending_plan_read_does_not_revive_expired_runtime():
    manager = SessionManager(session_ttl_seconds=1)
    manager.put_plan(
        session_id="s", user_id="u", persona_id="p", plan_id="plan", steps=[{"idx": 0, "tool": "rag_search"}]
    )
    manager.get("s").updated_at = datetime.now(timezone.utc) - timedelta(seconds=2)
    assert manager.get_latest_plan_snapshot(session_id="s", user_id="u", persona_id="p") is None
    assert manager.get("s") is None


@pytest.mark.parametrize(
    "steps",
    [
        [{"idx": i, "tool": "rag_search"} for i in range(101)],
        [{"idx": 0, "tool": "rag_search", "args": {"query": "x" * 65536}}],
    ],
)
def test_latest_pending_plan_projection_omits_oversized_plan_without_consuming(steps):
    manager = SessionManager()
    manager.put_plan(session_id="s", user_id="u", persona_id="p", plan_id="plan", steps=steps)
    assert manager.get_latest_plan_snapshot(session_id="s", user_id="u", persona_id="p") is None
    assert manager.get_plan(session_id="s", user_id="u", plan_id="plan") is not None


@pytest.fixture
def confirmation_plan():
    manager = SessionManager()
    pending = manager.put_plan(
        session_id="confirmation-session",
        user_id="owner",
        persona_id="research_assistant",
        plan_id="confirmation-plan",
        steps=[{"idx": 0, "tool": "rag_search", "args": {}}],
        requires_persisted_session=True,
    )
    return manager, pending


@pytest.mark.parametrize(
    "user_id,session_exists,session_terminal,reason",
    [
        ("other", True, False, "PLAN_NOT_FOUND"),
        ("owner", True, True, "SESSION_TERMINAL"),
        ("owner", False, False, "SESSION_NOT_FOUND"),
    ],
)
def test_confirmation_rejects_without_consuming(confirmation_plan, user_id, session_exists, session_terminal, reason):
    manager, pending = confirmation_plan
    with pytest.raises(PlanConfirmationError) as error:
        manager.consume_plan_for_confirmation(
            session_id=pending.session_id,
            plan_id=pending.plan_id,
            user_id=user_id,
            session_exists=session_exists,
            session_terminal=session_terminal,
        )
    assert error.value.reason_code == reason
    assert manager.get_plan(session_id=pending.session_id, plan_id=pending.plan_id, user_id="owner") is pending


def test_confirmation_prunes_expired_plan(confirmation_plan):
    manager, pending = confirmation_plan
    manager._sessions[pending.session_id].updated_at = datetime.now(timezone.utc) - timedelta(days=2)
    with pytest.raises(PlanConfirmationError) as error:
        manager.consume_plan_for_confirmation(
            session_id=pending.session_id,
            plan_id=pending.plan_id,
            user_id="owner",
            session_exists=True,
            session_terminal=False,
        )
    assert error.value.reason_code == "PLAN_NOT_FOUND"


def test_confirmation_preserves_runtime_only_sessions(confirmation_plan):
    manager, pending = confirmation_plan
    pending.requires_persisted_session = False
    assert (
        manager.consume_plan_for_confirmation(
            session_id=pending.session_id,
            plan_id=pending.plan_id,
            user_id="owner",
            session_exists=False,
            session_terminal=False,
        )
        is pending
    )


def test_concurrent_confirmations_consume_exactly_once(confirmation_plan):
    manager, pending = confirmation_plan
    start = Barrier(2)

    def confirm():
        start.wait(timeout=5)
        try:
            return manager.consume_plan_for_confirmation(
                session_id=pending.session_id,
                plan_id=pending.plan_id,
                user_id="owner",
                session_exists=True,
                session_terminal=False,
            )
        except PlanConfirmationError as error:
            return error.reason_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: confirm(), range(2)))
    assert sum(item is pending for item in outcomes) == 1
    assert outcomes.count("PLAN_NOT_FOUND") == 1
    assert manager.get_plan(session_id=pending.session_id, plan_id=pending.plan_id, user_id="owner") is None
