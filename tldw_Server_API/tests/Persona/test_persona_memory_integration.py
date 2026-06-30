from __future__ import annotations

import hashlib

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Personalization_DB import (
    PersonalizationDB,
    SemanticMemory,
    UsageEvent,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.memory_integration import (
    backfill_persona_memory_from_legacy,
    persist_persona_turn,
    persist_tool_outcome,
    retrieve_top_memories,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.debug_args: list[tuple[object, ...]] = []
        self.debug_kwargs: list[dict[str, object]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append(message.format(*args, **kwargs) if args or kwargs else message)
        self.debug_args.append(args)
        self.debug_kwargs.append(dict(kwargs))


def _seed_memory_db(tmp_path, monkeypatch, *, user_id: str, enabled: bool) -> PersonalizationDB:
    base = tmp_path / "user_db"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base))
    path = DatabasePaths.get_personalization_db_path(int(user_id))
    db = PersonalizationDB(str(path))
    db.update_profile(user_id, enabled=1 if enabled else 0)
    return db


def _seed_memory_db_for_raw_user(
    tmp_path,
    monkeypatch,
    *,
    raw_user_id: str,
    enabled: bool,
) -> tuple[PersonalizationDB, str]:
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    base = tmp_path / "user_db"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base))
    normalized_user_id = mem._normalize_personalization_user_id(raw_user_id)
    path = DatabasePaths.get_personalization_db_path(normalized_user_id)
    db = PersonalizationDB(str(path))
    db.update_profile(normalized_user_id, enabled=1 if enabled else 0)
    return db, normalized_user_id


def _chacha_entries_for_user(tmp_path, monkeypatch, *, user_id: str, persona_id: str) -> list[dict]:
    base = tmp_path / "user_db"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base))
    db_path = DatabasePaths.get_chacha_db_path(int(user_id))
    db = CharactersRAGDB(str(db_path), client_id=f"persona-memory-test-{user_id}")
    try:
        return db.list_persona_memory_entries(
            user_id=user_id,
            persona_id=persona_id,
            include_archived=True,
            include_deleted=True,
            limit=500,
            offset=0,
        )
    finally:
        db.close_connection()


def test_retrieve_top_memories_respects_opt_in_and_top_k(tmp_path, monkeypatch):
    user_id = "101"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    _ = db.add_semantic_memory(
        SemanticMemory(user_id=user_id, content="Prefers concise responses.", tags=["prefs"])
    )
    _ = db.add_semantic_memory(
        SemanticMemory(user_id=user_id, content="Working on FastAPI tests.", tags=["project"])
    )

    memories = retrieve_top_memories(user_id=user_id, query_text="FastAPI", top_k=1)
    assert len(memories) == 1
    assert "FastAPI" in memories[0].content


def test_retrieve_top_memories_returns_empty_when_opted_out(tmp_path, monkeypatch):
    user_id = "102"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=False)
    _ = db.add_semantic_memory(
        SemanticMemory(user_id=user_id, content="Should not be used when opted out.", tags=["prefs"])
    )

    memories = retrieve_top_memories(user_id=user_id, query_text="used", top_k=3)
    assert memories == []


def test_retrieve_top_memories_sanitizes_fail_open_fallback_log(monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    logger_stub = _LoggerStub()

    def _fail_get_db_for_user(_user_id: str):
        raise RuntimeError("retrieval exploded with token sk-live-secret at /private/persona.db")

    monkeypatch.setattr(mem, "logger", logger_stub)
    monkeypatch.setattr(mem, "is_personalization_enabled", lambda: True)
    monkeypatch.setattr(mem, "_get_db_for_user", _fail_get_db_for_user)

    memories = retrieve_top_memories(user_id="101", query_text="secret", top_k=3)

    assert memories == []
    assert logger_stub.debugs == ["persona memory retrieval skipped"]
    assert logger_stub.debug_args == [()]
    assert logger_stub.debug_kwargs == [{}]
    rendered = "\n".join(logger_stub.debugs)
    assert "sk-live-secret" not in rendered
    assert "/private/persona.db" not in rendered
    assert "exploded" not in rendered


def test_persist_turn_and_tool_outcome_when_opted_in(tmp_path, monkeypatch):
    user_id = "103"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    session_id = "sess_memory"

    ok_turn = persist_persona_turn(
        user_id=user_id,
        session_id=session_id,
        persona_id="research_assistant",
        role="assistant",
        content="Here is your summary.",
        turn_type="assistant_delta",
        metadata={"source": "test"},
        store_as_memory=True,
    )
    ok_tool = persist_tool_outcome(
        user_id=user_id,
        session_id=session_id,
        persona_id="research_assistant",
        tool_name="rag_search",
        step_idx=0,
        outcome={"ok": True, "result": {"hits": 2}},
    )
    assert ok_turn is True
    assert ok_tool is True

    events = db.list_recent_events(user_id=user_id, limit=20)
    assert len(events) >= 2
    assert any(evt["type"] == "persona.turn" for evt in events)

    memories, _ = db.list_semantic_memories(user_id=user_id, limit=20, offset=0)
    contents = [item["content"] for item in memories]
    assert any("Here is your summary." in c for c in contents)
    assert any("Tool=rag_search" in c for c in contents)


def test_persist_tool_outcome_is_bounded_and_privacy_safe(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "1031"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    monkeypatch.setattr(mem, "_get_persona_tool_outcome_summary_max_chars", lambda: 256)

    ok_tool = persist_tool_outcome(
        user_id=user_id,
        session_id="sess_tool_outcome_bounds",
        persona_id="research_assistant",
        tool_name="rag_search",
        step_idx=0,
        outcome={
            "ok": True,
            "reason_code": "DONE",
            "output": {
                "sensitive": "api-key-live-123456789",
                "blob": "x" * 6000,
            },
        },
        store_as_memory=True,
    )
    assert ok_tool is True

    memories, _ = db.list_semantic_memories(user_id=user_id, limit=20, offset=0)
    tool_entries = [str(item.get("content") or "") for item in memories if "Tool=rag_search" in str(item.get("content") or "")]
    assert tool_entries
    tool_entry = tool_entries[0]
    assert len(tool_entry) <= 256
    assert "api-key-live-123456789" not in tool_entry
    assert "output_digest" in tool_entry


def test_persist_turn_skips_when_opted_out(tmp_path, monkeypatch):
    user_id = "104"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=False)
    session_id = "sess_memory_off"

    ok_turn = persist_persona_turn(
        user_id=user_id,
        session_id=session_id,
        persona_id="research_assistant",
        role="assistant",
        content="Should not persist.",
        turn_type="assistant_delta",
        metadata=None,
        store_as_memory=True,
    )
    assert ok_turn is False
    assert db.list_recent_events(user_id=user_id, limit=10) == []
    memories, total = db.list_semantic_memories(user_id=user_id, limit=10, offset=0)
    assert total == 0
    assert memories == []


def test_persist_persona_turn_sanitizes_fail_open_fallback_log(monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    logger_stub = _LoggerStub()

    def _fail_get_db_for_user(_user_id: str):
        raise RuntimeError("persistence exploded with token sk-live-secret at /private/persona.db")

    monkeypatch.setattr(mem, "logger", logger_stub)
    monkeypatch.setattr(mem, "is_personalization_enabled", lambda: True)
    monkeypatch.setattr(mem, "_get_db_for_user", _fail_get_db_for_user)

    ok = persist_persona_turn(
        user_id="103",
        session_id="sess_memory",
        persona_id="research_assistant",
        role="assistant",
        content="Should fail open.",
        turn_type="assistant_delta",
    )

    assert ok is False
    assert logger_stub.debugs == ["persona turn persistence skipped"]
    assert logger_stub.debug_args == [()]
    assert logger_stub.debug_kwargs == [{}]
    rendered = "\n".join(logger_stub.debugs)
    assert "sk-live-secret" not in rendered
    assert "/private/persona.db" not in rendered
    assert "exploded" not in rendered


def test_dual_read_chacha_first_falls_back_to_legacy(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "201"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    _ = db.add_semantic_memory(
        SemanticMemory(user_id=user_id, content="Legacy fallback memory marker.", tags=["legacy"])
    )
    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_first_fallback_legacy")

    memories = retrieve_top_memories(
        user_id=user_id,
        persona_id="research_assistant",
        query_text="fallback memory marker",
        top_k=5,
    )
    assert memories
    assert "Legacy fallback memory marker." in {item.content for item in memories}


def test_dual_read_prefers_chacha_when_available(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "202"
    _ = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    monkeypatch.setattr(mem, "_get_persona_memory_write_mode", lambda: "chacha_only")
    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_first_fallback_legacy")

    ok = persist_persona_turn(
        user_id=user_id,
        session_id="sess_chacha_first",
        persona_id="research_assistant",
        role="assistant",
        content="Chacha-first memory marker.",
        turn_type="assistant_delta",
        metadata={"source": "test"},
        store_as_memory=True,
    )
    assert ok is True

    memories = retrieve_top_memories(
        user_id=user_id,
        persona_id="research_assistant",
        query_text="chacha-first memory marker",
        top_k=3,
    )
    assert memories
    assert memories[0].content == "Chacha-first memory marker."


def test_write_mode_switch_supports_rollback_between_legacy_and_chacha(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "203"
    persona_id = "research_assistant"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)

    monkeypatch.setattr(mem, "_get_persona_memory_write_mode", lambda: "dual_write")
    ok = persist_persona_turn(
        user_id=user_id,
        session_id="sess_dual_write",
        persona_id=persona_id,
        role="assistant",
        content="Rollback switch memory marker.",
        turn_type="assistant_delta",
        metadata={"source": "test"},
        store_as_memory=True,
    )
    assert ok is True

    legacy_memories, _ = db.list_semantic_memories(user_id=user_id, limit=20, offset=0)
    assert any("Rollback switch memory marker." == str(item["content"]) for item in legacy_memories)

    chacha_entries = _chacha_entries_for_user(tmp_path, monkeypatch, user_id=user_id, persona_id=persona_id)
    assert any(
        str(entry.get("memory_type")) == "summary"
        and str(entry.get("content")) == "Rollback switch memory marker."
        for entry in chacha_entries
    )

    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "legacy_only")
    legacy_only = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="rollback switch memory marker",
        top_k=5,
    )
    assert legacy_only
    assert legacy_only[0].content == "Rollback switch memory marker."

    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_only")
    chacha_only = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="rollback switch memory marker",
        top_k=5,
    )
    assert chacha_only
    assert chacha_only[0].content == "Rollback switch memory marker."


def test_backfill_legacy_to_chacha_is_idempotent_and_resumable(tmp_path, monkeypatch):
    user_id = "204"
    persona_id = "research_assistant"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    _ = db.add_semantic_memory(SemanticMemory(user_id=user_id, content="legacy-memory-1", tags=["m"]))
    _ = db.add_semantic_memory(SemanticMemory(user_id=user_id, content="legacy-memory-2", tags=["m"]))
    _ = db.add_semantic_memory(SemanticMemory(user_id=user_id, content="legacy-memory-3", tags=["m"]))
    _ = db.insert_usage_event(
        UsageEvent(user_id=user_id, type="persona.turn", resource_id="s1", tags=["persona"], metadata={"i": 1})
    )
    _ = db.insert_usage_event(
        UsageEvent(user_id=user_id, type="persona.turn", resource_id="s2", tags=["persona"], metadata={"i": 2})
    )
    _ = db.insert_usage_event(
        UsageEvent(user_id=user_id, type="persona.turn", resource_id="s3", tags=["persona"], metadata={"i": 3})
    )

    r1 = backfill_persona_memory_from_legacy(
        user_id=user_id,
        persona_id=persona_id,
        batch_size=2,
        checkpoint=None,
        include_usage_events=True,
    )
    assert r1.processed_semantic == 2
    assert r1.inserted_semantic + r1.skipped_semantic == 2
    assert r1.completed is False

    r2 = backfill_persona_memory_from_legacy(
        user_id=user_id,
        persona_id=persona_id,
        batch_size=2,
        checkpoint=r1.next_checkpoint,
        include_usage_events=True,
    )
    assert r2.processed_semantic == 1
    assert r2.processed_usage_events == 2
    assert r2.completed is False

    r3 = backfill_persona_memory_from_legacy(
        user_id=user_id,
        persona_id=persona_id,
        batch_size=2,
        checkpoint=r2.next_checkpoint,
        include_usage_events=True,
    )
    assert r3.processed_semantic == 0
    assert r3.processed_usage_events == 1
    assert r3.completed is True

    rows = _chacha_entries_for_user(tmp_path, monkeypatch, user_id=user_id, persona_id=persona_id)
    memory_types = [str(row.get("memory_type")) for row in rows]
    assert memory_types.count("legacy_semantic") == 3
    assert memory_types.count("legacy_usage_event") == 3

    rerun = backfill_persona_memory_from_legacy(
        user_id=user_id,
        persona_id=persona_id,
        batch_size=10,
        checkpoint=None,
        include_usage_events=True,
    )
    assert rerun.inserted_semantic == 0
    assert rerun.inserted_usage_events == 0
    assert rerun.skipped_semantic >= 3
    assert rerun.skipped_usage_events >= 3


def test_backfill_respects_opt_in_gate(tmp_path, monkeypatch):
    user_id = "205"
    persona_id = "research_assistant"
    db = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=False)
    _ = db.add_semantic_memory(SemanticMemory(user_id=user_id, content="should-not-backfill", tags=["m"]))

    result = backfill_persona_memory_from_legacy(
        user_id=user_id,
        persona_id=persona_id,
        batch_size=10,
        checkpoint=None,
        include_usage_events=True,
    )
    assert result.inserted_semantic == 0
    assert result.inserted_usage_events == 0
    assert result.completed is True

    rows = _chacha_entries_for_user(tmp_path, monkeypatch, user_id=user_id, persona_id=persona_id)
    assert rows == []


def test_chacha_memory_retrieval_is_scope_snapshot_namespaced(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "301"
    persona_id = "research_assistant"
    _ = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    monkeypatch.setattr(mem, "_get_persona_memory_write_mode", lambda: "chacha_only")
    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_only")

    assert persist_persona_turn(
        user_id=user_id,
        session_id="sess_scope_a",
        persona_id=persona_id,
        role="assistant",
        content="scope-a-memory",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_a",
    )
    assert persist_persona_turn(
        user_id=user_id,
        session_id="sess_scope_b",
        persona_id=persona_id,
        role="assistant",
        content="scope-b-memory",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_b",
    )

    scope_a_memories = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="scope-",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_a",
        session_id="sess_scope_a",
    )
    assert [item.content for item in scope_a_memories] == ["scope-a-memory"]

    scope_b_memories = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="scope-",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_b",
        session_id="sess_scope_b",
    )
    assert [item.content for item in scope_b_memories] == ["scope-b-memory"]

    missing_scope = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="scope-",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id=None,
        session_id="sess_scope_a",
    )
    assert missing_scope == []


def test_chacha_memory_retrieval_is_session_namespaced_for_session_mode(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "302"
    persona_id = "research_assistant"
    _ = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    monkeypatch.setattr(mem, "_get_persona_memory_write_mode", lambda: "chacha_only")
    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_only")

    assert persist_persona_turn(
        user_id=user_id,
        session_id="sess_one",
        persona_id=persona_id,
        role="assistant",
        content="session-one-memory",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="session_scoped",
        scope_snapshot_id="scope_same",
    )
    assert persist_persona_turn(
        user_id=user_id,
        session_id="sess_two",
        persona_id=persona_id,
        role="assistant",
        content="session-two-memory",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="session_scoped",
        scope_snapshot_id="scope_same",
    )

    sess_one_memories = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="session-",
        top_k=10,
        runtime_mode="session_scoped",
        scope_snapshot_id="scope_same",
        session_id="sess_one",
    )
    assert [item.content for item in sess_one_memories] == ["session-one-memory"]

    sess_two_memories = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="session-",
        top_k=10,
        runtime_mode="session_scoped",
        scope_snapshot_id="scope_same",
        session_id="sess_two",
    )
    assert [item.content for item in sess_two_memories] == ["session-two-memory"]

    missing_session = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="session-",
        top_k=10,
        runtime_mode="session_scoped",
        scope_snapshot_id="scope_same",
        session_id=None,
    )
    assert missing_session == []


def test_persistent_scope_namespace_falls_back_to_session_hash_when_scope_missing(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "303"
    persona_id = "research_assistant"
    _ = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    monkeypatch.setattr(mem, "_get_persona_memory_write_mode", lambda: "chacha_only")
    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_only")

    assert persist_persona_turn(
        user_id=user_id,
        session_id="sess_persistent_fallback_a",
        persona_id=persona_id,
        role="assistant",
        content="persistent-fallback-a",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="persistent_scoped",
        scope_snapshot_id=None,
    )
    assert persist_persona_turn(
        user_id=user_id,
        session_id="sess_persistent_fallback_b",
        persona_id=persona_id,
        role="assistant",
        content="persistent-fallback-b",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="persistent_scoped",
        scope_snapshot_id=None,
    )

    scope_a_memories = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="persistent-fallback-",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id=None,
        session_id="sess_persistent_fallback_a",
    )
    scope_b_memories = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="persistent-fallback-",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id=None,
        session_id="sess_persistent_fallback_b",
    )

    assert [item.content for item in scope_a_memories] == ["persistent-fallback-a"]
    assert [item.content for item in scope_b_memories] == ["persistent-fallback-b"]

    chacha_entries = _chacha_entries_for_user(tmp_path, monkeypatch, user_id=user_id, persona_id=persona_id)
    summary_entries = [entry for entry in chacha_entries if str(entry.get("memory_type")) == "summary"]
    assert len(summary_entries) >= 2
    scope_values = {str(entry.get("scope_snapshot_id") or "") for entry in summary_entries}
    assert all(value.startswith("persistent_fallback_sid_") for value in scope_values)
    assert len(scope_values) >= 2
    assert all(entry.get("session_id") in (None, "") for entry in summary_entries)


def test_persistent_scope_missing_backfills_legacy_unscoped_entries(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_id = "304"
    persona_id = "research_assistant"
    _ = _seed_memory_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    monkeypatch.setattr(mem, "_get_persona_memory_write_mode", lambda: "chacha_only")
    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_only")

    db_path = DatabasePaths.get_chacha_db_path(int(user_id))
    chacha_db = CharactersRAGDB(str(db_path), client_id="persona-memory-legacy-unscoped-test")
    try:
        _ = chacha_db.create_persona_profile(
            {
                "id": persona_id,
                "user_id": user_id,
                "name": "Legacy Unscoped Persona",
                "mode": "persistent_scoped",
                "system_prompt": "",
                "is_active": True,
            }
        )
        _ = chacha_db.add_persona_memory_entry(
            {
                "persona_id": persona_id,
                "user_id": user_id,
                "memory_type": "summary",
                "content": "legacy-unscoped-memory",
                "scope_snapshot_id": None,
                "session_id": None,
                "salience": 0.6,
            }
        )
    finally:
        chacha_db.close_connection()

    retrieved = retrieve_top_memories(
        user_id=user_id,
        persona_id=persona_id,
        query_text="legacy-unscoped-memory",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id=None,
        session_id="sess_backfill_lookup",
    )
    assert [item.content for item in retrieved] == ["legacy-unscoped-memory"]

    post_rows = _chacha_entries_for_user(tmp_path, monkeypatch, user_id=user_id, persona_id=persona_id)
    target_rows = [row for row in post_rows if str(row.get("content")) == "legacy-unscoped-memory"]
    assert len(target_rows) == 1
    target_row = target_rows[0]
    assert str(target_row.get("scope_snapshot_id") or "").startswith("persistent_legacy_pid_")
    assert target_row.get("session_id") in (None, "")


def test_non_numeric_user_ids_use_collision_safe_mapping(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Persona import memory_integration as mem

    user_a = "user-94641"
    user_b = "user-115688"
    legacy_a = int.from_bytes(hashlib.sha1(user_a.encode("utf-8"), usedforsecurity=False).digest()[:4], "big")
    legacy_b = int.from_bytes(hashlib.sha1(user_b.encode("utf-8"), usedforsecurity=False).digest()[:4], "big")
    assert legacy_a == legacy_b

    _, normalized_a = _seed_memory_db_for_raw_user(tmp_path, monkeypatch, raw_user_id=user_a, enabled=True)
    _, normalized_b = _seed_memory_db_for_raw_user(tmp_path, monkeypatch, raw_user_id=user_b, enabled=True)
    assert normalized_a != normalized_b

    persona_id = "research_assistant"
    monkeypatch.setattr(mem, "_get_persona_memory_write_mode", lambda: "chacha_only")
    monkeypatch.setattr(mem, "_get_persona_memory_read_mode", lambda: "chacha_only")

    assert persist_persona_turn(
        user_id=user_a,
        session_id="sess_user_a",
        persona_id=persona_id,
        role="assistant",
        content="memory-for-user-a",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_shared",
    )
    assert persist_persona_turn(
        user_id=user_b,
        session_id="sess_user_b",
        persona_id=persona_id,
        role="assistant",
        content="memory-for-user-b",
        turn_type="assistant_delta",
        store_as_memory=True,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_shared",
    )

    user_a_memories = retrieve_top_memories(
        user_id=user_a,
        persona_id=persona_id,
        query_text="memory-for-user-",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_shared",
        session_id="sess_user_a",
    )
    user_b_memories = retrieve_top_memories(
        user_id=user_b,
        persona_id=persona_id,
        query_text="memory-for-user-",
        top_k=10,
        runtime_mode="persistent_scoped",
        scope_snapshot_id="scope_shared",
        session_id="sess_user_b",
    )

    assert [item.content for item in user_a_memories] == ["memory-for-user-a"]
    assert [item.content for item in user_b_memories] == ["memory-for-user-b"]
