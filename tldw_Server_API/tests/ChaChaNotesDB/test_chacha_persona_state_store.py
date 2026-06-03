import ast
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.persona_state_store import (
    PersonaStateStore,
)


pytestmark = pytest.mark.unit


_DELEGATED_PERSONA_STATE_METHODS = {
    "_ensure_persona_live_voice_session_summaries_table",
    "upsert_persona_live_voice_session_summary",
    "list_persona_live_voice_session_summaries",
    "get_persona_live_voice_session_summary",
    "_ensure_persona_setup_events_table",
    "_persona_setup_event_row_to_dict",
    "_decode_persona_json_object",
    "record_persona_setup_event",
    "list_persona_setup_events",
    "get_persona_setup_analytics_summary",
    "_persona_profile_row_to_dict",
    "_persona_buddy_row_to_dict",
    "_decode_persona_json_list",
    "_persona_visual_pack_row_to_dict",
    "_persona_visual_asset_row_to_dict",
    "_persona_visual_candidate_row_to_dict",
    "_normalize_persona_visual_enum",
    "_require_persona_visual_pack_owner",
    "_persona_scope_rule_row_to_dict",
    "_persona_policy_rule_row_to_dict",
    "_persona_session_row_to_dict",
    "_normalize_persona_session_activity_surface",
    "_persona_memory_row_to_dict",
    "_persona_exemplar_row_to_dict",
    "_require_active_persona_profile_owner",
    "_normalize_persona_exemplar_tone",
    "_normalize_persona_exemplar_tags",
    "create_persona_exemplar",
    "get_persona_exemplar",
    "list_persona_exemplars",
    "update_persona_exemplar",
    "soft_delete_persona_exemplar",
    "create_persona_profile",
    "get_persona_profile",
    "list_persona_profiles",
    "update_persona_profile",
    "soft_delete_persona_profile",
    "restore_persona_profile",
    "get_persona_buddy",
    "list_persona_buddies",
    "upsert_persona_buddy",
    "create_persona_visual_pack",
    "get_persona_visual_pack",
    "list_persona_visual_packs",
    "get_active_persona_visual_pack",
    "activate_persona_visual_pack",
    "deactivate_persona_visual_pack",
    "update_persona_visual_pack_manifest",
    "create_persona_visual_asset",
    "get_persona_visual_asset",
    "list_persona_visual_assets",
    "create_persona_visual_candidate",
    "get_persona_visual_candidate",
    "list_persona_visual_candidates",
    "update_persona_visual_candidate_status",
    "list_persona_scope_rules",
    "replace_persona_scope_rules",
    "list_persona_policy_rules",
    "replace_persona_policy_rules",
    "create_persona_session",
    "get_persona_session",
    "list_persona_sessions",
    "update_persona_session",
    "add_persona_memory_entry",
    "list_persona_memory_entries",
    "get_persona_memory_entry_by_id",
    "count_persona_memory_entries",
    "set_persona_memory_archived",
    "update_persona_memory_entry",
    "backfill_persona_memory_scope_namespace",
    "soft_delete_persona_memory_entry",
}


def _class_method_names(class_obj: type[object]) -> set[str]:
    source_path = Path(inspect.getsourcefile(class_obj) or "")
    assert source_path.exists()
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_obj.__name__:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_obj.__name__} not found in {source_path}")


@pytest.fixture()
def db(tmp_path):
    instance = CharactersRAGDB(
        db_path=str(tmp_path / "persona_state_store.sqlite"),
        client_id="persona-state-store-user",
    )
    yield instance
    instance.close_connection()


@pytest.fixture()
def store(db):
    return PersonaStateStore(db)


def test_persona_state_store_owns_delegated_methods_without_monolith_duplicates(db, monkeypatch):
    class_method_names = _class_method_names(CharactersRAGDB)
    assert _DELEGATED_PERSONA_STATE_METHODS.isdisjoint(class_method_names)

    captured: dict[str, object] = {}

    def _fake_create_persona_profile(profile_data: dict[str, object]) -> str:
        captured["profile_data"] = profile_data
        return "persona-from-store"

    monkeypatch.setattr(db.persona_state_store, "create_persona_profile", _fake_create_persona_profile)

    assert db.create_persona_profile({"user_id": "user-1", "name": "Delegated Persona"}) == "persona-from-store"
    assert captured["profile_data"] == {"user_id": "user-1", "name": "Delegated Persona"}


def test_persona_profile_rules_soft_delete_and_restore_roundtrip(store):
    persona_id = store.create_persona_profile(
        {
            "user_id": "user-1",
            "name": "Research Persona",
            "mode": "session_scoped",
            "system_prompt": "Focus on signal.",
        }
    )

    profile = store.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    assert profile["name"] == "Research Persona"
    assert profile["mode"] == "session_scoped"

    listed_profiles = store.list_persona_profiles(user_id="user-1")
    assert [item["id"] for item in listed_profiles] == [persona_id]

    assert store.update_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        update_data={
            "mode": "persistent_scoped",
            "use_persona_state_context_default": False,
        },
        expected_version=int(profile["version"]),
    )

    updated_profile = store.get_persona_profile(persona_id, user_id="user-1")
    assert updated_profile is not None
    assert updated_profile["mode"] == "persistent_scoped"
    assert updated_profile["use_persona_state_context_default"] is False

    assert store.replace_persona_scope_rules(
        persona_id=persona_id,
        user_id="user-1",
        rules=[
            {"rule_type": "conversation_id", "rule_value": "conv-a", "include": True},
            {"rule_type": "media_tag", "rule_value": "physics", "include": True},
        ],
    ) == 2
    assert store.replace_persona_policy_rules(
        persona_id=persona_id,
        user_id="user-1",
        rules=[
            {"rule_kind": "mcp_tool", "rule_name": "knowledge.search", "allowed": True},
            {
                "rule_kind": "skill",
                "rule_name": "digest",
                "allowed": True,
                "require_confirmation": False,
                "max_calls_per_turn": 2,
            },
        ],
    ) == 2

    scope_rules = store.list_persona_scope_rules(persona_id=persona_id, user_id="user-1")
    policy_rules = store.list_persona_policy_rules(persona_id=persona_id, user_id="user-1")
    assert {item["rule_type"] for item in scope_rules} == {"conversation_id", "media_tag"}
    assert {item["rule_kind"] for item in policy_rules} == {"mcp_tool", "skill"}

    assert store.soft_delete_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        expected_version=int(updated_profile["version"]),
    )

    deleted_profile = store.get_persona_profile(
        persona_id,
        user_id="user-1",
        include_deleted=True,
    )
    assert deleted_profile is not None
    assert deleted_profile["deleted"] is True

    assert store.restore_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        expected_version=int(deleted_profile["version"]),
    )

    restored = store.get_persona_profile(persona_id, user_id="user-1")
    assert restored is not None
    assert restored["deleted"] is False
    assert restored["is_active"] is True


def test_persona_session_and_memory_roundtrip(store):
    persona_id = store.create_persona_profile({"user_id": "user-1", "name": "Memory Persona"})

    session_id = store.create_persona_session(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "mode": "persistent_scoped",
            "reuse_allowed": True,
            "status": "active",
            "scope_snapshot_json": {"conversations": ["conv-a"]},
            "preferences_json": {"memory_top_k": 4, "use_memory_context": True},
        }
    )
    session = store.get_persona_session(session_id, user_id="user-1")
    assert session is not None
    assert session["persona_id"] == persona_id
    assert session["preferences"]["memory_top_k"] == 4

    assert store.update_persona_session(
        session_id=session_id,
        user_id="user-1",
        update_data={
            "status": "paused",
            "activity_surface": "companion.conversation",
            "preferences_json": {"memory_top_k": 7, "use_memory_context": False},
        },
        expected_version=int(session["version"]),
    )

    paused_session = store.get_persona_session(session_id, user_id="user-1")
    assert paused_session is not None
    assert paused_session["status"] == "paused"
    assert paused_session["activity_surface"] == "companion.conversation"
    assert paused_session["preferences"]["memory_top_k"] == 7

    scoped_entry_id = store.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "fact",
            "content": "User likes astrophysics.",
            "scope_snapshot_id": "scope-a",
            "session_id": session_id,
            "salience": 0.8,
        }
    )
    unscoped_entry_id = store.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "preference",
            "content": "Prefer concise explanations.",
        }
    )

    listed_entries = store.list_persona_memory_entries(user_id="user-1", persona_id=persona_id)
    assert {item["id"] for item in listed_entries} == {scoped_entry_id, unscoped_entry_id}
    assert store.count_persona_memory_entries(user_id="user-1", persona_id=persona_id) == 2

    assert store.backfill_persona_memory_scope_namespace(
        user_id="user-1",
        persona_id=persona_id,
        scope_snapshot_id="scope-backfill",
    ) == 1

    backfilled_entry = store.get_persona_memory_entry_by_id(
        entry_id=unscoped_entry_id,
        user_id="user-1",
        persona_id=persona_id,
    )
    assert backfilled_entry is not None
    assert backfilled_entry["scope_snapshot_id"] == "scope-backfill"

    assert store.set_persona_memory_archived(
        entry_id=unscoped_entry_id,
        user_id="user-1",
        persona_id=persona_id,
        archived=True,
    )
    assert store.count_persona_memory_entries(user_id="user-1", persona_id=persona_id) == 1
    assert store.count_persona_memory_entries(
        user_id="user-1",
        persona_id=persona_id,
        include_archived=True,
    ) == 2

    assert store.update_persona_memory_entry(
        entry_id=scoped_entry_id,
        user_id="user-1",
        persona_id=persona_id,
        update_data={
            "content": "User likes astrophysics and cosmology.",
            "salience": 0.9,
        },
    )
    updated_entry = store.get_persona_memory_entry_by_id(
        entry_id=scoped_entry_id,
        user_id="user-1",
        persona_id=persona_id,
    )
    assert updated_entry is not None
    assert updated_entry["content"] == "User likes astrophysics and cosmology."
    assert updated_entry["salience"] == pytest.approx(0.9)

    assert store.soft_delete_persona_memory_entry(
        entry_id=scoped_entry_id,
        user_id="user-1",
        persona_id=persona_id,
    )
    deleted_entry = store.get_persona_memory_entry_by_id(
        entry_id=scoped_entry_id,
        user_id="user-1",
        persona_id=persona_id,
        include_deleted=True,
    )
    assert deleted_entry is not None
    assert deleted_entry["deleted"] is True


def test_persona_memory_filter_builder_normalizes_shared_filters(store: PersonaStateStore) -> None:
    where_sql, params = store._build_persona_memory_where_clause(
        entry_id="  entry-1  ",
        user_id="  user-1  ",
        persona_id="  persona-1  ",
        memory_type="  fact  ",
        scope_snapshot_id="  scope-a  ",
        session_id="  session-a  ",
        include_archived=False,
        include_deleted=False,
    )

    assert where_sql == (
        "id = ? AND user_id = ? AND persona_id = ? AND memory_type = ? "
        "AND scope_snapshot_id = ? AND session_id = ? AND archived = 0 AND deleted = 0"
    )
    assert params == ["entry-1", "user-1", "persona-1", "fact", "scope-a", "session-a"]


@pytest.mark.parametrize(
    ("filter_kwargs", "message"),
    [
        (
            {"scope_snapshot_id": "scope-a", "require_missing_scope_snapshot_id": True},
            "scope_snapshot_id",
        ),
        (
            {"session_id": "session-a", "require_missing_session_id": True},
            "session_id",
        ),
    ],
)
def test_persona_memory_filter_builder_rejects_conflicting_scope_filters(
    store: PersonaStateStore,
    filter_kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        store._build_persona_memory_where_clause(user_id="user-1", **filter_kwargs)


def test_persona_memory_facade_preserves_filter_lifecycle(db: CharactersRAGDB) -> None:
    persona_id = db.create_persona_profile({"user_id": "user-1", "name": "Facade Memory Persona"})
    matching_entry_id = db.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": " fact ",
            "content": "User prefers concise research notes.",
            "scope_snapshot_id": " scope-a ",
            "session_id": " session-a ",
        }
    )
    db.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "preference",
            "content": "User prefers examples.",
            "scope_snapshot_id": "scope-b",
            "session_id": "session-b",
        }
    )

    listed_entries = db.list_persona_memory_entries(
        user_id="  user-1  ",
        persona_id=f"  {persona_id}  ",
        memory_type=" fact ",
        scope_snapshot_id=" scope-a ",
        session_id=" session-a ",
    )
    assert [item["id"] for item in listed_entries] == [matching_entry_id]
    assert db.count_persona_memory_entries(
        user_id="  user-1  ",
        persona_id=f"  {persona_id}  ",
        memory_type=" fact ",
    ) == 1

    assert db.set_persona_memory_archived(
        entry_id=f"  {matching_entry_id}  ",
        user_id="  user-1  ",
        persona_id=f"  {persona_id}  ",
        archived=True,
    )
    assert db.list_persona_memory_entries(user_id="user-1", persona_id=persona_id, memory_type="fact") == []
    assert db.count_persona_memory_entries(
        user_id="user-1",
        persona_id=persona_id,
        memory_type="fact",
        include_archived=True,
    ) == 1

    assert db.soft_delete_persona_memory_entry(
        entry_id=f"  {matching_entry_id}  ",
        user_id="  user-1  ",
        persona_id=f"  {persona_id}  ",
    )
    deleted_entry = db.get_persona_memory_entry_by_id(
        entry_id=f"  {matching_entry_id}  ",
        user_id="  user-1  ",
        persona_id=f"  {persona_id}  ",
        include_deleted=True,
    )
    assert deleted_entry is not None
    assert deleted_entry["deleted"] is True


def test_persona_buddy_roundtrip(store):
    persona_id = store.create_persona_profile({"user_id": "user-1", "name": "Buddy Persona"})

    persisted = store.upsert_persona_buddy(
        persona_id=persona_id,
        user_id="user-1",
        derivation_version=1,
        source_fingerprint="fp-1",
        derived_core={
            "species_id": "owl",
            "silhouette_id": "owl_round",
            "palette_id": "moss",
            "behavior_family": "steady",
            "expression_profile": "warm",
        },
        overlay_preferences={"accessory_id": "scarf", "eye_style": "sleepy"},
    )

    assert persisted["persona_id"] == persona_id
    assert persisted["derived_core"]["species_id"] == "owl"
    assert persisted["overlay_preferences"]["accessory_id"] == "scarf"

    fetched = store.get_persona_buddy(persona_id=persona_id, user_id="user-1")
    listed = store.list_persona_buddies(user_id="user-1", persona_ids=[persona_id, "missing"])
    assert fetched is not None
    assert fetched["resolved_profile"]["species_id"] == "owl"
    assert fetched["resolved_profile"]["compatibility_status"] == "exact"
    assert listed[persona_id] is not None
    assert listed["missing"] is None


def test_persona_exemplar_roundtrip(store):
    persona_id = store.create_persona_profile({"user_id": "user-1", "name": "Exemplar Persona"})

    exemplar_id = store.create_persona_exemplar(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "kind": "style",
            "content": "Use crisp, grounded phrasing.",
            "tone": "Analytical",
            "scenario_tags": ["research"],
            "capability_tags": ["summaries"],
            "priority": 5,
        }
    )

    exemplar = store.get_persona_exemplar(
        exemplar_id=exemplar_id,
        persona_id=persona_id,
        user_id="user-1",
    )
    assert exemplar is not None
    assert exemplar["tone"] == "analytical"
    assert exemplar["scenario_tags"] == ["research"]

    listed = store.list_persona_exemplars(user_id="user-1", persona_id=persona_id)
    assert [item["id"] for item in listed] == [exemplar_id]

    assert store.update_persona_exemplar(
        exemplar_id=exemplar_id,
        persona_id=persona_id,
        user_id="user-1",
        update_data={"notes": "Updated note", "enabled": False},
    )

    updated = store.get_persona_exemplar(
        exemplar_id=exemplar_id,
        persona_id=persona_id,
        user_id="user-1",
        include_disabled=True,
    )
    assert updated is not None
    assert updated["notes"] == "Updated note"
    assert updated["enabled"] is False

    assert store.soft_delete_persona_exemplar(
        exemplar_id=exemplar_id,
        persona_id=persona_id,
        user_id="user-1",
    )
    deleted = store.get_persona_exemplar(
        exemplar_id=exemplar_id,
        persona_id=persona_id,
        user_id="user-1",
        include_deleted=True,
        include_disabled=True,
    )
    assert deleted is not None
    assert deleted["deleted"] is True


def test_persona_exemplar_facade_normalizes_without_monolith_helper_fallback(db, monkeypatch):
    persona_id = db.create_persona_profile({"user_id": "user-1", "name": "Facade Persona"})

    def _fail_on_monolith_helper(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("persona exemplar normalization used a monolith fallback")

    monkeypatch.setattr(db, "_normalize_exemplar_enum", _fail_on_monolith_helper)
    monkeypatch.setattr(db, "_normalize_exemplar_string_list", _fail_on_monolith_helper)
    monkeypatch.setattr(db, "_normalize_persona_exemplar_tags", _fail_on_monolith_helper, raising=False)

    exemplar_id = db.create_persona_exemplar(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "kind": "STYLE",
            "content": "Prefer grounded synthesis.",
            "tone": "Analytical",
            "scenario_tags": ["Research", "research"],
            "capability_tags": "[\"Summaries\", \"summaries\"]",
            "source_type": "MANUAL",
        }
    )

    exemplar = db.get_persona_exemplar(
        exemplar_id=exemplar_id,
        persona_id=persona_id,
        user_id="user-1",
    )
    assert exemplar is not None
    assert exemplar["kind"] == "style"
    assert exemplar["tone"] == "analytical"
    assert exemplar["scenario_tags"] == ["research"]
    assert exemplar["capability_tags"] == ["summaries"]
    assert exemplar["source_type"] == "manual"


def test_persona_setup_and_live_voice_analytics_roundtrip(store):
    first_event = store.record_persona_setup_event(
        user_id=7,
        persona_id="persona-analytics",
        event_id="event-1",
        run_id="run-1",
        event_type="setup_started",
        step="welcome",
        metadata={"surface": "voice"},
    )
    deduped_event = store.record_persona_setup_event(
        user_id=7,
        persona_id="persona-analytics",
        event_id="event-1",
        run_id="run-1",
        event_type="setup_started",
        step="welcome",
    )
    store.record_persona_setup_event(
        user_id=7,
        persona_id="persona-analytics",
        event_id="event-2",
        run_id="run-1",
        event_type="setup_completed",
        completion_type="live_session",
        action_target="persona-chat",
    )

    assert first_event["deduped"] is False
    assert first_event["metadata"]["surface"] == "voice"
    assert deduped_event["deduped"] is True

    assert store.upsert_persona_live_voice_session_summary(
        user_id=7,
        persona_id="persona-analytics",
        session_id="session-1",
        started_at="2026-04-19T12:00:00Z",
        auto_commit_enabled=True,
        commit_source="vad_auto",
    )
    assert store.upsert_persona_live_voice_session_summary(
        user_id=7,
        persona_id="persona-analytics",
        session_id="session-1",
        finalize=True,
        text_only_tts_increment=2,
    )

    session_summary = store.get_persona_live_voice_session_summary(
        user_id=7,
        persona_id="persona-analytics",
        session_id="session-1",
    )
    analytics_summary = store.get_persona_setup_analytics_summary(
        user_id=7,
        persona_id="persona-analytics",
    )
    listed_summaries = store.list_persona_live_voice_session_summaries(
        user_id=7,
        persona_id="persona-analytics",
        days=36500,
    )

    assert session_summary is not None
    assert session_summary["total_committed_turns"] == 1
    assert session_summary["text_only_tts_count"] == 2
    assert [item["session_id"] for item in listed_summaries] == ["session-1"]
    assert analytics_summary["summary"]["total_runs"] == 1
    assert analytics_summary["summary"]["completed_runs"] == 1
    assert analytics_summary["summary"]["live_session_completion_count"] == 1
