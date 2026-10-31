import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    DEFAULT_CHARACTER_NAME,
    get_chacha_db_for_user,
)


def _build_branch_conversation(populated_chacha_db):
    char = populated_chacha_db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
    assert char
    conv_id = populated_chacha_db.add_conversation(
        {"character_id": char["id"], "title": "Continuation Integration Branch"}
    )
    root_id = populated_chacha_db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "root-int"}
    )
    anchor_id = populated_chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "anchor-int",
            "parent_message_id": root_id,
        }
    )
    populated_chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "tip-int",
            "parent_message_id": anchor_id,
        }
    )
    return conv_id, anchor_id


@pytest.mark.integration
def test_branch_continuation_returns_metadata_and_parent_link(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
):
    def override_get_db():
        return populated_chacha_db

    credentialed_test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    try:
        conv_id, anchor_id = _build_branch_conversation(populated_chacha_db)

        response = credentialed_test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "conversation_id": conv_id,
                "save_to_db": True,
                "messages": [{"role": "user", "content": "continue please"}],
                "tldw_continuation": {
                    "from_message_id": anchor_id,
                    "mode": "branch",
                },
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload.get("tldw_continuation", {}).get("applied") is True
        assert payload.get("tldw_continuation", {}).get("mode") == "branch"
        assert payload.get("tldw_continuation", {}).get("from_message_id") == anchor_id

        saved_id = payload.get("tldw_message_id")
        assert isinstance(saved_id, str) and saved_id
        saved = populated_chacha_db.get_message_by_id(saved_id)
        assert saved is not None
        assert saved.get("parent_message_id") == anchor_id
    finally:
        credentialed_test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.integration
def test_append_continuation_non_tip_returns_409(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
):
    def override_get_db():
        return populated_chacha_db

    credentialed_test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    try:
        char = populated_chacha_db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        assert char
        conv_id = populated_chacha_db.add_conversation(
            {"character_id": char["id"], "title": "Continuation Integration Append"}
        )
        root_id = populated_chacha_db.add_message(
            {"conversation_id": conv_id, "sender": "user", "content": "append-root-int"}
        )
        anchor_id = populated_chacha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "assistant",
                "content": "append-anchor-int",
                "parent_message_id": root_id,
            }
        )
        populated_chacha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "assistant",
                "content": "append-latest-int",
                "parent_message_id": root_id,
            }
        )

        response = credentialed_test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "conversation_id": conv_id,
                "messages": [{"role": "user", "content": "append please"}],
                "tldw_continuation": {
                    "from_message_id": anchor_id,
                    "mode": "append",
                },
            },
            headers=auth_headers,
        )

        assert response.status_code == 409
        detail = response.json().get("detail")
        assert "Append continuation requires" in str(detail)
    finally:
        credentialed_test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.parametrize("case", ["no_skills", "stale", "unsupported_skills"])
def test_versioned_completion_never_discovers_mutable_skills(
    credentialed_test_client,
    populated_chacha_db,
    auth_headers,
    monkeypatch,
    tmp_path,
    case,
):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
    from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
    from tldw_Server_API.app.core.Skills.skills_service import SkillsService

    client, db = credentialed_test_client, populated_chacha_db
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        cid = db.add_conversation({"character_id": None, "client_id": "1", "title": "Read-only skill fence"})
        captured = client.post(
            f"/api/v1/chat/conversations/{cid}/history/selection",
            headers=auth_headers,
            json={
                "purpose": "send",
                "view": {
                    "view_session_id": "skills",
                    "conversation_id": cid,
                    "interpretation": {"kind": "parent_graph_v1"},
                    "cursor": {"kind": "empty"},
                    "selection_revision": 1,
                },
            },
        ).json()
        selection = resolve_history_selection(captured["snapshot"], captured["view"], "send", "skills-lease")[
            "selection"
        ]
        if case == "stale":
            db.upsert_conversation_settings(cid, {"model": "changed"})
        if case == "unsupported_skills":
            installed = tmp_path / "skills" / "visible"
            installed.mkdir(parents=True)
            (installed / "SKILL.md").write_text("---\nname: visible\n---\nInstructions")
            recovery = tmp_path / "skills" / ".replacing-preserved.v1"
            recovery.mkdir()
            (recovery / "marker").write_text("must remain unchanged")
        before_files = {
            str(p.relative_to(tmp_path)): p.read_bytes() for p in (tmp_path / "skills").rglob("*") if p.is_file()
        }
        before_settings = db.get_conversation_settings(cid)
        with db.transaction() as conn:
            before_registry = db.backend.table_exists("skill_registry", connection=conn)
        constructions = []

        def forbid_service(*args, **kwargs):
            constructions.append(True)
            raise AssertionError("Versioned endpoint must never construct SkillsService")

        monkeypatch.setattr(SkillsService, "__init__", forbid_service)
        monkeypatch.setattr(endpoint.DatabasePaths, "get_user_base_directory", lambda *_: tmp_path)
        seen_tools = []
        original_builder = endpoint._build_context_and_messages_compat

        async def record_tools(**kwargs):
            seen_tools.extend(kwargs["request_data"].tools or [])
            return await original_builder(**kwargs)

        monkeypatch.setattr(endpoint, "_build_context_and_messages_compat", record_tools)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "explicit_tool",
                    "description": "explicit",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        response = client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "gpt-4o-mini",
                "conversation_id": cid,
                "save_to_db": True,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": tools,
                "tldw_history_selection_v1": selection,
            },
        )
        assert constructions == []
        assert response.status_code == (200 if case == "no_skills" else 409), response.text
        if case == "no_skills":
            normalized = [
                item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item for item in seen_tools
            ]
            assert normalized == tools
            assert not (tmp_path / "skills").exists()
        else:
            assert db.count_messages_for_conversation(cid) == 0
        assert db.get_conversation_settings(cid) == before_settings
        assert {
            str(p.relative_to(tmp_path)): p.read_bytes() for p in (tmp_path / "skills").rglob("*") if p.is_file()
        } == before_files
        with db.transaction() as conn:
            assert db.backend.table_exists("skill_registry", connection=conn) is before_registry
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
