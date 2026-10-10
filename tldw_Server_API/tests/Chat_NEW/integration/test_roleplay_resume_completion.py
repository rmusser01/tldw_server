"""Authenticated protected Resume admission and generation contract coverage."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import roleplay_resume_fence
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def resume_api(credentialed_test_client, populated_chacha_db, auth_headers):
    db = CharactersRAGDB(db_path=populated_chacha_db.db_path_str, client_id="1", owner_user_id="1")
    client = credentialed_test_client
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    card = db.add_character_card(
        {"name": "Ari", "description": "Original saved archivist", "system_prompt": "Preserve continuity."}
    )
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Protected Resume", "client_id": "1"},
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "resume-ancestor", "sender": "user", "content": "Earlier question"},
            {
                "id": "resume-prior",
                "sender": "assistant",
                "content": "Earlier answer",
                "parent_message_id": "resume-ancestor",
            },
        ],
    )
    yield client, db, cid, auth_headers
    client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
    db.close_connection()


def input_request(db, cid):
    return {
        "message_id": "resume-chosen",
        "content": "Original proposal?\n  Keep spaces.  ",
        "expected": roleplay_resume_fence(db.get_roleplay_resume_state(cid)),
    }


def post_input(client, cid, headers, body):
    return client.post(f"/api/v1/chats/{cid}/roleplay/messages", headers=headers, json=body)


def test_idempotent_authenticated_append_returns_original_graph_and_fence(resume_api):
    client, db, cid, headers = resume_api
    body = input_request(db, cid)
    accepted = post_input(client, cid, headers, body)
    assert accepted.status_code == 201, accepted.text
    original = accepted.json()
    assert original["parent_message_id"] == "resume-prior"
    assert original["fence"]["tail"] == {"message_id": "resume-chosen", "message_version": 1}
    db.update_message("resume-ancestor", {"content": "Changed after admission"}, 1)
    replay = post_input(client, cid, headers, body)
    assert replay.status_code == 201, replay.text
    assert replay.json() == original
    changed = post_input(client, cid, headers, {**body, "content": "Conflicting replacement"})
    assert changed.status_code == 409, changed.text
    assert changed.json()["detail"] == {"status": "resume_conflict", "code": "idempotency_conflict"}
    assert db.count_messages_for_conversation(cid) == 3


@pytest.mark.parametrize("mutation", ["ancestor-edit", "append", "settings"])
def test_authenticated_resume_prior_fence_never_silently_branches(resume_api, mutation):
    client, db, cid, headers = resume_api
    body = input_request(db, cid)
    if mutation == "ancestor-edit":
        db.update_message("resume-ancestor", {"content": "Changed before admission"}, 1)
    elif mutation == "append":
        db.add_message(
            {
                "id": "other-tab",
                "conversation_id": cid,
                "sender": "user",
                "content": "Other tab",
                "parent_message_id": "resume-prior",
            }
        )
    else:
        db.upsert_conversation_settings(cid, db.get_roleplay_resume_state(cid)["settings"])
    before = db.count_messages_for_conversation(cid)
    response = post_input(client, cid, headers, body)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] in {"history_fence_changed", "settings_changed"}
    assert db.get_message_by_id("resume-chosen") is None
    assert db.count_messages_for_conversation(cid) == before


def test_resume_append_is_authenticated_before_any_write(resume_api):
    client, db, cid, headers = resume_api
    response = post_input(client, cid, {}, input_request(db, cid))
    assert response.status_code == 401, response.text
    assert db.count_messages_for_conversation(cid) == 2


def test_enrolled_sync_refuses_resume_before_any_append(resume_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages as endpoint

    client, db, cid, headers = resume_api
    monkeypatch.setattr(endpoint, "_active_message_sync_service", lambda *args: object())
    response = post_input(client, cid, headers, input_request(db, cid))
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"status": "resume_conflict", "code": "resume_sync_unsupported"}
    assert db.count_messages_for_conversation(cid) == 2


def test_append_quota_is_atomic_and_identical_replay_does_not_spend_it_again(resume_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages as endpoint
    from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter

    client, db, cid, headers = resume_api
    limiter = CharacterRateLimiter(max_messages_per_chat=3, enabled=False)
    monkeypatch.setattr(endpoint, "get_character_rate_limiter", lambda: limiter)
    body = input_request(db, cid)
    accepted = post_input(client, cid, headers, body)
    assert accepted.status_code == 201, accepted.text
    assert post_input(client, cid, headers, body).json() == accepted.json()
    next_body = {
        **body,
        "message_id": "quota-refused",
        "expected": roleplay_resume_fence(db.get_roleplay_resume_state(cid)),
    }
    before = db.get_roleplay_resume_state(cid)["history_version"]
    refused = post_input(client, cid, headers, next_body)
    assert refused.status_code == 403, refused.text
    assert db.get_message_by_id("quota-refused") is None
    assert db.get_roleplay_resume_state(cid)["history_version"] == before
    assert db.count_messages_for_conversation(cid) == 3


def test_resume_append_conceals_foreign_owner_before_any_write(resume_api):
    client, db, cid, headers = resume_api
    body = input_request(db, cid)
    foreign = db.add_conversation({"id": "foreign-conversation", "title": "Foreign", "client_id": "another-owner"})
    response = post_input(client, foreign, headers, body)
    assert response.status_code == 404, response.text
    assert db.count_messages_for_conversation(foreign) == 0
    assert db.count_messages_for_conversation(cid) == 2


def test_global_resume_never_ignores_an_explicit_workspace_scope(resume_api):
    client, db, cid, headers = resume_api
    response = client.post(
        f"/api/v1/chats/{cid}/roleplay/messages?scope_type=workspace&workspace_id=other-workspace",
        headers=headers,
        json=input_request(db, cid),
    )
    assert response.status_code == 404, response.text
    assert db.count_messages_for_conversation(cid) == 2
