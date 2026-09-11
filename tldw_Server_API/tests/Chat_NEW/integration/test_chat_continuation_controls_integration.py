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
