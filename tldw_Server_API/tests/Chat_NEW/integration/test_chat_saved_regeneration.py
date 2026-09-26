"""The HTTP completion path must preserve saved response-variant ancestry."""

import json

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
def test_saved_regeneration_api_reuses_user_and_persists_variant(
    credentialed_test_client, populated_chacha_db, auth_headers, stream,
):
    """Only provider generation is mocked; request handling and storage are real."""
    db = populated_chacha_db
    conversation = db.add_conversation({"title": "Saved API regeneration"})
    user = db.add_message({"conversation_id": conversation, "sender": "user", "content": "Original question"})
    original_reply = db.add_message({"conversation_id": conversation, "sender": "assistant", "content": "Original answer"})
    overrides = credentialed_test_client.app.dependency_overrides
    overrides[get_chacha_db_for_user] = lambda: db
    try:
        response = credentialed_test_client.post(
            "/api/v1/chat/completions", headers=auth_headers,
            json={"model": "gpt-4o-mini", "conversation_id": conversation, "save_to_db": True,
                  "stream": stream, "messages": [{"role": "user", "content": "Original question"}],
                  "metadata": {"tldw_regenerate_from_message_id": original_reply}},
        )
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ") and line[6:] != "[DONE]"] if stream else [response.json()]
        saved_ids = [event.get("tldw_message_id") for event in events if event.get("tldw_message_id")]
        assert saved_ids
        rows = db.get_messages_for_conversation(conversation)
        assert [row["id"] for row in rows if row["sender"] == "user"] == [user]
        assert [(row["id"], row["parent_message_id"]) for row in rows if row["sender"] == "assistant"] == [
            (original_reply, user), (saved_ids[-1], user),
        ]
    finally:
        overrides.pop(get_chacha_db_for_user, None)
