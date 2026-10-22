"""Native H1 Continue (CM-01): extend a saved reply as a new server-settled alternative.

Continue reuses the saved question's admission like Regenerate, but keeps the
selected reply in the model context and saves the original text plus the new
continuation as a sibling reply. The original question and reply are unchanged.
"""

import json

import pytest

from tldw_Server_API.tests.Chat_NEW.integration.test_history_branch_admission_api import (
    history_api as history_api,
)
from tldw_Server_API.tests.Chat_NEW.integration.test_history_branch_admission_api import (
    reply,
    select,
    send,
)


def answered(client, cid, headers):
    response = send(client, cid, headers, select(client, cid, headers), "u1")
    assert response.status_code == 201, response.text
    admission = response.json()["tldw_history_admission_v1"]
    reply(client, cid, headers, admission, "a1")
    return admission


def continue_reply(client, cid, headers, selection, *, stream=False, target="a1", **extra):
    from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter

    initialize_rate_limiter()
    return client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "conversation_id": cid,
            "save_to_db": True,
            "stream": stream,
            "messages": [{"role": "user", "content": "input u1"}],
            "metadata": {"tldw_continue_from_message_id": target},
            "tldw_history_selection_v1": selection,
            "tldw_history_branch": True,
            **extra,
        },
    )


def events(response, stream):
    return (
        [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        if stream
        else [response.json()]
    )


def generated_text(frames, stream):
    if not stream:
        return frames[0]["choices"][0]["message"]["content"]
    return "".join(
        (frame.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for frame in frames
        if frame.get("choices")
    )


@pytest.mark.parametrize("stream", [False, True])
def test_continue_saves_original_plus_continuation_as_sibling_reply(history_api, monkeypatch, stream):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    provider = endpoint.perform_chat_api_call
    provider_inputs = []

    def observe(api_endpoint, messages_payload, **kwargs):
        provider_inputs.append(messages_payload)
        return provider(api_endpoint, messages_payload, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", observe)
    original_admission = answered(client, cid, headers)
    before = db.get_messages_for_conversation(cid)
    response = continue_reply(client, cid, headers, select(client, cid, headers, after="a1"), stream=stream)
    assert response.status_code == 200, response.text
    frames = events(response, stream)
    admission = next(frame["tldw_history_admission_v1"] for frame in frames if "tldw_history_admission_v1" in frame)
    assert admission == original_admission
    saved = [frame["tldw_message_id"] for frame in frames if frame.get("tldw_message_id")]
    assert saved
    continuation = generated_text(frames, stream)
    assert continuation
    # The model continues from the saved reply: it is the final context message.
    assert len(provider_inputs) == 1
    context = [message for message in provider_inputs[0] if message["role"] != "system"]
    assert context[-1]["role"] == "assistant"
    assert "reply a1" in json.dumps(context[-1]["content"])
    assert [message["content"] for message in context if message["role"] == "user"] == [
        [{"type": "text", "text": "input u1"}]
    ]
    rows = db.get_messages_for_conversation(cid)
    assert rows[:-1] == before
    assert rows[-1]["id"] == saved[-1]
    assert rows[-1]["sender"] == "assistant"
    assert rows[-1]["parent_message_id"] == "u1"
    assert rows[-1]["content"] == "reply a1" + continuation
    assert [row["id"] for row in rows if row["sender"] == "user"] == ["u1"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "boundary",
    [
        "changed-reply",
        "wrong-target",
        "branch-false",
        "missing-authority",
        "with-regenerate",
        "with-retry",
        "without-selection",
        "changed-input",
    ],
)
def test_invalid_continue_never_reaches_provider_or_message_writes(history_api, monkeypatch, stream, boundary):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint

    client, db, cid, headers = history_api
    answered(client, cid, headers)
    selection = select(client, cid, headers, after="a1")
    target, extra = "a1", {}
    if boundary == "changed-reply":
        db.update_message("a1", {"content": "Changed reply"}, 1)
    elif boundary == "wrong-target":
        target = "u1"
    elif boundary == "branch-false":
        extra["tldw_history_branch"] = False
    elif boundary == "with-regenerate":
        extra["metadata"] = {"tldw_continue_from_message_id": "a1", "tldw_regenerate_from_message_id": "a1"}
    elif boundary == "with-retry":
        extra["metadata"] = {"tldw_continue_from_message_id": "a1", "tldw_retry_failed_turn": True}
    elif boundary == "without-selection":
        extra["tldw_history_selection_v1"] = None
        extra["tldw_history_branch"] = None
    elif boundary == "changed-input":
        extra["messages"] = [{"role": "user", "content": "input u1 changed"}]
    elif boundary == "missing-authority":
        with db.transaction() as conn:
            row = conn.execute("SELECT history_admission_json FROM messages WHERE id = ?", ("u1",)).fetchone()
            authority = json.loads(row["history_admission_json"])
            authority.pop("admission")
            conn.execute("UPDATE messages SET history_admission_json = ? WHERE id = ?", (json.dumps(authority), "u1"))
        selection = select(client, cid, headers, after="a1")
    before = (db.get_messages_for_conversation(cid), db.get_conversation_by_id(cid))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid continuation must not reach provider")

    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    response = continue_reply(client, cid, headers, selection, stream=stream, target=target, **extra)
    assert response.status_code == 409, response.text
    if boundary in {"missing-authority", "branch-false", "without-selection"}:
        assert response.json()["detail"]["code"] == "unsupported_history_continuation"
        assert response.json()["detail"]["status"] == "unsupported_history_capability"
    assert (db.get_messages_for_conversation(cid), db.get_conversation_by_id(cid)) == before


def test_continue_requires_a_valid_reply_id(history_api):
    client, db, cid, headers = history_api
    answered(client, cid, headers)
    before = db.get_messages_for_conversation(cid)
    response = continue_reply(client, cid, headers, select(client, cid, headers, after="a1"), target="not a valid id!")
    assert response.status_code == 400, response.text
    assert db.get_messages_for_conversation(cid) == before
