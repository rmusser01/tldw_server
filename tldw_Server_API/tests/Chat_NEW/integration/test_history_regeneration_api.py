"""Actual authenticated H1 regeneration: one original input, alternate saved replies."""

import base64
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

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


def regenerate(client, cid, headers, selection, *, stream=False, target="a1", **extra):
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
            "metadata": {"tldw_regenerate_from_message_id": target},
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("older_branch", [False, True])
def test_regeneration_reuses_original_input_and_preserves_selected_originals(
    history_api, monkeypatch, stream, older_branch
):
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint

    provider = endpoint.perform_chat_api_call
    provider_inputs = []

    def observe(api_endpoint, messages_payload, **kwargs):
        provider_inputs.append(messages_payload)
        return provider(api_endpoint, messages_payload, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", observe)
    original_admission = answered(client, cid, headers)
    if older_branch:
        later = send(client, cid, headers, select(client, cid, headers, after="a1"), "u2")
        reply(client, cid, headers, later.json()["tldw_history_admission_v1"], "a2")
    before = db.get_messages_for_conversation(cid)
    selection = select(client, cid, headers, after="a1")
    response = regenerate(client, cid, headers, selection, stream=stream)
    assert response.status_code == 200, response.text
    frames = events(response, stream)
    admission = next(frame["tldw_history_admission_v1"] for frame in frames if "tldw_history_admission_v1" in frame)
    assert admission == original_admission
    saved = [frame["tldw_message_id"] for frame in frames if frame.get("tldw_message_id")]
    assert saved
    rows = db.get_messages_for_conversation(cid)
    assert len(provider_inputs) == 1
    assert [message["content"] for message in provider_inputs[0] if message["role"] == "user"] == [
        [{"type": "text", "text": "input u1"}]
    ]
    assert all("reply a1" not in str(message) and "input u2" not in str(message) for message in provider_inputs[0])
    assert rows[:-1] == before
    assert rows[-1]["id"] == saved[-1]
    assert rows[-1]["sender"] == "assistant"
    assert rows[-1]["parent_message_id"] == "u1"
    assert [row["id"] for row in rows if row["sender"] == "user"] == (["u1", "u2"] if older_branch else ["u1"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "boundary",
    [
        "changed-input",
        "edited-input",
        "changed-reply",
        "deleted-reply",
        "wrong-target",
        "wrong-owner",
        "scope-header",
        "stale-context",
        "extra-user",
        "extra-tool",
        "renamed-input",
        "changed-body",
        "branch-absent",
        "branch-false",
        "conflicting-markers",
        "missing-authority",
        "forged-authority",
        "missing-digest",
        "missing-scope",
    ],
)
def test_invalid_regeneration_never_reaches_provider_or_message_writes(history_api, monkeypatch, stream, boundary):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint

    client, db, cid, headers = history_api
    answered(client, cid, headers)
    selection = select(client, cid, headers, after="a1")
    target, extra = "a1", {}
    if boundary in {"changed-input", "edited-input"}:
        db.update_message("u1", {"content": "Changed input"}, 1)
        if boundary == "edited-input":
            selection = select(client, cid, headers, after="a1")
    elif boundary == "changed-reply":
        db.update_message("a1", {"content": "Changed reply"}, 1)
    elif boundary == "deleted-reply":
        db.soft_delete_message("a1", 1)
    elif boundary == "wrong-target":
        target = "u1"
    elif boundary == "wrong-owner":
        selection = {**selection, "owner_key": "foreign-owner"}
    elif boundary == "scope-header":
        headers = {**headers, "X-TLDW-Expected-User-ID": "foreign-user"}
    elif boundary == "stale-context":
        db.upsert_conversation_settings(cid, {"temperature": 0.33})
    elif boundary == "extra-user":
        extra["messages"] = [{"role": "user", "content": "input u1"}, {"role": "user", "content": "Extra input"}]
    elif boundary == "extra-tool":
        extra["messages"] = [
            {"role": "user", "content": "input u1"},
            {"role": "tool", "content": "Extra tool", "tool_call_id": "t"},
        ]
    elif boundary == "renamed-input":
        extra["messages"] = [{"role": "user", "content": "input u1", "name": "Different"}]
    elif boundary == "changed-body":
        extra["messages"] = [{"role": "user", "content": "input u1 "}]
    elif boundary == "branch-absent":
        extra["tldw_history_branch"] = None
    elif boundary == "branch-false":
        extra["tldw_history_branch"] = False
    elif boundary == "conflicting-markers":
        extra["metadata"] = {"tldw_regenerate_from_message_id": "a1", "tldw_retry_failed_turn": True}
    elif boundary in {"missing-authority", "forged-authority", "missing-digest", "missing-scope"}:
        with db.transaction() as conn:
            row = conn.execute("SELECT history_admission_json FROM messages WHERE id = ?", ("u1",)).fetchone()
            authority = json.loads(row["history_admission_json"])
            if boundary == "missing-authority":
                authority.pop("admission")
            elif boundary == "missing-digest":
                authority.pop("input_state_digest")
            elif boundary == "missing-scope":
                authority.pop("scope")
            else:
                authority["admission"]["input_message_id"] = "foreign"
            conn.execute("UPDATE messages SET history_admission_json = ? WHERE id = ?", (json.dumps(authority), "u1"))
        selection = select(client, cid, headers, after="a1")
    before = (db.get_messages_for_conversation(cid), db.get_conversation_by_id(cid), db.get_conversation_settings(cid))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid regeneration must not reach provider")

    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    response = regenerate(client, cid, headers, selection, stream=stream, target=target, **extra)
    assert response.status_code == (412 if boundary == "scope-header" else 409), response.text
    if boundary in {"missing-authority", "branch-absent", "branch-false"}:
        assert response.json()["detail"]["code"] == "unsupported_history_regeneration"
        assert response.json()["detail"]["status"] == "unsupported_history_capability"
    assert (
        db.get_messages_for_conversation(cid),
        db.get_conversation_by_id(cid),
        db.get_conversation_settings(cid),
    ) == before


def test_regeneration_refuses_an_invalid_reply_id_as_a_bad_request(history_api):
    # The endpoint passes only allow-listed statuses through; an invalid id is a
    # client error, not an unexpected 500.
    client, db, cid, headers = history_api
    answered(client, cid, headers)
    before = db.get_messages_for_conversation(cid)
    response = regenerate(client, cid, headers, select(client, cid, headers, after="a1"), target="not a valid id!")
    assert response.status_code == 400, response.text
    assert "valid saved reply ID" in response.json()["detail"]
    assert db.get_messages_for_conversation(cid) == before


@pytest.mark.parametrize("stream", [False, True])
def test_fresh_assistant_edit_can_regenerate_with_its_current_opaque_revision(history_api, monkeypatch, stream):
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    admission = answered(client, cid, headers)
    db.update_message("a1", {"content": "Saved edited answer"}, 1)
    before = db.get_messages_for_conversation(cid)
    response = regenerate(client, cid, headers, select(client, cid, headers, after="a1"), stream=stream)
    assert response.status_code == 200, response.text
    assert db.get_messages_for_conversation(cid)[:-1] == before
    assert (
        next(
            frame["tldw_history_admission_v1"]
            for frame in events(response, stream)
            if "tldw_history_admission_v1" in frame
        )
        == admission
    )


def image_part(color, detail):
    with io.BytesIO() as output:
        Image.new("RGB", (1, 1), color).save(output, format="PNG")
        data = base64.b64encode(output.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}", "detail": detail}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["", "  Literal {{char}} input  "])
def test_server_admitted_ordered_images_are_reused_once_verbatim(history_api, monkeypatch, stream, text):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    content = ([{"type": "text", "text": text}] if text else []) + [
        image_part("red", "high"),
        image_part("blue", "low"),
    ]
    original = regenerate(
        client,
        cid,
        headers,
        select(client, cid, headers),
        metadata=None,
        messages=[{"role": "user", "content": content}],
    )
    assert original.status_code == 200, original.text
    admission = original.json()["tldw_history_admission_v1"]
    target = original.json()["tldw_message_id"]
    before = db.get_messages_for_conversation(cid, strict_images=True)
    provider = endpoint.perform_chat_api_call
    provider_inputs = []

    def observe(api_endpoint, messages_payload, **kwargs):
        provider_inputs.append(messages_payload)
        return provider(api_endpoint, messages_payload, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", observe)
    selection = select(client, cid, headers, after=target)
    response = regenerate(
        client, cid, headers, selection, stream=stream, target=target, messages=[{"role": "user", "content": content}]
    )
    assert response.status_code == 200, response.text
    assert (
        next(
            frame["tldw_history_admission_v1"]
            for frame in events(response, stream)
            if "tldw_history_admission_v1" in frame
        )
        == admission
    )
    assert db.get_messages_for_conversation(cid, strict_images=True)[:-1] == before
    assert len(provider_inputs) == 1
    users = [message for message in provider_inputs[0] if message["role"] == "user"]
    assert len(users) == 1
    assert [part for part in users[0]["content"] if part["type"] == "image_url"] == content[-2:]
    if text:
        assert users[0]["content"][0]["text"].startswith("  Literal ") and users[0]["content"][0]["text"].endswith(
            " input  "
        )


@pytest.mark.parametrize("stream", [False, True])
def test_selected_ancestors_and_current_system_instructions_survive_regeneration(history_api, monkeypatch, stream):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    ancestor = send(client, cid, headers, select(client, cid, headers), "u0")
    reply(client, cid, headers, ancestor.json()["tldw_history_admission_v1"], "a0")
    original = send(client, cid, headers, select(client, cid, headers, after="a0"), "u1")
    reply(client, cid, headers, original.json()["tldw_history_admission_v1"], "a1")
    provider = endpoint.perform_chat_api_call
    payloads = []

    def observe(api_endpoint, messages_payload, **kwargs):
        payloads.append((messages_payload, kwargs))
        return provider(api_endpoint, messages_payload, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", observe)
    response = regenerate(
        client,
        cid,
        headers,
        select(client, cid, headers, after="a1"),
        stream=stream,
        messages=[{"role": "system", "content": "CURRENT_REGEN_INSTRUCTION"}, {"role": "user", "content": "input u1"}],
    )
    assert response.status_code == 200, response.text
    assert len(payloads) == 1
    messages, options = payloads[0]
    assert [message["content"] for message in messages if message["role"] == "user"] == [
        [{"type": "text", "text": "input u0"}],
        [{"type": "text", "text": "input u1"}],
    ]
    assert [message["content"] for message in messages if message["role"] == "assistant"] == [
        [{"type": "text", "text": "reply a0"}]
    ]
    assert "CURRENT_REGEN_INSTRUCTION" in str(messages) + str(options)
    assert (
        "tldw_history_selection_v1" not in options
        and "tldw_history_branch" not in options
        and "metadata" not in options
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("boundary", ["edit", "delete"])
def test_input_retirement_during_provider_call_refuses_late_reply_settlement(
    history_api, monkeypatch, stream, boundary
):
    from tldw_Server_API.app.api.v1.endpoints import chat as endpoint
    from tldw_Server_API.app.core.Chat import streaming_utils

    monkeypatch.setattr(streaming_utils, "CHAT_STREAM_INCLUDE_METADATA", False)
    client, db, cid, headers = history_api
    answered(client, cid, headers)
    selection = select(client, cid, headers, after="a1")
    entered, release = threading.Event(), threading.Event()
    provider = endpoint.perform_chat_api_call

    def held(api_endpoint, messages_payload, **kwargs):
        entered.set()
        assert release.wait(15), "test must release the held provider"
        return provider(api_endpoint, messages_payload, **kwargs)

    monkeypatch.setattr(endpoint, "perform_chat_api_call", held)
    with ThreadPoolExecutor(max_workers=1) as pool:
        operation = pool.submit(regenerate, client, cid, headers, selection, stream=stream)
        try:
            assert entered.wait(15), "regeneration must reach provider after admission reuse"
            if boundary == "edit":
                db.update_message("u1", {"content": "Changed while generating"}, 1)
            else:
                db.soft_delete_message("u1", 1)
            before = db.get_messages_for_conversation(cid)
        finally:
            release.set()
        response = operation.result(timeout=20)
    assert response.status_code in {200, 409}, response.text
    if response.status_code == 200:
        assert not any(frame.get("tldw_message_id") for frame in events(response, stream))
    assert db.get_messages_for_conversation(cid) == before
