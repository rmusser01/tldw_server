"""Owned attachment listing and failed-turn recovery through the actual API/DB."""
from __future__ import annotations

import base64
import io
import json

import pytest
from fastapi import HTTPException
from PIL import Image

from tldw_Server_API.app.api.v1.endpoints import character_messages
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
from tldw_Server_API.tests.Chat.integration import test_persona_backed_chat_conversations as fixtures

persona_chat_db = fixtures.persona_chat_db
persona_chat_client = fixtures.persona_chat_client


def image_data(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    data = output.getvalue()
    return data, "data:image/png;base64," + base64.b64encode(data).decode()


def image_body(chat_id, text, url, local_id="local-image"):
    body = fixtures._chat_completion_body(chat_id)
    body["history_message_order"] = "asc"
    # This is generateHistory/ChatTldw's real image-first request order.
    parts = [{"type": "image_url", "image_url": {"url": url}}]
    if text:
        parts.append({"type": "text", "text": text})
    body["messages"] = [{"role": "user", "content": parts}]
    body["metadata"] = {"tldw_client_message_id": local_id}
    return body


@pytest.mark.parametrize("context", [False, True])
def test_opt_in_images_keep_default_payload_and_owned_pagination(persona_chat_client, persona_chat_db, context):
    client, headers, _ = persona_chat_client
    data, url = image_data()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Images"})
    first = persona_chat_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "First", "images": [{"data": data, "mime": "image/png"}]})
    second = persona_chat_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Second", "image_data": data, "image_mime_type": "image/png"})
    path = f"/api/v1/chats/{chat_id}/messages"
    default = client.get(path, headers=headers)
    assert default.status_code == 200
    assert all("images" not in row for row in default.json()["messages"])
    for offset, message_id in enumerate([first, second]):
        response = client.get(path, headers=headers, params={"include_images": True, "include_character_context": context, "limit": 1, "offset": offset})
        assert response.status_code == 200
        assert response.json()["messages"][0]["images"] == [url]
        assert response.json()["messages"][0]["id"] == message_id
        assert response.json()["total"] == 2
    assert client.get(path, headers=headers, params={"include_images": True, "scope_type": "workspace", "workspace_id": "other"}).status_code in (403, 404)
    foreign_id = persona_chat_db.add_conversation({"client_id": "2", "title": "Foreign"})
    assert client.get(f"/api/v1/chats/{foreign_id}/messages?include_images=true", headers=headers).status_code in (403, 404)
    completed = client.get(path, headers=headers, params={"include_images": True, "format_for_completions": True})
    assert completed.status_code == 200
    assert all("images" not in row for row in completed.json()["messages"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["Question", "", "<Image attachment x1>"])
def test_real_failed_image_turn_reuses_user_after_opt_in_listing(persona_chat_client, persona_chat_db, text, stream):
    client, headers, provider = persona_chat_client
    _, url = image_data()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Image retry"})
    body = image_body(chat_id, text, url)
    body["stream"] = stream
    provider.side_effect = HTTPException(502, "Provider failed")
    assert client.post("/api/v1/chat/completions", headers=headers, json=body).status_code == 502
    user = next(row for row in persona_chat_db.get_messages_for_conversation(chat_id) if row["sender"] == "user")
    listed = client.get(f"/api/v1/chats/{chat_id}/messages?include_images=true&include_metadata=true&render_placeholders=false", headers=headers)
    row = next(row for row in listed.json()["messages"] if row["id"] == user["id"])
    assert row["images"] == [url]
    if not text:
        assert row["metadata_extra"]["content_placeholder_reason"] == "image_attachment"
        assert row["version"] == 1
    provider.side_effect = None
    if stream:
        provider.return_value = iter([
            'data: {"choices":[{"delta":{"content":"Recovered"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ])
    body["metadata"]["tldw_retry_failed_turn"] = True
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200, response.text
    payloads = ([json.loads(line[6:]) for line in response.text.splitlines()
                 if line.startswith("data: ") and line[6:] != "[DONE]"] if stream else [response.json()])
    assert {p["tldw_user_message_id"] for p in payloads if p.get("tldw_user_message_id")} == {user["id"]}
    assert len([row for row in persona_chat_db.get_messages_for_conversation(chat_id) if row["sender"] == "user"]) == 1
    payload = provider.call_args.kwargs["messages_payload"]
    assert len([row for row in payload if row["role"] == "user"]) == 1
    assert "<Image attachment" not in str(payload)


def test_retry_with_prior_successful_image_turn_matches_entire_overlap(persona_chat_client, persona_chat_db):
    client, headers, provider = persona_chat_client
    _, url = image_data()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Image history"})
    first = image_body(chat_id, "", url, "first-local")
    assert client.post("/api/v1/chat/completions", headers=headers, json=first).status_code == 200
    second = image_body(chat_id, "Next question", url, "second-local")
    provider.side_effect = HTTPException(502, "Provider failed")
    assert client.post("/api/v1/chat/completions", headers=headers, json=second).status_code == 502
    rows = persona_chat_db.get_messages_for_conversation(chat_id)
    assistant = next(row["content"] for row in rows if row["sender"] == "assistant")
    provider.side_effect = None
    second["messages"] = [first["messages"][0], {"role": "assistant", "content": assistant}, second["messages"][0]]
    second["metadata"]["tldw_retry_failed_turn"] = True
    response = client.post("/api/v1/chat/completions", headers=headers, json=second)
    assert response.status_code == 200, response.text
    assert len([row for row in persona_chat_db.get_messages_for_conversation(chat_id) if row["sender"] == "user"]) == 2
    payload = provider.call_args.kwargs["messages_payload"]
    assert len([row for row in payload if row["role"] == "user"]) == 2
    assert "<Image attachment" not in str(payload)


@pytest.mark.parametrize("fault", ["ordered-read", "corrupt", "truncated", "position", "budget"])
def test_opt_in_read_never_returns_partial_attachments(persona_chat_client, persona_chat_db, monkeypatch, fault):
    client, headers, _ = persona_chat_client
    data, _ = image_data()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Incomplete"})
    message_id = persona_chat_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Question", "images": [{"data": data, "mime": "image/png"}, {"data": b"bad" if fault == "corrupt" else data[:12] if fault == "truncated" else data, "mime": "image/png"}]})
    if fault == "position":
        persona_chat_db.execute_query("UPDATE message_images SET position = 2 WHERE message_id = ? AND position = 1", (message_id,), commit=True)
    original = persona_chat_db.execute_query
    blob_reads = []
    def query(sql, *args, **kwargs):
        if "m.image_data, m.image_mime_type" in sql or "SELECT message_id, position, image_data" in sql:
            blob_reads.append(sql)
        if fault == "ordered-read" and "LEFT JOIN message_images" in sql:
            raise CharactersRAGDBError("Unavailable image table")
        return original(sql, *args, **kwargs)
    monkeypatch.setattr(persona_chat_db, "execute_query", query)
    if fault == "budget":
        monkeypatch.setattr(character_messages, "MAX_CHAT_ATTACHMENT_READ_BYTES", 1, raising=False)
    response = client.get(f"/api/v1/chats/{chat_id}/messages?include_images=true", headers=headers)
    assert response.status_code >= 400
    assert "messages" not in response.json()
    if fault == "budget":
        assert blob_reads == []


@pytest.mark.parametrize("mutation", ["text", "image", "order", "detail", "multiple-text", "edited-placeholder", "ranked-placeholder", "legacy-placeholder"])
def test_image_retry_conflicts_preserve_owned_rows(persona_chat_client, persona_chat_db, mutation):
    client, headers, provider = persona_chat_client
    _, url = image_data()
    _, other = image_data("blue")
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Image conflicts"})
    body = image_body(chat_id, "" if mutation.endswith("placeholder") else "Question", url)
    if mutation == "order":
        body["messages"][0]["content"].insert(1, {"type": "image_url", "image_url": {"url": other}})
    provider.side_effect = HTTPException(502, "Provider failed")
    assert client.post("/api/v1/chat/completions", headers=headers, json=body).status_code == 502
    user = next(row for row in persona_chat_db.get_messages_for_conversation(chat_id) if row["sender"] == "user")
    if mutation == "text":
        body["messages"][0]["content"][-1]["text"] = "Changed"
    elif mutation == "image":
        body["messages"][0]["content"][0]["image_url"]["url"] = other
    elif mutation == "order":
        body["messages"][0]["content"][:2] = list(reversed(body["messages"][0]["content"][:2]))
    elif mutation == "ranked-placeholder":
        persona_chat_db.update_message(user["id"], {"ranking": 1}, expected_version=user["version"])
    elif mutation == "legacy-placeholder":
        persona_chat_db.add_message_metadata(user["id"], extra={"client_message_id": "local-image"})
    elif mutation == "detail":
        body["messages"][0]["content"][0]["image_url"]["detail"] = "high"
    elif mutation == "multiple-text":
        body["messages"][0]["content"].append({"type": "text", "text": "More"})
    else:
        persona_chat_db.update_message(user["id"], {"content": "Edited"}, expected_version=user["version"])
        persona_chat_db.update_message(user["id"], {"content": user["content"]}, expected_version=user["version"] + 1)
    before = persona_chat_db.get_messages_for_conversation(chat_id)
    provider.reset_mock()
    body["metadata"]["tldw_retry_failed_turn"] = True
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 409
    provider.assert_not_called()
    assert persona_chat_db.get_messages_for_conversation(chat_id) == before


def test_image_retry_keeps_literal_template_text_in_prior_and_current_turns(persona_chat_client, persona_chat_db):
    client, headers, provider = persona_chat_client
    _, url = image_data()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Literal image questions"})
    first = image_body(chat_id, "Literal {{user}} and {{char}}", url, "first")
    assert client.post("/api/v1/chat/completions", headers=headers, json=first).status_code == 200
    assistant = next(row["content"] for row in persona_chat_db.get_messages_for_conversation(chat_id) if row["sender"] == "assistant")
    second = image_body(chat_id, "Again {{user}} and {{char}}", url, "second")
    provider.side_effect = HTTPException(502, "Provider failed")
    assert client.post("/api/v1/chat/completions", headers=headers, json=second).status_code == 502
    provider.side_effect = None
    second["messages"] = [first["messages"][0], {"role": "assistant", "content": assistant}, second["messages"][0]]
    second["metadata"]["tldw_retry_failed_turn"] = True
    response = client.post("/api/v1/chat/completions", headers=headers, json=second)
    assert response.status_code == 200, response.text
    users = [row for row in provider.call_args.kwargs["messages_payload"] if row["role"] == "user"]
    assert len(users) == 2
    assert [part["text"] for row in users for part in row["content"] if part["type"] == "text"] == ["Literal {{user}} and {{char}}", "Again {{user}} and {{char}}"]


def test_retry_cannot_skip_a_missing_second_saved_image(persona_chat_client, persona_chat_db):
    client, headers, provider = persona_chat_client
    data, url = image_data()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Incomplete retry"})
    message_id = persona_chat_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Question", "images": [{"data": data, "mime": "image/png"}, {"data": data, "mime": "image/png"}]})
    persona_chat_db.execute_query("UPDATE message_images SET image_data = ? WHERE message_id = ? AND position = 1", (b"", message_id), commit=True)
    body = image_body(chat_id, "Question", url)
    body["metadata"]["tldw_retry_failed_turn"] = True
    before = persona_chat_db.get_messages_for_conversation(chat_id)
    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 409
    provider.assert_not_called()
    assert persona_chat_db.get_messages_for_conversation(chat_id) == before


def assert_strict_image_page_snapshot(db, monkeypatch):
    """One SQL snapshot returns the exact page or only its over-budget size."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    data, _ = image_data()
    other, _ = image_data("blue")
    chat_id = db.add_conversation({"client_id": db.client_id, "title": "Bounded image snapshot"})
    first = db.add_message({"conversation_id": chat_id, "sender": "user", "content": "First", "images": [{"data": data, "mime": "image/png"}, {"data": other, "mime": "image/png"}]})
    second = db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Legacy", "image_data": data, "image_mime_type": "image/png"})
    db.execute_query("DELETE FROM message_images WHERE message_id = ?", (second,), commit=True)
    original = db.execute_query
    observed = []
    def query(sql, *args, **kwargs):
        cursor = original(sql, *args, **kwargs)
        if "WITH page AS" in sql:
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            observed.append([dict(row) if isinstance(row, dict) else dict(zip(columns, row)) for row in rows])
            return SimpleNamespace(description=cursor.description, fetchall=lambda: rows)
        assert "SELECT message_id, position, image_data" not in sql, "Strict reads must not refetch images outside their snapshot"
        return cursor
    monkeypatch.setattr(db, "execute_query", query)
    rows = db.get_messages_for_conversation(chat_id, limit=1, strict_images=True)
    assert len(observed) == 1
    assert [row["id"] for row in rows] == [first]
    assert [image["image_data"] for image in rows[0]["images"]] == [data, other]
    legacy = db.get_messages_for_conversation(chat_id, limit=1, offset=1, strict_images=True)
    assert legacy[0]["id"] == second
    assert legacy[0]["images"] == []
    assert legacy[0]["image_data"] == data
    with pytest.raises(InputError):
        db.get_messages_for_conversation(chat_id, strict_images=True, image_byte_limit=1)
    assert len(observed[-1]) == 2
    assert all(row["page_image_bytes"] == len(data) * 2 + len(other) for row in observed[-1])
    assert all(row["image_data"] is None and row["ordered_image_data"] is None for row in observed[-1])


def test_sqlite_strict_image_snapshot(persona_chat_db, monkeypatch):
    assert_strict_image_page_snapshot(persona_chat_db, monkeypatch)


def test_postgres_strict_image_snapshot(pg_database_config, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="1", backend=backend)
    try:
        assert_strict_image_page_snapshot(db, monkeypatch)
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_deleted_attachment_messages_and_conversations_stay_hidden(persona_chat_client, persona_chat_db):
    client, headers, _ = persona_chat_client
    data, _ = image_data()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Deleted image"})
    message_id = persona_chat_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Private image", "images": [{"data": data, "mime": "image/png"}]})
    persona_chat_db.execute_query("UPDATE messages SET deleted = TRUE WHERE id = ?", (message_id,), commit=True)
    response = client.get(f"/api/v1/chats/{chat_id}/messages?include_images=true", headers=headers)
    assert response.status_code == 200
    assert response.json()["messages"] == []
    persona_chat_db.execute_query("UPDATE conversations SET deleted = TRUE WHERE id = ?", (chat_id,), commit=True)
    assert client.get(f"/api/v1/chats/{chat_id}/messages?include_images=true", headers=headers).status_code in (403, 404)


@pytest.mark.parametrize("fault", ["empty-primary", "null-primary", "missing-mime"])
def test_opt_in_legacy_image_read_requires_complete_recognized_attachment(persona_chat_client, persona_chat_db, fault):
    client, headers, _ = persona_chat_client
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, format="TIFF" if fault == "missing-mime" else "PNG")
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "Incomplete legacy image"})
    message_id = persona_chat_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Image question", "image_data": stream.getvalue(), "image_mime_type": "image/tiff" if fault == "missing-mime" else "image/png"})
    persona_chat_db.execute_query("DELETE FROM message_images WHERE message_id = ?", (message_id,), commit=True)
    if fault in {"empty-primary", "null-primary"}:
        persona_chat_db.execute_query("UPDATE messages SET image_data = ? WHERE id = ?", (b"" if fault == "empty-primary" else None, message_id), commit=True)
    else:
        persona_chat_db.execute_query("UPDATE messages SET image_mime_type = NULL WHERE id = ?", (message_id,), commit=True)
    path = f"/api/v1/chats/{chat_id}/messages"
    assert client.get(path, headers=headers).status_code == 200
    response = client.get(path, headers=headers, params={"include_images": True})
    assert response.status_code == 409
    assert "messages" not in response.json()


@pytest.mark.parametrize("truncated", [False, True])
def test_opt_in_jpeg_requires_complete_pixel_data(persona_chat_client, persona_chat_db, truncated):
    client, headers, _ = persona_chat_client
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, format="JPEG")
    data = stream.getvalue()
    chat_id = persona_chat_db.add_conversation({"client_id": "1", "title": "JPEG image"})
    persona_chat_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "JPEG question", "images": [{"data": data[:-10] if truncated else data, "mime": "image/jpeg"}]})
    response = client.get(f"/api/v1/chats/{chat_id}/messages?include_images=true", headers=headers)
    assert response.status_code == (409 if truncated else 200)
    if not truncated:
        assert response.json()["messages"][0]["images"] == ["data:image/jpeg;base64," + base64.b64encode(data).decode()]
