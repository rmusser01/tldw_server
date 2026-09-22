"""Character completion keeps stored images through the real API/provider boundary."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgres"])
def character_db(request: pytest.FixtureRequest, test_db_path: Path) -> Iterator[CharactersRAGDB]:
    """Exercise the real endpoint with SQLite and the official Postgres fixture."""
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(db_path=str(test_db_path), client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def create_character_chat(test_client: TestClient, auth_headers: dict[str, str]) -> str:
    """Use the public API to create an owned Character and its conversation."""
    character = test_client.post("/api/v1/characters/", json={"name": "Vision Character"}, headers=auth_headers)
    assert character.status_code == 201, character.text
    chat = test_client.post("/api/v1/chats/", json={"character_id": character.json()["id"]}, headers=auth_headers)
    assert chat.status_code == 201, chat.text
    return chat.json()["id"]


@pytest.mark.parametrize("endpoint", ["context", "messages", "completions", "complete-v2"])
@pytest.mark.parametrize("text", ["", "Describe this image", None])
def test_character_completion_retains_saved_image(monkeypatch: pytest.MonkeyPatch, test_client: TestClient, auth_headers: dict[str, str], endpoint: str, text: str | None) -> None:
    """Stored PNG content reaches completion formatting without dropping text-only turns."""
    png = (Path(__file__).resolve().parents[4] / "apps/packages/ui/src/public/icon/128.png").read_bytes()
    chat_id = create_character_chat(test_client, auth_headers)
    payload = {"role": "user"}
    if text:
        payload["content"] = text
    if text is None:
        payload["content"] = "Text-only control"
    else:
        payload["image_base64"] = base64.b64encode(png).decode("ascii")
    saved = test_client.post(f"/api/v1/chats/{chat_id}/messages", json=payload, headers=auth_headers)
    assert saved.status_code == 201, saved.text
    captured = []

    def capture_completion(**kwargs: Any) -> dict[str, Any]:
        captured.extend(kwargs["messages_payload"])
        return {"choices": [{"message": {"content": "Image received"}}]}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", capture_completion)
    monkeypatch.setattr(character_chat_sessions, "is_model_known_for_provider", lambda *args: None)
    body = {"include_character_context": False}
    if endpoint == "complete-v2":
        body.update(provider="llama", model="vision-fixture", save_to_db=False)
    if endpoint in {"context", "messages"}:
        response = test_client.get(
            f"/api/v1/chats/{chat_id}/{endpoint}", params={"format_for_completions": True}, headers=auth_headers
        )
    else:
        response = test_client.post(f"/api/v1/chats/{chat_id}/{endpoint}", json=body, headers=auth_headers)
    assert response.status_code == 200, response.text
    messages = response.json()["messages"] if endpoint != "complete-v2" else captured
    content = next(message["content"] for message in messages if message["role"] == "user")
    if text is None:
        assert content == "Text-only control"
    else:
        assert isinstance(content, list), "The stored image must reach the provider as multimodal content"
        images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
        assert images == ["data:image/png;base64," + base64.b64encode(png).decode("ascii")]
        assert "".join(part["text"] for part in content if part["type"] == "text") == text


@pytest.mark.parametrize("stream", [False, True])
def test_character_provider_receives_ordered_png_and_jpeg(monkeypatch: pytest.MonkeyPatch, test_client: TestClient, auth_headers: dict[str, str], character_db: CharactersRAGDB, stream: bool) -> None:
    """Streaming and normal dispatch carry every attachment with exact MIME and bytes."""
    images = []
    for fmt, mime in [("PNG", "image/png"), ("JPEG", "image/jpeg")]:
        output = io.BytesIO()
        Image.new("RGB", (2, 2), "red").save(output, format=fmt)
        images.append({"data": output.getvalue(), "mime": mime})
    chat_id = create_character_chat(test_client, auth_headers)
    character_db.add_message(
        {"conversation_id": chat_id, "sender": "user", "content": "Hello {{user}}", "images": images}
    )
    captured = []

    def provider(**kwargs: Any) -> dict[str, Any] | Iterator[str]:
        captured.extend(kwargs["messages_payload"])
        if stream:
            return iter(
                ['data: {"choices":[{"delta":{"content":"Seen"},"finish_reason":null}]}\n\n', "data: [DONE]\n\n"]
            )
        return {"choices": [{"message": {"content": "Seen"}}]}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", provider)
    monkeypatch.setattr(character_chat_sessions, "is_model_known_for_provider", lambda *args: None)
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={
            "provider": "llama",
            "model": "vision-fixture",
            "include_character_context": False,
            "save_to_db": False,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    assert "Seen" in response.text
    content = next(row["content"] for row in captured if row["role"] == "user")
    assert content == [{"type": "text", "text": "Hello User"}] + [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image['mime']};base64,{base64.b64encode(image['data']).decode('ascii')}"},
        }
        for image in images
    ]


@pytest.mark.parametrize("fault", ["corrupt", "budget", "unavailable", "foreign"])
def test_character_rejects_incomplete_images_before_provider(
    monkeypatch: pytest.MonkeyPatch, test_client: TestClient, auth_headers: dict[str, str], character_db: CharactersRAGDB, fault: str
) -> None:
    """Completion never silently drops an unreadable attachment or crosses ownership."""
    from tldw_Server_API.app.api.v1.utils import chat_message_images
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    png = (Path(__file__).resolve().parents[4] / "apps/packages/ui/src/public/icon/128.png").read_bytes()
    chat_id = character_db.add_conversation({"client_id": "2" if fault == "foreign" else "1", "title": "Private image"})
    character_db.add_message(
        {
            "conversation_id": chat_id,
            "sender": "user",
            "content": "Describe",
            "images": [{"data": b"corrupt" if fault == "corrupt" else png, "mime": "image/png"}],
        }
    )
    called = []
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", lambda **kwargs: called.append(kwargs))
    if fault == "budget":
        monkeypatch.setitem(chat_message_images.settings, "MAX_MESSAGE_IMAGE_BYTES", 1)
    if fault == "unavailable":
        original = character_db.get_messages_for_conversation

        def read(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("strict_images"):
                raise CharactersRAGDBError("Unavailable attachment snapshot")
            return original(*args, **kwargs)

        monkeypatch.setattr(character_db, "get_messages_for_conversation", read)
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={"provider": "llama", "model": "vision-fixture", "save_to_db": False},
    )
    expected = {"corrupt": (409,), "budget": (413,), "unavailable": (503,), "foreign": (403, 404)}
    assert response.status_code in expected[fault], response.text
    assert called == []


def test_character_offline_image_only_does_not_echo_an_earlier_turn(test_client: TestClient, auth_headers: dict[str, str], character_db: CharactersRAGDB) -> None:
    """Offline simulation can read multipart input without treating old text as current."""
    png = (Path(__file__).resolve().parents[4] / "apps/packages/ui/src/public/icon/128.png").read_bytes()
    chat_id = create_character_chat(test_client, auth_headers)
    character_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Earlier turn"})
    character_db.add_message(
        {"conversation_id": chat_id, "sender": "user", "content": "", "images": [{"data": png, "mime": "image/png"}]}
    )
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={"provider": "local-llm", "model": "local-test", "include_character_context": False, "save_to_db": False},
    )
    assert response.status_code == 200, response.text
    assert response.json()["assistant_content"] == "OK"


@pytest.mark.parametrize("endpoint", ["context", "messages", "completions", "complete-v2"])
@pytest.mark.parametrize("details", [None, ["high", "low"], ["high"], ["high", "invalid"]])
def test_character_preserves_or_rejects_saved_image_details(
    monkeypatch: pytest.MonkeyPatch, test_client: TestClient, auth_headers: dict[str, str], character_db: CharactersRAGDB, endpoint: str, details: list[str] | None
) -> None:
    """Every formatter preserves the ordered options ordinary Chat uses for retry."""
    png = (Path(__file__).resolve().parents[4] / "apps/packages/ui/src/public/icon/128.png").read_bytes()
    chat_id = create_character_chat(test_client, auth_headers)
    message_id = character_db.add_message({
        "conversation_id": chat_id, "sender": "user", "content": "Describe",
        "images": [{"data": png, "mime": "image/png"}] * 2,
    })
    if details is not None:
        assert character_db.add_message_metadata(message_id, extra={"image_details": details})
    captured = []

    def provider(**kwargs: Any) -> dict[str, Any] | Iterator[str]:
        captured.extend(kwargs["messages_payload"])
        return {"choices": [{"message": {"content": "Seen"}}]}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", provider)
    monkeypatch.setattr(character_chat_sessions, "is_model_known_for_provider", lambda *args: None)
    body = {"include_character_context": False}
    if endpoint == "complete-v2":
        body.update(provider="llama", model="vision-fixture", save_to_db=False)
    if endpoint in {"context", "messages"}:
        response = test_client.get(f"/api/v1/chats/{chat_id}/{endpoint}",
                                   params={"format_for_completions": True}, headers=auth_headers)
    else:
        response = test_client.post(f"/api/v1/chats/{chat_id}/{endpoint}", json=body, headers=auth_headers)
    if details in (["high"], ["high", "invalid"]):
        assert response.status_code == 409, response.text
        assert captured == []
        return
    assert response.status_code == 200, response.text
    messages = captured if endpoint == "complete-v2" else response.json()["messages"]
    images = [part["image_url"] for row in messages if row["role"] == "user"
              for part in row["content"] if part["type"] == "image_url"]
    assert [part.get("detail", "auto") for part in images] == (details or ["auto", "auto"])
    assert [part["url"] for part in images] == ["data:image/png;base64," + base64.b64encode(png).decode("ascii")] * 2


def test_character_image_metadata_read_failure_prevents_dispatch(
    monkeypatch: pytest.MonkeyPatch, test_client: TestClient, auth_headers: dict[str, str], character_db: CharactersRAGDB
) -> None:
    """A real metadata read failure must not be mistaken for a legacy auto row."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    png = (Path(__file__).resolve().parents[4] / "apps/packages/ui/src/public/icon/128.png").read_bytes()
    chat_id = create_character_chat(test_client, auth_headers)
    mid = character_db.add_message({"conversation_id": chat_id, "sender": "user", "content": "Describe",
                                   "images": [{"data": png, "mime": "image/png"}]})
    assert character_db.add_message_metadata(mid, extra={"image_details": ["high"]})
    execute = character_db.execute_query

    def fail_metadata(query: str, *args: Any, **kwargs: Any) -> Any:
        if "FROM message_metadata" in query:
            raise CharactersRAGDBError("Test metadata read unavailable")
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(character_db, "execute_query", fail_metadata)
    called = []
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", lambda **kwargs: called.append(kwargs))
    response = test_client.post(f"/api/v1/chats/{chat_id}/complete-v2", headers=auth_headers,
                               json={"provider": "llama", "model": "vision-fixture", "save_to_db": False})
    assert response.status_code == 503, response.text
    assert called == []


@pytest.mark.parametrize("endpoint", ["context", "messages", "completions", "complete-v2"])
@pytest.mark.parametrize("text_kind", ["generated", "literal", "edited"])
def test_character_image_only_text_preserves_retry_content(
    monkeypatch: pytest.MonkeyPatch, test_client: TestClient, auth_headers: dict[str, str], character_db: CharactersRAGDB, endpoint: str, text_kind: str
) -> None:
    """Discard only the server's unedited placeholder, never user-authored text."""
    png = (Path(__file__).resolve().parents[4] / "apps/packages/ui/src/public/icon/128.png").read_bytes()
    chat_id = create_character_chat(test_client, auth_headers)
    placeholder = "<Image attachment x1>"
    mid = character_db.add_message({"conversation_id": chat_id, "sender": "user", "content": placeholder,
                                   "images": [{"data": png, "mime": "image/png"}]})
    extra = {"image_details": ["high"]}
    if text_kind != "literal":
        extra["content_placeholder_reason"] = "image_attachment"
    assert character_db.add_message_metadata(mid, extra=extra)
    if text_kind == "edited":
        assert character_db.update_message(mid, {"content": placeholder}, expected_version=1)
    raw = test_client.get(f"/api/v1/chats/{chat_id}/messages", params={"include_images": True}, headers=auth_headers)
    assert raw.status_code == 200, raw.text
    assert raw.json()["messages"][0]["content"] == placeholder
    captured = []

    def provider(**kwargs: Any) -> dict[str, Any] | Iterator[str]:
        captured.extend(kwargs["messages_payload"])
        return {"choices": [{"message": {"content": "Seen"}}]}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", provider)
    monkeypatch.setattr(character_chat_sessions, "is_model_known_for_provider", lambda *args: None)
    body = {"include_character_context": False}
    if endpoint == "complete-v2":
        body.update(provider="llama", model="vision-fixture", save_to_db=False)
    if endpoint in {"context", "messages"}:
        response = test_client.get(f"/api/v1/chats/{chat_id}/{endpoint}",
                                   params={"format_for_completions": True}, headers=auth_headers)
    else:
        response = test_client.post(f"/api/v1/chats/{chat_id}/{endpoint}", json=body, headers=auth_headers)
    assert response.status_code == 200, response.text
    messages = captured if endpoint == "complete-v2" else response.json()["messages"]
    parts = next(row["content"] for row in messages if row["role"] == "user")
    assert "".join(part["text"] for part in parts if part["type"] == "text") == ("" if text_kind == "generated" else placeholder)
    assert [part["image_url"]["detail"] for part in parts if part["type"] == "image_url"] == ["high"]
