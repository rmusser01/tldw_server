"""Raw message-list recovery preserves stored text and existing access checks."""

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints import character_messages
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def listing_state(tmp_path):
    """Create one owned neutral conversation using the real database interface."""
    db = CharactersRAGDB(str(tmp_path / "messages.db"), client_id="41")
    chat_id = db.add_conversation({"client_id": "41", "title": "Literal templates"})
    content = "Explain {{user}} and <CHAR>; keep {{char}} literally."
    message_id = db.add_message(
        {
            "conversation_id": chat_id,
            "sender": "user",
            "content": content,
        }
    )
    try:
        yield db, chat_id, message_id, content
    finally:
        db.close_all_connections()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("include_context", [False, True])
@pytest.mark.parametrize("render", [False, True])
async def test_standard_listing_can_preserve_raw_message_content(listing_state, include_context, render):
    """Only the requested standard response content changes, never stored rows."""
    db, chat_id, message_id, content = listing_state
    result = await character_messages.get_chat_messages(
        chat_id=chat_id,
        limit=200,
        offset=0,
        include_deleted=False,
        include_character_context=include_context,
        format_for_completions=False,
        include_tool_calls=False,
        include_metadata=False,
        include_message_ids=False,
        render_placeholders=render,
        scope_type="global",
        workspace_id=None,
        db=db,
        current_user=SimpleNamespace(id=41),
    )
    body = result if isinstance(result, dict) else result.model_dump()
    expected = "Explain User and Assistant; keep Assistant literally." if render else content
    assert body["messages"][0]["content"] == expected
    assert body["messages"][0]["id"] == message_id
    assert body["messages"][0]["sender"] == "user"
    assert db.get_messages_for_conversation(chat_id)[0]["content"] == content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_raw_listing_query_keeps_default_rendering_pagination_and_access(listing_state):
    """Exercise actual query parsing, owner guard and database-backed responses."""
    db, chat_id, message_id, content = listing_state
    app = FastAPI()
    app.include_router(character_messages.router, prefix="/api/v1")
    principal = SimpleNamespace(id=41)
    app.dependency_overrides[character_messages.get_request_user] = lambda: principal
    app.dependency_overrides[character_messages.get_chacha_db_for_user] = lambda: db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        path = f"/api/v1/chats/{chat_id}/messages"
        rendered = await client.get(path)
        raw = await client.get(path, params={"render_placeholders": "false", "limit": 1})
        assert rendered.status_code == raw.status_code == 200
        assert rendered.json()["messages"][0]["content"] == "Explain User and Assistant; keep Assistant literally."
        assert raw.json()["messages"][0]["content"] == content
        assert raw.json()["messages"][0]["id"] == message_id
        assert raw.json()["messages"][0]["sender"] == "user"
        assert raw.json()["total"] == 1
        assert raw.json()["offset"] == 0
        assert raw.json()["limit"] == 1
        assert (await client.get(path, params={"render_placeholders": "invalid"})).status_code == 422
        principal.id = 42
        assert (await client.get(path, params={"render_placeholders": "false"})).status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_raw_option_does_not_change_completion_format(listing_state):
    """Completion formatting continues to expand placeholders for inference."""
    db, chat_id, _, _ = listing_state
    app = FastAPI()
    app.include_router(character_messages.router, prefix="/api/v1")
    app.dependency_overrides[character_messages.get_request_user] = lambda: SimpleNamespace(id=41)
    app.dependency_overrides[character_messages.get_chacha_db_for_user] = lambda: db
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/chats/{chat_id}/messages",
            params={
                "format_for_completions": "true",
                "render_placeholders": "false",
            },
        )
    assert response.status_code == 200
    assert response.json()["messages"][0] == {
        "role": "user",
        "content": "Explain User and Assistant; keep Assistant literally.",
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_listing_runs_off_event_loop_with_operation_context(
    listing_state: tuple[CharactersRAGDB, str, str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attachment decoding must not block the loop or lose request-owned DB cleanup."""
    import threading

    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import (
        chacha_operation,
        current_connection_state,
    )

    db, chat_id, _, _ = listing_state
    original = character_messages.read_messages_with_images
    observed = []

    def read(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Capture the execution owner while performing the real complete read."""
        observed.append((threading.get_ident(), current_connection_state(db)))
        return original(*args, **kwargs)

    monkeypatch.setattr(character_messages, "read_messages_with_images", read)
    app = FastAPI()
    app.include_router(character_messages.router, prefix="/api/v1")
    app.dependency_overrides[character_messages.get_request_user] = lambda: SimpleNamespace(id=41)
    app.dependency_overrides[character_messages.get_chacha_db_for_user] = lambda: db
    loop_thread = threading.get_ident()
    with chacha_operation() as operation:
        expected_state = operation.state_for(db)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/v1/chats/{chat_id}/messages", params={"include_images": True})
    assert response.status_code == 200
    assert observed[0][0] != loop_thread
    assert observed[0][1] is expected_state


@pytest.mark.unit
@pytest.mark.parametrize("for_completions", [False, True])
def test_image_helper_retains_ordered_bytes_and_proven_placeholder_policy(for_completions: bool) -> None:
    """Direct reads preserve both attachments and remove only a proven placeholder."""
    import base64
    import io
    from unittest.mock import Mock

    from PIL import Image

    from tldw_Server_API.app.api.v1.utils.chat_message_images import read_messages_with_images

    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    png = output.getvalue()
    row = {"id": "image-turn", "sender": "user", "version": 1, "content": "<Image attachment x2>",
           "images": [{"image_data": memoryview(png), "image_mime_type": "image/png"},
                      {"image_data": png, "image_mime_type": "image/png"}]}
    db = Mock(spec=CharactersRAGDB)
    db.get_messages_for_conversation.return_value = [row]
    db.get_message_metadata.return_value = {"extra": {
        "image_details": ["high", "low"], "content_placeholder_reason": "image_attachment",
    }}
    messages, urls = read_messages_with_images(db, "owned", limit=2, for_completions=for_completions)
    assert urls == {"image-turn": ["data:image/png;base64," + base64.b64encode(png).decode("ascii")] * 2}
    assert messages[0]["content"] == ("" if for_completions else "<Image attachment x2>")
    assert messages[0]["image_details"] == ["high", "low"]


@pytest.mark.unit
@pytest.mark.parametrize("failure,expected_status", [("row_read", 503), ("budget", 413), ("corrupt", 409)])
def test_image_helper_never_returns_a_partial_page(failure: str, expected_status: int) -> None:
    """Unreadable, oversized or corrupt attachments fail the complete requested page."""
    from unittest.mock import Mock

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.utils.chat_message_images import read_messages_with_images
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError, InputError

    db = Mock(spec=CharactersRAGDB)
    if failure == "corrupt":
        db.get_messages_for_conversation.return_value = [{"id": "broken", "image_data": b"broken",
                                                          "image_mime_type": "image/png"}]
    else:
        db.get_messages_for_conversation.side_effect = (
            InputError("too large") if failure == "budget" else CharactersRAGDBError("incomplete")
        )
    with pytest.raises(HTTPException) as error:
        read_messages_with_images(db, "owned", limit=2)
    assert error.value.status_code == expected_status


@pytest.mark.unit
@pytest.mark.parametrize("details", [None, ["high", "low"], ["auto"], ["high", "invalid"]])
def test_image_content_formatter_keeps_order_and_rejects_invalid_options(details: list[str] | None) -> None:
    """Formatting preserves text-only output and validates ordered image option lists."""
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.utils.chat_message_images import format_message_content

    assert format_message_content("literal text", [], details) == "literal text"
    urls = ["data:image/png;base64,first", "data:image/jpeg;base64,second"]
    if details in (["auto"], ["high", "invalid"]):
        with pytest.raises(HTTPException) as error:
            format_message_content("", urls, details)
        assert error.value.status_code == 409
    else:
        parts = format_message_content("", urls, details)
        assert parts == [{"type": "image_url", "image_url": {"url": url, **({"detail": details[i]} if details else {})}}
                         for i, url in enumerate(urls)]
    assert urls == ["data:image/png;base64,first", "data:image/jpeg;base64,second"]
