import io

import pytest
from PIL import Image

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import MessageResponse
from tldw_Server_API.tests.Chat.integration import test_persona_backed_chat_conversations as fixtures

persona_chat_db = fixtures.persona_chat_db
persona_chat_client = fixtures.persona_chat_client


@pytest.mark.parametrize("fault", ["empty-primary", "missing-mime", "truncated-jpeg"])
def test_read_probe_invalid_legacy_attachment_is_not_complete(persona_chat_client, persona_chat_db, fault):
    client, headers, _ = persona_chat_client
    stream = io.BytesIO()
    format = {"missing-mime": "TIFF", "truncated-jpeg": "JPEG"}.get(fault, "PNG")
    Image.new("RGB", (2, 2), "red").save(stream, format=format)
    chat = persona_chat_db.add_conversation({"client_id": "1", "title": "Legacy attachment"})
    message = persona_chat_db.add_message({"conversation_id": chat, "sender": "user", "content": "Image question", "image_data": stream.getvalue(), "image_mime_type": f"image/{format.lower()}"})
    persona_chat_db.execute_query("DELETE FROM message_images WHERE message_id = ?", (message,), commit=True)
    if fault == "empty-primary":
        persona_chat_db.execute_query("UPDATE messages SET image_data = ? WHERE id = ?", (b"", message), commit=True)
    elif fault == "missing-mime":
        persona_chat_db.execute_query("UPDATE messages SET image_mime_type = NULL WHERE id = ?", (message,), commit=True)
    else:
        persona_chat_db.execute_query("UPDATE messages SET image_data = ? WHERE id = ?", (stream.getvalue()[:-10], message), commit=True)
    result = client.get(f"/api/v1/chats/{chat}/messages?include_images=true", headers=headers)
    assert result.status_code >= 400, {"status": result.status_code, "rows": [{"images": [url.split(",", 1)[0] for url in row.get("images", [])], "has_image": row["has_image"]} for row in result.json()["messages"]]}


def test_read_probe_serializer_preserves_explicit_images_only():
    message = MessageResponse(id="m", conversation_id="c", sender="user", content="Text", timestamp="2026-09-16T00:00:00Z")
    assert "images" not in message.model_dump()
    assert "images" not in message.model_dump(mode="json")
    assert message.model_copy(update={"images": []}).model_dump()["images"] == []
    assert message.model_copy(update={"images": ["data:image/png;base64,eA=="]}).model_dump()["images"] == ["data:image/png;base64,eA=="]
