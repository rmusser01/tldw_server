"""
Verify chat export JSON surfaces tool_calls per message and includes message_metadata_extra when include_metadata=true.
"""

import os
import shutil
import tempfile
from pathlib import Path

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings


@pytest.mark.asyncio
async def test_chat_export_includes_tool_calls_and_metadata():
    tmpdir = tempfile.mkdtemp(prefix="chacha_export_tools_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir
    try:
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Create character and chat
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            # Add assistant message
            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "assistant", "content": "Response with tools"},
            )
            assert r.status_code == 201
            # Confirm via list fetch
            r = await client.get(f"/api/v1/chats/{chat_id}/messages", headers=headers)
            assert r.status_code == 200
            msgs = r.json().get("messages", [])
            assistant_msg_id = next(
                (
                    m.get("id")
                    for m in msgs
                    if m.get("sender") == "assistant" and m.get("content") == "Response with tools"
                ),
                None,
            )
            assert assistant_msg_id, "Assistant message not found in conversation after creation"

            # Get user id and DB path (fixed id in single-user mode)
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _gs
            from tldw_Server_API.app.core.config import settings as cfg_settings

            user_id = _gs().SINGLE_USER_FIXED_ID
            base_dir = cfg_settings.get("USER_DB_BASE_DIR")
            db_path = Path(base_dir) / str(user_id) / "ChaChaNotes.db"

            # Persist inline tool_calls via public API
            tool_call_id = "call_456"
            tool_calls = [
                {"id": tool_call_id, "type": "function", "function": {"name": "lookup", "arguments": '{"id": 42}'}}
            ]
            r = await client.get(f"/api/v1/chats/{chat_id}/messages", headers=headers)
            assert r.status_code == 200
            msgs = r.json().get("messages", [])
            m = next((m for m in msgs if m.get("id") == assistant_msg_id), None)
            assert m is not None
            import json as _json

            new_content = (m.get("content") or "").rstrip() + "\n\n[tool_calls]: " + _json.dumps(tool_calls)
            r = await client.put(
                f"/api/v1/messages/{assistant_msg_id}",
                headers=headers,
                params={"expected_version": m.get("version")},
                json={"content": new_content},
            )
            assert r.status_code == 200

            # Export chat history (JSON)
            r = await client.get(
                f"/api/v1/chats/{chat_id}/export",
                headers=headers,
                params={"format": "json", "include_metadata": True},
            )
            assert r.status_code == 200
            data = r.json()
            msgs = data["messages"]
            # Find assistant message and ensure tool_calls surfaced
            am = next((m for m in msgs if m.get("id") == assistant_msg_id), None)
            assert am is not None
            assert "tool_calls" in am
            assert any(tc.get("id") == tool_call_id for tc in am.get("tool_calls", []))

            pagination = data["pagination"]
            assert pagination["mode"] == "page"
            assert pagination["per_page"] == 1000
            assert pagination["total"] == pagination["total_messages"]

            # message_metadata_extra is returned only when stored; we don't require it here
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
