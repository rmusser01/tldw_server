"""Raw message-list recovery preserves stored text and existing access checks."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints import character_messages
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def listing_state(request, tmp_path):
    """Create one owned neutral conversation using the real database interface."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(str(tmp_path / "messages.db"), client_id="41", backend=backend)
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
        if backend is not None:
            backend.get_pool().close_all()


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
async def test_recovery_opt_in_returns_unverified_without_public_metadata_authority(listing_state):
    """Legacy rows remain readable but cannot manufacture protected receipts."""
    db, chat_id, message_id, content = listing_state
    db.set_message_metadata_extra(message_id, {"history_result_v1": {"version": 1, "sources": []}})
    app = FastAPI()
    app.include_router(character_messages.router, prefix="/api/v1")
    app.dependency_overrides[character_messages.get_request_user] = lambda: SimpleNamespace(id=41)
    app.dependency_overrides[character_messages.get_chacha_db_for_user] = lambda: db
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        params = {"include_history_recovery_v1": "true", "scope_type": "global"}
        single = await client.get(f"/api/v1/messages/{message_id}", params=params)
        page = await client.get(f"/api/v1/chats/{chat_id}/messages", params={**params, "render_placeholders": "false"})
        assert single.status_code == page.status_code == 200
        for body in (single.json(), page.json()["messages"][0]):
            assert body["content"] == content
            assert body["tldw_history_recovery_v1"] == {
                "version": 1, "status": "unverified", "code": "no_protected_binding",
            }
            assert body.get("metadata_extra") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recovery_api_get_and_list_use_normal_operation_checkout(listing_state):
    """GET's preliminary message/image reads must leave isolation configurable."""
    db, chat_id, message_id, content = listing_state
    db.close_connection()
    app = FastAPI()
    app.add_middleware(ChaChaOperationMiddleware)
    app.include_router(character_messages.router, prefix="/api/v1")
    principal = SimpleNamespace(id=41)
    app.dependency_overrides[character_messages.get_request_user] = lambda: principal
    app.dependency_overrides[character_messages.get_chacha_db_for_user] = lambda: db
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        params = {"include_history_recovery_v1": "true", "scope_type": "global"}
        single_path = f"/api/v1/messages/{message_id}"
        page_path = f"/api/v1/chats/{chat_id}/messages"
        single = await client.get(single_path, params=params)
        page = await client.get(page_path, params={**params, "render_placeholders": "false"})
        assert single.status_code == page.status_code == 200
        assert single.json()["content"] == page.json()["messages"][0]["content"] == content
        assert single.json()["tldw_history_recovery_v1"]["status"] == "unverified"
        assert page.json()["total"] == 1
        assert (
            await client.get(single_path, params={**params, "scope_type": "workspace", "workspace_id": "other"})
        ).status_code == 404
        principal.id = 42
        assert (await client.get(single_path, params=params)).status_code == 403
        assert (await client.get(page_path, params={**params, "render_placeholders": "false"})).status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("incompatible", [
    {"render_placeholders": "true"}, {"include_deleted": "true"},
    {"format_for_completions": "true"}, {"include_character_context": "true"},
])
async def test_recovery_listing_rejects_transformed_or_deleted_views(listing_state, incompatible):
    db, chat_id, _, _ = listing_state
    app = FastAPI()
    app.include_router(character_messages.router, prefix="/api/v1")
    app.dependency_overrides[character_messages.get_request_user] = lambda: SimpleNamespace(id=41)
    app.dependency_overrides[character_messages.get_chacha_db_for_user] = lambda: db
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/chats/{chat_id}/messages", params={
            "include_history_recovery_v1": "true", "render_placeholders": "false", **incompatible,
        })
    assert response.status_code == 422
