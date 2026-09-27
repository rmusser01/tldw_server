"""Diagnose first-byte delivery through real Character and local-adapter streams."""

import asyncio
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import CharacterChatCompletionV2Request
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import _chat_with_openai_compatible_local_server

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def stream_db(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "stream.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("unified", ["0", "1"])
@pytest.mark.parametrize("scenario", ["complete", "disconnect", "fallback"])
async def test_local_bytes_forward_before_completion_and_cleanup(stream_db, monkeypatch, unified, scenario):
    """Control only upstream I/O, credential lookup and limits; keep DB/stream code real."""
    db = stream_db
    character = db.add_character_card(
        {"name": "Delivery TestBot", "system_prompt": "Always respond with exactly: BEEP BOOP."}
    )
    chat = db.add_conversation({"character_id": character, "client_id": "1", "title": "Delivery"})
    message = db.add_message({"conversation_id": chat, "sender": "user", "content": "Hello, who are you?"})
    entered, first, finish = threading.Event(), threading.Event(), threading.Event()
    lifecycle, requests = [], []
    runtime_closed = asyncio.Event()

    class Runtime:
        def __init__(self, **_kwargs):
            pass

        async def resolve(self, provider, *, model=None):
            return SimpleNamespace(provider=provider, api_key=None, app_config={}, credentials_resolved=True)

        async def mark_used(self, _credentials):
            lifecycle.append("used")

        async def close(self):
            lifecycle.append("runtime_closed")
            runtime_closed.set()

    async def allowed(*_args, **_kwargs):
        pass

    class Upstream:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            entered.set()
            assert first.wait(5)
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
            assert finish.wait(5)
            yield 'data: {"choices":[{"delta":{"content":"BEEP BOOP"}}]}'
            yield "data: [DONE]"

        def close(self):
            lifecycle.append("response_closed")

    @contextmanager
    def stream_http(**kwargs):
        requests.append(kwargs["json"])
        response = Upstream()
        try:
            yield response
        finally:
            response.close()

    def provider_call(**kwargs):
        if scenario == "fallback":
            return {"choices": [{"message": {"content": "BEEP BOOP"}}]}
        return _chat_with_openai_compatible_local_server(
            api_base_url="http://provider.invalid",
            model_name=kwargs["model"],
            input_data=kwargs["messages_payload"],
            streaming=True,
            http_client_factory=lambda *_: SimpleNamespace(close=lambda: lifecycle.append("client_closed")),
            http_streamer=stream_http,
        )

    monkeypatch.setenv("STREAMS_UNIFIED", unified)
    monkeypatch.setenv("DISABLE_OFFLINE_SIM", "true")
    monkeypatch.setattr(endpoint, "derive_trusted_credential_scope", lambda *_: (1, [], [], False))
    monkeypatch.setattr(endpoint, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(
        endpoint, "get_character_rate_limiter", lambda: SimpleNamespace(check_chat_completion_rate=allowed)
    )
    monkeypatch.setattr(endpoint, "_should_enforce_char_chat_strict_model_selection", lambda: False)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider_call)
    response = await endpoint.character_chat_completion(
        chat_id=chat,
        body=CharacterChatCompletionV2Request(provider="llama", model="delivery-model", stream=True, save_to_db=True),
        db=db,
        current_user=User(id=1, username="fixture", is_active=True),
        http_request=SimpleNamespace(state=SimpleNamespace()),
    )
    pending = None if scenario == "fallback" else asyncio.create_task(response.body_iterator.__anext__())
    try:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert set(response.headers.get("cache-control", "").replace(" ", "").split(",")) >= {"no-cache", "no-transform"}
        assert response.headers.get("x-accel-buffering") == "no"
        if scenario == "fallback":
            chunks = [str(chunk) async for chunk in response.body_iterator]
            assert any("BEEP BOOP" in chunk for chunk in chunks)
            assert sum("[DONE]" in chunk for chunk in chunks) == 1
            await asyncio.wait_for(runtime_closed.wait(), 3)
            assert [row["id"] for row in db.get_messages_for_conversation(chat)] == [message]
            assert requests == []
            return
        assert await asyncio.to_thread(entered.wait, 3)
        # Successful headers do not imply body delivery while the provider is silent.
        assert response.status_code == 200 and not pending.done()
        first.set()
        initial = await asyncio.wait_for(pending, 3)
        assert '"role"' in str(initial) and not finish.is_set()
        assert len(requests) == 1
        assert "BEEP BOOP" in json.dumps(requests[0]["messages"])
        if scenario == "disconnect":
            pending = asyncio.create_task(response.body_iterator.__anext__())
            await asyncio.sleep(0)
            pending.cancel()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            finish.set()
            chunks = [str(chunk) async for chunk in response.body_iterator]
            assert any("BEEP BOOP" in chunk for chunk in chunks)
            assert sum("[DONE]" in chunk for chunk in chunks) == 1
        await asyncio.wait_for(runtime_closed.wait(), 3)
        assert "response_closed" in lifecycle and "client_closed" in lifecycle
        assert lifecycle.index("client_closed") < lifecycle.index("runtime_closed")
        assert [row["id"] for row in db.get_messages_for_conversation(chat)] == [message]
        assert len(requests) == 1
    finally:
        first.set()
        finish.set()
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await response.body_iterator.aclose()
        if response.background:
            await response.background()
