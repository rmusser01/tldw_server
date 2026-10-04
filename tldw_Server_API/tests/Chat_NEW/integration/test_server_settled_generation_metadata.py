"""Server-settled replies keep generation metadata and partial output (D7 P2, S3-1).

The disconnect tests drive the real ASGI app and drop the client mid-stream the
way uvicorn does: ``receive()`` returns ``http.disconnect`` while the provider
is still generating. Starlette then cancels the response task through anyio,
whose cancellation is level-triggered, so the partial reply must be written by
a detached task rather than by an ordinary ``await`` in a ``finally``.

Partial replies are kept only for native turns (``tldw_history_selection_v1``).
Other ``save_to_db`` turns keep the unanswered-turn retry contract.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio
import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

_SETTLE_TIMEOUT_SECONDS = 10.0


@pytest.fixture
def settled_api(credentialed_test_client, populated_chacha_db, auth_headers):
    """Authenticated owner, a fresh conversation, and the shared database handle."""
    client = credentialed_test_client
    db = CharactersRAGDB(
        db_path=populated_chacha_db.db_path_str,
        client_id="1",
        owner_user_id="1",
    )
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    cid = db.add_conversation({"character_id": None, "title": "P2 settled", "client_id": "1"})
    yield client, db, cid, auth_headers
    client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
    db.close_connection()


def _selection(client, cid: str, headers: dict[str, str], seed: str) -> dict[str, Any]:
    response = client.post(
        f"/api/v1/chat/conversations/{cid}/history/selection",
        headers=headers,
        json={
            "purpose": "send",
            "view": {
                "view_session_id": f"view-{seed}",
                "conversation_id": cid,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "empty"},
                "selection_revision": 1,
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    return resolve_history_selection(body["snapshot"], body["view"], "send", seed)["selection"]


def _completion_body(cid: str, *, versioned_selection: dict[str, Any] | None, stream: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "conversation_id": cid,
        "save_to_db": True,
        "stream": stream,
        "messages": [{"role": "user", "content": "tell me a story"}],
    }
    if versioned_selection is not None:
        body["tldw_history_selection_v1"] = versioned_selection
    return body


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


def _content(text: str, finish_reason: str | None = None) -> str:
    choice: dict[str, Any] = {"index": 0, "delta": {"content": text}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return _sse({"choices": [choice]})


def _install_provider(monkeypatch, make_stream: Callable[[], AsyncIterator[str]]) -> None:
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

    def provider(*_args, **_kwargs):
        return make_stream()

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", provider)


def _drop_client_after(app, marker: bytes, delivered: list[bytes], occurrences: int = 1):
    """Wrap an ASGI app so the client disconnects once ``marker`` was delivered.

    This mirrors uvicorn (ASGI spec 2.3): after the drop ``receive()`` returns
    ``http.disconnect`` and later ``send()`` calls are silently discarded.
    Bodies the client received before the drop are appended to ``delivered``.
    """

    async def wrapped(scope, receive, send):
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return
        dropped = anyio.Event()
        request_delivered = False
        seen = 0

        async def receive_until_dropped():
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return await receive()
            await dropped.wait()
            return {"type": "http.disconnect"}

        async def send_until_dropped(message):
            nonlocal seen
            if dropped.is_set():
                return
            await send(message)
            if message.get("type") == "http.response.body":
                delivered.append(message.get("body") or b"")
                seen += (message.get("body") or b"").count(marker)
                if seen >= occurrences:
                    dropped.set()

        await app(scope, receive_until_dropped, send_until_dropped)

    return wrapped


def _assistant_rows(db: CharactersRAGDB, cid: str) -> list[dict[str, Any]]:
    return [row for row in db.get_messages_for_conversation(cid) if row["sender"] == "assistant"]


def _wait_for_assistant(db: CharactersRAGDB, cid: str) -> dict[str, Any]:
    deadline = time.monotonic() + _SETTLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        rows = _assistant_rows(db, cid)
        if rows:
            return rows[-1]
        time.sleep(0.05)
    raise AssertionError("No assistant reply was settled for the conversation")


def _extra(db: CharactersRAGDB, message_id: str) -> dict[str, Any]:
    metadata = db.get_message_metadata(message_id) or {}
    return metadata.get("extra") or {}


def test_client_disconnect_mid_stream_saves_partial_as_interrupted(settled_api, monkeypatch):
    client, db, cid, headers = settled_api
    # No moderation holdback, so the client has seen exactly what is saved.
    monkeypatch.setenv("MODERATION_STREAM_BUFFER_CHARS", "0")

    async def hanging_provider() -> AsyncIterator[str]:
        yield _content("Once upon ")
        yield _content("a time")
        # The provider is still generating when the client goes away.
        await asyncio.Event().wait()

    _install_provider(monkeypatch, hanging_provider)
    selection = _selection(client, cid, headers, "disconnect")
    delivered: list[bytes] = []
    monkeypatch.setattr(client._transport, "app", _drop_client_after(client.app, b"a time", delivered))

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )
    assert response.status_code == 200
    assert b"a time" in b"".join(delivered)

    reply = _wait_for_assistant(db, cid)
    assert reply["content"] == "Once upon a time"
    extra = _extra(db, reply["id"])
    assert extra["generation_status"] == "interrupted"
    assert extra["model_id"] == "gpt-4o-mini"
    assert extra["provider"] == "openai"
    assert "finish_reason" not in extra



def test_client_disconnect_saves_moderation_holdback_tail(settled_api, monkeypatch):
    """Text held back for cross-chunk moderation is part of the saved partial."""
    client, db, cid, headers = settled_api
    monkeypatch.setenv("MODERATION_STREAM_BUFFER_CHARS", "8")

    async def hanging_provider() -> AsyncIterator[str]:
        yield _content("Once upon a time, ")
        yield _content("the end")
        await asyncio.Event().wait()

    _install_provider(monkeypatch, hanging_provider)
    selection = _selection(client, cid, headers, "holdback")
    delivered: list[bytes] = []
    monkeypatch.setattr(client._transport, "app", _drop_client_after(client.app, b"a time,", delivered))

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )
    assert response.status_code == 200
    # The last eight characters never reached the client.
    assert b"the end" not in b"".join(delivered)

    reply = _wait_for_assistant(db, cid)
    assert reply["content"] == "Once upon a time, the end"
    assert _extra(db, reply["id"])["generation_status"] == "interrupted"


def test_stop_signal_mid_stream_saves_partial_as_stopped(settled_api, monkeypatch):
    """A stop signal (``request_stop``) settles the partial as ``stopped``, not ``interrupted``.

    No endpoint raises the signal yet (that is D7 P12's cancel route), so the
    test reaches the request's stream handler directly.
    """
    from tldw_Server_API.app.core.Chat import streaming_utils

    client, db, cid, headers = settled_api
    monkeypatch.setenv("MODERATION_STREAM_BUFFER_CHARS", "0")
    handlers: list[streaming_utils.StreamingResponseHandler] = []

    class RecordingHandler(streaming_utils.StreamingResponseHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            handlers.append(self)

    monkeypatch.setattr(streaming_utils, "StreamingResponseHandler", RecordingHandler)

    async def provider() -> AsyncIterator[str]:
        yield _content("Stopped ")
        yield _content("here")
        handlers[-1].request_stop()
        yield _content(" and never saved")

    _install_provider(monkeypatch, provider)
    selection = _selection(client, cid, headers, "stop")

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )
    assert response.status_code == 200, response.text
    assert "never saved" not in response.text

    reply = _wait_for_assistant(db, cid)
    assert reply["content"] == "Stopped here"
    extra = _extra(db, reply["id"])
    assert extra["generation_status"] == "stopped"
    assert extra["model_id"] == "gpt-4o-mini"
    assert extra["provider"] == "openai"


def _frames(response) -> list[dict[str, Any]]:
    frames = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            frames.append(json.loads(line[6:]))
    return frames


@pytest.mark.parametrize("versioned", [True, False], ids=["native_history", "legacy_save_to_db"])
def test_complete_stream_settles_with_model_provider_finish_reason_and_usage(settled_api, monkeypatch, versioned):
    client, db, cid, headers = settled_api

    async def provider() -> AsyncIterator[str]:
        yield _content("All ")
        yield _content("done", "stop")
        yield _sse({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}})
        yield "data: [DONE]\n\n"

    _install_provider(monkeypatch, provider)
    selection = _selection(client, cid, headers, "complete") if versioned else None

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )
    assert response.status_code == 200, response.text

    replies = _assistant_rows(db, cid)
    assert [row["content"] for row in replies] == ["All done"]
    extra = _extra(db, replies[0]["id"])
    assert extra["generation_status"] == "complete"
    assert extra["finish_reason"] == "stop"
    assert extra["model_id"] == "gpt-4o-mini"
    assert extra["provider"] == "openai"
    assert extra["usage"] == {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
    # Existing metadata is kept next to the generation keys.
    assert extra["sender_role"] == "assistant"


def test_length_finish_reason_settles_as_length(settled_api, monkeypatch):
    client, db, cid, headers = settled_api

    async def provider() -> AsyncIterator[str]:
        yield _content("Cut off mid-sent", "length")
        yield "data: [DONE]\n\n"

    _install_provider(monkeypatch, provider)
    selection = _selection(client, cid, headers, "length")

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )
    assert response.status_code == 200, response.text

    replies = _assistant_rows(db, cid)
    assert [row["content"] for row in replies] == ["Cut off mid-sent"]
    extra = _extra(db, replies[0]["id"])
    assert extra["generation_status"] == "length"
    assert extra["finish_reason"] == "length"


def test_upstream_error_after_partial_output_saves_interrupted_partial(settled_api, monkeypatch):
    client, db, cid, headers = settled_api

    async def failing_provider() -> AsyncIterator[str]:
        yield _content("Half an ")
        yield _content("answer")
        raise RuntimeError("upstream connection reset")

    _install_provider(monkeypatch, failing_provider)
    selection = _selection(client, cid, headers, "upstream-error")

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )
    assert response.status_code == 200, response.text
    frames = _frames(response)
    assert any("error" in frame for frame in frames)

    replies = _assistant_rows(db, cid)
    assert [row["content"] for row in replies] == ["Half an answer"]
    extra = _extra(db, replies[0]["id"])
    assert extra["generation_status"] == "interrupted"
    assert extra["model_id"] == "gpt-4o-mini"
    assert extra["provider"] == "openai"
    # The connected client learns which message holds the partial reply.
    assert {frame["tldw_message_id"] for frame in frames if "tldw_message_id" in frame} == {replies[0]["id"]}


def test_upstream_error_before_any_output_saves_nothing(settled_api, monkeypatch):
    client, db, cid, headers = settled_api

    async def failing_provider() -> AsyncIterator[str]:
        raise RuntimeError("upstream refused the connection")
        yield  # pragma: no cover - makes this an async generator

    _install_provider(monkeypatch, failing_provider)
    selection = _selection(client, cid, headers, "error-first")

    client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )

    time.sleep(0.5)
    assert _assistant_rows(db, cid) == []


def test_legacy_save_to_db_turn_keeps_no_partial_after_a_disconnect(settled_api, monkeypatch):
    """Turns without a native history selection keep their unanswered-turn retry contract."""
    client, db, cid, headers = settled_api
    monkeypatch.setenv("MODERATION_STREAM_BUFFER_CHARS", "0")

    async def hanging_provider() -> AsyncIterator[str]:
        yield _content("Once upon ")
        yield _content("a time")
        await asyncio.Event().wait()

    _install_provider(monkeypatch, hanging_provider)
    delivered: list[bytes] = []
    monkeypatch.setattr(client._transport, "app", _drop_client_after(client.app, b"a time", delivered))

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=None),
    )
    assert response.status_code == 200
    assert b"a time" in b"".join(delivered)

    time.sleep(1.0)
    assert _assistant_rows(db, cid) == []


def test_legacy_failed_turn_retry_still_reuses_the_unanswered_user_message(settled_api, monkeypatch):
    """A provider failure mid-stream leaves no reply, so ``tldw_retry_failed_turn`` still works."""
    client, db, cid, headers = settled_api

    async def failing_provider() -> AsyncIterator[str]:
        yield _content("Half an ")
        yield _content("answer")
        raise RuntimeError("upstream connection reset")

    _install_provider(monkeypatch, failing_provider)
    body = _completion_body(cid, versioned_selection=None)
    failed = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert failed.status_code == 200, failed.text
    assert any("error" in frame for frame in _frames(failed))
    assert _assistant_rows(db, cid) == []

    async def working_provider() -> AsyncIterator[str]:
        yield _content("Whole answer", "stop")
        yield "data: [DONE]\n\n"

    _install_provider(monkeypatch, working_provider)
    body["metadata"] = {"tldw_retry_failed_turn": True}
    retried = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert retried.status_code == 200, retried.text

    rows = db.get_messages_for_conversation(cid)
    assert len([row for row in rows if row["sender"] == "user"]) == 1
    assert [row["content"] for row in rows if row["sender"] == "assistant"] == ["Whole answer"]


def test_client_managed_history_is_unchanged_by_a_disconnect(settled_api, monkeypatch):
    """``save_to_db: false`` keeps the client in charge; the server stores no partial."""
    client, db, cid, headers = settled_api
    monkeypatch.setenv("MODERATION_STREAM_BUFFER_CHARS", "0")

    async def hanging_provider() -> AsyncIterator[str]:
        yield _content("Once upon ")
        yield _content("a time")
        await asyncio.Event().wait()

    _install_provider(monkeypatch, hanging_provider)
    delivered: list[bytes] = []
    monkeypatch.setattr(client._transport, "app", _drop_client_after(client.app, b"a time", delivered))
    body = _completion_body(cid, versioned_selection=None)
    body["save_to_db"] = False

    response = client.post("/api/v1/chat/completions", headers=headers, json=body)
    assert response.status_code == 200
    assert b"a time" in b"".join(delivered)

    time.sleep(1.0)
    assert _assistant_rows(db, cid) == []


def test_non_stream_completion_settles_with_generation_metadata(settled_api):
    client, db, cid, headers = settled_api
    selection = _selection(client, cid, headers, "non-stream")

    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection, stream=False),
    )
    assert response.status_code == 200, response.text

    extra = _extra(db, response.json()["tldw_message_id"])
    assert extra["generation_status"] == "complete"
    assert extra["finish_reason"] == "stop"
    assert extra["model_id"] == "gpt-4o-mini"
    assert extra["provider"] == "openai"
    assert extra["usage"] == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}


def test_message_reads_return_generation_metadata(settled_api, monkeypatch):
    client, db, cid, headers = settled_api

    async def provider() -> AsyncIterator[str]:
        yield _content("Readable", "stop")
        yield _sse({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}})
        yield "data: [DONE]\n\n"

    _install_provider(monkeypatch, provider)
    selection = _selection(client, cid, headers, "read-side")
    response = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json=_completion_body(cid, versioned_selection=selection),
    )
    assert response.status_code == 200, response.text
    reply_id = _assistant_rows(db, cid)[0]["id"]
    expected = {
        "generation_status": "complete",
        "finish_reason": "stop",
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    }

    listed = client.get(f"/api/v1/chats/{cid}/messages", headers=headers, params={"include_metadata": "true"})
    assert listed.status_code == 200, listed.text
    listed_reply = next(message for message in listed.json()["messages"] if message["id"] == reply_id)
    assert expected.items() <= listed_reply["metadata_extra"].items()

    single = client.get(f"/api/v1/messages/{reply_id}", headers=headers, params={"include_metadata": "true"})
    assert single.status_code == 200, single.text
    assert expected.items() <= single.json()["metadata_extra"].items()

    # Clients that do not ask for metadata see the same payload as before.
    plain = client.get(f"/api/v1/chats/{cid}/messages", headers=headers)
    assert plain.status_code == 200, plain.text
    plain_reply = next(message for message in plain.json()["messages"] if message["id"] == reply_id)
    assert plain_reply.get("metadata_extra") is None


async def test_save_function_stores_only_allow_listed_generation_metadata(chacha_db):
    from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db

    cid = chacha_db.add_conversation({"character_id": None, "title": "allow-list", "client_id": "1"})
    junk = {
        "generation_status": "interrupted",
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "usage": {"total_tokens": 3, "cost_usd": 1},
        "api_key": "sk-secret",
        "sender_role": "system",
    }

    assistant_id = await _save_message_turn_to_db(
        chacha_db,
        cid,
        {"role": "assistant", "content": "partial", "generation_metadata": junk},
        use_transaction=True,
    )
    user_id = await _save_message_turn_to_db(
        chacha_db,
        cid,
        {"role": "user", "content": "question", "generation_metadata": junk},
        use_transaction=True,
    )

    assert _extra(chacha_db, assistant_id) == {
        "sender_role": "assistant",
        "generation_status": "interrupted",
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "usage": {"total_tokens": 3},
    }
    assert _extra(chacha_db, user_id) == {"sender_role": "user"}
