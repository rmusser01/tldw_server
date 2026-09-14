"""Accepted Buddy turns exercise the real authenticated Chat ASGI boundary."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest
from fastapi import HTTPException, Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.schemas.buddy_turns import BuddyTurnCreate
from tldw_Server_API.app.core.Buddy.service import BuddyService
from tldw_Server_API.app.core.Buddy.turns import BuddyTurnRuntime
from tldw_Server_API.app.core.DB_Management.Buddy_Turns_DB import (
    BuddyPublicationRevokedError,
    BuddyRuntimeBusyError,
    BuddyTurnRepository,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    _create_persona_conversation,
)
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_client as _persona_chat_client,
)
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_db as _persona_chat_db,
)

pytestmark = pytest.mark.integration
persona_chat_client = _persona_chat_client
persona_chat_db = _persona_chat_db


@pytest.fixture(autouse=True)
def _fake_provider_credentials(monkeypatch):
    # The provider itself is mocked by persona_chat_client; run all HTTP policy
    # and persistence while avoiding dependence on a developer's configured key.
    monkeypatch.setenv("CHAT_FORCE_MOCK", "1")


def _attach(client, headers, conversation_id, *, slot="default", workspace_id=None):
    created = client.post(
        "/api/v1/buddies",
        headers=headers,
        json={
            "name": "Turn Buddy",
            "source": {"kind": "starter", "starter_id": "pixel-migu"},
        },
    )
    assert created.status_code == 201, created.text
    response = client.put(
        f"/api/v1/buddies/attachment?client_slot={slot}",
        headers=headers,
        json={
            "expected_version": 0,
            "buddy_id": created.json()["id"],
            "scope_type": "workspace" if workspace_id else "conversation",
            "scope_id": workspace_id or conversation_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["version"]


def _send(client, headers, conversation_id, *, slot="default", key=None, text="Remember this.", **updates):
    return client.post(
        f"/api/v1/buddies/turns?client_slot={slot}",
        headers=headers,
        json={
            "conversation_id": conversation_id,
            "text": text,
            "client_request_id": key or uuid.uuid4().hex,
            "expected_attachment_version": 1,
            "provider": "openai",
            "model": "gpt-4",
            **updates,
        },
    )


def _terminal(client, headers, turn_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/buddies/turns/{turn_id}", headers=headers)
        assert response.status_code == 200, response.text
        item = response.json()
        if item["status"] not in {"queued", "running"}:
            return item
        time.sleep(0.02)
    pytest.fail("accepted turn did not reach a terminal status")


@pytest.mark.parametrize("identity", ["none", "character", "persona"])
def test_main_chat_persists_exact_conversation_for_each_identity(persona_chat_client, persona_chat_db, identity):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    if identity == "persona":
        conversation_id, _ = _create_persona_conversation(db, persona_id="turn-persona")
    else:
        values = {"title": identity, "client_id": "1"}
        if identity == "character":
            values["character_id"] = db.get_character_card_by_name("Source Character")["id"]
        conversation_id = db.add_conversation(values)
    _attach(client, headers, conversation_id)
    response = _send(client, headers, conversation_id)
    assert response.status_code == 202, response.text
    done = _terminal(client, headers, response.json()["id"])
    assert done["status"] == "completed", str(done)
    assert done["conversation_id"] == conversation_id
    assert done["result_message_id"]
    assert provider.call_count == 1
    messages = db.get_messages_for_conversation(conversation_id)
    assert any(row["id"] == done["result_message_id"] for row in messages)


def test_detach_does_not_stop_accepted_fifo_and_other_conversation_runs(persona_chat_client, persona_chat_db):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    first = db.add_conversation({"title": "First", "client_id": "1"})
    second = db.add_conversation({"title": "Second", "client_id": "1"})
    _attach(client, headers, first)
    _attach(client, headers, second, slot="other")
    entered, release = threading.Event(), threading.Event()
    normal = provider.return_value

    def blocked(**kwargs):
        content = str(kwargs.get("messages_payload"))
        if "hold first" in content and "next first" not in content:
            entered.set()
            assert release.wait(8)
        return normal

    provider.side_effect = blocked
    try:
        one = _send(client, headers, first, text="hold first").json()
        assert entered.wait(5)
        two = _send(client, headers, first, text="next first").json()
        other = _send(client, headers, second, slot="other").json()
        assert _terminal(client, headers, other["id"])["status"] == "completed"
        assert client.get(f"/api/v1/buddies/turns/{two['id']}", headers=headers).json()["status"] == "queued"
        assert client.delete("/api/v1/buddies/attachment?expected_version=1", headers=headers).status_code == 200
    finally:
        release.set()
    assert _terminal(client, headers, one["id"])["status"] == "completed"
    assert _terminal(client, headers, two["id"])["status"] == "completed"


def test_idempotency_and_stop_prevent_duplicate_dispatch_and_late_reply(persona_chat_client, persona_chat_db):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    conversation_id = db.add_conversation({"title": "Stop", "client_id": "1"})
    _attach(client, headers, conversation_id)
    entered, release, provider_finished = threading.Event(), threading.Event(), threading.Event()
    normal = provider.return_value

    def blocked(**kwargs):
        entered.set()
        try:
            assert release.wait(8)
            return normal
        finally:
            provider_finished.set()

    provider.side_effect = blocked
    try:
        response = _send(client, headers, conversation_id, key="same-turn")
        assert response.status_code == 202, response.text
        turn_id = response.json()["id"]
        assert entered.wait(5)
        worker = app.state.buddy_turn_runtime._workers[("1", conversation_id)]
        duplicate = _send(client, headers, conversation_id, key="same-turn")
        assert duplicate.json()["id"] == turn_id
        assert _send(client, headers, conversation_id, key="same-turn", text="different").status_code == 409
        stopped = client.post(f"/api/v1/buddies/turns/{turn_id}/stop", headers=headers)
        assert stopped.json()["status"] == "stopped"
    finally:
        release.set()
    assert _terminal(client, headers, turn_id)["status"] == "stopped"
    assert provider_finished.wait(5)

    async def wait_for_publication_owner() -> None:
        await asyncio.wait_for(asyncio.shield(worker), timeout=5)

    client.portal.call(wait_for_publication_owner)
    assert provider.call_count == 1
    assert not any(row["sender"] == "assistant" for row in db.get_messages_for_conversation(conversation_id))


def test_starter_artwork_content_is_authenticated_and_private(persona_chat_client, monkeypatch):
    client, headers, _ = persona_chat_client
    detail = client.get("/api/v1/persona/visual-starter-packs/pixel-migu", headers=headers)
    assert detail.status_code == 200, detail.text
    asset = detail.json()["assets"][0]
    url = f"/api/v1/persona/visual-starter-packs/pixel-migu/assets/{asset['asset_key']}/content"
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(response.content) == asset["byte_size"]
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    missing = client.get("/api/v1/persona/visual-starter-packs/pixel-migu/assets/missing/content", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "starter_asset_not_found"
    assert client.get(url.replace("pixel-migu", "missing-starter"), headers=headers).status_code == 404

    async def unauthenticated() -> None:
        raise HTTPException(401, "Authentication required")

    monkeypatch.setitem(app.dependency_overrides, get_request_user, unauthenticated)
    rejected = client.get(url)
    assert rejected.status_code == 401
    assert rejected.content != response.content


def test_unexpected_configuration_value_error_is_not_a_client_validation_error(
    persona_chat_client, persona_chat_db, monkeypatch
):
    client, headers, _ = persona_chat_client
    conversation_id = persona_chat_db.add_conversation({"title": "Configuration failure", "client_id": "1"})
    _attach(client, headers, conversation_id)

    def broken_settings(*_args, **_kwargs):
        raise ValueError("unexpected implementation failure")

    monkeypatch.setattr(persona_chat_db, "get_roleplay_resume_state", broken_settings)
    # TestClient re-raises unhandled server exceptions. The endpoint must not
    # turn unexpected implementation errors into a client configuration 422.
    with pytest.raises(ValueError, match="unexpected implementation failure"):
        _send(client, headers, conversation_id)


def test_owned_workspace_scope_and_missing_model_fail_closed(persona_chat_client, persona_chat_db):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    db.upsert_workspace("turn-workspace", "Research")
    conversation_id = db.add_conversation(
        {"title": "Scoped", "client_id": "1", "scope_type": "workspace", "workspace_id": "turn-workspace"}
    )
    outside = db.add_conversation({"title": "Outside", "client_id": "1"})
    _attach(client, headers, conversation_id, workspace_id="turn-workspace")
    assert _send(client, headers, outside).status_code == 404
    assert _send(client, headers, conversation_id, model=None, provider=None).status_code == 422
    assert _send(client, headers, conversation_id, expected_attachment_version=2).status_code == 409
    response = _send(client, headers, conversation_id)
    assert response.status_code == 202, response.text
    done = _terminal(client, headers, response.json()["id"])
    assert done["status"] == "completed", str(done)
    assert done["workspace_id"] == "turn-workspace"
    assert provider.call_count == 1


@pytest.mark.parametrize("change", ["revision", "workspace_deleted"])
def test_access_or_identity_change_during_provider_revokes_publication(persona_chat_client, persona_chat_db, change):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    db.upsert_workspace("mutable-workspace", "Research")
    conversation_id = db.add_conversation(
        {"title": "Mutable", "client_id": "1", "scope_type": "workspace", "workspace_id": "mutable-workspace"}
    )
    _attach(client, headers, conversation_id)
    entered, release = threading.Event(), threading.Event()
    normal = provider.return_value

    def blocked(**kwargs):
        entered.set()
        assert release.wait(8)
        return normal

    provider.side_effect = blocked
    try:
        response = _send(client, headers, conversation_id)
        assert response.status_code == 202, response.text
        assert entered.wait(5)
        with db.transaction() as conn:
            if change == "revision":
                conn.execute("UPDATE conversations SET version = version + 1 WHERE id = ?", (conversation_id,))
            else:
                conn.execute("UPDATE workspaces SET deleted = TRUE WHERE id = ?", ("mutable-workspace",))
    finally:
        release.set()
    done = _terminal(client, headers, response.json()["id"])
    assert done["status"] == "failed", str(done)
    assert done["result_message_id"] is None
    assert not any(row["sender"] == "assistant" for row in db.get_messages_for_conversation(conversation_id))


def test_cancelled_acceptance_subscriber_does_not_cancel_owner(persona_chat_client, persona_chat_db, monkeypatch):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    conversation_id = db.add_conversation({"title": "Disconnected", "client_id": "1"})
    _attach(client, headers, conversation_id)
    entered, release = threading.Event(), threading.Event()
    original_create = BuddyTurnRepository.create

    def held_create(repository, value, **kwargs):
        result = original_create(repository, value, **kwargs)
        entered.set()
        assert release.wait(8)
        return result

    monkeypatch.setattr(BuddyTurnRepository, "create", held_create)

    async def disconnected_request():
        runtime = BuddyTurnRuntime(app)
        app.state.buddy_turn_runtime = runtime
        request = BuddyTurnCreate(
            conversation_id=conversation_id,
            text="Continue after disconnect",
            client_request_id="disconnected",
            expected_attachment_version=1,
            provider="openai",
            model="gpt-4",
        )
        task = asyncio.create_task(runtime.accept(BuddyService(db, "1"), "default", request, dict(headers), None))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()

    try:
        client.portal.call(disconnected_request)
    finally:
        release.set()
    row = BuddyTurnRepository(db, "1").by_key("disconnected")
    assert _terminal(client, headers, row["id"])["status"] == "completed"
    assert provider.call_count == 1


def test_expired_process_owner_is_terminal_and_cannot_publish(persona_chat_db):
    db = persona_chat_db
    conversation_id = db.add_conversation({"title": "Restart", "client_id": "1"})
    repository = BuddyTurnRepository(db, "1")
    repository.claim("old-owner")
    row = repository.create(
        {
            "id": "interrupted",
            "owner_id": "old-owner",
            "client_slot": "default",
            "client_request_id": "restart-key",
            "request_digest": "digest",
            "conversation_id": conversation_id,
            "conversation_title": "Restart",
            "conversation_version": 1,
            "workspace_id": None,
            "attachment_version": 1,
            "created_at": "2026-09-08T00:00:00Z",
        }
    )
    repository.transition(row["id"], "running")
    with pytest.raises(BuddyRuntimeBusyError):
        repository.claim("new-owner")
    with db.transaction() as conn:
        conn.execute("UPDATE buddy_turn_owners SET expires_at = 0 WHERE user_id = ?", ("1",))
    repository.expire_interrupted()
    assert repository.get(row["id"])["error_code"] == "interrupted_unknown"
    repository.claim("new-owner")
    with db.transaction() as conn, pytest.raises(BuddyPublicationRevokedError):
        repository.assert_publication(conn, row, conversation_id)
    assert repository.by_key("restart-key")["id"] == "interrupted"
    columns = set(repository.get(row["id"]))
    assert not {"text", "headers", "authorization", "api_key", "messages"} & columns


def test_inner_chat_authentication_is_reentered_after_acceptance(persona_chat_client, persona_chat_db, monkeypatch):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    conversation_id = db.add_conversation({"title": "Policy", "client_id": "1"})
    _attach(client, headers, conversation_id)
    admitted_paths = []

    async def user_for_request(request: Request):
        admitted_paths.append(request.url.path)
        if request.url.path == "/api/v1/chat/completions":
            raise HTTPException(403, "Chat access revoked")
        return User(id=1, username="test_user", is_active=True)

    with monkeypatch.context() as patch:
        patch.setitem(app.dependency_overrides, get_request_user, user_for_request)
        response = _send(client, headers, conversation_id)
        assert response.status_code == 202, response.text
        done = _terminal(client, headers, response.json()["id"])
    assert done["status"] == "failed"
    assert done["error_code"] == "chat_http_403"
    assert "/api/v1/chat/completions" in admitted_paths
    assert provider.call_count == 0
    assert db.get_messages_for_conversation(conversation_id) == []


def test_queue_is_bounded_and_failed_key_is_not_retried(persona_chat_client, persona_chat_db):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    conversation_id = db.add_conversation({"title": "Bounded", "client_id": "1"})
    _attach(client, headers, conversation_id)
    entered, release = threading.Event(), threading.Event()

    def failed_provider(**kwargs):
        entered.set()
        assert release.wait(8)
        raise RuntimeError("controlled provider failure")

    provider.side_effect = failed_provider
    try:
        response = _send(client, headers, conversation_id, key="failed-key")
        assert response.status_code == 202, response.text
        assert entered.wait(5)
        app.state.buddy_turn_runtime.per_user_capacity = 1
        assert _send(client, headers, conversation_id).status_code == 429
    finally:
        release.set()
    done = _terminal(client, headers, response.json()["id"])
    assert done["status"] == "failed", str(done)
    calls = provider.call_count
    replay = _send(client, headers, conversation_id, key="failed-key")
    assert replay.json()["id"] == done["id"]
    assert replay.json()["status"] == "failed"
    assert provider.call_count == calls


def test_turn_status_and_stop_are_principal_scoped(persona_chat_client, persona_chat_db, monkeypatch):
    client, headers, _ = persona_chat_client
    conversation_id = persona_chat_db.add_conversation({"title": "Private", "client_id": "1"})
    _attach(client, headers, conversation_id)
    response = _send(client, headers, conversation_id)
    assert response.status_code == 202, response.text
    turn_id = response.json()["id"]
    assert _terminal(client, headers, turn_id)["status"] == "completed"
    with monkeypatch.context() as patch:
        patch.setitem(app.dependency_overrides, get_request_user, lambda: User(id=2, username="other", is_active=True))
        assert client.get("/api/v1/buddies/turns", headers=headers).json()["turns"] == []
        assert client.get(f"/api/v1/buddies/turns/{turn_id}", headers=headers).status_code == 404
        assert client.post(f"/api/v1/buddies/turns/{turn_id}/stop", headers=headers).status_code == 404


@pytest.mark.parametrize("failure_point", ["read", "transition"])
def test_pre_dispatch_database_failure_does_not_drop_next_accepted_turn(
    persona_chat_client, persona_chat_db, monkeypatch, failure_point
):
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    conversation_id = db.add_conversation({"title": "Keep the queue", "client_id": "1"})
    _attach(client, headers, conversation_id)
    entered, release = threading.Event(), threading.Event()
    original_get = BuddyTurnRepository.get
    original_transition = BuddyTurnRepository.transition
    first_reads = 0
    failed = False

    def fail_once():
        nonlocal failed
        failed = True
        entered.set()
        assert release.wait(8)
        raise CharactersRAGDBError("controlled transient database error")

    def read(repository, turn_id):
        nonlocal first_reads
        row = original_get(repository, turn_id)
        if row["client_request_id"] == "first-in-queue":
            first_reads += 1
            # create() returns the first read; the second is worker preflight.
            if failure_point == "read" and first_reads == 2:
                fail_once()
        return row

    def transition(repository, turn_id, status, **kwargs):
        if failure_point == "transition" and status == "running" and not failed:
            fail_once()
        return original_transition(repository, turn_id, status, **kwargs)

    monkeypatch.setattr(BuddyTurnRepository, "get", read)
    monkeypatch.setattr(BuddyTurnRepository, "transition", transition)
    try:
        first = _send(client, headers, conversation_id, key="first-in-queue")
        assert first.status_code == 202, first.text
        assert entered.wait(5)
        second = _send(client, headers, conversation_id, key="second-in-queue")
        assert second.status_code == 202, second.text
    finally:
        release.set()
    assert _terminal(client, headers, first.json()["id"])["status"] == "failed"
    assert _terminal(client, headers, second.json()["id"])["status"] == "completed"
    assert provider.call_count == 1
