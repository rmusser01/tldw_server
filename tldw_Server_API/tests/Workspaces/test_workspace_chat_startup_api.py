"""Strict authenticated startup HTTP behavior without legacy fallback or side effects."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from threading import get_ident
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as sessions
from tldw_Server_API.app.api.v1.endpoints.workspace_chat_startup_transport import WorkspaceStartupRoute
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Workspaces import chat_startup
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_acceptance import _counts

_PATH = "/api/v1/chats/workspace-startup"
_PRIVATE = "private-credential-request-content"
_FORBIDDEN = {
    "character_id": 1, "assistant_kind": "persona", "assistant_id": "persona-a",
    "persona_memory_mode": "read_only", "assistant_startup": {}, "participant_character_ids": [],
    "prompt_preset_id": "preset", "memory_by_character_id": {}, "provider": "openai", "model": "auto",
    "temperature": 0.7, "top_p": 1.0, "repetition_penalty": 1.0, "stop": [],
    "parent_conversation_id": "parent", "forked_from_message_id": "message", "user_id": 1,
    "created_at": "2026-09-27T00:00:00Z", _PRIVATE: _PRIVATE,
}


def _payload(**values: Any) -> dict[str, Any]:
    """Build the current saved-default request without caller identity authority."""
    return {"scope_type": "workspace", "workspace_id": "ws", "workspace_assistant_selection": "inherit",
            "workspace_assistant_default_version": 2, **values}


def _app(db: Any, monkeypatch: pytest.MonkeyPatch, *, authenticated: bool = True):
    """Reuse real routes with owner-scoped DB/auth overrides and observable dependencies."""
    app = FastAPI()
    app.include_router(sessions.router, prefix="/api/v1/chats")
    effects: dict[str, list[Any]] = {"db": [], "rate": [], "loop": []}
    limiter = CharacterRateLimiter(enabled=False, max_chats_per_user=100)

    async def user():
        """Retain authentication before the side-effecting database dependency."""
        if not authenticated:
            raise HTTPException(401, "Not authenticated")
        effects["loop"].append(get_ident())
        return SimpleNamespace(id="user-1")

    async def expected_user(current_user: Any = Depends(sessions.get_request_user)):
        """Keep the existing expected-user authentication dependency graph."""
        return current_user

    async def database():
        """Count initialization separately from transaction/message effects."""
        effects["db"].append(get_ident())
        return db

    async def rate(user_id: Any, operation: str):
        """Observe the existing chat-create operation without contacting external RG."""
        effects["rate"].append((user_id, operation))
        return True, 0

    monkeypatch.setattr(limiter, "check_rate_limit", rate)
    monkeypatch.setattr(sessions, "get_character_rate_limiter", lambda: limiter)
    app.dependency_overrides[sessions.get_request_user] = user
    app.dependency_overrides[sessions.require_expected_user] = expected_user
    app.dependency_overrides[get_chacha_db_for_user] = database
    return app, limiter, effects


@pytest.mark.unit
@pytest.mark.parametrize("field,value", _FORBIDDEN.items())
@pytest.mark.parametrize("explicit_null", [False, True])
def test_forbidden_fields_reject_before_db_and_rate_dependencies(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any, explicit_null: bool,
) -> None:
    """Default-valued and null extras cannot cause a downgraded legacy creation."""
    app, _, effects = _app(object(), monkeypatch)
    with TestClient(app) as client:
        response = client.post(_PATH, content=json.dumps(_payload(**{field: None if explicit_null else value})),
                               headers={"Content-Type": "application/json", "Idempotency-Key": "accepted"})
    assert response.status_code == 422
    assert effects["db"] == effects["rate"] == []
    assert _PRIVATE not in response.text


@pytest.mark.unit
@pytest.mark.parametrize("field", ["workspace_assistant_selection", "workspace_assistant_default_version"])
@pytest.mark.parametrize("value", [None, "none"])
def test_legacy_create_rejects_strict_selector_presence(field: str, value: Any) -> None:
    """An old create model cannot silently ignore a new selector, even explicit null."""
    with pytest.raises(ValidationError):
        ChatSessionCreate.model_validate({field: value})


@pytest.mark.unit
def test_legacy_create_keeps_unrelated_extra_ignore_compatibility() -> None:
    """Only strict selector names are newly rejected by the legacy model."""
    assert ChatSessionCreate.model_validate({"unrelated_legacy_extra": _PRIVATE}).assistant_kind is None


@pytest.mark.unit
def test_strict_route_is_static_and_uses_only_its_bounded_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dynamic chat-id route cannot capture strict startup or inherit its limits."""
    app, _, _ = _app(object(), monkeypatch)
    paths = [route.path for route in app.routes]
    assert _PATH in paths
    assert paths.index(_PATH) < paths.index("/api/v1/chats/{chat_id}")
    strict = next(route for route in app.routes if route.path == _PATH)
    legacy = next(route for route in app.routes if route.path == "/api/v1/chats/")
    assert isinstance(strict, WorkspaceStartupRoute)
    assert not isinstance(legacy, WorkspaceStartupRoute)
    operation = app.openapi()["paths"][_PATH]["post"]
    assert next(item for item in operation["parameters"] if item["name"] == "Idempotency-Key")["required"]


@pytest.mark.unit
@pytest.mark.parametrize("status_code,required_word,excluded_word", [
    ("409", "rejected", "quota"), ("410", "deleted", "permanently"), ("429", "quota", "lifetime"),
])
def test_openapi_describes_conflict_deletion_and_quota_responses(
    monkeypatch: pytest.MonkeyPatch, status_code: str, required_word: str, excluded_word: str,
) -> None:
    """Document current deletion and quota outcomes without promising permanent deletion."""
    app, _, _ = _app(object(), monkeypatch)
    responses = app.openapi()["paths"][_PATH]["post"]["responses"]
    assert status_code in responses
    description = responses[status_code]["description"].lower()
    assert required_word in description
    assert excluded_word not in description


@pytest.mark.unit
def test_openapi_key_pattern_rejects_invalid_embedded_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated clients see a whole-key pattern, not a matching valid substring."""
    app, _, _ = _app(object(), monkeypatch)
    parameters = app.openapi()["paths"][_PATH]["post"]["parameters"]
    pattern = next(item for item in parameters if item["name"] == "Idempotency-Key")["schema"]["pattern"]
    assert all(re.search(pattern, key) is None for key in ("_first", "private space", "!accepted"))


@pytest.mark.unit
def test_authentication_failure_cannot_initialize_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid startup body never bypasses existing authenticated creation policy."""
    app, _, effects = _app(object(), monkeypatch, authenticated=False)
    with TestClient(app) as client:
        response = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "accepted"})
    assert response.status_code == 401
    assert effects["db"] == effects["rate"] == []


@pytest.mark.integration
def test_accept_replay_current_metadata_capacity_and_quota(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP success follows commit; replay retains identity and consumes neither budget."""
    monkeypatch.setenv("WORKSPACE_CHAT_STARTUP_RECEIPT_LIMIT_PER_USER", "1")
    app, limiter, effects = _app(creation_db, monkeypatch)
    worker_threads: list[int] = []
    start = chat_startup.start_workspace_chat

    def observe(*args: Any, **kwargs: Any):
        """Observe real orchestration and committed records, not a replacement result."""
        worker_threads.append(get_ident())
        result = start(*args, **kwargs)
        creation_db.workspace_chat_startups.require_outermost()
        return result

    monkeypatch.setattr(sessions, "start_workspace_chat", observe, raising=False)
    with TestClient(app) as client:
        first = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "private-key-accepted"})
        assert first.status_code == 201
        row = first.json()
        assert (row["assistant_id"], row["persona_memory_mode"], row["assistant_startup"]["workspace_version"]) == (
            "persona-a", "read_only", 2,
        )
        assert "Idempotency-Replayed" not in first.headers
        assert _counts(creation_db) == (1, 1)
        with creation_db.transaction():
            creation_db.update_conversation(row["id"], {"title": "Edited"}, row["version"])
            creation_db.update_workspace("ws", {"assistant_defaults_json": None, "archived": True}, 2)
        limiter._limits = replace(limiter._limits, max_chats_per_user=0)
        replay = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "private-key-accepted"})
        assert replay.status_code == 200
        assert replay.headers["Idempotency-Replayed"] == "true"
        assert (replay.json()["id"], replay.json()["assistant_id"], replay.json()["title"]) == (
            row["id"], "persona-a", "Edited",
        )
        assert _counts(creation_db) == (1, 1)
        assert worker_threads and all(thread not in effects["loop"] for thread in worker_threads)
        for response in (first, replay):
            assert all(private not in response.text for private in (
                "private-key-accepted", "key_digest", "request_fingerprint", "binding_digest", "invalidated_at", "receipt",
            ))
        assert effects["rate"] == [("user-1", "chat_create")] * 2


@pytest.mark.integration
@pytest.mark.parametrize("gate,status,code", [
    ("version", 409, "workspace_assistant_version_conflict"),
    ("archive", 409, "workspace_archived"), ("closing", 409, "workspace_chat_admission_closed"),
    ("deleted", 404, "workspace_not_found"), ("transferred", 404, "workspace_not_found"),
    ("quota", 429, "workspace_chat_quota_exceeded"),
    ("inactive", 409, "workspace_assistant_unavailable"), ("disabled", 503, "persona_feature_disabled"),
    ("configuration", 503, "workspace_chat_startup_configuration_invalid"),
])
def test_fresh_domain_errors_are_bounded_and_leave_no_partial_acceptance(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, gate: str, status: int, code: str,
) -> None:
    """Transport maps only typed failures, with no partial receipt/chat or fallback."""
    app, limiter, _ = _app(creation_db, monkeypatch)
    payload = _payload()
    if gate == "version":
        payload["workspace_assistant_default_version"] = 1
    elif gate in {"archive", "closing", "deleted", "transferred"}:
        field, value = {"archive": ("archived", True), "closing": ("native_chat_admission_closed", True),
                        "deleted": ("deleted", True), "transferred": ("client_id", "other")}[gate]
        with creation_db.transaction() as conn:
            conn.execute(f"UPDATE workspaces SET {field} = ? WHERE id = ?", (value, "ws"))  # nosec B608
    elif gate == "quota":
        limiter._limits = replace(limiter._limits, max_chats_per_user=0)
    elif gate == "inactive":
        with creation_db.transaction() as conn:
            conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
    elif gate == "disabled":
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    else:
        monkeypatch.setenv("WORKSPACE_CHAT_STARTUP_RECEIPT_LIMIT_PER_USER", _PRIVATE)
    with TestClient(app) as client:
        response = client.post(_PATH, json=payload, headers={"Idempotency-Key": "accepted"})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert _PRIVATE not in response.text
    assert _counts(creation_db) == (0, 0)


@pytest.mark.integration
@pytest.mark.parametrize("hard", [False, True])
def test_deleted_accepted_target_is_410_without_recreation(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, hard: bool,
) -> None:
    """A permanent accepted key cannot restore, rebind or recreate a deleted target."""
    app, _, _ = _app(creation_db, monkeypatch)
    with TestClient(app) as client:
        first = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "accepted"})
        assert first.status_code == 201
        if hard:
            creation_db.hard_delete_conversation(first.json()["id"])
        else:
            creation_db.soft_delete_conversation(first.json()["id"], first.json()["version"])
        replay = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "accepted"})
    assert replay.status_code == 410
    assert replay.json()["detail"] == {"code": "workspace_chat_deleted"}
    assert _counts(creation_db) == (1, 0)


@pytest.mark.integration
def test_capacity_request_conflict_and_binding_change_do_not_recreate(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted keys remain permanent across budget rejection and actual identity changes."""
    monkeypatch.setenv("WORKSPACE_CHAT_STARTUP_RECEIPT_LIMIT_PER_USER", "1")
    app, _, _ = _app(creation_db, monkeypatch)
    with TestClient(app) as client:
        first = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "accepted"})
        assert first.status_code == 201
        capacity = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "unseen"})
        assert (capacity.status_code, capacity.json()["detail"]) == (
            409, {"code": "workspace_chat_receipt_capacity_exceeded"},
        )
        conflict = client.post(_PATH, json=_payload(title=None), headers={"Idempotency-Key": "accepted"})
        assert (conflict.status_code, conflict.json()["detail"]) == (409, {"code": "idempotency_key_conflict"})
        with creation_db.transaction():
            creation_db.update_conversation(first.json()["id"], {
                "assistant_kind": None, "assistant_id": None, "character_id": None, "persona_memory_mode": None,
            }, first.json()["version"])
        changed = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "accepted"})
    assert (changed.status_code, changed.json()["detail"]) == (409, {"code": "workspace_chat_startup_changed"})
    assert _counts(creation_db) == (1, 1)


@pytest.mark.integration
@pytest.mark.parametrize("gate,status,code", [
    ("inactive", 409, "persona_unavailable"), ("deleted", 404, "persona_not_found"),
    ("disabled", 503, "persona_feature_disabled"),
])
def test_replay_rechecks_original_persona_after_default_change(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, gate: str, status: int, code: str,
) -> None:
    """Today's missing default cannot downgrade the accepted original Persona on replay."""
    app, _, _ = _app(creation_db, monkeypatch)
    with TestClient(app) as client:
        first = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "accepted"})
        assert first.status_code == 201
        with creation_db.transaction() as conn:
            creation_db.update_workspace("ws", {"assistant_defaults_json": None}, 2)
            if gate != "disabled":
                field = "is_active" if gate == "inactive" else "deleted"
                conn.execute(f"UPDATE persona_profiles SET {field} = ? WHERE id = ?",  # nosec B608
                             (gate == "deleted", "persona-a"))
        if gate == "disabled":
            monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
        replay = client.post(_PATH, json=_payload(), headers={"Idempotency-Key": "accepted"})
    assert replay.status_code == status
    assert replay.json()["detail"]["code"] == code
    assert _counts(creation_db) == (1, 1)


@pytest.mark.integration
def test_explicit_none_does_not_lookup_unavailable_persona_or_seed_sync(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit None is usable without Persona, provider, greeting or Sync authority."""
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)

    def unexpected(*args: Any, **kwargs: Any):
        """None startup must not touch unavailable Persona or legacy side effects."""
        pytest.fail("Strict startup performed a forbidden legacy/Persona effect")

    monkeypatch.setattr(creation_db, "get_persona_profile", unexpected)
    monkeypatch.setattr(sessions, "_active_chat_sync_service", unexpected)
    monkeypatch.setattr(sessions, "post_message_to_conversation", unexpected)
    monkeypatch.setattr(sessions, "ProviderCredentialRuntime", unexpected)
    app, _, _ = _app(creation_db, monkeypatch)
    with TestClient(app) as client:
        response = client.post(_PATH, json={
            "scope_type": "workspace", "workspace_id": "ws", "workspace_assistant_selection": "none",
        }, headers={"Idempotency-Key": "explicit-none"})
    assert response.status_code == 201
    row = response.json()
    assert (row["assistant_kind"], row["assistant_id"], row["assistant_startup"]["source"]) == (
        None, None, "explicit_none",
    )
    assert row["message_count"] == 0
    assert _counts(creation_db) == (1, 1)
