"""Current Persona admission is owner-bound and never silently falls back."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as sessions
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db


def _admission() -> Any:
    """Report a missing contract as an intended assertion rather than import failure."""
    name = "tldw_Server_API.app.core.Persona.conversation_admission"
    assert importlib.util.find_spec(name) is not None, "Shared current Persona admission must exist"
    return importlib.import_module(name).require_current_persona


def _unexpected_lookup(*args: Any, **kwargs: Any) -> Any:
    """Admission rejection/bypass must precede profile access."""
    pytest.fail("Persona lookup must not run")


@pytest.mark.integration
@pytest.mark.parametrize("kind", [None, "character"])
def test_nonpersona_admission_does_not_lookup_profile(
    creation_db: CharactersRAGDB, kind: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None/Character admission does not depend on Persona feature availability."""
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    monkeypatch.setattr(creation_db, "get_persona_profile", _unexpected_lookup)
    row = {"assistant_kind": kind, "character_id": 7 if kind == "character" else None}
    assert _admission()(creation_db, owner_id="user-1", conversation=row) is None


@pytest.mark.integration
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "binding",
    [
        {"assistant_kind": None, "assistant_id": "private-persona", "persona_memory_mode": "read_only"},
        {"assistant_kind": None, "persona_memory_mode": "read_only"},
        {"assistant_kind": "private-invalid", "assistant_id": "private-persona"},
    ],
)
def test_null_or_invalid_kind_with_persona_binding_rejects_before_lookup(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, enabled: bool, binding: dict[str, Any]
) -> None:
    """Contradictory bindings cannot become explicit None when the kind is absent."""
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: enabled)
    monkeypatch.setattr(creation_db, "get_persona_profile", _unexpected_lookup)
    with pytest.raises(HTTPException) as caught:
        _admission()(creation_db, owner_id="user-1", conversation=binding)
    assert (caught.value.status_code, caught.value.detail) == (409, {"code": "persona_binding_invalid"})


@pytest.mark.integration
def test_legacy_character_precedence_bypasses_persona(creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing normalizer, not a second policy, recognizes a legacy Character."""
    monkeypatch.setattr(creation_db, "get_persona_profile", _unexpected_lookup)
    assert _admission()(
        creation_db, owner_id="user-1", conversation={"assistant_kind": None, "character_id": 7, "assistant_id": "old"}
    ) is None


@pytest.mark.unit
@pytest.mark.parametrize("kind", [None, "character", "persona"])
def test_supplied_owner_mismatch_rejects_before_lookup(kind: str | None) -> None:
    """Even non-Persona bypass cannot change the database's immutable owner."""
    db = SimpleNamespace(owner_user_id="owner", client_id="other", get_persona_profile=_unexpected_lookup)
    with pytest.raises(HTTPException) as caught:
        _admission()(db, owner_id="other", conversation={"assistant_kind": kind, "assistant_id": "private-id"})
    assert caught.value.status_code == 404
    assert "private-id" not in str(caught.value.detail)


@pytest.mark.integration
def test_active_persona_uses_owner_not_mutable_writer(creation_db: CharactersRAGDB) -> None:
    """A writer/device name change cannot revoke or grant the owner's profile access."""
    creation_db.client_id = "another-device"
    row = {"assistant_kind": "persona", "assistant_id": "persona-a", "persona_memory_mode": "read_only"}
    profile = _admission()(creation_db, owner_id="user-1", conversation=row)
    assert (profile["id"], profile["user_id"], profile["is_active"]) == ("persona-a", "user-1", True)


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["inactive", "deleted", "missing", "wrong_owner", "disabled"])
def test_current_profile_failures_are_bounded(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Actual current state rejects without returning a fallback or inaccessible identity."""
    row = {"assistant_kind": "persona", "assistant_id": "persona-a"}
    expected = {
        "inactive": (409, "persona_unavailable"),
        "deleted": (404, "persona_not_found"),
        "missing": (404, "persona_not_found"),
        "wrong_owner": (404, "persona_not_found"),
        "disabled": (503, "persona_feature_disabled"),
    }[failure]
    if failure in {"inactive", "deleted", "wrong_owner"}:
        with creation_db.transaction() as conn:
            field, value = {
                "inactive": ("is_active", False),
                "deleted": ("deleted", True),
                "wrong_owner": ("user_id", "other"),
            }[failure]
            # Only fixed test-owned schema columns are interpolated.
            conn.execute(f"UPDATE persona_profiles SET {field} = ? WHERE id = ?", (value, "persona-a"))  # nosec B608
    elif failure == "missing":
        row["assistant_id"] = "private-missing-id"
    else:
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
        monkeypatch.setattr(creation_db, "get_persona_profile", _unexpected_lookup)
    with pytest.raises(HTTPException) as caught:
        _admission()(creation_db, owner_id="user-1", conversation=row)
    assert (caught.value.status_code, caught.value.detail["code"]) == expected
    assert "persona-a" not in str(caught.value.detail)
    assert "private-missing-id" not in str(caught.value.detail)
    if failure == "inactive":
        assert caught.value.detail["reason"] == "persona_unavailable"


@pytest.mark.integration
@pytest.mark.parametrize(
    "binding",
    [
        {},
        {"assistant_id": ""},
        {"assistant_id": "   "},
        {"assistant_id": True},
        {"assistant_id": "persona-a", "persona_memory_mode": "private-invalid"},
    ],
)
def test_malformed_persona_binding_is_not_a_fallback(
    creation_db: CharactersRAGDB, binding: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing or invalid Persona identity fails before any profile lookup."""
    monkeypatch.setattr(creation_db, "get_persona_profile", _unexpected_lookup)
    with pytest.raises(HTTPException) as caught:
        _admission()(creation_db, owner_id="user-1", conversation={"assistant_kind": "persona", **binding})
    assert (caught.value.status_code, caught.value.detail) == (409, {"code": "persona_binding_invalid"})


@pytest.mark.integration
def test_transaction_admission_uses_supplied_lock_and_storage_errors_propagate(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creation/replay can lock current state; failed storage must never grant admission."""
    row = {"assistant_kind": "persona", "assistant_id": "persona-a"}
    getter = creation_db.get_persona_profile
    calls: list[dict[str, Any]] = []

    def record(persona_id: str, **kwargs: Any) -> Any:
        """Observe actual transaction-local lookup without replacing its result."""
        calls.append(kwargs)
        return getter(persona_id, **kwargs)

    monkeypatch.setattr(creation_db, "get_persona_profile", record)
    with creation_db.transaction() as conn:
        assert _admission()(creation_db, owner_id="user-1", conversation=row, conn=conn)["id"] == "persona-a"
        assert calls == [{"user_id": "user-1", "include_deleted": False, "conn": conn, "for_update": True}]

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Represent a storage outage, not Persona absence."""
        raise RuntimeError("private-storage-error")

    monkeypatch.setattr(creation_db, "get_persona_profile", fail)
    with pytest.raises(RuntimeError, match="private-storage-error"):
        _admission()(creation_db, owner_id="user-1", conversation=row)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["inactive", "deleted", "missing", "wrong_owner", "disabled", "malformed"])
async def test_direct_service_rechecks_current_persona_before_context(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Direct service use cannot bypass revocation or use mutable device ownership."""
    cid = creation_db.add_conversation({"assistant_kind": "persona", "assistant_id": "persona-a"})
    expected = {"inactive": (409, "persona_unavailable"), "deleted": (404, "persona_not_found"),
                "missing": (404, "persona_not_found"), "wrong_owner": (404, "persona_not_found"),
                "disabled": (503, "persona_feature_disabled"), "malformed": (409, "persona_binding_invalid")}[failure]
    with creation_db.transaction() as conn:
        if failure == "inactive":
            conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
        elif failure == "deleted":
            conn.execute("UPDATE persona_profiles SET deleted = ? WHERE id = ?", (True, "persona-a"))
        elif failure == "wrong_owner":
            conn.execute("UPDATE persona_profiles SET user_id = ? WHERE id = ?", ("other", "persona-a"))
        elif failure in {"missing", "malformed"}:
            conn.execute("UPDATE conversations SET assistant_id = ? WHERE id = ?",
                         (None if failure == "malformed" else "private-missing", cid))
    if failure == "wrong_owner":
        creation_db.client_id = "other"
    if failure == "disabled":
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    context_calls = []

    async def record_context(*args: Any, **kwargs: Any) -> None:
        """Observe a pre-admission Character lookup without generating anything."""
        context_calls.append((args, kwargs))

    monkeypatch.setattr(chat_service, "_resolve_default_character_id", record_context)
    with pytest.raises(HTTPException) as caught:
        await chat_service._resolve_assistant_context_for_chat(
            chat_db=creation_db, request_data=SimpleNamespace(character_id=None),
            loop=asyncio.get_running_loop(), conversation_id=cid,
        )
    assert (caught.value.status_code, caught.value.detail["code"]) == expected
    assert context_calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_direct_service_uses_current_immutable_owner_profile(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching owner with a different writer still gets the real Persona projection."""
    cid = creation_db.add_conversation({"assistant_kind": "persona", "assistant_id": "persona-a"})
    creation_db.client_id = "other-device"
    profile = creation_db.get_persona_profile("persona-a", user_id="user-1")

    async def unexpected_context(*args: Any, **kwargs: Any) -> None:
        """A Persona context does not need a default Character lookup."""
        pytest.fail("Persona must be admitted before Character context")

    monkeypatch.setattr(chat_service, "_resolve_default_character_id", unexpected_context)
    projection, character_id, row, context = await chat_service._resolve_assistant_context_for_chat(
        chat_db=creation_db, request_data=SimpleNamespace(character_id=None),
        loop=asyncio.get_running_loop(), conversation_id=cid,
    )
    assert projection["name"] == profile["name"]
    assert character_id is None
    assert row["id"] == cid
    assert context["assistant_kind"] == "persona"


def _session_app(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, list[str]]:
    """Use the real session routes; stop before any prompt/provider/message effect."""
    app = FastAPI()
    app.include_router(sessions.router, prefix="/api/v1/chats")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    effects = []

    def record_effect(*args: Any, **kwargs: Any) -> Any:
        """An admission or scope rejection must happen before context assembly."""
        effects.append("context")
        raise RuntimeError("Context must not run on rejected admission")

    monkeypatch.setattr(sessions, "_get_completion_settings_row", record_effect)
    return app, effects


@pytest.mark.integration
@pytest.mark.parametrize("route", ["completions", "prompt-preview", "complete-v2"])
@pytest.mark.parametrize("failure", ["inactive", "deleted", "missing", "wrong_owner", "disabled", "malformed"])
def test_global_session_rechecks_persona_before_context(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, route: str, failure: str,
) -> None:
    """Every global session consumer rejects unusable Persona before prompt assembly."""
    cid = creation_db.add_conversation({"assistant_kind": "persona", "assistant_id": "persona-a"})
    expected = {"inactive": (409, "persona_unavailable"), "deleted": (404, "persona_not_found"),
                "missing": (404, "persona_not_found"), "wrong_owner": (404, "persona_not_found"),
                "disabled": (503, "persona_feature_disabled"), "malformed": (409, "persona_binding_invalid")}[failure]
    with creation_db.transaction() as conn:
        if failure == "inactive":
            conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "persona-a"))
        elif failure == "deleted":
            conn.execute("UPDATE persona_profiles SET deleted = ? WHERE id = ?", (True, "persona-a"))
        elif failure == "wrong_owner":
            conn.execute("UPDATE persona_profiles SET user_id = ? WHERE id = ?", ("other", "persona-a"))
        elif failure in {"missing", "malformed"}:
            conn.execute("UPDATE conversations SET assistant_id = ? WHERE id = ?",
                         (None if failure == "malformed" else "private-missing", cid))
    if failure == "disabled":
        monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: False)
    app, effects = _session_app(creation_db, monkeypatch)
    with TestClient(app) as client:
        response = client.post(f"/api/v1/chats/{cid}/{route}", json={})
    assert response.status_code == expected[0], response.text
    assert response.json()["detail"]["code"] == expected[1]
    assert effects == []


@pytest.mark.integration
@pytest.mark.parametrize("route", ["completions", "prompt-preview", "complete-v2"])
@pytest.mark.parametrize("query", [{"scope_type": "global"}, {"scope_type": ""}, {"workspace_id": ""}, {"workspace_id": "ws"}])
def test_global_session_rejects_supplied_workspace_scope_options(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, route: str, query: dict[str, str],
) -> None:
    """Even empty/default-valued Workspace query options cannot be silently ignored."""
    cid = creation_db.add_conversation({"assistant_kind": "persona", "assistant_id": "persona-a"})
    app, effects = _session_app(creation_db, monkeypatch)
    with TestClient(app) as client:
        response = client.post(f"/api/v1/chats/{cid}/{route}", json={}, params=query)
    assert response.status_code == 422, response.text
    assert effects == []


@pytest.mark.integration
@pytest.mark.parametrize("route", ["completions", "prompt-preview", "complete-v2"])
def test_global_session_keeps_workspace_chat_inaccessible(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    """Owned Workspace ids retain the global-only 404, not a generation downgrade."""
    cid = creation_db.add_conversation({"assistant_kind": "persona", "assistant_id": "persona-a",
                                      "scope_type": "workspace", "workspace_id": "ws"})
    app, effects = _session_app(creation_db, monkeypatch)
    with TestClient(app) as client:
        response = client.post(f"/api/v1/chats/{cid}/{route}", json={})
    assert response.status_code == 404, response.text
    assert effects == []


@pytest.mark.integration
@pytest.mark.parametrize("route", ["completions", "complete-v2"])
def test_character_only_session_cannot_downgrade_active_persona(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    """An active Persona cannot be sent through a Character-only preparation/generator."""
    cid = creation_db.add_conversation({"assistant_kind": "persona", "assistant_id": "persona-a"})
    app, effects = _session_app(creation_db, monkeypatch)
    with TestClient(app) as client:
        response = client.post(f"/api/v1/chats/{cid}/{route}", json={})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "persona_session_generation_unsupported"
    assert effects == []
