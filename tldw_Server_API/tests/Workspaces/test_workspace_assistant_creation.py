"""Workspace defaults are resolved once at server conversation creation."""

from __future__ import annotations

import importlib

import pytest
from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def persona_enabled(monkeypatch):
    """Make selection independent of the deployment's Persona feature flag."""
    monkeypatch.setattr(feature_flags, "settings", {"PERSONA_ENABLED": True})


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "defaults.sqlite", "1")
    character_id = database.add_character_card({"name": "Default source", "system_prompt": "Help."})
    database.create_persona_profile(
        {
            "id": "workspace-persona",
            "user_id": "1",
            "name": "Workspace Persona",
            "character_card_id": character_id,
            "mode": "session_scoped",
            "is_active": True,
        }
    )
    workspace = database.upsert_workspace("ws", "Workspace")
    database.update_workspace(
        "ws",
        {
            "assistant_defaults_json": {
                "assistant_kind": "persona",
                "assistant_id": "workspace-persona",
                "persona_memory_mode": "read_only",
            }
        },
        expected_version=workspace["version"],
    )
    yield database
    database.close_connection()


def _resolve(db, payload):
    module = importlib.import_module("tldw_Server_API.app.core.Workspaces.assistant_defaults")
    return module.resolve_new_conversation_assistant(db, user_id="1", request=ChatSessionCreate.model_validate(payload))


def test_omitted_choice_inherits_default_without_mutating_existing_conversations(db):
    existing_id = db.add_conversation(
        {"title": "Existing", "client_id": "1", "scope_type": "workspace", "workspace_id": "ws"}
    )
    result = _resolve(db, {"scope_type": "workspace", "workspace_id": "ws"})
    assert result.assistant_kind == "persona"
    assert result.assistant_id == "workspace-persona"
    assert result.persona_memory_mode == "read_only"
    assert db.get_conversation_by_id(existing_id)["assistant_id"] is None


@pytest.mark.parametrize(
    "choice",
    [
        {"assistant_kind": None},
        {"assistant_id": None},
        {"character_id": None},
        {"assistant_kind": "persona", "assistant_id": "explicit-persona"},
    ],
)
def test_explicit_choice_including_none_wins_over_workspace_default(db, choice):
    result = _resolve(db, {"scope_type": "workspace", "workspace_id": "ws", **choice})
    assert result.assistant_id == choice.get("assistant_id")


def test_global_and_fork_requests_do_not_inherit_workspace_default(db):
    assert _resolve(db, {}).assistant_id is None
    assert (
        _resolve(
            db, {"scope_type": "workspace", "workspace_id": "ws", "parent_conversation_id": "existing"}
        ).assistant_id
        is None
    )


def test_deleted_default_persona_fails_closed(db):
    profile = db.get_persona_profile("workspace-persona", user_id="1")
    db.soft_delete_persona_profile(persona_id="workspace-persona", user_id="1", expected_version=profile["version"])
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        _resolve(db, {"scope_type": "workspace", "workspace_id": "ws"})
    assert error.value.status_code == 409


def test_missing_workspace_fails_before_default_lookup(db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        _resolve(db, {"scope_type": "workspace", "workspace_id": "missing"})
    assert error.value.status_code == 404


def test_create_chat_endpoint_persists_inherited_and_explicit_none_choices(db, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint

    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/chats")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[endpoint.require_expected_user] = lambda: None
    monkeypatch.setattr(
        endpoint,
        "get_character_rate_limiter",
        lambda: SimpleNamespace(check_rate_limit=AsyncMock(), check_chat_limit=AsyncMock()),
    )
    monkeypatch.setattr(endpoint, "_active_chat_sync_service", lambda *args: None)
    with TestClient(app) as client:
        inherited = client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws"})
        assert inherited.status_code == 201, inherited.text
        row = db.get_conversation_by_id(inherited.json()["id"])
        assert row["assistant_kind"] == "persona"
        assert row["assistant_id"] == "workspace-persona"
        assert row["persona_memory_mode"] == "read_only"
        explicit = client.post(
            "/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws", "assistant_kind": None}
        )
        assert explicit.status_code == 201, explicit.text
        assert db.get_conversation_by_id(explicit.json()["id"])["assistant_id"] is None


def _resolve_startup(db, payload, **kwargs):
    """Exercise the provenance-returning boundary on schema-normalized requests."""
    module = importlib.import_module("tldw_Server_API.app.core.Workspaces.assistant_defaults")
    return module.resolve_workspace_assistant_startup(
        db, user_id="1", request=ChatSessionCreate.model_validate(payload), **kwargs,
    )


@pytest.mark.parametrize(
    "payload,source,assistant_id",
    [
        ({}, "unknown", None),
        ({"scope_type": "global", "assistant_kind": "persona", "assistant_id": "workspace-persona"},
         "unknown", "workspace-persona"),
        ({"scope_type": "workspace", "workspace_id": "ws"}, "workspace_default", "workspace-persona"),
        *[
            ({"scope_type": "workspace", "workspace_id": "ws", field: None}, "explicit_none", None)
            for field in ("assistant_kind", "assistant_id", "character_id")
        ],
        ({"scope_type": "workspace", "workspace_id": "ws", "assistant_kind": "persona",
          "assistant_id": "workspace-persona", "character_id": None}, "explicit", "workspace-persona"),
        ({"scope_type": "workspace", "workspace_id": "ws", "character_id": 1,
          "assistant_id": None}, "explicit", "1"),
        ({"scope_type": "workspace", "workspace_id": "ws", "parent_conversation_id": "validated-parent"},
         "fork", None),
    ],
)
def test_startup_selection_uses_original_choice_not_identity_equality(db, payload, source, assistant_id):
    """Omission, explicit null and equal explicit identity retain distinct origins."""
    resolved = _resolve_startup(db, payload)
    assert resolved.startup.model_dump() == {
        "schema_version": 1, "source": source,
        "workspace_id": "ws" if source == "workspace_default" else None,
        "workspace_version": 2 if source == "workspace_default" else None,
    }
    assert resolved.request.assistant_id == assistant_id
    if source == "workspace_default":
        assert resolved.display_name == "Workspace Persona"
        assert resolved.request.persona_memory_mode == "read_only"


@pytest.mark.parametrize("cleared", [False, True])
def test_unset_and_cleared_defaults_record_examined_workspace(db, cleared):
    """A stored opt-out is system fallback, not request-level explicit None."""
    workspace = db.upsert_workspace("empty", "Empty")
    if cleared:
        workspace = db.update_workspace("empty", {"assistant_defaults_json": None}, workspace["version"])
    resolved = _resolve_startup(db, {"scope_type": "workspace", "workspace_id": "empty"})
    assert resolved.startup.model_dump() == {
        "schema_version": 1, "source": "system_fallback", "workspace_id": "empty",
        "workspace_version": 2 if cleared else 1,
    }
    assert resolved.request.assistant_id is None


def test_inherited_read_write_is_saved_mode_and_does_not_mutate_request(db):
    """A confirmed saved mode is inherited while the original omission stays intact."""
    workspace = db.get_workspace("ws")
    stored = dict(workspace["assistant_defaults_json"], persona_memory_mode="read_write")
    db.update_workspace("ws", {"assistant_defaults_json": stored}, workspace["version"])
    request = ChatSessionCreate(scope_type="workspace", workspace_id="ws")
    module = importlib.import_module("tldw_Server_API.app.core.Workspaces.assistant_defaults")
    result = module.resolve_workspace_assistant_startup(db, user_id="1", request=request)
    assert result.request.persona_memory_mode == "read_write"
    assert request.model_fields_set == {"scope_type", "workspace_id"}
    assert request.assistant_id is None


@pytest.mark.parametrize("raw", ["PRIVATE-RAW", "[]", {}, {"assistant_kind": "PRIVATE-KIND"}])
def test_corrupt_normalized_defaults_fail_closed_without_private_logs(db, monkeypatch, raw):
    """DB normalization must not turn malformed non-null storage into successful fallback."""
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        workspace = db._workspace_row_to_dict(dict(db.get_workspace("ws"), assistant_defaults_json=raw))
        monkeypatch.setattr(db, "get_workspace", lambda workspace_id: workspace)
        with pytest.raises(HTTPException) as error:
            _resolve(db, {"scope_type": "workspace", "workspace_id": "ws"})
    finally:
        logger.remove(sink)
    assert error.value.status_code == 409
    assert "PRIVATE" not in error.value.detail + "".join(messages)
    assert "workspace-persona" not in error.value.detail


def test_non_null_default_with_opt_out_fails_even_without_corruption_marker(db, monkeypatch):
    """The shared resolver checks inconsistent pairs instead of trusting a diagnostic bit."""
    workspace = dict(db.get_workspace("ws"), assistant_defaults_explicit_none=True, _assistant_defaults_invalid=False)
    monkeypatch.setattr(db, "get_workspace", lambda workspace_id: workspace)
    with pytest.raises(HTTPException) as error:
        _resolve(db, {"scope_type": "workspace", "workspace_id": "ws"})
    assert error.value.status_code == 409


@pytest.mark.parametrize("state", ["inactive", "deleted", "hidden", "disabled"])
def test_unavailable_default_has_bounded_startup_error(db, monkeypatch, state):
    """Configured unavailable Personas never fall back or disclose identity/name."""
    if state == "disabled":
        monkeypatch.setattr(feature_flags, "settings", {"PERSONA_ENABLED": False})

        def unexpected_lookup(*args, **kwargs):
            """Disabled defaults must be rejected before touching Persona storage."""
            pytest.fail("disabled feature queried a Persona")

        monkeypatch.setattr(db, "get_persona_profile", unexpected_lookup)
    elif state == "inactive":
        db.update_persona_profile(persona_id="workspace-persona", user_id="1", update_data={"is_active": False})
    elif state == "deleted":
        profile = db.get_persona_profile("workspace-persona", user_id="1")
        db.soft_delete_persona_profile(persona_id="workspace-persona", user_id="1", expected_version=profile["version"])
    else:
        workspace = db.get_workspace("ws")
        db.update_workspace("ws", {"assistant_defaults_json": dict(
            workspace["assistant_defaults_json"], assistant_id="PRIVATE-HIDDEN-PERSONA",
        )}, workspace["version"])
    with pytest.raises(HTTPException) as error:
        _resolve(db, {"scope_type": "workspace", "workspace_id": "ws"})
    assert error.value.status_code == (503 if state == "disabled" else 409)
    assert len(error.value.detail) < 256
    assert all(marker not in error.value.detail for marker in ("PRIVATE", "workspace-persona", "Workspace Persona"))


@pytest.mark.parametrize("field", ["assistant_kind", "assistant_id", "character_id"])
def test_explicit_none_bypasses_disabled_persona_storage(db, monkeypatch, field):
    """Each explicit null suppresses inheritance even when Persona storage is unavailable."""
    monkeypatch.setattr(feature_flags, "settings", {"PERSONA_ENABLED": False})

    def unavailable_storage(*args, **kwargs):
        """Prove the opt-out path never depends on Persona lookup availability."""
        raise CharactersRAGDBError("PRIVATE-STORAGE-ERROR")

    monkeypatch.setattr(db, "get_persona_profile", unavailable_storage)
    resolved = _resolve_startup(db, {"scope_type": "workspace", "workspace_id": "ws", field: None})
    assert resolved.startup.source == "explicit_none"
    assert resolved.request.assistant_id is None


@pytest.mark.parametrize("method", ["get_workspace", "get_persona_profile"])
def test_startup_propagates_database_failures_to_endpoint_mapper(db, monkeypatch, method):
    """Storage failures are errors, never a successful degraded selection."""
    def failed_read(*args, **kwargs):
        """Simulate a DB boundary failure without altering normalization."""
        raise CharactersRAGDBError("PRIVATE-STORAGE-ERROR")

    monkeypatch.setattr(db, method, failed_read)
    with pytest.raises(CharactersRAGDBError):
        _resolve_startup(db, {"scope_type": "workspace", "workspace_id": "ws"})


@pytest.mark.parametrize("conn", [None, object()])
def test_optional_connection_locks_workspace_then_selected_persona(db, monkeypatch, conn):
    """Only transaction calls pass the future locking kwargs; legacy reads still work."""
    get_workspace, get_profile = db.get_workspace, db.get_persona_profile
    calls = []

    def workspace_read(workspace_id, **kwargs):
        """Record lock order and delegate to today's real DB API."""
        calls.append(("workspace", kwargs))
        return get_workspace(workspace_id)

    def profile_read(persona_id, *, user_id, include_deleted=False, **kwargs):
        """Record transaction arguments without requiring Stage 3 DB implementation."""
        calls.append(("persona", kwargs))
        return get_profile(persona_id, user_id=user_id, include_deleted=include_deleted)

    monkeypatch.setattr(db, "get_workspace", workspace_read)
    monkeypatch.setattr(db, "get_persona_profile", profile_read)
    result = _resolve_startup(db, {"scope_type": "workspace", "workspace_id": "ws"}, conn=conn)
    assert result.request.assistant_id == "workspace-persona"
    expected_kwargs = {} if conn is None else {"conn": conn, "for_update": True}
    assert calls == [("workspace", expected_kwargs), ("persona", expected_kwargs)]


@pytest.mark.parametrize("workspace_id", ["PRIVATE-" + "x" * 1024, "PRIVATE-\ud800"])
def test_unrepresentable_origin_raises_bounded_input_error(db, monkeypatch, workspace_id):
    """Legacy Workspace access stays intact but a new origin cannot be truncated."""
    workspace = dict(db.get_workspace("ws"), id=workspace_id)
    monkeypatch.setattr(db, "get_workspace", lambda requested_id: workspace)
    with pytest.raises(InputError) as error:
        _resolve_startup(db, {"scope_type": "workspace", "workspace_id": workspace_id})
    assert "PRIVATE" not in str(error.value)
    assert len(str(error.value)) < 256


@pytest.mark.parametrize("parent_id", [None, "validated-parent"])
def test_explicit_persona_transaction_uses_selected_profile_and_request_memory(db, monkeypatch, parent_id):
    """Explicit/fork titles use the locked profile, never configured default availability."""
    get_workspace, get_profile = db.get_workspace, db.get_persona_profile
    connection = object()

    def workspace_read(workspace_id, *, conn, for_update):
        """Expose corrupt defaults to prove explicit selection bypasses them."""
        assert conn is connection and for_update
        return dict(get_workspace(workspace_id), _assistant_defaults_invalid=True)

    def profile_read(persona_id, *, user_id, include_deleted, conn, for_update):
        """Require a locked read for the explicit Persona's transaction-time label."""
        assert conn is connection and for_update
        return get_profile(persona_id, user_id=user_id, include_deleted=include_deleted)

    monkeypatch.setattr(db, "get_workspace", workspace_read)
    monkeypatch.setattr(db, "get_persona_profile", profile_read)
    result = _resolve_startup(db, {
        "scope_type": "workspace", "workspace_id": "ws", "assistant_kind": "persona",
        "assistant_id": "workspace-persona", "persona_memory_mode": "read_write",
        "parent_conversation_id": parent_id,
    }, conn=connection)
    assert result.startup.source == ("fork" if parent_id else "explicit")
    assert result.request.persona_memory_mode == "read_write"
    assert result.display_name == "Workspace Persona"


def test_locked_effective_resolution_does_not_reuse_unlocked_profile_cache(db, monkeypatch):
    """An earlier active cached profile cannot authorize a later locked inactive read."""
    module = importlib.import_module("tldw_Server_API.app.core.Workspaces.assistant_defaults")
    workspace = db.get_workspace("ws")
    cache = {}
    assert module.resolve_effective_workspace_assistant_default(
        db, workspace=workspace, user_id="1", persona_profile_cache=cache,
    ).status == "available"
    db.update_persona_profile(persona_id="workspace-persona", user_id="1", update_data={"is_active": False})
    get_profile = db.get_persona_profile

    def profile_read(persona_id, *, user_id, include_deleted, conn, for_update):
        """Delegate the future locked signature to the current owned-profile DB API."""
        return get_profile(persona_id, user_id=user_id, include_deleted=include_deleted)

    monkeypatch.setattr(db, "get_persona_profile", profile_read)
    effective = module.resolve_effective_workspace_assistant_default(
        db, workspace=workspace, user_id="1", persona_profile_cache=cache, conn=object(),
    )
    assert effective.status == "unavailable"
    assert effective.degraded_reason == "persona_unavailable"


def test_archived_workspace_preserves_legacy_inheritance(db):
    """Stage 1 does not import the deferred strict route's archived-Workspace rejection."""
    workspace = db.get_workspace("ws")
    db.update_workspace("ws", {"archived": True}, workspace["version"])
    result = _resolve_startup(db, {"scope_type": "workspace", "workspace_id": "ws"})
    assert result.startup.source == "workspace_default"
    assert result.startup.workspace_version == 3


@pytest.mark.parametrize("choice", [{}, {"assistant_kind": None}, {"character_id": 1}])
def test_deleted_workspace_rejects_all_choices(db, monkeypatch, choice):
    """Explicit choices still require a visible Workspace before bypassing defaults."""
    workspace = dict(db.get_workspace("ws"), deleted=True)
    monkeypatch.setattr(db, "get_workspace", lambda workspace_id: workspace)
    with pytest.raises(HTTPException) as error:
        _resolve_startup(db, {"scope_type": "workspace", "workspace_id": "ws", **choice})
    assert error.value.status_code == 404
