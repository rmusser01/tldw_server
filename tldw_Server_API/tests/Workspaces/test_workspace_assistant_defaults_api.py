"""API coverage for Workspace Assistant Defaults effective state."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)


@pytest.fixture(autouse=True)
def persona_feature_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep effective-default tests independent of deployment feature settings."""
    monkeypatch.setattr(feature_flags, "settings", {"PERSONA_ENABLED": True})


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="1")
    database.add_character_card(
        {
            "name": "Workspace Persona Source",
            "description": "Source character for workspace persona default tests",
            "personality": "Focused",
            "scenario": "Research",
            "system_prompt": "You support workspace research.",
            "first_message": "Ready.",
            "creator_notes": "Test fixture",
        }
    )
    return database


@pytest.fixture
def workspace_app() -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


def _allow_rate_limit() -> None:
    return None


def _install_workspace_overrides(
    app: FastAPI,
    db: CharactersRAGDB,
    *,
    user_id: int = 1,
    write: bool = False,
) -> None:
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=user_id)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    if write:
        app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit


def _clear_workspace_overrides(app: FastAPI) -> None:
    app.dependency_overrides.pop(get_request_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)
    app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
    app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _create_persona(
    db: CharactersRAGDB,
    *,
    persona_id: str = "persona-1",
    user_id: str = "1",
    name: str = "Literature Review Assistant",
) -> str:
    character = db.get_character_card_by_name("Workspace Persona Source")
    assert character is not None  # nosec B101
    return db.create_persona_profile(
        {
            "id": persona_id,
            "user_id": user_id,
            "name": name,
            "character_card_id": int(character["id"]),
            "mode": "session_scoped",
            "system_prompt": "You support literature review workflows.",
            "is_active": True,
        }
    )


def _assistant_defaults_payload(
    persona_id: str = "persona-1",
    memory_mode: str = "read_only",
) -> dict[str, Any]:
    return {
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "persona_memory_mode": memory_mode,
    }


@pytest.mark.integration
@pytest.mark.parametrize("initial_state", ["unset", "saved", "cleared"])
@pytest.mark.parametrize("operation", ["save", "clear", "omit_patch", "omit_put"])
def test_workspace_explicit_none_transitions_and_reads(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    initial_state: str,
    operation: str,
) -> None:
    """Writes derive durable choice and every management read preserves it."""
    workspace = db.upsert_workspace("ws-choice", "Assistant choice")
    persona_id = _create_persona(db)
    if initial_state != "unset":
        workspace = db.update_workspace(
            workspace["id"],
            {
                "assistant_defaults_json": (
                    _assistant_defaults_payload(persona_id) if initial_state == "saved" else None
                )
            },
            workspace["version"],
        )
    body = {"version": workspace["version"]}
    if operation == "save":
        body["assistant_defaults"] = _assistant_defaults_payload(persona_id)
    elif operation == "clear":
        body["assistant_defaults"] = None
    else:
        body["name"] = "Renamed choice"
    _install_workspace_overrides(workspace_app, db, write=True)
    try:
        with TestClient(workspace_app) as client:
            response = client.request(
                "PUT" if operation == "omit_put" else "PATCH",
                "/api/v1/workspaces/ws-choice",
                json={"name": "Renamed choice"} if operation == "omit_put" else body,
            )
            get_response = client.get("/api/v1/workspaces/ws-choice")
            list_response = client.get("/api/v1/workspaces/")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == get_response.status_code == list_response.status_code == 200
    expected_none = operation == "clear" or (operation.startswith("omit_") and initial_state == "cleared")
    expected_saved = operation == "save" or (operation.startswith("omit_") and initial_state == "saved")
    for payload in [response.json(), get_response.json(), *list_response.json()["items"]]:
        assert payload["assistant_defaults_explicit_none"] is expected_none
        assert payload["version"] == workspace["version"] + 1
        if expected_saved:
            assert payload["assistant_defaults"]["assistant_id"] == persona_id
            assert payload["effective_assistant_default"]["status"] == "available"
        else:
            assert payload["assistant_defaults"] is None
            assert payload["effective_assistant_default"]["status"] == "none"


@pytest.mark.integration
def test_upsert_workspace_creates_unset_explicit_none(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    """A newly created Workspace is unset, not a durable opt-out."""
    _install_workspace_overrides(workspace_app, db, write=True)
    try:
        with TestClient(workspace_app) as client:
            response = client.put("/api/v1/workspaces/ws-new", json={"name": "New choice"})
            get_response = client.get("/api/v1/workspaces/ws-new")
            list_response = client.get("/api/v1/workspaces/")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == get_response.status_code == list_response.status_code == 200
    for payload in [response.json(), get_response.json(), *list_response.json()["items"]]:
        assert payload["assistant_defaults_explicit_none"] is False
        assert payload["assistant_defaults"] is None
        assert payload["effective_assistant_default"]["status"] == "none"


@pytest.mark.integration
@pytest.mark.parametrize("value", [True, False, None])
@pytest.mark.parametrize("operation", ["create", "upsert", "patch"])
def test_workspace_rejects_readonly_explicit_none_without_mutation(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    value: bool | None,
    operation: str,
) -> None:
    """Supplying the known read-only field always fails before any write."""
    workspace = None
    body = {"name": "Must not be written", "assistant_defaults_explicit_none": value}
    if operation != "create":
        workspace = db.upsert_workspace("ws-readonly", "Original name")
        if operation == "patch":
            body["version"] = workspace["version"]
            body["assistant_defaults"] = None
    _install_workspace_overrides(workspace_app, db, write=True)
    try:
        with TestClient(workspace_app) as client:
            response = client.request(
                "PATCH" if operation == "patch" else "PUT",
                "/api/v1/workspaces/ws-readonly",
                json=body,
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 422, response.text
    assert "assistant_defaults_explicit_none" in response.text
    assert db.get_workspace("ws-readonly") == workspace


@pytest.mark.integration
@pytest.mark.parametrize("method", ["PUT", "PATCH"])
def test_workspace_explicit_none_validation_preserves_unrelated_extra_compatibility(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    method: str,
) -> None:
    """Rejecting the derived choice does not globally forbid legacy extras."""
    workspace = db.upsert_workspace("ws-extra", "Original name")
    body = {"name": "Renamed", "legacy_extra": "ignored"}
    if method == "PATCH":
        body["version"] = workspace["version"]
    _install_workspace_overrides(workspace_app, db, write=True)
    try:
        with TestClient(workspace_app) as client:
            response = client.request(method, "/api/v1/workspaces/ws-extra", json=body)
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200, response.text
    assert response.json()["name"] == "Renamed"
    assert "legacy_extra" not in response.json()


@pytest.mark.integration
def test_workspace_inconsistent_explicit_none_is_unavailable_on_reads(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normalized non-null default plus opt-out fails closed on detail/list."""
    workspace = db.upsert_workspace("ws-inconsistent", "Inconsistent choice")
    persona_id = _create_persona(db)
    workspace.update(
        assistant_defaults_json=_assistant_defaults_payload(persona_id),
        assistant_defaults_explicit_none=True,
        _assistant_defaults_invalid=True,
    )
    monkeypatch.setattr(db, "get_workspace", Mock(return_value=workspace))
    monkeypatch.setattr(db, "list_workspaces", Mock(return_value=[workspace]))
    lookup = Mock(side_effect=AssertionError("inconsistent defaults must not resolve a Persona"))
    monkeypatch.setattr(db, "get_persona_profile", lookup)
    _install_workspace_overrides(workspace_app, db)
    try:
        with TestClient(workspace_app) as client:
            get_response = client.get("/api/v1/workspaces/ws-inconsistent")
            list_response = client.get("/api/v1/workspaces/")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert get_response.status_code == list_response.status_code == 200
    for payload in [get_response.json(), *list_response.json()["items"]]:
        assert payload["assistant_defaults_explicit_none"] is True
        assert payload["effective_assistant_default"] == {
            "status": "unavailable",
            "source": "workspace",
            "assistant_kind": None,
            "assistant_id": None,
            "label": None,
            "persona_memory_mode": None,
            "degraded_reason": "invalid_default",
        }
        assert "_assistant_defaults_invalid" not in payload
    lookup.assert_not_called()


@pytest.mark.integration
def test_patch_workspace_returns_effective_default_for_existing_persona(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    persona_id = _create_persona(db)
    _install_workspace_overrides(workspace_app, db, write=True)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": _assistant_defaults_payload(persona_id),
                },
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["assistant_defaults"] == {
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "persona_memory_mode": "read_only",
        "voice": None,
        "style": None,
        "tool_policy_profile_id": None,
    }
    assert payload["effective_assistant_default"] == {
        "status": "available",
        "source": "workspace",
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "label": "Literature Review Assistant",
        "persona_memory_mode": "read_only",
        "degraded_reason": None,
    }


@pytest.mark.integration
def test_get_and_list_workspaces_include_effective_default(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    persona_id = _create_persona(db)
    db.update_workspace(
        "ws-assistant",
        {"assistant_defaults_json": _assistant_defaults_payload(persona_id)},
        expected_version=int(workspace["version"]),
    )
    _install_workspace_overrides(workspace_app, db)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            get_response = client.get("/api/v1/workspaces/ws-assistant")
            list_response = client.get("/api/v1/workspaces/")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert get_response.status_code == 200, get_response.text
    assert list_response.status_code == 200, list_response.text
    assert get_response.json()["effective_assistant_default"]["status"] == "available"
    [listed] = list_response.json()["items"]
    assert listed["id"] == "ws-assistant"
    assert listed["effective_assistant_default"]["assistant_id"] == persona_id
    assert listed["effective_assistant_default"]["label"] == "Literature Review Assistant"


@pytest.mark.integration
def test_list_workspaces_caches_repeated_persona_default_lookups(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_id = _create_persona(db)
    first_workspace = db.upsert_workspace("ws-assistant-a", "Assistant A")
    second_workspace = db.upsert_workspace("ws-assistant-b", "Assistant B")
    for workspace in (first_workspace, second_workspace):
        db.update_workspace(
            workspace["id"],
            {"assistant_defaults_json": _assistant_defaults_payload(persona_id)},
            expected_version=int(workspace["version"]),
        )

    original_get_persona_profile = db.get_persona_profile
    profile_lookup_count = 0

    def _counting_get_persona_profile(*args: Any, **kwargs: Any) -> Any:
        nonlocal profile_lookup_count
        profile_lookup_count += 1
        return original_get_persona_profile(*args, **kwargs)

    monkeypatch.setattr(db, "get_persona_profile", _counting_get_persona_profile)
    _install_workspace_overrides(workspace_app, db)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert {item["id"] for item in payload["items"]} == {
        "ws-assistant-a",
        "ws-assistant-b",
    }
    assert profile_lookup_count == 1


@pytest.mark.integration
@pytest.mark.parametrize("fail_deleted_lookup", [False, True])
def test_get_workspace_maps_persona_lookup_database_errors(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    fail_deleted_lookup: bool,
) -> None:
    """Both live and deleted-profile lookup failures remain mapped service errors."""
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    db.update_workspace(
        "ws-assistant",
        {"assistant_defaults_json": _assistant_defaults_payload("persona-1")},
        expected_version=int(workspace["version"]),
    )

    def _raise_get_persona_profile(*args: Any, **kwargs: Any) -> Any:
        """Fail at the selected lookup boundary without returning an unset default."""
        if fail_deleted_lookup and not kwargs.get("include_deleted"):
            return None
        raise CharactersRAGDBError("lookup failed")

    monkeypatch.setattr(db, "get_persona_profile", _raise_get_persona_profile)
    _install_workspace_overrides(workspace_app, db)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-assistant")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to resolve workspace assistant default"}


@pytest.mark.integration
def test_patch_workspace_maps_persona_lookup_input_errors(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")

    def _raise_get_persona_profile(*args: Any, **kwargs: Any) -> Any:
        raise InputError("invalid persona lookup")

    monkeypatch.setattr(db, "get_persona_profile", _raise_get_persona_profile)
    _install_workspace_overrides(workspace_app, db, write=True)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": _assistant_defaults_payload("persona-1"),
                },
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "invalid persona lookup"}
    assert db.get_workspace("ws-assistant")["assistant_defaults_json"] is None


@pytest.mark.integration
def test_patch_workspace_rejects_missing_persona_default(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    _install_workspace_overrides(workspace_app, db, write=True)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": _assistant_defaults_payload("missing-persona"),
                },
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 422, response.text
    assert "assistant_defaults.assistant_id" in response.text
    assert db.get_workspace("ws-assistant")["assistant_defaults_json"] is None


@pytest.mark.integration
def test_effective_default_redacts_deleted_persona_drift(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    persona_id = _create_persona(db)
    db.update_workspace(
        "ws-assistant",
        {"assistant_defaults_json": _assistant_defaults_payload(persona_id)},
        expected_version=int(workspace["version"]),
    )
    profile = db.get_persona_profile(persona_id, user_id="1")
    assert profile is not None  # nosec B101
    assert db.soft_delete_persona_profile(
        persona_id=persona_id,
        user_id="1",
        expected_version=int(profile["version"]),
    )
    _install_workspace_overrides(workspace_app, db)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-assistant")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["assistant_defaults"]["assistant_id"] == persona_id
    assert payload["effective_assistant_default"] == {
        "status": "unavailable",
        "source": "workspace",
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "label": None,
        "persona_memory_mode": "read_only",
        "degraded_reason": "persona_deleted",
    }


@pytest.mark.integration
@pytest.mark.parametrize("hidden_owner", [None, "another-user"])
def test_effective_default_redacts_permission_denied(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    hidden_owner: str | None,
) -> None:
    """Missing and other-owner Personas reveal no identity in the effective view."""
    workspace = db.upsert_workspace("ws-private", "Private default")
    if hidden_owner is not None:
        _create_persona(db, persona_id="hidden-persona", user_id=hidden_owner)
    db.update_workspace(
        workspace["id"],
        {"assistant_defaults_json": _assistant_defaults_payload("hidden-persona")},
        workspace["version"],
    )
    _install_workspace_overrides(workspace_app, db)
    try:
        with TestClient(workspace_app) as client:
            get_response = client.get("/api/v1/workspaces/ws-private")
            list_response = client.get("/api/v1/workspaces/")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert get_response.status_code == list_response.status_code == 200
    for payload in [get_response.json(), *list_response.json()["items"]]:
        assert payload["assistant_defaults"]["assistant_id"] == "hidden-persona"
        assert payload["effective_assistant_default"] == {
            "status": "unavailable",
            "source": "workspace",
            "assistant_kind": None,
            "assistant_id": None,
            "label": None,
            "persona_memory_mode": None,
            "degraded_reason": "permission_denied",
        }


@pytest.mark.parametrize("mixed_keys", [False, True])
def test_invalid_default_logs_omit_payload_values_and_keys(mixed_keys: bool) -> None:
    """Validation logging never renders private input or sorts heterogeneous keys."""
    private_marker = "PRIVATE-DEFAULT-CONTENT"
    raw = {"assistant_kind": private_marker, private_marker: "private-value"}
    if mixed_keys:
        raw[42] = "private-numeric-key"
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        parsed, invalid = workspaces_endpoint._parse_workspace_assistant_defaults(
            raw,
            workspace_id="PRIVATE-WORKSPACE\n" + "x" * 1000,
        )
    finally:
        logger.remove(sink_id)

    assert parsed is None and invalid
    assert len(messages) == 1
    assert "invalid stored workspace assistant defaults" in messages[0]
    assert private_marker not in messages[0]
    assert "private-value" not in messages[0]
    assert "private-numeric-key" not in messages[0]
    assert "PRIVATE-WORKSPACE" not in messages[0]
    assert len(messages[0]) < 256


@pytest.mark.integration
@pytest.mark.parametrize("defaults", [None, {}, {"assistant_kind": "invalid"}])
def test_effective_default_distinguishes_unset_and_invalid_objects(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    defaults: dict | None,
) -> None:
    """Unset defaults stay none while invalid objects degrade without a reference."""
    workspace = db.upsert_workspace("ws-default-state", "Default state")
    db.update_workspace(
        workspace["id"],
        {"assistant_defaults_json": defaults},
        workspace["version"],
    )
    _install_workspace_overrides(workspace_app, db)
    try:
        with TestClient(workspace_app) as client:
            response = client.get("/api/v1/workspaces/ws-default-state")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200
    payload = response.json()
    assert payload["assistant_defaults"] is None
    assert payload["effective_assistant_default"] == {
        "status": "none" if defaults is None else "unavailable",
        "source": "none" if defaults is None else "workspace",
        "assistant_kind": None,
        "assistant_id": None,
        "label": None,
        "persona_memory_mode": None,
        "degraded_reason": None if defaults is None else "invalid_default",
    }


def test_effective_default_preserves_db_corruption_diagnostic(db: CharactersRAGDB) -> None:
    """A normalized corrupt row must not be projected as an unset default."""
    workspace = db.upsert_workspace("ws-corrupt", "Corrupt defaults")
    workspace["_assistant_defaults_invalid"] = True
    lookup = Mock(side_effect=AssertionError("corrupt defaults must not resolve a Persona"))
    db.get_persona_profile = lookup
    payload = workspaces_endpoint._ws_to_response(
        workspace,
        db=db,
        current_user=SimpleNamespace(id=1),
    ).model_dump()

    assert payload["effective_assistant_default"]["degraded_reason"] == "invalid_default"
    assert payload["assistant_defaults"] is None
    assert "_assistant_defaults_invalid" not in payload
    lookup.assert_not_called()


@pytest.mark.integration
def test_effective_default_marks_inactive_persona_unavailable(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    """An inactive owned Persona retains its reference but cannot be effective."""
    workspace = db.upsert_workspace("ws-inactive", "Inactive default")
    persona_id = _create_persona(db)
    db.update_workspace(
        workspace["id"],
        {"assistant_defaults_json": _assistant_defaults_payload(persona_id)},
        workspace["version"],
    )
    db.update_persona_profile(persona_id=persona_id, user_id="1", update_data={"is_active": False})
    _install_workspace_overrides(workspace_app, db)
    try:
        with TestClient(workspace_app) as client:
            response = client.get("/api/v1/workspaces/ws-inactive")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200
    assert response.json()["effective_assistant_default"] == {
        "status": "unavailable",
        "source": "workspace",
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "label": None,
        "persona_memory_mode": "read_only",
        "degraded_reason": "persona_unavailable",
    }


@pytest.mark.integration
def test_disabled_persona_feature_skips_lookup_and_allows_clearing(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling Persona hides effective identity but leaves settings clearable."""
    workspace = db.upsert_workspace("ws-disabled", "Disabled default")
    workspace = db.update_workspace(
        workspace["id"],
        {"assistant_defaults_json": _assistant_defaults_payload()},
        workspace["version"],
    )
    monkeypatch.setattr(feature_flags, "settings", {"PERSONA_ENABLED": False})
    lookup = Mock(side_effect=AssertionError("disabled Persona feature must not query profiles"))
    monkeypatch.setattr(db, "get_persona_profile", lookup)
    _install_workspace_overrides(workspace_app, db, write=True)
    try:
        with TestClient(workspace_app) as client:
            get_response = client.get("/api/v1/workspaces/ws-disabled")
            list_response = client.get("/api/v1/workspaces/")
            clear_response = client.patch(
                "/api/v1/workspaces/ws-disabled",
                json={"version": workspace["version"], "assistant_defaults": None},
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert get_response.status_code == list_response.status_code == clear_response.status_code == 200
    for payload in [get_response.json(), *list_response.json()["items"]]:
        assert payload["assistant_defaults"]["assistant_id"] == "persona-1"
        assert payload["effective_assistant_default"] == {
            "status": "unavailable",
            "source": "workspace",
            "assistant_kind": None,
            "assistant_id": None,
            "label": None,
            "persona_memory_mode": None,
            "degraded_reason": "persona_feature_disabled",
        }
    assert clear_response.json()["effective_assistant_default"]["status"] == "none"
    lookup.assert_not_called()


@pytest.mark.integration
def test_disabled_persona_feature_rejects_new_default_without_lookup(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled Persona writes fail before lookup or Workspace mutation."""
    workspace = db.upsert_workspace("ws-disabled", "Disabled default")
    monkeypatch.setattr(feature_flags, "settings", {"PERSONA_ENABLED": False})
    lookup = Mock(side_effect=AssertionError("disabled Persona feature must not query profiles"))
    monkeypatch.setattr(db, "get_persona_profile", lookup)
    _install_workspace_overrides(workspace_app, db, write=True)
    try:
        with TestClient(workspace_app) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-disabled",
                json={"version": workspace["version"], "assistant_defaults": _assistant_defaults_payload()},
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 503
    assert db.get_workspace(workspace["id"])["version"] == workspace["version"]
    lookup.assert_not_called()
