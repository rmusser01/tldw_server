"""Real database/router controls for optional visual metadata support (UAT208)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import visual_identities as api
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository

PREFIX = "/api/v1/visual-identities"
UNSUPPORTED = "visual_identity_metadata_backend_unsupported"


@dataclass
class Route:
    """Actual router and selected database; only external request context is replaced."""

    app: FastAPI
    client: TestClient
    db: CharactersRAGDB
    backend_name: str
    character_id: int
    persona_id: str

    def resolve(self, **params):
        """Issue an ordinary optional lookup unless a test supplies override fields."""
        return self.client.get(
            f"{PREFIX}/bindings/resolve",
            params={"actor_kind": "character", "actor_id": self.character_id, **params},
        )


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def route(request, tmp_path: Path) -> Iterator[Route]:
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    db = CharactersRAGDB(tmp_path / "visual.db", client_id="2", backend=backend)
    character = db.add_character_card({"name": "Owned optional visual actor"})
    assert character is not None
    persona = db.create_persona_profile({"name": "Owned persona", "user_id": "2"})
    app = FastAPI()
    app.include_router(api.router, prefix=PREFIX)
    app.dependency_overrides[api.get_request_user] = lambda: User(
        id=2, username="visual-fixture-owner", roles=["user"], permissions=[]
    )
    app.dependency_overrides[api.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[api._job_manager] = lambda: object()
    app.dependency_overrides[api._READ_LIMIT.dependency] = lambda: None
    app.dependency_overrides[api._WRITE_LIMIT.dependency] = lambda: None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield Route(app, client, db, request.param, character, persona)
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("actor_kind", ["character", "persona"])
@pytest.mark.parametrize("expression,normalized", [("neutral", "neutral"), ("thoughtful", "thinking")])
def test_owned_optional_resolution_returns_explicit_no_asset(route, actor_kind, expression, normalized):
    actor_id = route.character_id if actor_kind == "character" else route.persona_id
    response = route.resolve(
        actor_kind=actor_kind,
        actor_id=actor_id,
        expression_key=expression,
        role_id="narrator",
        role_label="Narrator",
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["actor_id"] == actor_id
    assert payload["requested_expression_key"] == normalized
    assert payload["role_id"] == "narrator"
    assert payload["role_label"] == "Narrator"
    assert payload["fallback_reason"] == (
        "metadata_backend_unsupported" if route.backend_name == "postgres" else "placeholder"
    )
    assert payload["resolution_source"] == "placeholder"
    assert all(
        payload[key] is None
        for key in (
            "pack_id",
            "pack_version_id",
            "asset_id",
            "asset_url",
            "storage_relpath",
            "preview_url",
        )
    )


def test_capabilities_report_selected_backend_without_changing_formats(route):
    response = route.client.get(f"{PREFIX}/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata_supported"] is (route.backend_name == "sqlite")
    assert payload["metadata_unavailable_reason"] == (UNSUPPORTED if route.backend_name == "postgres" else None)
    assert {"image/gif", "image/jpeg", "image/png", "image/webp"} <= set(payload["supported_mime_types"])
    assert payload["upload_max_bytes"] > 0
    assert payload["archive_max_bytes"] > 0


@pytest.mark.parametrize("actor_kind", ["character", "persona"])
def test_missing_actor_is_not_hidden_by_optional_fallback(route, actor_kind):
    response = route.resolve(actor_kind=actor_kind, actor_id="999999")
    assert response.status_code == 404
    assert response.json()["detail"] == f"visual_identity_{actor_kind}_not_found"


@pytest.mark.parametrize("actor_kind", ["character", "persona"])
def test_deleted_actor_is_not_hidden_by_optional_fallback(route, actor_kind):
    if actor_kind == "character":
        actor_id = route.character_id
        assert route.db.soft_delete_character_card(actor_id, expected_version=1)
    else:
        actor_id = route.persona_id
        assert route.db.soft_delete_persona_profile(persona_id=actor_id, user_id="2", expected_version=1)
    response = route.resolve(actor_kind=actor_kind, actor_id=actor_id)
    assert response.status_code == 404
    assert response.json()["detail"] == f"visual_identity_{actor_kind}_not_found"


def test_foreign_persona_is_not_hidden_by_optional_fallback(route):
    actor_id = route.db.create_persona_profile({"name": "Foreign persona", "user_id": "3"})
    response = route.resolve(actor_kind="persona", actor_id=actor_id)
    assert response.status_code == 404
    assert response.json()["detail"] == "visual_identity_persona_not_found"


def test_character_owner_filter_preserves_backend_contract(route, tmp_path):
    other = CharactersRAGDB(
        route.db.db_path if route.backend_name == "sqlite" else tmp_path / "other.db",
        client_id="3",
        backend=route.db.backend if route.backend_name == "postgres" else None,
    )
    try:
        actor_id = other.add_character_card({"name": "Other client actor"})
    finally:
        # The route fixture owns the shared backend pool; release only this checkout.
        other.close_connection()
    response = route.resolve(actor_id=actor_id)
    # SQLite client IDs are sync devices within one user's file; PG shares owner-scoped rows.
    assert response.status_code == (404 if route.backend_name == "postgres" else 200)


@pytest.mark.parametrize(
    "params",
    [
        {"expression_key": "!!!"},
        {"actor_kind": "invalid"},
        {"override_pack_id": 0},
    ],
)
def test_invalid_query_remains_validation_failure(route, params):
    assert route.resolve(**params).status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"override_pack_id": 999999},
        {"override_pack_version_id": 999999, "allow_override_fallback": "true"},
    ],
)
def test_explicit_override_never_becomes_ordinary_placeholder(route, params):
    response = route.resolve(**params)
    assert response.status_code == (501 if route.backend_name == "postgres" else 404)
    assert response.json()["detail"] == (UNSUPPORTED if route.backend_name == "postgres" else "pack_not_found")


@pytest.mark.parametrize("method", ["get", "post"])
def test_metadata_authoring_stays_explicitly_unsupported(route, method):
    response = getattr(route.client, method)(
        f"{PREFIX}/packs", **({"json": {"title": "Expressions"}} if method == "post" else {})
    )
    if route.backend_name == "postgres":
        assert response.status_code == 501
        assert response.json()["detail"] == UNSUPPORTED
    else:
        assert response.status_code == (201 if method == "post" else 200)
        if method == "post":
            assert response.json()["title"] == "Expressions"


@pytest.mark.parametrize("dependency,code", [(api.get_request_user, 401), (api._READ_LIMIT.dependency, 429)])
def test_auth_and_rate_failures_are_not_optional_success(route, dependency, code):
    def deny():
        raise HTTPException(status_code=code, detail="request_denied")

    route.app.dependency_overrides[dependency] = deny
    response = route.resolve()
    assert response.status_code == code
    assert response.json()["detail"] == "request_denied"


def test_real_lookup_database_failure_is_not_a_no_asset_result(route, monkeypatch):
    failures = []

    def broken_lookup(_actor_id):
        try:
            return route.db.execute_query(
                "SELECT * FROM visual_identity_missing_fixture_relation",
                read_only=True,
            ).fetchone()
        except Exception as exc:
            failures.append(type(exc).__name__)
            raise

    monkeypatch.setattr(route.db, "get_character_card_by_id", broken_lookup)
    response = route.resolve()
    assert response.status_code == 500
    assert len(failures) == 1  # The real lookup failed; an early repository failure cannot satisfy this.


def test_direct_repository_support_guard_is_preserved(route):
    if route.backend_name == "postgres":
        with pytest.raises(NotImplementedError, match="SQLite ChaChaNotes"):
            VisualIdentityRepository(route.db)
    else:
        assert isinstance(VisualIdentityRepository(route.db), VisualIdentityRepository)
