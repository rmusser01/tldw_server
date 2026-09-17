"""Actual suggestion read routes on a fresh inactive-Sync Notes database."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import notes_graph_suggestions as endpoint
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph import suggestion_api, suggestion_provider
from tldw_Server_API.app.core.Sync.v2 import notes_link_coordinator, server_origin

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def fresh_suggestions(request, tmp_path, monkeypatch):
    # Other transport fixtures install a FakeAPI directly on the route module.
    # This fixture always exercises the real factory, even in a mixed suite.
    monkeypatch.setattr(endpoint, "build_notes_graph_suggestions_api", suggestion_api.build_notes_graph_suggestions_api)
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    db = CharactersRAGDB(tmp_path / "suggestions.db", client_id="1", backend=backend)
    monkeypatch.setattr(server_origin, "get_active_server_origin_sync_service_for_user", lambda _owner: None)
    monkeypatch.setattr(notes_link_coordinator, "get_active_server_origin_sync_service_for_user", lambda _owner: None)
    monkeypatch.setattr(suggestion_provider, "get_default_provider", lambda: None)
    monkeypatch.setattr(suggestion_provider, "get_default_model_for_provider", lambda _provider: None)
    monkeypatch.delenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", raising=False)
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/notes")
    app.dependency_overrides[endpoint.get_request_user] = lambda: SimpleNamespace(id=1, id_str="1")
    app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[endpoint.try_get_job_manager] = lambda: None
    app.dependency_overrides[endpoint._require_suggestion_permissions] = lambda: AuthPrincipal(
        kind="user", user_id=1, roles=["admin"], permissions=[], is_admin=True
    )
    for route in app.routes:
        for dep in getattr(getattr(route, "dependant", None), "dependencies", []):
            if getattr(dep.call, "_tldw_token_scope", False) or getattr(dep.call, "_tldw_rate_limit_resource", None):
                app.dependency_overrides[dep.call] = lambda: None
    try:
        with chacha_operation(independent=True):
            note = db.add_note("Synthetic suggestion source", "A short owned fixture note.")
        with TestClient(app, raise_server_exceptions=False) as client:
            yield db, client, note, request.param
    finally:
        app.dependency_overrides.clear()
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("registered", [False, True], ids=["fresh", "registered-scope-control"])
@pytest.mark.parametrize("route", ["capabilities", "list"])
def test_fresh_suggestion_reads_report_expected_unavailability_instead_of_storage_error(
    fresh_suggestions, monkeypatch, registered, route
):
    db, client, note, _kind = fresh_suggestions
    dataset = endpoint._dataset_key(owner_user_id="1", dataset_id=None)
    assert dataset == "legacy:1"
    if registered:
        # Established storage fixtures explicitly insert this authority. This
        # control isolates the factory/store integration gap without schema edits.
        with chacha_operation(independent=True), db.transaction() as conn:
            conn.execute("INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)", ("1", dataset))
    causes = []
    original = suggestion_api.NotesGraphSuggestionsAPI._translate

    def observe(exc):
        causes.append((type(exc).__name__, str(exc)))
        return original(exc)

    monkeypatch.setattr(suggestion_api.NotesGraphSuggestionsAPI, "_translate", staticmethod(observe))
    suffix = "/capabilities" if route == "capabilities" else "?state=pending%2Caccepting&limit=100"
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions{suffix}")
    assert response.status_code == 200, {"response": response.json(), "causes": causes}
    body = response.json()
    if route == "capabilities":
        assert body["generation_available"] is False
        assert body["unavailable_reason"] == "notes_graph_suggestions_worker_unavailable"
    else:
        assert body["items"] == [] and body["next_cursor"] is None
    with chacha_operation(independent=True):
        assert db.get_note_by_id(note)["version"] == 1
