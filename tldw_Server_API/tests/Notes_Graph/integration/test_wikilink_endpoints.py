"""API coverage for ``[[Title]]`` wikilinks (NE-02, #3110, decision D2).

- ``POST /api/v1/notes/wikilinks/resolve`` resolves link texts the way the
  graph projection does, so the WebUI preview and backlinks agree.
- ``GET /api/v1/notes/search?title_only=true`` powers ``[[`` autocomplete
  across the whole library.
- A ``[[Title]]`` link written through the API shows up as wikilink and
  backlink edges in ``/neighbors``.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def _make_token(scope: str) -> str:
    svc = JWTService(get_settings())
    return svc.create_virtual_access_token(
        user_id=1, username="tester", role="user", scope=scope, ttl_minutes=5,
    )


@pytest.fixture()
def client_and_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "wikilink_endpoint_tests_secret_1234567890")
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    reset_settings()

    db = CharactersRAGDB(str(tmp_path / "wikilink_endpoints.db"), client_id="1")

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            roles=["user"],
            permissions=["notes.graph.read", "notes.graph.write"],
        )

    from tldw_Server_API.app import main as app_main
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():
        return db

    importlib.reload(app_main)
    fastapi_app = app_main.app
    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client, db

    fastapi_app.dependency_overrides.clear()
    db.close_connection()

    # Restore the default lightweight test profile for later tests.
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    reset_settings()
    importlib.reload(app_main)
    reset_settings()


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_token('notes')}"}


def _create_note(client: TestClient, title: str, content: str = "body") -> str:
    resp = client.post("/api/v1/notes/", json={"title": title, "content": content}, headers=_headers())
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _set_created_at(db: CharactersRAGDB, note_id: str, created_at: str) -> None:
    db.execute_query("UPDATE notes SET created_at = ? WHERE id = ?", (created_at, note_id))


def test_resolve_endpoint_resolves_titles_like_the_projection(client_and_db) -> None:
    client, db = client_and_db
    older = _create_note(client, "Shared")
    newer = _create_note(client, "Shared")
    _set_created_at(db, older, "2026-01-01T00:00:00.000Z")
    _set_created_at(db, newer, "2026-02-01T00:00:00.000Z")
    target = _create_note(client, "Target Note")

    resp = client.post(
        "/api/v1/notes/wikilinks/resolve",
        json={"titles": ["target  NOTE", "Shared", "Missing note"]},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    by_title = {item["title"]: item for item in resp.json()["titles"]}
    assert by_title["target  NOTE"] == {
        "title": "target  NOTE",
        "note_id": target,
        "note_title": "Target Note",
        "candidate_count": 1,
    }
    assert by_title["Shared"]["note_id"] == older
    assert by_title["Shared"]["candidate_count"] == 2
    assert by_title["Missing note"]["note_id"] is None
    assert by_title["Missing note"]["candidate_count"] == 0


def test_resolve_endpoint_excludes_the_linking_note(client_and_db) -> None:
    client, _db = client_and_db
    source = _create_note(client, "Weekly sync")

    resp = client.post(
        "/api/v1/notes/wikilinks/resolve",
        json={"titles": ["Weekly sync"], "source_note_id": source},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["titles"][0]["note_id"] is None


def test_resolve_endpoint_reports_id_links(client_and_db) -> None:
    client, db = client_and_db
    live = _create_note(client, "Live note")
    trashed = _create_note(client, "Trashed note")
    db.soft_delete_note(trashed, expected_version=1)
    missing = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

    resp = client.post(
        "/api/v1/notes/wikilinks/resolve",
        json={"ids": [live.upper(), trashed, missing, "not-a-uuid"]},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    by_id = {item["id"]: item for item in resp.json()["ids"]}
    assert by_id[live.upper()] == {"id": live.upper(), "note_id": live, "note_title": "Live note"}
    assert by_id[trashed]["note_id"] is None
    assert by_id[missing]["note_id"] is None
    assert by_id["not-a-uuid"]["note_id"] is None


def test_resolve_endpoint_bounds_its_input(client_and_db) -> None:
    client, _db = client_and_db

    resp = client.post(
        "/api/v1/notes/wikilinks/resolve",
        json={"titles": [f"Title {index}" for index in range(201)]},
        headers=_headers(),
    )

    assert resp.status_code == 422


def test_title_only_search_finds_titles_across_the_whole_library(client_and_db) -> None:
    client, _db = client_and_db
    for index in range(25):
        _create_note(client, f"Filler {index:02d}", "nothing to see")
    content_only = _create_note(client, "Unrelated", "This body mentions Attention only in content.")
    infix = _create_note(client, "Notes on attention", "body")
    prefix = _create_note(client, "Attention is all you need", "body")

    resp = client.get(
        "/api/v1/notes/search/",
        params={"query": "atten", "title_only": "true", "limit": 10},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    ids = [note["id"] for note in payload["items"]]
    # Partial words match, prefix matches sort first, and content is ignored.
    assert ids == [prefix, infix]
    assert content_only not in ids
    assert payload["total"] == 2


def test_title_only_search_treats_like_wildcards_literally(client_and_db) -> None:
    client, _db = client_and_db
    literal = _create_note(client, "100% done", "body")
    _create_note(client, "1000 done", "body")

    resp = client.get(
        "/api/v1/notes/search/",
        params={"query": "100%", "title_only": "true"},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    assert [note["id"] for note in resp.json()["items"]] == [literal]


def test_title_wikilink_creates_wikilink_and_backlink_edges(client_and_db) -> None:
    client, _db = client_and_db
    target = _create_note(client, "Target Note", "the target")
    source = _create_note(client, "Source", "See [[Target Note]] for details.")

    resp = client.get(
        f"/api/v1/notes/{target}/neighbors",
        params={"edge_types": "wikilink,backlink"},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    edges = {(edge["type"], edge["source"], edge["target"]) for edge in resp.json()["edges"]}
    assert ("wikilink", source, target) in edges
    assert ("backlink", target, source) in edges


def test_creating_the_missing_note_adds_the_backlink(client_and_db) -> None:
    client, _db = client_and_db
    source = _create_note(client, "Source", "Plan: [[Future Note]]")

    target = _create_note(client, "Future Note", "created from the unresolved link")
    resp = client.get(
        f"/api/v1/notes/{target}/neighbors",
        params={"edge_types": "wikilink"},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    assert ("wikilink", source, target) in {
        (edge["type"], edge["source"], edge["target"]) for edge in resp.json()["edges"]
    }
