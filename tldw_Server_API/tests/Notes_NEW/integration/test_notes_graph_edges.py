"""
Integration tests for Notes Graph manual edge creation/deletion.

Uses a temporary ChaChaNotes DB via dependency override and JWT-based scope checks.
"""

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
    return svc.create_virtual_access_token(user_id=1, username="tester", role="user", scope=scope, ttl_minutes=5)


@pytest.fixture()
def client_with_graph_db(tmp_path, monkeypatch):
     # Configure multi-user mode with a virtual JWT secret
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "graph_edges_tests_secret_9876543210")
    # Use full app profile so Notes + Notes Graph routes are included
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    reset_settings()

    # Real temp ChaChaNotes DB
    db_path = tmp_path / "graph_edges.db"
    db = CharactersRAGDB(str(db_path), client_id="1")

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            roles=["user"],
            permissions=["notes.graph.read", "notes.graph.write"],
        )

    # Inject per-user DB via dependency override
    from tldw_Server_API.app import main as app_main
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():

        return db

    # Reload app after env tweaks so router gating sees MINIMAL_TEST_APP=0
    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()
    reset_settings()


def test_create_and_delete_manual_link(client_with_graph_db: TestClient):
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}

    # Create two notes to link
    n1 = client.post("/api/v1/notes/", json={"title": "N1", "content": "A"}, headers=headers)
    n2 = client.post("/api/v1/notes/", json={"title": "N2", "content": "B"}, headers=headers)
    assert n1.status_code == 201 and n2.status_code == 201
    id1 = n1.json()["id"]
    id2 = n2.json()["id"]

    # Create link
    resp = client.post(f"/api/v1/notes/{id1}/links", json={"to_note_id": id2, "directed": False}, headers=headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload.get("status") == "created"
    edge = payload.get("edge")
    assert isinstance(edge, dict) and (edge.get("id") or edge.get("edge_id"))
    # created_by principal set
    assert edge.get("created_by", "").startswith("user:")
    # Undirected canonicalization: from <= to (lexicographic)
    f = edge.get("from_note_id")
    t = edge.get("to_note_id")
    assert f <= t

    edge_id = edge.get("edge_id") or edge.get("id")
    del_resp = client.delete(f"/api/v1/notes/links/{edge_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json().get("deleted") is True


def test_delete_manual_link_with_typed_edge_id(client_with_graph_db: TestClient):
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}

    n1 = client.post("/api/v1/notes/", json={"title": "T1", "content": "A"}, headers=headers).json()["id"]
    n2 = client.post("/api/v1/notes/", json={"title": "T2", "content": "B"}, headers=headers).json()["id"]

    resp = client.post(f"/api/v1/notes/{n1}/links", json={"to_note_id": n2, "directed": False}, headers=headers)
    assert resp.status_code == 200, resp.text
    edge = resp.json().get("edge") or {}
    edge_id = edge.get("edge_id") or edge.get("id")
    assert edge_id

    del_resp = client.delete(f"/api/v1/notes/links/e:{edge_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json().get("deleted") is True


def test_create_link_with_typed_note_ids(client_with_graph_db: TestClient):
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}

    n1 = client.post("/api/v1/notes/", json={"title": "Typed A", "content": "A"}, headers=headers).json()
    n2 = client.post("/api/v1/notes/", json={"title": "Typed B", "content": "B"}, headers=headers).json()
    typed_from = f"note:{n1['id']}"
    typed_to = f"note:{n2['id']}"

    resp = client.post(
        f"/api/v1/notes/{typed_from}/links",
        json={"to_note_id": typed_to, "directed": False},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    edge = resp.json().get("edge") or {}
    # Typed IDs should be normalized to raw UUIDs; undirected edges are canonicalized.
    assert {edge.get("from_note_id"), edge.get("to_note_id")} == {n1["id"], n2["id"]}
    assert edge.get("from_note_id") <= edge.get("to_note_id")


def test_duplicate_undirected_conflict(client_with_graph_db: TestClient):
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}

    # Create two notes
    a = client.post("/api/v1/notes/", json={"title": "A", "content": "A"}, headers=headers).json()["id"]
    b = client.post("/api/v1/notes/", json={"title": "B", "content": "B"}, headers=headers).json()["id"]

    # First link succeeds
    ok = client.post(f"/api/v1/notes/{a}/links", json={"to_note_id": b, "directed": False}, headers=headers)
    assert ok.status_code == 200

    # Duplicate (same endpoints, undirected) should 409
    dup = client.post(f"/api/v1/notes/{b}/links", json={"to_note_id": a, "directed": False}, headers=headers)
    assert dup.status_code == 409
    assert dup.json()["detail"] == "duplicate manual link"


def test_directed_both_directions_allowed(client_with_graph_db: TestClient):
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}

    a = client.post("/api/v1/notes/", json={"title": "A2", "content": "A2"}, headers=headers).json()["id"]
    b = client.post("/api/v1/notes/", json={"title": "B2", "content": "B2"}, headers=headers).json()["id"]

    r1 = client.post(f"/api/v1/notes/{a}/links", json={"to_note_id": b, "directed": True}, headers=headers)
    r2 = client.post(f"/api/v1/notes/{b}/links", json={"to_note_id": a, "directed": True}, headers=headers)
    assert r1.status_code == 200 and r2.status_code == 200


def test_self_loop_rejected(client_with_graph_db: TestClient):
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}

    # Create one note
    n = client.post("/api/v1/notes/", json={"title": "Solo", "content": "One"}, headers=headers)
    assert n.status_code == 201
    nid = n.json()["id"]

    # Attempt to link note to itself should be rejected
    resp = client.post(f"/api/v1/notes/{nid}/links", json={"to_note_id": nid, "directed": False}, headers=headers)
    assert resp.status_code == 400
    assert "self" in resp.json().get("detail", "").lower()


def test_inactive_link_lifecycle_and_legacy_metadata_label(client_with_graph_db: TestClient):
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}
    source = client.post(
        "/api/v1/notes/",
        json={"title": "Lifecycle source", "content": "A"},
        headers=headers,
    ).json()["id"]
    target = client.post(
        "/api/v1/notes/",
        json={"title": "Lifecycle target", "content": "B"},
        headers=headers,
    ).json()["id"]

    created_response = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "metadata": {"label": "related", "origin": "legacy"},
        },
        headers=headers,
    )
    assert created_response.status_code == 200, created_response.text
    created = created_response.json()["edge"]
    edge_id = created["edge_id"]
    assert created["label"] == "related"
    assert created["properties"] == {"origin": "legacy"}

    detail = client.get(f"/api/v1/notes/links/{edge_id}", headers=headers)
    listing = client.get("/api/v1/notes/links", headers=headers)
    assert detail.status_code == 200 and detail.json()["version"] == 1
    assert [item["edge_id"] for item in listing.json()["links"]] == [edge_id]

    updated_response = client.patch(
        f"/api/v1/notes/links/{edge_id}",
        json={"weight": 2.0, "metadata": {"label": "strong", "rank": 2}},
        headers=headers,
    )
    assert updated_response.status_code == 200, updated_response.text
    assert updated_response.json()["version"] == 2
    assert updated_response.json()["label"] == "strong"

    deleted = client.delete(f"/api/v1/notes/links/{edge_id}", headers=headers)
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    tombstone = client.get(f"/api/v1/notes/links/{edge_id}", headers=headers).json()
    assert tombstone["deleted"] is True and tombstone["version"] == 3

    restored = client.post(
        f"/api/v1/notes/links/{edge_id}/restore",
        json={},
        headers=headers,
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["edge"]["deleted"] is False
    assert restored.json()["edge"]["version"] == 4


def test_link_listing_is_keyset_paginated_and_uses_bounded_summaries(
    client_with_graph_db: TestClient,
) -> None:
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}
    note_ids = [
        client.post(
            "/api/v1/notes/",
            json={"title": f"Page {index}", "content": "plain"},
            headers=headers,
        ).json()["id"]
        for index in range(3)
    ]
    for target in note_ids[1:]:
        response = client.post(
            f"/api/v1/notes/{note_ids[0]}/links",
            json={"to_note_id": target, "properties": {"private": "detail-only"}},
            headers=headers,
        )
        assert response.status_code == 200, response.text

    first = client.get("/api/v1/notes/links", params={"limit": 1}, headers=headers)
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert len(first_payload["links"]) == 1
    assert first_payload["has_more"] is True
    assert first_payload["next_cursor"]
    assert "properties" not in first_payload["links"][0]
    assert "metadata" not in first_payload["links"][0]

    second = client.get(
        "/api/v1/notes/links",
        params={"limit": 1, "cursor": first_payload["next_cursor"]},
        headers=headers,
    )
    assert second.status_code == 200, second.text
    assert second.json()["links"][0]["edge_id"] != first_payload["links"][0]["edge_id"]


def test_orphans_endpoint_ignores_tag_membership_and_is_keyset_paginated(
    client_with_graph_db: TestClient,
) -> None:
    client = client_with_graph_db
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}
    first = client.post(
        "/api/v1/notes/",
        json={"title": "Orphan A", "content": "plain"},
        headers=headers,
    ).json()["id"]
    second = client.post(
        "/api/v1/notes/",
        json={"title": "Orphan B", "content": "plain"},
        headers=headers,
    ).json()["id"]
    keyword_response = client.post(
        "/api/v1/notes/keywords/",
        json={"keyword": "orphan-tag"},
        headers=headers,
    )
    assert keyword_response.status_code == 201, keyword_response.text
    keyword_id = keyword_response.json()["id"]
    tag_response = client.post(
        f"/api/v1/notes/{first}/keywords/{keyword_id}",
        headers=headers,
    )
    assert tag_response.status_code == 200, tag_response.text

    page_one = client.get(
        "/api/v1/notes/graph/orphans",
        params={"limit": 1},
        headers=headers,
    )
    assert page_one.status_code == 200, page_one.text
    payload = page_one.json()
    assert len(payload["notes"]) == 1
    assert payload["has_more"] is True
    assert payload["next_cursor"]

    page_two = client.get(
        "/api/v1/notes/graph/orphans",
        params={"limit": 1, "cursor": payload["next_cursor"]},
        headers=headers,
    )
    assert page_two.status_code == 200, page_two.text
    returned = {payload["notes"][0]["id"], page_two.json()["notes"][0]["id"]}
    assert returned == {first, second}
