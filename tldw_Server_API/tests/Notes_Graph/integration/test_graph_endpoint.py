"""
Integration tests for the Notes Graph /graph and /neighbors endpoints.

Uses a real ChaChaNotes DB with dependency overrides, matching the pattern
in tests/Notes_NEW/integration/test_notes_graph_edges.py.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.notes_graph import _can_use_heavy_graph_limits
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
    monkeypatch.setenv("JWT_SECRET_KEY", "graph_endpoint_tests_secret_1234567890")
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    reset_settings()

    db_path = tmp_path / "graph_endpoint.db"
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

    # This fixture reloads the shared app module in full-app mode so the
    # notes graph routes are available. Restore the default lightweight test
    # profile before later tests reuse app_main.app via client_user_only.
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    reset_settings()
    importlib.reload(app_main)
    reset_settings()


def _headers():
    return {"Authorization": f"Bearer {_make_token('notes')}"}


def _graph_user(**overrides) -> User:
    data = {
        "id": 1,
        "username": "tester",
        "email": "t@e.com",
        "is_active": True,
        "roles": ["user"],
        "permissions": ["notes.graph.read"],
    }
    data.update(overrides)
    return User(**data)


def _create_note(client, title="N", content="body"):
    resp = client.post("/api/v1/notes/", json={"title": title, "content": content}, headers=_headers())
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_returns_manual_links(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "A", "aaa")
    n2 = _create_note(client, "B", "bbb")

    # Create manual link
    link_resp = client.post(
        f"/api/v1/notes/{n1}/links",
        json={"to_note_id": n2, "directed": False},
        headers=h,
    )
    assert link_resp.status_code == 200, link_resp.text

    resp = client.get(f"/api/v1/notes/graph?center_note_id={n1}&radius=1", headers=h)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    note_ids = {n["id"] for n in data["nodes"] if n["type"] == "note"}
    assert n1 in note_ids
    assert n2 in note_ids
    manual_edges = [e for e in data["edges"] if e["type"] == "manual"]
    assert len(manual_edges) >= 1


def test_graph_returns_wikilink_edges(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "A")
    n2 = _create_note(client, "B")
    # Use the canonical note writer so the derived graph projection advances
    # atomically with the note content.
    db.update_note(
        n1,
        {"content": f"See [[id:{n2}]] for details"},
        expected_version=1,
    )

    resp = client.get(
        f"/api/v1/notes/graph?center_note_id={n1}&radius=1&edge_types=wikilink",
        headers=h,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    wl = [e for e in data["edges"] if e["type"] == "wikilink"]
    assert len(wl) >= 1
    assert wl[0]["directed"] is True


def test_graph_returns_tag_edges(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "Tagged")
    kw_id = db.add_keyword("ml")
    db.link_note_to_keyword(n1, kw_id)

    resp = client.get(
        f"/api/v1/notes/graph?center_note_id={n1}&radius=1&edge_types=tag_membership",
        headers=h,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    tm = [e for e in data["edges"] if e["type"] == "tag_membership"]
    assert len(tm) >= 1
    tag_nodes = [n for n in data["nodes"] if n["type"] == "tag"]
    assert len(tag_nodes) >= 1
    assert tag_nodes[0]["label"] == "ml"


def test_neighbors_endpoint(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "Center", "center content")
    n2 = _create_note(client, "Neighbor", "neighbor content")
    client.post(
        f"/api/v1/notes/{n1}/links",
        json={"to_note_id": n2, "directed": False},
        headers=h,
    )

    resp = client.get(f"/api/v1/notes/{n1}/neighbors", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    note_ids = {n["id"] for n in data["nodes"] if n["type"] == "note"}
    assert n1 in note_ids
    assert n2 in note_ids


def test_cytoscape_format(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "Cy", "cy content")

    resp = client.get(f"/api/v1/notes/graph?center_note_id={n1}&format=cytoscape", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert "elements" in data
    assert "nodes" in data["elements"]
    assert "edges" in data["elements"]
    assert "truncated" in data
    assert "limits" in data
    assert data["active_note_count"] == 1
    assert data["all_notes_note_cap"] == min(100, data["limits"]["max_nodes"])
    assert data["all_notes_eligible"] is True


def test_graph_read_metadata_is_owner_scoped_and_effective_cap_is_clamped(
    client_and_db,
    monkeypatch,
):
    client, db = client_and_db
    h = _headers()
    note_ids = [_create_note(client, f"Owner note {index}") for index in range(3)]
    other_owner = CharactersRAGDB(f"{db.db_path_str}.owner-2", client_id="2")
    try:
        other_owner.add_note("Other owner private note", "private body")
        monkeypatch.setenv("NOTES_GRAPH_ALL_NOTES_NOTE_CAP", "100")
        response = client.get(
            "/api/v1/notes/graph",
            params={"center_note_id": note_ids[0], "max_nodes": 2},
            headers=h,
        )
    finally:
        other_owner.close_all_connections()

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["active_note_count"] == 3
    assert data["all_notes_note_cap"] == 2
    assert data["all_notes_eligible"] is False
    assert data["limits"]["max_nodes"] == 2


def test_seedless_small_collection(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "A")
    n2 = _create_note(client, "B")
    n3 = _create_note(client, "C")

    resp = client.get("/api/v1/notes/graph?radius=1", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    note_ids = {n["id"] for n in data["nodes"] if n["type"] == "note"}
    assert {n1, n2, n3} <= note_ids


def test_allow_heavy_requires_graph_admin_permission(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "Heavy", "heavy content")

    resp = client.get(
        "/api/v1/notes/graph",
        params={"center_note_id": n1, "allow_heavy": "true", "max_nodes": 1000},
        headers=h,
    )

    assert resp.status_code == 403
    assert "notes.graph.admin" in resp.json()["detail"]


@pytest.mark.parametrize(
    "user",
    [
        _graph_user(role="admin", roles=[], permissions=[]),
        _graph_user(is_admin=True, roles=[], permissions=[]),
        _graph_user(is_superuser=True, roles=[], permissions=[]),
        _graph_user(roles=["Admin"], permissions=[]),
        _graph_user(roles=[], permissions=["SYSTEM.CONFIGURE"]),
        _graph_user(roles=[], permissions=["notes.graph.admin"]),
        _graph_user(roles=[], permissions=["*"]),
    ],
)
def test_heavy_graph_permission_accepts_authnz_admin_signals(user):
    assert _can_use_heavy_graph_limits(user) is True


def test_edge_type_filter(client_and_db):
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "A")
    n2 = _create_note(client, "B")
    client.post(f"/api/v1/notes/{n1}/links", json={"to_note_id": n2}, headers=h)
    kw_id = db.add_keyword("filter_test")
    db.link_note_to_keyword(n1, kw_id)

    # Only request manual edges
    resp = client.get(
        "/api/v1/notes/graph",
        params={"center_note_id": n1, "edge_types": ["manual"]},
        headers=h,
    )
    assert resp.status_code == 200
    data = resp.json()
    types = {e["type"] for e in data["edges"]}
    if types:
        assert types <= {"manual"}, f"Unexpected edge types: {types}"


def test_empty_graph(client_and_db):
    client, db = client_and_db
    h = _headers()
    resp = client.get("/api/v1/notes/graph?radius=1", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["truncated"] is False


def test_cursor_pagination_returns_more(client_and_db):
    """When max_nodes truncates, response has has_more=True and a cursor string."""
    client, db = client_and_db
    h = _headers()
    center = _create_note(client, "Center", "center content")
    # Create enough neighbors to trigger truncation at max_nodes=3
    neighbors = []
    for i in range(5):
        nid = _create_note(client, f"N{i}", f"content {i}")
        neighbors.append(nid)
        client.post(f"/api/v1/notes/{center}/links", json={"to_note_id": nid}, headers=h)

    resp = client.get(
        f"/api/v1/notes/graph?center_note_id={center}&radius=1&max_nodes=3",
        headers=h,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is True
    assert data["has_more"] is True
    assert data["cursor"] is not None
    assert len(data["cursor"]) > 0


def test_invalid_edge_type_returns_400(client_and_db):
    """Invalid edge_types CSV value should return 400, not 500."""
    client, db = client_and_db
    h = _headers()
    n1 = _create_note(client, "A")
    resp = client.get(
        f"/api/v1/notes/graph?center_note_id={n1}&edge_types=bogus_type",
        headers=h,
    )
    assert resp.status_code == 400
    assert "Invalid edge_type" in resp.json()["detail"]
