import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=321, username="reader", email=None, is_active=True)

    # Use full app profile for reading/Collections endpoints
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))

    from tldw_Server_API.app.main import app as fastapi_app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    with TestClient(fastapi_app) as client:
        yield client
    fastapi_app.dependency_overrides.clear()


def test_highlights_crud(client_with_user):

    client = client_with_user
    saved = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.org/highlight",
            "title": "Highlight parent",
            "content": "Important sentence",
        },
    )
    assert saved.status_code == 200, saved.text
    item_id = saved.json()["id"]

    # Create
    payload = {
        "item_id": item_id,
        "quote": "Important sentence",
        "start_offset": 10,
        "end_offset": 28,
        "color": "yellow",
        "note": "Check this",
        "anchor_strategy": "fuzzy_quote",
    }
    r = client.post(f"/api/v1/reading/items/{item_id}/highlight", json=payload)
    assert r.status_code == 200, r.text
    h = r.json()
    hid = h["id"]
    assert h["item_id"] == item_id
    assert h["quote"] == "Important sentence"

    # List for item
    r = client.get(f"/api/v1/reading/items/{item_id}/highlights")
    assert r.status_code == 200
    lst = r.json()
    assert any(x["id"] == hid for x in lst)

    # Update
    r = client.patch(f"/api/v1/reading/highlights/{hid}", json={"note": "Updated note", "state": "active"})
    assert r.status_code == 200
    upd = r.json()
    assert upd["note"] == "Updated note"

    # Delete
    r = client.delete(f"/api/v1/reading/highlights/{hid}")
    assert r.status_code == 200
    assert r.json().get("success") is True

    # Update non-existent -> 404
    r = client.patch(f"/api/v1/reading/highlights/{hid}", json={"note": "oops"})
    assert r.status_code == 404


def test_highlight_create_rejects_missing_parent(client_with_user):
    response = client_with_user.post(
        "/api/v1/reading/items/99999/highlight",
        json={"item_id": 99999, "quote": "No surviving capture"},
    )
    assert response.status_code == 404
