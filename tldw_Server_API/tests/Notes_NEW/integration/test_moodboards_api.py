"""
Integration tests for Notes moodboard API endpoints using a lightweight app.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.integration


class _AllowRateLimiter:
    async def check_user_rate_limit(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        return True, {}


@pytest.fixture()
def moodboard_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "moodboards_api.db"
    db = CharactersRAGDB(str(db_path), client_id="integration_user")

    app = FastAPI()
    app.include_router(notes_endpoints.router, prefix="/api/v1/notes")

    async def override_user():
        return User(id=1, username="tester", email="tester@example.com", is_active=True, is_admin=True)

    def override_db_dep():
        return db

    async def override_rate_limiter_dep():
        return _AllowRateLimiter()

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_db_dep
    app.dependency_overrides[get_rate_limiter_dep] = override_rate_limiter_dep

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    try:
        db.close()
    except Exception:
        _ = None


def test_moodboard_crud_and_membership_flow(moodboard_client: TestClient):
    client = moodboard_client

    note_manual = client.post("/api/v1/notes/", json={"title": "Manual mood", "content": "A"}).json()
    note_both = client.post("/api/v1/notes/", json={"title": "Palette mood", "content": "B"}).json()

    keyword = client.post("/api/v1/notes/keywords/", json={"keyword": "palette"}).json()
    kw_id = keyword["id"]
    assert client.post(f"/api/v1/notes/{note_both['id']}/keywords/{kw_id}").status_code == 200

    create_resp = client.post(
        "/api/v1/notes/moodboards",
        json={
            "name": "Inspiration",
            "description": "Visual ideas",
            "smart_rule": {"keyword_tokens": ["palette"]},
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    moodboard = create_resp.json()
    moodboard_id = moodboard["id"]

    get_resp = client.get(f"/api/v1/notes/moodboards/{moodboard_id}")
    assert get_resp.status_code == 200
    fetched = get_resp.json()
    assert fetched["name"] == "Inspiration"
    assert fetched["smart_rule"]["keyword_tokens"] == ["palette"]

    list_resp = client.get("/api/v1/notes/moodboards")
    assert list_resp.status_code == 200
    list_payload = list_resp.json()
    boards = list_payload["moodboards"]
    assert any(int(item["id"]) == int(moodboard_id) for item in boards)
    assert list_payload["count"] == 1
    assert list_payload["total"] == 1
    assert list_payload["limit"] == 100
    assert list_payload["offset"] == 0
    assert list_payload["pagination"] == {
        "mode": "offset",
        "limit": 100,
        "offset": 0,
        "total": 1,
        "has_more": False,
        "next_offset": None,
    }
    assert list_payload["has_more"] is False
    assert list_payload["next_offset"] is None

    pin_manual = client.post(f"/api/v1/notes/moodboards/{moodboard_id}/notes/{note_manual['id']}")
    pin_both = client.post(f"/api/v1/notes/moodboards/{moodboard_id}/notes/{note_both['id']}")
    pin_both_duplicate = client.post(f"/api/v1/notes/moodboards/{moodboard_id}/notes/{note_both['id']}")
    assert pin_manual.status_code == 200
    assert pin_both.status_code == 200
    assert pin_both_duplicate.status_code == 200
    assert pin_both_duplicate.json()["success"] is False

    board_notes = client.get(f"/api/v1/notes/moodboards/{moodboard_id}/notes")
    assert board_notes.status_code == 200
    notes_payload = board_notes.json()["notes"]
    by_id = {row["id"]: row for row in notes_payload}
    assert by_id[note_manual["id"]]["membership_source"] == "manual"
    assert by_id[note_both["id"]]["membership_source"] == "both"

    paged = client.get(f"/api/v1/notes/moodboards/{moodboard_id}/notes?limit=1&offset=0")
    assert paged.status_code == 200
    paged_body = paged.json()
    assert paged_body["count"] == 1
    assert paged_body["total"] == 2
    assert paged_body["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
        "next_offset": 1,
    }
    assert paged_body["has_more"] is True
    assert paged_body["next_offset"] == 1

    unpin_both = client.delete(f"/api/v1/notes/moodboards/{moodboard_id}/notes/{note_both['id']}")
    assert unpin_both.status_code == 200
    assert unpin_both.json()["success"] is True

    board_notes_after = client.get(f"/api/v1/notes/moodboards/{moodboard_id}/notes")
    assert board_notes_after.status_code == 200
    by_id_after = {row["id"]: row for row in board_notes_after.json()["notes"]}
    assert by_id_after[note_both["id"]]["membership_source"] == "smart"

    patch_resp = client.patch(
        f"/api/v1/notes/moodboards/{moodboard_id}",
        json={"name": "Inspiration Updated"},
        headers={"expected-version": str(fetched["version"])},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    patched = patch_resp.json()
    assert patched["name"] == "Inspiration Updated"

    delete_resp = client.delete(
        f"/api/v1/notes/moodboards/{moodboard_id}",
        headers={"expected-version": str(patched["version"])},
    )
    assert delete_resp.status_code == 204
    assert client.get(f"/api/v1/notes/moodboards/{moodboard_id}").status_code == 404


def test_moodboard_not_found_endpoints(moodboard_client: TestClient):
    client = moodboard_client
    assert client.get("/api/v1/notes/moodboards/999999").status_code == 404
    assert client.get("/api/v1/notes/moodboards/999999/notes").status_code == 404
