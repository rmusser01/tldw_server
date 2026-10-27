"""
The notes list payload can skip its duplicate arrays (NS-04, #3101).

``GET /api/v1/notes/``, ``/notes/trash`` and ``/notes/search/`` return the page
in ``items`` and, for older clients, two more copies in ``notes`` and
``results``. Every autosave in the /notes WebUI refreshes the list, so the
copies tripled the payload. ``legacy_aliases=false`` leaves them out; the
default keeps them so existing clients see no change.
"""

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


class _AllowRateLimiter:
    async def check_user_rate_limit(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        return True, {}


@contextmanager
def _notes_client(db: CharactersRAGDB) -> Iterator[TestClient]:
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


@pytest.fixture()
def notes_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    db = CharactersRAGDB(str(tmp_path / "notes_aliases.db"), client_id="integration_user")
    for index in range(3):
        db.add_note(title=f"Alias note {index}", content=f"Body {index}")
    trashed_id = db.add_note(title="Alias note trashed", content="Gone")
    trashed = db.get_note_by_id(trashed_id)
    db.soft_delete_note(trashed_id, expected_version=trashed["version"])
    with _notes_client(db) as client:
        yield client
    db.close_all_connections()


_LIST_PATHS = [
    ("/api/v1/notes/", {}),
    ("/api/v1/notes/trash", {}),
    ("/api/v1/notes/search/", {"query": "Alias"}),
]


@pytest.mark.parametrize(("path", "params"), _LIST_PATHS)
def test_legacy_aliases_are_returned_by_default(notes_client: TestClient, path: str, params: dict) -> None:
    response = notes_client.get(path, params=params)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["items"], "fixture should produce at least one row"
    assert payload["notes"] == payload["items"]
    assert payload["results"] == payload["items"]


@pytest.mark.parametrize(("path", "params"), _LIST_PATHS)
def test_legacy_aliases_can_be_skipped(notes_client: TestClient, path: str, params: dict) -> None:
    response = notes_client.get(path, params={**params, "legacy_aliases": "false"})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["items"], "fixture should produce at least one row"
    assert payload.get("notes") is None
    assert payload.get("results") is None
    assert payload["count"] == len(payload["items"])
    assert payload["pagination"]["total"] == payload["total"]


def test_skipping_aliases_shrinks_the_payload(notes_client: TestClient) -> None:
    full = notes_client.get("/api/v1/notes/")
    compact = notes_client.get("/api/v1/notes/", params={"legacy_aliases": "false"})

    assert full.status_code == 200 and compact.status_code == 200
    assert len(compact.content) < len(full.content) / 2
