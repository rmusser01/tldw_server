"""
Filtering notes by tag matches tags exactly and can ask for all or any (NL-N1, #3101).

The tag filter in /notes sends ``tokens`` to ``GET /api/v1/notes/search/``.
It used to match each token as a substring and OR the tokens together, so
picking the tag "eval" listed notes tagged only "evaluation", and adding a
second tag widened the list instead of narrowing it. Tokens now match a tag
exactly (ignoring case). ``tokens_match=all`` (the default) keeps notes that
have every token; ``tokens_match=any`` keeps notes that have at least one.
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


def _tag(db: CharactersRAGDB, note_id: str, keyword: str) -> None:
    existing = db.get_keyword_by_text(keyword)
    keyword_id = existing["id"] if existing else db.add_keyword(keyword)
    db.link_note_to_keyword(note_id, keyword_id)


@pytest.fixture()
def notes_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    db = CharactersRAGDB(str(tmp_path / "notes_tag_filter.db"), client_id="integration_user")
    notes = {
        "evaluation only": ["evaluation"],
        "eval only": ["Eval"],
        "rag only": ["rag"],
        "baking only": ["baking"],
        "rag and baking": ["rag", "baking"],
        "rag and baking and eval": ["rag", "baking", "eval"],
    }
    for title, tags in notes.items():
        note_id = db.add_note(title=title, content=f"Body of {title}")
        for tag in tags:
            _tag(db, note_id, tag)
    with _notes_client(db) as client:
        yield client
    db.close_all_connections()


def _titles(client: TestClient, **params) -> tuple[list[str], int]:
    response = client.get("/api/v1/notes/search/", params={"limit": 100, **params})
    assert response.status_code == 200, response.text
    body = response.json()
    return sorted(note["title"] for note in body["items"]), body["total"]


def test_a_tag_matches_exactly_not_as_a_substring(notes_client: TestClient) -> None:
    titles, total = _titles(notes_client, tokens=["eval"])

    assert titles == ["eval only", "rag and baking and eval"]
    assert total == 2


def test_a_tag_matches_regardless_of_case(notes_client: TestClient) -> None:
    titles, _ = _titles(notes_client, tokens=["EVAL"])

    assert "eval only" in titles


def test_several_tags_keep_notes_that_have_all_of_them_by_default(notes_client: TestClient) -> None:
    titles, total = _titles(notes_client, tokens=["rag", "baking"])

    assert titles == ["rag and baking", "rag and baking and eval"]
    assert total == 2


def test_match_any_keeps_notes_that_have_at_least_one_tag(notes_client: TestClient) -> None:
    titles, total = _titles(notes_client, tokens=["rag", "baking"], tokens_match="any")

    assert titles == ["baking only", "rag and baking", "rag and baking and eval", "rag only"]
    assert total == 4


def test_match_all_with_three_tags_narrows_further(notes_client: TestClient) -> None:
    titles, total = _titles(notes_client, tokens=["rag", "baking", "eval"], tokens_match="all")

    assert titles == ["rag and baking and eval"]
    assert total == 1


def test_a_repeated_tag_counts_once(notes_client: TestClient) -> None:
    titles, _ = _titles(notes_client, tokens=["rag", "RAG"])

    assert titles == ["rag and baking", "rag and baking and eval", "rag only"]


def test_match_all_combines_with_a_text_query(notes_client: TestClient) -> None:
    titles, total = _titles(notes_client, query="Body", tokens=["rag", "baking"])

    assert titles == ["rag and baking", "rag and baking and eval"]
    assert total == 2


def test_an_unknown_match_mode_is_rejected(notes_client: TestClient) -> None:
    response = notes_client.get("/api/v1/notes/search/", params={"tokens": ["rag"], "tokens_match": "some"})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["loc"][-1] == "tokens_match" for error in errors)
