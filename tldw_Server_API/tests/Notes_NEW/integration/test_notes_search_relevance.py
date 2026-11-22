"""GET /api/v1/notes/search/ keeps relevance, honours a sort and returns a snippet (NL-07, #3101).

The endpoint documented ``sort_by``/``sort_order`` as accepted and ignored, and
returned no match context, so the WebUI re-sorted relevance pages by date and
showed the first line of body-only hits. Without a sort (or with
``sort_by=relevance``) a query orders by relevance; ``last_modified``,
``created_at`` and ``title`` order the whole result in SQL. Each item carries
``snippet``: a short excerpt of the body around the match, split into segments
whose ``match`` flag marks a matched term, or null when only the title matched.

Clients that still send other ``sort_by``/``sort_order`` values keep working:
known values match ignoring case, an unknown ``sort_by`` falls back to
relevance and an unknown ``sort_order`` to the default (desc), never a 422.
"""

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

_FILLER = "Notes about flour, water and salt. " * 12


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


def _add(db: CharactersRAGDB, title: str, content: str, modified: str) -> str:
    note_id = db.add_note(title=title, content=content)
    db.execute_query(
        "UPDATE notes SET last_modified = ? WHERE id = ?",
        (modified, note_id),
        commit=True,
    )
    return note_id


def _tag(db: CharactersRAGDB, note_id: str, keyword: str) -> None:
    existing = db.get_keyword_by_text(keyword)
    keyword_id = existing["id"] if existing else db.add_keyword(keyword)
    db.link_note_to_keyword(note_id, keyword_id)


@pytest.fixture()
def notes_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    db = CharactersRAGDB(str(tmp_path / "notes_search_relevance.db"), client_id="integration_user")
    loaf = _add(
        db,
        "Sourdough country loaf",
        "Feed the sourdough starter, then mix the sourdough levain.",
        "2026-01-01T00:00:00+00:00",
    )
    bread = _add(db, "bread log", f"{_FILLER}Tried a sourdough crumb test. {_FILLER}", "2026-02-01T00:00:00+00:00")
    apple = _add(db, "Apple pie", f"{_FILLER}A friend mentioned sourdough once. {_FILLER}", "2026-03-01T00:00:00+00:00")
    _add(db, "Sourdough shopping", "Flour and a new banneton.", "2026-04-01T00:00:00+00:00")
    for note_id in (loaf, bread, apple):
        _tag(db, note_id, "baking")
    with _notes_client(db) as client:
        yield client
    db.close_all_connections()


def _search(client: TestClient, **params) -> list[dict]:
    response = client.get("/api/v1/notes/search/", params={"limit": 20, **params})
    assert response.status_code == 200, response.text
    return response.json()["items"]


def _titles(items: list[dict]) -> list[str]:
    return [item["title"] for item in items]


def test_query_without_a_sort_orders_by_relevance(notes_client: TestClient) -> None:
    items = _search(notes_client, query="sourdough")
    assert _titles(items)[0] == "Sourdough country loaf"
    assert _titles(_search(notes_client, query="sourdough", sort_by="relevance")) == _titles(items)


@pytest.mark.parametrize(
    ("sort_by", "sort_order", "expected"),
    [
        ("last_modified", "desc", ["Sourdough shopping", "Apple pie", "bread log", "Sourdough country loaf"]),
        ("title", "asc", ["Apple pie", "bread log", "Sourdough country loaf", "Sourdough shopping"]),
    ],
)
def test_explicit_sort_is_honoured(
    notes_client: TestClient, sort_by: str, sort_order: str, expected: list[str]
) -> None:
    assert _titles(_search(notes_client, query="sourdough", sort_by=sort_by, sort_order=sort_order)) == expected


def test_tag_search_honours_sort(notes_client: TestClient) -> None:
    items = _search(notes_client, query="sourdough", tokens="baking", sort_by="title", sort_order="desc")
    assert _titles(items) == ["Sourdough country loaf", "bread log", "Apple pie"]


@pytest.mark.parametrize("sort_order", [None, "asc", "sideways"])
def test_unknown_sort_by_falls_back_to_relevance(notes_client: TestClient, sort_order: str | None) -> None:
    """The endpoint once accepted and ignored any sort; old clients must not get a 422."""
    relevance = _titles(_search(notes_client, query="sourdough"))
    params = {"query": "sourdough", "sort_by": "updated_at"}
    if sort_order is not None:
        params["sort_order"] = sort_order
    assert _titles(_search(notes_client, **params)) == relevance


def test_unknown_sort_by_falls_back_to_relevance_for_tag_searches(notes_client: TestClient) -> None:
    relevance = _titles(_search(notes_client, query="sourdough", tokens="baking"))
    assert _titles(_search(notes_client, query="sourdough", tokens="baking", sort_by="bogus")) == relevance


@pytest.mark.parametrize(
    ("sort_by", "sort_order", "expected"),
    [
        ("last_modified", "DESC", ["Sourdough shopping", "Apple pie", "bread log", "Sourdough country loaf"]),
        ("Title", "Asc", ["Apple pie", "bread log", "Sourdough country loaf", "Sourdough shopping"]),
        # An unknown direction uses the default, desc.
        ("LAST_MODIFIED", "sideways", ["Sourdough shopping", "Apple pie", "bread log", "Sourdough country loaf"]),
    ],
)
def test_known_sorts_match_ignoring_case(
    notes_client: TestClient, sort_by: str, sort_order: str, expected: list[str]
) -> None:
    assert _titles(_search(notes_client, query="sourdough", sort_by=sort_by, sort_order=sort_order)) == expected


def test_unknown_sort_is_logged_at_debug_not_warning(notes_client: TestClient) -> None:
    records: list[tuple[str, str]] = []
    sink_id = logger.add(lambda message: records.append((message.record["level"].name, message.record["message"])))
    try:
        _search(notes_client, query="sourdough", sort_by="updated_at", sort_order="sideways")
    finally:
        logger.remove(sink_id)
    mentions = [(level, text) for level, text in records if "updated_at" in text or "sideways" in text]
    assert ("DEBUG" in {level for level, _ in mentions}), records
    # Request logging can echo the query string at INFO; only WARNING and above would be noise.
    assert not [m for m in mentions if m[0] in {"WARNING", "ERROR", "CRITICAL"}], mentions


def test_openapi_documents_the_sort_values_without_rejecting_others(notes_client: TestClient) -> None:
    spec = notes_client.app.openapi()
    parameters = {param["name"]: param for param in spec["paths"]["/api/v1/notes/search/"]["get"]["parameters"]}
    for name, values in (
        ("sort_by", ("relevance", "last_modified", "created_at", "title")),
        ("sort_order", ("asc", "desc")),
    ):
        param = parameters[name]
        schema_text = repr(param["schema"])
        assert "enum" not in schema_text, param
        assert all(value in param["description"] for value in values), param


def test_items_carry_a_highlighted_snippet_for_body_matches(notes_client: TestClient) -> None:
    items = {item["title"]: item for item in _search(notes_client, query="sourdough")}

    apple = items["Apple pie"]["snippet"]
    assert {"text": "sourdough", "match": True} in apple
    assert "A friend mentioned sourdough once." in "".join(segment["text"] for segment in apple)
    # Only the title matched, so there is nothing to show from the body.
    assert items["Sourdough shopping"]["snippet"] is None


def test_title_only_search_has_no_snippet(notes_client: TestClient) -> None:
    items = _search(notes_client, query="Apple", title_only="true")
    assert _titles(items) == ["Apple pie"]
    assert items[0]["snippet"] is None


@pytest.mark.parametrize("tokens", [None, "baking"])
def test_snippets_can_be_left_out(notes_client: TestClient, tokens: str | None) -> None:
    """Callers that only need ids and titles (select all, export) skip the excerpt work."""
    params = {"query": "sourdough", "include_snippets": "false"}
    if tokens:
        params["tokens"] = tokens
    items = _search(notes_client, **params)
    assert "Apple pie" in _titles(items)
    assert all(item["snippet"] is None for item in items)


def test_openapi_documents_include_snippets(notes_client: TestClient) -> None:
    spec = notes_client.app.openapi()
    parameters = {param["name"]: param for param in spec["paths"]["/api/v1/notes/search/"]["get"]["parameters"]}
    assert parameters["include_snippets"]["schema"]["default"] is True
