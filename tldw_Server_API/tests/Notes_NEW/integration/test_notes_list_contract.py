"""
Contract tests for the notes list endpoints (NL-01, #3103).

The /notes WebUI pages through ``GET /api/v1/notes/`` and ``GET /api/v1/notes/trash``.
These tests pin the contract the client relies on, against the real notes
router and a real ChaChaNotes database:

- ``limit``/``offset`` select a page, and ``pagination.total`` reports the
  size of the whole library, not the page.
- ``sort_by``/``sort_order`` order the whole library before it is paged, with
  ``id`` as a deterministic tie-breaker so pages never overlap or skip notes.
- Sort parameters are whitelisted.
"""

import itertools
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

pytestmark = pytest.mark.integration

_LIBRARY_SIZE = 120
_PAGE_SIZE = 20
_CLOCK_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
_EDITED_CONTENT = "edited last"
_LOWERCASE_TITLE = "aardvark first alphabetically"


class _AllowRateLimiter:
    async def check_user_rate_limit(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        return True, {}


@contextmanager
def _controlled_clock(db: CharactersRAGDB, *, step: timedelta) -> Iterator[None]:
    """Make the DB stamp rows from a controlled clock instead of wall time."""
    ticks = itertools.count()

    def _now() -> str:
        moment = _CLOCK_START + step * next(ticks)
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(db, "_get_current_utc_timestamp_iso", _now)
        yield


def _seed_library(db: CharactersRAGDB) -> None:
    """120 notes created one second apart, then the oldest one edited.

    Titles are a permutation of "Note 000".."Note 119" so alphabetical order
    differs from creation order. One lowercase title proves title sorting is
    case-insensitive: a binary sort would place every "Note …" before it.
    """
    titles = [f"Note {(index * 37) % _LIBRARY_SIZE:03d}" for index in range(_LIBRARY_SIZE)]
    titles[_LIBRARY_SIZE // 2] = _LOWERCASE_TITLE
    with _controlled_clock(db, step=timedelta(seconds=1)):
        for title in titles:
            db.add_note(title=title, content=f"Body of {title}")
        oldest = min(db.list_notes(limit=_LIBRARY_SIZE, offset=0), key=lambda row: row["created_at"])
        db.update_note(oldest["id"], {"content": _EDITED_CONTENT}, expected_version=oldest["version"])


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


@pytest.fixture(scope="module")
def library_client(tmp_path_factory: pytest.TempPathFactory) -> Generator[TestClient, None, None]:
    """Read-only client over a shared seeded library."""
    db_path = tmp_path_factory.mktemp("notes_list_contract") / "library.db"
    db = CharactersRAGDB(str(db_path), client_id="integration_user")
    _seed_library(db)
    with _notes_client(db) as client:
        yield client
    db.close_all_connections()


@pytest.fixture()
def notes_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    """An empty database for tests that write."""
    db = CharactersRAGDB(str(tmp_path / "notes_list_contract.db"), client_id="integration_user")
    yield db
    db.close_all_connections()


@pytest.fixture()
def notes_client(notes_db: CharactersRAGDB) -> Generator[TestClient, None, None]:
    with _notes_client(notes_db) as client:
        yield client


def _get_page(client: TestClient, path: str, **params) -> dict:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _ids(payload: dict) -> list[str]:
    return [note["id"] for note in payload["items"]]


def _collect_all_pages(client: TestClient, path: str, **params) -> list[dict]:
    collected: list[dict] = []
    offset = 0
    while True:
        payload = _get_page(client, path, limit=_PAGE_SIZE, offset=offset, **params)
        collected.extend(payload["items"])
        if not payload["pagination"]["has_more"]:
            return collected
        offset = payload["pagination"]["next_offset"]


def _assert_ordered(
    rows: list[dict],
    key: Callable[[dict], object],
    *,
    descending: bool,
) -> None:
    """Rows follow ``key`` in the given direction, ties broken by ascending id."""
    for earlier, later in itertools.pairwise(rows):
        earlier_key, later_key = key(earlier), key(later)
        if earlier_key == later_key:
            assert earlier["id"] < later["id"], (earlier, later)
        elif descending:
            assert earlier_key > later_key, (earlier, later)
        else:
            assert earlier_key < later_key, (earlier, later)


def test_list_pages_are_disjoint_and_report_the_library_total(library_client: TestClient) -> None:
    first = _get_page(library_client, "/api/v1/notes/", limit=_PAGE_SIZE, offset=0)
    second = _get_page(library_client, "/api/v1/notes/", limit=_PAGE_SIZE, offset=_PAGE_SIZE)

    assert len(first["items"]) == _PAGE_SIZE
    assert len(second["items"]) == _PAGE_SIZE
    assert set(_ids(first)).isdisjoint(_ids(second))
    assert first["pagination"] == {
        "mode": "offset",
        "limit": _PAGE_SIZE,
        "offset": 0,
        "total": _LIBRARY_SIZE,
        "has_more": True,
        "next_offset": _PAGE_SIZE,
    }
    assert first["total"] == _LIBRARY_SIZE


def test_last_list_page_is_partial_and_ends_pagination(library_client: TestClient) -> None:
    last = _get_page(library_client, "/api/v1/notes/", limit=_PAGE_SIZE, offset=110)

    assert len(last["items"]) == _LIBRARY_SIZE - 110
    assert last["pagination"]["total"] == _LIBRARY_SIZE
    assert last["pagination"]["has_more"] is False
    assert last["pagination"]["next_offset"] is None


def test_default_order_is_most_recently_modified_first(library_client: TestClient) -> None:
    every_note = _collect_all_pages(library_client, "/api/v1/notes/")

    assert every_note[0]["content"] == _EDITED_CONTENT
    _assert_ordered(every_note, lambda note: note["last_modified"], descending=True)


@pytest.mark.parametrize(
    ("sort_by", "sort_order", "key", "descending"),
    [
        ("last_modified", "desc", lambda note: note["last_modified"], True),
        ("last_modified", "asc", lambda note: note["last_modified"], False),
        ("created_at", "desc", lambda note: note["created_at"], True),
        ("created_at", "asc", lambda note: note["created_at"], False),
        ("title", "asc", lambda note: note["title"].lower(), False),
        ("title", "desc", lambda note: note["title"].lower(), True),
    ],
)
def test_sort_orders_the_whole_library_before_paging(
    library_client: TestClient,
    sort_by: str,
    sort_order: str,
    key: Callable[[dict], object],
    descending: bool,
) -> None:
    every_note = _collect_all_pages(
        library_client, "/api/v1/notes/", sort_by=sort_by, sort_order=sort_order
    )

    assert len(every_note) == _LIBRARY_SIZE
    assert len({note["id"] for note in every_note}) == _LIBRARY_SIZE
    _assert_ordered(every_note, key, descending=descending)


def test_title_sort_puts_the_global_minimum_on_page_one(library_client: TestClient) -> None:
    first = _get_page(
        library_client, "/api/v1/notes/", limit=_PAGE_SIZE, offset=0, sort_by="title", sort_order="asc"
    )

    assert first["items"][0]["title"] == _LOWERCASE_TITLE
    assert [note["title"] for note in first["items"][1:4]] == ["Note 000", "Note 001", "Note 002"]


def test_created_and_modified_sorts_differ_after_an_edit(library_client: TestClient) -> None:
    newest_modified = _get_page(
        library_client, "/api/v1/notes/", limit=1, offset=0, sort_by="last_modified", sort_order="desc"
    )
    oldest_created = _get_page(
        library_client, "/api/v1/notes/", limit=1, offset=0, sort_by="created_at", sort_order="asc"
    )
    newest_created = _get_page(
        library_client, "/api/v1/notes/", limit=1, offset=0, sort_by="created_at", sort_order="desc"
    )

    assert newest_modified["items"][0]["content"] == _EDITED_CONTENT
    assert oldest_created["items"][0]["content"] == _EDITED_CONTENT
    assert newest_created["items"][0]["content"] != _EDITED_CONTENT


@pytest.mark.parametrize(
    "params",
    [
        {"sort_by": "content"},
        {"sort_by": "title; DROP TABLE notes"},
        {"sort_order": "sideways"},
    ],
)
def test_unknown_sort_parameters_are_rejected(library_client: TestClient, params: dict[str, str]) -> None:
    for path in ("/api/v1/notes/", "/api/v1/notes/trash"):
        response = library_client.get(path, params=params)
        assert response.status_code == 422, (path, response.text)


def test_pages_cover_every_note_once_when_timestamps_tie(
    notes_client: TestClient,
    notes_db: CharactersRAGDB,
) -> None:
    with _controlled_clock(notes_db, step=timedelta(0)):
        for index in range(45):
            notes_db.add_note(title="Same title", content=f"Body {index}")

    for sort_by in ("last_modified", "created_at", "title"):
        every_note = _collect_all_pages(notes_client, "/api/v1/notes/", sort_by=sort_by, sort_order="desc")
        ids = [note["id"] for note in every_note]
        assert ids == sorted(ids), sort_by


def test_trash_pages_and_sorts_like_the_active_list(
    notes_client: TestClient,
    notes_db: CharactersRAGDB,
) -> None:
    _seed_library(notes_db)
    by_title = sorted(notes_db.list_notes(limit=_LIBRARY_SIZE, offset=0), key=lambda note: note["title"].lower())
    trashed = by_title[:25]
    for note in trashed:
        notes_db.soft_delete_note(note["id"], expected_version=note["version"])

    first = _get_page(
        notes_client, "/api/v1/notes/trash", limit=_PAGE_SIZE, offset=0, sort_by="title", sort_order="asc"
    )
    every_trashed = _collect_all_pages(notes_client, "/api/v1/notes/trash", sort_by="title", sort_order="asc")
    active = _get_page(notes_client, "/api/v1/notes/", limit=1, offset=0)

    assert first["pagination"]["total"] == 25
    assert first["items"][0]["title"] == _LOWERCASE_TITLE
    assert [note["id"] for note in every_trashed] == [note["id"] for note in trashed]
    assert active["pagination"]["total"] == _LIBRARY_SIZE - 25


@pytest.mark.parametrize(
    ("sort_by", "sort_order"),
    [("content", "desc"), ("title; DROP TABLE notes", "asc"), ("title", "sideways")],
)
def test_note_store_rejects_unlisted_sort_parameters(
    notes_db: CharactersRAGDB,
    sort_by: str,
    sort_order: str,
) -> None:
    with pytest.raises(InputError):
        notes_db.list_notes(limit=10, offset=0, sort_by=sort_by, sort_order=sort_order)
