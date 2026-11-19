"""Tags a view implies always match, whatever the any/all toggle says (NL-12, #3101).

The Inbox view lists notes tagged ``captured``. The WebUI used to send that
token with the tags the reader picked, under one ``tokens_match``: with Inbox
plus two tags and "Any", notes that were never captured showed up in the Inbox.
``required_tokens`` are tags a note must have, every one of them, on top of
``tokens`` and their ``tokens_match``: "captured AND (rag OR baking)".
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Notes_NEW.integration.test_notes_tag_filter_match import (
    _notes_client,
    _tag,
)

pytestmark = pytest.mark.integration


@pytest.fixture()
def notes_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    db = CharactersRAGDB(str(tmp_path / "notes_required_tokens.db"), client_id="integration_user")
    notes = {
        "captured rag": ["captured", "rag"],
        "captured baking": ["Captured", "baking"],
        "captured only": ["captured"],
        "rag only": ["rag"],
        "baking only": ["baking"],
        "captured rag and baking": ["captured", "rag", "baking"],
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


def test_required_tokens_and_the_any_match(notes_client: TestClient) -> None:
    titles, total = _titles(
        notes_client,
        tokens=["rag", "baking"],
        tokens_match="any",
        required_tokens=["captured"],
    )

    assert titles == ["captured baking", "captured rag", "captured rag and baking"]
    assert total == 3


def test_required_tokens_alone_filter_the_list(notes_client: TestClient) -> None:
    titles, total = _titles(notes_client, required_tokens=["CAPTURED"])

    assert titles == ["captured baking", "captured only", "captured rag", "captured rag and baking"]
    assert total == 4


def test_every_required_token_must_match(notes_client: TestClient) -> None:
    titles, total = _titles(notes_client, required_tokens=["captured", "baking"])

    assert titles == ["captured baking", "captured rag and baking"]
    assert total == 2


def test_an_unknown_required_token_matches_nothing(notes_client: TestClient) -> None:
    titles, total = _titles(
        notes_client,
        tokens=["rag"],
        tokens_match="any",
        required_tokens=["never-used"],
    )

    assert titles == []
    assert total == 0


def test_required_tokens_combine_with_a_text_query(notes_client: TestClient) -> None:
    titles, total = _titles(
        notes_client,
        query="Body",
        tokens=["rag", "baking"],
        tokens_match="any",
        required_tokens=["captured"],
    )

    assert titles == ["captured baking", "captured rag", "captured rag and baking"]
    assert total == 3
