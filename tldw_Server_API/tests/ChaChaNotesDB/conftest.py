"""Shared fixtures for ChaChaNotesDB quiz persistence tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def quiz_db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(tmp_path / "quiz-persistence.db", client_id="quiz-persistence-test")
    try:
        yield db
    finally:
        db.close_all_connections()
