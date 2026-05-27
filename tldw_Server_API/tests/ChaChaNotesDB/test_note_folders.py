from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path):
    notes_db = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="folder-test")
    try:
        yield notes_db
    finally:
        notes_db.close_connection()


def test_create_note_folder_path_is_idempotent_and_listed(db: CharactersRAGDB) -> None:
    created = db.create_note_folder_path("Inbox/Captured Articles")
    duplicate = db.create_note_folder_path("inbox/captured articles")

    assert duplicate == created

    folders = db.list_note_folders()
    assert [folder["path"] for folder in folders] == [
        "Inbox",
        "Inbox/Captured Articles",
    ]
    assert folders[0]["parent_id"] is None
    assert folders[1]["parent_id"] == folders[0]["id"]
