"""A tags-only relink advances the note version, on SQLite and PostgreSQL (#3157).

PATCH checks expected-version before it relinks, so the rollback below covers the
race that check cannot: another write lands between the check and the relink.
These tests drive the relink through ``_sync_note_organization``, the helper the
PATCH endpoint uses, with a version that has gone stale.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def db(request, tmp_path):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    database = CharactersRAGDB(tmp_path / "notes.db", client_id="930010", backend=backend)
    try:
        yield database
    finally:
        database.close_all_connections()
        if backend:
            backend.get_pool().close_all()


def _tag(db: CharactersRAGDB, note_id: str, keyword: str) -> None:
    existing = db.get_keyword_by_text(keyword)
    keyword_id = existing["id"] if existing else db.add_keyword(keyword)
    db.link_note_to_keyword(note_id, keyword_id)


def _tags(db: CharactersRAGDB, note_id: str) -> list[str]:
    return sorted(row["keyword"] for row in db.get_keywords_for_note(note_id))


def _version(db: CharactersRAGDB, note_id: str) -> int:
    note = db.get_note_by_id(note_id)
    assert note is not None
    return int(note["version"])


def test_stale_bump_note_version_raises_and_keeps_the_version(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Tagged", content="Body")

    assert db.note_store.bump_note_version(note_id, 1) == 2
    with pytest.raises(ConflictError):
        db.note_store.bump_note_version(note_id, 1)

    assert _version(db, note_id) == 2


def test_stale_tags_only_relink_rolls_back_its_keyword_links(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Tagged", content="Body")
    _tag(db, note_id, "Alpha")

    first = notes_endpoint._sync_note_organization(
        db, note_id=note_id, keywords=["Beta"], folder_paths=None, bump_from_version=1
    )
    assert first is not None and first["failed_count"] == 0
    assert (_tags(db, note_id), _version(db, note_id)) == (["Beta"], 2)

    # A second client that also read version 1 relinks after the first one won.
    with pytest.raises(ConflictError):
        notes_endpoint._sync_note_organization(
            db, note_id=note_id, keywords=["Gamma"], folder_paths=None, bump_from_version=1
        )

    assert (_tags(db, note_id), _version(db, note_id)) == (["Beta"], 2)


def test_tag_created_concurrently_before_the_relink_still_links(db: CharactersRAGDB, monkeypatch) -> None:
    """Another request creates the tag between our lookup and our insert.

    On PostgreSQL the losing insert raises a unique violation, which aborts the
    transaction it runs in. Keyword rows are resolved before the relink transaction,
    so the conflict only fails that insert, and the lookup that follows finds the row.
    """
    note_id = db.add_note(title="Tagged", content="Body")
    db.add_keyword("Gamma")  # the other request's row
    lookup = db.keyword_store.get_keyword_by_text
    misses = {"left": 2}

    def stale_lookup(keyword_text: str):
        # Our existence check and add_keyword's own check both ran before the other insert.
        if keyword_text.strip().lower() == "gamma" and misses["left"] > 0:
            misses["left"] -= 1
            return None
        return lookup(keyword_text)

    monkeypatch.setattr(db.keyword_store, "get_keyword_by_text", stale_lookup)

    summary = notes_endpoint._sync_note_organization(
        db, note_id=note_id, keywords=["Gamma"], folder_paths=None, bump_from_version=1
    )

    assert misses["left"] == 0
    assert summary is not None and summary["failed_count"] == 0
    assert (_tags(db, note_id), _version(db, note_id)) == (["Gamma"], 2)


def test_relink_creates_new_tags_before_its_transaction_opens(tmp_path, monkeypatch) -> None:
    """Creation is idempotent by identity, so it needs no lock; only link, unlink and bump do."""
    database = CharactersRAGDB(tmp_path / "notes.db", client_id="930011")
    try:
        note_id = database.add_note(title="Tagged", content="Body")
        add_keyword = database.add_keyword
        in_transaction: list[bool] = []

        def spy_add_keyword(keyword_text: str):
            in_transaction.append(database.get_connection().in_transaction)
            return add_keyword(keyword_text)

        monkeypatch.setattr(database, "add_keyword", spy_add_keyword)

        notes_endpoint._sync_note_organization(
            database, note_id=note_id, keywords=["Delta"], folder_paths=None, bump_from_version=1
        )

        assert in_transaction == [False]
        assert _tags(database, note_id) == ["Delta"]
    finally:
        database.close_all_connections()
