import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)


@contextmanager
def _temporary_chacha_db(client_id: str = "owner-1") -> Iterator[CharactersRAGDB]:
    """Yield a temp CharactersRAGDB and close all SQLite handles before cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = CharactersRAGDB(os.path.join(tmpdir, "ChaChaNotes.db"), client_id=client_id)
        try:
            yield db
        finally:
            db.close_all_connections()


def test_deck_visibility_and_share_records_persist():
    with _temporary_chacha_db() as db:
        deck_id = db.add_deck("Shared Biology", visibility="team")
        deck = db.get_deck(deck_id)

        assert deck is not None
        assert deck["visibility"] == "team"

        share = db.upsert_deck_share(
            deck_id,
            user_id=22,
            role="viewer",
            shared_by=11,
        )

        assert share["deck_id"] == deck_id
        assert share["user_id"] == 22
        assert share["role"] == "viewer"
        assert share["shared_by"] == 11
        assert share["shared_at"]
        assert db.get_deck_share(deck_id, user_id=22)["role"] == "viewer"
        assert [item["user_id"] for item in db.list_deck_shares(deck_id)] == [22]


def test_deck_share_upsert_normalizes_role_and_updates_existing_share():
    with _temporary_chacha_db() as db:
        deck_id = db.add_deck("Role Update Deck")
        timestamps = iter(("2026-04-30T12:00:00Z", "2026-04-30T12:00:01Z"))
        db._get_current_utc_timestamp_iso = lambda: next(timestamps)

        first = db.upsert_deck_share(deck_id, user_id=22, role="viewer", shared_by=11)
        second = db.upsert_deck_share(deck_id, user_id=22, role="OWNER", shared_by=12)

        shares = db.list_deck_shares(deck_id)
        assert len(shares) == 1
        assert first["deck_id"] == second["deck_id"] == deck_id
        assert second["role"] == "owner"
        assert second["shared_by"] == 12
        assert second["shared_at"] == first["shared_at"]
        assert second["last_modified"] == "2026-04-30T12:00:01Z"


def test_deck_share_rejects_invalid_role_missing_deck_and_self_share():
    with _temporary_chacha_db() as db:
        deck_id = db.add_deck("Private Deck")

        with pytest.raises(InputError):
            db.upsert_deck_share(deck_id, user_id=22, role="admin", shared_by=11)

        with pytest.raises(InputError):
            db.upsert_deck_share(deck_id, user_id=11, role="viewer", shared_by=11)

        with pytest.raises(ConflictError):
            db.upsert_deck_share(9999, user_id=22, role="viewer", shared_by=11)


def test_deleting_deck_removes_deck_share_records():
    with _temporary_chacha_db() as db:
        deck_id = db.add_deck("Temporary Shared Deck")
        db.upsert_deck_share(deck_id, user_id=22, role="viewer", shared_by=11)

        db.soft_delete_deck_by_id(deck_id)

        assert db.list_deck_shares(deck_id) == []


def test_shared_with_user_filter_lists_matching_active_decks_only():
    with _temporary_chacha_db() as db:
        shared_deck = db.add_deck("Shared Deck")
        private_deck = db.add_deck("Private Deck")
        deleted_deck = db.add_deck("Deleted Shared Deck")
        db.upsert_deck_share(shared_deck, user_id=22, role="viewer", shared_by=11)
        db.upsert_deck_share(deleted_deck, user_id=22, role="viewer", shared_by=11)
        db.soft_delete_deck_by_id(deleted_deck)

        decks = db.list_decks(shared_with_user_id=22)

        assert [deck["id"] for deck in decks] == [shared_deck]
        assert all(deck["id"] != private_deck for deck in decks)


def test_add_deck_undelete_preserves_visibility_when_omitted():
    with _temporary_chacha_db() as db:
        deck_id = db.add_deck("Restored Shared Deck", visibility="team")
        db.soft_delete_deck_by_id(deck_id)

        restored_id = db.add_deck("Restored Shared Deck")

        assert restored_id == deck_id
        restored = db.get_deck(deck_id)
        assert restored is not None
        assert restored["deleted"] == 0
        assert restored["visibility"] == "team"

        db.soft_delete_deck_by_id(deck_id)
        db.add_deck("Restored Shared Deck", visibility="public")
        assert db.get_deck(deck_id)["visibility"] == "public"


def test_shared_with_user_workspace_filter_can_include_general_scope():
    with _temporary_chacha_db() as db:
        db.upsert_workspace("ws-1", "Workspace One")
        db.upsert_workspace("ws-2", "Workspace Two")
        general_deck = db.add_deck("General Shared Deck")
        workspace_deck = db.add_deck("Workspace Shared Deck", workspace_id="ws-1")
        other_workspace_deck = db.add_deck("Other Workspace Shared Deck", workspace_id="ws-2")
        for deck_id in (general_deck, workspace_deck, other_workspace_deck):
            db.upsert_deck_share(deck_id, user_id=22, role="viewer", shared_by=11)

        strict_decks = db.list_decks(shared_with_user_id=22, workspace_id="ws-1")
        inclusive_decks = db.list_decks(
            shared_with_user_id=22,
            workspace_id="ws-1",
            include_workspace_items=True,
        )

        assert {deck["id"] for deck in strict_decks} == {workspace_deck}
        assert {deck["id"] for deck in inclusive_decks} == {general_deck, workspace_deck}
