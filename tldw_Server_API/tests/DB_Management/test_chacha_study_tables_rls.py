"""ChaCha study tables isolate accounts by RLS on a shared PostgreSQL.

On PostgreSQL every account's decks, flashcards and study packs share one set
of tables. Until TASK-13950.3 they had no policy, so the owner filter each
query happened to carry was the only isolation. This reads with raw SQL that
carries no owner filter, as a NOSUPERUSER NOBYPASSRLS role (a superuser is
exempt from RLS even under FORCE), so only the policy can hide the rows.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseBackend
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

STUDY_TABLES = (
    "decks",
    "flashcards",
    "flashcard_reviews",
    "flashcard_review_sessions",
    "flashcard_assets",
    "flashcard_templates",
    "deck_shares",
    "study_packs",
    "study_pack_cards",
    "flashcard_citations",
)


def _seed_every_study_table(db: CharactersRAGDB) -> None:
    deck = db.add_deck("Alice private deck")
    card = db.add_flashcard({"deck_id": deck, "front": "Alice front", "back": "Alice back"})
    db.review_flashcard(card, 3)
    db.add_flashcard_asset(image_bytes=b"alice-private", mime_type="image/png")
    db.add_flashcard_template(
        name="Alice template", model_type="basic", front_template="Alice front", back_template="Alice back"
    )
    db.upsert_deck_share(deck, user_id=4, shared_by=int(db.client_id))
    pack = db.create_study_pack(
        title="Alice pack",
        workspace_id=None,
        deck_id=deck,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "alice-note"}]},
        generation_options_json={"deck_mode": "new"},
    )
    db.add_study_pack_cards(pack, [card])
    db.add_flashcard_citations(
        card, [{"source_type": "note", "source_id": "alice-note", "citation_text": "Alice evidence"}]
    )


def _owned_counts(db: CharactersRAGDB, owner: str) -> dict[str, int]:
    """Count ``owner``'s rows per table with no app-level owner filter."""
    with db.transaction() as conn:
        return {
            table: conn.execute(
                f"SELECT count(*) AS n FROM {table} WHERE client_id = ?",  # nosec B608 - fixed table list
                (owner,),
            ).fetchone()["n"]
            for table in STUDY_TABLES
        }


def test_second_user_cannot_read_first_users_study_rows(
    pg_restricted_backend: DatabaseBackend,
) -> None:
    alice = CharactersRAGDB(db_path=":memory:", client_id="930001", backend=pg_restricted_backend)
    bob = CharactersRAGDB(db_path=":memory:", client_id="930002", backend=pg_restricted_backend)
    try:
        _seed_every_study_table(alice)

        seen_by_owner = _owned_counts(alice, alice.client_id)
        assert all(seen_by_owner.values()), f"seed missed a table: {seen_by_owner}"

        assert _owned_counts(bob, alice.client_id) == dict.fromkeys(STUDY_TABLES, 0)

        with bob.transaction() as conn:
            forced = {
                row["relname"]
                for row in conn.execute(
                    "SELECT relname FROM pg_class WHERE relname = ANY(?) "
                    "AND relrowsecurity AND relforcerowsecurity",
                    (list(STUDY_TABLES),),
                ).fetchall()
            }
        assert forced == set(STUDY_TABLES)
    finally:
        alice.close_connection()
        bob.close_connection()
