import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-packs.db"), client_id="study-pack-tests")
    chacha.upsert_workspace("ws-1", "Workspace 1")
    try:
        yield chacha
    finally:
        chacha.close_connection()


def _create_card(db: CharactersRAGDB, *, deck_id: int, front: str) -> str:
    return db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": front,
            "back": f"{front} answer",
            "notes": "study pack fixture",
        }
    )


def test_study_pack_workspace_id_persists_alongside_destination_deck(db: CharactersRAGDB):
    deck_id = db.add_deck("Networking Deck", workspace_id="ws-1")

    pack_id = db.create_study_pack(
        title="Networking",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "n1"}]},
        generation_options_json={"deck_mode": "new"},
    )

    pack = db.get_study_pack(pack_id)

    assert pack["workspace_id"] == "ws-1"  # nosec B101
    assert pack["deck_id"] == deck_id  # nosec B101
    assert pack["status"] == "active"  # nosec B101


def test_create_study_pack_rejects_whitespace_only_source_id(db: CharactersRAGDB):
    deck_id = db.add_deck("Invalid Source Deck", workspace_id="ws-1")

    with pytest.raises(InputError, match="source_id"):
        db.create_study_pack(
            title="Invalid Source Pack",
            workspace_id="ws-1",
            deck_id=deck_id,
            source_bundle_json={"items": [{"source_type": "note", "source_id": "   "}]},
            generation_options_json={"deck_mode": "new"},
        )


def test_study_pack_card_membership_survives_manual_deck_changes(db: CharactersRAGDB):
    original_deck_id = db.add_deck("Original Deck")
    moved_deck_id = db.add_deck("Moved Deck")
    first_card_uuid = _create_card(db, deck_id=original_deck_id, front="TCP handshake")
    second_card_uuid = _create_card(db, deck_id=original_deck_id, front="CIDR notation")

    pack_id = db.create_study_pack(
        title="Networking",
        workspace_id="ws-1",
        deck_id=original_deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "n1"}]},
        generation_options_json={"deck_mode": "new"},
    )
    db.add_study_pack_cards(pack_id, [first_card_uuid, second_card_uuid])

    assert [row["flashcard_uuid"] for row in db.list_study_pack_cards(pack_id)] == [  # nosec B101
        first_card_uuid,
        second_card_uuid,
    ]

    assert db.update_flashcard(first_card_uuid, {"deck_id": moved_deck_id}, expected_version=1) is True  # nosec B101
    assert db.update_flashcard(second_card_uuid, {"deck_id": moved_deck_id}, expected_version=1) is True  # nosec B101

    rows = db.list_study_pack_cards(pack_id)

    assert [row["flashcard_uuid"] for row in rows] == [first_card_uuid, second_card_uuid]  # nosec B101


def test_add_study_pack_cards_returns_actual_insert_count_when_duplicates_are_ignored(db: CharactersRAGDB):
    deck_id = db.add_deck("Distributed Systems Deck")
    first_card_uuid = _create_card(db, deck_id=deck_id, front="Consensus")
    second_card_uuid = _create_card(db, deck_id=deck_id, front="Raft")
    pack_id = db.create_study_pack(
        title="Distributed Systems",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "n2"}]},
        generation_options_json={"deck_mode": "new"},
    )

    inserted = db.add_study_pack_cards(pack_id, [first_card_uuid, first_card_uuid, second_card_uuid])

    assert inserted == 2  # nosec B101
    assert [row["flashcard_uuid"] for row in db.list_study_pack_cards(pack_id)] == [  # nosec B101
        first_card_uuid,
        second_card_uuid,
    ]


def test_add_flashcard_citations_rejects_negative_ordinals(db: CharactersRAGDB):
    deck_id = db.add_deck("Computer Architecture Deck")
    card_uuid = _create_card(db, deck_id=deck_id, front="What is a cache line?")

    with pytest.raises(InputError, match="ordinal"):
        db.add_flashcard_citations(
            card_uuid,
            [
                {
                    "source_type": "note",
                    "source_id": "note-999",
                    "citation_text": "Cache lines are fixed-size blocks.",
                    "locator": "chapter-3",
                    "ordinal": -1,
                }
            ],
        )


def test_add_flashcard_citations_rejects_non_numeric_ordinals(db: CharactersRAGDB):
    deck_id = db.add_deck("Operating Systems Input Deck")
    card_uuid = _create_card(db, deck_id=deck_id, front="What is thrashing?")

    with pytest.raises(InputError, match="ordinal"):
        db.add_flashcard_citations(
            card_uuid,
            [
                {
                    "source_type": "note",
                    "source_id": "note-bad-ordinal",
                    "citation_text": "Thrashing is excessive paging.",
                    "locator": "chapter-8",
                    "ordinal": "abc",
                }
            ],
        )


def test_add_flashcard_citations_rejects_blank_source_id(db: CharactersRAGDB):
    deck_id = db.add_deck("Compilers Deck")
    card_uuid = _create_card(db, deck_id=deck_id, front="What is a parse tree?")

    with pytest.raises(InputError, match="source_id"):
        db.add_flashcard_citations(
            card_uuid,
            [
                {
                    "source_type": "note",
                    "source_id": "   ",
                    "citation_text": "A parse tree shows syntactic structure.",
                    "locator": "section-1",
                    "ordinal": 0,
                }
            ],
        )


def test_flashcard_citations_preserve_storage_metadata(db: CharactersRAGDB):
    deck_id = db.add_deck("Operating Systems Deck")
    card_uuid = _create_card(db, deck_id=deck_id, front="What is paging?")

    db.add_flashcard_citations(
        card_uuid,
        [
            {
                "source_type": "note",
                "source_id": "note-123",
                "citation_text": "Paging swaps fixed-size pages between memory and disk.",
                "locator": "chapter-7",
                "ordinal": 0,
            },
            {
                "source_type": "media",
                "source_id": "media-9",
                "citation_text": "Thrashing happens when page faults dominate execution.",
                "locator": "00:10:04",
                "ordinal": 1,
            },
        ],
    )

    citations = db.list_flashcard_citations(card_uuid)

    assert [citation["ordinal"] for citation in citations] == [0, 1]  # nosec B101
    assert citations[0]["client_id"] == "study-pack-tests"  # nosec B101
    assert citations[0]["version"] == 1  # nosec B101
    assert citations[0]["deleted"] in (False, 0)  # nosec B101


def test_soft_delete_study_pack_enforces_version_and_writes_delete_metadata(db: CharactersRAGDB):
    deck_id = db.add_deck("Security Deck", workspace_id="ws-1")
    pack_id = db.create_study_pack(
        title="Security",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "sec-1"}]},
        generation_options_json={"deck_mode": "new"},
    )

    with pytest.raises(ConflictError, match="Version mismatch deleting study pack"):
        db.soft_delete_study_pack(pack_id, expected_version=99)

    assert db.soft_delete_study_pack(pack_id, expected_version=1) is True  # nosec B101
    assert db.get_study_pack(pack_id) is None  # nosec B101

    row = db.execute_query(
        """
        SELECT deleted, version, client_id
          FROM study_packs
         WHERE id = ?
        """,
        (pack_id,),
    ).fetchone()
    sync_row = db.execute_query(
        """
        SELECT operation, version, client_id
          FROM sync_log
         WHERE entity = 'study_packs' AND entity_id = ?
         ORDER BY change_id DESC
         LIMIT 1
        """,
        (str(pack_id),),
    ).fetchone()

    assert row["deleted"] in (True, 1)  # nosec B101
    assert row["version"] == 2  # nosec B101
    assert row["client_id"] == "study-pack-tests"  # nosec B101
    assert sync_row["operation"] == "delete"  # nosec B101
    assert sync_row["version"] == 2  # nosec B101
    assert sync_row["client_id"] == "study-pack-tests"  # nosec B101


def test_supersede_study_pack_enforces_version_and_writes_update_metadata(db: CharactersRAGDB):
    deck_id = db.add_deck("Databases Deck", workspace_id="ws-1")
    original_pack_id = db.create_study_pack(
        title="Databases v1",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "db-1"}]},
        generation_options_json={"deck_mode": "new"},
    )
    replacement_pack_id = db.create_study_pack(
        title="Databases v2",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "db-2"}]},
        generation_options_json={"deck_mode": "new"},
    )

    with pytest.raises(ConflictError, match="Version mismatch updating study pack"):
        db.supersede_study_pack(original_pack_id, superseded_by_pack_id=replacement_pack_id, expected_version=99)

    assert db.supersede_study_pack(original_pack_id, superseded_by_pack_id=replacement_pack_id, expected_version=1) is True  # nosec B101

    row = db.execute_query(
        """
        SELECT status, superseded_by_pack_id, version, deleted, client_id
          FROM study_packs
         WHERE id = ?
        """,
        (original_pack_id,),
    ).fetchone()
    sync_row = db.execute_query(
        """
        SELECT operation, version, client_id
          FROM sync_log
         WHERE entity = 'study_packs' AND entity_id = ?
         ORDER BY change_id DESC
         LIMIT 1
        """,
        (str(original_pack_id),),
    ).fetchone()

    assert row["status"] == "superseded"  # nosec B101
    assert row["superseded_by_pack_id"] == replacement_pack_id  # nosec B101
    assert row["version"] == 2  # nosec B101
    assert row["deleted"] in (False, 0)  # nosec B101
    assert row["client_id"] == "study-pack-tests"  # nosec B101
    assert sync_row["operation"] == "update"  # nosec B101
    assert sync_row["version"] == 2  # nosec B101
    assert sync_row["client_id"] == "study-pack-tests"  # nosec B101


def test_supersede_study_pack_rejects_self_supersede(db: CharactersRAGDB):
    deck_id = db.add_deck("Networking Self-Supersede Deck", workspace_id="ws-1")
    pack_id = db.create_study_pack(
        title="Networking Pack",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "net-1"}]},
        generation_options_json={"deck_mode": "new"},
    )

    with pytest.raises(InputError, match="self-supersede"):
        db.supersede_study_pack(pack_id, superseded_by_pack_id=pack_id, expected_version=1)


def test_supersede_study_pack_rejects_nonexistent_replacement_pack(db: CharactersRAGDB):
    deck_id = db.add_deck("Replacement Validation Deck", workspace_id="ws-1")
    pack_id = db.create_study_pack(
        title="Original Pack",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "replace-1"}]},
        generation_options_json={"deck_mode": "new"},
    )

    with pytest.raises(ConflictError, match="replacement"):
        db.supersede_study_pack(pack_id, superseded_by_pack_id=999_999, expected_version=1)


def test_supersede_study_pack_rejects_deleted_replacement_pack(db: CharactersRAGDB):
    deck_id = db.add_deck("Deleted Replacement Deck", workspace_id="ws-1")
    pack_id = db.create_study_pack(
        title="Original Pack",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "replace-2"}]},
        generation_options_json={"deck_mode": "new"},
    )
    replacement_pack_id = db.create_study_pack(
        title="Replacement Pack",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "replace-3"}]},
        generation_options_json={"deck_mode": "new"},
    )
    assert db.soft_delete_study_pack(replacement_pack_id, expected_version=1) is True  # nosec B101

    with pytest.raises(ConflictError, match="replacement"):
        db.supersede_study_pack(pack_id, superseded_by_pack_id=replacement_pack_id, expected_version=1)
