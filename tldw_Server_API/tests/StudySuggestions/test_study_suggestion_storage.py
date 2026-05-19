from __future__ import annotations

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.StudySuggestions import snapshot_service
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-suggestions.db"), client_id="study-suggestion-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        username="admin",
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
    )


def test_create_suggestion_snapshot_defaults_to_active_status(db: CharactersRAGDB):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"summary": {"score": 7}, "topics": [{"display_label": "Renal basics"}]},
    )

    row = db.get_suggestion_snapshot(snapshot_id)

    assert row["service"] == "quiz"  # nosec B101
    assert row["status"] == "active"  # nosec B101
    assert row["user_selection_json"] is None  # nosec B101


def test_create_suggestion_snapshot_rejects_unknown_status(db: CharactersRAGDB):
    with pytest.raises(ValueError):
        db.create_suggestion_snapshot(
            service="quiz",
            activity_type="quiz_attempt",
            anchor_type="quiz_attempt",
            anchor_id=101,
            suggestion_type="study_suggestions",
            status="archived",
            payload_json={"summary": {"score": 7}},
        )


def test_suggestion_snapshot_payload_defaults_to_permission_safe_fields_only(db: CharactersRAGDB):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "summary": {"score": 7, "correct_count": 7},
            "topics": [
                {
                    "display_label": "Renal basics",
                    "source_type": "note",
                    "source_id": "note-7",
                    "selected": True,
                    "excerpt_text": "This long explanation should not be persisted.",
                    "quote_text": "A long quote that should not be stored by default.",
                    "rich_excerpt": {"markdown": "> quoted block"},
                    "unsafe_blob": "x" * 500,
                    "analysis_markdown": "## Detailed reasoning that should not persist",
                }
            ],
            "narrative": "A verbose explanation that is not a safe label/ref/count/flag field.",
            "quote_cache": ["remove this"],
        },
    )

    row = db.get_suggestion_snapshot(snapshot_id)
    payload = row["payload_json"]

    assert payload["summary"]["score"] == 7  # nosec B101
    assert payload["topics"][0]["display_label"] == "Renal basics"  # nosec B101
    assert payload["topics"][0]["source_id"] == "note-7"  # nosec B101
    assert payload["topics"][0]["selected"] is True  # nosec B101
    assert "excerpt_text" not in payload["topics"][0]  # nosec B101
    assert "quote_text" not in payload["topics"][0]  # nosec B101
    assert "rich_excerpt" not in payload["topics"][0]  # nosec B101
    assert "unsafe_blob" not in payload["topics"][0]  # nosec B101
    assert "analysis_markdown" not in payload["topics"][0]  # nosec B101
    assert "narrative" not in payload  # nosec B101
    assert "quote_cache" not in payload  # nosec B101


def test_suggestion_snapshot_payload_preserves_v2_topic_identity_fields(db: CharactersRAGDB):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "summary": {"score": 7},
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:renal-physiology",
                    "normalization_version": "norm-v2",
                    "display_label": "Renal Physiology",
                    "canonical_label": "renal physiology",
                    "source_count": 2,
                    "evidence_reasons": [
                        "missed_question",
                        "source_citation",
                        "tag_match",
                        "derived_label",
                        "this should not persist",
                    ],
                    "source_type": "note",
                    "source_id": "note-7",
                    "selected": True,
                    "excerpt_text": "This explanation should not be persisted.",
                    "analysis_markdown": "## This explanation should not be persisted either",
                }
            ],
        },
    )

    row = db.get_suggestion_snapshot(snapshot_id)
    topic = row["payload_json"]["topics"][0]

    assert topic["topic_key"] == "renal:renal-physiology"  # nosec B101
    assert topic["normalization_version"] == "norm-v2"  # nosec B101
    assert topic["source_count"] == 2  # nosec B101
    assert topic["evidence_reasons"] == [  # nosec B101
        "missed_question",
        "source_citation",
        "tag_match",
        "derived_label",
    ]
    assert topic["canonical_label"] == "renal physiology"  # nosec B101
    assert topic["source_id"] == "note-7"  # nosec B101
    assert "excerpt_text" not in topic  # nosec B101
    assert "analysis_markdown" not in topic  # nosec B101


def test_suggestion_snapshot_user_selection_survives_refresh_lineage_without_resubmission(db: CharactersRAGDB):
    original_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}]},
        user_selection_json={"selected_topic_ids": ["topic-1"]},
    )
    refreshed_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}, {"display_label": "Electrolytes"}]},
        refreshed_from_snapshot_id=original_id,
    )

    refreshed = db.get_suggestion_snapshot(refreshed_id)
    snapshots = db.list_suggestion_snapshots_for_anchor("quiz_attempt", 101)

    assert refreshed["refreshed_from_snapshot_id"] == original_id  # nosec B101
    assert refreshed["user_selection_json"] == {"selected_topic_ids": ["topic-1"]}  # nosec B101
    assert [row["id"] for row in snapshots] == [refreshed_id, original_id]  # nosec B101


def test_suggestion_snapshot_rejects_invalid_user_selection_json_string(db: CharactersRAGDB):
    with pytest.raises(InputError, match="user_selection_json"):
        db.create_suggestion_snapshot(
            service="quiz",
            activity_type="quiz_attempt",
            anchor_type="quiz_attempt",
            anchor_id=101,
            suggestion_type="study_suggestions",
            payload_json={"topics": [{"display_label": "Renal basics"}]},
            user_selection_json="{not-valid-json",
        )


def test_suggestion_snapshot_rejects_refresh_reference_from_different_lineage(db: CharactersRAGDB):
    original_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}]},
        user_selection_json={"selected_topic_ids": ["topic-1"]},
    )

    with pytest.raises(InputError, match="same suggestion lineage"):
        db.create_suggestion_snapshot(
            service="quiz",
            activity_type="quiz_attempt",
            anchor_type="quiz_attempt",
            anchor_id=202,
            suggestion_type="study_suggestions",
            payload_json={"topics": [{"display_label": "Electrolytes"}]},
            refreshed_from_snapshot_id=original_id,
        )


def test_generation_links_persist_one_row_per_snapshot_action_result(db: CharactersRAGDB):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}]},
    )

    first_link_id = db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-55",
        selection_fingerprint="sel-1",
    )
    second_link_id = db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id="deck-8",
        selection_fingerprint="sel-2",
    )

    first_row = db.find_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-55",
        selection_fingerprint="sel-1",
    )
    second_row = db.find_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id="deck-8",
        selection_fingerprint="sel-2",
    )
    count_row = db.execute_query(
        "SELECT COUNT(*) AS count FROM suggestion_generation_links WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()

    assert first_link_id != second_link_id  # nosec B101
    assert first_row["target_service"] == "quiz"  # nosec B101
    assert second_row["target_service"] == "flashcards"  # nosec B101
    assert count_row["count"] == 2  # nosec B101


def test_flashcard_review_session_rollup_preserves_provenance_fields(db: CharactersRAGDB):
    deck_id = db.add_deck("Grounded Deck", "desc")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter="renal basics",
        scope_key=f"due:deck:{deck_id}",
    )
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET source_bundle_json = ?,
               study_pack_id = ?
         WHERE id = ?
        """,
        (
            '[{"source_type":"note","source_id":"note-7","label":"Renal basics"}]',
            44,
            session["id"],
        ),
        commit=True,
    )

    rollup = db.get_flashcard_review_session_rollup(session["id"], repair_session_aggregates=False)

    assert rollup["study_pack_id"] == 44  # nosec B101
    assert rollup["source_bundle"] == [  # nosec B101
        {"source_type": "note", "source_id": "note-7", "label": "Renal basics"},
    ]


def test_flashcard_review_session_rollup_unwraps_study_pack_bundle_items_shape(db: CharactersRAGDB):
    deck_id = db.add_deck("Packed Grounded Deck", "desc")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter="renal basics",
        scope_key=f"due:deck:{deck_id}",
    )
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET source_bundle_json = ?
         WHERE id = ?
        """,
        (
            '{"items":[{"source_type":"note","source_id":"note-7","label":"Renal basics"}]}',
            session["id"],
        ),
        commit=True,
    )

    rollup = db.get_flashcard_review_session_rollup(session["id"], repair_session_aggregates=False)

    assert rollup["source_bundle"] == [  # nosec B101
        {"source_type": "note", "source_id": "note-7", "label": "Renal basics"},
    ]


def test_get_flashcard_reviewed_cards_returns_ordered_card_metadata(db: CharactersRAGDB):
    deck_id = db.add_deck("Reviewed Cards Deck", "desc")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )
    first_card = db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Front 1",
            "back": "Back 1",
            "tags": ["Renal basics"],
            "source_ref_type": "note",
            "source_ref_id": "note-7",
        }
    )
    second_card = db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Front 2",
            "back": "Back 2",
            "tags": ["Electrolytes"],
            "source_ref_type": "note",
            "source_ref_id": "note-8",
        }
    )

    db.review_flashcard(first_card, rating=4, answer_time_ms=500, review_session_id=int(session["id"]))
    db.review_flashcard(second_card, rating=2, answer_time_ms=650, review_session_id=int(session["id"]))

    reviewed_cards = db.get_flashcard_reviewed_cards(int(session["id"]))

    assert [card["uuid"] for card in reviewed_cards] == [first_card, second_card]  # nosec B101
    assert reviewed_cards[0]["deck_name"] == "Reviewed Cards Deck"  # nosec B101
    assert reviewed_cards[0]["source_ref_id"] == "note-7"  # nosec B101
    assert reviewed_cards[1]["source_ref_id"] == "note-8"  # nosec B101


def test_refresh_snapshot_serializes_quiz_v2_topic_fields(db: CharactersRAGDB):
    quiz_id = db.create_quiz(
        name="Renal Quiz",
        source_bundle_json=[{"source_type": "note", "source_id": "note-7", "label": "Renal basics"}],
    )
    question_id = db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which organ filters blood?",
        options=["Heart", "Kidney", "Lung"],
        correct_answer=1,
        explanation="The kidney filters blood.",
        tags=["Renal basics"],
        source_citations=[{"source_type": "note", "source_id": "note-7", "label": "Renal basics"}],
    )
    attempt = db.start_attempt(quiz_id)
    db.submit_attempt(
        attempt["id"],
        [
            {
                "question_id": question_id,
                "user_answer": 0,
                "time_spent_ms": 900,
            }
        ],
    )

    snapshot_id = snapshot_service.refresh_snapshot_for_anchor(
        note_db=db,
        anchor_type="quiz_attempt",
        anchor_id=int(attempt["id"]),
        principal=_admin_principal(),
    )
    row = db.get_suggestion_snapshot(snapshot_id)
    topic = row["payload_json"]["topics"][0]

    assert topic["topic_key"]  # nosec B101
    assert topic["normalization_version"] == "norm-v2"  # nosec B101
    assert topic["canonical_label"]  # nosec B101
    assert isinstance(topic["evidence_reasons"], list)  # nosec B101
    assert topic["evidence_reasons"]  # nosec B101
    assert "missed_question" in topic["evidence_reasons"]  # nosec B101
    assert "question_text" not in topic  # nosec B101
    assert "excerpt_text" not in topic  # nosec B101


def test_refresh_snapshot_serializes_semantic_canonical_label_for_alias_source_text(db: CharactersRAGDB):
    quiz_id = db.create_quiz(
        name="Alias Canonical Quiz",
        source_bundle_json=[{"source_type": "note", "source_id": "note-8", "label": "Kidney physiology"}],
    )
    question_id = db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which system handles filtration?",
        options=["Respiratory", "Renal", "Digestive"],
        correct_answer=1,
        explanation="The renal system handles filtration.",
        tags=[],
        source_citations=[{"source_type": "note", "source_id": "note-8", "label": "Kidney physiology"}],
    )
    attempt = db.start_attempt(quiz_id)
    db.submit_attempt(
        attempt["id"],
        [
            {
                "question_id": question_id,
                "user_answer": 0,
                "time_spent_ms": 700,
            }
        ],
    )

    snapshot_id = snapshot_service.refresh_snapshot_for_anchor(
        note_db=db,
        anchor_type="quiz_attempt",
        anchor_id=int(attempt["id"]),
        principal=_admin_principal(),
    )
    topic = db.get_suggestion_snapshot(snapshot_id)["payload_json"]["topics"][0]

    assert topic["topic_key"] == "renal:renal-physiology"  # nosec B101
    assert topic["canonical_label"] == "renal physiology"  # nosec B101
    assert topic["display_label"] == "Kidney Physiology"  # nosec B101
    assert topic["source_id"] == "note-8"  # nosec B101


def test_quiz_snapshot_assigns_source_refs_only_to_matching_topics(db: CharactersRAGDB):
    quiz_id = db.create_quiz(
        name="Mixed Quiz",
        source_bundle_json=[{"source_type": "note", "source_id": "note-7", "label": "Cited source"}],
    )
    question_id = db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which label should stay tag-only?",
        options=["Tagged only", "Cited source", "Both"],
        correct_answer=1,
        explanation="The citation and tag should remain separate topics.",
        tags=["Tagged only"],
        source_citations=[{"source_type": "note", "source_id": "note-7", "label": "Cited source"}],
    )
    attempt = db.start_attempt(quiz_id)
    db.submit_attempt(
        attempt["id"],
        [
            {
                "question_id": question_id,
                "user_answer": 0,
                "time_spent_ms": 850,
            }
        ],
    )

    snapshot_id = snapshot_service.refresh_snapshot_for_anchor(
        note_db=db,
        anchor_type="quiz_attempt",
        anchor_id=int(attempt["id"]),
        principal=_admin_principal(),
    )
    payload = db.get_suggestion_snapshot(snapshot_id)["payload_json"]
    topics_by_label = {
        topic["display_label"]: topic
        for topic in payload["topics"]
        if isinstance(topic, dict) and isinstance(topic.get("display_label"), str)
    }

    assert topics_by_label["Cited Source"]["source_id"] == "note-7"  # nosec B101
    assert "source_id" not in topics_by_label["Tagged Only"]  # nosec B101


def test_suggestion_generation_link_duplicate_is_conflict_but_fk_failure_is_not(db: CharactersRAGDB):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}]},
    )

    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-55",
        selection_fingerprint="sel-1",
    )

    with pytest.raises(ConflictError, match="already exists"):
        db.create_suggestion_generation_link(
            snapshot_id=snapshot_id,
            target_service="quiz",
            target_type="quiz",
            target_id="quiz-55",
            selection_fingerprint="sel-1",
        )

    with pytest.raises(Exception) as exc_info:
        db.create_suggestion_generation_link(
            snapshot_id=999999,
            target_service="quiz",
            target_type="quiz",
            target_id="quiz-56",
            selection_fingerprint="sel-fk",
        )

    assert not isinstance(exc_info.value, ConflictError)  # nosec B101


def test_suggestion_generation_link_duplicate_identity_ignores_target_id_for_active_rows(
    db: CharactersRAGDB,
):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}]},
    )

    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-55",
        selection_fingerprint="sel-1",
    )

    with pytest.raises(ConflictError, match="already exists"):
        db.create_suggestion_generation_link(
            snapshot_id=snapshot_id,
            target_service="quiz",
            target_type="quiz",
            target_id="quiz-99",
            selection_fingerprint="sel-1",
        )


def test_replace_suggestion_generation_link_returns_retained_link_id_when_updating_existing_row(
    db: CharactersRAGDB,
):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}]},
    )
    original_link_id = db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-55",
        selection_fingerprint="sel-1",
    )

    updated_link_id = db.replace_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-99",
        selection_fingerprint="sel-1",
    )
    row = db.find_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-99",
        selection_fingerprint="sel-1",
    )

    assert updated_link_id == original_link_id  # nosec B101
    assert row is not None  # nosec B101
    assert int(row["id"]) == original_link_id  # nosec B101


def test_suggestion_generation_link_schema_dedupes_legacy_active_duplicates_before_unique_index(
    db: CharactersRAGDB,
):
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Renal basics"}]},
    )

    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-55",
        selection_fingerprint="sel-1",
    )
    db.execute_query("DROP INDEX IF EXISTS idx_suggestion_generation_links_unique_active", commit=True)

    now = db._get_current_utc_timestamp_iso()
    insert_sql = """
        INSERT INTO suggestion_generation_links(
            snapshot_id, target_service, target_type, target_id, selection_fingerprint,
            created_at, last_modified, deleted, client_id, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    db.execute_query(
        insert_sql,
        (
            snapshot_id,
            "quiz",
            "quiz",
            "pending:sel-1",
            "sel-1",
            now,
            now,
            False,
            "legacy-test",
            1,
        ),
        commit=True,
    )
    db.execute_query(
        insert_sql,
        (
            snapshot_id,
            "quiz",
            "quiz",
            "quiz-99",
            "sel-1",
            now,
            now,
            False,
            "legacy-test",
            1,
        ),
        commit=True,
    )

    with db.transaction() as conn:
        db._ensure_study_pack_schema_sqlite(conn)

    rows = [
        dict(row)
        for row in db.execute_query(
            """
            SELECT target_id, deleted
              FROM suggestion_generation_links
             WHERE snapshot_id = ? AND target_service = ? AND target_type = ? AND selection_fingerprint = ?
             ORDER BY id
            """,
            (snapshot_id, "quiz", "quiz", "sel-1"),
        ).fetchall()
    ]
    active_rows = [row for row in rows if not row["deleted"]]

    assert [row["target_id"] for row in active_rows] == ["quiz-99"]  # nosec B101

    with pytest.raises(ConflictError, match="already exists"):
        db.create_suggestion_generation_link(
            snapshot_id=snapshot_id,
            target_service="quiz",
            target_type="quiz",
            target_id="quiz-100",
            selection_fingerprint="sel-1",
        )


def test_study_pack_postgres_schema_dedupes_before_unique_index_recreation() -> None:
    executed_statements: list[str] = []

    class RecordingBackend:
        def execute(self, statement, connection=None):  # noqa: ANN001
            executed_statements.append(" ".join(str(statement).split()))
            return None

    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db.backend = RecordingBackend()
    db._dedupe_suggestion_generation_links_postgres = lambda conn: executed_statements.append("__DEDUPE__")  # type: ignore[method-assign]

    db._ensure_study_pack_schema_postgres(conn=object())

    dedupe_index = executed_statements.index("__DEDUPE__")
    drop_index = next(
        index
        for index, statement in enumerate(executed_statements)
        if "DROP INDEX IF EXISTS idx_suggestion_generation_links_unique_active" in statement
    )
    create_index = next(
        index
        for index, statement in enumerate(executed_statements)
        if "CREATE UNIQUE INDEX IF NOT EXISTS idx_suggestion_generation_links_unique_active" in statement
    )

    assert dedupe_index < drop_index < create_index  # nosec B101


def test_suggestion_storage_tables_include_sync_and_version_fields(db: CharactersRAGDB):
    snapshot_columns = {
        column["name"]
        for column in db.backend.get_table_info("suggestion_snapshots")
    }
    link_columns = {
        column["name"]
        for column in db.backend.get_table_info("suggestion_generation_links")
    }

    assert {"created_at", "last_modified", "deleted", "client_id", "version"} <= snapshot_columns  # nosec B101
    assert {"created_at", "last_modified", "deleted", "client_id", "version"} <= link_columns  # nosec B101


@pytest.mark.integration
def test_suggestion_storage_postgres_round_trip_and_integrity(pg_database_config: DatabaseConfig):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="study-suggestion-pg", backend=backend)

    try:
        snapshot_id = db.create_suggestion_snapshot(
            service="quiz",
            activity_type="quiz_attempt",
            anchor_type="quiz_attempt",
            anchor_id=101,
            suggestion_type="study_suggestions",
            payload_json={"topics": [{"display_label": "Renal basics", "source_id": "note-1"}]},
            user_selection_json={"selected_topic_ids": ["topic-1"]},
        )

        row = db.get_suggestion_snapshot(snapshot_id)
        assert row is not None  # nosec B101
        assert row["user_selection_json"] == {"selected_topic_ids": ["topic-1"]}  # nosec B101

        db.create_suggestion_generation_link(
            snapshot_id=snapshot_id,
            target_service="quiz",
            target_type="quiz",
            target_id="quiz-55",
            selection_fingerprint="sel-1",
        )

        with pytest.raises(ConflictError, match="already exists"):
            db.create_suggestion_generation_link(
                snapshot_id=snapshot_id,
                target_service="quiz",
                target_type="quiz",
                target_id="quiz-55",
                selection_fingerprint="sel-1",
            )

        with pytest.raises(Exception) as exc_info:
            db.create_suggestion_generation_link(
                snapshot_id=999999,
                target_service="quiz",
                target_type="quiz",
                target_id="quiz-56",
                selection_fingerprint="sel-fk",
            )

        assert not isinstance(exc_info.value, ConflictError)  # nosec B101
    finally:
        try:
            db.close_connection()
            if db.backend_type.name == "POSTGRESQL":
                db.backend.get_pool().close_all()
        except Exception:
            _ = None
