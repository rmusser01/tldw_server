from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import (
    NotesGraphFTSNotReadyError,
    NotesGraphSourceTooLargeError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.suggestion_retrieval import (
    MAX_CANDIDATES,
    MAX_RETRIEVAL_TERMS,
    MAX_TAG_CATALOG,
    RETRIEVAL_OVERFETCH,
    SuggestionRetriever,
    derive_retrieval_terms,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "suggestion-retrieval.db"), client_id="owner-1")
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (database.client_id, "dataset-1"),
        )
    yield database
    database.close_connection()


def _note(
    db: CharactersRAGDB,
    note_id: str,
    title: str,
    content: str,
    *,
    conversation_id: str | None = None,
) -> str:
    return str(db.add_note(title, content, note_id=note_id, conversation_id=conversation_id))


def _id(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def test_derive_retrieval_terms_prioritizes_title_then_frequency_with_stable_ties() -> None:
    terms = derive_retrieval_terms(
        "Zeta alpha alpha", "beta beta beta gamma gamma delta-theory and the alpha"
    )

    assert terms[:2] == ("alpha", "zeta")
    assert terms[2:] == ("beta", "gamma", "delta-theory")
    assert len(terms) <= MAX_RETRIEVAL_TERMS


def test_derive_retrieval_terms_is_bounded_to_24() -> None:
    title = " ".join(f"title{i}" for i in range(40))

    assert len(derive_retrieval_terms(title, "")) == MAX_RETRIEVAL_TERMS


def test_retrieval_excludes_only_selected_trash_and_direct_links(db: CharactersRAGDB) -> None:
    character_id = db.add_character_card({"name": "Shared source"})
    conversation_id = db.add_conversation(
        {"character_id": character_id, "title": "Shared source", "source": "youtube", "external_ref": "video-1"}
    )
    source = _note(
        db, _id(1), "Neural retrieval", "lexical graph analysis", conversation_id=conversation_id
    )
    eligible = _note(db, _id(2), "Graph retrieval", "neural lexical evidence")
    shared_tag = _note(db, _id(3), "Neural index", "retrieval evidence")
    shared_source = _note(
        db, _id(4), "Graph index", "neural retrieval", conversation_id=conversation_id
    )
    manual = _note(db, _id(5), "Neural graph", "retrieval evidence")
    wikilink = _note(db, _id(6), "Retrieval link", "neural graph")
    trashed = _note(db, _id(7), "Neural trash", "retrieval graph")
    db.create_manual_note_edge(
        user_id="owner-1", from_note_id=source, to_note_id=manual,
        directed=False, weight=1.0, created_by="test",
    )
    db.update_note(source, {"content": f"lexical graph analysis [[id:{wikilink}]]"}, expected_version=1)
    keyword_id = db.add_keyword("shared")
    db.link_note_to_keyword(source, keyword_id)
    db.link_note_to_keyword(shared_tag, keyword_id)
    db.soft_delete_note(trashed, expected_version=1)

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id="dataset-1", source_note_id=source
    )

    ids = [candidate.note_id for candidate in result.candidates]
    assert eligible in ids
    assert shared_tag in ids
    assert shared_source in ids
    assert source not in ids
    assert manual not in ids
    assert wikilink not in ids
    assert trashed not in ids


def test_retrieval_uses_60_row_overfetch_then_prunes_to_30_in_stable_rank_order(
    db: CharactersRAGDB,
) -> None:
    source = _note(db, _id(100), "alpha", "alpha")
    for number in range(65):
        _note(db, _id(200 + number), "alpha", "alpha")

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id="dataset-1", source_note_id=source
    )

    assert result.backend_overfetch_count == RETRIEVAL_OVERFETCH
    assert len(result.candidates) == MAX_CANDIDATES
    assert [candidate.note_id for candidate in result.candidates] == sorted(
        candidate.note_id for candidate in result.candidates
    )


def test_retrieval_caps_tag_catalog_and_reports_projection_freshness(db: CharactersRAGDB) -> None:
    source = _note(db, _id(300), "retrieval", "graph")
    for number in range(101):
        db.add_keyword(f"retrieval-tag-{number:03d}")

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id="dataset-1", source_note_id=source
    )

    assert len(result.tag_catalog) == MAX_TAG_CATALOG
    assert result.projection_fresh is True


def test_retrieval_excludes_oversized_candidates_and_counts_them(db: CharactersRAGDB) -> None:
    source = _note(db, _id(400), "retrieval", "graph")
    eligible = _note(db, _id(401), "retrieval", "graph")
    _note(db, _id(402), "retrieval", "x" * 250_001)

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id="dataset-1", source_note_id=source
    )

    assert [candidate.note_id for candidate in result.candidates] == [eligible]
    assert result.excluded_oversized_candidate_count == 1


def test_retrieval_rejects_an_oversized_selected_note_without_truncation(db: CharactersRAGDB) -> None:
    source = _note(db, _id(500), "retrieval", "x" * 1_000_001)

    with pytest.raises(NotesGraphSourceTooLargeError, match="notes_graph_source_too_large"):
        SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
            dataset_id="dataset-1", source_note_id=source
        )


def test_retrieval_estimates_only_bounded_evidence_windows(db: CharactersRAGDB) -> None:
    source = _note(db, _id(600), "retrieval", "source " * 20_000)
    _note(db, _id(601), "retrieval", "candidate " * 20_000)

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id="dataset-1", source_note_id=source
    )

    assert result.estimated_input_tokens <= 24_000


def test_retrieval_rejects_an_unbound_dataset_without_disclosing_notes(db: CharactersRAGDB) -> None:
    source = _note(db, _id(700), "retrieval", "graph")

    with pytest.raises(RuntimeError, match="notes_graph_dataset_scope_invalid"):
        SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
            dataset_id="dataset-not-authorized", source_note_id=source
        )


def test_retrieval_fails_closed_when_a_notes_fts_trigger_definition_drifts(db: CharactersRAGDB) -> None:
    source = _note(db, _id(701), "retrieval", "graph")
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER notes_au")
        conn.execute(
            "CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN SELECT 1; END"
        )

    with pytest.raises(NotesGraphFTSNotReadyError, match="notes_graph_fts_not_ready"):
        SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
            dataset_id="dataset-1", source_note_id=source
        )


def test_retrieval_fails_closed_when_notes_fts_is_not_the_expected_virtual_table(
    db: CharactersRAGDB,
) -> None:
    source = _note(db, _id(702), "retrieval", "graph")
    with db.transaction() as conn:
        for trigger_name in ("notes_ai", "notes_au", "notes_ad"):
            conn.execute(f"DROP TRIGGER {trigger_name}")
        conn.execute("DROP TABLE notes_fts")
        conn.execute("CREATE TABLE notes_fts(title TEXT, content TEXT)")
        for trigger_name in ("notes_ai", "notes_au", "notes_ad"):
            conn.execute(
                f"CREATE TRIGGER {trigger_name} AFTER UPDATE ON notes BEGIN SELECT 1; END"
            )

    with pytest.raises(NotesGraphFTSNotReadyError, match="notes_graph_fts_not_ready"):
        SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
            dataset_id="dataset-1", source_note_id=source
        )
