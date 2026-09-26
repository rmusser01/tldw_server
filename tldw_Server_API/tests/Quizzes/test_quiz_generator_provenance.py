import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.services.quiz_generator import (
    QuizProvenanceValidationError,
    _canonicalize_selected_source_citations,
    _validate_strict_provenance,
)

pytestmark = pytest.mark.unit


def test_rejects_questions_without_source_citations():
    with pytest.raises(QuizProvenanceValidationError, match="missing required source_citations"):
        _validate_strict_provenance(
            [{"question_text": "Q1", "source_citations": []}],
            [{"source_type": "note", "source_id": "n1"}],
        )


def test_rejects_citations_not_in_selected_sources():
    with pytest.raises(QuizProvenanceValidationError, match="do not map to selected sources"):
        _validate_strict_provenance(
            [{"source_citations": [{"source_type": "media", "source_id": "999"}]}],
            [{"source_type": "note", "source_id": "n1"}],
        )


def test_rejects_mixed_valid_and_invalid_citations():
    with pytest.raises(QuizProvenanceValidationError, match="do not map to selected sources"):
        _validate_strict_provenance(
            [
                {
                    "source_citations": [
                        {"source_type": "note", "source_id": "n1"},
                        {"source_type": "media", "source_id": "999"},
                    ]
                }
            ],
            [{"source_type": "note", "source_id": "n1"}],
        )


def test_accepts_valid_citations_for_selected_sources():
    _validate_strict_provenance(
        [{"source_citations": [{"source_type": "note", "source_id": "n1"}]}],
        [{"source_type": "note", "source_id": "n1"}],
    )


@given(
    source_type=st.sampled_from(["note", "media", "flashcard_deck", "flashcard_card"]),
    source_id=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789:-_", min_size=1, max_size=40),
)
def test_qualified_selected_ids_canonicalize_without_changing_identity(source_type, source_id):
    selected = [{"source_type": source_type, "source_id": source_id}]
    questions = [{"source_citations": [{"source_type": source_type, "source_id": f"{source_type}:{source_id}"}]}]

    _canonicalize_selected_source_citations(questions, selected)

    assert questions[0]["source_citations"][0]["source_id"] == source_id
    _validate_strict_provenance(questions, selected)


def test_exact_selected_id_containing_type_prefix_is_not_stripped():
    citation = {"source_type": "note", "source_id": "note:n1"}

    _canonicalize_selected_source_citations(
        [{"source_citations": [citation]}], [{"source_type": "note", "source_id": "note:n1"}]
    )

    assert citation["source_id"] == "note:n1"


def test_qualified_citation_cannot_change_source_type():
    questions = [{"source_citations": [{"source_type": "media", "source_id": "media:59"}]}]
    selected = [{"source_type": "note", "source_id": "59"}]

    _canonicalize_selected_source_citations(questions, selected)

    with pytest.raises(QuizProvenanceValidationError, match="do not map"):
        _validate_strict_provenance(questions, selected)


def test_qualified_media_citation_retains_numeric_media_reference():
    citation = {"source_type": "media", "source_id": "media:59"}

    _canonicalize_selected_source_citations(
        [{"source_citations": [citation]}], [{"source_type": "media", "source_id": "59"}]
    )

    assert citation == {"source_type": "media", "source_id": "59", "media_id": 59}
