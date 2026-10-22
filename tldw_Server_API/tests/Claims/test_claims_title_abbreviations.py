"""Complete title-bearing Claims sentences without changing source offsets."""

import pytest

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
    HeuristicSentenceExtractor,
)
from tldw_Server_API.app.core.Claims_Extraction.extractor_catalog import (
    split_claims_sentences,
)
from tldw_Server_API.app.core.Claims_Extraction.ingestion_claims import (
    extract_claims_for_chunks,
    store_claims,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("title", ["Mr", "Mrs", "Dr", "Ms", "Prof"])
@pytest.mark.parametrize("case", [str.upper, str.lower, str.title], ids=["upper", "lower", "title"])
def test_title_keeps_person_name_in_claim(title, case):
    claim = f"Its director is {case(title)}. Mira Vale."
    following = "The telescope is named Selene."

    assert split_claims_sentences(f"{claim} {following}", "en") == [claim, following]  # nosec B101 # Pytest regression assertion, not an application guard.


@pytest.mark.parametrize("word", ["ADR", "AlDr", "ChiefProf", "OtherMrs"])
def test_title_suffix_inside_word_still_ends_sentence(word):
    first = f"The record identifies {word}."
    second = "another sentence begins here."

    assert split_claims_sentences(f"{first} {second}", "en") == [first, second]  # nosec B101 # Pytest regression assertion, not an application guard.


@pytest.mark.parametrize("ending", [".", "!", "?", "…", "؟"])
def test_ordinary_punctuation_retains_existing_boundaries(ending):
    first = f"The observatory opened in 2019{ending}"
    second = "the telescope is named Selene."

    assert split_claims_sentences(f"{first}\n{second}", "en") == [first, second]  # nosec B101 # Pytest regression assertion, not an application guard.


def test_title_sentence_keeps_internal_whitespace():
    claim = "Its director is Dr.\tMira\nVale."

    assert split_claims_sentences(f"  {claim}  ", "en") == [claim]  # nosec B101 # Pytest regression assertion, not an application guard.


def test_title_sentence_respects_filtered_claim_budget():
    first = "Its director is Dr. Mira Vale."
    second = "The telescope is named Selene."

    assert split_claims_sentences(f"Tiny. {first} {second}", "en", max_sentences=1) == [first]  # nosec B101 # Pytest regression assertion, not an application guard.


def test_explicit_minimum_still_filters_complete_title_sentence():
    assert split_claims_sentences("Its director is Dr. Mira Vale.", "en", min_length=40) == []  # nosec B101 # Pytest regression assertion, not an application guard.


def test_decimal_and_lowercase_sentence_semantics_remain():
    first = "The measured value is 3.14."
    second = "another measurement is pending."

    assert split_claims_sentences(f"{first} {second}", "en") == [first, second]  # nosec B101 # Pytest regression assertion, not an application guard.


@pytest.mark.parametrize("language", ["zh", "ja", "ko", "th"])
def test_no_space_language_punctuation_is_unchanged(language):
    text = "这是一个比较长的中文句子。这里还有另一个比较长的句子！"

    assert split_claims_sentences(text, language) == [  # nosec B101 # Pytest regression assertion, not an application guard.
        "这是一个比较长的中文句子。",
        "这里还有另一个比较长的句子！",
    ]


@pytest.mark.parametrize("text", ["", "  \n\t", "Short."])
def test_empty_and_short_inputs_remain_filtered(text):
    assert split_claims_sentences(text, "en") == []  # nosec B101 # Pytest regression assertion, not an application guard.


def test_ingestion_preserves_complete_title_and_exact_chunk_offsets():
    text = "Opening year: 2019.  Its director is Dr.\tMira Vale.  The telescope is named Selene."
    chunks = [{"text": text, "metadata": {"chunk_index": 7}}]

    claims = extract_claims_for_chunks(chunks, extractor_mode="heuristic", max_per_chunk=3, language="en")

    assert [(row["chunk_index"], row["claim_text"], row["span"]) for row in claims] == [  # nosec B101 # Pytest regression assertion, not an application guard.
        (7, "Opening year: 2019.", [0, 19]),
        (7, "Its director is Dr.\tMira Vale.", [21, 51]),
        (7, "The telescope is named Selene.", [53, 83]),
    ]


def test_each_chunk_retains_its_own_claim_budget():
    text = "Its director is Dr. Mira Vale. The telescope is named Selene."
    chunks = [
        {"text": text, "metadata": {"chunk_index": 4}},
        {"text": text, "metadata": {"chunk_index": 9}},
    ]

    claims = extract_claims_for_chunks(chunks, extractor_mode="heuristic", max_per_chunk=1, language="en")

    assert [(row["chunk_index"], row["claim_text"], row["span"]) for row in claims] == [  # nosec B101 # Pytest regression assertion, not an application guard.
        (4, "Its director is Dr. Mira Vale.", [0, 30]),
        (9, "Its director is Dr. Mira Vale.", [0, 30]),
    ]


@pytest.mark.asyncio
async def test_answer_extractor_uses_complete_title_sentence():
    claims = await HeuristicSentenceExtractor().extract(
        "Its director is Dr. Mira Vale. The telescope is named Selene.", max_claims=1
    )

    assert [(claim.id, claim.text) for claim in claims] == [("c1", "Its director is Dr. Mira Vale.")]  # nosec B101 # Pytest regression assertion, not an application guard.


def test_stored_claim_retains_full_text_and_source_span(tmp_path):
    text = "Its director is Dr.\tMira Vale."
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="title_claim_test")
    db.initialize_db()
    try:
        media_id, _, _ = db.add_media_with_keywords(title="Observatory", media_type="text", content=text, keywords=None)
        claims = extract_claims_for_chunks(
            [{"text": text, "metadata": {"chunk_index": 0}}],
            extractor_mode="heuristic",
            max_per_chunk=1,
            language="en",
        )
        inserted = store_claims(db, media_id=media_id, chunk_texts_by_index={0: text}, claims=claims)
        rows = db.get_claims_by_media(media_id)

        assert (inserted, [(row["claim_text"], row["span_start"], row["span_end"]) for row in rows]) == (  # nosec B101 # Pytest regression assertion, not an application guard.
            1,
            [(text, 0, 30)],
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize("suffix", ["Jr", "Sr"])
def test_name_suffix_still_ends_sentence(suffix):
    first = f"The director is Rowan Vale {suffix}."
    second = "The telescope is named Selene."

    assert split_claims_sentences(f"{first} {second}", "en") == [first, second]  # nosec B101 # Pytest sentence-boundary regression.


@pytest.mark.parametrize("suffix", ["Jr", "Sr"])
def test_suffix_claims_keep_source_offsets(suffix):
    text = f"The director is Rowan Vale {suffix}. The telescope is named Selene."
    chunks = [{"text": text, "metadata": {"chunk_index": 2}}]

    claims = extract_claims_for_chunks(chunks, extractor_mode="heuristic", max_per_chunk=2, language="en")

    assert [(row["claim_text"], row["span"]) for row in claims] == [  # nosec B101 # Pytest source-span regression.
        (f"The director is Rowan Vale {suffix}.", [0, 30]),
        ("The telescope is named Selene.", [31, 61]),
    ]
    limited = extract_claims_for_chunks(chunks, extractor_mode="heuristic", max_per_chunk=1, language="en")
    assert [(row["claim_text"], row["span"]) for row in limited] == [  # nosec B101 # Pytest chunk-budget regression.
        (f"The director is Rowan Vale {suffix}.", [0, 30]),
    ]


@pytest.mark.parametrize("suffix", ["Jr", "Sr"])
@pytest.mark.asyncio
async def test_answer_budget_stops_at_name_suffix(suffix):
    first = f"The director is Rowan Vale {suffix}."
    claims = await HeuristicSentenceExtractor().extract(f"{first} The telescope is named Selene.", max_claims=1)

    assert [(claim.id, claim.text) for claim in claims] == [("c1", first)]  # nosec B101 # Pytest answer-budget regression.
