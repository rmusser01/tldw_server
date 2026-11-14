"""Tests for wikilink_parser.extract_wikilinks()."""

import uuid

import pytest

from tldw_Server_API.app.core.Notes.wikilinks import (
    MAX_WIKILINK_TARGETS,
    WIKILINK_PARSER_VERSION,
    WikilinkTitleCandidate,
    normalize_wikilink_title,
    parse_wikilinks,
    select_wikilink_title_target,
    wikilink_title_reference_key,
)
from tldw_Server_API.app.core.Notes_Graph.wikilink_parser import (
    WikilinkRef,
    extract_wikilinks,
)

pytestmark = pytest.mark.unit


class TestExtractWikilinks:
    """Core extraction tests."""

    def test_empty_input(self):
        assert extract_wikilinks("") == []

    def test_none_like_empty(self):
        # Empty string, whitespace only
        assert extract_wikilinks("   ") == []

    def test_no_links(self):
        assert extract_wikilinks("Just some plain text with no links.") == []

    def test_single_link(self):
        content = "See [[id:a1b2c3d4-e5f6-7890-abcd-ef1234567890]] for details."
        result = extract_wikilinks(content)
        assert result == [WikilinkRef(target_note_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890")]

    def test_multiple_links(self):
        content = (
            "Ref [[id:11111111-1111-1111-1111-111111111111]] and "
            "[[id:22222222-2222-2222-2222-222222222222]] here."
        )
        result = extract_wikilinks(content)
        assert len(result) == 2
        assert result[0].target_note_id == "11111111-1111-1111-1111-111111111111"
        assert result[1].target_note_id == "22222222-2222-2222-2222-222222222222"

    def test_dedup(self):
        content = (
            "[[id:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee]] and again "
            "[[id:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee]]"
        )
        result = extract_wikilinks(content)
        assert len(result) == 1

    def test_dedup_case_insensitive(self):
        content = (
            "[[id:AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE]] and "
            "[[id:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee]]"
        )
        result = extract_wikilinks(content)
        assert len(result) == 1
        assert result[0].target_note_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_case_normalization_to_lower(self):
        content = "[[id:AABBCCDD-1122-3344-5566-778899AABBCC]]"
        result = extract_wikilinks(content)
        assert result[0].target_note_id == "aabbccdd-1122-3344-5566-778899aabbcc"

    def test_malformed_uuid_rejected(self):
        # Too short
        assert extract_wikilinks("[[id:abc]]") == []
        # Missing dashes
        assert extract_wikilinks("[[id:a1b2c3d4e5f67890abcdef1234567890]]") == []
        # Wrong segment lengths
        assert extract_wikilinks("[[id:a1b2c3d4-e5f6-7890-abcd-ef12345678]]") == []

    def test_title_style_is_not_an_id_ref(self):
        """[[Title]] links need owner-scoped resolution, so the id-only helper skips them."""
        assert extract_wikilinks("[[My Note Title]]") == []
        assert extract_wikilinks("[[Some-Other-Note]]") == []

    def test_mixed_valid_and_invalid(self):
        content = (
            "Valid: [[id:11111111-2222-3333-4444-555555555555]] "
            "Invalid: [[title:Foo]] [[id:short]] "
            "Valid2: [[id:66666666-7777-8888-9999-aaaaaaaaaaaa]]"
        )
        result = extract_wikilinks(content)
        assert len(result) == 2
        assert result[0].target_note_id == "11111111-2222-3333-4444-555555555555"
        assert result[1].target_note_id == "66666666-7777-8888-9999-aaaaaaaaaaaa"

    def test_link_at_start_and_end(self):
        content = "[[id:11111111-1111-1111-1111-111111111111]]text[[id:22222222-2222-2222-2222-222222222222]]"
        result = extract_wikilinks(content)
        assert len(result) == 2

    def test_multiline_content(self):
        content = "Line1\n[[id:11111111-1111-1111-1111-111111111111]]\nLine3\n[[id:22222222-2222-2222-2222-222222222222]]"
        result = extract_wikilinks(content)
        assert len(result) == 2

    def test_frozen_dataclass(self):
        ref = WikilinkRef(target_note_id="test-id")
        with pytest.raises(AttributeError):
            ref.target_note_id = "other"


def test_projection_parser_is_bounded_deterministic_and_omits_self_links() -> None:
    source_id = str(uuid.UUID(int=1))
    distinct = [str(uuid.UUID(int=index)) for index in range(1, MAX_WIKILINK_TARGETS + 3)]
    content = " ".join(
        [f"[[id:{source_id}]]", *[f"[[id:{value.upper()}]]" for value in distinct], f"[[id:{distinct[1]}]]"]
    )

    projection = parse_wikilinks(content, source_note_id=source_id)

    assert projection.parser_version == WIKILINK_PARSER_VERSION
    assert projection.truncated is True
    assert len(projection.target_note_ids) == MAX_WIKILINK_TARGETS
    assert source_id not in projection.target_note_ids
    assert projection.target_note_ids[:2] == (distinct[1], distinct[2])
    assert parse_wikilinks(content, source_note_id=source_id) == projection


def test_legacy_parser_is_a_compatibility_reexport() -> None:
    target_id = str(uuid.UUID(int=44))
    projection = parse_wikilinks(f"[[id:{target_id}]]")

    assert extract_wikilinks(f"[[id:{target_id}]]") == [
        WikilinkRef(target_note_id=projection.target_note_ids[0])
    ]


# ---------------------------------------------------------------------------
# NE-02 (#3110), decision D2: both ``[[Title]]`` and ``[[id:UUID]]`` are links.
# ---------------------------------------------------------------------------


def test_parser_version_is_bumped_for_title_links() -> None:
    # Title links change what the parser projects, so existing projections must rebuild.
    assert WIKILINK_PARSER_VERSION >= 2


def test_title_links_are_collected_in_first_occurrence_order() -> None:
    target_id = str(uuid.UUID(int=7))
    projection = parse_wikilinks(
        f"See [[Beta]], [[id:{target_id}]] and [[ Alpha   Note ]] then [[beta]] again."
    )

    assert projection.target_note_ids == (target_id,)
    # Whitespace is collapsed, and a repeated title (any case) is one target.
    assert projection.target_titles == ("Beta", "Alpha Note")
    assert projection.truncated is False


def test_title_links_may_contain_single_brackets() -> None:
    projection = parse_wikilinks("Read [[[Draft] Proposal]] and [[Q3 [final]]].")

    assert projection.target_titles == ("[Draft] Proposal", "Q3 [final]")


def test_nested_open_brackets_link_the_innermost_title() -> None:
    projection = parse_wikilinks("[[outer [[Inner]] tail]]")

    assert projection.target_titles == ("Inner",)


def test_title_links_do_not_span_lines_or_stay_empty() -> None:
    projection = parse_wikilinks("[[Broken\nTitle]] [[   ]] [[]]")

    assert projection.target_titles == ()
    assert projection.target_note_ids == ()


def test_reserved_id_prefix_never_falls_back_to_title() -> None:
    # A malformed id link is not silently treated as a note titled "id:short".
    projection = parse_wikilinks("[[id:short]] [[id:]]")

    assert projection == parse_wikilinks("")


def test_title_and_id_links_share_the_target_bound() -> None:
    ids = [f"[[id:{uuid.UUID(int=index)}]]" for index in range(1, 4)]
    titles = [f"[[Title {index}]]" for index in range(1, 4)]

    projection = parse_wikilinks(" ".join([ids[0], titles[0], ids[1], titles[1], ids[2], titles[2]]), max_targets=4)

    assert projection.truncated is True
    assert len(projection.target_note_ids) + len(projection.target_titles) == 4
    assert projection.target_note_ids == (str(uuid.UUID(int=1)), str(uuid.UUID(int=2)))
    assert projection.target_titles == ("Title 1", "Title 2")


def test_title_normalization_matches_the_client_rule() -> None:
    assert normalize_wikilink_title("  Weekly   SYNC \t notes ") == "weekly sync notes"
    assert normalize_wikilink_title("") == ""


def test_title_reference_key_is_stable_and_case_insensitive() -> None:
    key = wikilink_title_reference_key("Weekly  Sync")

    assert key == wikilink_title_reference_key("weekly sync")
    assert key != wikilink_title_reference_key("Weekly Sync 2")
    # Reference keys can never collide with canonical UUID targets.
    with pytest.raises(ValueError):
        uuid.UUID(key)


def _candidate(note_id: str, title: str, created_at: str) -> WikilinkTitleCandidate:
    return WikilinkTitleCandidate(note_id=note_id, title=title, created_at=created_at)


def test_select_target_is_case_and_whitespace_insensitive() -> None:
    candidates = [_candidate("n-1", "Target  Note", "2026-01-01T00:00:00Z")]

    assert select_wikilink_title_target("target note", candidates) == "n-1"
    assert select_wikilink_title_target("Other", candidates) is None


def test_select_target_prefers_exact_title_then_oldest_then_lowest_id() -> None:
    candidates = [
        _candidate("n-3", "shared", "2026-01-01T00:00:00Z"),
        _candidate("n-2", "Shared", "2026-03-01T00:00:00Z"),
        _candidate("n-1", "Shared", "2026-03-01T00:00:00Z"),
        _candidate("n-0", "Shared", "2026-02-01T00:00:00Z"),
    ]

    # An exact-case match wins over an older case-insensitive match.
    assert select_wikilink_title_target("Shared", candidates) == "n-0"
    # Without an exact match the oldest note wins; equal ages fall back to the lowest id.
    assert select_wikilink_title_target("SHARED", candidates) == "n-3"
    assert select_wikilink_title_target("SHARED", candidates[1:3]) == "n-1"
