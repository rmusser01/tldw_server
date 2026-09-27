"""Tests for wikilink_parser.extract_wikilinks()."""

import uuid

import pytest

from tldw_Server_API.app.core.Notes.wikilinks import (
    MAX_WIKILINK_TARGETS,
    WIKILINK_PARSER_VERSION,
    parse_wikilinks,
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

    def test_title_style_not_matched(self):
        """[[Title]] syntax should NOT be matched."""
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
