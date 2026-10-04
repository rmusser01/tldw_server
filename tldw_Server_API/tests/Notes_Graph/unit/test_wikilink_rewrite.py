"""Rewriting ``[[Old title]]`` links after a rename (#3110, follow-up to NE-02).

The rewrite must agree with the parser about what a link is: it replaces
exactly the tokens ``parse_wikilinks`` reads as a link to the old title and
never touches other text. Undo puts the original tokens back, byte for byte.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Notes.wikilinks import (
    MAX_WIKILINK_TITLE_LENGTH,
    WikilinkTokenReplacement,
    is_single_wikilink,
    iter_wikilink_tokens,
    normalize_wikilink_title,
    parse_wikilinks,
    restore_wikilink_tokens,
    rewrite_wikilink_title_tokens,
    wikilink_id_link_text,
    wikilink_title_link_text,
)

pytestmark = pytest.mark.unit

NOTE_ID = "11111111-1111-4111-8111-111111111111"
NEW_LINK = "[[New title]]"


def _rewrite(content: str, old_title: str = "Old title", replacement: str = NEW_LINK):
    return rewrite_wikilink_title_tokens(content, old_title=old_title, replacement=replacement)


# --- which tokens are rewritten -------------------------------------------


def test_rewrites_an_exact_title_link() -> None:
    content, replaced = _rewrite("See [[Old title]] for details.")

    assert content == "See [[New title]] for details."
    assert replaced == (WikilinkTokenReplacement(token_index=0, original="[[Old title]]"),)


def test_matches_like_the_parser_ignoring_case_and_extra_whitespace() -> None:
    content, replaced = _rewrite("a [[old TITLE]] b [[  Old   title ]] c [[Old\ttitle]]")

    assert content == "a [[New title]] b [[New title]] c [[New title]]"
    assert [item.original for item in replaced] == [
        "[[old TITLE]]",
        "[[  Old   title ]]",
        "[[Old\ttitle]]",
    ]


@pytest.mark.parametrize(
    "content",
    [
        "[[Old title 2]]",
        "[[Old titles]]",
        "[[The Old title]]",
        "[[Old title notes]]",
        "[[Old]] [[title]]",
        "[[Old-title]]",
    ],
)
def test_similar_titles_are_left_alone(content: str) -> None:
    assert _rewrite(content) == (content, ())


@pytest.mark.parametrize(
    "content",
    [
        "Old title appears as plain text.",
        "[Old title] in single brackets",
        "[Old title](https://example.test/old-title) is a Markdown link",
        "# Old title\n\nA heading is not a link.",
        "[[Old\ntitle]] spans a newline, so the parser does not read it as a link",
        "[[Old title] is unterminated",
    ],
)
def test_the_title_outside_a_wikilink_is_never_touched(content: str) -> None:
    assert _rewrite(content) == (content, ())


def test_a_title_inside_a_longer_link_is_not_a_substring_match() -> None:
    # The parser reads these as the titles "[Old title" and "Old title]".
    content = "[[[Old title]] and [[Old title]]]"

    assert [token.title for token in iter_wikilink_tokens(content)] == ["[Old title", "Old title]"]
    assert _rewrite(content) == (content, ())


def test_id_links_are_left_alone() -> None:
    content = f"[[id:{NOTE_ID}]] and [[ID:{NOTE_ID}]] and [[id:Old title]] and [[Old title]]"

    rewritten, replaced = _rewrite(content)

    # ``[[ID:...]]`` is a title to the parser, and a malformed ``id:`` link is not a link at all.
    assert rewritten == f"[[id:{NOTE_ID}]] and [[ID:{NOTE_ID}]] and [[id:Old title]] and [[New title]]"
    assert replaced == (WikilinkTokenReplacement(token_index=3, original="[[Old title]]"),)


def test_a_reserved_id_prefix_old_title_matches_nothing() -> None:
    # A note titled "id:draft" can't be linked by title: the prefix is reserved.
    content = "[[id:draft]]"

    assert _rewrite(content, old_title="id:draft") == (content, ())


def test_alias_syntax_is_not_supported_so_it_is_a_different_title() -> None:
    # The parser has no ``[[Title|label]]`` form: the whole text is the title.
    content = "[[Old title|shown text]] and [[Old title#Section]]"

    assert parse_wikilinks(content).target_titles == ("Old title|shown text", "Old title#Section")
    assert _rewrite(content) == (content, ())


def test_links_in_code_are_links_to_the_parser_so_they_are_rewritten_too() -> None:
    content = "Inline `[[Old title]]` and fenced:\n```\n[[Old title]]\n```\n"

    # Mirror the parser: it projects an edge for a link inside code.
    assert parse_wikilinks(content).target_titles == ("Old title",)
    rewritten, replaced = _rewrite(content)

    assert rewritten == "Inline `[[New title]]` and fenced:\n```\n[[New title]]\n```\n"
    assert len(replaced) == 2


def test_over_long_titles_are_not_links() -> None:
    long_title = "x" * (MAX_WIKILINK_TITLE_LENGTH + 1)
    content = f"[[{long_title}]]"

    assert parse_wikilinks(content).target_titles == ()
    assert _rewrite(content, old_title=long_title) == (content, ())


def test_blank_old_title_matches_nothing() -> None:
    assert _rewrite("[[ ]] [[Old title]]", old_title="   ") == ("[[ ]] [[Old title]]", ())


def test_every_matching_token_is_rewritten_and_the_rest_is_byte_identical() -> None:
    content = "α [[Old title]]\n\n- [[Other]] — [[old title]]\r\n\t[[id:" + NOTE_ID + "]] ✓ [[Old title]]"

    rewritten, replaced = _rewrite(content)

    assert rewritten == (
        "α [[New title]]\n\n- [[Other]] — [[New title]]\r\n\t[[id:" + NOTE_ID + "]] ✓ [[New title]]"
    )
    assert [(item.token_index, item.original) for item in replaced] == [
        (0, "[[Old title]]"),
        (2, "[[old title]]"),
        (4, "[[Old title]]"),
    ]


def test_rewrites_to_an_id_link() -> None:
    id_link = wikilink_id_link_text(NOTE_ID)

    rewritten, replaced = _rewrite("See [[Old title]].", replacement=id_link)

    assert id_link == f"[[id:{NOTE_ID}]]"
    assert rewritten == f"See [[id:{NOTE_ID}]]."
    assert parse_wikilinks(rewritten).target_note_ids == (NOTE_ID,)
    assert len(replaced) == 1


@pytest.mark.parametrize("replacement", ["New title", "[[New]] and more", "[[a]][[b]]", "", "[[id:not-a-uuid]]"])
def test_the_replacement_must_be_exactly_one_link(replacement: str) -> None:
    with pytest.raises(ValueError, match="replacement"):
        _rewrite("[[Old title]]", replacement=replacement)


# --- link text for the renamed note ---------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("New title", "[[New title]]"),
        ("  New   title ", "[[New title]]"),
        ("[Draft] Proposal", "[[[Draft] Proposal]]"),
        ("Ends with ]", "[[Ends with ]]]"),
        ("a | b", "[[a | b]]"),
    ],
)
def test_title_link_text_round_trips_through_the_parser(title: str, expected: str) -> None:
    text = wikilink_title_link_text(title)

    assert text == expected
    assert [normalize_wikilink_title(found) for found in parse_wikilinks(text).target_titles] == [
        normalize_wikilink_title(title)
    ]


@pytest.mark.parametrize(
    "title",
    ["", "   ", "a]]b", "a[[b", "id:draft", "id:" + NOTE_ID, "x" * (MAX_WIKILINK_TITLE_LENGTH + 1)],
)
def test_titles_a_link_cannot_name_have_no_title_link_text(title: str) -> None:
    assert wikilink_title_link_text(title) is None


def test_id_link_text_needs_a_canonical_uuid() -> None:
    assert wikilink_id_link_text(NOTE_ID.upper()) == f"[[id:{NOTE_ID}]]"
    assert wikilink_id_link_text("notes-note-0123456789abcdef0123456789abcdef") is None
    assert wikilink_id_link_text(None) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[[New title]]", True),
        (f"[[id:{NOTE_ID}]]", True),
        ("[[id:not-a-uuid]]", False),
        ("[[New title]] and more", False),
        ("[[a]][[b]]", False),
        ("New title", False),
        ("", False),
        (None, False),
    ],
)
def test_is_single_wikilink(text: str | None, expected: bool) -> None:
    assert is_single_wikilink(text) is expected


# --- undo ------------------------------------------------------------------


def test_restore_puts_back_the_original_text_exactly() -> None:
    original = "a [[old   TITLE]] b [[New title]] c [[ Old title ]]"
    rewritten, replaced = _rewrite(original)

    # The note already linked the new title; undo must not touch that link.
    assert rewritten == "a [[New title]] b [[New title]] c [[New title]]"
    assert (
        restore_wikilink_tokens(rewritten, replaced, old_title="Old title", replacement=NEW_LINK)
        == original
    )


def test_restore_refuses_when_the_text_no_longer_holds_the_rewritten_link() -> None:
    _rewritten, replaced = _rewrite("x [[Old title]] y")

    for edited in ("x [[Something else]] y", "x y", "[[Extra]] x [[New title]] y"):
        assert (
            restore_wikilink_tokens(edited, replaced, old_title="Old title", replacement=NEW_LINK)
            is None
        )


@pytest.mark.parametrize(
    "original",
    ["arbitrary text", "[[Another title]]", "[[Old title]] trailing", "[[id:" + NOTE_ID + "]]", ""],
)
def test_restore_only_writes_links_to_the_old_title(original: str) -> None:
    replacements = (WikilinkTokenReplacement(token_index=0, original=original),)

    assert (
        restore_wikilink_tokens("[[New title]]", replacements, old_title="Old title", replacement=NEW_LINK)
        is None
    )


def test_restore_rejects_repeated_or_missing_token_indexes() -> None:
    twice = (
        WikilinkTokenReplacement(token_index=0, original="[[Old title]]"),
        WikilinkTokenReplacement(token_index=0, original="[[old title]]"),
    )
    beyond = (WikilinkTokenReplacement(token_index=5, original="[[Old title]]"),)

    for replacements in (twice, beyond, ()):
        assert (
            restore_wikilink_tokens(
                "[[New title]]", replacements, old_title="Old title", replacement=NEW_LINK
            )
            is None
        )


_FRAGMENTS = st.sampled_from(
    [
        "[[Old title]]",
        "[[old  TITLE]]",
        "[[ Old title ]]",
        "[[New title]]",
        "[[Old title 2]]",
        "[[[Old title]]",
        "[[Old title]]]",
        f"[[id:{NOTE_ID}]]",
        "[[id:Old title]]",
        "[[Old\ntitle]]",
        "[[",
        "]]",
        "[",
        "]",
        "Old title",
        " ",
        "\n",
        "`",
        "é",
        "|",
    ]
)
_REPLACEMENTS = st.sampled_from(
    [NEW_LINK, f"[[id:{NOTE_ID}]]", "[[[Draft] New]]", "[[New ]]]", "[[Old title 2]]"]
)


@settings(max_examples=300, deadline=None)
@given(fragments=st.lists(_FRAGMENTS, max_size=12), replacement=_REPLACEMENTS)
def test_rewrite_then_restore_is_the_identity(fragments: list[str], replacement: str) -> None:
    content = "".join(fragments)
    before = list(iter_wikilink_tokens(content))

    rewritten, replaced = rewrite_wikilink_title_tokens(
        content, old_title="Old title", replacement=replacement
    )
    after = list(iter_wikilink_tokens(rewritten))

    replaced_indexes = {item.token_index for item in replaced}
    old_key = normalize_wikilink_title("Old title")
    # Exactly the parser's links to the old title are replaced ...
    assert replaced_indexes == {
        token.index for token in before if token.title is not None and token.title.lower() == old_key
    }
    # ... every other token is untouched, and nothing shifts.
    assert len(after) == len(before)
    for old_token, new_token in zip(before, after):
        expected = replacement if old_token.index in replaced_indexes else old_token.raw
        assert new_token.raw == expected
    if not replaced:
        assert rewritten == content
    else:
        assert (
            restore_wikilink_tokens(rewritten, replaced, old_title="Old title", replacement=replacement)
            == content
        )
