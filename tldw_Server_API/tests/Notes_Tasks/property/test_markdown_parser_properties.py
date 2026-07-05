"""Property tests for the note markdown checklist parser (RA4, plan Task 7).

``parse_note_checklists`` is parse-only (no renderer, so no round-trip). The
real invariants: total (never raises on arbitrary content), deterministic,
every item's locator offsets stay within the source and slice back to the raw
line, and locators are monotone / occurrence-indexed consistently.
"""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists

pytestmark = [pytest.mark.unit, pytest.mark.property]

_COMMON = hyp_settings(max_examples=200, deadline=None)

# text likely to contain checklist lines, fences, indentation, unicode
_line = st.one_of(
    st.just("- [ ] todo item"),
    st.just("- [x] done item"),
    st.just("  - [ ] nested"),
    st.just("* [ ] star bullet"),
    st.just("```"),
    st.just("plain paragraph"),
    st.just(""),
    st.text(max_size=40),
)
_content = st.lists(_line, max_size=30).map(lambda lines: "\n".join(lines))


@_COMMON
@given(content=_content)
def test_parser_is_total(content: str) -> None:
    result = parse_note_checklists(note_id="n1", note_version=1, content=content)
    assert result is not None
    assert isinstance(result.items, list)


@_COMMON
@given(content=_content)
def test_parser_is_deterministic(content: str) -> None:
    a = parse_note_checklists(note_id="n1", note_version=1, content=content)
    b = parse_note_checklists(note_id="n1", note_version=1, content=content)
    assert [i.locator for i in a.items] == [i.locator for i in b.items]
    assert [i.text for i in a.items] == [i.text for i in b.items]


@_COMMON
@given(content=_content)
def test_locator_offsets_stay_within_source_and_slice_the_raw_line(content: str) -> None:
    result = parse_note_checklists(note_id="n1", note_version=1, content=content)
    n = len(content)
    for item in result.items:
        loc = item.locator
        assert 0 <= loc.start_offset <= loc.end_offset <= n, "offsets escape the source bounds"
        # the span must slice back to exactly the raw line the item came from
        assert content[loc.start_offset:loc.end_offset] == item.raw_line


@_COMMON
@given(content=_content)
def test_locators_are_strictly_increasing(content: str) -> None:
    result = parse_note_checklists(note_id="n1", note_version=1, content=content)
    line_numbers = [i.locator.line_number for i in result.items]
    start_offsets = [i.locator.start_offset for i in result.items]
    assert line_numbers == sorted(line_numbers)
    assert start_offsets == sorted(start_offsets)
    # distinct items occupy distinct lines
    assert len(set(line_numbers)) == len(line_numbers)


@_COMMON
@given(content=_content)
def test_occurrence_index_increments_per_normalized_text(content: str) -> None:
    result = parse_note_checklists(note_id="n1", note_version=1, content=content)
    seen: dict[str, int] = {}
    for item in result.items:
        key = item.locator.normalized_text_hash
        seen[key] = seen.get(key, 0) + 1
        assert item.locator.occurrence_index == seen[key], "occurrence_index out of sequence"
