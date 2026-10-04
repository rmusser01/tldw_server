"""Direct compatibility contracts for hierarchical joins and grouping."""

from copy import deepcopy
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Chunking.hierarchical.grouping import (
    group_items_by_elements,
    group_section_by_kind_weight,
    merge_texts,
)

pytestmark = pytest.mark.unit


def item(text: str, kind: str = "paragraph", **metadata: Any) -> dict[str, Any]:
    """Build an input element with caller-owned metadata."""
    return {"type": "text", "text": text, "metadata": {"paragraph_kind": kind, **metadata}}


@pytest.mark.parametrize("parts,expected", [([], ""), ([("only\n", {})], "only\n")])
def test_merge_empty_and_single(parts: list[tuple[str, dict[str, Any]]], expected: str) -> None:
    assert merge_texts(parts, method="words") == expected


@pytest.mark.parametrize(
    "kind,method,left,right,expected",
    [
        ("paragraph", "words", "a", "b", "a b"),
        ("paragraph", "words", "a ", "b", "a b"),
        ("paragraph", "words", "a", " b", "a b"),
        ("paragraph", "structure_aware", "a", "b", "a\n\nb"),
        ("paragraph", "structure_aware", "a\n", "b", "a\n\nb"),
        ("paragraph", "structure_aware", "a\n\n", "b", "a\n\nb"),
        ("header_atx", "words", "a", "b", "a \n\nb"),
        ("header_atx", "words", "a\n", "b", "a \n\nb"),
        ("hr", "words", "a", "b", "a\n\nb"),
        ("list_unordered", "words", "a\n", "b", "a\nb"),
        ("list_ordered", "words", "a", "b", "a\nb"),
        ("table_md", "words", "a", "b", "a\nb"),
        ("code_fence", "words", "a", "b", "a\nb"),
    ],
)
def test_merge_separators(kind: str, method: str, left: str, right: str, expected: str) -> None:
    assert merge_texts([(left, {}), (right, {"paragraph_kind": kind})], method=method) == expected


@pytest.mark.parametrize("language", ["zh", "ZH-CN", "zh-tw", "ja", "th"])
@pytest.mark.parametrize("method", ["words", "structure_aware"])
def test_no_space_language_from_previous_metadata(language: str, method: str) -> None:
    assert merge_texts([("a", {"language": language}), ("b", {})], method=method) == "ab"


@pytest.mark.parametrize("kind,expected", [("header_atx", "a\n\nb"), ("code_fence", "a\nb")])
def test_no_space_language_keeps_structural_separator(kind: str, expected: str) -> None:
    assert merge_texts([("a", {}), ("b", {"language": "ja"})], method="words", kind_hint=kind) == expected


def test_merge_kind_hint_overrides_metadata_and_default_separator() -> None:
    assert (
        merge_texts(
            [("a\n", {}), ("b", {"paragraph_kind": "code_fence"})], method=None, default_sep="|", kind_hint="header_atx"
        )
        == "a \n\nb"
    )


@pytest.mark.parametrize("limit", [None, 0, -1])
@pytest.mark.parametrize("weighted", [False, True])
def test_disabled_grouping_returns_original_list(limit: int | None, weighted: bool) -> None:
    items = [item("a", nested={"values": [1]})]
    result = (
        group_section_by_kind_weight(items, method="words", max_weight=limit, overlap=1, weights={})
        if weighted
        else group_items_by_elements(items, method="words", max_elements=limit, overlap=1)
    )
    assert result is items


@pytest.mark.parametrize(
    "overlap,expected",
    [
        (0, ["a b c", "d e"]),
        (-2, ["a b c", "d e"]),
        (1, ["a b c", "c d e"]),
        (3, ["a b c", "b c d", "c d e"]),
        (10, ["a b c", "b c d", "c d e"]),
    ],
)
def test_element_windows_progress_and_suppress_overlap_only_tail(overlap: int, expected: list[str]) -> None:
    result = group_items_by_elements([item(t) for t in "abcde"], method="words", max_elements=3, overlap=overlap)
    assert [group["text"] for group in result] == expected


@pytest.mark.parametrize("weighted", [False, True])
def test_group_offsets_conversion_and_fresh_metadata(weighted: bool) -> None:
    items = [
        item("a", start_offset="9", end_offset="12"),
        item("b", start_offset=2, end_offset=20),
        item("c", start_offset="bad", end_offset=[]),
    ]
    result = (
        group_section_by_kind_weight(items, method="words", max_weight=3, overlap=0, weights={})
        if weighted
        else group_items_by_elements(items, method="words", max_elements=3, overlap=0)
    )
    expected = {"method": "words", "start_offset": 2, "end_offset": 20, "grouped_elements": 3}
    if weighted:
        expected["group_kind"] = "paragraph"
    assert result == [{"type": "text", "text": "a b c", "metadata": expected}]
    assert result[0] is not items[0]
    assert all(result[0]["metadata"] is not source["metadata"] for source in items)


@pytest.mark.parametrize("weighted", [False, True])
def test_group_missing_offsets_use_text_length(weighted: bool) -> None:
    items = [item("a"), item("b", start_offset="bad", end_offset="bad")]
    result = (
        group_section_by_kind_weight(items, method="words", max_weight=2, overlap=0, weights={})
        if weighted
        else group_items_by_elements(items, method="words", max_elements=2, overlap=0)
    )
    assert (result[0]["metadata"]["start_offset"], result[0]["metadata"]["end_offset"]) == (0, 3)


def test_weighted_groups_do_not_cross_kind_boundaries() -> None:
    result = group_section_by_kind_weight(
        [item("a"), item("b"), item("c", "code_fence"), item("d", "code_fence"), item("e")],
        method="words",
        max_weight=3,
        overlap=0,
        weights={"code_fence": 2},
    )
    assert [(g["text"], g["metadata"]["group_kind"], g["metadata"]["grouped_elements"]) for g in result] == [
        ("a b", "paragraph", 2),
        ("c", "code_fence", 1),
        ("d", "code_fence", 1),
        ("e", "paragraph", 1),
    ]


@pytest.mark.parametrize("weights", [{}, {"paragraph": 0}, {"paragraph": -2}, {"paragraph": "1"}])
def test_default_and_nonpositive_weights(weights: dict[str, int | str]) -> None:
    assert [
        g["text"]
        for g in group_section_by_kind_weight(
            [item(t) for t in "abc"], method="words", max_weight=2, overlap=0, weights=weights
        )
    ] == ["a b", "c"]


@pytest.mark.parametrize(
    "overlap,expected", [(-1, ["a b", "c"]), (0, ["a b", "c"]), (1, ["a b", "b c", "c"]), (20, ["a b", "b c", "c"])]
)
def test_weighted_overlap_clamps_and_progresses(overlap: int, expected: list[str]) -> None:
    assert [
        g["text"]
        for g in group_section_by_kind_weight(
            [item(t) for t in "abc"], method="words", max_weight=2, overlap=overlap, weights={}
        )
    ] == expected


def test_overweight_item_still_consumed() -> None:
    assert [
        g["text"]
        for g in group_section_by_kind_weight(
            [item("a"), item("b")], method="words", max_weight=1, overlap=5, weights={"paragraph": 3}
        )
    ] == ["a", "b"]


def test_invalid_string_weight_propagates() -> None:
    with pytest.raises(ValueError):
        group_section_by_kind_weight(
            [item("a")], method="words", max_weight=2, overlap=0, weights={"paragraph": "invalid"}
        )


@pytest.mark.parametrize("weighted", [False, True])
def test_grouping_does_not_mutate_input_or_nested_metadata(weighted: bool) -> None:
    nested = {"values": [1, 2]}
    items = [item("a", nested=nested), item("b", nested=nested)]
    originals = list(items)
    metadata = [it["metadata"] for it in items]
    before = deepcopy(items)
    if weighted:
        group_section_by_kind_weight(items, method="structure_aware", max_weight=2, overlap=1, weights={})
    else:
        group_items_by_elements(items, method="structure_aware", max_elements=2, overlap=1)
    merge_texts([(it["text"], it["metadata"]) for it in items], method="words")
    assert items == before
    assert all(
        it is original and it["metadata"] is md and md["nested"] is nested
        for it, original, md in zip(items, originals, metadata)
    )


@given(st.integers(min_value=0, max_value=30), st.integers(min_value=1, max_value=10))
def test_nonoverlapping_element_groups_preserve_element_count(count: int, limit: int) -> None:
    result = group_items_by_elements(
        [item(str(i)) for i in range(count)], method="words", max_elements=limit, overlap=0
    )
    assert sum(g["metadata"]["grouped_elements"] for g in result) == count


@pytest.mark.parametrize("by_kind", [False, True])
def test_flatten_uses_call_time_grouping_helpers(monkeypatch: pytest.MonkeyPatch, by_kind: bool) -> None:
    from tldw_Server_API.app.core.Chunking import Chunker
    from tldw_Server_API.app.core.Chunking.hierarchical import flatten as module

    calls = []

    def merge(parts: list[tuple[str, dict[str, Any]]], *, method: str, default_sep: str, kind_hint: str) -> str:
        calls.append(("merge", method, default_sep, kind_hint))
        return "merged"

    def group(items: list[dict[str, Any]], *, method: str, overlap: int, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append(("group", method, overlap, kwargs))
        assert items[0]["text"] == "merged"
        return items

    monkeypatch.setattr(module, "merge_texts", merge)
    monkeypatch.setattr(module, "group_section_by_kind_weight" if by_kind else "group_items_by_elements", group)
    tree = {
        "method": "structure_aware",
        "max_size": 2,
        "overlap": 1,
        "grouping": {"by_kind": by_kind, "element_weights": {"paragraph": 1}},
        "root": {"children": [{"kind": "section", "children": [{"chunks": [item("# h", "header_atx"), item("a")]}]}]},
    }
    Chunker().flatten_hierarchical(tree)
    assert calls == [
        ("merge", "structure_aware", "\n\n", "header_atx"),
        (
            "group",
            "structure_aware",
            1,
            {"max_weight": 2, "weights": {"paragraph": 1}} if by_kind else {"max_elements": 2},
        ),
    ]
