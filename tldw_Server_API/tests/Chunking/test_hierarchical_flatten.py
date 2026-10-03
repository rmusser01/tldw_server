"""Direct traversal, malformed-input, and aliasing contracts for flattening."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tldw_Server_API.app.core.Chunking.hierarchical import flatten as module
from tldw_Server_API.app.core.Chunking.hierarchical.flatten import flatten_tree


def normalize(value: Any) -> str | None:
    return {"paragraph": "text", "header_atx": "heading", "code_fence": "code"}.get(value)


def item(text: str, kind: str = "paragraph", **metadata: Any) -> dict[str, Any]:
    return {"text": text, "metadata": {"paragraph_kind": kind, **metadata}}


@pytest.mark.parametrize(
    ("tree", "expected_texts", "error", "message"),
    [
        ({}, [], None, ""),
        ({"root": None, "blocks": [{"chunks": [item("legacy")]}]}, ["legacy"], None, ""),
        ({"root": {}, "blocks": [{"chunks": [item("legacy")]}]}, ["legacy"], None, ""),
        ({"root": "invalid"}, [], AttributeError, "'str' object has no attribute 'get'"),
        ({"root": {"children": ["skip", 3, None]}}, [], None, ""),
        ({"root": {"children": "skip"}}, [], None, ""),
        ({"root": {"children": 3}}, [], TypeError, "'int' object is not iterable"),
        ({"root": {"chunks": "ab"}}, ["a", "b"], None, ""),
        ({"root": {"chunks": [3, None, {"metadata": None}]}}, ["3", "None", None], None, ""),
        ({"root": {"chunks": 3}}, [], TypeError, "'int' object is not iterable"),
        (
            {"root": {"chunks": [{"text": "bad", "metadata": "invalid"}]}},
            [],
            ValueError,
            "dictionary update sequence element #0",
        ),
        ({"root": {"chunks": [{"text": "pairs", "metadata": [("key", "value")]}]}}, ["pairs"], None, ""),
        ({"root": {"chunks": None, "children": None}}, [], None, ""),
        (
            {"method": "structure_aware", "max_size": "2", "root": {"kind": "section", "chunks": [item("ungrouped")]}},
            ["ungrouped"],
            None,
            "",
        ),
        (
            {
                "method": "structure_aware",
                "max_size": 2,
                "root": {"kind": "section", "children": [{"chunks": ["skip", item("valid")]}]},
            },
            ["valid"],
            None,
            "",
        ),
        (
            {
                "method": "structure_aware",
                "max_size": 2,
                "grouping": {"by_kind": True, "element_weights": {"paragraph": "invalid"}},
                "root": {"kind": "section", "children": [{"chunks": [item("bad")]}]},
            },
            [],
            ValueError,
            r"invalid literal for int\(\)",
        ),
    ],
    ids=[
        "empty",
        "falsey-root",
        "empty-root",
        "invalid-root",
        "skip-children",
        "string-children",
        "invalid-children",
        "string-chunks",
        "scalar-chunks",
        "invalid-chunks",
        "invalid-metadata",
        "metadata-pairs",
        "falsey-lists",
        "uncoerced-size",
        "skip-group-rows",
        "invalid-weight",
    ],
)
def test_malformed_dictionary_matrix(tree, expected_texts, error, message) -> None:
    if error is not None:
        with pytest.raises(error, match=message):
            flatten_tree(tree, normalize)
    else:
        rows = flatten_tree(tree, normalize)
        assert [row["text"] for row in rows] == expected_texts
        assert [row["metadata"]["chunk_index"] for row in rows] == list(range(1, len(rows) + 1))
        assert all(row["metadata"]["total_chunks"] == len(rows) for row in rows)


@pytest.mark.parametrize("tree", [None, "invalid", []])
def test_direct_function_does_not_own_public_non_dictionary_guard(tree) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'get'"):
        flatten_tree(tree, normalize)


@pytest.mark.parametrize("kind", ["root", "section"])
@pytest.mark.parametrize("title", ["Section", "", "   "])
def test_input_metadata_and_shared_ancestry_identity(kind: str, title: str) -> None:
    nested = {"values": []}
    first_metadata = {"paragraph_kind": "paragraph", "nested": nested, "chunk_index": 41, "total_chunks": 99}
    second_metadata = {"paragraph_kind": "paragraph", "nested": nested}
    tree = {
        "method": "words",
        "root": {
            "kind": kind,
            "title": title,
            "chunks": [{"text": "first", "metadata": first_metadata}, {"text": "second", "metadata": second_metadata}],
            "children": [],
        },
    }
    original_tree = copy.deepcopy(tree)

    rows = flatten_tree(tree, normalize)

    assert tree == original_tree
    assert first_metadata == original_tree["root"]["chunks"][0]["metadata"]
    assert rows[0]["metadata"] is not first_metadata
    assert rows[1]["metadata"] is not second_metadata
    assert rows[0]["metadata"]["nested"] is nested
    assert rows[1]["metadata"]["nested"] is nested
    assert rows[0]["metadata"]["ancestry_titles"] is rows[1]["metadata"]["ancestry_titles"]
    titles = [title] if kind == "section" and title.strip() else []
    assert rows[0]["metadata"]["ancestry_titles"] == titles
    if titles:
        assert rows[0]["metadata"]["section_path"] == title
    else:
        assert "section_path" not in rows[0]["metadata"]
    assert rows[0]["metadata"]["chunk_index"] == 41
    assert rows[0]["metadata"]["total_chunks"] == 99
    assert rows[1]["metadata"]["chunk_index"] == 2
    assert rows[1]["metadata"]["total_chunks"] == 2


def test_nested_titles_and_empty_title_do_not_duplicate_content() -> None:
    tree = {
        "root": {
            "kind": "section",
            "title": " Outer ",
            "chunks": [item("outer")],
            "children": [
                {
                    "kind": "section",
                    "title": " ",
                    "chunks": [item("empty")],
                    "children": [{"kind": "section", "title": "Inner", "chunks": [item("inner")]}],
                }
            ],
        }
    }
    rows = flatten_tree(tree, normalize)
    assert [row["text"] for row in rows] == ["outer", "empty", "inner"]
    assert [row["metadata"]["ancestry_titles"] for row in rows] == [["Outer"], ["Outer"], ["Outer", "Inner"]]
    assert [row["metadata"]["section_path"] for row in rows] == ["Outer", "Outer", "Outer > Inner"]
    assert rows[0]["metadata"]["ancestry_titles"] is not rows[1]["metadata"]["ancestry_titles"]


@pytest.mark.parametrize("chunk_type", [None, "", "code_fence", "unknown"])
def test_normalizer_receives_chunk_type_or_paragraph_kind_each_call(chunk_type) -> None:
    metadata = {"paragraph_kind": "paragraph"}
    if chunk_type is not None:
        metadata["chunk_type"] = chunk_type
    tree = {"root": {"chunks": [{"text": "body", "metadata": metadata}]}}
    calls = []

    def callback(value):
        calls.append(value)
        return "first" if len(calls) == 1 else None

    assert flatten_tree(tree, callback)[0]["metadata"]["chunk_type"] == "first"
    second = flatten_tree(tree, callback)[0]["metadata"]
    assert second.get("chunk_type") == metadata.get("chunk_type")
    assert calls == [chunk_type or "paragraph"] * 2


@pytest.mark.parametrize("by_kind", [False, True])
def test_header_buffering_and_nested_section_grouping(by_kind: bool) -> None:
    tree = {
        "method": "structure_aware",
        "max_size": 2,
        "overlap": 0,
        "grouping": {"by_kind": by_kind},
        "root": {
            "kind": "section",
            "title": "Outer",
            "children": [
                {"chunks": [item("# H", "header_atx", start_offset=0, end_offset=3)]},
                {"chunks": [item("a", start_offset=4, end_offset=5), item("b", start_offset=6, end_offset=7)]},
                {
                    "kind": "section",
                    "title": "Inner",
                    "children": [{"chunks": [item("c", start_offset=8, end_offset=9)]}],
                },
            ],
        },
    }
    original = copy.deepcopy(tree)
    rows = flatten_tree(tree, normalize)
    assert tree == original
    assert [row["text"] for row in rows] == ["# H \n\na\n\nb", "c"]
    assert [row["metadata"]["grouped_elements"] for row in rows] == [2, 1]
    assert [(row["metadata"]["start_offset"], row["metadata"]["end_offset"]) for row in rows] == [(0, 7), (8, 9)]
    assert [row["metadata"]["section_path"] for row in rows] == ["Outer", "Outer > Inner"]
    if by_kind:
        assert [row["metadata"]["group_kind"] for row in rows] == ["paragraph", "paragraph"]


@pytest.mark.parametrize("by_kind", [False, True])
def test_header_only_section_preserves_header_content(by_kind: bool) -> None:
    tree = {
        "method": "structure_aware",
        "max_size": 2,
        "grouping": {"by_kind": by_kind},
        "root": {"kind": "section", "title": "Only", "children": [{"chunks": [item("# Only", "header_atx")]}]},
    }
    rows = flatten_tree(tree, normalize)
    assert [row["text"] for row in rows] == ["# Only"]
    assert rows[0]["metadata"]["grouped_elements"] == 1
    assert rows[0]["metadata"]["section_path"] == "Only"
    if by_kind:
        assert rows[0]["metadata"]["group_kind"] == "header_atx"


def test_by_kind_grouping_respects_kind_boundaries_and_weights() -> None:
    tree = {
        "method": "structure_aware",
        "max_size": 2,
        "grouping": {"by_kind": True, "element_weights": {"code_fence": 2}},
        "root": {
            "kind": "section",
            "children": [{"chunks": [item("a"), item("b"), item("code1", "code_fence"), item("code2", "code_fence")]}],
        },
    }
    rows = flatten_tree(tree, normalize)
    assert [row["text"] for row in rows] == ["a\n\nb", "code1", "code2"]
    assert [row["metadata"]["group_kind"] for row in rows] == ["paragraph", "code_fence", "code_fence"]


def test_buffered_header_metadata_copy_preserves_target_nested_identity(monkeypatch) -> None:
    nested = {"values": []}
    target = item("body", nested=nested, start_offset=4, end_offset=8)
    header = item("# H", "header_atx", start_offset=0, end_offset=3)
    gathered = []

    def group(items, **_kwargs):
        gathered.extend(items)
        return items

    monkeypatch.setattr(module, "group_items_by_elements", group)
    tree = {
        "method": "structure_aware",
        "max_size": 2,
        "root": {"kind": "section", "children": [{"chunks": [header, target]}]},
    }
    original = copy.deepcopy(tree)
    rows = flatten_tree(tree, normalize)
    assert tree == original
    assert gathered[0] is not target
    assert gathered[0]["metadata"] is not target["metadata"]
    assert rows[0]["metadata"] is not gathered[0]["metadata"]
    assert rows[0]["metadata"]["nested"] is nested
    assert rows[0]["metadata"]["has_section_header"] is True
    assert rows[0]["metadata"]["paragraph_kind"] == "paragraph"
    assert (rows[0]["metadata"]["start_offset"], rows[0]["metadata"]["end_offset"]) == (0, 8)
