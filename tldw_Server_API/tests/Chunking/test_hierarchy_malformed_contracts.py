"""Malformed-input and aliasing contracts for hierarchy flattening."""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking import flatten_hierarchical as package_flatten


@pytest.fixture(params=["public", "package"])
def flatten(request: pytest.FixtureRequest) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    if request.param == "public":
        return Chunker().flatten_hierarchical
    return package_flatten


def test_falsey_root_uses_legacy_blocks(
    flatten: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    tree = {
        "root": None,
        "blocks": [
            {
                "kind": "paragraph",
                "chunks": [
                    {
                        "text": "legacy",
                        "metadata": {"paragraph_kind": "paragraph"},
                    }
                ],
            }
        ],
    }

    assert flatten(tree) == [
        {
            "text": "legacy",
            "metadata": {
                "paragraph_kind": "paragraph",
                "ancestry_titles": [],
                "chunk_type": "text",
                "chunk_index": 1,
                "total_chunks": 1,
            },
        }
    ]


def test_truthy_non_mapping_root_raises_attribute_error(
    flatten: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    with pytest.raises(
        AttributeError,
        match="'str' object has no attribute 'get'",
    ):
        flatten({"root": "invalid"})


def test_non_mapping_root_children_are_skipped(
    flatten: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    assert flatten({"root": {"children": ["skip", 3, None]}}) == []


def test_string_chunks_are_iterated_as_individual_rows(
    flatten: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    assert flatten({"root": {"chunks": "ab"}}) == [
        {
            "text": "a",
            "metadata": {
                "ancestry_titles": [],
                "chunk_index": 1,
                "total_chunks": 2,
            },
        },
        {
            "text": "b",
            "metadata": {
                "ancestry_titles": [],
                "chunk_index": 2,
                "total_chunks": 2,
            },
        },
    ]


def test_invalid_chunk_metadata_public_raises_and_package_suppresses() -> None:
    tree = {"root": {"chunks": [{"text": "bad", "metadata": "invalid"}]}}

    with pytest.raises(ValueError, match="dictionary update sequence element #0"):
        Chunker().flatten_hierarchical(tree)
    assert package_flatten(tree) == []


def test_invalid_structure_aware_weight_public_raises_and_package_suppresses() -> None:
    tree = {
        "method": "structure_aware",
        "max_size": 2,
        "overlap": 0,
        "grouping": {
            "by_kind": True,
            "element_weights": {"paragraph": "invalid"},
        },
        "root": {
            "kind": "section",
            "children": [
                {
                    "kind": "paragraph",
                    "chunks": [
                        {
                            "text": "bad",
                            "metadata": {"paragraph_kind": "paragraph"},
                        }
                    ],
                }
            ],
        },
    }

    with pytest.raises(ValueError, match=r"invalid literal for int\(\)"):
        Chunker().flatten_hierarchical(tree)
    assert package_flatten(tree) == []


def test_flatten_preserves_input_and_metadata_aliasing_contracts() -> None:
    nested = {"values": []}
    first_metadata = {
        "paragraph_kind": "paragraph",
        "nested": nested,
        "chunk_index": 41,
        "total_chunks": 99,
    }
    second_metadata = {"paragraph_kind": "paragraph", "nested": nested}
    tree = {
        "method": "words",
        "root": {
            "kind": "section",
            "title": "Section",
            "chunks": [
                {"text": "first", "metadata": first_metadata},
                {"text": "second", "metadata": second_metadata},
            ],
            "children": [],
        },
    }
    original_tree = copy.deepcopy(tree)

    rows = Chunker().flatten_hierarchical(tree)

    assert tree == original_tree
    assert first_metadata == original_tree["root"]["chunks"][0]["metadata"]
    assert rows[0]["metadata"] is not first_metadata
    assert rows[1]["metadata"] is not second_metadata
    assert rows[0]["metadata"]["nested"] is nested
    assert rows[1]["metadata"]["nested"] is nested
    assert rows[0]["metadata"]["ancestry_titles"] is rows[1]["metadata"]["ancestry_titles"]
    assert rows[0]["metadata"]["ancestry_titles"] == ["Section"]
    assert rows[0]["metadata"]["chunk_index"] == 41
    assert rows[0]["metadata"]["total_chunks"] == 99
    assert rows[1]["metadata"]["chunk_index"] == 2
    assert rows[1]["metadata"]["total_chunks"] == 2
