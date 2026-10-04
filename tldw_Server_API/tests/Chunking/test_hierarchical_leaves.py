"""Direct contracts for hierarchical leaf construction."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from loguru import logger

from tldw_Server_API.app.core.Chunking.hierarchical.leaves import build_leaf_block
from tldw_Server_API.app.core.Chunking.hierarchical.models import (
    HierarchyTextViews,
    ResolvedHierarchyOptions,
)

pytestmark = pytest.mark.unit

Call = tuple[str, str, Any, Any, Any, Any, dict[str, Any]]


class FakeLeafChunkingContext:
    """Scripted leaf context that records exact calls without doing real work."""

    def __init__(
        self,
        *,
        metadata_effects: list[Any] | None = None,
        plain_effects: list[Any] | None = None,
    ) -> None:
        self.calls: list[Call] = []
        self._metadata_effects = list(metadata_effects or [])
        self._plain_effects = list(plain_effects or [])

    @staticmethod
    def _resolve_effect(effects: list[Any]) -> list[Any]:
        effect = effects.pop(0) if effects else []
        if isinstance(effect, BaseException):
            raise effect
        return effect

    def chunk_text(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]:
        self.calls.append(("plain", text, method, max_size, overlap, language, options))
        return self._resolve_effect(self._plain_effects)

    def chunk_text_with_metadata(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]:
        self.calls.append(("metadata", text, method, max_size, overlap, language, options))
        return self._resolve_effect(self._metadata_effects)


def _options(
    method: str,
    *,
    sanitize_output: bool = True,
    method_options: dict[str, Any] | None = None,
) -> ResolvedHierarchyOptions:
    return ResolvedHierarchyOptions(
        method=method,
        max_size=5,
        overlap=1,
        language="fr",
        method_options=method_options or {},
        sanitize_output=sanitize_output,
    )


def _metadata_result(**metadata: Any) -> SimpleNamespace:
    return SimpleNamespace(metadata=SimpleNamespace(**metadata))


def _capture_log_records(action: Callable[[], Any]) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: records.append((message.record["level"].name, message.record["message"])),
        level="DEBUG",
    )
    try:
        action()
    finally:
        logger.remove(sink_id)
    return records


@pytest.mark.parametrize("span", [(2, 2, "paragraph"), (3, 2, "paragraph")])
def test_empty_or_reversed_span_returns_none_without_context_calls(
    span: tuple[int, int, str],
) -> None:
    context = FakeLeafChunkingContext()
    texts = HierarchyTextViews(original="text", sanitized="text", output="text")

    result = build_leaf_block(context, texts, span, _options("fixed_size"))

    assert result is None
    assert context.calls == []


@pytest.mark.parametrize(
    "method",
    [
        "semantic",
        "propositions",
        "json",
        "xml",
        "ebook_chapters",
        "rolling_summarize",
        "code",
        "code_ast",
    ],
)
def test_rewrite_methods_emit_invalid_offsets_and_call_plain_once(method: str) -> None:
    nested = {"items": []}
    method_options = {"flag": "kept", "nested": nested}
    context = FakeLeafChunkingContext(plain_effects=[["rewrite", 7]])
    texts = HierarchyTextViews(
        original="--source--",
        sanitized="--SOURCE--",
        output="--OUTPUT--",
    )

    result = build_leaf_block(
        context,
        texts,
        (2, 8, "paragraph"),
        _options(method, method_options=method_options),
    )

    assert context.calls == [("plain", "source", method, 5, 1, "fr", {"flag": "kept", "nested": nested})]
    assert result == {
        "kind": "paragraph",
        "start_offset": 2,
        "end_offset": 8,
        "chunks": [
            {
                "type": "text",
                "text": "rewrite",
                "metadata": {
                    "method": method,
                    "start_offset": None,
                    "end_offset": None,
                    "language": "fr",
                    "paragraph_kind": "paragraph",
                    "offsets_valid": False,
                },
            },
            {
                "type": "text",
                "text": "7",
                "metadata": {
                    "method": method,
                    "start_offset": None,
                    "end_offset": None,
                    "language": "fr",
                    "paragraph_kind": "paragraph",
                    "offsets_valid": False,
                },
            },
        ],
        "children": [],
    }


@pytest.mark.parametrize("method", ["words", "sentences", "tokens"])
def test_metadata_methods_call_metadata_once_and_forward_exact_arguments(method: str) -> None:
    nested = {"items": []}
    context = FakeLeafChunkingContext(metadata_effects=[[_metadata_result(start_char=1, end_char=4)]])
    texts = HierarchyTextViews(
        original="--abcdef--",
        sanitized="--ABCDEF--",
        output="--UVWXYZ--",
    )

    result = build_leaf_block(
        context,
        texts,
        (2, 8, "paragraph"),
        _options(method, method_options={"nested": nested}),
    )

    assert context.calls == [("metadata", "abcdef", method, 5, 1, "fr", {"nested": nested})]
    assert result == {
        "kind": "paragraph",
        "start_offset": 2,
        "end_offset": 8,
        "chunks": [
            {
                "type": "text",
                "text": "VWX",
                "metadata": {
                    "method": method,
                    "start_offset": 3,
                    "end_offset": 6,
                    "language": "fr",
                    "paragraph_kind": "paragraph",
                },
            }
        ],
        "children": [],
    }


@pytest.mark.parametrize("method", ["words", "sentences", "tokens"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"end_char": 3},
        {"start_char": "0", "end_char": 3},
        {"start_char": 0},
        {"start_char": 0, "end_char": "3"},
    ],
    ids=["missing-start", "non-integer-start", "missing-end", "non-integer-end"],
)
def test_invalid_metadata_offsets_are_skipped_without_plain_fallback(
    method: str,
    metadata: dict[str, Any],
) -> None:
    context = FakeLeafChunkingContext(metadata_effects=[[_metadata_result(**metadata)]])
    texts = HierarchyTextViews(original="raw", sanitized="raw", output="raw")

    result = build_leaf_block(context, texts, (0, 3, "paragraph"), _options(method))

    assert context.calls == [("metadata", "raw", method, 5, 1, "fr", {})]
    assert result == {
        "kind": "paragraph",
        "start_offset": 0,
        "end_offset": 3,
        "chunks": [],
        "children": [],
    }


@pytest.mark.parametrize(
    ("method", "expected_message"),
    [
        ("words", "words metadata mapping failed, using fallback: metadata sentinel"),
        ("sentences", "sentences metadata mapping failed, using fallback: metadata sentinel"),
        ("tokens", "Token metadata mapping failed, using fallback: metadata sentinel"),
    ],
)
def test_metadata_exception_logs_debug_and_calls_plain_fallback_once(
    method: str,
    expected_message: str,
) -> None:
    context = FakeLeafChunkingContext(
        metadata_effects=[RuntimeError("metadata sentinel")],
        plain_effects=[["abc"]],
    )
    texts = HierarchyTextViews(original="--abcabc--", sanitized="--abcabc--", output="--abcabc--")
    result: dict[str, Any] | None = None

    def call() -> None:
        nonlocal result
        result = build_leaf_block(context, texts, (2, 8, "paragraph"), _options(method))

    records = _capture_log_records(call)

    assert records == [("DEBUG", expected_message)]
    assert context.calls == [
        ("metadata", "abcabc", method, 5, 1, "fr", {}),
        ("plain", "abcabc", method, 5, 1, "fr", {}),
    ]
    assert result is not None
    assert result["chunks"] == [
        {
            "type": "text",
            "text": "abc",
            "metadata": {
                "method": method,
                "start_offset": 2,
                "end_offset": 5,
                "language": "fr",
                "paragraph_kind": "paragraph",
            },
        }
    ]


def test_failed_first_plain_fallback_is_retried_once_by_outer_fallback() -> None:
    context = FakeLeafChunkingContext(
        metadata_effects=[RuntimeError("metadata sentinel")],
        plain_effects=[RuntimeError("outer plain sentinel"), ["abcdef"]],
    )
    texts = HierarchyTextViews(original="--abcdef--", sanitized="--abcdef--", output="--abcdef--")
    result: dict[str, Any] | None = None

    def call() -> None:
        nonlocal result
        result = build_leaf_block(context, texts, (2, 8, "paragraph"), _options("words"))

    records = _capture_log_records(call)

    assert [entry[0] for entry in context.calls] == ["metadata", "plain", "plain"]
    assert context.calls == [
        ("metadata", "abcdef", "words", 5, 1, "fr", {}),
        ("plain", "abcdef", "words", 5, 1, "fr", {}),
        ("plain", "abcdef", "words", 5, 1, "fr", {}),
    ]
    assert records == [
        ("DEBUG", "words metadata mapping failed, using fallback: metadata sentinel"),
        (
            "WARNING",
            "Offset mapping failed for method=words: outer plain sentinel; using naive offsets",
        ),
    ]
    assert result is not None
    assert result["chunks"] == [
        {
            "type": "text",
            "text": "abcdef",
            "metadata": {
                "method": "words",
                "start_offset": 2,
                "end_offset": 8,
                "language": "fr",
                "paragraph_kind": "paragraph",
            },
        }
    ]


def test_structure_aware_emits_exact_output_slice_without_context_calls() -> None:
    context = FakeLeafChunkingContext()
    texts = HierarchyTextViews(
        original="--source--",
        sanitized="--SOURCE--",
        output="--OUTPUT--",
    )

    result = build_leaf_block(
        context,
        texts,
        (2, 8, "header_atx"),
        _options("structure_aware"),
    )

    assert context.calls == []
    assert result == {
        "kind": "header_atx",
        "start_offset": 2,
        "end_offset": 8,
        "chunks": [
            {
                "type": "text",
                "text": "OUTPUT",
                "metadata": {
                    "method": "structure_aware",
                    "start_offset": 2,
                    "end_offset": 8,
                    "language": "fr",
                    "paragraph_kind": "header_atx",
                },
            }
        ],
        "children": [],
    }


def test_ordinary_mapping_is_rolling_monotonic_and_bounded_to_the_block() -> None:
    context = FakeLeafChunkingContext(plain_effects=[["alpha", "alpha", "OUT", "missing-long"]])
    texts = HierarchyTextViews(
        original="OUTalpha--alphaOUT",
        sanitized="OUTalpha--alphaOUT",
        output="OUTalpha--alphaOUT",
    )

    result = build_leaf_block(context, texts, (3, 15, "paragraph"), _options("fixed_size"))

    assert context.calls == [("plain", "alpha--alpha", "fixed_size", 5, 1, "fr", {})]
    assert result is not None
    assert result["chunks"] == [
        {
            "type": "text",
            "text": "alpha",
            "metadata": {
                "method": "fixed_size",
                "start_offset": 3,
                "end_offset": 8,
                "language": "fr",
                "paragraph_kind": "paragraph",
            },
        },
        {
            "type": "text",
            "text": "alpha",
            "metadata": {
                "method": "fixed_size",
                "start_offset": 10,
                "end_offset": 15,
                "language": "fr",
                "paragraph_kind": "paragraph",
            },
        },
        {
            "type": "text",
            "text": "",
            "metadata": {
                "method": "fixed_size",
                "start_offset": 15,
                "end_offset": 15,
                "language": "fr",
                "paragraph_kind": "paragraph",
            },
        },
        {
            "type": "text",
            "text": "",
            "metadata": {
                "method": "fixed_size",
                "start_offset": 15,
                "end_offset": 15,
                "language": "fr",
                "paragraph_kind": "paragraph",
            },
        },
    ]
    offsets = [(chunk["metadata"]["start_offset"], chunk["metadata"]["end_offset"]) for chunk in result["chunks"]]
    assert offsets == sorted(offsets)
    assert all(3 <= chunk_start <= chunk_end <= 15 for chunk_start, chunk_end in offsets)


@pytest.mark.parametrize(
    ("sanitize_output", "expected_text"),
    [(False, "RAW"), (True, "SAN")],
)
def test_output_slice_uses_the_selected_text_view(
    sanitize_output: bool,
    expected_text: str,
) -> None:
    context = FakeLeafChunkingContext()
    texts = HierarchyTextViews(
        original="--RAW--",
        sanitized="--MID--",
        output="--SAN--",
    )

    result = build_leaf_block(
        context,
        texts,
        (2, 5, "paragraph"),
        _options("structure_aware", sanitize_output=sanitize_output),
    )

    assert context.calls == []
    assert result is not None
    assert result["chunks"][0]["text"] == expected_text


def test_results_are_fresh_and_inputs_are_not_mutated() -> None:
    nested = {"items": ["kept"]}
    method_options = {"nested": nested}
    options = _options("structure_aware", method_options=method_options)
    texts = HierarchyTextViews(original="--RAW--", sanitized="--MID--", output="--SAN--")
    span = (2, 5, "paragraph")
    texts_snapshot = (texts.original, texts.sanitized, texts.output)
    span_snapshot = tuple(span)
    options_snapshot = deepcopy(method_options)
    context = FakeLeafChunkingContext()

    first = build_leaf_block(context, texts, span, options)
    second = build_leaf_block(context, texts, span, options)

    assert first == second
    assert first is not second
    assert first is not None
    assert second is not None
    assert first["chunks"] is not second["chunks"]
    assert first["children"] is not second["children"]
    first["chunks"].append({"changed": True})
    first["children"].append({"changed": True})
    assert len(second["chunks"]) == 1
    assert second["children"] == []
    assert (texts.original, texts.sanitized, texts.output) == texts_snapshot
    assert span == span_snapshot
    assert options.method_options is method_options
    assert options.method_options["nested"] is nested
    assert options.method_options == options_snapshot
    assert context.calls == []
